"""Refit frozen inner NCA selections and evaluate all outer cells, no tuning."""
import argparse,json,logging,shutil,traceback
from datetime import datetime,timezone
from pathlib import Path
import joblib,numpy as np
from scripts.train_nca_curve_fidelity import fit,predict,OLD
from scripts.finalize_curve_fidelity import summarize
from scripts.prepare_same_type_forecast import save_json,sha
from battery_smetimes.curve_fidelity import curve_metrics


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    run=parser.parse_args().run;cfg=json.loads((run/'config.json').read_text())
    assert not cfg['smoke'] and len(cfg['seeds'])==5 and len(cfg['plans'])==5
    assert json.loads((run/'progress.json').read_text())['training_complete']
    assert json.loads((run/'independent_validation_verification.json').read_text())['passed']
    for name,digest in cfg['source_sha256'].items():
        assert sha(run/'source_snapshot'/name)==digest
    for path in (Path('scripts/train_nca_curve_fidelity.py'),Path('scripts/train_zhu2022_nca.py'),
            Path('scripts/prepare_zhu2022_nca.py'),Path('battery_smetimes/curve_fidelity.py'),
            Path('battery_smetimes/research_baselines.py')):
        assert sha(path)==cfg['source_sha256'][path.name], 'Training implementation changed; use archived source'
    oldcfg=json.loads((OLD/'config.json').read_text());assert cfg['plans']==oldcfg['plans']
    assert sha(OLD/'config.json')==cfg['original_config_sha256']
    source=Path(cfg['cells_directory']);assert sha(source/'index.json')==cfg['index_sha256']
    index=json.loads((source/'index.json').read_text())
    out=Path('experiments/nca_curve_fidelity_outer')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True);snap=out/'source_snapshot';snap.mkdir()
    files=[Path(__file__),Path('scripts/train_nca_curve_fidelity.py'),Path('scripts/train_zhu2022_nca.py'),
        Path('scripts/prepare_zhu2022_nca.py'),Path('battery_smetimes/curve_fidelity.py'),
        Path('scripts/finalize_curve_fidelity.py')]
    for p in files:shutil.copy2(p,snap/p.name)
    save_json(out/'config.json',dict(dataset='Zhu2022_NCA',validation_source=str(run.resolve()),
        validation_config_sha256=sha(run/'config.json'),cells_directory=str(source.resolve()),
        index_sha256=cfg['index_sha256'],source_sha256={p.name:sha(p) for p in files},
        baseline=str((OLD/'extra_trees').resolve()),origin=100,not_blind=True,
        selection_sha256={p['name']:{str(s):sha(run/p['name']/str(s)/'selection.json') for s in cfg['seeds']} for p in cfg['plans']}))
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',
        handlers=[logging.FileHandler(out/'training.log',encoding='utf-8'),logging.StreamHandler()])
    rows=[];old_rows=[]
    try:
        for plan in cfg['plans']:
            cells={}
            for key in plan['train']+plan['validation']+plan['test']:
                entry=index['cells'][key];p=source/entry['file'];assert sha(p)==entry['sha256'];cells[key]=joblib.load(p)
            dev=plan['train']+plan['validation'];assert not set(dev)&set(plan['test'])
            for seed in cfg['seeds']:
                selection=json.loads((run/plan['name']/str(seed)/'selection.json').read_text())
                selected=min(selection['candidates'],key=lambda r:r['selection_score']);assert selection['selected']==selected
                folder=out/plan['name']/str(seed);folder.mkdir(parents=True);save_json(folder/'selection.json',selection)
                logging.info('REFIT %s seed=%s %s',plan['name'],seed,selected['parameters'])
                b=fit(cells,dev,selected['parameters'],seed);joblib.dump(b,folder/'model.joblib')
                metrics=[];arrays={}
                for key in plan['test']:
                    c=cells[key];x,y=c['cycle'],c['truth'];clean={k:v for k,v in c.items() if k not in ('cycle','truth')}
                    p=predict(b,clean,x)
                    poison=dict(clean,prefix=np.full_like(clean['prefix'],np.inf),truth=np.inf,cycle_life=np.inf,
                        early_cycle=np.r_[clean['early_cycle'],101,10000],
                        early_raw=np.vstack([clean['early_raw'],np.full((2,clean['early_raw'].shape[1]),np.inf)]))
                    np.testing.assert_allclose(p,predict(b,poison,x),atol=1e-12,rtol=0)
                    try:predict(b,dict(clean,physical_family='different'),x)
                    except ValueError:pass
                    else:raise AssertionError('family gate failed')
                    metrics.append(dict(cell_id=key,**curve_metrics(x,y,p)))
                    arrays.update({key+'__100__'+f:a for f,a in [('cycle',x),('truth',y),('prediction',p)]})
                save_json(folder/'test_metrics.json',metrics);np.savez_compressed(folder/'test_predictions.npz',**arrays)
                rows.extend(dict(r,fold=plan['name'],seed=seed,dataset='Zhu2022_NCA') for r in metrics)
                save_json(out/'by_cell.json',rows);logging.info('SAVED %s seed=%s',plan['name'],seed)
            for seed in oldcfg['seeds']:
                with np.load(OLD/'extra_trees'/plan['name']/str(seed)/'predictions.npz') as a:
                    for key in plan['test']:
                        x,y,p=[a[key+'__100__'+f] for f in ('cycle','truth','prediction')]
                        np.testing.assert_array_equal(x,cells[key]['cycle']);np.testing.assert_array_equal(y,cells[key]['truth'])
                        old_rows.append(dict(cell_id=key,seed=seed,**curve_metrics(x,y,p)))
        summary,by_cell=summarize(rows);baseline,_=summarize(old_rows)
        gates=dict(full_mape=summary['mape_percent']['mean']<=6,late_mape=summary['late_mape']['mean']<=6,
            rmse=summary['rmse_soh']['mean']<=.04,endpoint_mean=summary['endpoint_pp']['mean']<=5,
            drop_mean=summary['drop_error_pp']['mean']<=4,joint_fraction=summary['joint_pass_fraction']>=.9,
            severe_fraction=summary['severe_flat_fraction']<=.1,
            full_no_regression=summary['mape_percent']['mean']<=baseline['mape_percent']['mean']+.5)
        for field in ('endpoint_p90','drop_error_p90'):
            gates[field]=summary[field]<=(3 if baseline[field]<=3 else .8*baseline[field])
        save_json(out/'cell_seed_means.json',by_cell)
        save_json(out/'summary.json',dict(dataset='Zhu2022_NCA',new=summary,baseline=baseline,gates=gates,
            numeric_gates_passed=bool(all(gates.values())),curve_delivery_complete=False,not_blind=True))
        save_json(out/'progress.json',dict(training_complete=True,all_outer_cells_evaluated=True,
            plots_exported=False,independent_artifact_audit_complete=False,goal_complete=False))
        print('OUTPUT='+str(out.resolve()))
    except BaseException:
        save_json(out/'failure.json',dict(traceback=traceback.format_exc(),completed_rows=len(rows)));raise


if __name__=='__main__':main()
