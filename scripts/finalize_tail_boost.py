"""Validation-only frozen old/new policy, then all outer cells, no tuning."""
import argparse,json,logging,shutil,traceback
from datetime import datetime,timezone
from pathlib import Path
import joblib,numpy as np
from threadpoolctl import threadpool_limits
from battery_smetimes.tail_selection import choose,fit,predict
from battery_smetimes.tail_boost import labels,is_nca
from battery_smetimes.curve_fidelity import curve_metrics
from scripts.train_tail_boost import load_cells,NCA
from scripts.train_curve_fidelity import AUDIT,load_subset
from scripts.finalize_curve_fidelity import summarize,OLD as CORE_OLD,NASA_OLD
from scripts.train_nca_curve_fidelity import OLD as NCA_OLD
from scripts.prepare_same_type_forecast import save_json,sha


def all_cells(plan):
    if plan['dataset']!='Zhu2022_NCA':return load_subset(plan['train']+plan['validation']+plan['test'])
    index=json.loads((NCA/'index.json').read_text());cells={}
    for key in plan['train']+plan['validation']+plan['test']:
        entry=index['cells'][key];p=NCA/entry['file'];assert sha(p)==entry['sha256'];cells[key]=joblib.load(p)
    return cells


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    run=parser.parse_args().run;cfg=json.loads((run/'config.json').read_text())
    assert not cfg['smoke'] and len(cfg['seeds'])==5
    assert json.loads((run/'progress.json').read_text())['training_complete']
    assert json.loads((run/'independent_verification.json').read_text())['passed']
    lock=json.loads((run/'reference_lock.json').read_text());assert lock['passed']
    for name,digest in cfg['source_sha256'].items():assert sha(run/'source_snapshot'/name)==digest
    datasets={p['dataset'] for p in cfg['plans']};assert len(datasets)==1;dataset=datasets.pop()
    out=Path('experiments/tail_boost_outer')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True);snap=out/'source_snapshot';snap.mkdir()
    sources=[Path(__file__),Path('battery_smetimes/tail_selection.py'),Path('battery_smetimes/tail_boost.py'),
        Path('battery_smetimes/curve_fidelity.py'),Path('battery_smetimes/research_baselines.py'),
        Path('battery_smetimes/forecast.py'),Path('scripts/train_nca_curve_fidelity.py'),Path('scripts/train_zhu2022_nca.py'),
        Path('scripts/prepare_zhu2022_nca.py'),Path('scripts/train_curve_fidelity.py'),Path('scripts/train_tail_boost.py'),
        Path('scripts/finalize_curve_fidelity.py'),Path('scripts/train_stm32_compact_trees.py'),Path('battery_smetimes/physical_types.py')]
    for p in sources:shutil.copy2(p,snap/p.name)
    baseline=NCA_OLD/'extra_trees' if dataset=='Zhu2022_NCA' else NASA_OLD if dataset=='NASA' else CORE_OLD
    save_json(out/'config.json',dict(dataset=dataset,training_source=str(run.resolve()),training_config_sha256=sha(run/'config.json'),
        reference_lock_sha256=sha(run/'reference_lock.json'),source_sha256={p.name:sha(p) for p in sources},
        baseline=str(baseline.resolve()),policy=cfg['primary_outer_policy'],not_blind=True,origin=100))
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',
        handlers=[logging.FileHandler(out/'training.log',encoding='utf-8'),logging.StreamHandler()])
    rows=[];old_rows=[];decisions=[]
    try:
        # Freeze EVERY primary choice before any outer cell is loaded/evaluated.
        for plan in cfg['plans']:
            dev=load_cells(plan);assert not set(dev)&set(plan['test'])
            for seed in cfg['seeds']:
                decision=choose(run,cfg,plan,seed,dev);decisions.append(dict(fold=plan['name'],seed=seed,decision=decision))
        save_json(out/'frozen_decisions.json',decisions)
        for plan in cfg['plans']:
            cells=all_cells(plan);dev=plan['train']+plan['validation'];assert not set(dev)&set(plan['test'])
            for seed in cfg['seeds']:
                decision=next(d['decision'] for d in decisions if d['fold']==plan['name'] and d['seed']==seed)
                folder=out/plan['name']/str(seed);folder.mkdir(parents=True);save_json(folder/'selection.json',decision)
                logging.info('REFIT %s seed=%s mode=%s',plan['name'],seed,decision['selected']['mode'])
                b=fit(cells,dev,decision['selected'],seed);joblib.dump(b,folder/'model.joblib');metrics=[];arrays={}
                for key in plan['test']:
                    c=cells[key];x,y=labels(c);remove=('cycle','truth') if is_nca(c) else ('soh','label_cycle','unfiltered_soh','reference_capacity_ah')
                    clean={k:v for k,v in c.items() if k not in remove};p=predict(b,clean,x)
                    if is_nca(c):
                        poison=dict(clean,early_cycle=np.r_[clean['early_cycle'],101,10000],
                            early_raw=np.vstack([clean['early_raw'],np.full((2,clean['early_raw'].shape[1]),np.inf)]),prefix=np.full_like(clean['prefix'],np.inf),truth=np.inf,cycle_life=np.inf)
                    else:
                        poison=dict(clean,cycle=np.r_[clean['cycle'],101,10000],
                            raw=np.vstack([clean['raw'],np.full((2,clean['raw'].shape[1]),np.inf)]),soh=np.inf,cycle_life=np.inf)
                    np.testing.assert_allclose(p,predict(b,poison,x),atol=1e-12,rtol=0)
                    try:predict(b,dict(clean,physical_family='different'),x)
                    except ValueError:pass
                    else:raise AssertionError('family gate failed')
                    metrics.append(dict(cell_id=key,**curve_metrics(x,y,p)))
                    arrays.update({key+'__100__'+f:a for f,a in [('cycle',x),('truth',y),('prediction',p)]})
                save_json(folder/'test_metrics.json',metrics);np.savez_compressed(folder/'test_predictions.npz',**arrays)
                rows.extend(dict(r,fold=plan['name'],seed=seed,dataset=dataset) for r in metrics);save_json(out/'by_cell.json',rows)
                logging.info('SAVED %s seed=%s',plan['name'],seed)
            for seed in range(20261004,20261009):
                p=baseline/plan['name']/str(seed)/('test/predictions.npz' if dataset not in ('Zhu2022_NCA','NASA') else 'predictions.npz')
                with np.load(p) as a:
                    for key in plan['test']:
                        x,y,pred=[a[key+'__100__'+f] for f in ('cycle','truth','prediction')]
                        xx,yy=labels(cells[key]);np.testing.assert_array_equal(x,xx);np.testing.assert_array_equal(y,yy)
                        old_rows.append(dict(cell_id=key,seed=seed,**curve_metrics(x,y,pred)))
        summary,by_cell=summarize(rows);old,_=summarize(old_rows)
        gates=dict(full_mape=summary['mape_percent']['mean']<=6,late_mape=summary['late_mape']['mean']<=6,
            rmse=summary['rmse_soh']['mean']<=.04,endpoint_mean=summary['endpoint_pp']['mean']<=5,
            drop_mean=summary['drop_error_pp']['mean']<=4,joint_fraction=summary['joint_pass_fraction']>=.9,
            severe_fraction=summary['severe_flat_fraction']<=.1,full_no_regression=summary['mape_percent']['mean']<=old['mape_percent']['mean']+.5)
        for field in ('endpoint_p90','drop_error_p90'):gates[field]=summary[field]<=(3 if old[field]<=3 else .8*old[field])
        save_json(out/'summary.json',dict(dataset=dataset,new=summary,baseline=old,gates=gates,
            numeric_gates_passed=bool(all(gates.values())),curve_delivery_complete=False,not_blind=True))
        save_json(out/'cell_seed_means.json',by_cell)
        save_json(out/'progress.json',dict(training_complete=True,all_outer_cells_evaluated=True,
            plots_exported=False,independent_artifact_audit_complete=False,goal_complete=False))
        print('OUTPUT='+str(out.resolve()))
    except BaseException:
        save_json(out/'failure.json',dict(traceback=traceback.format_exc(),completed_rows=len(rows)));raise


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
