"""NCA curve-score validation, with independent passive per-cell conversion."""
import argparse,hashlib,json,logging,shutil,traceback
from datetime import datetime,timezone
from pathlib import Path
import joblib,numpy as np
from scripts.train_zhu2022_nca import fit as original_fit,predict as original_predict
from scripts.prepare_zhu2022_nca import prefix
from scripts.prepare_same_type_forecast import save_json,sha
from battery_smetimes.curve_fidelity import curve_metrics,selection_score

OLD=Path('experiments/zhu2022_nca_forecast/20261004T161453_449950Z')
CANDIDATES=[dict(local=False,trees=64,depth=8,leaf=l) for l in (3,6,12)]+[
    dict(local=False,trees=128,depth=8,leaf=1),dict(local=False,trees=128,depth=None,leaf=1),
    dict(local=False,trees=128,depth=None,leaf=4),dict(local=True,trees=128,depth=None,leaf=1),
    dict(local=True,trees=128,depth=None,leaf=4)]


def condition(c):return tuple(c['conditions'])


def fit(cells,ids,parameters,seed):
    base=original_fit(cells,ids,'extra_trees',{k:parameters[k] for k in ('trees','depth','leaf')},seed)
    local={}
    if parameters['local']:
        groups={}
        for key in ids:groups.setdefault(condition(cells[key]),[]).append(key)
        for key,members in groups.items():
            if len(members)>=6:
                local[key]=original_fit(cells,members,'extra_trees',{k:parameters[k] for k in ('trees','depth','leaf')},seed)
    return dict(global_model=base,local=local,train_ids=list(ids),parameters=parameters)


def predict(bundle,cell,future):
    cycle=np.asarray(cell['early_cycle']);raw=np.asarray(cell['early_raw'])
    take=cycle<=100
    signal_only=dict(cell,prefix=prefix(cycle[take],raw[take],cell['conditions']))
    return original_predict(bundle['local'].get(condition(cell),bundle['global_model']),signal_only,future)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--cells',type=Path);parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();oldcfg=json.loads((OLD/'config.json').read_text())
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    if args.prepare:
        source=Path(oldcfg['prepared_directory'])/'prepared_cells.joblib'
        assert sha(source)==oldcfg['prepared_sha256']
        out=Path('data/interim/nca_curve_cells')/stamp;out.mkdir(parents=True)
        files={}
        for key,c in joblib.load(source).items():
            dest=out/(hashlib.sha256(key.encode()).hexdigest()+'.joblib');joblib.dump(c,dest)
            files[key]=dict(file=dest.name,sha256=sha(dest))
        save_json(out/'index.json',dict(cells=files,source_sha256=sha(source),source=str(source.resolve()),
            passive_conversion_only=True,no_statistics_or_models_fitted=True))
        print('OUTPUT='+str(out.resolve()));return
    if args.cells is None:raise ValueError('--cells required')
    index=json.loads((args.cells/'index.json').read_text());assert index['passive_conversion_only']
    assert index['source_sha256']==oldcfg['prepared_sha256']
    out=Path('experiments/nca_curve_fidelity')/stamp;out.mkdir(parents=True)
    sources=[Path(__file__),Path('scripts/train_zhu2022_nca.py'),Path('scripts/prepare_zhu2022_nca.py'),
        Path('battery_smetimes/curve_fidelity.py'),Path('battery_smetimes/research_baselines.py')]
    snap=out/'source_snapshot';snap.mkdir()
    for p in sources:shutil.copy2(p,snap/p.name)
    shutil.copy2('docs/nca_curve_fidelity_stage_20261005.md',out/'plan.md')
    plans=oldcfg['plans'][:1] if args.smoke else oldcfg['plans']
    seeds=[20261005] if args.smoke else list(range(20261005,20261010))
    save_json(out/'config.json',dict(plans=plans,seeds=seeds,candidates=CANDIDATES,smoke=args.smoke,
        cells_directory=str(args.cells.resolve()),index_sha256=sha(args.cells/'index.json'),
        prepared_sha256=index['source_sha256'],original_config_sha256=sha(OLD/'config.json'),
        source_sha256={p.name:sha(p) for p in sources},selection='fixed curve-fidelity selection_score',
        outer_test_loaded=False))
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',
        handlers=[logging.FileHandler(out/'training.log',encoding='utf-8'),logging.StreamHandler()])
    reports=[]
    try:
        for plan in plans:
            ids=plan['train']+plan['validation'];assert not set(ids)&set(plan['test']);cells={}
            for key in ids:
                entry=index['cells'][key];source=args.cells/entry['file'];assert sha(source)==entry['sha256']
                c=joblib.load(source)
                assert np.all(c['early_cycle']<=100)
                np.testing.assert_array_equal(c['prefix'],prefix(c['early_cycle'],c['early_raw'],c['conditions']))
                cells[key]=c
            for seed in seeds:
                scores=[]
                for i,parameters in enumerate(CANDIDATES):
                    logging.info('TRAIN %s seed=%s candidate=%s %s',plan['name'],seed,i,parameters)
                    folder=out/plan['name']/str(seed)/f'candidate_{i}';folder.mkdir(parents=True)
                    b=fit(cells,plan['train'],parameters,seed);joblib.dump(b,folder/'model.joblib')
                    rows=[];arrays={}
                    for key in plan['validation']:
                        c=cells[key];x,y=c['cycle'],c['truth'];clean={k:v for k,v in c.items() if k not in ('truth','cycle')}
                        p=predict(b,clean,x)
                        np.testing.assert_allclose(p,predict(b,dict(clean,truth=np.inf,cycle_life=np.inf),x),atol=1e-12,rtol=0)
                        poison=dict(clean,early_cycle=np.r_[clean['early_cycle'],101,10000],
                            early_raw=np.vstack([clean['early_raw'],np.full((2,clean['early_raw'].shape[1]),np.inf)]))
                        np.testing.assert_allclose(p,predict(b,poison,x),atol=1e-12,rtol=0)
                        try:predict(b,dict(clean,physical_family='different'),x)
                        except ValueError:pass
                        else:raise AssertionError('family gate failed')
                        rows.append(dict(cell_id=key,**curve_metrics(x,y,p)))
                        arrays.update({key+'__100__'+f:a for f,a in [('cycle',x),('truth',y),('prediction',p)]})
                    save_json(folder/'validation_metrics.json',rows);np.savez_compressed(folder/'validation_predictions.npz',**arrays)
                    report=dict(fold=plan['name'],seed=seed,candidate=i,parameters=parameters,selection_score=selection_score(rows),
                        **{k:float(np.mean([r[k] for r in rows])) for k in ('mape_percent','late_mape','endpoint_pp','drop_error_pp','joint_pass')})
                    reports.append(report);scores.append(report);save_json(out/'summary.json',reports)
                    logging.info('SAVED %s seed=%s candidate=%s score=%.3f MAPE=%.3f endpoint=%.3f',
                        plan['name'],seed,i,report['selection_score'],report['mape_percent'],report['endpoint_pp'])
                save_json(out/plan['name']/str(seed)/'selection.json',dict(selected=min(scores,key=lambda r:r['selection_score']),
                    candidates=scores,used_only_validation=True))
        save_json(out/'progress.json',dict(training_complete=True,outer_evaluation_complete=False,
            smoke=args.smoke,goal_complete=False,source_prefix_recomputed=True))
        print('OUTPUT='+str(out.resolve()))
    except BaseException:
        save_json(out/'failure.json',dict(traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
