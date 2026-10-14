"""Fixed one candidate per family; all internal folds, no outer data."""
import json,logging,shutil,time
from pathlib import Path
from datetime import datetime,timezone
import joblib,numpy as np
from threadpoolctl import threadpool_limits
from battery_smetimes.decorrelated_forest import fit,predict
from battery_smetimes.tail_boost import labels,is_nca
from battery_smetimes.curve_fidelity import curve_metrics,selection_score
from scripts.train_crossfit_drift_smoke import clean
from scripts.train_tail_boost import load_cells
from scripts.prepare_same_type_forecast import save_json,sha


def main():
    smoke=Path('experiments/decorrelated_forest/20261005T052350_857439Z')
    assert json.loads((smoke/'independent_verification.json').read_text())['passed']
    choices={'MIT':dict(max_features=.5,augmented=False),'XJTU':dict(max_features=.25,augmented=True),'Zhu2022_NCA':dict(max_features=.25,augmented=False)}
    routes={'MIT':'20261005T034441_870182Z','XJTU':'20261005T034452_066902Z','Zhu2022_NCA':'20261005T034431_681399Z'}
    root=Path('experiments/decorrelated_forest_fivefold')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');root.mkdir(parents=True)
    snap=root/'source_snapshot';snap.mkdir()
    files=[Path(__file__),Path('battery_smetimes/decorrelated_forest.py'),Path('battery_smetimes/temporal_features.py'),Path('battery_smetimes/tail_boost.py'),Path('battery_smetimes/research_baselines.py'),Path('battery_smetimes/forecast.py'),Path('scripts/train_tail_boost.py'),Path('scripts/train_curve_fidelity.py'),Path('scripts/prepare_zhu2022_nca.py'),Path('scripts/train_zhu2022_nca.py')]
    for f in files:shutil.copy2(f,snap/f.name)
    plans=[];provenance=[];hashes={};controls={};control_sha={}
    for dataset,stamp in routes.items():
        source=Path('experiments/tail_boost')/stamp;cfg=json.loads((source/'config.json').read_text())
        assert len(cfg['plans'])==5
        provenance.append(dict(file=str((source/'config.json').resolve()),sha256=sha(source/'config.json')))
        for p in cfg['plans']:
            assert p['dataset']==dataset;plans.append(p)
            for k in p['train']+p['validation']:hashes[k]=cfg['prepared_sha256'][k]
            path=source/p['name']/'20261005'/'candidate_0'/'model.joblib'
            controls[p['name']]=str(path.resolve());control_sha[p['name']]=sha(path)
    save_json(root/'config.json',dict(plans=plans,seed=20261005,fivefold=True,candidates_by_dataset={d:[None,p] for d,p in choices.items()},provenance=provenance,control_paths=controls,base_sha256=control_sha,prepared_sha256=hashes,source_sha256={f.name:sha(f) for f in files},preregistration_file='docs/soh_decorrelated_forest_fivefold_20261005.md',preregistration_sha256=sha('docs/soh_decorrelated_forest_fivefold_20261005.md'),smoke=True))
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(root/'training.log',encoding='utf-8'),logging.StreamHandler()])
    reports=[];start=time.perf_counter()
    for plan in plans:
        cells=load_cells(plan);assert set(cells)==set(plan['train']+plan['validation']) and not set(cells)&set(plan['test'])
        for index,params in enumerate([None,choices[plan['dataset']]]):
            folder=root/plan['name']/f'candidate_{index}';folder.mkdir(parents=True)
            logging.info('FIT %s candidate=%s params=%s',plan['name'],index,params)
            if params is None:
                base=joblib.load(controls[plan['name']]);assert base['train_ids']==plan['train'] and base['seed']==20261005
                assert base['parameters']==dict(model='extra_trees',leaf=1,tail_weight=3,local=False)
                b=dict(mode='control',base=base,train_ids=plan['train'])
            else:b=fit(cells,plan['train'],seed=20261005,**params)
            joblib.dump(b,folder/'model.joblib');rows=[];arrays={}
            for key in plan['validation']:
                cc=clean(cells[key]);x,y=labels(cells[key]);p=predict(b,cc,x)
                if is_nca(cc):poison=dict(cc,early_cycle=np.r_[cc['early_cycle'],101],early_raw=np.vstack([cc['early_raw'],np.full((1,cc['early_raw'].shape[1]),np.inf)]),truth=np.inf,cycle_life=np.inf)
                else:poison=dict(cc,cycle=np.r_[cc['cycle'],101],raw=np.vstack([cc['raw'],np.full((1,cc['raw'].shape[1]),np.inf)]),soh=np.inf,cycle_life=np.inf)
                np.testing.assert_allclose(p,predict(b,poison,x),atol=1e-12,rtol=0)
                try:predict(b,dict(cc,physical_family='different'),x)
                except ValueError:pass
                else:raise AssertionError('family isolation failed')
                rows.append(dict(cell_id=key,**curve_metrics(x,y,p)))
                arrays.update({key+'__100__'+f:a for f,a in [('cycle',x),('truth',y),('prediction',p)]})
            save_json(folder/'validation_metrics.json',rows);np.savez_compressed(folder/'validation_predictions.npz',**arrays)
            report=dict(dataset=plan['dataset'],fold=plan['name'],candidate=index,parameters=params,score=selection_score(rows),**{f:float(np.mean([r[f] for r in rows])) for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp')})
            reports.append(report);save_json(root/'summary.json',reports);logging.info('SAVED %s',report)
    save_json(root/'progress.json',dict(training_complete=True,seconds=time.perf_counter()-start,outer_evaluated=False,goal_complete=False))
    print('OUTPUT='+str(root.resolve()))


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
