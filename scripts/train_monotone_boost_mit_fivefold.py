"""Preregistered MIT fivefold validation only; no outer cells loaded."""
import json,logging,shutil,time
from pathlib import Path
from datetime import datetime,timezone
import joblib,numpy as np
from threadpoolctl import threadpool_limits
from battery_smetimes.monotone_boost import fit,predict
from battery_smetimes.tail_boost import fit as tree_fit,labels
from battery_smetimes.curve_fidelity import curve_metrics,selection_score
from scripts.train_crossfit_drift_smoke import clean
from scripts.train_tail_boost import load_cells
from scripts.prepare_same_type_forecast import save_json,sha


def main():
    smoke=Path('experiments/monotone_boost/20261005T050748_433940Z')
    assert json.loads((smoke/'independent_verification.json').read_text())['passed']
    source=Path('experiments/tail_boost/20261005T034441_870182Z')
    original=json.loads((source/'config.json').read_text());plans=original['plans']
    assert len(plans)==5 and all(p['dataset']=='MIT' for p in plans)
    root=Path('experiments/monotone_boost_mit_fivefold')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');root.mkdir(parents=True)
    snap=root/'source_snapshot';snap.mkdir()
    files=[Path(__file__),Path('battery_smetimes/monotone_boost.py'),Path('battery_smetimes/conditional_drift.py'),Path('battery_smetimes/crossfit_drift.py'),Path('battery_smetimes/tail_boost.py'),Path('battery_smetimes/research_baselines.py'),Path('battery_smetimes/forecast.py'),Path('scripts/train_tail_boost.py'),Path('scripts/train_curve_fidelity.py'),Path('scripts/prepare_zhu2022_nca.py'),Path('scripts/train_zhu2022_nca.py')]
    for f in files:shutil.copy2(f,snap/f.name)
    candidates=[None,dict(l2=1,monotone=True),dict(l2=10,monotone=True)]
    control=dict(model='extra_trees',leaf=1,tail_weight=3,local=False);seed=20261005
    save_json(root/'config.json',dict(plans=plans,seed=seed,candidates=candidates,smoke=True,control_refit=True,control_parameters=control,source_run=str(source.resolve()),source_config_sha256=sha(source/'config.json'),prepared_sha256=original['prepared_sha256'],source_sha256={f.name:sha(f) for f in files},preregistration_file='docs/soh_monotone_boost_mit_fivefold_20261005.md',preregistration_sha256=sha('docs/soh_monotone_boost_mit_fivefold_20261005.md')))
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(root/'training.log',encoding='utf-8'),logging.StreamHandler()])
    reports=[];start=time.perf_counter()
    for plan in plans:
        cells=load_cells(plan);assert set(cells)==set(plan['train']+plan['validation']) and not set(cells)&set(plan['test'])
        for index,params in enumerate(candidates):
            folder=root/plan['name']/f'candidate_{index}';folder.mkdir(parents=True)
            logging.info('FIT %s candidate=%s',plan['name'],index)
            b=dict(mode='control',base=tree_fit(cells,plan['train'],control,seed),train_ids=plan['train']) if params is None else fit(cells,plan['train'],seed=seed,**params)
            joblib.dump(b,folder/'model.joblib');rows=[];arrays={}
            for key in plan['validation']:
                cc=clean(cells[key]);x,y=labels(cells[key]);p=predict(b,cc,x)
                if params is not None:assert np.all(np.diff(p)<=1e-12)
                poison=dict(cc,cycle=np.r_[cc['cycle'],101],raw=np.vstack([cc['raw'],np.full((1,cc['raw'].shape[1]),np.inf)]),soh=np.inf,cycle_life=np.inf)
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
