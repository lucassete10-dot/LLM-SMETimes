"""One frozen shape candidate on all XJTU internal folds; no outer access."""
import json, logging, shutil, time
from pathlib import Path
from datetime import datetime, timezone
import joblib, numpy as np
from threadpoolctl import threadpool_limits
from battery_smetimes.shape_forecast import fit, predict
from battery_smetimes.tail_boost import labels
from battery_smetimes.curve_fidelity import curve_metrics, selection_score
from scripts.train_crossfit_drift_smoke import clean
from scripts.train_tail_boost import load_cells
from scripts.prepare_same_type_forecast import save_json, sha


def main():
    smoke=Path('experiments/shape_forecast/20261005T081014_843192Z')
    assert json.loads((smoke/'independent_verification.json').read_text())['passed']
    source=Path('experiments/tail_boost/20261005T034452_066902Z')
    old=json.loads((source/'config.json').read_text()); plans=old['plans']
    assert len(plans)==5 and all(p['dataset']=='XJTU' for p in plans)
    root=Path('experiments/shape_xjtu_fivefold')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    root.mkdir(parents=True); snap=root/'source_snapshot'; snap.mkdir()
    files=[Path(__file__),*[Path('battery_smetimes')/(n+'.py') for n in ('shape_forecast','waveform_shape','tail_boost','research_baselines','forecast')],*[Path('scripts')/(n+'.py') for n in ('train_tail_boost','train_curve_fidelity','prepare_zhu2022_nca','train_zhu2022_nca')]]
    for f in files: shutil.copy2(f,snap/f.name)
    candidates=[None,dict(kind='extra_trees',alpha=None)]
    controls={p['name']:str((source/p['name']/'20261005'/'candidate_0'/'model.joblib').resolve()) for p in plans}
    prereg='docs/soh_shape_xjtu_fivefold_20261005.md'
    save_json(root/'config.json',dict(plans=plans,seed=20261005,candidates=candidates,fivefold_shape=True,smoke=True,source_run=str(source.resolve()),source_config_sha256=sha(source/'config.json'),prepared_sha256=old['prepared_sha256'],source_sha256={f.name:sha(f) for f in files},control_paths=controls,base_sha256={k:sha(v) for k,v in controls.items()},preregistration_file=prereg,preregistration_sha256=sha(prereg)))
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(root/'training.log',encoding='utf-8'),logging.StreamHandler()])
    reports=[]; start=time.perf_counter()
    for plan in plans:
        cells=load_cells(plan); assert set(cells)==set(plan['train']+plan['validation']) and not set(cells)&set(plan['test'])
        for index,params in enumerate(candidates):
            folder=root/plan['name']/f'candidate_{index}';folder.mkdir(parents=True)
            logging.info('FIT %s candidate=%s',plan['name'],index)
            if params is None:
                base=joblib.load(controls[plan['name']]);assert base['train_ids']==plan['train'] and base['seed']==20261005
                assert base['parameters']==dict(model='extra_trees',leaf=1,tail_weight=3,local=False)
                b=dict(mode='control',base=base,train_ids=plan['train'])
            else:b=fit(cells,plan['train'],seed=20261005,**params)
            joblib.dump(b,folder/'model.joblib');rows=[];arrays={}
            for key in plan['validation']:
                cc=clean(cells[key]);x,y=labels(cells[key]);p=predict(b,cc,x)
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
