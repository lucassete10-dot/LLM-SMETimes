"""Fixed NASA rate candidate: lock all inner fits before outer evaluation."""
import json,logging,shutil,time
from datetime import datetime,timezone
from pathlib import Path
import joblib,numpy as np
from threadpoolctl import threadpool_limits
from battery_smetimes.fade_rate import fit,predict
from battery_smetimes.tail_boost import labels
from battery_smetimes.curve_fidelity import curve_metrics
from scripts.train_curve_fidelity import load_subset,AUDIT
from scripts.train_crossfit_drift_smoke import clean
from scripts.prepare_same_type_forecast import save_json,sha
from scripts.audit_tail_numeric_gates import aggregate,gates,metric


def evaluate(model,cells,ids,folder,name,predictor=predict):
    rows=[];arrays={}
    for key in ids:
        c=clean(cells[key]);x,y=labels(cells[key]);p=predictor(model,c,x)
        rows.append(dict(cell_id=key,**curve_metrics(x,y,p)))
        arrays.update({key+'__100__'+f:a for f,a in [('cycle',x),('truth',y),('prediction',p)]})
        poison=dict(c,cycle=np.r_[c['cycle'],101],raw=np.vstack([c['raw'],np.full((1,c['raw'].shape[1]),np.inf)]),soh=np.inf,cycle_life=np.inf)
        np.testing.assert_allclose(p,predictor(model,poison,x),atol=1e-12,rtol=0)
        try:predictor(model,dict(c,physical_family='different'),x)
        except ValueError:pass
        else:raise AssertionError('family isolation failed')
    save_json(folder/(name+'_metrics.json'),rows);np.savez_compressed(folder/(name+'_predictions.npz'),**arrays)
    return rows


def main():
    screened=Path('experiments/nasa_fade_rate_fourfold/20261005T085132_456474Z')
    assert json.loads((screened/'verification.json').read_text())['passed'] and json.loads((screened/'screening_decision.json').read_text())['expand_formal']
    cfg=json.loads((screened/'config.json').read_text());plans=cfg['plans'];seeds=list(range(20261005,20261010));baseline=Path(cfg['source_run'])
    root=Path('experiments/nasa_rate_formal')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');root.mkdir(parents=True);snap=root/'source_snapshot';snap.mkdir()
    files=[Path(__file__),*[Path('battery_smetimes')/(n+'.py') for n in ('fade_rate','conditional_drift','crossfit_drift','tail_boost','forecast','research_baselines')],*[Path('scripts')/(n+'.py') for n in ('train_curve_fidelity','prepare_zhu2022_nca','train_zhu2022_nca')]]
    for f in files:shutil.copy2(f,snap/f.name)
    save_json(root/'config.json',dict(dataset='NASA',plans=plans,seeds=seeds,parameters=dict(learned_anchor=False),screened_run=str(screened.resolve()),screened_config_sha256=sha(screened/'config.json'),baseline=str(baseline.resolve()),baseline_config_sha256=sha(baseline/'config.json'),prepared_sha256=cfg['prepared_sha256'],source_sha256={f.name:sha(f) for f in files},preregistration_sha256=sha('docs/soh_nasa_rate_formal_20261005.md'),policy='fixed rate candidate; all inner SHA locked before outer access',development_cv=True))
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(root/'training.log',encoding='utf-8'),logging.StreamHandler()]);start=time.perf_counter();locks=[]
    for plan in plans:
        cells=load_subset(plan['train']+plan['validation']);assert not set(cells)&set(plan['test'])
        for seed in seeds:
            folder=root/plan['name']/str(seed)/'inner';folder.mkdir(parents=True);logging.info('INNER %s seed=%s',plan['name'],seed)
            model=fit(cells,plan['train'],False,seed,[dict(cell_id=k) for k in plan['train']]);joblib.dump(model,folder/'model.joblib')
            evaluate(model,cells,plan['train'],folder,'training');evaluate(model,cells,plan['validation'],folder,'validation')
            locks.append(dict(fold=plan['name'],seed=seed,model_file=str((folder/'model.joblib').resolve()),model_sha256=sha(folder/'model.joblib'),validation_file=str((folder/'validation_predictions.npz').resolve()),validation_sha256=sha(folder/'validation_predictions.npz')))
    assert len(locks)==20;save_json(root/'inner_lock.json',dict(fixed_policy=True,outer_loaded=False,models=locks))
    lock_sha=sha(root/'inner_lock.json');save_json(root/'progress.json',dict(inner_complete=True,inner_lock_sha256=lock_sha,outer_complete=False))
    rows=[];old_rows=[];models=[]
    for plan in plans:
        ids=plan['train']+plan['validation'];assert not set(ids)&set(plan['test'])
        cells=load_subset(ids+plan['test']);assert sha(root/'inner_lock.json')==lock_sha
        for seed in seeds:
            folder=root/plan['name']/str(seed)/'outer';folder.mkdir();logging.info('OUTER %s seed=%s',plan['name'],seed)
            model=fit(cells,ids,False,seed,[dict(cell_id=k) for k in ids]);joblib.dump(model,folder/'model.joblib')
            evaluate(model,cells,ids,folder,'training');rr=evaluate(model,cells,plan['test'],folder,'test')
            with np.load(folder/'test_predictions.npz') as a:
                for key in plan['test']:
                    x,y,p=[a[key+'__100__'+f] for f in ('cycle','truth','prediction')];rows.append(dict(cell_id=key,seed=seed,**metric(x,y,p)))
            models.append(dict(fold=plan['name'],seed=seed,model_sha256=sha(folder/'model.joblib'),prediction_sha256=sha(folder/'test_predictions.npz')))
        for seed in range(20261004,20261009):
            with np.load(baseline/plan['name']/str(seed)/'predictions.npz') as a:
                for key in plan['test']:
                    x,y,p=[a[key+'__100__'+f] for f in ('cycle','truth','prediction')]
                    cx,cy=labels(cells[key]);np.testing.assert_array_equal(x,cx);np.testing.assert_array_equal(y,cy)
                    old_rows.append(dict(cell_id=key,seed=seed,**metric(x,y,p)))
    summary,cell_rows=aggregate(rows);old,_=aggregate(old_rows);checks=gates(summary,old)
    save_json(root/'by_cell_seed.json',rows);save_json(root/'by_cell.json',cell_rows)
    save_json(root/'summary.json',dict(new=summary,baseline=old,gates=checks,numeric_gates_passed=all(checks.values()),goal_complete=False))
    save_json(root/'progress.json',dict(inner_complete=True,inner_lock_sha256=lock_sha,outer_complete=True,training_complete=True,models=models,seconds=time.perf_counter()-start,goal_complete=False))
    print('OUTPUT='+str(root.resolve()),flush=True);print(json.dumps(dict(mape=summary['mape_percent'],endpoint=summary['endpoint_pp'],drop=summary['drop_error_pp'],joint_fraction=summary['joint_pass_fraction'],gates=checks)),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
