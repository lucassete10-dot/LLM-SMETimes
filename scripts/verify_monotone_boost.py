"""Audit training-only statistics, matched constraints and every validation curve."""
import argparse,json
from pathlib import Path
import joblib,numpy as np
from threadpoolctl import threadpool_limits
from battery_smetimes.monotone_boost import fit,predict,inputs
from battery_smetimes.conditional_drift import prefix
from battery_smetimes.tail_boost import labels,is_nca
from battery_smetimes.tail_boost import fit as tree_fit
from scripts.train_crossfit_drift_smoke import clean
from scripts.train_tail_boost import load_cells
from scripts.prepare_same_type_forecast import save_json,sha
from scripts.audit_tail_numeric_gates import metric


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    root=parser.parse_args().run;cfg=json.loads((root/'config.json').read_text())
    assert json.loads((root/'progress.json').read_text())['training_complete']
    source=Path(cfg['source_run']);assert sha(source/'config.json')==cfg['source_config_sha256']
    assert sha(cfg.get('preregistration_file','docs/soh_monotone_boost_stage_20261005.md'))==cfg['preregistration_sha256']
    for name,digest in cfg['source_sha256'].items():
        assert sha(root/'source_snapshot'/name)==digest
    for file in ('battery_smetimes/monotone_boost.py','battery_smetimes/conditional_drift.py','battery_smetimes/tail_boost.py','battery_smetimes/forecast.py'):
        assert sha(file)==cfg['source_sha256'][Path(file).name]
    expected_candidates=([None,dict(l2=1,monotone=True),dict(l2=10,monotone=True)] if cfg.get('control_refit') else [None]+[dict(l2=l,monotone=m) for l in (1,10) for m in (False,True)])
    assert cfg['candidates']==expected_candidates
    reports=json.loads((root/'summary.json').read_text());checked=[]
    for plan in cfg['plans']:
        cells=load_cells(plan);assert set(cells)==set(plan['train']+plan['validation'])
        assert not set(cells)&set(plan['test'])
        if not cfg.get('control_refit'):assert sha(source/plan['name']/'base.joblib')==cfg['base_sha256'][plan['name']]
        for index,params in enumerate(cfg['candidates']):
            folder=root/plan['name']/f'candidate_{index}';b=joblib.load(folder/'model.joblib')
            assert b['train_ids']==plan['train']
            replay=None
            if params is None and cfg.get('control_refit'):
                assert cfg['control_parameters']==dict(model='extra_trees',leaf=1,tail_weight=3,local=False)
                replay=dict(mode='control',base=tree_fit(cells,plan['train'],cfg['control_parameters'],cfg['seed']),train_ids=plan['train'])
            if params is not None:
                assert b['seed']==cfg['seed'] and b['l2']==params['l2'] and b['monotone']==params['monotone']
                z=np.stack([prefix(cells[k]) for k in plan['train']]);expected=[]
                for j in range(z.shape[1]):
                    v=z[:,j][np.isfinite(z[:,j])];expected.append(np.median(v) if len(v) else 0)
                np.testing.assert_allclose(expected,b['imputer'].statistics_,atol=1e-12,rtol=0)
                model=b['model'];p=model.get_params()
                for key,value in dict(max_iter=120,learning_rate=.05,max_leaf_nodes=31,min_samples_leaf=20,max_bins=64,early_stopping=False,l2_regularization=params['l2'],random_state=cfg['seed']).items():assert p[key]==value
                assert model.n_iter_==120
                np.testing.assert_array_equal(p['monotonic_cst'],np.r_[np.zeros(len(expected),dtype=int),np.full(3,-1 if params['monotone'] else 0)])
                replay=fit(cells,plan['train'],seed=cfg['seed'],**params)
            rows={r['cell_id']:r for r in json.loads((folder/'validation_metrics.json').read_text())}
            assert set(rows)==set(plan['validation']);scores=[];values_all=[]
            with np.load(folder/'validation_predictions.npz') as arrays:
                assert len(arrays.files)==3*len(rows)
                for key in plan['validation']:
                    c=cells[key];x,y=labels(c);cc=clean(c);pred=predict(b,cc,x)
                    np.testing.assert_array_equal(x,arrays[key+'__100__cycle'])
                    np.testing.assert_array_equal(y,arrays[key+'__100__truth'])
                    np.testing.assert_allclose(pred,arrays[key+'__100__prediction'],atol=1e-12,rtol=0)
                    if replay is not None:
                        np.testing.assert_allclose(pred,predict(replay,cc,x),atol=1e-12,rtol=0)
                        if params is not None and params['monotone']:assert np.all(np.diff(pred)<=1e-12)
                    if is_nca(cc):
                        poison=dict(cc,early_cycle=np.r_[cc['early_cycle'],101],early_raw=np.vstack([cc['early_raw'],np.full((1,cc['early_raw'].shape[1]),np.inf)]),truth=np.inf,cycle_life=np.inf)
                    else:
                        poison=dict(cc,cycle=np.r_[cc['cycle'],101],raw=np.vstack([cc['raw'],np.full((1,cc['raw'].shape[1]),np.inf)]),soh=np.inf,cycle_life=np.inf)
                    np.testing.assert_allclose(pred,predict(b,poison,x),atol=1e-12,rtol=0)
                    try:predict(b,dict(cc,physical_family='different'),x)
                    except ValueError:pass
                    else:raise AssertionError('family isolation failed')
                    values=metric(x,y,pred)
                    for f,v in values.items():np.testing.assert_allclose(v,rows[key][f],atol=1e-10,rtol=0)
                    scores.append(sum(values[f] for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp'))/4)
                    values_all.append(values);checked.append(dict(cell_id=key,candidate=index,points=len(x),model_sha256=sha(folder/'model.joblib'),predictions_sha256=sha(folder/'validation_predictions.npz')))
            report=next(r for r in reports if r['fold']==plan['name'] and r['candidate']==index)
            np.testing.assert_allclose(np.mean(scores)+.25*np.quantile(scores,.9),report['score'],atol=1e-10,rtol=0)
            for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp'):
                np.testing.assert_allclose(np.mean([v[f] for v in values_all]),report[f],atol=1e-10,rtol=0)
            print('PASS '+plan['name']+' candidate='+str(index),flush=True)
    assert len(checked)==sum(len(p['validation'])*len(cfg['candidates']) for p in cfg['plans'])
    count=len(cfg['plans'])*(len(cfg['candidates'])-(0 if cfg.get('control_refit') else 1))
    save_json(root/'independent_verification.json',dict(passed=True,checked=checked,retrained_models=count,source_sha256=sha(__file__),outer_evaluated=False,goal_complete=False))
    print('CURVES='+str(len(checked)))


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
