"""Independent train-statistic/PLS refit and explicit curve replay, including training diagnostics."""
import argparse,json
from pathlib import Path
import joblib,numpy as np
from scipy.special import expit
from sklearn.cross_decomposition import PLSRegression
from threadpoolctl import threadpool_limits
from battery_smetimes.parameter_pls import predict
from battery_smetimes.conditional_drift import prefix
from battery_smetimes.parameter_ridge import FIELDS
from battery_smetimes.warm_monotone import derive_prior
from battery_smetimes.tail_boost import labels,is_nca
from scripts.verify_conditional_drift import manual_scale
from scripts.train_crossfit_drift_smoke import clean
from scripts.train_tail_boost import load_cells
from scripts.prepare_same_type_forecast import save_json,sha
from scripts.audit_tail_numeric_gates import metric


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    root=parser.parse_args().run;cfg=json.loads((root/'config.json').read_text())
    assert json.loads((root/'progress.json').read_text())['training_complete']
    source=Path(cfg['source_run']);assert sha(source/'config.json')==cfg['source_config_sha256']
    assert sha('docs/soh_parameter_pls_stage_20261005.md')==cfg['preregistration_sha256']
    for name,digest in cfg['source_sha256'].items():assert sha(root/'source_snapshot'/name)==digest
    for f in ('parameter_pls','parameter_ridge','conditional_drift','tail_boost','forecast'):
        p=Path('battery_smetimes')/(f+'.py');assert sha(p)==cfg['source_sha256'][p.name]
    assert cfg['candidates']==[None]+[dict(components=n,logarithmic=l) for n in (2,4) for l in (False,True)]
    reports=json.loads((root/'summary.json').read_text());checked=[]
    for plan in cfg['plans']:
        cells=load_cells(plan);assert set(cells)==set(plan['train']+plan['validation']) and not set(cells)&set(plan['test'])
        assert sha(cfg['base_paths'][plan['name']])==cfg['base_sha256'][plan['name']]
        assert sha(source/(plan['name']+'_training_parameters.json'))==cfg['parameter_fits_sha256'][plan['name']]
        parameter_rows=derive_prior(cells,plan['train'])[1]
        assert parameter_rows==json.loads((root/(plan['name']+'_training_parameters.json')).read_text())
        for index,params in enumerate(cfg['candidates']):
            folder=root/plan['name']/f'candidate_{index}';b=joblib.load(folder/'model.joblib');assert b['train_ids']==plan['train']
            if params is not None:
                assert b['training_parameter_fits']==parameter_rows and b['seed']==cfg['seed'] and b['components']==params['components'] and b['logarithmic']==params['logarithmic']
                raw=np.stack([prefix(clean(cells[k])) for k in plan['train']]);med=[]
                for j in range(raw.shape[1]):
                    v=raw[:,j][np.isfinite(raw[:,j])];med.append(np.median(v) if len(v) else 0)
                med=np.asarray(med);np.testing.assert_allclose(med,b['imputer'].statistics_,atol=1e-12,rtol=0)
                raw=np.where(np.isfinite(raw),raw,med);mean=raw.mean(axis=0);var=raw.var(axis=0);scale=manual_scale(var,mean,len(raw))
                for expected,actual in [(mean,b['scaler'].mean_),(var,b['scaler'].var_),(scale,b['scaler'].scale_)]:np.testing.assert_allclose(expected,actual,atol=1e-10,rtol=1e-10)
                target=np.array([[r[f] for f in FIELDS] for r in parameter_rows]);frac=np.clip((target[:,0]-.5)/.75,1e-6,1-1e-6);target[:,0]=np.log(frac/(1-frac))
                if params['logarithmic']:target[:,1:]=np.log(np.maximum(target[:,1:],1e-6))
                tm=target.mean(axis=0);ts=np.maximum(target.std(axis=0),1e-6)
                np.testing.assert_allclose(tm,b['target_mean'],atol=1e-12,rtol=0);np.testing.assert_allclose(ts,b['target_scale'],atol=1e-12,rtol=0)
                fixed=dict(n_components=min(params['components'],len(plan['train'])-1,raw.shape[1]),scale=False,max_iter=1000,tol=1e-8)
                for k,v in fixed.items():assert b['model'].get_params()[k]==v
                assert all(n<1000 for n in b['model'].n_iter_)
                independent=PLSRegression(**fixed).fit((raw-mean)/scale,(target-tm)/ts)
                np.testing.assert_allclose(independent.coef_,b['model'].coef_,atol=1e-9,rtol=1e-9)
            report=next(r for r in reports if r['fold']==plan['name'] and r['candidate']==index)
            for role,ids in [('training',plan['train']),('validation',plan['validation'])]:
                rows={r['cell_id']:r for r in json.loads((folder/(role+'_metrics.json')).read_text())};assert set(rows)==set(ids)
                saved_params={r['cell_id']:r['values'] for r in json.loads((folder/(role+'_predicted_parameters.json')).read_text())}
                assert set(saved_params)==(set(ids) if params is not None else set())
                scores=[];values_all=[]
                with np.load(folder/(role+'_predictions.npz')) as arrays:
                    assert len(arrays.files)==3*len(rows)
                    for key in ids:
                        c=cells[key];cycle,y=labels(c);cc=clean(c);pred=predict(b,cc,cycle)
                        np.testing.assert_array_equal(cycle,arrays[key+'__100__cycle']);np.testing.assert_array_equal(y,arrays[key+'__100__truth'])
                        np.testing.assert_allclose(pred,arrays[key+'__100__prediction'],atol=1e-12,rtol=0)
                        if params is not None:
                            q=prefix(cc);q=np.where(np.isfinite(q),q,med);qx=(q-mean)/scale
                            u=((qx-independent._x_mean)@independent.coef_.T+independent.intercept_)*ts+tm
                            a=.5+.75*expit(u[0]);rates=np.exp(u[1:]) if params['logarithmic'] else np.maximum(u[1:],0)
                            np.testing.assert_allclose(np.r_[a,rates],saved_params[key],atol=1e-9,rtol=1e-9)
                            h=(cycle-100)/100;hinge=.1*np.logaddexp(0,(h-rates[2])/.1)
                            np.testing.assert_allclose(pred,a-rates[0]*h-rates[1]*hinge**2,atol=1e-9,rtol=1e-9)
                            assert np.all(np.diff(pred)<=1e-12)
                        if role=='validation':
                            if is_nca(cc):poison=dict(cc,early_cycle=np.r_[cc['early_cycle'],101],early_raw=np.vstack([cc['early_raw'],np.full((1,cc['early_raw'].shape[1]),np.inf)]),truth=np.inf,cycle_life=np.inf)
                            else:poison=dict(cc,cycle=np.r_[cc['cycle'],101],raw=np.vstack([cc['raw'],np.full((1,cc['raw'].shape[1]),np.inf)]),soh=np.inf,cycle_life=np.inf)
                            np.testing.assert_allclose(pred,predict(b,poison,cycle),atol=1e-12,rtol=0)
                            try:predict(b,dict(cc,physical_family='different'),cycle)
                            except ValueError:pass
                            else:raise AssertionError('family isolation failed')
                        values=metric(cycle,y,pred)
                        for f,v in values.items():np.testing.assert_allclose(v,rows[key][f],atol=1e-10,rtol=0)
                        scores.append(sum(values[f] for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp'))/4);values_all.append(values)
                        checked.append(dict(cell_id=key,candidate=index,role=role,points=len(cycle),model_sha256=sha(folder/'model.joblib'),predictions_sha256=sha(folder/(role+'_predictions.npz'))))
                if role=='training':np.testing.assert_allclose(np.mean([v['mape_percent'] for v in values_all]),report['training_mape'],atol=1e-10,rtol=0)
                else:
                    np.testing.assert_allclose(np.mean(scores)+.25*np.quantile(scores,.9),report['score'],atol=1e-10,rtol=0)
                    for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp'):np.testing.assert_allclose(np.mean([v[f] for v in values_all]),report[f],atol=1e-10,rtol=0)
            print('PASS '+plan['name']+' candidate='+str(index),flush=True)
    assert len(checked)==990
    save_json(root/'independent_verification.json',dict(passed=True,checked=checked,source_sha256=sha(__file__),outer_evaluated=False,goal_complete=False))
    print('CURVES='+str(len(checked)))


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
