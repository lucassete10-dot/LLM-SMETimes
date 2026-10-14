"""Training statistics, parameter transforms and independent normal-equation replay."""
import argparse,json
from pathlib import Path
import joblib,numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits
from battery_smetimes.parameter_ridge import predict,FIELDS
from battery_smetimes.conditional_drift import prefix
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
    assert sha('docs/soh_parameter_ridge_stage_20261005.md')==cfg['preregistration_sha256']
    for name,digest in cfg['source_sha256'].items():assert sha(root/'source_snapshot'/name)==digest
    for f in ('parameter_ridge','warm_monotone','conditional_drift','tail_boost','forecast'):
        path=Path('battery_smetimes')/(f+'.py');assert sha(path)==cfg['source_sha256'][path.name]
    assert cfg['candidates']==[None]+[dict(alpha=a,logarithmic=l) for a in (10,100) for l in (False,True)]
    reports=json.loads((root/'summary.json').read_text());checked=[]
    for plan in cfg['plans']:
        cells=load_cells(plan);assert set(cells)==set(plan['train']+plan['validation']) and not set(cells)&set(plan['test'])
        assert sha(source/plan['name']/'base.joblib')==cfg['base_sha256'][plan['name']]
        parameter_rows=derive_prior(cells,plan['train'])[1]
        saved=json.loads((root/(plan['name']+'_training_parameters.json')).read_text());assert parameter_rows==saved
        for index,params in enumerate(cfg['candidates']):
            folder=root/plan['name']/f'candidate_{index}';b=joblib.load(folder/'model.joblib')
            assert b['train_ids']==plan['train']
            if params is not None:
                assert b['training_parameter_fits']==parameter_rows and b['seed']==cfg['seed'] and b['alpha']==params['alpha'] and b['logarithmic']==params['logarithmic']
                e=b['encoder'];raw=np.stack([prefix(clean(cells[k])) for k in plan['train']]);med=[]
                for j in range(raw.shape[1]):
                    v=raw[:,j][np.isfinite(raw[:,j])];med.append(np.median(v) if len(v) else 0)
                med=np.asarray(med);np.testing.assert_allclose(med,e['imputer'].statistics_,atol=1e-12,rtol=0)
                raw=np.where(np.isfinite(raw),raw,med)
                mean=raw.mean(axis=0);variance=raw.var(axis=0);scale=manual_scale(variance,mean,len(raw))
                for expected,actual in [(mean,e['scaler'].mean_),(variance,e['scaler'].var_),(scale,e['scaler'].scale_)]:np.testing.assert_allclose(expected,actual,atol=1e-10,rtol=1e-10)
                pc=e['pca'];assert pc.n_components_==min(2,len(plan['train'])-1)
                np.testing.assert_allclose(pc.components_@pc.components_.T,np.eye(pc.n_components_),atol=1e-10,rtol=0)
                x=((raw-mean)/scale-pc.mean_)@pc.components_.T
                target=np.array([[r[f] for f in FIELDS] for r in parameter_rows])
                frac=np.clip((target[:,0]-.5)/.75,1e-6,1-1e-6);target[:,0]=np.log(frac/(1-frac))
                if params['logarithmic']:target[:,1:]=np.log(np.maximum(target[:,1:],1e-6))
                tm=target.mean(axis=0);ts=np.maximum(target.std(axis=0),1e-6)
                np.testing.assert_allclose(tm,b['target_mean'],atol=1e-12,rtol=0);np.testing.assert_allclose(ts,b['target_scale'],atol=1e-12,rtol=0)
                y=(target-tm)/ts;xm=x.mean(axis=0);ym=y.mean(axis=0);xc=x-xm;yc=y-ym
                coef=np.linalg.solve(xc.T@xc+params['alpha']*np.eye(x.shape[1]),xc.T@yc);intercept=ym-xm@coef
                np.testing.assert_allclose(coef.T,b['model'].coef_,atol=1e-10,rtol=1e-10);np.testing.assert_allclose(intercept,b['model'].intercept_,atol=1e-10,rtol=0)
            rows={r['cell_id']:r for r in json.loads((folder/'validation_metrics.json').read_text())};assert set(rows)==set(plan['validation'])
            scores=[];values_all=[]
            with np.load(folder/'validation_predictions.npz') as arrays:
                assert len(arrays.files)==3*len(rows)
                for key in plan['validation']:
                    c=cells[key];cycle,y=labels(c);cc=clean(c);pred=predict(b,cc,cycle)
                    np.testing.assert_array_equal(cycle,arrays[key+'__100__cycle']);np.testing.assert_array_equal(y,arrays[key+'__100__truth'])
                    np.testing.assert_allclose(pred,arrays[key+'__100__prediction'],atol=1e-12,rtol=0)
                    if params is not None:
                        q=prefix(cc);q=np.where(np.isfinite(q),q,med);qx=((q-mean)/scale-pc.mean_)@pc.components_.T
                        u=(qx@coef+intercept)*ts+tm;a=.5+.75*expit(u[0]);rates=np.exp(u[1:]) if params['logarithmic'] else np.maximum(u[1:],0)
                        h=(cycle-100)/100;hinge=.1*np.logaddexp(0,(h-rates[2])/.1)
                        independent=a-rates[0]*h-rates[1]*hinge**2
                        np.testing.assert_allclose(pred,independent,atol=1e-10,rtol=1e-10);assert np.all(np.diff(pred)<=1e-12)
                    if is_nca(cc):poison=dict(cc,early_cycle=np.r_[cc['early_cycle'],101],early_raw=np.vstack([cc['early_raw'],np.full((1,cc['early_raw'].shape[1]),np.inf)]),truth=np.inf,cycle_life=np.inf)
                    else:poison=dict(cc,cycle=np.r_[cc['cycle'],101],raw=np.vstack([cc['raw'],np.full((1,cc['raw'].shape[1]),np.inf)]),soh=np.inf,cycle_life=np.inf)
                    np.testing.assert_allclose(pred,predict(b,poison,cycle),atol=1e-12,rtol=0)
                    try:predict(b,dict(cc,physical_family='different'),cycle)
                    except ValueError:pass
                    else:raise AssertionError('family isolation failed')
                    values=metric(cycle,y,pred)
                    for f,v in values.items():np.testing.assert_allclose(v,rows[key][f],atol=1e-10,rtol=0)
                    scores.append(sum(values[f] for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp'))/4);values_all.append(values)
                    checked.append(dict(cell_id=key,candidate=index,points=len(cycle),model_sha256=sha(folder/'model.joblib'),predictions_sha256=sha(folder/'validation_predictions.npz')))
            report=next(r for r in reports if r['fold']==plan['name'] and r['candidate']==index)
            np.testing.assert_allclose(np.mean(scores)+.25*np.quantile(scores,.9),report['score'],atol=1e-10,rtol=0)
            for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp'):np.testing.assert_allclose(np.mean([v[f] for v in values_all]),report[f],atol=1e-10,rtol=0)
            print('PASS '+plan['name']+' candidate='+str(index),flush=True)
    assert len(checked)==215
    save_json(root/'independent_verification.json',dict(passed=True,checked=checked,source_sha256=sha(__file__),outer_evaluated=False,goal_complete=False))
    print('CURVES='+str(len(checked)))


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
