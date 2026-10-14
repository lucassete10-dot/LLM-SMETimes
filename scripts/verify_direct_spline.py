"""Training-only preprocessing, normal equations and full curve replay."""
import argparse,json
from pathlib import Path
import joblib,numpy as np
from threadpoolctl import threadpool_limits
from battery_smetimes.conditional_drift import prefix
from battery_smetimes.tail_boost import labels
from scripts.train_crossfit_drift_smoke import clean
from scripts.train_tail_boost import load_cells
from scripts.verify_tail_boost import preprocessing
from scripts.verify_conditional_drift import manual_scale
from scripts.audit_tail_numeric_gates import metric
from scripts.prepare_same_type_forecast import save_json,sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    root=parser.parse_args().run;cfg=json.loads((root/'config.json').read_text())
    assert json.loads((root/'progress.json').read_text())['training_complete']
    source=Path(cfg['source_run']);assert sha(source/'config.json')==cfg['source_config_sha256']
    for name,digest in cfg['source_sha256'].items():assert sha(root/'source_snapshot'/name)==digest
    for file in ('battery_smetimes/direct_spline.py','battery_smetimes/conditional_drift.py','battery_smetimes/tail_boost.py'):
        assert sha(file)==cfg['source_sha256'][Path(file).name]
    reports=json.loads((root/'summary.json').read_text());checked=[];solvers=[]
    for plan in cfg['plans']:
        cells=load_cells(plan);assert not set(cells)&set(plan['test'])
        assert sha(source/plan['name']/'base.joblib')==cfg['base_sha256'][plan['name']]
        for index,parameters in enumerate(cfg['candidates']):
            folder=root/plan['name']/f'candidate_{index}';b=joblib.load(folder/'model.joblib');assert b['train_ids']==plan['train']
            if parameters is None:
                from battery_smetimes.direct_spline import predict
                preprocessing(b['base'],cells)
            else:
                if b['mode']=='full_spline':
                    from battery_smetimes.direct_full_spline import predict,training_arrays
                    assert sha('battery_smetimes/direct_full_spline.py')==cfg['source_sha256']['direct_full_spline.py']
                else:
                    from battery_smetimes.direct_spline import predict,training_arrays
                encoder=b['encoder'];assert encoder['train_ids']==plan['train']
                raw=np.stack([prefix(cells[k]) for k in plan['train']]);median=[]
                for j in range(raw.shape[1]):
                    finite=raw[:,j][np.isfinite(raw[:,j])];median.append(np.median(finite) if len(finite) else 0)
                np.testing.assert_allclose(median,encoder['imputer'].statistics_,atol=1e-12,rtol=0)
                filled=np.where(np.isnan(raw),np.array(median)[None],raw);mean=filled.mean(axis=0);var=filled.var(axis=0)
                np.testing.assert_allclose(mean,encoder['scaler'].mean_,atol=1e-12,rtol=0)
                np.testing.assert_allclose(var,encoder['scaler'].var_,atol=1e-12,rtol=0)
                scale=manual_scale(var,mean,len(plan['train']));np.testing.assert_allclose(scale,encoder['scaler'].scale_,atol=1e-12,rtol=0)
                if 'pca' in encoder:
                    pca=encoder['pca'];z=(filled-mean)/scale;centered=z-z.mean(axis=0)
                    n=min(parameters['components'],len(plan['train'])-1,z.shape[1]);assert pca.n_components_==n
                    np.testing.assert_allclose(pca.mean_,z.mean(axis=0),atol=1e-12,rtol=0)
                    singular=np.linalg.svd(centered,compute_uv=False)[:n];projected=centered@pca.components_.T
                    np.testing.assert_allclose(projected.T@projected,np.diag(singular**2),atol=1e-8,rtol=1e-10)
                    np.testing.assert_allclose(pca.components_@pca.components_.T,np.eye(n),atol=1e-10,rtol=0)
                x,y,w=training_arrays(cells,plan['train'],encoder);mean=np.average(x,axis=0,weights=w);var=np.average((x-mean)**2,axis=0,weights=w)
                np.testing.assert_allclose(mean,b['scaler'].mean_,atol=1e-9,rtol=0)
                np.testing.assert_allclose(var,b['scaler'].var_,atol=1e-9,rtol=0)
                assert np.isfinite(b['scaler'].var_).all() and np.all(b['scaler'].var_>=0)
                z=b['scaler'].transform(x);zm=np.average(z,axis=0,weights=w);ym=np.average(y,weights=w);zz=z-zm
                coef=b['model'].coef_;rhs=zz.T@(w*(y-ym));gradient=zz.T@(w*(zz@coef-(y-ym)))+b['alpha']*coef
                residual=float(np.linalg.norm(gradient)/max(np.linalg.norm(rhs),1));assert residual<=1e-5
                np.testing.assert_allclose(b['model'].intercept_,ym-zm@coef,atol=1e-10,rtol=0)
                solvers.append(dict(fold=plan['name'],candidate=index,normal_equation_relative_residual=residual,iterations=None if b['model'].n_iter_ is None else b['model'].n_iter_.tolist()))
            stored={r['cell_id']:r for r in json.loads((folder/'validation_metrics.json').read_text())};assert set(stored)==set(plan['validation']);scores=[]
            with np.load(folder/'validation_predictions.npz') as a:
                for key in plan['validation']:
                    x,y=labels(cells[key]);p=predict(b,clean(cells[key]),x)
                    for f,v in [('cycle',x),('truth',y),('prediction',p)]:np.testing.assert_allclose(v,a[key+'__100__'+f],atol=1e-12,rtol=0)
                    values=metric(x,y,p)
                    for f,v in values.items():np.testing.assert_allclose(v,stored[key][f],atol=1e-10,rtol=0)
                    scores.append(sum(values[f] for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp'))/4)
                    checked.append(dict(cell_id=key,candidate=index,targets=len(x),model_sha256=sha(folder/'model.joblib'),prediction_sha256=sha(folder/'validation_predictions.npz')))
            report=next(r for r in reports if r['fold']==plan['name'] and r['candidate']==index)
            np.testing.assert_allclose(np.mean(scores)+.25*np.quantile(scores,.9),report['score'],atol=1e-10,rtol=0)
            print('PASS '+plan['name']+' candidate='+str(index),flush=True)
    save_json(root/'independent_verification.json',dict(passed=True,rows=checked,solvers=solvers,training_only_encoder=True,
        source_sha256=sha(__file__),normal_equation_relative_tolerance=1e-5,outer_evaluated=False,goal_complete=False))
    print('CURVES='+str(len(checked)))


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
