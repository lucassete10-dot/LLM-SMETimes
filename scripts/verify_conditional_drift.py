"""Independent train-only PCA algebra, Ridge solution and validation replay."""
import argparse,json
from pathlib import Path
import joblib,numpy as np
from threadpoolctl import threadpool_limits
from battery_smetimes.conditional_drift import prefix,query,predict
from battery_smetimes.tail_boost import labels
from scripts.train_crossfit_drift_smoke import clean
from scripts.train_tail_boost import load_cells
from scripts.verify_tail_boost import preprocessing
from scripts.audit_tail_numeric_gates import metric
from scripts.prepare_same_type_forecast import save_json,sha


def manual_scale(var, mean, n_samples):
    eps=np.finfo(np.float64).eps
    constant=var<=n_samples*eps*var+(n_samples*mean*eps)**2
    scale=np.sqrt(var);scale[constant]=1
    return scale


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    root=parser.parse_args().run;cfg=json.loads((root/'config.json').read_text())
    assert json.loads((root/'progress.json').read_text())['training_complete']
    source=Path(cfg['source_run']);assert sha(source/'config.json')==cfg['source_config_sha256']
    assert sha(source/'independent_verification.json')==cfg['source_verification_sha256']
    assert json.loads((source/'independent_verification.json').read_text())['passed']
    for name,digest in cfg['source_sha256'].items():assert sha(root/'source_snapshot'/name)==digest
    for file in ('battery_smetimes/conditional_drift.py','battery_smetimes/tail_boost.py'):
        assert sha(file)==cfg['source_sha256'][Path(file).name]
    reports=json.loads((root/'summary.json').read_text());checked=[]
    for plan in cfg['plans']:
        cells=load_cells(plan);assert not set(cells)&set(plan['test'])
        assert sha(source/plan['name']/'oof_residuals.npz')==cfg['oof_sha256'][plan['name']]
        assert sha(source/plan['name']/'base.joblib')==cfg['base_sha256'][plan['name']]
        residuals={}
        with np.load(source/plan['name']/'oof_residuals.npz') as a:
            for key in plan['train']:residuals[key]={f:a[key+'__'+f] for f in ('cycle','truth','prediction')}
        for index,parameters in enumerate(cfg['candidates']):
            dest=root/plan['name']/f'candidate_{index}';b=joblib.load(dest/'model.joblib')
            assert b['train_ids']==plan['train'] and b['base']['train_ids']==plan['train']
            original=joblib.load(source/plan['name']/'base.joblib')
            np.testing.assert_array_equal(b['base']['imputer'].statistics_,original['imputer'].statistics_)
            preprocessing(b['base'],cells)
            if parameters is None:assert b['head'] is None
            else:
                head=b['head'];assert head['degree']==parameters['degree'] and head['components']==parameters['components']
                encoder=head['encoder'];assert encoder['train_ids']==plan['train'] and head['train_ids']==plan['train']
                features=np.stack([prefix(cells[k]) for k in plan['train']]);expected=[]
                for j in range(features.shape[1]):
                    finite=features[:,j][np.isfinite(features[:,j])];expected.append(np.median(finite) if len(finite) else 0)
                np.testing.assert_allclose(expected,encoder['imputer'].statistics_,atol=1e-12,rtol=0)
                filled=np.where(np.isnan(features),np.array(expected)[None],features)
                mean=filled.mean(axis=0);var=filled.var(axis=0)
                np.testing.assert_allclose(mean,encoder['scaler'].mean_,atol=1e-12,rtol=0)
                np.testing.assert_allclose(var,encoder['scaler'].var_,atol=1e-12,rtol=0)
                # StandardScaler's float64 two-pass error bound also treats
                # near-constant columns as constant, not only exact zero.
                scale=manual_scale(var,mean,len(plan['train']))
                np.testing.assert_allclose(scale,encoder['scaler'].scale_,atol=1e-12,rtol=0)
                z=(filled-mean)/scale
                pca=encoder['pca'];n=min(parameters['components'],len(plan['train'])-1,z.shape[1])
                assert pca.n_components_==n and not pca.whiten
                centered=z-z.mean(axis=0);components=pca.components_
                np.testing.assert_allclose(pca.mean_,z.mean(axis=0),atol=1e-12,rtol=0)
                np.testing.assert_allclose(components@components.T,np.eye(n),atol=1e-10,rtol=0)
                singular=np.linalg.svd(centered,compute_uv=False)[:n]
                np.testing.assert_allclose(pca.singular_values_,singular,atol=1e-9,rtol=1e-10)
                projected=centered@components.T
                np.testing.assert_allclose(projected.T@projected,np.diag(singular**2),atol=1e-8,rtol=1e-10)
                xs=[];ys=[];ws=[]
                for key in plan['train']:
                    item=residuals[key];x=item['cycle'];xs.append(query(encoder,cells[key],x,head['degree']))
                    ys.append(item['truth']-item['prediction']);w=np.linspace(1,3,len(x));ws.append(w/w.sum())
                x,y,w=np.concatenate(xs),np.concatenate(ys),np.concatenate(ws);w*=len(w)/w.sum()
                np.testing.assert_allclose(head['scaler'].mean_,x.mean(axis=0),atol=1e-10,rtol=0)
                np.testing.assert_allclose(head['scaler'].var_,x.var(axis=0),atol=1e-10,rtol=0)
                z=head['scaler'].transform(x);zm=np.average(z,axis=0,weights=w);ym=np.average(y,weights=w)
                zz=z-zm;coef=np.linalg.solve((zz*w[:,None]).T@zz+head['alpha']*np.eye(z.shape[1]),(zz*w[:,None]).T@(y-ym))
                np.testing.assert_allclose(coef,head['model'].coef_,atol=1e-10,rtol=0)
                np.testing.assert_allclose(ym-zm@coef,head['model'].intercept_,atol=1e-10,rtol=0)
            stored={r['cell_id']:r for r in json.loads((dest/'validation_metrics.json').read_text())};assert set(stored)==set(plan['validation']);scores=[]
            with np.load(dest/'validation_predictions.npz') as a:
                for key in plan['validation']:
                    x,y=labels(cells[key]);p=predict(b,clean(cells[key]),x)
                    for f,v in [('cycle',x),('truth',y),('prediction',p)]:np.testing.assert_allclose(v,a[key+'__100__'+f],atol=1e-12,rtol=0)
                    values=metric(x,y,p)
                    for f,v in values.items():np.testing.assert_allclose(v,stored[key][f],atol=1e-10,rtol=0)
                    scores.append(sum(values[f] for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp'))/4)
                    checked.append(dict(cell_id=key,candidate=index,points=len(x),model_sha256=sha(dest/'model.joblib'),prediction_sha256=sha(dest/'validation_predictions.npz')))
            report=next(r for r in reports if r['fold']==plan['name'] and r['candidate']==index)
            np.testing.assert_allclose(np.mean(scores)+.25*np.quantile(scores,.9),report['score'],atol=1e-10,rtol=0)
            print('PASS '+plan['name']+' candidate='+str(index),flush=True)
    save_json(root/'independent_verification.json',dict(passed=True,rows=checked,training_only_pca_algebra_checked=True,
        ridge_closed_form_checked=True,source_sha256=sha(__file__),outer_evaluated=False,goal_complete=False))
    print('CURVES='+str(len(checked)))


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
