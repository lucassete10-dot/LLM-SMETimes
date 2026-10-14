"""Replay training determinism, independent curve equation and full metrics."""
import argparse,json
from pathlib import Path
import joblib,numpy as np,torch
from threadpoolctl import threadpool_limits
from battery_smetimes.monotone_network import predict,CurveNet
from battery_smetimes.warm_monotone import fit,inverse_bias
from battery_smetimes.conditional_drift import prefix
from battery_smetimes.tail_boost import labels,is_nca
from scripts.train_tail_boost import load_cells
from scripts.train_crossfit_drift_smoke import clean
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
    for file in ('battery_smetimes/warm_monotone.py','battery_smetimes/monotone_network.py','battery_smetimes/conditional_drift.py','battery_smetimes/tail_boost.py'):
        assert sha(file)==cfg['source_sha256'][Path(file).name]
    reports=json.loads((root/'summary.json').read_text());checked=[];models=[]
    for plan in cfg['plans']:
        cells=load_cells(plan);assert not set(cells)&set(plan['test'])
        for index,params in enumerate(cfg['candidates']):
            folder=root/plan['name']/f'candidate_{index}';b=joblib.load(folder/'model.joblib');assert b['train_ids']==plan['train']
            if params is None:preprocessing(b['base'],cells)
            else:
                assert b['width']==params['width'] and b['decay']==params['decay'] and b['epochs']==params['epochs']
                assert b['warm_start'] and {r['cell_id'] for r in b['training_parameter_fits']}==set(plan['train'])
                prior=np.median([[r[f] for f in ('anchor','rate','acceleration','knee_h')] for r in b['training_parameter_fits']],axis=0)
                np.testing.assert_array_equal(prior,b['initialization_prior'])
                np.testing.assert_array_equal(inverse_bias(prior),b['initialization_bias'])
                initialization=json.loads((folder/'training_initialization.json').read_text())
                assert initialization==dict(prior=b['initialization_prior'],bias=b['initialization_bias'],training_parameter_fits=b['training_parameter_fits'])
                raw=np.stack([prefix(cells[k]) for k in plan['train']]);median=[]
                for j in range(raw.shape[1]):
                    finite=raw[:,j][np.isfinite(raw[:,j])];median.append(np.median(finite) if len(finite) else 0)
                np.testing.assert_allclose(median,b['imputer'].statistics_,atol=1e-12,rtol=0)
                filled=np.where(np.isnan(raw),np.array(median)[None],raw);mean=filled.mean(axis=0);var=filled.var(axis=0)
                np.testing.assert_allclose(mean,b['scaler'].mean_,atol=1e-12,rtol=0)
                np.testing.assert_allclose(var,b['scaler'].var_,atol=1e-12,rtol=0)
                np.testing.assert_allclose(manual_scale(var,mean,len(plan['train'])),b['scaler'].scale_,atol=1e-12,rtol=0)
                assert b['parameter_count']==sum(v.size for v in b['state'].values())
                assert len(b['history'])==params['epochs'] and np.isfinite(b['history']).all() and np.all(np.array(b['history'])>=0)
                assert json.loads((folder/'epoch_loss.json').read_text())==b['history']
                # Actual deterministic full retraining, not just loading state.
                replay=fit(cells,plan['train'],seed=cfg['seed'],**params)
                for name,array in b['state'].items():np.testing.assert_array_equal(array,replay['state'][name])
                np.testing.assert_array_equal(b['history'],replay['history'])
                assert b['training_parameter_fits']==replay['training_parameter_fits']
                models.append(dict(fold=plan['name'],candidate=index,parameter_count=b['parameter_count'],training_state_exactly_reproduced=True))
            stored={r['cell_id']:r for r in json.loads((folder/'validation_metrics.json').read_text())};assert set(stored)==set(plan['validation']);scores=[]
            with np.load(folder/'validation_predictions.npz') as a:
                for key in plan['validation']:
                    x,y=labels(cells[key]);cc=clean(cells[key]);p=predict(b,cc,x)
                    for field,v in [('cycle',x),('truth',y),('prediction',p)]:np.testing.assert_allclose(v,a[key+'__100__'+field],atol=1e-12,rtol=0)
                    if params is not None:
                        assert np.all(np.diff(p)<=1e-7)
                        z=b['scaler'].transform(b['imputer'].transform(prefix(cc)[None]))
                        net=CurveNet(b['input_width'],b['width']);net.load_state_dict({k:torch.from_numpy(v) for k,v in b['state'].items()})
                        with torch.no_grad():anchor,rate,accel,knee=net.parameters_for(torch.tensor(z,dtype=torch.float32))[0].numpy().astype(float)
                        assert .5<=anchor<=1.25 and min(rate,accel,knee)>=0
                        h=(x-100)/100;hinge=.1*np.logaddexp(0,(h-knee)/.1)
                        np.testing.assert_allclose(p,anchor-rate*h-accel*hinge**2,atol=5e-6,rtol=0)
                        if is_nca(cc):
                            poison=dict(cc,early_cycle=np.r_[cc['early_cycle'],101],early_raw=np.vstack([cc['early_raw'],np.full((1,cc['early_raw'].shape[1]),np.inf)]),truth=np.inf,cycle_life=np.inf)
                        else:
                            poison=dict(cc,cycle=np.r_[cc['cycle'],101],raw=np.vstack([cc['raw'],np.full((1,cc['raw'].shape[1]),np.inf)]),soh=np.inf,cycle_life=np.inf)
                        np.testing.assert_array_equal(p,predict(b,poison,x))
                    values=metric(x,y,p)
                    for field,v in values.items():np.testing.assert_allclose(v,stored[key][field],atol=1e-10,rtol=0)
                    scores.append(sum(values[f] for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp'))/4)
                    checked.append(dict(cell_id=key,candidate=index,points=len(x),model_sha256=sha(folder/'model.joblib'),prediction_sha256=sha(folder/'validation_predictions.npz')))
            report=next(r for r in reports if r['fold']==plan['name'] and r['candidate']==index)
            np.testing.assert_allclose(np.mean(scores)+.25*np.quantile(scores,.9),report['score'],atol=1e-10,rtol=0)
            print('PASS '+plan['name']+' candidate='+str(index),flush=True)
    save_json(root/'independent_verification.json',dict(passed=True,rows=checked,models=models,training_only_scaler=True,
        independent_float64_curve_equation_tolerance=5e-6,torch_version=torch.__version__,source_sha256=sha(__file__),
        outer_evaluated=False,goal_complete=False))
    print('CURVES='+str(len(checked)))


if __name__=='__main__':
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):main()
