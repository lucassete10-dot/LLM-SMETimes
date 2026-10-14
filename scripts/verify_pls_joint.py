"""Full deterministic retraining, train-statistic/PLS algebra and independent curves."""
import argparse,json
from pathlib import Path
import joblib,numpy as np,torch
from scipy.special import expit
from threadpoolctl import threadpool_limits
from battery_smetimes.pls_joint import fit,predict
from battery_smetimes.conditional_drift import prefix
from battery_smetimes.tail_boost import labels,is_nca
from scripts.verify_conditional_drift import manual_scale
from scripts.train_crossfit_drift_smoke import clean
from scripts.train_tail_boost import load_cells
from scripts.prepare_same_type_forecast import save_json,sha
from scripts.audit_tail_numeric_gates import metric


def main():
    torch.set_num_threads(2)
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    root=parser.parse_args().run;cfg=json.loads((root/'config.json').read_text())
    assert json.loads((root/'progress.json').read_text())['training_complete']
    source=Path(cfg['source_run']);assert sha(source/'config.json')==cfg['source_config_sha256']
    assert json.loads((source/'independent_verification.json').read_text())['passed']
    assert sha('docs/soh_pls_joint_stage_20261005.md')==cfg['preregistration_sha256']
    for name,digest in cfg['source_sha256'].items():assert sha(root/'source_snapshot'/name)==digest
    for f in ('pls_joint','parameter_pls','parameter_ridge','conditional_drift','tail_boost','forecast'):
        path=Path('battery_smetimes')/(f+'.py');assert sha(path)==cfg['source_sha256'][path.name]
    assert cfg['epochs']==2000 and cfg['candidates']==[None]+[dict(components=n,decay=d) for n in (2,4) for d in (.01,.1)]
    reports=json.loads((root/'summary.json').read_text());checked=[]
    for plan in cfg['plans']:
        cells=load_cells(plan);assert set(cells)==set(plan['train']+plan['validation']) and not set(cells)&set(plan['test'])
        assert sha(cfg['base_paths'][plan['name']])==cfg['base_sha256'][plan['name']]
        for index,params in enumerate(cfg['candidates']):
            folder=root/plan['name']/f'candidate_{index}';b=joblib.load(folder/'model.joblib');assert b['train_ids']==plan['train']
            if params is not None:
                path=cfg['encoder_paths'][plan['name']][str(params['components'])];assert sha(path)==cfg['encoder_sha256'][plan['name']][str(params['components'])]
                original=joblib.load(path);e=b['encoder'];assert e['train_ids']==plan['train'] and e['logarithmic']
                assert b['epochs']==2000 and b['decay']==params['decay'] and b['seed']==cfg['seed']
                for name in ('coef_','x_rotations_','_x_mean','_x_std'):np.testing.assert_array_equal(getattr(e['model'],name),getattr(original['model'],name))
                raw=np.stack([prefix(clean(cells[k])) for k in plan['train']]);med=[]
                for j in range(raw.shape[1]):
                    v=raw[:,j][np.isfinite(raw[:,j])];med.append(np.median(v) if len(v) else 0)
                med=np.asarray(med);np.testing.assert_allclose(med,e['imputer'].statistics_,atol=1e-12,rtol=0)
                raw=np.where(np.isfinite(raw),raw,med);mean=raw.mean(axis=0);var=raw.var(axis=0);scale=manual_scale(var,mean,len(raw))
                for expected,actual in [(mean,e['scaler'].mean_),(var,e['scaler'].var_),(scale,e['scaler'].scale_)]:np.testing.assert_allclose(expected,actual,atol=1e-10,rtol=1e-10)
                latent=(((raw-mean)/scale-e['model']._x_mean)/e['model']._x_std)@e['model'].x_rotations_
                lm=latent.mean(axis=0);lv=latent.var(axis=0);ls=manual_scale(lv,lm,len(latent))
                for expected,actual in [(lm,b['latent_scaler'].mean_),(lv,b['latent_scaler'].var_),(ls,b['latent_scaler'].scale_)]:np.testing.assert_allclose(expected,actual,atol=1e-10,rtol=1e-10)
                replay=fit(cells,plan['train'],original,params['decay'],cfg['seed'],epochs=2000)
                for name in ('coefficient','intercept','initialization','history'):np.testing.assert_array_equal(b[name],replay[name])
                np.testing.assert_array_equal(b['history'],json.loads((folder/'loss_history.json').read_text()));assert len(b['history'])==2000
                assert b['head_parameter_count']==4*(latent.shape[1]+1)
                z=(latent-lm)/ls;u0=np.column_stack([z,np.ones(len(z))])@b['initialization']
                reference=((raw-mean)/scale-e['model']._x_mean)@e['model'].coef_.T+e['model'].intercept_
                np.testing.assert_allclose(u0,reference*e['target_scale']+e['target_mean'],atol=1e-9,rtol=1e-9)
            report=next(r for r in reports if r['fold']==plan['name'] and r['candidate']==index)
            for role,ids in [('training',plan['train']),('validation',plan['validation'])]:
                rows={r['cell_id']:r for r in json.loads((folder/(role+'_metrics.json')).read_text())};assert set(rows)==set(ids)
                parameter_rows={r['cell_id']:r['values'] for r in json.loads((folder/(role+'_predicted_parameters.json')).read_text())}
                assert set(parameter_rows)==(set(ids) if params is not None else set());values_all=[];scores=[]
                with np.load(folder/(role+'_predictions.npz')) as arrays:
                    assert len(arrays.files)==3*len(ids)
                    for key in ids:
                        cc=clean(cells[key]);cycle,y=labels(cells[key]);p=predict(b,cc,cycle)
                        np.testing.assert_array_equal(cycle,arrays[key+'__100__cycle']);np.testing.assert_array_equal(y,arrays[key+'__100__truth'])
                        np.testing.assert_allclose(p,arrays[key+'__100__prediction'],atol=1e-12,rtol=0)
                        if params is not None:
                            q=prefix(cc);q=np.where(np.isfinite(q),q,med);lat=(((q-mean)/scale-e['model']._x_mean)/e['model']._x_std)@e['model'].x_rotations_
                            u=((lat-lm)/ls)@b['coefficient'].T+b['intercept'];a=.5+.75*expit(u[0]);rates=np.exp(u[1:])
                            np.testing.assert_allclose(np.r_[a,rates],parameter_rows[key],atol=1e-9,rtol=1e-9)
                            h=(cycle-100)/100;hinge=.1*np.logaddexp(0,(h-rates[2])/.1)
                            np.testing.assert_allclose(p,a-rates[0]*h-rates[1]*hinge**2,atol=1e-9,rtol=1e-9);assert np.all(np.diff(p)<=1e-12)
                        if role=='validation':
                            if is_nca(cc):poison=dict(cc,early_cycle=np.r_[cc['early_cycle'],101],early_raw=np.vstack([cc['early_raw'],np.full((1,cc['early_raw'].shape[1]),np.inf)]),truth=np.inf,cycle_life=np.inf)
                            else:poison=dict(cc,cycle=np.r_[cc['cycle'],101],raw=np.vstack([cc['raw'],np.full((1,cc['raw'].shape[1]),np.inf)]),soh=np.inf,cycle_life=np.inf)
                            np.testing.assert_allclose(p,predict(b,poison,cycle),atol=1e-12,rtol=0)
                            try:predict(b,dict(cc,physical_family='different'),cycle)
                            except ValueError:pass
                            else:raise AssertionError('family isolation failed')
                        values=metric(cycle,y,p)
                        for f,v in values.items():np.testing.assert_allclose(v,rows[key][f],atol=1e-10,rtol=0)
                        values_all.append(values);scores.append(sum(values[f] for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp'))/4)
                        checked.append(dict(cell_id=key,role=role,candidate=index,points=len(cycle),model_sha256=sha(folder/'model.joblib'),arrays_sha256=sha(folder/(role+'_predictions.npz'))))
                if role=='training':np.testing.assert_allclose(np.mean([v['mape_percent'] for v in values_all]),report['training_mape'],atol=1e-10,rtol=0)
                else:
                    np.testing.assert_allclose(np.mean(scores)+.25*np.quantile(scores,.9),report['score'],atol=1e-10,rtol=0)
                    for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp'):np.testing.assert_allclose(np.mean([v[f] for v in values_all]),report[f],atol=1e-10,rtol=0)
            print('PASS '+plan['name']+' candidate='+str(index),flush=True)
    assert len(checked)==990
    save_json(root/'independent_verification.json',dict(passed=True,checked=checked,full_epoch_retrains=16,source_sha256=sha(__file__),outer_evaluated=False,goal_complete=False));print('CURVES='+str(len(checked)))


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
