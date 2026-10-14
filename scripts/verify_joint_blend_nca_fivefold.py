"""Reconstruct all NCA encoders/joint epochs and every train/validation blend."""
import argparse,json
from pathlib import Path
import joblib,numpy as np,torch
from threadpoolctl import threadpool_limits
from battery_smetimes.parameter_pls import fit as pls_fit
from battery_smetimes.pls_joint import fit as joint_fit,predict as joint_predict
from battery_smetimes.warm_monotone import derive_prior
from battery_smetimes.joint_blend import make,predict
from battery_smetimes.tail_selection import choose,predict as primary_predict
from battery_smetimes.conditional_drift import prefix
from scripts.verify_conditional_drift import manual_scale
from battery_smetimes.tail_boost import labels
from scripts.train_crossfit_drift_smoke import clean
from scripts.train_tail_boost import load_cells
from scripts.prepare_same_type_forecast import save_json,sha
from scripts.audit_tail_numeric_gates import metric


def main():
    torch.set_num_threads(2)
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    root=parser.parse_args().run;cfg=json.loads((root/'config.json').read_text())
    assert cfg['weights']==[0,.5] and cfg['components']==4 and cfg['logarithmic'] and cfg['decay']==.01 and cfg['epochs']==2000
    assert len(cfg['plans'])==5 and json.loads((root/'progress.json').read_text())['training_complete']
    source=Path(cfg['source_run']);assert sha(source/'config.json')==cfg['source_config_sha256'];oldcfg=json.loads((source/'config.json').read_text())
    assert sha('docs/soh_joint_blend_nca_fivefold_20261005.md')==cfg['preregistration_sha256']
    for name,digest in cfg['source_sha256'].items():assert sha(root/'source_snapshot'/name)==digest
    for f in ('joint_blend','pls_joint','parameter_pls','warm_monotone','tail_selection','conditional_drift','tail_boost','forecast'):
        p=Path('battery_smetimes')/(f+'.py');assert sha(p)==cfg['source_sha256'][p.name]
    assert cfg['plans']==oldcfg['plans'] and cfg['prepared_sha256']==oldcfg['prepared_sha256']
    reports=json.loads((root/'summary.json').read_text());checked=[]
    for plan in cfg['plans']:
        cells=load_cells(plan);assert set(cells)==set(plan['train']+plan['validation']) and not set(cells)&set(plan['test'])
        folder=root/plan['name'];prov=json.loads((folder/'provenance.json').read_text())
        for f in ('primary_config','selection_file','primary_model_file','joint_model_file'):
            name='selection_sha256' if f=='selection_file' else f.replace('_file','')+'_sha256';assert sha(prov[f])==prov[name]
        decision=choose(source,oldcfg,plan,cfg['seed'],cells);assert decision==prov['decision']
        fits=derive_prior(cells,plan['train'])[1];assert fits==json.loads((folder/'training_parameter_fits.json').read_text())
        encoder=joblib.load(folder/'encoder.joblib');original_encoder=pls_fit(cells,plan['train'],4,True,cfg['seed'],fits)
        np.testing.assert_array_equal(encoder['model'].coef_,original_encoder['model'].coef_)
        raw=np.stack([prefix(clean(cells[k])) for k in plan['train']]);med=[]
        for j in range(raw.shape[1]):
            v=raw[:,j][np.isfinite(raw[:,j])];med.append(np.median(v) if len(v) else 0)
        med=np.asarray(med);np.testing.assert_allclose(med,encoder['imputer'].statistics_,atol=1e-12,rtol=0)
        raw=np.where(np.isfinite(raw),raw,med);mean=raw.mean(axis=0);var=raw.var(axis=0);scale=manual_scale(var,mean,len(raw))
        for expected,actual in [(mean,encoder['scaler'].mean_),(var,encoder['scaler'].var_),(scale,encoder['scaler'].scale_)]:np.testing.assert_allclose(expected,actual,atol=1e-10,rtol=1e-10)
        joint=joblib.load(folder/'joint.joblib');replay=joint_fit(cells,plan['train'],original_encoder,.01,cfg['seed'],epochs=2000)
        for field in ('coefficient','intercept','initialization','history'):np.testing.assert_array_equal(joint[field],replay[field])
        np.testing.assert_array_equal(joint['history'],json.loads((folder/'loss_history.json').read_text()))
        stored=joblib.load(folder/'components.joblib');original_primary=joblib.load(prov['primary_model_file'])
        for b in (stored['primary'],stored['joint'],joint,original_primary):assert b['train_ids']==plan['train']
        primary=dict(kind='tail_primary',mode=decision['selected']['mode'],inner=original_primary,family=joint['family'],train_ids=plan['train'])
        for index,w in enumerate(cfg['weights']):
            dest=folder/f'candidate_{index}';manifest=json.loads((dest/'model.json').read_text())
            assert manifest['weight']==w and manifest['train_ids']==plan['train'] and sha(manifest['components_file'])==manifest['components_sha256']
            b=make(stored['primary'],stored['joint'],w);report=next(r for r in reports if r['fold']==plan['name'] and r['candidate']==index)
            for role,ids in [('training',plan['train']),('validation',plan['validation'])]:
                rows={r['cell_id']:r for r in json.loads((dest/(role+'_metrics.json')).read_text())};assert set(rows)==set(ids)
                values_all=[];scores=[]
                with np.load(dest/(role+'_predictions.npz')) as arrays:
                    assert len(arrays.files)==5*len(rows)
                    for key in ids:
                        cc=clean(cells[key]);x,y=labels(cells[key]);p0=primary_predict(primary,cc,x);p1=joint_predict(replay,cc,x);p=(1-w)*p0+w*p1
                        for f,a in [('cycle',x),('truth',y)]:np.testing.assert_array_equal(a,arrays[key+'__100__'+f])
                        for f,a in [('prediction',p),('primary',p0),('joint',p1)]:np.testing.assert_allclose(a,arrays[key+'__100__'+f],atol=1e-12,rtol=0)
                        np.testing.assert_allclose(p,predict(b,cc,x),atol=1e-12,rtol=0)
                        if role=='validation':
                            poison=dict(cc,early_cycle=np.r_[cc['early_cycle'],101],early_raw=np.vstack([cc['early_raw'],np.full((1,cc['early_raw'].shape[1]),np.inf)]),truth=np.inf,cycle_life=np.inf)
                            np.testing.assert_allclose(p,predict(b,poison,x),atol=1e-12,rtol=0)
                            try:predict(b,dict(cc,physical_family='different'),x)
                            except ValueError:pass
                            else:raise AssertionError('family isolation failed')
                        values=metric(x,y,p)
                        for f,v in values.items():np.testing.assert_allclose(v,rows[key][f],atol=1e-10,rtol=0)
                        values_all.append(values);scores.append(sum(values[f] for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp'))/4)
                        checked.append(dict(cell_id=key,weight=w,role=role,points=len(x),manifest_sha256=sha(dest/'model.json'),arrays_sha256=sha(dest/(role+'_predictions.npz'))))
                if role=='training':np.testing.assert_allclose(np.mean([v['mape_percent'] for v in values_all]),report['training_mape'],atol=1e-10,rtol=0)
                else:
                    np.testing.assert_allclose(np.mean(scores)+.25*np.quantile(scores,.9),report['score'],atol=1e-10,rtol=0)
                    for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp'):np.testing.assert_allclose(np.mean([v[f] for v in values_all]),report[f],atol=1e-10,rtol=0)
        print('PASS '+plan['name'],flush=True)
    assert len(checked)==440
    save_json(root/'independent_verification.json',dict(passed=True,checked=checked,full_epoch_retrains=5,source_sha256=sha(__file__),outer_evaluated=False,goal_complete=False));print('CURVES='+str(len(checked)))


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
