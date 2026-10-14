"""Replay original train-only component models and independent blend metrics."""
import argparse,json
from pathlib import Path
import joblib,numpy as np
from threadpoolctl import threadpool_limits
from battery_smetimes.joint_blend import make,predict
from battery_smetimes.pls_joint import predict as joint_predict
from battery_smetimes.tail_selection import choose,predict as primary_predict
from battery_smetimes.research_baselines import predict_baseline
from battery_smetimes.tail_boost import labels,is_nca
from scripts.train_crossfit_drift_smoke import clean
from scripts.train_tail_boost import load_cells
from scripts.prepare_same_type_forecast import save_json,sha
from scripts.audit_tail_numeric_gates import metric


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    root=parser.parse_args().run;cfg=json.loads((root/'config.json').read_text())
    assert cfg['weights']==[0,.25,.5,.75,1] and cfg['joint_candidates']=={'MIT':3,'XJTU':4,'Zhu2022_NCA':3,'NASA':2}
    assert json.loads((root/'progress.json').read_text())['training_complete']
    source=Path(cfg['source_run']);assert sha(source/'config.json')==cfg['source_config_sha256']
    assert json.loads((source/'independent_verification.json').read_text())['passed']
    assert sha('docs/soh_joint_blend_stage_20261005.md')==cfg['preregistration_sha256']
    for name,digest in cfg['source_sha256'].items():assert sha(root/'source_snapshot'/name)==digest
    for f in ('joint_blend','pls_joint','tail_selection','research_baselines','forecast'):
        path=Path('battery_smetimes')/(f+'.py');assert sha(path)==cfg['source_sha256'][path.name]
    reports=json.loads((root/'summary.json').read_text());checked=[]
    for plan in cfg['plans']:
        cells=load_cells(plan);assert set(cells)==set(plan['train']+plan['validation']) and not set(cells)&set(plan['test'])
        folder=root/plan['name'];prov=json.loads((folder/'provenance.json').read_text())
        for f in ('primary_config','selection_file','primary_model_file','joint_model_file'):
            digest=prov[('selection_sha256' if f=='selection_file' else f.replace('_file','')+'_sha256')]
            assert sha(prov[f])==digest
        oldcfg=json.loads(Path(prov['primary_config']).read_text());oldplan=next(p for p in oldcfg['plans'] if p['name']==plan['name'])
        for role in ('train','validation','test'):assert oldplan[role]==plan[role]
        stored=joblib.load(folder/'components.joblib');original_joint=joblib.load(prov['joint_model_file']);original_primary=joblib.load(prov['primary_model_file'])
        for b in (stored['primary'],stored['joint'],original_joint,original_primary):assert b['train_ids']==plan['train']
        if plan['dataset']=='NASA':
            selection=json.loads(Path(prov['selection_file']).read_text());idx=oldcfg['candidates'].index(selection['selected']['parameters'])
            assert Path(prov['primary_model_file']).parent.name==f'candidate_{idx}' and original_primary['used_train_ids']==plan['train']
            original_wrapper=dict(kind='research_baseline',inner=original_primary,family=original_primary['physical_family'],train_ids=plan['train'])
        else:
            decision=choose(Path(prov['primary_config']).parent,oldcfg,plan,cfg['seed'],cells);assert decision==prov['decision']
            original_wrapper=dict(kind='tail_primary',mode=decision['selected']['mode'],inner=original_primary,family=original_joint['family'],train_ids=plan['train'])
        for index,w in enumerate(cfg['weights']):
            dest=folder/f'candidate_{index}';manifest=json.loads((dest/'model.json').read_text())
            assert manifest['weight']==w and manifest['train_ids']==plan['train'] and sha(manifest['components_file'])==manifest['components_sha256']
            b=make(stored['primary'],stored['joint'],w);rows={r['cell_id']:r for r in json.loads((dest/'validation_metrics.json').read_text())};assert set(rows)==set(plan['validation'])
            values_all=[];scores=[]
            with np.load(dest/'validation_predictions.npz') as arrays:
                assert len(arrays.files)==5*len(rows)
                for key in plan['validation']:
                    cc=clean(cells[key]);x,y=labels(cells[key])
                    p0=predict_baseline(original_primary,cc,100,x) if plan['dataset']=='NASA' else primary_predict(original_wrapper,cc,x)
                    p1=joint_predict(original_joint,cc,x);p=(1-w)*p0+w*p1
                    for f,a in [('cycle',x),('truth',y)]:np.testing.assert_array_equal(a,arrays[key+'__100__'+f])
                    for f,a in [('prediction',p),('primary',p0),('joint',p1)]:np.testing.assert_allclose(a,arrays[key+'__100__'+f],atol=1e-12,rtol=0)
                    np.testing.assert_allclose(p,predict(b,cc,x),atol=1e-12,rtol=0)
                    if is_nca(cc):poison=dict(cc,early_cycle=np.r_[cc['early_cycle'],101],early_raw=np.vstack([cc['early_raw'],np.full((1,cc['early_raw'].shape[1]),np.inf)]),truth=np.inf,cycle_life=np.inf)
                    else:poison=dict(cc,cycle=np.r_[cc['cycle'],101],raw=np.vstack([cc['raw'],np.full((1,cc['raw'].shape[1]),np.inf)]),soh=np.inf,cycle_life=np.inf)
                    np.testing.assert_allclose(p,predict(b,poison,x),atol=1e-12,rtol=0)
                    try:predict(b,dict(cc,physical_family='different'),x)
                    except ValueError:pass
                    else:raise AssertionError('family isolation failed')
                    values=metric(x,y,p)
                    for f,v in values.items():np.testing.assert_allclose(v,rows[key][f],atol=1e-10,rtol=0)
                    values_all.append(values);scores.append(sum(values[f] for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp'))/4)
                    checked.append(dict(cell_id=key,weight=w,points=len(x),manifest_sha256=sha(dest/'model.json'),arrays_sha256=sha(dest/'validation_predictions.npz')))
            report=next(r for r in reports if r['fold']==plan['name'] and r['candidate']==index)
            np.testing.assert_allclose(np.mean(scores)+.25*np.quantile(scores,.9),report['score'],atol=1e-10,rtol=0)
            for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp'):np.testing.assert_allclose(np.mean([v[f] for v in values_all]),report[f],atol=1e-10,rtol=0)
        expected=min([r for r in reports if r['fold']==plan['name']],key=lambda r:(r['score'],r['weight']))
        assert json.loads((folder/'selection.json').read_text())['selected']==expected
        print('PASS '+plan['name'],flush=True)
    assert len(checked)==215
    save_json(root/'independent_verification.json',dict(passed=True,checked=checked,source_sha256=sha(__file__),outer_evaluated=False,goal_complete=False));print('CURVES='+str(len(checked)))


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
