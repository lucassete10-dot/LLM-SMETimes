"""Independent float targets/median/ET refit, all curves and goal arithmetic."""
import argparse,json
from pathlib import Path
import joblib,numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from threadpoolctl import threadpool_limits
from battery_smetimes.fade_rate import predict
from battery_smetimes.tail_boost import inputs,labels
from scripts.train_curve_fidelity import load_subset,AUDIT
from scripts.train_crossfit_drift_smoke import clean
from scripts.prepare_same_type_forecast import save_json,sha
from scripts.audit_tail_numeric_gates import metric


def stats(rows):
    seeds=sorted({r['seed'] for r in rows});ids=sorted({r['cell_id'] for r in rows});assert len(seeds)==5 and len(ids)==4
    fields=tuple(metric(np.arange(2),np.ones(2),np.ones(2)));cells=[]
    for key in ids:
        rr=[r for r in rows if r['cell_id']==key];assert len(rr)==5 and {r['seed'] for r in rr}==set(seeds)
        v=dict(cell_id=key,**{f:float(np.mean([r[f] for r in rr])) for f in fields})
        v['joint_pass']=all(v[f]<=bound for f,bound in [('mape_percent',6),('late_mape',8),('endpoint_pp',8),('drop_error_pp',5)])
        v['severe_flat_failure']=v['truth_drop_pp']>=5 and v['prediction_drop_pp']<.3*v['truth_drop_pp'];cells.append(v)
    result={f:dict(mean=float(np.mean([c[f] for c in cells])),seed_sd=float(np.std([np.mean([r[f] for r in rows if r['seed']==s]) for s in seeds],ddof=1))) for f in fields}
    result.update(cells=4,joint_pass_fraction=sum(c['joint_pass'] for c in cells)/4,severe_flat_fraction=sum(c['severe_flat_failure'] for c in cells)/4,endpoint_p90=float(np.quantile([c['endpoint_pp'] for c in cells],.9)),drop_error_p90=float(np.quantile([c['drop_error_pp'] for c in cells],.9)))
    return result,cells


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);root=parser.parse_args().run
    cfg=json.loads((root/'config.json').read_text());progress=json.loads((root/'progress.json').read_text())
    assert progress['training_complete'] and progress['outer_complete'] and cfg['parameters']==dict(learned_anchor=False)
    assert cfg['seeds']==list(range(20261005,20261010)) and len(cfg['plans'])==4
    baseline=Path(cfg['baseline']);assert sha(baseline/'config.json')==cfg['baseline_config_sha256']
    assert cfg['plans']==json.loads((baseline/'config.json').read_text())['plans']
    screened=Path(cfg['screened_run']);assert sha(screened/'config.json')==cfg['screened_config_sha256']
    assert json.loads((screened/'screening_decision.json').read_text())['expand_formal']
    assert sha('docs/soh_nasa_rate_formal_20261005.md')==cfg['preregistration_sha256']
    for name,digest in cfg['source_sha256'].items():assert sha(root/'source_snapshot'/name)==digest
    for name in ('fade_rate','conditional_drift','crossfit_drift','tail_boost','forecast','research_baselines'):
        assert sha(Path('battery_smetimes')/(name+'.py'))==cfg['source_sha256'][name+'.py']
    registry=json.loads((AUDIT/'cell_registry.json').read_text())['cells'];members={k for p in cfg['plans'] for k in p['train']+p['validation']+p['test']}
    assert set(cfg['prepared_sha256'])==members
    for key in members:assert cfg['prepared_sha256'][key]==registry[key]['prepared_sha256']
    lock=json.loads((root/'inner_lock.json').read_text());assert lock['fixed_policy'] and not lock['outer_loaded'] and len(lock['models'])==20
    assert sha(root/'inner_lock.json')==progress['inner_lock_sha256'];checked=[];outer_rows=[];old_rows=[];evidence=[]
    for plan in cfg['plans']:
        assert not set(plan['train'])&set(plan['validation']) and not set(plan['train']+plan['validation'])&set(plan['test'])
        for phase,ids,roles in [('inner',plan['train'],[('training',plan['train']),('validation',plan['validation'])]),('outer',plan['train']+plan['validation'],[('training',plan['train']+plan['validation']),('test',plan['test'])])]:
            loaded=set(ids+[k for _,rr in roles for k in rr]);cells=load_subset(list(loaded));assert set(cells)==loaded
            if phase=='inner':assert not set(cells)&set(plan['test'])
            for seed in cfg['seeds']:
                folder=root/plan['name']/str(seed)/phase;b=joblib.load(folder/'model.joblib');assert b['train_ids']==ids and b['seed']==seed and not b['learned_anchor']
                np.testing.assert_array_equal(b['training_anchors'],np.ones(len(ids)))
                if phase=='inner':
                    record=next(r for r in lock['models'] if r['fold']==plan['name'] and r['seed']==seed)
                    assert sha(folder/'model.joblib')==record['model_sha256'] and sha(folder/'validation_predictions.npz')==record['validation_sha256']
                xs=[];rates=[];weights=[]
                for key in ids:
                    c,y=labels(cells[key]);idx=np.unique(np.linspace(0,len(c)-1,min(96,len(c))).round().astype(int));xs.append(inputs(clean(cells[key]),c[idx]))
                    r=(1-y[idx])/(1+(c[idx]-100)/100);np.testing.assert_allclose(r,100*(1-y[idx])/c[idx],atol=1e-14,rtol=0);rates.append(r);w=np.linspace(1,3,len(idx));weights.append(w/w.sum())
                x=np.concatenate(xs);r=np.concatenate(rates);w=np.concatenate(weights);w*=len(w)/w.sum();median=[]
                for col in x.T:
                    finite=col[np.isfinite(col)];median.append(np.median(finite) if len(finite) else 0.)
                median=np.asarray(median);np.testing.assert_allclose(median,b['imputer'].statistics_,atol=1e-12,rtol=0)
                z=np.where(np.isfinite(x),x,median);replay=ExtraTreesRegressor(n_estimators=128,min_samples_leaf=1,max_depth=None,random_state=seed,n_jobs=2).fit(z,r,sample_weight=w)
                assert b['model'].n_estimators==128 and b['model'].min_samples_leaf==1
                for role,keys in roles:
                    file=folder/(role+'_predictions.npz');rows={v['cell_id']:v for v in json.loads((folder/(role+'_metrics.json')).read_text())};assert set(rows)==set(keys)
                    with np.load(file) as arrays:
                        assert len(arrays.files)==3*len(keys)
                        for key in keys:
                            cc=clean(cells[key]);c,y=labels(cells[key]);p=predict(b,cc,c)
                            np.testing.assert_array_equal(c,arrays[key+'__100__cycle']);np.testing.assert_array_equal(y,arrays[key+'__100__truth']);np.testing.assert_allclose(p,arrays[key+'__100__prediction'],atol=1e-12,rtol=0)
                            independent=1-c/100*replay.predict(b['imputer'].transform(inputs(cc,c)));np.testing.assert_allclose(p,independent,atol=1e-12,rtol=0)
                            poison=dict(cc,cycle=np.r_[cc['cycle'],101],raw=np.vstack([cc['raw'],np.full((1,cc['raw'].shape[1]),np.inf)]),soh=np.inf,cycle_life=np.inf)
                            np.testing.assert_allclose(p,predict(b,poison,c),atol=1e-12,rtol=0)
                            try:predict(b,dict(cc,physical_family='different'),c)
                            except ValueError:pass
                            else:raise AssertionError('family gate failed')
                            values=metric(c,y,p)
                            for field,value in values.items():np.testing.assert_allclose(value,rows[key][field],atol=1e-10,rtol=0)
                            if role=='test':outer_rows.append(dict(cell_id=key,seed=seed,**values))
                            checked.append(dict(fold=plan['name'],seed=seed,phase=phase,role=role,cell_id=key,model_sha256=sha(folder/'model.joblib'),arrays_sha256=sha(file)))
                print('PASS '+plan['name']+' '+phase+' '+str(seed),flush=True)
        for seed in range(20261004,20261009):
            file=baseline/plan['name']/str(seed)/'predictions.npz'
            with np.load(file) as arrays:
                for key in plan['test']:
                    c,y,p=[arrays[key+'__100__'+f] for f in ('cycle','truth','prediction')];cc,yy=labels(cells[key]);np.testing.assert_array_equal(c,cc);np.testing.assert_array_equal(y,yy)
                    old_rows.append(dict(cell_id=key,seed=seed,**metric(c,y,p)))
            evidence.append(dict(file=str(file.resolve()),sha256=sha(file)))
    new,cells=stats(outer_rows);old,_=stats(old_rows)
    checks=dict(full_mape=new['mape_percent']['mean']<=6,late_mape=new['late_mape']['mean']<=6,rmse=new['rmse_soh']['mean']<=.04,endpoint_mean=new['endpoint_pp']['mean']<=5,drop_mean=new['drop_error_pp']['mean']<=4,joint_fraction=new['joint_pass_fraction']==1,severe_fraction=new['severe_flat_fraction']<=.1,full_no_regression=new['mape_percent']['mean']<=old['mape_percent']['mean']+.5)
    for f in ('endpoint_p90','drop_error_p90'):checks[f]=new[f]<=(3 if old[f]<=3 else .8*old[f])
    saved=json.loads((root/'summary.json').read_text());assert saved['gates']==checks and saved['numeric_gates_passed']==all(checks.values())
    for label,expected in [('new',new),('baseline',old)]:
        for field,value in expected.items():
            if isinstance(value,dict):
                for k,v in value.items():np.testing.assert_allclose(v,saved[label][field][k],atol=1e-10,rtol=0)
            else:np.testing.assert_allclose(value,saved[label][field],atol=1e-10,rtol=0)
    assert len(checked)==140 and len(outer_rows)==20
    save_json(root/'independent_verification.json',dict(passed=True,models_retrained=40,checked=checked,new=new,baseline=old,cells=cells,gates=checks,numeric_gates_passed=all(checks.values()),baseline_evidence=evidence,source_sha256=sha(__file__),goal_complete=False))
    print('AUDIT_PASS '+str(sum(checks.values()))+'/10',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
