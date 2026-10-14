"""Refit every formal NASA component and audit frozen selections/full curves."""
import argparse,json
from pathlib import Path
import joblib,numpy as np
from threadpoolctl import threadpool_limits
from battery_smetimes.pooled_anchor import fit as pooled_fit
from battery_smetimes.fade_rate import fit as rate_fit
from battery_smetimes.decoupled_anchor import make,predict
from battery_smetimes.tail_boost import labels
from scripts.train_curve_fidelity import load_subset,AUDIT
from scripts.train_crossfit_drift_smoke import clean
from scripts.prepare_same_type_forecast import save_json,sha
from scripts.audit_tail_numeric_gates import metric
from scripts.verify_nasa_rate_formal import stats


def median_check(x,actual):
    x=np.asarray(x,dtype=float);med=[]
    for col in x.T:
        v=col[np.isfinite(col)];med.append(np.median(v) if len(v) else 0.)
    np.testing.assert_allclose(med,actual,atol=1e-12,rtol=0)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);root=parser.parse_args().run;cfg=json.loads((root/'config.json').read_text());progress=json.loads((root/'progress.json').read_text());assert progress['training_complete']
    source=Path(cfg['source_run']);assert sha(source/'config.json')==cfg['source_config_sha256'] and json.loads((source/'independent_verification.json').read_text())['passed']
    assert cfg['plans']==json.loads((source/'config.json').read_text())['plans'] and cfg['seeds']==list(range(20261005,20261010))
    assert cfg['candidates']==[dict(kind='ridge',alpha=a) for a in (1,10,100)]+[dict(kind='extra_trees',alpha=None)]
    assert sha('docs/soh_decoupled_anchor_formal_20261005.md')==cfg['preregistration_sha256']
    for name,digest in cfg['source_sha256'].items():assert sha(root/'source_snapshot'/name)==digest
    for name in ('decoupled_anchor','pooled_anchor','fade_rate','conditional_drift','crossfit_drift','tail_boost','forecast','research_baselines'):
        assert sha(Path('battery_smetimes')/(name+'.py'))==cfg['source_sha256'][name+'.py']
    baseline=Path(cfg['baseline']);assert sha(baseline/'config.json')==cfg['baseline_config_sha256']
    registry=json.loads((AUDIT/'cell_registry.json').read_text())['cells'];members={k for p in cfg['plans'] for role in ('train','validation','test') for k in p[role]}
    assert set(cfg['prepared_sha256'])==members
    for key in members:assert cfg['prepared_sha256'][key]==registry[key]['prepared_sha256']
    assert sha(root/'inner_lock.json')==progress['inner_lock_sha256'];lock=json.loads((root/'inner_lock.json').read_text());assert not lock['outer_loaded'] and len(lock['selections'])==20
    checked=[];outer_rows=[];old_rows=[];models=0
    for plan in cfg['plans']:
        assert not set(plan['train'])&set(plan['validation']) and not set(plan['train']+plan['validation'])&set(plan['test'])
        for phase,ids in [('inner',plan['train']),('outer',plan['train']+plan['validation'])]:
            cells=load_subset(ids+(plan['validation'] if phase=='inner' else plan['test']))
            if phase=='inner':assert not set(cells)&set(plan['test'])
            for seed in cfg['seeds']:
                selection=root/plan['name']/str(seed)/'inner'/'selection.json';sel=json.loads(selection.read_text());record=next(r for r in lock['selections'] if r['fold']==plan['name'] and r['seed']==seed)
                assert sha(selection)==record['selection_sha256'] and sel['selected']==record['selected'] and sel['selected']==min(sel['candidates'],key=lambda r:r['score'])
                choices=list(enumerate(cfg['candidates'])) if phase=='inner' else [(sel['selected']['candidate'],sel['selected']['parameters'])]
                rate=rate_fit(cells,ids,False,seed,[dict(cell_id=k) for k in ids])
                for index,params in choices:
                    folder=root/plan['name']/str(seed)/phase
                    if phase=='inner':folder=folder/('candidate_'+str(index))
                    b=joblib.load(folder/'model.joblib');p=b['pooled'];r=b['rate']
                    for part in (p,r):assert part['train_ids']==ids and part['seed']==seed and part['family']==b['family']
                    assert b['train_ids']==ids and p['kind']==params['kind'] and p['alpha']==params['alpha'] and not r['learned_anchor']
                    raw=[];anchors=[];membership=[]
                    for key in ids:
                        c=cells[key];lookup={float(n):float(y) for n,y in zip(c['label_cycle'],c['soh']) if 0<n<=100 and np.isfinite(y) and y>0};take=[i for i,n in enumerate(c['cycle']) if 0<n<=100 and float(n) in lookup]
                        cycle=np.asarray(c['cycle'])[take];yy=np.array([lookup[float(n)] for n in cycle]);raw.append(np.asarray(c['raw'],dtype=float)[take]);anchors.append(float(np.median(yy[-3:])));membership.append(dict(cell_id=key,cycles=cycle.tolist(),rows=len(yy)))
                    assert p['supervision']==membership;np.testing.assert_array_equal(anchors,p['anchors']);median_check(np.concatenate(raw),p['imputer'].statistics_)
                    replay=make(pooled_fit(cells,ids,seed=seed,**params),rate);models+=1
                    roles=[('training',ids),('validation',plan['validation'])] if phase=='inner' else [('training',ids),('test',plan['test'])];scores=[]
                    for role,keys in roles:
                        file=folder/(role+'_predictions.npz');rows={v['cell_id']:v for v in json.loads((folder/(role+'_metrics.json')).read_text())};assert set(rows)==set(keys)
                        with np.load(file) as arrays:
                            assert len(arrays.files)==3*len(keys)
                            for key in keys:
                                c=clean(cells[key]);x,y=labels(cells[key]);prediction=predict(b,c,x)
                                np.testing.assert_array_equal(x,arrays[key+'__100__cycle']);np.testing.assert_array_equal(y,arrays[key+'__100__truth']);np.testing.assert_allclose(prediction,arrays[key+'__100__prediction'],atol=1e-12,rtol=0)
                                np.testing.assert_allclose(prediction,predict(replay,c,x),atol=1e-12,rtol=0)
                                poison=dict(c,cycle=np.r_[c['cycle'],101],raw=np.vstack([c['raw'],np.full((1,c['raw'].shape[1]),np.inf)]),soh=np.inf,cycle_life=np.inf)
                                np.testing.assert_allclose(prediction,predict(b,poison,x),atol=1e-12,rtol=0)
                                try:predict(b,dict(c,physical_family='different'),x)
                                except ValueError:pass
                                else:raise AssertionError('family gate failed')
                                values=metric(x,y,prediction)
                                for field,value in values.items():np.testing.assert_allclose(value,rows[key][field],atol=1e-10,rtol=0)
                                if role=='validation':scores.append(sum(values[f] for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp'))/4)
                                if role=='test':outer_rows.append(dict(cell_id=key,seed=seed,**values))
                                checked.append(dict(fold=plan['name'],seed=seed,phase=phase,candidate=index,role=role,cell_id=key,model_sha256=sha(folder/'model.joblib'),predictions_sha256=sha(file)))
                    if phase=='inner':
                        sc=next(v for v in sel['candidates'] if v['candidate']==index);assert params==sc['parameters']
                        np.testing.assert_allclose(np.mean(scores)+.25*np.quantile(scores,.9),sc['score'],atol=1e-10,rtol=0)
                        assert sha(folder/'model.joblib')==sc['model_sha256'] and sha(folder/'validation_predictions.npz')==sc['validation_sha256']
                        assert (folder/'validation_predictions.npz').stat().st_mtime_ns<=(root/'inner_lock.json').stat().st_mtime_ns
                    else:assert (root/'inner_lock.json').stat().st_mtime_ns<=(folder/'model.joblib').stat().st_mtime_ns
                print('PASS '+plan['name']+' '+phase+' '+str(seed),flush=True)
        for seed in range(20261004,20261009):
            with np.load(baseline/plan['name']/str(seed)/'predictions.npz') as arrays:
                for key in plan['test']:
                    x,y,p=[arrays[key+'__100__'+f] for f in ('cycle','truth','prediction')];cx,cy=labels(cells[key]);np.testing.assert_array_equal(x,cx);np.testing.assert_array_equal(y,cy);old_rows.append(dict(cell_id=key,seed=seed,**metric(x,y,p)))
    new,cells=stats(outer_rows);old,_=stats(old_rows);checks=dict(full_mape=new['mape_percent']['mean']<=6,late_mape=new['late_mape']['mean']<=6,rmse=new['rmse_soh']['mean']<=.04,endpoint_mean=new['endpoint_pp']['mean']<=5,drop_mean=new['drop_error_pp']['mean']<=4,joint_fraction=new['joint_pass_fraction']==1,severe_fraction=new['severe_flat_fraction']<=.1,full_no_regression=new['mape_percent']['mean']<=old['mape_percent']['mean']+.5)
    for f in ('endpoint_p90','drop_error_p90'):checks[f]=new[f]<=(3 if old[f]<=3 else .8*old[f])
    saved=json.loads((root/'summary.json').read_text());assert saved['gates']==checks and saved['numeric_gates_passed']==all(checks.values())
    for label,expected in [('new',new),('baseline',old)]:
        for field,value in expected.items():
            if isinstance(value,dict):
                for k,v in value.items():np.testing.assert_allclose(v,saved[label][field][k],atol=1e-10,rtol=0)
            else:np.testing.assert_allclose(value,saved[label][field],atol=1e-10,rtol=0)
    assert models==100 and len(checked)==320
    save_json(root/'independent_verification.json',dict(passed=True,retrained_bundles=models,checked=checked,new=new,baseline=old,cells=cells,gates=checks,numeric_gates_passed=all(checks.values()),source_sha256=sha(__file__),goal_complete=False));print('AUDIT_PASS '+str(sum(checks.values()))+'/10',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
