"""Separate neighbor routing/weight/trajectory arithmetic with solver replay."""
import json,sys,copy
from pathlib import Path
import joblib,numpy as np
from threadpoolctl import threadpool_limits
from battery_smetimes.mit_neighbor_templates import fit,predict,route
from battery_smetimes.mit_segment_features import prefix,condition_key
from battery_smetimes.mit_fixed_origin_dynamics import origin
from battery_smetimes.curve_fidelity import targets
from scripts.train_curve_fidelity import load_subset
from scripts.train_crossfit_drift_smoke import clean
from scripts.verify_mit_fixed_origin_dynamics import serial_trees
from scripts.train_nasa_robust_inner import detail
from scripts.verify_nasa_robust_inner import close
from scripts.prepare_same_type_forecast import sha,save_json


def curve(parameters,c):
    a,r,d,t=parameters;h=(c-100)/100
    return a-r*h-d*np.square(np.clip(h-t,0,None))


def main(root):
    cfg=json.loads((root/'config.json').read_text());plan=cfg['plan'];cells=load_subset(plan['train']+plan['validation'])
    assert (len(plan['train']),len(plan['validation']),len(plan['test']))==(83,24,27)
    assert set(cells)==set(plan['train']+plan['validation']) and not set(cells)&set(plan['test'])
    parent=Path(cfg['parent']);assert sha(parent/'config.json')==cfg['parent_config_sha256']
    assert sha(parent/'independent_verification.json')==cfg['parent_proof_sha256']
    close(cfg['control'],json.loads((parent/'config.json').read_text())['control'])
    assert sha(Path(cfg['origin_model_file']))==cfg['origin_model_sha256']
    frozen=joblib.load(cfg['origin_model_file'])['frozen'];serial_trees(frozen)
    for f,d in cfg['source_sha256'].items():assert sha(root/'source_snapshot'/Path(f).name)==d
    for k,d in cfg['prepared_sha256'].items():assert cells[k]['prepared_sha256']==d
    fresh=fit(cells,plan['train'],frozen);scores=[];count=0;support=[]
    for i,n in enumerate(cfg['neighbor_counts']):
        out=root/f'candidate_{i}';b=joblib.load(out/'model.joblib');serial_trees(b['frozen']);fresh['neighbors']=n
        assert b['train_ids']==plan['train'] and b['neighbors']==n
        np.testing.assert_array_equal(b['z'],fresh['z'])
        for obj,fields in [('imputer',['statistics_']),('scaler',['mean_','scale_','var_'])]:
            for field in fields:np.testing.assert_array_equal(getattr(b[obj],field),getattr(fresh[obj],field))
        for t,ft in zip(b['templates'],fresh['templates']):
            assert t['cell_id']==ft['cell_id'] and t['condition']==ft['condition'] and t['record_end']==ft['record_end']
            np.testing.assert_array_equal(t['parameters'],ft['parameters'])
        altered=copy.deepcopy(b)
        for t in altered['templates']:t['record_end']=-1e10
        for role,ids in [('training',plan['train']),('validation',plan['validation'])]:
            records=json.loads((out/(role+'_metrics.json')).read_text());routes=json.loads((out/(role+'_routes.json')).read_text())
            assert len(records)==len(routes)==len(ids);rows=[]
            with np.load(out/(role+'_predictions.npz'),allow_pickle=False) as arrays:
                assert set(arrays.files)=={k+'__100__'+field for k in ids for field in ('cycle','truth','prediction')}
                for pos,k in enumerate(ids):
                    c,y,p=[arrays[k+'__100__'+field] for field in ('cycle','truth','prediction')]
                    cc,yy=targets(cells[k]);np.testing.assert_array_equal(c,cc);np.testing.assert_array_equal(y,yy)
                    q=clean(cells[k]);np.testing.assert_allclose(p,predict(b,q,c),rtol=0,atol=1e-12)
                    np.testing.assert_allclose(p,predict(fresh,q,c),rtol=0,atol=1e-12)
                    # End metadata must not influence distances, choices, weights or forecast.
                    np.testing.assert_allclose(p,predict(altered,q,c),rtol=0,atol=1e-12)
                    group=condition_key(q);local=[j for j,t in enumerate(b['templates']) if t['condition']==group]
                    pool=local if len(local)>=6 else list(range(83))
                    z=b['scaler'].transform(b['imputer'].transform(prefix(q,b['parameters'])[None]))[0]
                    distances=np.array([np.linalg.norm(b['z'][j]-z)/np.sqrt(len(z)) for j in pool])
                    order=np.argsort(distances,kind='stable')[:min(n,len(pool))];ix=np.array(pool)[order];d=distances[order]
                    weights=1/(d+.001);weights/=np.sum(weights)
                    rr=dict(cell_id=k,scope='same_condition' if len(local)>=6 else 'family_fallback',neighbor_ids=[b['train_ids'][j] for j in ix],
                        distances=d.tolist(),weights=weights.tolist(),neighbor_record_ends=[b['templates'][j]['record_end'] for j in ix],
                        fraction_neighbors_extrapolated_at_query_end=float(np.mean([c[-1]>b['templates'][j]['record_end'] for j in ix])))
                    # Lists require elementwise numeric comparisons rather than dict-only close.
                    assert routes[pos]['cell_id']==k and routes[pos]['neighbor_ids']==rr['neighbor_ids'] and routes[pos]['scope']==rr['scope']
                    for field in ('distances','weights','neighbor_record_ends'):np.testing.assert_allclose(routes[pos][field],rr[field],rtol=0,atol=1e-12)
                    assert routes[pos]['fraction_neighbors_extrapolated_at_query_end']==rr['fraction_neighbors_extrapolated_at_query_end']
                    rel=np.stack([curve(b['templates'][j]['parameters'],c)-curve(b['templates'][j]['parameters'],np.array([101.]))[0] for j in ix])
                    expected=origin(frozen,q)+np.einsum('i,ij->j',weights,rel)
                    np.testing.assert_allclose(p,expected,rtol=0,atol=1e-12)
                    np.testing.assert_allclose(p[0],origin(frozen,q),rtol=0,atol=1e-12);assert np.all(np.diff(p)<=1e-10)
                    poison=dict(q,soh=np.inf,label_cycle=np.inf,reference_capacity_ah=np.inf,cycle_life=np.inf,
                        cycle=np.r_[q['cycle'],101.],raw=np.vstack([q['raw'],np.full((1,51),np.inf)]))
                    np.testing.assert_allclose(p,predict(b,poison,c),rtol=0,atol=1e-12)
                    try:predict(b,dict(q,physical_family='INVALID'),c)
                    except ValueError:pass
                    else:raise AssertionError('family gate')
                    r=dict(cell_id=k,**detail(c,y,p),start_error_pp=float(100*abs(p[0]-y[0])));close(records[pos],r);rows.append(r);count+=1
                    if role=='validation':support.append(dict(candidate=i,**rr,endpoint_pp=r['endpoint_pp']))
            if role=='validation':vr=rows
        s=np.array([np.mean([r[field] for field in ('mape_percent','late_mape','endpoint_pp','drop_error_pp')]) for r in vr])
        scores.append(dict(candidate=i,neighbors=n,score=float(s.mean()+.25*np.percentile(s,90)),
            **{field:float(np.mean([r[field] for r in vr])) for field in ('mape_percent','late_mape','endpoint_pp','drop_error_pp','start_error_pp')}))
    selection=json.loads((root/'selection.json').read_text())
    for a,b in zip(selection['candidates'],scores):close(a,b)
    best=min(scores,key=lambda r:(r['score'],r['candidate']));old=cfg['control']
    expand=best['score']<old['score'] and best['mape_percent']<=old['mape_percent']+.1 and best['endpoint_pp']<=old['endpoint_pp'] and best['drop_error_pp']<=old['drop_error_pp']
    close(selection['selected'],best);assert selection['expand']==expand and count==214
    save_json(root/'verified_validation_routes.json',support)
    save_json(root/'independent_verification.json',dict(passed=True,curves=count,template_solver_replays=83,expand=expand,outer_loaded=False,
        scope='Shared knee solver/features replay; separate neighbor distances/weights/relative curve/selection arithmetic; end metadata poisoning invariant.',verifier_sha256=sha(Path(__file__))))
    print('PASS:83template replays,214curves,routes/end metadata invariance/poison/family/metrics; expand='+str(expand))


if __name__=='__main__':
    with threadpool_limits(limits=1):main(Path(sys.argv[1]))
