"""Separate rate-target refits/composition checks, shared early-feature helper."""
import json, sys
from pathlib import Path
import joblib, numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from battery_smetimes.tail_boost import inputs
from battery_smetimes.nca_level_rate import predict
from battery_smetimes.tail_selection import predict as control_predict
from scripts.train_tail_boost import load_cells, NCA
from scripts.train_crossfit_drift_smoke import clean
from scripts.train_nasa_robust_inner import detail
from scripts.prepare_same_type_forecast import sha, save_json
from scripts.verify_nasa_robust_inner import close


def fresh(cells, ids, weight, seed, origin):
    x, targets, weights = [], [], []
    for k in ids:
        c,y=cells[k]['cycle'],cells[k]['truth']; assert c[0]==101
        ix=np.unique(np.rint(np.linspace(0,len(c)-1,min(96,len(c)))).astype(int))
        x.append(inputs(clean(cells[k]),c[ix]))
        targets.append((float(y[0])-y[ix])*100/(c[ix]-1))
        w=np.linspace(1,weight,len(ix)); weights.append(w/np.sum(w))
    imputer=SimpleImputer(strategy='median',keep_empty_features=True)
    z=imputer.fit_transform(np.concatenate(x)); w=np.concatenate(weights)
    w=w*len(w)/np.sum(w)
    model=HistGradientBoostingRegressor(max_iter=120,learning_rate=.05,max_leaf_nodes=15,
        min_samples_leaf=20,l2_regularization=1,max_bins=64,early_stopping=False,random_state=seed)
    model.fit(z,np.concatenate(targets),sample_weight=w)
    return dict(origin,rate=model,rate_imputer=imputer)


def main(root):
    cfg=json.loads((root/'config.json').read_text()); plan=cfg['plan']; cells=load_cells(plan)
    assert (len(plan['train']),len(plan['validation']),len(plan['test']))==(39,5,11)
    assert set(cells)==set(plan['train']+plan['validation']) and not set(cells)&set(plan['test'])
    assert not set(plan['train'])&set(plan['validation'])
    assert sha(Path(cfg['parent'])/'config.json')==cfg['parent_config_sha256']
    for prefix in ('control_model','origin_model','origin_config'):
        assert sha(Path(cfg[prefix+'_file']))==cfg[prefix+'_sha256']
    assert sha(NCA/'index.json')==cfg['nca_index_sha256']
    ix=json.loads((NCA/'index.json').read_text())
    for k,d in cfg['prepared_sha256'].items(): assert ix['cells'][k]['sha256']==d
    for f,d in cfg['source_sha256'].items(): assert sha(root/'source_snapshot'/Path(f).name)==d
    origin=joblib.load(cfg['origin_model_file']); origin['level'].n_jobs=1
    assert origin['train_ids']==plan['train'] and origin['seed']==cfg['seed']
    scores=[]; curves=0
    for i,weight in enumerate([None]+cfg['tail_weights']):
        out=root/f'candidate_{i}'; b=joblib.load(out/'model.joblib'); fn=control_predict if i==0 else predict
        fb=None
        if i:
            assert b['train_ids']==plan['train'] and b['seed']==cfg['seed'] and b['tail_weight']==weight
            fb=fresh(cells,plan['train'],weight,cfg['seed'],origin)
            np.testing.assert_array_equal(b['rate_imputer'].statistics_,fb['rate_imputer'].statistics_)
            np.testing.assert_array_equal(b['level_imputer'].statistics_,origin['level_imputer'].statistics_)
        for role,ids in [('training',plan['train']),('validation',plan['validation'])]:
            records=json.loads((out/(role+'_metrics.json')).read_text()); rows=[]
            assert len(records)==len(ids)
            with np.load(out/(role+'_predictions.npz'),allow_pickle=False) as a:
                assert set(a.files)=={k+'__100__'+f for k in ids for f in ('cycle','truth','prediction')}
                for pos,k in enumerate(ids):
                    c,y,p=[a[k+'__100__'+f] for f in ('cycle','truth','prediction')]
                    np.testing.assert_array_equal(c,cells[k]['cycle']); np.testing.assert_array_equal(y,cells[k]['truth'])
                    query=clean(cells[k]); np.testing.assert_allclose(p,fn(b,query,c),rtol=0,atol=1e-12)
                    if fb:
                        np.testing.assert_allclose(p,predict(fb,query,c),rtol=0,atol=1e-12)
                        z0=inputs(query,[101.]); z=inputs(query,c)
                        anchor=origin['level'].predict(origin['level_imputer'].transform(z0))[0]
                        np.testing.assert_allclose(b['level'].predict(b['level_imputer'].transform(z)),
                            origin['level'].predict(origin['level_imputer'].transform(z)),rtol=0,atol=1e-12)
                        rates=b['rate'].predict(b['rate_imputer'].transform(z))
                        r0=b['rate'].predict(b['rate_imputer'].transform(z0))[0]
                        np.testing.assert_allclose(p,anchor-rates*(c-1)/100+r0,rtol=0,atol=1e-12)
                        np.testing.assert_allclose(p[0],anchor,rtol=0,atol=1e-12)
                        for bad in ([],[100.],[np.inf],[[101.]]):
                            try: predict(b,query,bad)
                            except ValueError: pass
                            else: raise AssertionError('invalid coordinates accepted')
                    poison=dict(query,truth=np.inf,cycle=np.inf,soh=np.inf,capacity=np.inf,cycle_life=np.inf,
                        early_cycle=np.r_[query['early_cycle'],101.],
                        early_raw=np.vstack([query['early_raw'],np.full((1,32),np.inf)]))
                    np.testing.assert_allclose(p,fn(b,poison,c),rtol=0,atol=1e-12)
                    try: fn(b,dict(query,physical_family='INVALID'),c)
                    except ValueError: pass
                    else: raise AssertionError('family gate')
                    r=dict(cell_id=k,**detail(c,y,p),start_error_pp=float(100*abs(p[0]-y[0])))
                    close(records[pos],r)
                    for f,v in dict(mape_percent=100*np.mean(abs(p-y)/y),endpoint_pp=100*abs(p[-1]-y[-1]),
                        drop_error_pp=100*abs((p[0]-p[-1])-(y[0]-y[-1]))).items():
                        np.testing.assert_allclose(r[f],v,rtol=0,atol=1e-10)
                    rows.append(r); curves+=1
            if role=='validation': vr=rows
        s=np.array([np.mean([r[f] for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp')]) for r in vr])
        scores.append(dict(candidate=i,tail_weight=weight,score=float(s.mean()+.25*np.percentile(s,90)),
            **{f:float(np.mean([r[f] for r in vr])) for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp','start_error_pp')}))
    selection=json.loads((root/'selection.json').read_text())
    for a,b in zip(selection['candidates'],scores): close(a,b)
    best=min(scores,key=lambda r:(r['score'],r['candidate'])); old=scores[0]
    expand=best['candidate']>0 and best['score']<old['score'] and best['mape_percent']<=old['mape_percent']+.1 and best['endpoint_pp']<=old['endpoint_pp'] and best['drop_error_pp']<=old['drop_error_pp']
    close(selection['selected'],best); close(selection['control'],old)
    assert selection['expand']==expand and curves==132
    save_json(root/'independent_verification.json',dict(passed=True,curves=curves,fresh_rate_refits=2,
        frozen_origin_checked=True,expand=expand,outer_loaded=False,verifier_sha256=sha(Path(__file__)),
        scope='Separate rate target/sampling/refits/composition; shared feature helper, prediction equality not full-state identity.'))
    print('PASS: 2 fresh rate heads, 132 curves, frozen origin, poisoning, family and selection; expand='+str(expand))


if __name__=='__main__':
    with threadpool_limits(limits=2): main(Path(sys.argv[1]))
