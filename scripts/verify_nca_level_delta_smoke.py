"""Separate train sampling/head refits and all saved-curve checks."""
import json,sys
from pathlib import Path
import joblib,numpy as np
from sklearn.ensemble import ExtraTreesRegressor,HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from threadpoolctl import threadpool_limits
from battery_smetimes.tail_boost import inputs
from battery_smetimes.nca_level_delta import predict
from battery_smetimes.tail_selection import predict as control_predict
from scripts.train_tail_boost import load_cells,NCA
from scripts.train_crossfit_drift_smoke import clean
from scripts.prepare_same_type_forecast import sha,save_json
from scripts.train_nasa_robust_inner import detail
from scripts.verify_nasa_robust_inner import close


def fresh(cells,ids,weight,seed,b):
    starts=[];levels=[];xs=[];changes=[];ws=[]
    for k in ids:
        c,y=cells[k]['cycle'],cells[k]['truth'];assert c[0]==101
        starts.append(inputs(clean(cells[k]),[101.])[0]);levels.append(float(y[0]))
        idx=np.unique(np.rint(np.linspace(0,len(y)-1,min(96,len(y)))).astype(int))
        xs.append(inputs(clean(cells[k]),c[idx]));changes.append(y[idx]-float(y[0]))
        w=np.linspace(1,weight,len(idx));ws.append(w/np.sum(w))
    ai=SimpleImputer(strategy='median',keep_empty_features=True);az=ai.fit_transform(starts)
    level=ExtraTreesRegressor(n_estimators=128,min_samples_leaf=3,max_depth=8,random_state=seed,n_jobs=2).fit(az,levels)
    di=SimpleImputer(strategy='median',keep_empty_features=True);z=di.fit_transform(np.concatenate(xs));w=np.concatenate(ws);w*=len(w)/w.sum()
    delta=HistGradientBoostingRegressor(max_iter=120,learning_rate=.05,max_leaf_nodes=15,min_samples_leaf=20,l2_regularization=1,
        max_bins=64,early_stopping=False,random_state=seed).fit(z,np.concatenate(changes),sample_weight=w)
    np.testing.assert_array_equal(ai.statistics_,b['level_imputer'].statistics_);np.testing.assert_array_equal(di.statistics_,b['delta_imputer'].statistics_)
    return dict(b,level=level,level_imputer=ai,delta=delta,delta_imputer=di)


def main(root):
    cfg=json.loads((root/'config.json').read_text());plan=cfg['plan'];cells=load_cells(plan)
    assert (len(plan['train']),len(plan['validation']),len(plan['test']))==(39,5,11)
    assert set(cells)==set(plan['train']+plan['validation']) and not set(cells)&set(plan['test'])
    assert not set(plan['train'])&set(plan['validation'])
    assert sha(Path(cfg['parent'])/'config.json')==cfg['parent_config_sha256']
    assert sha(Path(cfg['control_model_file']))==cfg['control_model_sha256']
    assert sha(NCA/'index.json')==cfg['nca_index_sha256']
    index=json.loads((NCA/'index.json').read_text())
    for k,d in cfg['prepared_sha256'].items():assert d==index['cells'][k]['sha256']
    for f,d in cfg['source_sha256'].items():assert sha(root/'source_snapshot'/Path(f).name)==d
    scores=[];curves=0
    for i,weight in enumerate([None]+cfg['tail_weights']):
        out=root/f'candidate_{i}';b=joblib.load(out/'model.joblib');fn=control_predict if i==0 else predict
        fb=None
        if i:
            assert b['train_ids']==plan['train'] and b['tail_weight']==weight and b['seed']==cfg['seed']
            fb=fresh(cells,plan['train'],weight,cfg['seed'],b)
        for role,ids in [('training',plan['train']),('validation',plan['validation'])]:
            records=json.loads((out/(role+'_metrics.json')).read_text());rows=[]
            with np.load(out/(role+'_predictions.npz'),allow_pickle=False) as a:
                assert set(a.files)=={k+'__100__'+f for k in ids for f in ('cycle','truth','prediction')}
                for pos,k in enumerate(ids):
                    c,y,p=[a[k+'__100__'+f] for f in ('cycle','truth','prediction')]
                    np.testing.assert_array_equal(c,cells[k]['cycle']);np.testing.assert_array_equal(y,cells[k]['truth'])
                    query=clean(cells[k]);np.testing.assert_allclose(p,fn(b,query,c),rtol=0,atol=1e-12)
                    if fb:
                        np.testing.assert_allclose(p,predict(fb,query,c),rtol=0,atol=1e-12)
                        origin=inputs(query,[101.]);anchor=b['level'].predict(b['level_imputer'].transform(origin))[0]
                        np.testing.assert_allclose(p[0],anchor,rtol=0,atol=1e-12)
                        # Direct independent composition, including subtracting model's own origin.
                        d=b['delta'].predict(b['delta_imputer'].transform(inputs(query,c)))
                        d0=b['delta'].predict(b['delta_imputer'].transform(origin))[0]
                        np.testing.assert_allclose(p,anchor+(d-d0),rtol=0,atol=1e-12)
                    poison=dict(query,truth=np.inf,cycle=np.inf,soh=np.inf,capacity=np.inf,cycle_life=np.inf,
                        early_cycle=np.r_[query['early_cycle'],101.],early_raw=np.vstack([query['early_raw'],np.full((1,32),np.inf)]))
                    np.testing.assert_allclose(p,fn(b,poison,c),rtol=0,atol=1e-12)
                    try:fn(b,dict(query,physical_family='INVALID'),c)
                    except ValueError:pass
                    else:raise AssertionError('family gate')
                    r=dict(cell_id=k,**detail(c,y,p),start_error_pp=float(100*abs(p[0]-y[0])));close(records[pos],r)
                    for f,v in dict(mape_percent=100*np.average(abs(p-y)/y),endpoint_pp=100*abs(p[-1]-y[-1]),
                        drop_error_pp=100*abs((p[0]-p[-1])-(y[0]-y[-1]))).items():np.testing.assert_allclose(r[f],v,rtol=0,atol=1e-10)
                    rows.append(r);curves+=1
            if role=='validation':vr=rows
        s=np.array([np.mean([r[f] for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp')]) for r in vr])
        scores.append(dict(candidate=i,tail_weight=weight,score=float(s.mean()+.25*np.percentile(s,90)),
            **{f:float(np.mean([r[f] for r in vr])) for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp','start_error_pp')}))
    selection=json.loads((root/'selection.json').read_text())
    for a,b in zip(selection['candidates'],scores):close(a,b)
    best=min(scores,key=lambda r:(r['score'],r['candidate']));old=scores[0]
    expand=best['candidate']>0 and best['score']<old['score'] and best['mape_percent']<=old['mape_percent']+.1 and best['endpoint_pp']<=old['endpoint_pp'] and best['drop_error_pp']<=old['drop_error_pp']
    close(selection['selected'],best);assert selection['expand']==expand and curves==132
    save_json(root/'independent_verification.json',dict(passed=True,curves=curves,fresh_component_refits=4,expand=expand,
        scope='Separate sampling and head refits, shared feature helper; prediction equality, not full model-state comparison.',
        outer_loaded=False,verifier_sha256=sha(Path(__file__))))
    print('PASS: 4 component refits, 132 curves, anchor identity, leakage/family checks and selection; expand='+str(expand))


if __name__=='__main__':
    with threadpool_limits(limits=2):main(Path(sys.argv[1]))
