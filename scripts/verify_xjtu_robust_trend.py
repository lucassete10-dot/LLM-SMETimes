"""Independent scipy Theil-Sen features, fresh forests and all curve/selection checks."""
import json,sys
from pathlib import Path
import joblib,numpy as np
from scipy.stats import theilslopes
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesRegressor
from threadpoolctl import threadpool_limits
from prepare_xjtu import FEATURE_NAMES
from battery_smetimes.xjtu_robust_trend import inputs,predict
from battery_smetimes.xjtu_segment_features import predict as base_predict
from battery_smetimes.curve_fidelity import targets,condition_key
from scripts.verify_xjtu_segment_features import independent_inputs as original_inputs
from scripts.train_curve_fidelity import load_subset
from scripts.train_crossfit_drift_smoke import clean
from scripts.train_nasa_robust_inner import detail
from scripts.verify_mit_fixed_origin_dynamics import serial_trees
from scripts.verify_nasa_robust_inner import close
from scripts.prepare_same_type_forecast import sha,save_json


def independent_inputs(cell,c,p):
    z=original_inputs(cell,c,p)
    cols=[i for i,n in enumerate(FEATURE_NAMES) if any(t in n for t in ('voltage','current','temperature'))]
    keep=cell['cycle']<=100;t=cell['cycle'][keep][-10:].astype(float);x=cell['raw'][keep][-10:][:,cols].astype(float)
    slope=[]
    for j in range(len(cols)):
        mask=np.isfinite(x[:,j]);slope.append(float(theilslopes(x[mask,j],t[mask])[0]*100) if mask.sum()>=2 else np.nan)
    z[:,3*len(cols):4*len(cols)]=slope
    return z


def refit(cells,ids,b,p,seed):
    assert b['train_ids']==ids and b['parameters']==p and b['seed']==seed and b['trend']=='theil_sen'
    xs=[];ys=[];weights=[]
    for k in ids:
        c,y=targets(cells[k]);ix=np.unique(np.rint(np.linspace(0,len(c)-1,min(96,len(c)))).astype(int));x=independent_inputs(cells[k],c[ix],p)
        np.testing.assert_allclose(x,inputs(cells[k],c[ix],p),rtol=0,atol=1e-12)
        xs.append(x);ys.append(y[ix]);w=np.linspace(1,3,len(ix));weights.append(w/w.sum())
    x,y,w=np.concatenate(xs),np.concatenate(ys),np.concatenate(weights);w*=len(w)/w.sum()
    imp=SimpleImputer(strategy='median',keep_empty_features=True);z=imp.fit_transform(x)
    np.testing.assert_allclose(imp.statistics_,b['imputer'].statistics_,rtol=0,atol=1e-12)
    model=ExtraTreesRegressor(n_estimators=128,min_samples_leaf=1,max_depth=None,random_state=seed,n_jobs=2).fit(z,y,sample_weight=w)
    assert len(model.estimators_)==len(b['model'].estimators_)==128
    for t,u in zip(model.estimators_,b['model'].estimators_):
        for f in ('children_left','children_right','feature','threshold','value'):np.testing.assert_array_equal(getattr(t.tree_,f),getattr(u.tree_,f))
    model.n_jobs=1
    return dict(b,model=model,imputer=imp)


def replay(b,q,c):
    key=condition_key(q)
    if key in b['local']:return replay(b['local'][key],q,c)
    return b['model'].predict(b['imputer'].transform(independent_inputs(q,c,b['parameters'])))


def main(root):
    cfg=json.loads((root/'config.json').read_text());plan=cfg['plan'];cells=load_subset(plan['train']+plan['validation'])
    assert (len(plan['train']),len(plan['validation']),len(plan['test']))==(31,13,11) and not set(cells)&set(plan['test'])
    for f,d in cfg['input_sha256'].items():assert sha(Path(f))==d
    for f,d in cfg['source_sha256'].items():assert sha(root/'source_snapshot'/Path(f).name)==d
    for k,d in cfg['prepared_sha256'].items():assert cells[k]['prepared_sha256']==d
    base=joblib.load(cfg['model_file']);serial_trees(base);scores=[];curves=0;components=0
    for i,kind in enumerate(['original_ols','theil_sen']):
        out=root/f'candidate_{i}';b=joblib.load(out/'model.joblib');serial_trees(b)
        assert b['train_ids']==plan['train'] and b['family']==base['family'] and b['parameters']==cfg['parameters']
        if i:
            b=refit(cells,plan['train'],b,cfg['parameters'],cfg['seed']);components+=1;groups={}
            for k in plan['train']:groups.setdefault(condition_key(cells[k]),[]).append(k)
            required={k:v for k,v in groups.items() if len(v)>=6};assert set(b['local'])==set(required)
            b['local']={k:refit(cells,ids,b['local'][k],cfg['parameters'],cfg['seed']) for k,ids in required.items()};components+=len(required)
        for role,ids in [('training',plan['train']),('validation',plan['validation'])]:
            saved=json.loads((out/(role+'_metrics.json')).read_text());rows=[];assert len(saved)==len(ids)
            with np.load(out/(role+'_predictions.npz'),allow_pickle=False) as a:
                assert set(a.files)=={k+'__100__'+f for k in ids for f in ('cycle','truth','prediction')}
                for pos,k in enumerate(ids):
                    c,y=targets(cells[k]);q=clean(cells[k]);p=replay(b,q,c) if i else base_predict(base,q,c)
                    np.testing.assert_array_equal(a[k+'__100__cycle'],c);np.testing.assert_array_equal(a[k+'__100__truth'],y);np.testing.assert_allclose(a[k+'__100__prediction'],p,rtol=0,atol=1e-12)
                    fn=predict if i else base_predict;np.testing.assert_allclose(fn(b,q,c),p,rtol=0,atol=1e-12)
                    raw=q['raw'].copy();raw[:,[22,45]]=np.inf
                    poison=dict(q,soh=np.inf,label_cycle=np.inf,reference_capacity_ah=np.inf,cycle_life=np.inf,cycle=np.r_[q['cycle'],101.],raw=np.vstack([raw,np.full((1,raw.shape[1]),np.inf)]))
                    np.testing.assert_allclose(fn(b,poison,c),p,rtol=0,atol=1e-12)
                    try:fn(b,dict(q,physical_family='INVALID'),c)
                    except ValueError:pass
                    else:raise AssertionError('family')
                    for bad in ([],[100.],[np.inf],[[101.]]):
                        try:fn(b,q,bad)
                        except ValueError:pass
                        else:raise AssertionError('coordinates')
                    r=dict(cell_id=k,**detail(c,y,p),start_error_pp=float(100*abs(p[0]-y[0])));close(saved[pos],r);rows.append(r);curves+=1
            if role=='validation':vr=rows
        v=np.array([np.mean([r[f] for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp')]) for r in vr])
        scores.append(dict(candidate=i,kind=kind,score=float(v.mean()+.25*np.percentile(v,90)),**{f:float(np.mean([r[f] for r in vr])) for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp','start_error_pp')}))
    sel=json.loads((root/'selection.json').read_text());assert len(sel['candidates'])==2
    for x,y in zip(sel['candidates'],scores):close(x,y)
    best=min(scores,key=lambda r:(r['score'],r['candidate']));old=scores[0];expand=best['candidate']>0 and best['score']<old['score'] and best['mape_percent']<=old['mape_percent']+.1 and best['endpoint_pp']<=old['endpoint_pp'] and best['drop_error_pp']<=old['drop_error_pp']
    close(sel['selected'],best);assert sel['expand']==expand and curves==88 and components==2
    save_json(root/'independent_verification.json',dict(passed=True,independent_component_refits=components,exact_forest_state=True,curves=88,expand=expand,outer_loaded=False,
        scope='Scipy Theil-Sen vs manual pairwise median, prior independent remaining prefix construction, fresh forests/stats and separate score; shared metric helpers.',verifier_sha256=sha(Path(__file__))))
    print('PASS:2freshcomponents/exactforests/independentTS,88curves/metrics/selection/poison/family/SHA;expand='+str(expand))


if __name__=='__main__':
    with threadpool_limits(limits=1):main(Path(sys.argv[1]))
