"""Independent residual normal equations, fresh heads, bounded formula and full scores."""
import json,sys
from pathlib import Path
import joblib,numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from battery_smetimes.xjtu_segment_features import prefix,predict as base_predict
from battery_smetimes.xjtu_personal_residual import predict,correction
from battery_smetimes.curve_fidelity import targets
from scripts.train_curve_fidelity import load_subset
from scripts.train_crossfit_drift_smoke import clean
from scripts.train_nasa_robust_inner import detail
from scripts.verify_mit_fixed_origin_dynamics import serial_trees
from scripts.verify_nasa_robust_inner import close
from scripts.prepare_same_type_forecast import sha,save_json


def matrix(c):
    h=(np.asarray(c)-101)/100
    return np.stack([h,h-1+1/(1+h)],axis=1)


def main(root):
    cfg=json.loads((root/'config.json').read_text());plan=cfg['plan'];cells=load_subset(plan['train']+plan['validation'])
    assert (len(plan['train']),len(plan['validation']),len(plan['test']))==(31,13,11) and not set(cells)&set(plan['test'])
    for f,d in cfg['input_sha256'].items():assert sha(Path(f))==d
    for f,d in cfg['source_sha256'].items():assert sha(root/'source_snapshot'/Path(f).name)==d
    for k,d in cfg['prepared_sha256'].items():assert cells[k]['prepared_sha256']==d
    parent=Path(cfg['parent']);assert json.loads((parent/'independent_verification.json').read_text())['passed']
    supervision={}
    for s in json.loads((parent/'oof_splits.json').read_text()):
        assert not set(s['train'])&set(s['held']) and set(s['train']+s['held'])==set(plan['train'])
        with np.load(parent/f"oof_{s['index']}"/'predictions.npz',allow_pickle=False) as a:
            for k in s['held']:
                assert k not in supervision;c,y,p=[a[k+'__'+f] for f in ('cycle','truth','prediction')];cc,yy=targets(cells[k])
                np.testing.assert_array_equal(c,cc);np.testing.assert_array_equal(y,yy)
                ix=np.unique(np.rint(np.linspace(0,len(c)-1,min(96,len(c)))).astype(int))
                w=np.linspace(1,3,len(ix));w=w/(2*w.sum());z=matrix(c[ix])-matrix(c[:1])[0]
                r=y[ix]-p[ix]-(y[0]-p[0]);ze=matrix(c[-1:])[0]-matrix(c[:1])[0];re=y[-1]-p[-1]-(y[0]-p[0])
                xx=np.vstack([z*np.sqrt(w)[:,None],ze[None]/np.sqrt(2),np.eye(2)*.1])
                ry=np.r_[r*np.sqrt(w),re/np.sqrt(2),0,0]
                supervision[k]=np.clip(np.linalg.lstsq(xx,ry,rcond=None)[0],-.1,.1)
    assert set(supervision)==set(plan['train'])
    y=np.stack([supervision[k] for k in plan['train']]);x=np.stack([prefix(cells[k],dict(block='VIT',context='segments')) for k in plan['train']])
    base=joblib.load(cfg['model_file']);serial_trees(base);scores=[];curves=0;heads=0;extrema=[]
    for i,alpha in enumerate([None]+cfg['alphas']):
        out=root/f'candidate_{i}';b=joblib.load(out/'model.joblib');assert b['train_ids']==plan['train']
        serial_trees(b if alpha is None else b['base'])
        if alpha is not None:
            # Actual producer supervision differs only floating precision; refit heads from stored targets after independent check.
            np.testing.assert_allclose(b['supervision'],y,rtol=0,atol=1e-10)
            saved_supervision=json.loads((out/'supervision.json').read_text());assert set(saved_supervision)==set(plan['train'])
            np.testing.assert_array_equal(np.array([saved_supervision[k] for k in plan['train']]),b['supervision'])
            imp=SimpleImputer(strategy='median',keep_empty_features=True);z=imp.fit_transform(x);sc=StandardScaler();z=sc.fit_transform(z);model=Ridge(alpha=alpha).fit(z,b['supervision']);heads+=1
            np.testing.assert_allclose(imp.statistics_,b['imputer'].statistics_,rtol=0,atol=1e-12)
            for f in ('mean_','var_','scale_'):np.testing.assert_allclose(getattr(sc,f),getattr(b['scaler'],f),rtol=0,atol=1e-12)
            for f in ('coef_','intercept_'):np.testing.assert_allclose(getattr(model,f),getattr(b['model'],f),rtol=0,atol=1e-12)
            np.testing.assert_array_equal(b['lower'],b['supervision'].min(0));np.testing.assert_array_equal(b['upper'],b['supervision'].max(0));assert b['alpha']==alpha
        for role,ids in [('training',plan['train']),('validation',plan['validation'])]:
            saved=json.loads((out/(role+'_metrics.json')).read_text());rows=[];assert len(saved)==len(ids)
            with np.load(out/(role+'_predictions.npz'),allow_pickle=False) as a:
                assert set(a.files)=={k+'__100__'+f for k in ids for f in ('cycle','truth','prediction')}
                for pos,k in enumerate(ids):
                    c,y0=targets(cells[k]);q=clean(cells[k]);bp=base_predict(base,q,c);p=bp
                    if alpha is not None:
                        aa=np.clip(model.predict(sc.transform(imp.transform(prefix(q,dict(block='VIT',context='segments'))[None])))[0],b['lower'],b['upper'])
                        delta=.04*np.tanh((matrix(c)@aa)/.04);assert np.max(abs(delta))<=.04;p=bp+delta
                        np.testing.assert_allclose(correction(b,q,[101.]),[0.],rtol=0,atol=1e-15)
                    for f,v in [('cycle',c),('truth',y0)]:np.testing.assert_array_equal(a[k+'__100__'+f],v)
                    np.testing.assert_allclose(a[k+'__100__prediction'],p,rtol=0,atol=1e-12)
                    fn=base_predict if alpha is None else predict
                    np.testing.assert_allclose(fn(b,q,c),p,rtol=0,atol=1e-12)
                    np.testing.assert_allclose(fn(b,q,[101.]),base_predict(base,q,[101.]),rtol=0,atol=1e-12)
                    poison=dict(q,soh=np.inf,label_cycle=np.inf,reference_capacity_ah=np.inf,cycle_life=np.inf,cycle=np.r_[q['cycle'],101.],raw=np.vstack([q['raw'],np.full((1,q['raw'].shape[1]),np.inf)]))
                    np.testing.assert_allclose(fn(b,poison,c),p,rtol=0,atol=1e-12)
                    try:fn(b,dict(q,physical_family='INVALID'),c)
                    except ValueError:pass
                    else:raise AssertionError('family')
                    for bad in ([],[100.],[np.inf],[[101.]]):
                        try:fn(b,q,bad)
                        except ValueError:pass
                        else:raise AssertionError('coordinates')
                    r=dict(cell_id=k,**detail(c,y0,p),start_error_pp=float(100*abs(p[0]-y0[0])));close(saved[pos],r);rows.append(r);curves+=1;extrema.append((float(min(p)),float(max(p))))
            if role=='validation':vr=rows
        vals=np.array([np.mean([r[f] for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp')]) for r in vr])
        scores.append(dict(candidate=i,alpha=alpha,score=float(vals.mean()+.25*np.percentile(vals,90)),**{f:float(np.mean([r[f] for r in vr])) for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp','start_error_pp')}))
    sel=json.loads((root/'selection.json').read_text());assert len(sel['candidates'])==3
    for s,t in zip(sel['candidates'],scores):close(s,t)
    best=min(scores,key=lambda r:(r['score'],r['candidate']));old=scores[0]
    expand=best['candidate']>0 and best['score']<old['score'] and best['mape_percent']<=old['mape_percent']+.1 and best['endpoint_pp']<=old['endpoint_pp'] and best['drop_error_pp']<=old['drop_error_pp']
    close(sel['selected'],best);assert sel['expand']==expand and curves==132 and heads==2
    save_json(root/'independent_verification.json',dict(passed=True,independent_lstsq_targets=31,fresh_heads=heads,curves=curves,expand=expand,outer_loaded=False,
        prediction_min=min(v[0] for v in extrema),prediction_max=max(v[1] for v in extrema),
        scope='Independent lstsq vs normal-equation supervision; fresh Ridge/TRAINstats, separate bounded residual and score algebra, shared prefix/metrics and previously verified OOF.',verifier_sha256=sha(Path(__file__))))
    print('PASS:31independentlstsq/2freshheads/TRAINstats,132curves/bounds/anchor/metrics/selection/poison/family/SHA;expand='+str(expand))


if __name__=='__main__':
    with threadpool_limits(limits=1):main(Path(sys.argv[1]))
