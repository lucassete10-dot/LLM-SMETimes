"""Own-label shape solver replay plus independently refitted fold heads and all held curves."""
import json,sys
from pathlib import Path
import joblib,numpy as np
from scipy.optimize import lsq_linear
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from battery_smetimes.xjtu_segment_features import prefix
from battery_smetimes.xjtu_personal_shape import predict
from battery_smetimes.curve_fidelity import targets
from scripts.train_curve_fidelity import load_subset
from scripts.train_crossfit_drift_smoke import clean
from scripts.train_nasa_robust_inner import detail
from scripts.verify_nasa_robust_inner import close
from scripts.prepare_same_type_forecast import sha,save_json


def matrix(c):
    h=(np.asarray(c)-100)/100
    return np.stack([np.ones(len(h)),-h,-(h-1+1/(1+h))],axis=1)


def main(root):
    cfg=json.loads((root/'config.json').read_text());plan=cfg['plan'];cells=load_subset(plan['train'])
    assert len(cells)==31 and not set(cells)&set(plan['test']+plan['validation'])
    for f,d in cfg['input_sha256'].items():assert sha(Path(f))==d
    for f,d in cfg['source_sha256'].items():assert sha(root/'source_snapshot'/Path(f).name)==d
    for k,d in cfg['prepared_sha256'].items():assert cells[k]['prepared_sha256']==d
    shapes={};oracle=json.loads((root/'oracle_metrics.json').read_text())
    with np.load(root/'oracle.npz',allow_pickle=False) as a:
        assert set(a.files)=={k+'__'+f for k in plan['train'] for f in ('cycle','truth','prediction','coefficients')}
        for pos,k in enumerate(plan['train']):
            c,y=targets(cells[k]);ix=np.unique(np.rint(np.linspace(0,len(c)-1,min(96,len(c)))).astype(int))
            w=np.sqrt(np.linspace(1,3,len(ix))/(2*sum(np.linspace(1,3,len(ix)))))
            design=np.vstack([matrix(c[ix])*w[:,None],matrix(c[-1:])/np.sqrt(2),np.eye(3)*np.sqrt(.001)])
            target=np.r_[y[ix]*w,y[-1]/np.sqrt(2),np.sqrt(.001),0,0]
            solver=lsq_linear(design,target,bounds=([.2,0,0],[1.3,5,5]),tol=1e-12,max_iter=500)
            assert solver.success
            np.testing.assert_allclose(a[k+'__coefficients'],solver.x,rtol=0,atol=1e-8)
            # Stored targets preserve producer algebra; solver replay independent design close above.
            shapes[k]=a[k+'__coefficients'];p=matrix(c)@shapes[k]
            np.testing.assert_array_equal(a[k+'__cycle'],c);np.testing.assert_array_equal(a[k+'__truth'],y)
            np.testing.assert_allclose(a[k+'__prediction'],p,rtol=0,atol=1e-12)
            close(oracle[pos],dict(cell_id=k,**detail(c,y,p)))
    scores=[];metrics=[];curves=0;heads=0;prep=0;extrema=[]
    for j,alpha in enumerate(cfg['alphas']):
        out=root/f'candidate_{j}';saved=json.loads((out/'metrics.json').read_text());rows=[];pos=0
        with np.load(out/'predictions.npz',allow_pickle=False) as a:
            assert set(a.files)=={k+'__'+f for k in plan['train'] for f in ('cycle','truth','prediction','coefficients')}
            held_seen=[]
            for s in cfg['splits']:
                tr,held=s['train'],s['held'];assert not set(tr)&set(held) and set(tr+held)==set(plan['train']);held_seen+=held
                b=joblib.load(out/f"fold_{s['index']}.joblib");assert b['train_ids']==tr and b['alpha']==alpha
                x=np.stack([prefix(cells[k],dict(block='VIT',context='segments')) for k in tr]);y=np.stack([shapes[k] for k in tr])
                imp=SimpleImputer(strategy='median',keep_empty_features=True);z=imp.fit_transform(x);scaler=StandardScaler();z=scaler.fit_transform(z);prep+=1
                np.testing.assert_allclose(b['imputer'].statistics_,imp.statistics_,rtol=0,atol=1e-12)
                for f in ('mean_','var_','scale_'):np.testing.assert_allclose(getattr(b['scaler'],f),getattr(scaler,f),rtol=0,atol=1e-12)
                for f,v in [('mean',y.mean(0)),('lower',y.min(0)),('upper',y.max(0))]:np.testing.assert_array_equal(b[f],v)
                model=None if alpha is None else Ridge(alpha=alpha).fit(z,y)
                if model is not None:
                    heads+=1
                    for f in ('coef_','intercept_'):np.testing.assert_allclose(getattr(b['model'],f),getattr(model,f),rtol=0,atol=1e-12)
                else:assert b['model'] is None
                for k in held:
                    c,y=targets(cells[k]);q=clean(cells[k])
                    aa=b['mean'] if model is None else np.clip(model.predict(scaler.transform(imp.transform(prefix(q,dict(block='VIT',context='segments'))[None])))[0],b['lower'],b['upper'])
                    p=matrix(c)@aa
                    np.testing.assert_array_equal(a[k+'__cycle'],c);np.testing.assert_array_equal(a[k+'__truth'],y)
                    np.testing.assert_allclose(a[k+'__coefficients'],aa,rtol=0,atol=1e-12)
                    np.testing.assert_allclose(a[k+'__prediction'],p,rtol=0,atol=1e-12)
                    np.testing.assert_allclose(p,predict(b,q,c),rtol=0,atol=1e-12)
                    poison=dict(q,soh=np.inf,label_cycle=np.inf,reference_capacity_ah=np.inf,cycle_life=np.inf,
                        cycle=np.r_[q['cycle'],101.],raw=np.vstack([q['raw'],np.full((1,q['raw'].shape[1]),np.inf)]))
                    np.testing.assert_allclose(p,predict(b,poison,c),rtol=0,atol=1e-12)
                    try:predict(b,dict(q,physical_family='INVALID'),c)
                    except ValueError:pass
                    else:raise AssertionError('family gate')
                    for bad in ([],[100.],[np.inf],[[101.]]):
                        try:predict(b,q,bad)
                        except ValueError:pass
                        else:raise AssertionError('coordinates')
                    r=dict(cell_id=k,fold=s['index'],**detail(c,y,p),start_error_pp=float(100*abs(p[0]-y[0])),coefficient_mae=float(np.mean(abs(aa-shapes[k]))))
                    close(saved[pos],r);rows.append(r);pos+=1;curves+=1;extrema.append((float(min(p)),float(max(p))))
            assert len(held_seen)==31 and set(held_seen)==set(plan['train'])
        vals=np.array([np.mean([r[f] for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp')]) for r in rows]);metrics.append(vals)
        scores.append(dict(candidate=j,alpha=alpha,score=float(vals.mean()+.25*np.percentile(vals,90)),**{f:float(np.mean([r[f] for r in rows])) for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp','coefficient_mae')}))
    old=scores[0]
    for j,s in enumerate(scores):
        s['improved_cells']=int(np.sum(metrics[j]<metrics[0]))
        s['passes_diagnostic_gate']=j>0 and s['score']<old['score'] and s['mape_percent']<=old['mape_percent']+.1 and s['endpoint_pp']<=old['endpoint_pp'] and s['drop_error_pp']<=old['drop_error_pp'] and s['improved_cells']/31>=.6
    sel=json.loads((root/'selection.json').read_text());assert len(sel['candidates'])==3
    for s,t in zip(sel['candidates'],scores):close(s,t)
    assert sel['proceed_original_inner']==any(s['passes_diagnostic_gate'] for s in scores)
    assert (heads,prep,curves)==(10,15,93)
    save_json(root/'independent_verification.json',dict(passed=True,own_shape_solver_replays=31,fresh_ridge_heads=heads,train_only_preprocessors=prep,held_curves=curves,
        validation_loaded=False,outer_loaded=False,prediction_min=min(v[0] for v in extrema),prediction_max=max(v[1] for v in extrema),
        scope='Separate shape design/shared solver; fresh Ridge and fold statistics, separate curve algebra; shared prefix/metrics helpers. Single split seed TRAIN diagnostic.',verifier_sha256=sha(Path(__file__))))
    print('PASS:31shape solver replays,10freshRidge/15preprocessors,93heldcurves/metrics/poison/family/SHA')


if __name__=='__main__':
    with threadpool_limits(limits=1):main(Path(sys.argv[1]))
