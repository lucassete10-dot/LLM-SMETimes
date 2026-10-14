"""Independent TRAIN-stat/head replay and 132-curve arithmetic, no outer loading."""
import json,sys
from pathlib import Path
import joblib,numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from battery_smetimes.xjtu_segment_features import prefix,predict as base_predict
from battery_smetimes.xjtu_personal_shape import predict
from battery_smetimes.curve_fidelity import targets
from scripts.verify_xjtu_personal_shape import matrix
from scripts.train_curve_fidelity import load_subset
from scripts.train_crossfit_drift_smoke import clean
from scripts.train_nasa_robust_inner import detail
from scripts.verify_mit_fixed_origin_dynamics import serial_trees
from scripts.verify_nasa_robust_inner import close
from scripts.prepare_same_type_forecast import sha,save_json


def main(root):
    cfg=json.loads((root/'config.json').read_text());plan=cfg['plan'];cells=load_subset(plan['train']+plan['validation'])
    assert (len(plan['train']),len(plan['validation']),len(plan['test']))==(31,13,11) and not set(cells)&set(plan['test'])
    for f,d in cfg['input_sha256'].items():assert sha(Path(f))==d
    for f,d in cfg['source_sha256'].items():assert sha(root/'source_snapshot'/Path(f).name)==d
    for k,d in cfg['prepared_sha256'].items():assert cells[k]['prepared_sha256']==d
    parent=Path(cfg['parent']);assert json.loads((parent/'independent_verification.json').read_text())['passed']
    with np.load(parent/'oracle.npz',allow_pickle=False) as a:y=np.stack([a[k+'__coefficients'] for k in plan['train']])
    x=np.stack([prefix(cells[k],dict(block='VIT',context='segments')) for k in plan['train']])
    scores=[];curves=0;heads=0
    for i,alpha in enumerate([None]+cfg['alphas']):
        out=root/f'candidate_{i}';b=joblib.load(out/'model.joblib');assert b['train_ids']==plan['train']
        if alpha is None:serial_trees(b)
        else:
            imp=SimpleImputer(strategy='median',keep_empty_features=True);z=imp.fit_transform(x);sc=StandardScaler();z=sc.fit_transform(z);model=Ridge(alpha=alpha).fit(z,y);heads+=1
            np.testing.assert_allclose(imp.statistics_,b['imputer'].statistics_,rtol=0,atol=1e-12)
            for f in ('mean_','var_','scale_'):np.testing.assert_allclose(getattr(sc,f),getattr(b['scaler'],f),rtol=0,atol=1e-12)
            for f in ('coef_','intercept_'):np.testing.assert_allclose(getattr(model,f),getattr(b['model'],f),rtol=0,atol=1e-12)
            for f,v in [('mean',y.mean(0)),('lower',y.min(0)),('upper',y.max(0))]:np.testing.assert_array_equal(b[f],v)
            assert b['alpha']==alpha
        for role,ids in [('training',plan['train']),('validation',plan['validation'])]:
            saved=json.loads((out/(role+'_metrics.json')).read_text());rows=[];assert len(saved)==len(ids)
            with np.load(out/(role+'_predictions.npz'),allow_pickle=False) as a:
                assert set(a.files)=={k+'__100__'+f for k in ids for f in ('cycle','truth','prediction')}
                for pos,k in enumerate(ids):
                    c,y0=targets(cells[k]);q=clean(cells[k])
                    if alpha is None:p=base_predict(b,q,c)
                    else:
                        aa=np.clip(model.predict(sc.transform(imp.transform(prefix(q,dict(block='VIT',context='segments'))[None])))[0],b['lower'],b['upper']);p=matrix(c)@aa
                    for f,v in [('cycle',c),('truth',y0)]:np.testing.assert_array_equal(a[k+'__100__'+f],v)
                    np.testing.assert_allclose(a[k+'__100__prediction'],p,rtol=0,atol=1e-12)
                    fn=base_predict if alpha is None else predict
                    np.testing.assert_allclose(fn(b,q,c),p,rtol=0,atol=1e-12)
                    poison=dict(q,soh=np.inf,label_cycle=np.inf,reference_capacity_ah=np.inf,cycle_life=np.inf,cycle=np.r_[q['cycle'],101.],raw=np.vstack([q['raw'],np.full((1,q['raw'].shape[1]),np.inf)]))
                    np.testing.assert_allclose(fn(b,poison,c),p,rtol=0,atol=1e-12)
                    try:fn(b,dict(q,physical_family='INVALID'),c)
                    except ValueError:pass
                    else:raise AssertionError('family')
                    r=dict(cell_id=k,**detail(c,y0,p),start_error_pp=float(100*abs(p[0]-y0[0])));close(saved[pos],r);rows.append(r);curves+=1
            if role=='validation':vr=rows
        vals=np.array([np.mean([r[f] for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp')]) for r in vr])
        scores.append(dict(candidate=i,alpha=alpha,score=float(vals.mean()+.25*np.percentile(vals,90)),**{f:float(np.mean([r[f] for r in vr])) for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp','start_error_pp')}))
    sel=json.loads((root/'selection.json').read_text());assert len(sel['candidates'])==3
    for s,t in zip(sel['candidates'],scores):close(s,t)
    best=min(scores,key=lambda r:(r['score'],r['candidate']));old=scores[0]
    expand=best['candidate']>0 and best['score']<old['score'] and best['mape_percent']<=old['mape_percent']+.1 and best['endpoint_pp']<=old['endpoint_pp'] and best['drop_error_pp']<=old['drop_error_pp']
    close(sel['selected'],best);assert sel['expand']==expand and curves==132 and heads==2
    save_json(root/'independent_verification.json',dict(passed=True,fresh_heads=heads,curves=curves,expand=expand,outer_loaded=False,
        scope='Fresh Ridge and TRAIN statistics, separate shape/score algebra, shared prefix/metrics and previously verified cached shape targets.',verifier_sha256=sha(Path(__file__))))
    print('PASS:2freshheads/TRAINstats,132curves/metrics/selection/poison/family/SHA;expand='+str(expand))


if __name__=='__main__':
    with threadpool_limits(limits=1):main(Path(sys.argv[1]))
