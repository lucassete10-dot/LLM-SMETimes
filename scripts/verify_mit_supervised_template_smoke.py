"""Separate TRAIN head fit and all absolute SOH route/curve/score checks."""
import json,sys
from pathlib import Path
import joblib,numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from battery_smetimes.mit_pace_forecast import pace_label,FEATURES
from battery_smetimes.mit_segment_features import prefix,condition_key
from battery_smetimes.mit_supervised_templates import predict
from battery_smetimes.mit_fixed_origin_dynamics import origin
from scripts.verify_mit_template_tangent import relative
from battery_smetimes.curve_fidelity import targets
from scripts.train_curve_fidelity import load_subset
from scripts.train_crossfit_drift_smoke import clean
from scripts.verify_mit_fixed_origin_dynamics import serial_trees
from scripts.train_nasa_robust_inner import detail
from scripts.verify_nasa_robust_inner import close
from scripts.prepare_same_type_forecast import sha,save_json


def main(root):
    cfg=json.loads((root/'config.json').read_text());plan=cfg['plan'];cells=load_subset(plan['train']+plan['validation'])
    assert (len(plan['train']),len(plan['validation']),len(plan['test']))==(83,24,27)
    assert set(cells)==set(plan['train']+plan['validation']) and not set(cells)&set(plan['test'])
    for f,d in cfg['input_sha256'].items():assert sha(Path(f))==d
    for f,d in cfg['source_sha256'].items():assert sha(root/'source_snapshot'/Path(f).name)==d
    for k,d in cfg['prepared_sha256'].items():assert cells[k]['prepared_sha256']==d
    parent=Path(cfg['parent']);close(cfg['control'],json.loads((parent/'config.json').read_text())['control'])
    assert json.loads((parent/'independent_verification.json').read_text())['passed']
    cache=joblib.load(cfg['cache_file']);serial_trees(cache['frozen']);assert cache['train_ids']==plan['train']
    labeled=[k for k in plan['train'] if pace_label(cells[k]) is not None];unobserved=[k for k in plan['train'] if k not in labeled]
    assert len(labeled)==81 and len(unobserved)==2
    raw=np.stack([prefix(cells[k],FEATURES) for k in labeled]);ylog=np.log([pace_label(cells[k]) for k in labeled])
    scores=[];count=0
    for i,alpha in enumerate(cfg['alphas']):
        out=root/f'candidate_{i}';b=joblib.load(out/'model.joblib');serial_trees(b['frozen']);h=b['pace_head']
        assert b['train_ids']==h['train_ids']==plan['train'] and h['labeled_ids']==labeled and h['unobserved_ids']==unobserved
        assert b['alpha']==alpha and h['lower']==ylog.min() and h['upper']==ylog.max()
        imp=SimpleImputer(strategy='median',keep_empty_features=True);sc=StandardScaler();z=sc.fit_transform(imp.fit_transform(raw))
        model=Ridge(alpha=alpha).fit(z,ylog)
        np.testing.assert_array_equal(imp.statistics_,h['imputer'].statistics_)
        for f in ('mean_','scale_','var_'):np.testing.assert_array_equal(getattr(sc,f),getattr(h['scaler'],f))
        np.testing.assert_allclose(model.coef_,h['model'].coef_,rtol=0,atol=1e-12);np.testing.assert_allclose(model.intercept_,h['model'].intercept_,rtol=0,atol=1e-12)
        def coordinate(q):
            x=sc.transform(imp.transform(prefix(q,FEATURES)[None]))[0]
            return float(np.clip(x@model.coef_+model.intercept_,ylog.min(),ylog.max()))
        coords=np.array([coordinate(clean(cells[k])) for k in plan['train']]);np.testing.assert_allclose(coords,b['donor_coordinates'],rtol=0,atol=1e-12)
        for t,cached in zip(b['templates'],cache['templates']):
            assert t['cell_id']==cached['cell_id'] and t['condition']==cached['condition'] and t['record_end']==cached['record_end']
            np.testing.assert_array_equal(t['parameters'],cached['parameters'])
        for role,ids in [('training',plan['train']),('validation',plan['validation'])]:
            records=json.loads((out/(role+'_metrics.json')).read_text());routes=json.loads((out/(role+'_routes.json')).read_text());assert len(records)==len(routes)==len(ids);rows=[]
            with np.load(out/(role+'_predictions.npz'),allow_pickle=False) as a:
                assert set(a.files)=={k+'__100__'+f for k in ids for f in ('cycle','truth','prediction')}
                for pos,k in enumerate(ids):
                    c,y,p=[a[k+'__100__'+f] for f in ('cycle','truth','prediction')]
                    cc,yy=targets(cells[k]);np.testing.assert_array_equal(c,cc);np.testing.assert_array_equal(y,yy)
                    q=clean(cells[k]);np.testing.assert_allclose(p,predict(b,q,c),rtol=0,atol=1e-12)
                    local=[j for j,t in enumerate(b['templates']) if t['condition']==condition_key(q)];pool=local if len(local)>=6 else list(range(83))
                    query=coordinate(q);d=abs(coords[pool]-query);order=np.argsort(d,kind='stable')[:min(3,len(pool))];ii=np.array(pool)[order]
                    w=1/(d[order]+.001);w/=w.sum()
                    assert routes[pos]['cell_id']==k and routes[pos]['neighbor_ids']==[plan['train'][j] for j in ii]
                    assert routes[pos]['scope']==('same_condition' if len(local)>=6 else 'family_fallback')
                    np.testing.assert_allclose(routes[pos]['distances'],d[order],rtol=0,atol=1e-12);np.testing.assert_allclose(routes[pos]['weights'],w,rtol=0,atol=1e-12)
                    np.testing.assert_allclose(routes[pos]['predicted_logpace'],query,rtol=0,atol=1e-12)
                    anchor=origin(cache['frozen'],q);expected=anchor+sum(ww*relative(b['templates'][j],c) for j,ww in zip(ii,w))
                    np.testing.assert_allclose(p,expected,rtol=0,atol=1e-10);np.testing.assert_allclose(p[0],anchor,rtol=0,atol=1e-12)
                    poison=dict(q,soh=np.inf,label_cycle=np.inf,reference_capacity_ah=np.inf,cycle_life=np.inf,
                        cycle=np.r_[q['cycle'],101.],raw=np.vstack([q['raw'],np.full((1,51),np.inf)]))
                    np.testing.assert_allclose(p,predict(b,poison,c),rtol=0,atol=1e-12)
                    try:predict(b,dict(q,physical_family='INVALID'),c)
                    except ValueError:pass
                    else:raise AssertionError('family gate')
                    for bad in ([],[100.],[np.inf],[[101.]]):
                        try:predict(b,q,bad)
                        except ValueError:pass
                        else:raise AssertionError('coordinates')
                    assert np.all(np.diff(p)<=1e-10)
                    r=dict(cell_id=k,**detail(c,y,p),start_error_pp=float(100*abs(p[0]-y[0])));close(records[pos],r)
                    rows.append(r);count+=1
            if role=='validation':vr=rows
        s=np.array([np.mean([r[f] for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp')]) for r in vr])
        scores.append(dict(candidate=i,alpha=alpha,score=float(s.mean()+.25*np.percentile(s,90)),
            **{f:float(np.mean([r[f] for r in vr])) for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp','start_error_pp')}))
    selection=json.loads((root/'selection.json').read_text())
    for a,b in zip(selection['candidates'],scores):close(a,b)
    best=min(scores,key=lambda r:(r['score'],r['candidate']));old=cfg['control']
    expand=best['score']<old['score'] and best['mape_percent']<=old['mape_percent']+.1 and best['endpoint_pp']<=old['endpoint_pp'] and best['drop_error_pp']<=old['drop_error_pp']
    close(selection['selected'],best);assert selection['expand']==expand and count==214
    save_json(root/'independent_verification.json',dict(passed=True,curves=count,fresh_pace_refits=2,expand=expand,outer_loaded=False,
        scope='Separate TRAIN Ridge/stats/coordinate/route/tangent relative formula/selection; template cache inherits Stage87 solver replay; shared early/pace/metric helpers.',verifier_sha256=sha(Path(__file__))))
    print('PASS:2fresh pace heads,214SOH curves,routes/origin/poison/family/metrics/SHA; expand='+str(expand))


if __name__=='__main__':
    with threadpool_limits(limits=1):main(Path(sys.argv[1]))
