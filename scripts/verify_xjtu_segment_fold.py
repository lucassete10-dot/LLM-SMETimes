"""Independent feature arithmetic, forest refits and all saved curve metrics."""
import argparse,json,warnings
from pathlib import Path
import joblib,numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesRegressor
from threadpoolctl import threadpool_limits
from scripts.train_curve_fidelity import load_subset
from scripts.train_crossfit_drift_smoke import clean
from scripts.prepare_same_type_forecast import sha,save_json
from scripts.verify_nasa_robust_inner import close
from scripts.audit_rate_blend_metrics import metrics
from battery_smetimes.curve_fidelity import condition_key,selection_score
from battery_smetimes.forecast import metadata_vector,type_key
from battery_smetimes.tail_selection import choose,predict as old_predict
from battery_smetimes.xjtu_segment_features import predict as actual_predict
from prepare_xjtu import FEATURE_NAMES

def independent_inputs(cell,future,p):
    terms={'V':['voltage'],'VI':['voltage','current'],'VIT':['voltage','current','temperature']}[p['block']]
    indices=[i for i,name in enumerate(FEATURE_NAMES) if any(t in name for t in terms)]
    assert not set(indices)&{22,45}
    c0=cell['cycle'];take=c0<=100;c=c0[take].astype(float);x=cell['raw'][take][:,indices].astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        first=np.nanmedian(x[:5],axis=0);last=np.nanmedian(x[-3:],axis=0);values=x[-10:];times=c[-10:];valid=np.isfinite(values);count=valid.sum(axis=0)
        mean_t=(valid*times[:,None]).sum(axis=0)/np.maximum(count,1);mean_x=np.nansum(values,axis=0)/np.maximum(count,1);center=times[:,None]-mean_t;den=(center*center*valid).sum(axis=0)
        slopes=np.divide(np.nansum(center*(values-mean_x),axis=0),den,out=np.full(x.shape[1],np.nan),where=(count>=2)&(den>0))*100
        parts=[first,last,last-first,slopes]
        if p['context']=='segments':
            bins=[];counts=[]
            for start in (0,25,50,75):
                mask=(c>start)&(c<=start+25);counts.append(int(mask.sum()));bins.append(np.nanmedian(x[mask],axis=0) if mask.any() else np.full(x.shape[1],np.nan))
            parts+=bins+[bins[i+1]-bins[i] for i in range(3)]+[np.asarray(counts)/100]
    meta=cell['metadata'];parts+=[np.array([c[0],c[-1],len(c),c[-1]-c[0]])/100,metadata_vector(meta,[type_key(meta)])[-5:]]
    prefix=np.concatenate(parts);future=np.asarray(future,dtype=float);h=future-100
    return np.c_[np.tile(prefix,(len(h),1)),future/100,h/100,np.log1p(h)]

def targets(cell):
    take=(cell['label_cycle']>100)&np.isfinite(cell['soh'])&(cell['soh']>0)
    return cell['label_cycle'][take].astype(float),cell['soh'][take].astype(float)

def refit(cells,ids,bundle,p,seed):
    assert bundle['train_ids']==ids and bundle['parameters']==p and bundle['seed']==seed
    xs=[];ys=[];weights=[]
    for key in ids:
        c,y=targets(cells[key]);ix=np.unique(np.linspace(0,len(y)-1,min(96,len(y))).round().astype(int));xs.append(independent_inputs(cells[key],c[ix],p));ys.append(y[ix]);w=np.linspace(1,3,len(ix));weights.append(w/w.sum())
    x,y,w=np.concatenate(xs),np.concatenate(ys),np.concatenate(weights);w*=len(w)/w.sum()
    imputer=SimpleImputer(strategy='median',keep_empty_features=True);z=imputer.fit_transform(x)
    model=ExtraTreesRegressor(n_estimators=128,min_samples_leaf=1,max_depth=None,random_state=seed,n_jobs=2).fit(z,y,sample_weight=w)
    np.testing.assert_array_equal(imputer.statistics_,bundle['imputer'].statistics_)
    for a,b in zip(model.estimators_,bundle['model'].estimators_):
        for field in ('children_left','children_right','feature','threshold','value'):np.testing.assert_array_equal(getattr(a.tree_,field),getattr(b.tree_,field))
    return dict(bundle,model=model,imputer=imputer)

def replay(bundle,q,c):
    local=bundle['local'].get(condition_key(q))
    if local is not None:return replay(local,q,c)
    return bundle['model'].predict(bundle['imputer'].transform(independent_inputs(q,c,bundle['parameters'])))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);root=ap.parse_args().run;cfg=json.loads((root/'config.json').read_text());plan=cfg['plan'];parent=Path(cfg['parent']);pcfg=json.loads((parent/'config.json').read_text());seed=cfg['seed']
    assert cfg['outer_loaded'] is False and sha(parent/'config.json')==cfg['parent_config_sha256']
    assert plan==next(p for p in pcfg['plans'] if p['name']==plan['name'])
    for file,d in cfg['source_sha256'].items():assert sha(file)==d==sha(root/'source_snapshot'/Path(file).name)
    cells=load_subset(plan['train']+plan['validation']);assert not set(cells)&set(plan['test'])
    assert cfg['prepared_sha256']=={k:c['prepared_sha256'] for k,c in cells.items()}
    choice=choose(parent,pcfg,plan,seed,cells);assert choice==cfg['control_choice'];s=choice['selected'];selection=Path(s['selection_file']);assert sha(selection)==s['selection_sha256'];model_file=selection.parent/f"candidate_{s['candidate']}"/'model.joblib';assert sha(model_file)==s['model_sha256']
    expected=[None]+[dict(block=b,context=c) for b in ('V','VI','VIT') for c in ('basic','segments')];assert cfg['candidates']==expected
    allrows=[];scores=[];refits=0
    for i,p in enumerate(expected):
        out=root/f'candidate_{i}';bundle=joblib.load(out/'model.joblib')
        if i:
            bundle=refit(cells,plan['train'],bundle,p,seed);refits+=1;groups={}
            for key in plan['train']:groups.setdefault(condition_key(cells[key]),[]).append(key)
            required={k:v for k,v in groups.items() if len(v)>=6};assert set(bundle['local'])==set(required)
            bundle['local']={k:refit(cells,ids,bundle['local'][k],p,seed) for k,ids in required.items()};refits+=len(required)
        else:
            assert bundle['mode']==s['mode'];original=joblib.load(model_file)
            assert joblib.hash(bundle['inner'])==joblib.hash(original)
        vr=[]
        for role,ids in [('training',plan['train']),('validation',plan['validation'])]:
            stored=json.loads((out/(role+'_metrics.json')).read_text());assert [r['cell_id'] for r in stored]==ids
            with np.load(out/(role+'_predictions.npz')) as a:
                assert set(a.files)=={k+'__100__'+f for k in ids for f in ('cycle','truth','prediction')}
                for j,key in enumerate(ids):
                    c,y=targets(cells[key]);q=clean(cells[key]);p0=old_predict(bundle,q,c) if not i else replay(bundle,q,c)
                    np.testing.assert_array_equal(c,a[key+'__100__cycle']);np.testing.assert_array_equal(y,a[key+'__100__truth']);np.testing.assert_allclose(p0,a[key+'__100__prediction'],rtol=0,atol=1e-12)
                    if i:
                        poisoned=dict(q,soh=np.inf,label_cycle=np.inf,reference_capacity_ah=np.inf,cycle_life=np.inf,raw=q['raw'].copy())
                        poisoned['raw'][:,[22,45]]=np.inf;poisoned['cycle']=np.r_[q['cycle'],10000];poisoned['raw']=np.vstack([poisoned['raw'],np.full((1,46),np.inf)])
                        np.testing.assert_allclose(p0,actual_predict(bundle,poisoned,c),rtol=0,atol=1e-12)
                        try:actual_predict(bundle,dict(q,physical_family='WRONG'),c)
                        except ValueError:pass
                        else:raise AssertionError('family gate failed')
                    e=p0-y;ss=float(np.sum((y-y.mean())**2));r=dict(cell_id=key,candidate=i,role=role,**metrics(c,y,p0),mae_soh=float(np.mean(abs(e))),r2=1-float(e@e)/ss if ss else None,r2_undefined=not bool(ss),points=len(c),future_observation_span=float(c[-1]-c[0]),short_coverage=bool(len(c)<10 or c[-1]-c[0]<100),start_error_pp=float(100*abs(p0[0]-y[0])))
                    close(r,stored[j]);allrows.append(r)
                    if role=='validation':vr.append(r)
        summary=dict(candidate=i,parameters=p,score=selection_score(vr),**{f:float(np.mean([r[f] for r in vr])) for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp','start_error_pp')});close(summary,json.loads((out/'summary.json').read_text()));scores.append(summary);print('VERIFIED candidate '+str(i),flush=True)
    stored=json.loads((root/'selection.json').read_text());best=min(scores,key=lambda r:(r['score'],r['candidate']));control=scores[0];close(best,stored['selected']);close(control,stored['control'])
    expand=best['candidate']>0 and best['score']<control['score'] and best['mape_percent']<=control['mape_percent']+.1 and best['endpoint_pp']<=control['endpoint_pp'] and best['drop_error_pp']<=control['drop_error_pp'];assert expand==stored['expand']
    report=json.loads((root/'comprehensive_curve_metrics.json').read_text());assert report['count']==len(allrows)
    for r,s in zip(allrows,report['rows']):close(r,s)
    save_json(root/'independent_verification.json',dict(passed=True,independent_component_refits=refits,curves=len(allrows),expand=expand,label_future_integral_poison_passed=True,outer_loaded=False,source_sha256=sha(__file__),goal_complete=False));print('PASSED '+str(refits)+' independent fits / '+str(len(allrows))+' curves',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()


