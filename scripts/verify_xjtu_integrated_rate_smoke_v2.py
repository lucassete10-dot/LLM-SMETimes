"""Independent TRAIN slope labels, fresh forests, and fixed-grid integration."""
import argparse,json
from pathlib import Path
import joblib,numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesRegressor
from threadpoolctl import threadpool_limits
from scripts.verify_xjtu_segment_features import independent_inputs,targets,replay as anchor_replay
from scripts.verify_xjtu_segment_fold_state import equal_state
from scripts.train_curve_fidelity import load_subset
from scripts.train_crossfit_drift_smoke import clean
from scripts.prepare_same_type_forecast import sha,save_json
from scripts.audit_rate_blend_metrics import metrics
from scripts.verify_nasa_robust_inner import close
from battery_smetimes.curve_fidelity import condition_key,selection_score
from battery_smetimes.xjtu_integrated_rate_v2 import predict as actual_predict

FEATURES=dict(block='VIT',context='segments')

def training_data(cells,ids,parameters):
    xs=[];ys=[];weights=[];fallbacks=0
    for key in ids:
        c,y=targets(cells[key]);indices=np.unique(np.linspace(0,len(c)-1,min(96,len(c))).round().astype(int))
        centers=[];values=[]
        for ix in indices:
            chosen=np.where(abs(c-c[ix])<=parameters['window']/2)[0]
            if len(chosen)<3:
                assert len(c)>=3
                chosen=np.sort(np.argsort(abs(c-c[ix]),kind='stable')[:3]);fallbacks+=1
            cc=c[chosen];yy=y[chosen];dx=cc-cc.mean();dy=yy-yy.mean()
            value=max(0.,-float(dx@dy/(dx@dx))*100)
            centers.append(c[ix]);values.append(value)
        xs.append(independent_inputs(cells[key],centers,FEATURES));ys.append(values)
        w=np.linspace(1,3,len(values));weights.append(w/w.sum())
    x,y,w=np.concatenate(xs),np.concatenate(ys),np.concatenate(weights);w*=len(w)/w.sum()
    return x,y,w,fallbacks

def refit(cells,ids,saved,params,seed,local=True):
    assert saved['train_ids']==ids and saved['parameters']==params and saved['seed']==seed
    assert saved['family']==cells[ids[0]]['physical_family']
    x,y,w,fallbacks=training_data(cells,ids,params)
    imp=SimpleImputer(strategy='median',keep_empty_features=True);z=imp.fit_transform(x)
    model=ExtraTreesRegressor(n_estimators=128,min_samples_leaf=params['leaf'],max_depth=None,random_state=seed,n_jobs=2).fit(z,y,sample_weight=w)
    equal_state(model,saved['model']);equal_state(imp,saved['imputer']);np.testing.assert_array_equal(y,saved['training_rate_targets'])
    groups={}
    for k in ids:groups.setdefault(condition_key(cells[k]),[]).append(k)
    required={k:v for k,v in groups.items() if local and len(v)>=6};assert set(required)==set(saved['local'])
    children={};count=1
    for k,keys in required.items():
        children[k],n,nf=refit(cells,keys,saved['local'][k],params,seed,False);count+=n;fallbacks+=nf
    return dict(saved,model=model,imputer=imp,local=children),count,fallbacks

def rate_replay(b,q,c):
    assert b['family']==q['physical_family']
    if condition_key(q) in b['local']:return rate_replay(b['local'][condition_key(q)],q,c)
    return b['model'].predict(b['imputer'].transform(independent_inputs(q,c,FEATURES)))

def replay(b,q,c):
    assert q['physical_family']==b['family']
    a=anchor_replay(b['anchor'],q,[101.])[0]
    n=int(np.ceil(max(c)))-101
    if n==0:return np.full(len(c),a)
    mid=np.arange(n,dtype=float)+101.5;rate=rate_replay(b['rate'],q,mid)
    integrals=np.r_[0.,np.cumsum(rate/100)]
    h=np.asarray(c)-101;whole=np.floor(h).astype(int);fraction=h-whole
    tail=np.zeros(len(c));mask=whole<n;tail[mask]=fraction[mask]*rate[whole[mask]]/100
    return a-integrals[whole]-tail

def row(key,c,y,p):
    e=p-y;ss=float(np.sum((y-y.mean())**2))
    return dict(cell_id=key,**metrics(c,y,p),mae_soh=float(np.mean(abs(e))),r2=1-float(e@e)/ss if ss else None,
        r2_undefined=not bool(ss),points=len(c),future_observation_span=float(c[-1]-c[0]),
        short_coverage=bool(len(c)<10 or c[-1]-c[0]<100),start_error_pp=float(100*abs(p[0]-y[0])))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);root=ap.parse_args().run
    cfg=json.loads((root/'config.json').read_text());parent=Path(cfg['parent']);plan=cfg['plan'];seed=cfg['seed']
    assert not cfg['outer_loaded'] and cfg['smoke'] and seed==20261005
    for f,s in cfg['source_sha256'].items():assert sha(f)==s==sha(root/'source_snapshot'/Path(f).name)
    assert sha(parent/'config.json')==cfg['parent_config_sha256'] and plan==json.loads((parent/'config.json').read_text())['plan']
    assert sha(parent/'independent_verification.json')==cfg['parent_proof_sha256'] and json.loads((parent/'independent_verification.json').read_text())['passed']
    assert sha(cfg['model_file'])==cfg['model_sha256']
    specs=[None]+[dict(window=w,leaf=l) for w in (10,25) for l in (1,2)];assert cfg['candidates']==specs
    cells=load_subset(plan['train']+plan['validation']);assert not set(cells)&set(plan['test'])
    assert cfg['prepared_sha256']=={k:v['prepared_sha256'] for k,v in cells.items()}
    anchor=joblib.load(cfg['model_file']);scores=[];fits=count=fallbacks=0
    for i,params in enumerate(specs):
        dest=root/f'candidate_{i}';saved=joblib.load(dest/'model.joblib')
        if i:
            equal_state(saved['anchor'],anchor);rb,n,nf=refit(cells,plan['train'],saved['rate'],params,seed)
            b=dict(saved,rate=rb);fits+=n;fallbacks+=nf
        else:equal_state(saved,anchor);b=saved
        vr=[]
        for role,ids in [('training',plan['train']),('validation',plan['validation'])]:
            stored=json.loads((dest/(role+'_metrics.json')).read_text());assert [r['cell_id'] for r in stored]==ids
            with np.load(dest/(role+'_predictions.npz'),allow_pickle=False) as arrays:
                assert set(arrays.files)=={k+'__100__'+f for k in ids for f in ('cycle','truth','prediction')}
                for j,key in enumerate(ids):
                    c,y=targets(cells[key]);q=clean(cells[key]);p=replay(b,q,c) if i else anchor_replay(b,q,c)
                    stem=key+'__100__';np.testing.assert_array_equal(c,arrays[stem+'cycle']);np.testing.assert_array_equal(y,arrays[stem+'truth']);np.testing.assert_allclose(p,arrays[stem+'prediction'],atol=1e-12,rtol=0)
                    if i:
                        poison=dict(q,soh=np.inf,label_cycle=np.inf,reference_capacity_ah=np.inf,cycle_life=np.inf,raw=q['raw'].copy())
                        poison['raw'][:,[22,45]]=np.inf;poison['cycle']=np.r_[q['cycle'],10000];poison['raw']=np.vstack([poison['raw'],np.full((1,46),np.inf)])
                        np.testing.assert_allclose(p,actual_predict(b,poison,c),atol=1e-12,rtol=0)
                        np.testing.assert_allclose(p[::2],actual_predict(b,q,c[::2]),atol=1e-12,rtol=0)
                        try:actual_predict(b,dict(q,physical_family='WRONG'),c)
                        except ValueError:pass
                        else:raise AssertionError('family gate')
                    r=row(key,c,y,p);close(r,stored[j]);count+=1
                    if role=='validation':vr.append(r)
        scores.append(dict(candidate=i,parameters=params,score=selection_score(vr),**{f:float(np.mean([r[f] for r in vr])) for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp','start_error_pp')}))
        print('VERIFIED candidate '+str(i),flush=True)
    best=min(scores,key=lambda r:(r['score'],r['candidate']));old=scores[0]
    expand=best['candidate']>0 and best['score']<old['score'] and best['mape_percent']<=old['mape_percent']+.1 and best['endpoint_pp']<=old['endpoint_pp'] and best['drop_error_pp']<=old['drop_error_pp']
    saved=json.loads((root/'selection.json').read_text());close(best,saved['selected']);close(old,saved['control'])
    for a,b in zip(scores,saved['candidates']):close(a,b)
    assert len(scores)==len(saved['candidates']) and expand==saved['expand'] and not saved['outer_loaded']
    progress=json.loads((root/'progress.json').read_text());assert fits==progress['component_fits'] and count==progress['curves']
    save_json(root/'independent_verification.json',dict(passed=True,independent_component_refits=fits,curves=count,nearest3_fallback_training_centers=fallbacks,expand=expand,source_sha256=sha(__file__),label_future_integral_poison_passed=True,family_gate_passed=True,query_grid_invariance_passed=True,outer_loaded=False,goal_complete=False))
    print(json.dumps(dict(passed=True,refits=fits,curves=count,expand=expand)),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
