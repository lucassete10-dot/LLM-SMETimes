"""Independent constrained coefficient fits, forest refits, and curve audits."""
import argparse,json
from pathlib import Path
import joblib,numpy as np
from scipy.optimize import lsq_linear
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesRegressor
from threadpoolctl import threadpool_limits
from scripts.verify_xjtu_segment_features import independent_inputs,targets,replay as old_replay
from scripts.verify_xjtu_segment_fold_state import equal_state
from scripts.train_curve_fidelity import load_subset
from scripts.train_crossfit_drift_smoke import clean
from scripts.prepare_same_type_forecast import sha,save_json
from scripts.audit_rate_blend_metrics import metrics
from scripts.verify_nasa_robust_inner import close
from battery_smetimes.curve_fidelity import condition_key,selection_score
from battery_smetimes.xjtu_segment_shape import predict as actual_predict

def coefficients(c,y,degree):
    h=(c-100.)/100.;a=np.c_[np.ones(len(h)),-h]
    if degree==2:a=np.c_[a,-h*h]
    w=np.sqrt(1+3*h/max(h));penalty=np.eye(degree+1)*np.sqrt(.001)
    anchor=np.zeros(degree+1);anchor[0]=1
    solution=lsq_linear(np.r_[a*w[:,None],penalty],np.r_[y*w,penalty@anchor],
        bounds=(np.r_[.2,np.zeros(degree)],np.r_[1.3,np.full(degree,5.)]),tol=1e-10)
    assert solution.success
    return solution.x

def feature(q):
    return independent_inputs(q,[101.],dict(block='VIT',context='segments'))[0,:-3]

def refit(cells,ids,saved,params,seed,local=True):
    assert saved['train_ids']==ids and saved['parameters']==params and saved['seed']==seed
    assert saved['family']==cells[ids[0]]['physical_family']
    x=np.stack([feature(cells[k]) for k in ids]);y=np.stack([coefficients(*targets(cells[k]),params['degree']) for k in ids])
    scale=np.maximum(np.std(y,axis=0),.001);imp=SimpleImputer(strategy='median',keep_empty_features=True)
    z=imp.fit_transform(x)
    model=ExtraTreesRegressor(n_estimators=128,min_samples_leaf=params['leaf'],max_depth=None,random_state=seed,n_jobs=2).fit(z,y/scale)
    equal_state(model,saved['model']);equal_state(imp,saved['imputer'])
    np.testing.assert_array_equal(scale,saved['target_scale']);np.testing.assert_array_equal(y,saved['training_shape_parameters'])
    groups={}
    for k in ids:groups.setdefault(condition_key(cells[k]),[]).append(k)
    required={k:v for k,v in groups.items() if local and len(v)>=6}
    assert set(required)==set(saved['local'])
    children={};count=1
    for k,keys in required.items():
        children[k],n=refit(cells,keys,saved['local'][k],params,seed,False);count+=n
    return dict(saved,model=model,imputer=imp,local=children),count

def curve(a,c):
    h=(np.asarray(c)-100)/100
    p=a[0]-a[1]*h
    if len(a)==3:p=p-a[2]*h*h
    return p

def replay(b,q,c):
    assert b['family']==q['physical_family']
    if condition_key(q) in b['local']:return replay(b['local'][condition_key(q)],q,c)
    z=b['imputer'].transform(feature(q)[None]);a=b['model'].predict(z)[0]*b['target_scale']
    return curve(a,c)

def row(key,c,y,p):
    e=p-y;ss=float(np.sum((y-y.mean())**2))
    return dict(cell_id=key,**metrics(c,y,p),mae_soh=float(np.mean(abs(e))),
        r2=1-float(e@e)/ss if ss else None,r2_undefined=not bool(ss),points=len(c),
        future_observation_span=float(c[-1]-c[0]),short_coverage=bool(len(c)<10 or c[-1]-c[0]<100))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);root=ap.parse_args().run
    cfg=json.loads((root/'config.json').read_text());parent=Path(cfg['parent']);plan=cfg['plan'];seed=cfg['seed']
    assert not cfg['outer_loaded'] and cfg['smoke'] and seed==20261005
    for f,s in cfg['source_sha256'].items():assert sha(f)==s==sha(root/'source_snapshot'/Path(f).name)
    assert sha(parent/'config.json')==cfg['parent_config_sha256'] and plan==json.loads((parent/'config.json').read_text())['plan']
    assert sha(parent/'independent_verification.json')==cfg['parent_proof_sha256']
    assert json.loads((parent/'independent_verification.json').read_text())['passed']
    assert sha(cfg['model_file'])==cfg['model_sha256']
    specs=[None]+[dict(degree=d,leaf=l) for d in (1,2) for l in (1,2)];assert cfg['candidates']==specs
    cells=load_subset(plan['train']+plan['validation']);assert not set(cells)&set(plan['test'])
    assert cfg['prepared_sha256']=={k:v['prepared_sha256'] for k,v in cells.items()}
    scores=[];fits=count=0;oracle_report=[]
    for i,params in enumerate(specs):
        dest=root/f'candidate_{i}';saved=joblib.load(dest/'model.joblib')
        if i:b,n=refit(cells,plan['train'],saved,params,seed);fits+=n
        else:
            equal_state(saved,joblib.load(cfg['model_file']));b=saved
        if i:
            oracle=[]
            for key in plan['train']:
                c,y=targets(cells[key]);a=coefficients(c,y,params['degree']);oracle.append(row(key,c,y,curve(a,c)))
            stored=json.loads((dest/'training_oracle_metrics.json').read_text())
            assert len(stored)==len(oracle)
            for a,b0 in zip(oracle,stored):close(a,b0)
            oracle_report.append(dict(candidate=i,**{f:float(np.mean([r[f] for r in oracle])) for f in ('mape_percent','endpoint_pp','drop_error_pp')}))
        vr=[]
        for role,ids in [('training',plan['train']),('validation',plan['validation'])]:
            stored=json.loads((dest/(role+'_metrics.json')).read_text());assert [r['cell_id'] for r in stored]==ids
            with np.load(dest/(role+'_predictions.npz'),allow_pickle=False) as arrays:
                assert set(arrays.files)=={k+'__100__'+f for k in ids for f in ('cycle','truth','prediction')}
                for j,key in enumerate(ids):
                    c,y=targets(cells[key]);q=clean(cells[key]);p=replay(b,q,c) if i else old_replay(b,q,c)
                    stem=key+'__100__';np.testing.assert_array_equal(c,arrays[stem+'cycle']);np.testing.assert_array_equal(y,arrays[stem+'truth']);np.testing.assert_allclose(p,arrays[stem+'prediction'],atol=1e-12,rtol=0)
                    if i:
                        poison=dict(q,soh=np.inf,label_cycle=np.inf,reference_capacity_ah=np.inf,cycle_life=np.inf,raw=q['raw'].copy())
                        poison['raw'][:,[22,45]]=np.inf;poison['cycle']=np.r_[q['cycle'],10000];poison['raw']=np.vstack([poison['raw'],np.full((1,46),np.inf)])
                        np.testing.assert_allclose(p,actual_predict(b,poison,c),atol=1e-12,rtol=0)
                        try:actual_predict(b,dict(q,physical_family='WRONG'),c)
                        except ValueError:pass
                        else:raise AssertionError('family gate')
                    r=dict(**row(key,c,y,p),start_error_pp=float(100*abs(p[0]-y[0])));close(r,stored[j]);count+=1
                    if role=='validation':vr.append(r)
        scores.append(dict(candidate=i,parameters=params,score=selection_score(vr),**{f:float(np.mean([r[f] for r in vr])) for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp','start_error_pp')}))
        print('VERIFIED candidate '+str(i),flush=True)
    best=min(scores,key=lambda r:(r['score'],r['candidate']));old=scores[0]
    expand=best['candidate']>0 and best['score']<old['score'] and best['mape_percent']<=old['mape_percent']+.1 and best['endpoint_pp']<=old['endpoint_pp'] and best['drop_error_pp']<=old['drop_error_pp']
    saved=json.loads((root/'selection.json').read_text());close(best,saved['selected']);close(old,saved['control'])
    for a,b in zip(scores,saved['candidates']):close(a,b)
    assert len(scores)==len(saved['candidates']) and expand==saved['expand'] and not saved['outer_loaded']
    assert fits==json.loads((root/'progress.json').read_text())['component_fits']
    assert count==json.loads((root/'progress.json').read_text())['curves']
    save_json(root/'independent_verification.json',dict(passed=True,independent_component_refits=fits,curves=count,training_oracle=oracle_report,expand=expand,source_sha256=sha(__file__),label_future_integral_poison_passed=True,family_gate_passed=True,outer_loaded=False,goal_complete=False))
    print(json.dumps(dict(passed=True,refits=fits,curves=count,expand=expand,oracle=oracle_report)),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
