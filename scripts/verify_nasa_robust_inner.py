"""Independent component refits,900-curve arithmetic and robust selection audit."""
import argparse,json
from pathlib import Path
import joblib,numpy as np
from sklearn.linear_model import HuberRegressor
from threadpoolctl import threadpool_limits
from scripts.train_curve_fidelity import load_subset,AUDIT
from scripts.train_crossfit_drift_smoke import clean
from scripts.prepare_same_type_forecast import save_json,sha
from scripts.verify_nasa_cycle_calibration import electrical,scaler_model,score
from scripts.audit_rate_blend_metrics import metrics
from battery_smetimes.research_baselines import build_training_pairs,predict_baseline
from scripts.train_nasa_robust_inner import predict as actual_predict


def compare_model(bundle,x,y,w,alpha):
    imputer,scaler,model=scaler_model(x,y,w,alpha)
    np.testing.assert_array_equal(imputer.statistics_,bundle['imputer'].statistics_)
    for f in ('mean_','var_','scale_'):np.testing.assert_allclose(getattr(scaler,f),getattr(bundle['scaler'],f),rtol=0,atol=1e-12)
    np.testing.assert_allclose(model.coef_,bundle['model'].coef_,rtol=0,atol=1e-12);np.testing.assert_allclose(model.intercept_,bundle['model'].intercept_,rtol=0,atol=1e-12)
    return dict(bundle,imputer=imputer,scaler=scaler,model=model)


def inferred(bundle,query,future):
    c,x=electrical(query,bundle['parameters']['mode']);y=bundle['model'].predict(bundle['scaler'].transform(bundle['imputer'].transform(x)));take=c>100-bundle['parameters']['window'];trend=HuberRegressor(alpha=.01,max_iter=500).fit(((c[take]-100)/100)[:,None],y[take])
    return trend.predict(((np.asarray(future)-100)/100)[:,None])


def close(actual,expected):
    assert set(actual)==set(expected)
    for key,value in actual.items():
        if isinstance(value,(dict,list,str)) or value is None:assert value==expected[key],key
        else:np.testing.assert_allclose(value,expected[key],rtol=0,atol=1e-10)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);root=ap.parse_args().run;cfg=json.loads((root/'config.json').read_text());assert not cfg['outer_loaded'];assert json.loads((root/'progress.json').read_text())['training_complete']
    for file,digest in cfg['source_sha256'].items():assert sha(file)==digest==sha(root/'source_snapshot'/Path(file).name)
    assert sha(AUDIT/'cell_registry.json')==cfg['registry_sha256'];assert sha(AUDIT/'source_snapshot/battery_physical_registry.json')==cfg['physical_registry_sha256']
    parent=Path(cfg['parent']);level=Path(cfg['level_parent']);assert sha(parent/'config.json')==cfg['parent_config_sha256'];assert sha(level/'config.json')==cfg['level_parent_config_sha256'];pcfg=json.loads((parent/'config.json').read_text());assert cfg['plans']==pcfg['plans'] and cfg['control']==pcfg['control']
    grid=[dict(mode=m,alpha=a,window=w) for m in ('absolute','delta') for a in (.1,10.,100.) for w in (30,60)];specs=[dict(arm='control',calibration=None)]+[dict(arm=arm,calibration=p) for arm in ('direct','level') for p in grid];assert specs==cfg['candidates'];curves=refits=0;records=[];allrows=[]
    for plan in cfg['plans']:
        ids=plan['train']+plan['validation'];assert len(ids)==3 and len(set(ids))==3 and not set(ids)&set(plan['test']);cells=load_subset(ids);assert set(cells)==set(ids);folder=root/plan['name'];assert json.loads((folder/'data_sha256.json').read_text())=={k:cells[k]['prepared_sha256'] for k in ids};vr=[]
        for held in ids:
            train=[k for k in ids if k!=held];dest=folder/('hold_'+held);assert json.loads((dest/'split.json').read_text())==dict(train=train,validation=[held],excluded_outer=plan['test']);primary=joblib.load(dest/'primary.joblib');assert primary['train_ids']==train;x,y,w,_=build_training_pairs(cells,train,[100]);primary=compare_model(primary,x,y,w,cfg['control'][plan['name']]['parameters']['alpha']);refits+=1;calibrations=[]
            for j,params in enumerate(grid,1):
                bundle=joblib.load(dest/f'calibration_{j}.joblib');assert bundle['train_ids']==train and bundle['parameters']==params;xs=[];ys=[];ws=[];members=[]
                for key in train:
                    cell=cells[key];c,x=electrical(cell,params['mode']);lookup={int(n):float(v) for n,v in zip(cell['label_cycle'],cell['soh']) if n<=100 and np.isfinite(v) and v>0};take=np.array([int(n) in lookup for n in c]);yy=np.array([lookup[int(n)] for n in c[take]]);xs.append(x[take]);ys.append(yy);ws.append(np.full(len(yy),1/len(yy)));members.extend(dict(cell_id=key,cycle=int(n)) for n in c[take])
                assert members==bundle['training_members'];x,y,w=np.concatenate(xs),np.concatenate(ys),np.concatenate(ws);w*=len(w)/w.sum();calibrations.append(compare_model(bundle,x,y,w,params['alpha']));refits+=1
            for i,spec in enumerate(specs):
                calibration=None if i==0 else calibrations[(i-1)%12];out=dest/f'candidate_{i}'
                for role,keys in [('training',train),('validation',[held])]:
                    stored=json.loads((out/(role+'_metrics.json')).read_text());assert [r['cell_id'] for r in stored]==keys
                    with np.load(out/(role+'_predictions.npz')) as a:
                        assert set(a.files)=={key+'__100__'+f for key in keys for f in ('cycle','truth','prediction')}
                        for j,key in enumerate(keys):
                            lc,y0=cells[key]['label_cycle'],cells[key]['soh'];take=(lc>100)&np.isfinite(y0)&(y0>0);c,y=lc[take],y0[take];np.testing.assert_array_equal(c,a[key+'__100__cycle']);np.testing.assert_array_equal(y,a[key+'__100__truth']);query=clean(cells[key]);base=predict_baseline(primary,query,100,c)
                            if spec['arm']=='direct':p=inferred(calibration,query,c)
                            elif spec['arm']=='level':p=base+inferred(calibration,query,[101.])[0]-predict_baseline(primary,query,100,[101.])[0]
                            else:p=base
                            np.testing.assert_allclose(p,a[key+'__100__prediction'],rtol=0,atol=1e-12);np.testing.assert_allclose(p,actual_predict(primary,calibration,spec['arm'],query,c),rtol=0,atol=1e-12)
                            poisoned=dict(query,soh=np.inf,label_cycle=np.inf,reference_capacity_ah=np.inf,cycle_life=np.inf,cycle=np.r_[query['cycle'],101],raw=np.vstack([query['raw'],np.full((1,query['raw'].shape[1]),np.inf)]));np.testing.assert_allclose(p,actual_predict(primary,calibration,spec['arm'],poisoned,c),rtol=0,atol=1e-12)
                            try:actual_predict(primary,calibration,spec['arm'],dict(query,physical_family='WRONG'),c)
                            except ValueError:pass
                            else:raise AssertionError('no family gate')
                            e=p-y;ss=float(np.sum((y-y.mean())**2));r=dict(cell_id=key,seed=cfg['seed'],candidate=i,fold=plan['name'],heldout=held,role=role,**metrics(c,y,p),mae_soh=float(np.mean(abs(e))),r2=1-float(e@e)/ss if ss else None,r2_undefined=not bool(ss),points=len(c),future_observation_span=float(c[-1]-c[0]),short_coverage=bool(len(c)<10 or c[-1]-c[0]<100));close(r,stored[j]);allrows.append(r);curves+=1
                            if role=='validation':vr.append(r)
            print('VERIFIED '+plan['name']+' hold'+held,flush=True)
        choices=[]
        for i in range(25):
            rr=[r for r in vr if r['candidate']==i];assert len(rr)==3;choices.append(dict(candidate=i,score=score(rr),**{f:float(np.mean([r[f] for r in rr])) for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp')}))
        winner=min(choices,key=lambda r:(r['score'],r['candidate']));single=json.loads((level/plan['name']/'selection.json').read_text())['selected']['candidate'];single=single+12 if single else 0;stored=json.loads((folder/'selection.json').read_text());assert len(stored['candidates'])==25
        for r,s in zip(choices,stored['candidates']):close(r,s)
        for name,r in [('selected',winner),('control',choices[0]),('single_validation_choice',choices[single])]:close(r,stored[name])
        jackknife=[dict(omitted_validation_cell=k,candidate=min((score([r for r in vr if r['candidate']==i and r['cell_id']!=k]),i) for i in range(25))[1]) for k in sorted(ids)];assert jackknife==stored['selector_jackknife'];records.append(stored)
    fields=('score','mape_percent','endpoint_pp','drop_error_pp');summary={name:{f:float(np.mean([r[name][f] for r in records])) for f in fields} for name in ('selected','control','single_validation_choice')};new,old=summary['selected'],summary['control'];wins=sum(r['selected']['score']<r['control']['score'] for r in records);expand=wins>=3 and new['mape_percent']<=old['mape_percent']+.1 and new['endpoint_pp']<=old['endpoint_pp']+1e-10 and new['drop_error_pp']<=old['drop_error_pp']+1e-10;stored=json.loads((root/'summary.json').read_text());assert wins==stored['wins'] and expand==stored['expand']
    for name,r in summary.items():close(r,stored[name])
    report=json.loads((root/'comprehensive_curve_metrics.json').read_text());assert report['count']==curves==900 and refits==156 and len(report['rows'])==900
    for r,s in zip(allrows,report['rows']):close(r,s)
    save_json(root/'independent_verification.json',dict(passed=True,curves=curves,independent_component_refits=refits,expand=expand,outer_loaded=False,source_sha256=sha(__file__),goal_complete=False));print(json.dumps(dict(passed=True,curves=curves,refits=refits,expand=expand)),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
