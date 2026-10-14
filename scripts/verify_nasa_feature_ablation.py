"""Independent 144 refits and all 468 curve/selection checks."""
import argparse,json
from pathlib import Path
import joblib,numpy as np
from sklearn.linear_model import HuberRegressor
from threadpoolctl import threadpool_limits
from scripts.train_curve_fidelity import load_subset
from scripts.train_crossfit_drift_smoke import clean
from scripts.verify_nasa_robust_inner import compare_model,close
from scripts.verify_nasa_cycle_calibration import score
from scripts.audit_rate_blend_metrics import metrics
from scripts.prepare_same_type_forecast import sha,save_json
from battery_smetimes.nasa_feature_calibration import predict

def electrical(cell,params):
    columns={'V':list(range(11))+[33,34,35],
             'VI':list(range(22))+[33,34,35,36,37],
             'VIT':list(range(42))}[params['block']]
    c0=np.asarray(cell['cycle']);take=c0<=100;c=c0[take];x=np.asarray(cell['raw'])[take][:,columns].astype(np.float64)
    if params['mode']=='delta':x-=np.nanmedian(x[:5],axis=0)
    return c,np.c_[x,np.repeat(float(cell['metadata']['discharge_cutoff_v']),len(c))]

def infer(bundle,cell,future):
    c,x=electrical(cell,bundle['parameters']);eh=bundle['model'].predict(bundle['scaler'].transform(bundle['imputer'].transform(x)))
    t=c>70;trend=HuberRegressor(alpha=.01,max_iter=500).fit(((c[t]-100)/100).reshape(-1,1),eh[t])
    return trend.predict(((future-100)/100).reshape(-1,1)),c,eh

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);root=ap.parse_args().run;cfg=json.loads((root/'config.json').read_text());parent=Path(cfg['parent']);pcfg=json.loads((parent/'config.json').read_text())
    assert cfg['outer_loaded'] is False and cfg['plans']==pcfg['plans']
    assert json.loads((parent/'independent_verification.json').read_text())['passed']
    for f,d in cfg['parent_sha256'].items():assert sha(parent/f)==d
    for f,d in cfg['source_sha256'].items():assert sha(f)==d==sha(root/'source_snapshot'/Path(f).name)
    expected=[None]+[dict(block=b,mode=m,alpha=a) for b in ('V','VI','VIT') for m in ('absolute','delta') for a in (10.,100.)];assert cfg['candidates']==expected
    allrows=[];records=[];fits=0
    for plan in cfg['plans']:
        ids=plan['train']+plan['validation'];assert not set(ids)&set(plan['test']);cells=load_subset(ids);folder=root/plan['name'];vr=[]
        assert json.loads((folder/'data_sha256.json').read_text())=={k:cells[k]['prepared_sha256'] for k in ids}
        for f,d in json.loads((folder/'inherited_sha256.json').read_text()).items():assert sha(f)==d
        old=json.loads((parent/plan['name']/'selection.json').read_text())['selected']['candidate']
        for held in ids:
            train=[k for k in ids if k!=held];dest=folder/('hold_'+held);assert json.loads((dest/'split.json').read_text())==dict(train=train,validation=[held],excluded_outer=plan['test'])
            for i,params in enumerate(expected):
                out=dest/f'candidate_{i}';bundle=None
                if i:
                    b=joblib.load(out/'model.joblib');assert b['train_ids']==train and b['parameters']==params;xs=[];ys=[];ws=[];members=[]
                    for key in train:
                        cell=cells[key];c,x=electrical(cell,params);lut={int(n):float(v) for n,v in zip(cell['label_cycle'],cell['soh']) if n<=100 and np.isfinite(v) and v>0};keep=np.array([int(n) in lut for n in c]);y=np.array([lut[int(n)] for n in c[keep]])
                        xs.append(x[keep]);ys.append(y);ws.append(np.full(len(y),1/len(y)));members.extend(dict(cell_id=key,cycle=int(n)) for n in c[keep])
                    assert b['training_members']==members==json.loads((out/'training_members.json').read_text());x,y,w=np.concatenate(xs),np.concatenate(ys),np.concatenate(ws);w*=len(w)/w.sum();bundle=compare_model(b,x,y,w,params['alpha']);fits+=1
                for role,keys in [('training',train),('validation',[held])]:
                    stored=json.loads((out/(role+'_metrics.json')).read_text());assert [r['cell_id'] for r in stored]==keys
                    with np.load(out/(role+'_predictions.npz')) as a:
                        for j,key in enumerate(keys):
                            cell=cells[key];lc,yy=cell['label_cycle'],cell['soh'];take=(lc>100)&np.isfinite(yy)&(yy>0);c,y=lc[take],yy[take];q=clean(cell)
                            if i:
                                p,ec,eh=infer(bundle,q,c)
                                with np.load(out/(role+'_early_health.npz')) as e:np.testing.assert_array_equal(ec,e[key+'__cycle']);np.testing.assert_allclose(eh,e[key+'__predicted_health'],rtol=0,atol=1e-12)
                                polluted=dict(q,raw=q['raw'].copy(),soh=np.inf,label_cycle=np.inf,reference_capacity_ah=np.inf,cycle_life=np.inf)
                                polluted['raw'][:,42:]=np.inf
                                polluted['cycle']=np.r_[q['cycle'],10000]
                                polluted['raw']=np.vstack([polluted['raw'],np.full((1,44),np.inf)])
                                np.testing.assert_allclose(p,predict(bundle,polluted,c),rtol=0,atol=1e-12)
                                try:predict(bundle,dict(q,physical_family='WRONG'),c)
                                except ValueError:pass
                                else:raise AssertionError('family gate missing')
                            else:
                                file=parent/plan['name']/('hold_'+held)/f'candidate_{old}'/(role+'_predictions.npz')
                                with np.load(file) as legacy:np.testing.assert_array_equal(legacy[key+'__100__cycle'],c);np.testing.assert_array_equal(legacy[key+'__100__truth'],y);p=legacy[key+'__100__prediction']
                            np.testing.assert_array_equal(c,a[key+'__100__cycle']);np.testing.assert_array_equal(y,a[key+'__100__truth']);np.testing.assert_allclose(p,a[key+'__100__prediction'],rtol=0,atol=1e-12)
                            e=p-y;ss=float(np.sum((y-y.mean())**2));r=dict(cell_id=key,candidate=i,fold=plan['name'],heldout=held,role=role,seed=20261006,**metrics(c,y,p),mae_soh=float(np.mean(abs(e))),r2=1-float(e@e)/ss if ss else None,r2_undefined=not bool(ss),points=len(c),future_observation_span=float(c[-1]-c[0]),short_coverage=bool(len(c)<10 or c[-1]-c[0]<100),start_error_pp=float(100*abs(p[0]-y[0])))
                            close(r,stored[j]);allrows.append(r)
                            if role=='validation':vr.append(r)
            print('VERIFIED '+plan['name']+' hold '+held,flush=True)
        choices=[]
        for i in range(13):
            rr=[r for r in vr if r['candidate']==i];choices.append(dict(candidate=i,score=score(rr),**{f:float(np.mean([r[f] for r in rr])) for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp','start_error_pp')}))
        stored=json.loads((folder/'selection.json').read_text());assert len(stored['candidates'])==13
        for r,s in zip(choices,stored['candidates']):close(r,s)
        close(min(choices,key=lambda r:(r['score'],r['candidate'])),stored['selected']);close(choices[0],stored['control'])
        jack=[dict(omitted_validation_cell=k,candidate=min((score([r for r in vr if r['candidate']==i and r['cell_id']!=k]),i) for i in range(13))[1]) for k in sorted(ids)];assert stored['selector_jackknife']==jack;records.append(stored)
    fields=('score','mape_percent','endpoint_pp','drop_error_pp');summ={n:{f:float(np.mean([r[n][f] for r in records])) for f in fields} for n in ('selected','control')};n,o=summ['selected'],summ['control'];wins=sum(r['selected']['score']<r['control']['score'] for r in records);nc=sum(r['selected']['candidate']>0 for r in records)
    expand=wins>=3 and nc>=2 and n['endpoint_pp']<=o['endpoint_pp']+1e-10 and n['drop_error_pp']<=o['drop_error_pp']+1e-10 and n['mape_percent']<=o['mape_percent']+.1;stored=json.loads((root/'summary.json').read_text())
    for k,r in summ.items():close(r,stored[k])
    assert (wins,nc,expand)==(stored['wins'],stored['new_choices'],stored['expand']);report=json.loads((root/'comprehensive_curve_metrics.json').read_text());assert report['count']==len(allrows)==468 and fits==144
    for r,s in zip(allrows,report['rows']):close(r,s)
    save_json(root/'independent_verification.json',dict(passed=True,curves=468,independent_component_refits=144,label_and_future_and_AhWh_poison_passed=True,outer_loaded=False,expand=expand,source_sha256=sha(__file__),goal_complete=False));print('PASS 144 refits / 468 curves',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
