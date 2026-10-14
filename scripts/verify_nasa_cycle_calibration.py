"""Independent joins, weighted scaling, Ridge/Huber refits and curve arithmetic."""
import argparse,json
from pathlib import Path
import joblib,numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,HuberRegressor
from threadpoolctl import threadpool_limits
from scripts.train_curve_fidelity import load_subset,AUDIT
from scripts.train_crossfit_drift_smoke import clean
from scripts.prepare_same_type_forecast import sha,save_json
from scripts.audit_rate_blend_metrics import metrics
from battery_smetimes.research_baselines import build_training_pairs,predict_baseline
from battery_smetimes.nasa_cycle_calibration import predict


def electrical(cell,mode):
    take=np.asarray(cell['cycle'])<=100;c=np.asarray(cell['cycle'])[take];x=np.asarray(cell['raw'],dtype=float)[take][:,[5,10,34,35,38,40]]
    if mode=='delta':x-=np.nanmedian(x[:5],axis=0)
    return c,np.c_[x,np.full(len(c),cell['metadata']['discharge_cutoff_v'])]


def scaler_model(x,y,w,alpha):
    x=np.asarray(x,dtype=np.float64)
    imputer=SimpleImputer(strategy='median',keep_empty_features=True).fit(x);z=imputer.transform(x);offset=z[0].copy();scaler=StandardScaler().fit(z-offset,sample_weight=w);scaler.mean_+=offset
    model=Ridge(alpha=alpha,solver='svd').fit(scaler.transform(z),y,sample_weight=w)
    return imputer,scaler,model


def score(rows):
    v=np.array([(r['mape_percent']+r['late_mape']+r['endpoint_pp']+r['drop_error_pp'])/4 for r in rows]);return float(v.mean()+.25*np.quantile(v,.9))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);root=parser.parse_args().run;cfg=json.loads((root/'config.json').read_text());assert json.loads((root/'progress.json').read_text())['training_complete'];assert cfg['original_test_loaded'] is False
    for f,h in cfg['source_sha256'].items():assert sha(f)==h and sha(root/'source_snapshot'/Path(f).name)==h
    assert sha(AUDIT/'cell_registry.json')==cfg['registry_sha256'] and sha(AUDIT/'source_snapshot/battery_physical_registry.json')==cfg['physical_registry_sha256'];assert sha(Path(cfg['source_run'])/'config.json')==cfg['source_config_sha256']
    expected=[dict(mode=m,alpha=a,window=w) for m in ('absolute','delta') for a in (.1,10.,100.) for w in (30,60)];assert cfg['candidates']==expected;records=[];curves=0;refits=0
    for plan in cfg['plans']:
        ids=plan['train'];cells=load_subset(ids+plan['validation']);assert set(cells)==set(ids+plan['validation']) and not set(cells)&set(plan['test']);folder=root/plan['name'];provenance=cfg['control'][plan['name']];assert sha(provenance['file'])==provenance['sha256'];results=[]
        assert json.loads((folder/'data_sha256.json').read_text())=={k:c['prepared_sha256'] for k,c in cells.items()}
        for i in range(13):
            dest=folder/f'candidate_{i}';bundle=joblib.load(dest/'model.joblib');assert bundle['train_ids']==ids
            if i:
                params=expected[i-1];assert bundle['parameters']==params;xs=[];ys=[];ws=[];members=[]
                for key in ids:
                    cell=cells[key];c,x=electrical(cell,params['mode']);lookup={int(n):float(y) for n,y in zip(cell['label_cycle'],cell['soh']) if n<=100 and np.isfinite(y) and y>0};take=np.array([int(n) in lookup for n in c]);y=np.array([lookup[int(n)] for n in c[take]]);xs.append(x[take]);ys.append(y);ws.append(np.full(len(y),1/len(y)));members.extend(dict(cell_id=key,cycle=int(n)) for n in c[take])
                x,y,w=np.concatenate(xs),np.concatenate(ys),np.concatenate(ws);w*=len(w)/w.sum();assert members==bundle['training_members']==json.loads((dest/'members.json').read_text())
                with np.load(dest/'supervision.npz') as a:
                    for f,v in [('features',x),('truth',y),('weight',w)]:np.testing.assert_array_equal(v,a[f])
            else:
                params=provenance['parameters'];x,y,w,_=build_training_pairs(cells,ids,[100])
            imputer,scaler,model=scaler_model(x,y,w,params['alpha']);refits+=1
            for f,v in [('statistics_',imputer.statistics_)]:np.testing.assert_array_equal(v,getattr(bundle['imputer'],f))
            for f in ('mean_','var_','scale_'):np.testing.assert_allclose(getattr(scaler,f),getattr(bundle['scaler'],f),rtol=0,atol=1e-12)
            np.testing.assert_allclose(model.coef_,bundle['model'].coef_,rtol=0,atol=1e-12)
            for role,keys in [('training',ids),('validation',plan['validation'])]:
                rows=[];saved=json.loads((dest/(role+'_metrics.json')).read_text());assert [r['cell_id'] for r in saved]==keys
                with np.load(dest/(role+'_predictions.npz')) as a:
                    assert set(a.files)=={key+'__100__'+f for key in keys for f in ('cycle','truth','prediction')}
                    for j,key in enumerate(keys):
                        query=clean(cells[key]);c= a[key+'__100__cycle'];truth=a[key+'__100__truth'];lc=cells[key]['label_cycle'];sy=cells[key]['soh'];keep=(lc>100)&np.isfinite(sy)&(sy>0);np.testing.assert_array_equal(c,lc[keep]);np.testing.assert_array_equal(truth,sy[keep])
                        if i:
                            ec,xx=electrical(query,params['mode']);health=model.predict(scaler.transform(imputer.transform(xx)));take=ec>100-params['window'];trend=HuberRegressor(alpha=.01,max_iter=500).fit(((ec[take]-100)/100)[:,None],health[take]);p=trend.predict(((c-100)/100)[:,None])
                            with np.load(dest/(role+'_early_health.npz')) as e:np.testing.assert_array_equal(ec,e[key+'__cycle']);np.testing.assert_allclose(health,e[key+'__predicted_health'],rtol=0,atol=1e-12)
                            np.testing.assert_allclose(p,predict(bundle,query,c),rtol=0,atol=1e-12)
                            try:predict(bundle,dict(query,physical_family='wrong'),c)
                            except ValueError:pass
                            else:raise AssertionError('family gate absent')
                        else:
                            replay=dict(bundle,imputer=imputer,scaler=scaler,model=model);p=predict_baseline(replay,query,100,c)
                        np.testing.assert_allclose(p,a[key+'__100__prediction'],rtol=0,atol=1e-12);r=dict(cell_id=key,**metrics(c,truth,p));rows.append(r);curves+=1
                        for f,v in r.items():
                            if f!='cell_id':np.testing.assert_allclose(v,saved[j][f],rtol=0,atol=1e-10)
                if role=='validation':results.append(dict(candidate=i,score=score(rows),**{f:float(np.mean([r[f] for r in rows])) for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp')}))
            stored=json.loads((folder/'selection.json').read_text())['candidates'][i]
            for f,v in results[-1].items():np.testing.assert_allclose(v,stored[f],rtol=0,atol=1e-10)
        record=json.loads((folder/'selection.json').read_text());winner=min(results,key=lambda r:(r['score'],r['candidate']));assert record['selected']['candidate']==winner['candidate'];records.append(dict(selected=winner,control=results[0]));print('VERIFIED '+plan['name'],flush=True)
    fields=('score','mape_percent','endpoint_pp','drop_error_pp');new={f:float(np.mean([r['selected'][f] for r in records])) for f in fields};old={f:float(np.mean([r['control'][f] for r in records])) for f in fields};wins=sum(r['selected']['score']<r['control']['score'] for r in records);expand=wins>=3 and new['endpoint_pp']<=old['endpoint_pp'] and new['drop_error_pp']<=old['drop_error_pp'] and new['mape_percent']<=old['mape_percent']+.1;stored=json.loads((root/'summary.json').read_text());assert wins==stored['wins'] and expand==stored['expand']
    for label,actual in [('new',new),('control',old)]:
        for f,v in actual.items():np.testing.assert_allclose(v,stored[label][f],rtol=0,atol=1e-10)
    save_json(root/'independent_verification.json',dict(passed=True,curves=curves,independent_ridge_refits=refits,expand=expand,outer_loaded=False,source_sha256=sha(__file__),goal_complete=False));print(json.dumps(dict(passed=True,curves=curves,refits=refits,expand=expand)),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
