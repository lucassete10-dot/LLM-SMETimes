"""Fresh independent NASA outer Ridge/calibration refits and locked arithmetic."""
import argparse,json
from pathlib import Path
import joblib,numpy as np
from sklearn.linear_model import HuberRegressor
from threadpoolctl import threadpool_limits
from scripts.train_curve_fidelity import load_subset,AUDIT
from scripts.train_crossfit_drift_smoke import clean
from scripts.verify_nasa_cycle_calibration import electrical,scaler_model
from scripts.prepare_same_type_forecast import sha,save_json
from scripts.audit_rate_blend_metrics import metrics,summarize,assert_summary,checks
from battery_smetimes.research_baselines import build_training_pairs,predict_baseline
from battery_smetimes.nasa_calibrated_level import predict


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);root=parser.parse_args().run;cfg=json.loads((root/'config.json').read_text());progress=json.loads((root/'progress.json').read_text());assert progress['training_complete'];assert cfg['seeds']==list(range(20261005,20261010))
    for f,h in cfg['source_sha256'].items():assert sha(f)==h and sha(root/'source_snapshot'/Path(f).name)==h
    assert sha(AUDIT/'cell_registry.json')==cfg['prepared_registry_sha256'];screen=Path(cfg['screened']);assert sha(screen/'config.json')==cfg['screen_config_sha256'] and sha(screen/'independent_verification.json')==cfg['screen_verification_sha256'];assert json.loads((screen/'independent_verification.json').read_text())['passed'];parent=Path(cfg['parent']);pcfg=json.loads((parent/'config.json').read_text());assert sha(parent/'config.json')==cfg['parent_config_sha256'];baseline=Path(cfg['baseline']);assert sha(baseline/'config.json')==cfg['baseline_config_sha256'];assert sha(root/'inner_lock.json')==progress['inner_lock_sha256'];lockdoc=json.loads((root/'inner_lock.json').read_text());assert lockdoc['outer_loaded'] is False;locks=lockdoc['selections'];assert len(locks)==20 and len({(l['fold'],l['seed']) for l in locks})==20;locktime=(root/'inner_lock.json').stat().st_mtime_ns;rows=[];oldrows=[];curves=0;refits=0
    for plan in cfg['plans']:
        assert not set(plan['train']+plan['validation'])&set(plan['test']);ids=plan['train']+plan['validation'];cells=load_subset(ids+plan['test']);assert json.loads((root/plan['name']/'data_sha256.json').read_text())=={k:c['prepared_sha256'] for k,c in cells.items()}
        for seed in cfg['seeds']:
            lock=next(l for l in locks if l['fold']==plan['name'] and l['seed']==seed);selection=screen/plan['name']/'selection.json';assert sha(selection)==lock['selection_sha256'] and selection.stat().st_mtime_ns<=locktime;chosen=json.loads(selection.read_text())['selected']['candidate'];assert chosen==lock['candidate'];assert sha(screen/plan['name']/f'candidate_{chosen}/model.joblib')==lock['screened_model_sha256'];assert lock['control_parameters']==pcfg['control'][plan['name']]['parameters'];assert lock['calibration_parameters']==(pcfg['candidates'][chosen-1] if chosen else None)
            folder=root/plan['name']/str(seed);assert (folder/'model.joblib').stat().st_mtime_ns>=locktime;bundle=joblib.load(folder/'model.joblib');assert bundle['train_ids']==bundle['primary']['train_ids']==ids;x,y,w,_=build_training_pairs(cells,ids,[100]);imputer,scaler,model=scaler_model(x,y,w,lock['control_parameters']['alpha']);primary=dict(bundle['primary'],imputer=imputer,scaler=scaler,model=model);np.testing.assert_array_equal(imputer.statistics_,bundle['primary']['imputer'].statistics_);np.testing.assert_allclose(model.coef_,bundle['primary']['model'].coef_,rtol=0,atol=1e-12);refits+=1
            cal=lock['calibration_parameters'];ci=cs=cm=None
            if cal:
                assert bundle['calibration']['train_ids']==ids and bundle['calibration']['parameters']==cal;xx=[];yy=[];ww=[]
                for key in ids:
                    c,z=electrical(cells[key],cal['mode']);lookup={int(n):float(v) for n,v in zip(cells[key]['label_cycle'],cells[key]['soh']) if n<=100 and np.isfinite(v) and v>0};take=np.array([int(n) in lookup for n in c]);targets=np.array([lookup[int(n)] for n in c[take]]);xx.append(z[take]);yy.append(targets);ww.append(np.full(len(targets),1/len(targets)))
                xx,yy,ww=np.concatenate(xx),np.concatenate(yy),np.concatenate(ww);ww*=len(ww)/ww.sum();ci,cs,cm=scaler_model(xx,yy,ww,cal['alpha']);np.testing.assert_array_equal(ci.statistics_,bundle['calibration']['imputer'].statistics_);np.testing.assert_allclose(cm.coef_,bundle['calibration']['model'].coef_,rtol=0,atol=1e-12);refits+=1
            for role,keys in [('training',ids),('test',plan['test'])]:
                stored=json.loads((folder/(role+'_metrics.json')).read_text());assert [r['cell_id'] for r in stored]==keys
                with np.load(folder/(role+'_predictions.npz')) as a:
                    assert set(a.files)=={k+'__100__'+f for k in keys for f in ('cycle','truth','prediction')}
                    for j,key in enumerate(keys):
                        query=clean(cells[key]);cycle=a[key+'__100__cycle'];truth=a[key+'__100__truth'];base=predict_baseline(primary,query,100,cycle);offset=0.
                        if cal:
                            ec,z=electrical(query,cal['mode']);health=cm.predict(cs.transform(ci.transform(z)));take=ec>100-cal['window'];trend=HuberRegressor(alpha=.01,max_iter=500).fit(((ec[take]-100)/100)[:,None],health[take]);offset=float(trend.predict([[.01]])[0]-predict_baseline(primary,query,100,[101.])[0])
                        p=base+offset;np.testing.assert_allclose(p,a[key+'__100__prediction'],rtol=0,atol=1e-12);poison=dict(query,soh=np.inf,reference_capacity_ah=np.inf,cycle_life=np.inf,cycle=np.r_[query['cycle'],101],raw=np.vstack([query['raw'],np.full((1,query['raw'].shape[1]),np.inf)]));np.testing.assert_allclose(p,predict(bundle,poison,cycle),rtol=0,atol=1e-12)
                        try:predict(bundle,dict(query,physical_family='wrong'),cycle)
                        except ValueError:pass
                        else:raise AssertionError('family gate absent')
                        keep=(cells[key]['label_cycle']>100)&np.isfinite(cells[key]['soh'])&(cells[key]['soh']>0);np.testing.assert_array_equal(cycle,cells[key]['label_cycle'][keep]);np.testing.assert_array_equal(truth,cells[key]['soh'][keep]);r=dict(cell_id=key,seed=seed,**metrics(cycle,truth,p));curves+=1
                        for f,v in r.items():
                            if f!='cell_id':np.testing.assert_allclose(v,stored[j][f],rtol=0,atol=1e-10)
                        if role=='test':rows.append(r)
            print('VERIFIED '+plan['name']+' '+str(seed),flush=True)
        for seed in range(20261004,20261009):
            with np.load(baseline/plan['name']/str(seed)/'predictions.npz') as a,np.load(root/plan['name']/str(cfg['seeds'][0])/'test_predictions.npz') as actual:
                for key in plan['test']:
                    c,y,p=[a[key+'__100__'+f] for f in ('cycle','truth','prediction')];np.testing.assert_array_equal(c,actual[key+'__100__cycle']);np.testing.assert_array_equal(y,actual[key+'__100__truth']);oldrows.append(dict(cell_id=key,seed=seed,**metrics(c,y,p)))
    new,cells=summarize(rows);old,_=summarize(oldrows);stored=json.loads((root/'summary.json').read_text());assert_summary(new,stored['new']);assert_summary(old,stored['baseline']);g=checks(new,old);assert g==stored['gates'] and all(g.values())==stored['numeric_gates_passed'];saved=json.loads((root/'by_cell.json').read_text());assert len(saved)==4
    for a,b in zip(cells,saved):
        assert a['cell_id']==b['cell_id']
        for f,v in a.items():
            if f!='cell_id':np.testing.assert_allclose(v,b[f],rtol=0,atol=1e-10)
    save_json(root/'independent_verification.json',dict(passed=True,curves=curves,independent_ridge_refits=refits,new=new,gates=g,numeric_gates_passed=all(g.values()),failed_cells=[c for c in cells if not c['joint_pass']],source_sha256=sha(__file__),goal_complete=False));print(json.dumps(dict(passed=True,curves=curves,refits=refits,gates=g)),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
