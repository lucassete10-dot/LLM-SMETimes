"""Independent inherited model inference and all2196 blend curves/selection."""
import argparse,json
from pathlib import Path
import joblib,numpy as np
from threadpoolctl import threadpool_limits
from scripts.prepare_same_type_forecast import sha,save_json
from scripts.train_curve_fidelity import load_subset
from scripts.train_crossfit_drift_smoke import clean
from scripts.verify_nasa_robust_inner import inferred,close
from scripts.verify_nasa_cycle_calibration import score
from scripts.audit_rate_blend_metrics import metrics
from battery_smetimes.research_baselines import predict_baseline
from battery_smetimes.nasa_slope_blend import predict as blend_predict


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);root=ap.parse_args().run;cfg=json.loads((root/'config.json').read_text());assert cfg['outer_loaded'] is False and cfg['new_component_fits']==0;assert json.loads((root/'progress.json').read_text())['complete'];parent=Path(cfg['parent']);pcfg=json.loads((parent/'config.json').read_text());proof=json.loads((parent/'independent_verification.json').read_text());assert proof['passed'] and proof['curves']==900 and proof['independent_component_refits']==156;assert sha('scripts/verify_nasa_robust_inner.py')==proof['source_sha256']
    for key,file in [('parent_config_sha256','config.json'),('parent_proof_sha256','independent_verification.json'),('parent_summary_sha256','summary.json')]:assert cfg[key]==sha(parent/file)
    for file,digest in cfg['source_sha256'].items():assert sha(file)==digest==sha(root/'source_snapshot'/Path(file).name)
    for file,digest in pcfg['source_sha256'].items():assert sha(file)==digest==sha(parent/'source_snapshot'/Path(file).name)
    assert cfg['plans']==pcfg['plans'];specs=pcfg['candidates']+[dict(arm='blend',calibration=s['calibration'],parent_direct=i,parent_level=i+12,weight=w) for i,s in enumerate(pcfg['candidates'][1:13],1) for w in (.25,.5,.75)];assert specs==cfg['candidates'] and len(specs)==61;records=[];curves=0;allrows=[]
    for plan in cfg['plans']:
        ids=plan['train']+plan['validation'];assert len(ids)==3 and not set(ids)&set(plan['test']);cells=load_subset(ids);assert set(cells)==set(ids);folder=root/plan['name'];provenance=json.loads((folder/'inherited_sha256.json').read_text());assert provenance
        for file,digest in provenance.items():assert sha(file)==digest
        assert json.loads((parent/plan['name']/'data_sha256.json').read_text())=={k:cells[k]['prepared_sha256'] for k in ids};vr=[]
        for held in ids:
            pd=parent/plan['name']/('hold_'+held);dest=folder/('hold_'+held);train=[k for k in ids if k!=held];assert json.loads((dest/'split.json').read_text())==dict(train=train,validation=[held],excluded_outer=plan['test']);primary=joblib.load(pd/'primary.joblib');assert primary['train_ids']==train;calibrations=[joblib.load(pd/f'calibration_{i}.joblib') for i in range(1,13)];assert all(b['train_ids']==train for b in calibrations);cache={}
            for key in ids:
                cell=cells[key];c0,y0=cell['label_cycle'],cell['soh'];take=(c0>100)&np.isfinite(y0)&(y0>0);c,y=c0[take],y0[take];query=clean(cell);base=predict_baseline(primary,query,100,c);anchor=predict_baseline(primary,query,100,[101.])[0];direct=[inferred(b,query,c) for b in calibrations];levels=[base+inferred(b,query,[101.])[0]-anchor for b in calibrations];cache[key]=(c,y,[base]+direct+levels)
            for i,spec in enumerate(specs):
                for role,keys in [('training',train),('validation',[held])]:
                    out=dest/f'candidate_{i}';stored=json.loads((out/(role+'_metrics.json')).read_text());assert [r['cell_id'] for r in stored]==keys
                    with np.load(out/(role+'_predictions.npz')) as a:
                        assert set(a.files)=={key+'__100__'+f for key in keys for f in ('cycle','truth','prediction')}
                        for j,key in enumerate(keys):
                            c,y,preds=cache[key];p=preds[i] if i<25 else (1-spec['weight'])*preds[spec['parent_level']]+spec['weight']*preds[spec['parent_direct']];np.testing.assert_array_equal(c,a[key+'__100__cycle']);np.testing.assert_array_equal(y,a[key+'__100__truth']);np.testing.assert_allclose(p,a[key+'__100__prediction'],rtol=0,atol=1e-12)
                            if i>=25:
                                calibration=calibrations[spec['parent_direct']-1];query=clean(cells[key]);poison=dict(query,soh=np.inf,label_cycle=np.inf,reference_capacity_ah=np.inf,cycle_life=np.inf,cycle=np.r_[query['cycle'],101],raw=np.vstack([query['raw'],np.full((1,query['raw'].shape[1]),np.inf)]));np.testing.assert_allclose(p,blend_predict(primary,calibration,poison,c,spec['weight']),rtol=0,atol=1e-12)
                                try:blend_predict(primary,calibration,dict(query,physical_family='WRONG'),c,spec['weight'])
                                except ValueError:pass
                                else:raise AssertionError('family gate absent')
                            e=p-y;ss=float(np.sum((y-y.mean())**2));r=dict(cell_id=key,candidate=i,fold=plan['name'],heldout=held,role=role,seed=cfg['seed'],**metrics(c,y,p),mae_soh=float(np.mean(abs(e))),r2=1-float(e@e)/ss if ss else None,r2_undefined=not bool(ss),points=len(c),future_observation_span=float(c[-1]-c[0]),short_coverage=bool(len(c)<10 or c[-1]-c[0]<100));close(r,stored[j]);allrows.append(r);curves+=1
                            if role=='validation':vr.append(r)
            print('VERIFIED '+plan['name']+' hold'+held,flush=True)
        choices=[]
        for i in range(61):
            rr=[r for r in vr if r['candidate']==i];assert len(rr)==3;choices.append(dict(candidate=i,score=score(rr),**{f:float(np.mean([r[f] for r in rr])) for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp')}))
        stored=json.loads((folder/'selection.json').read_text());assert len(stored['candidates'])==61
        for r,s in zip(choices,stored['candidates']):close(r,s)
        winner=min(choices,key=lambda r:(r['score'],r['candidate']));old=json.loads((parent/plan['name']/'selection.json').read_text())['selected']['candidate'];close(winner,stored['selected']);close(choices[old],stored['control']);jackknife=[dict(omitted_validation_cell=k,candidate=min((score([r for r in vr if r['candidate']==i and r['cell_id']!=k]),i) for i in range(61))[1]) for k in sorted(ids)];assert jackknife==stored['selector_jackknife'];records.append(stored)
    fields=('score','mape_percent','endpoint_pp','drop_error_pp');summ={name:{f:float(np.mean([r[name][f] for r in records])) for f in fields} for name in ('selected','control')};new,old=summ['selected'],summ['control'];wins=sum(r['selected']['score']<r['control']['score'] for r in records);newchoices=sum(r['selected']['candidate']>=25 for r in records);expand=wins>=3 and newchoices>=2 and new['endpoint_pp']<=old['endpoint_pp']+1e-10 and new['drop_error_pp']<=old['drop_error_pp']+1e-10 and new['mape_percent']<=old['mape_percent']+.1;stored=json.loads((root/'summary.json').read_text());assert wins==stored['wins'] and newchoices==stored['new_choices'] and expand==stored['expand']
    for k,r in summ.items():close(r,stored[k])
    report=json.loads((root/'comprehensive_curve_metrics.json').read_text());assert report['count']==curves==2196 and len(report['rows'])==2196
    for r,s in zip(allrows,report['rows']):close(r,s)
    save_json(root/'independent_verification.json',dict(passed=True,curves=curves,inherited_component_fits=156,new_component_fits=0,expand=expand,outer_loaded=False,source_sha256=sha(__file__),goal_complete=False));print(json.dumps(dict(passed=True,curves=curves,expand=expand)),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
