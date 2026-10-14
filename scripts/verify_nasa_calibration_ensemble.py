"""Independent signal-to-curve replay of all 2844 training-side predictions."""
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

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);root=ap.parse_args().run
    cfg=json.loads((root/'config.json').read_text());parent=Path(cfg['parent']);pcfg=json.loads((parent/'config.json').read_text());grand=Path(pcfg['parent'])
    assert cfg['outer_loaded'] is False and cfg['new_component_fits']==0
    assert json.loads((parent/'independent_verification.json').read_text())['passed']
    assert json.loads((grand/'independent_verification.json').read_text())['independent_component_refits']==156
    for f,d in cfg['parent_sha256'].items():assert sha(parent/f)==d
    for f,d in cfg['source_sha256'].items():assert sha(f)==d==sha(root/'source_snapshot'/Path(f).name)
    for f,d in pcfg['source_sha256'].items():assert sha(f)==d
    assert cfg['plans']==pcfg['plans']
    specs=pcfg['candidates']+[dict(arm='ensemble',members=g,reducer=r,weight=w)
          for g in [list(range(1,7)),list(range(7,13)),list(range(1,13))]
          for r in ('mean','median') for w in (0.,.25,.5)]
    assert cfg['candidates']==specs and len(specs)==79
    allrows=[];records=[]
    for plan in cfg['plans']:
        ids=plan['train']+plan['validation'];assert not set(ids)&set(plan['test']);cells=load_subset(ids);folder=root/plan['name'];vr=[]
        assert json.loads((grand/plan['name']/'data_sha256.json').read_text())=={k:cells[k]['prepared_sha256'] for k in ids}
        for f,d in json.loads((folder/'inherited_sha256.json').read_text()).items():assert sha(f)==d
        for held in ids:
            train=[k for k in ids if k!=held];dest=folder/('hold_'+held);pd=grand/plan['name']/('hold_'+held)
            assert json.loads((dest/'split.json').read_text())==dict(train=train,validation=[held],excluded_outer=plan['test'])
            primary=joblib.load(pd/'primary.joblib');cal=[joblib.load(pd/f'calibration_{i}.joblib') for i in range(1,13)]
            assert primary['train_ids']==train and all(b['train_ids']==train for b in cal)
            for role,keys in [('training',train),('validation',[held])]:
                cache={}
                for key in keys:
                    cell=cells[key];c0,y0=cell['label_cycle'],cell['soh'];keep=(c0>100)&np.isfinite(y0)&(y0>0);c,y=c0[keep],y0[keep];q=clean(cell)
                    base=predict_baseline(primary,q,100,c);ba=predict_baseline(primary,q,100,[101.])[0]
                    direct=[inferred(b,q,c) for b in cal];anchors=[inferred(b,q,[101.])[0] for b in cal]
                    levels=[base+a-ba for a in anchors];preds=[base]+direct+levels
                    for s in specs[25:61]:preds.append((1-s['weight'])*levels[s['parent_direct']-1]+s['weight']*direct[s['parent_direct']-1])
                    for s in specs[61:]:
                        agg=np.mean if s['reducer']=='mean' else np.median;idx=[j-1 for j in s['members']]
                        p=(1-s['weight'])*(base+agg(np.array(anchors)[idx])-ba)+s['weight']*agg(np.stack(direct)[idx],axis=0)
                        preds.append(p)
                    cache[key]=(c,y,preds)
                for i in range(79):
                    out=dest/f'candidate_{i}';stored=json.loads((out/(role+'_metrics.json')).read_text());assert [r['cell_id'] for r in stored]==keys
                    with np.load(out/(role+'_predictions.npz')) as a:
                        assert set(a.files)=={k+'__100__'+f for k in keys for f in ('cycle','truth','prediction')}
                        for j,key in enumerate(keys):
                            c,y,preds=cache[key];p=preds[i];prefix=key+'__100__'
                            np.testing.assert_array_equal(c,a[prefix+'cycle']);np.testing.assert_array_equal(y,a[prefix+'truth']);np.testing.assert_allclose(p,a[prefix+'prediction'],rtol=0,atol=1e-12)
                            e=p-y;ss=float(np.sum((y-y.mean())**2));r=dict(cell_id=key,candidate=i,fold=plan['name'],heldout=held,role=role,seed=cfg['seed'],**metrics(c,y,p),mae_soh=float(np.mean(abs(e))),r2=1-float(e@e)/ss if ss else None,r2_undefined=not bool(ss),points=len(c),future_observation_span=float(c[-1]-c[0]),short_coverage=bool(len(c)<10 or c[-1]-c[0]<100))
                            close(r,stored[j]);allrows.append(r)
                            if role=='validation':vr.append(r)
            print('VERIFIED '+plan['name']+' hold '+held,flush=True)
        choices=[]
        for i in range(79):
            rr=[r for r in vr if r['candidate']==i];assert len(rr)==3
            choices.append(dict(candidate=i,score=score(rr),**{f:float(np.mean([r[f] for r in rr])) for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp')}))
        stored=json.loads((folder/'selection.json').read_text());assert len(stored['candidates'])==79
        for r,s in zip(choices,stored['candidates']):close(r,s)
        close(min(choices,key=lambda r:(r['score'],r['candidate'])),stored['selected'])
        old=json.loads((parent/plan['name']/'selection.json').read_text())['selected']['candidate'];close(choices[old],stored['control'])
        jack=[dict(omitted_validation_cell=k,candidate=min((score([r for r in vr if r['candidate']==i and r['cell_id']!=k]),i) for i in range(79))[1]) for k in sorted(ids)]
        assert jack==stored['selector_jackknife'];records.append(stored)
    fields=('score','mape_percent','endpoint_pp','drop_error_pp');sums={n:{f:float(np.mean([r[n][f] for r in records])) for f in fields} for n in ('selected','control')}
    n,o=sums['selected'],sums['control'];wins=sum(r['selected']['score']<r['control']['score'] for r in records);nc=sum(r['selected']['candidate']>=61 for r in records)
    expand=wins>=3 and nc>=2 and n['endpoint_pp']<=o['endpoint_pp']+1e-10 and n['drop_error_pp']<=o['drop_error_pp']+1e-10 and n['mape_percent']<=o['mape_percent']+.1
    stored=json.loads((root/'summary.json').read_text())
    for k,r in sums.items():close(r,stored[k])
    assert (wins,nc,expand)==(stored['wins'],stored['new_choices'],stored['expand'])
    report=json.loads((root/'comprehensive_curve_metrics.json').read_text());assert len(allrows)==report['count']==2844
    # Export order is role/candidate/cell, identical to this replay.
    assert len(report['rows'])==len(allrows)
    for r,s in zip(allrows,report['rows']):close(r,s)
    save_json(root/'independent_verification.json',dict(passed=True,curves=len(allrows),new_component_fits=0,inherited_component_fits=156,outer_loaded=False,expand=expand,source_sha256=sha(__file__),goal_complete=False))
    print(json.dumps(dict(passed=True,curves=len(allrows),expand=expand)),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
