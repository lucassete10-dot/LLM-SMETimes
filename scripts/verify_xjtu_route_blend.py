"""Independent feature arithmetic and per-tree replay of fixed routing arms."""
import argparse,json
from pathlib import Path
import joblib,numpy as np
from battery_smetimes.curve_fidelity import condition_key,selection_score
from scripts.verify_xjtu_segment_features import independent_inputs,targets
from scripts.train_curve_fidelity import load_subset
from scripts.train_crossfit_drift_smoke import clean
from scripts.audit_rate_blend_metrics import metrics
from scripts.verify_nasa_robust_inner import close
from scripts.prepare_same_type_forecast import sha,save_json
from scripts.screen_xjtu_route_blend import route

def one(bundle,q,c):
    assert q['physical_family']==bundle['family']
    z=bundle['imputer'].transform(independent_inputs(q,c,bundle['parameters']))
    return np.mean(np.stack([t.predict(z) for t in bundle['model'].estimators_]),axis=0)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);root=ap.parse_args().run
    cfg=json.loads((root/'config.json').read_text());plan=cfg['plan'];parent=Path(cfg['parent'])
    assert cfg['weights']==[1.,0.,.5] and cfg['new_fits']==0 and not cfg['outer_loaded']
    assert sha(cfg['model_file'])==cfg['model_sha256']
    assert sha(parent/'config.json')==cfg['parent_config_sha256']
    assert sha(parent/'independent_verification.json')==cfg['parent_proof_sha256']
    assert json.loads((parent/'independent_verification.json').read_text())['passed']
    for f,s in cfg['source_sha256'].items():assert sha(f)==s==sha(root/'source_snapshot'/Path(f).name)
    cells=load_subset(plan['train']+plan['validation']);assert not set(cells)&set(plan['test'])
    assert cfg['prepared_sha256']=={k:c['prepared_sha256'] for k,c in cells.items()}
    bundle=joblib.load(cfg['model_file']);assert bundle['train_ids']==plan['train']
    scores=[];count=0
    for i,w in enumerate(cfg['weights']):
        dest=root/f'candidate_{i}';vr=[]
        for role,ids in [('training',plan['train']),('validation',plan['validation'])]:
            stored=json.loads((dest/(role+'_metrics.json')).read_text());assert [r['cell_id'] for r in stored]==ids
            with np.load(dest/(role+'_predictions.npz'),allow_pickle=False) as a:
                assert set(a.files)=={k+'__100__'+f for k in ids for f in ('cycle','truth','prediction')}
                for j,key in enumerate(ids):
                    c,y=targets(cells[key]);q=clean(cells[key]);local=bundle['local'].get(condition_key(q),bundle)
                    p=(1-w)*one(bundle,q,c)+w*one(local,q,c);stem=key+'__100__'
                    np.testing.assert_array_equal(c,a[stem+'cycle']);np.testing.assert_array_equal(y,a[stem+'truth']);np.testing.assert_allclose(p,a[stem+'prediction'],atol=1e-12,rtol=0)
                    poison=dict(q,soh=np.inf,label_cycle=np.inf,cycle_life=np.inf,reference_capacity_ah=np.inf,raw=q['raw'].copy())
                    poison['raw'][:,[22,45]]=np.inf;poison['cycle']=np.r_[q['cycle'],10000];poison['raw']=np.vstack([poison['raw'],np.full((1,46),np.inf)])
                    np.testing.assert_allclose(p,route(bundle,poison,c,w),atol=1e-12,rtol=0)
                    try:route(bundle,dict(q,physical_family='WRONG'),c,w)
                    except ValueError:pass
                    else:raise AssertionError('family gate')
                    e=p-y;ss=float(np.sum((y-y.mean())**2))
                    r=dict(cell_id=key,**metrics(c,y,p),mae_soh=float(np.mean(abs(e))),r2=1-float(e@e)/ss if ss else None,r2_undefined=not bool(ss),points=len(c),future_observation_span=float(c[-1]-c[0]),short_coverage=bool(len(c)<10 or c[-1]-c[0]<100),start_error_pp=float(100*abs(p[0]-y[0])),has_local=condition_key(q) in bundle['local'])
                    close(r,stored[j]);count+=1
                    if role=='validation':vr.append(r)
        scores.append(dict(candidate=i,local_weight=w,score=selection_score(vr),**{f:float(np.mean([r[f] for r in vr])) for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp','start_error_pp')}))
    best=min(scores,key=lambda r:(r['score'],r['candidate']));control=scores[0]
    expand=best['candidate']!=0 and best['score']<control['score'] and best['mape_percent']<=control['mape_percent']+.1 and best['endpoint_pp']<=control['endpoint_pp'] and best['drop_error_pp']<=control['drop_error_pp']
    saved=json.loads((root/'selection.json').read_text())
    close(best,saved['selected']);close(control,saved['control'])
    assert len(scores)==len(saved['candidates'])
    for a,b in zip(scores,saved['candidates']):close(a,b)
    assert expand==saved['expand'] and not saved['outer_loaded'] and not saved['goal_complete']
    save_json(root/'independent_verification.json',dict(passed=True,curves=count,new_fits=0,individual_tree_replay=True,label_future_integral_poison_passed=True,family_gate_passed=True,expand=expand,source_sha256=sha(__file__),outer_loaded=False,goal_complete=False))
    print(json.dumps(dict(passed=True,curves=count,expand=expand)),flush=True)

if __name__=='__main__':main()
