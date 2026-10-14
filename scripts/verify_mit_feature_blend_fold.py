"""Fresh feature forests and frozen-control exact state before full mixtures."""
import argparse,json
from pathlib import Path
import joblib,numpy as np
from threadpoolctl import threadpool_limits
from scripts.verify_mit_segment_features import refit,replay,targets
from scripts.verify_model_state_portable import equal_state
from battery_smetimes.tail_selection import choose,predict as old_predict
from battery_smetimes.mit_segment_features import predict as actual_new
from battery_smetimes.curve_fidelity import selection_score
from scripts.train_curve_fidelity import load_subset
from scripts.train_crossfit_drift_smoke import clean
from scripts.verify_xjtu_segment_shape_smoke import row
from scripts.verify_nasa_robust_inner import close
from scripts.prepare_same_type_forecast import sha,save_json

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);root=ap.parse_args().run
    cfg=json.loads((root/'config.json').read_text());parent=Path(cfg['parent']);pcfg=json.loads((parent/'config.json').read_text());plan=cfg['plan'];seed=cfg['seed']
    assert cfg['weights']==[0.,.25,.5,.75,1.] and not cfg['outer_loaded'] and not cfg['smoke']
    assert cfg['parameters']==dict(block='V',context='segments')
    assert sha(parent/'config.json')==cfg['parent_config_sha256'] and plan==next(p for p in pcfg['plans'] if p['name']==plan['name'])
    for f,d in cfg['source_sha256'].items():assert sha(f)==d==sha(root/'source_snapshot'/Path(f).name)
    cells=load_subset(plan['train']+plan['validation']);assert not set(cells)&set(plan['test'])
    assert cfg['prepared_sha256']=={k:v['prepared_sha256'] for k,v in cells.items()}
    choice=choose(parent,pcfg,plan,seed,cells);assert choice==cfg['control_choice']
    s=choice['selected'];f=Path(s['selection_file']).parent/f"candidate_{s['candidate']}"/'model.joblib'
    assert str(f.resolve())==cfg['control_model_file'] and sha(f)==cfg['control_model_sha256']==s['model_sha256']
    old=joblib.load(root/'control.joblib');assert old['mode']==s['mode'] and old['family']==cells[plan['train'][0]]['physical_family']
    equal_state(old['inner'],joblib.load(f))
    new=joblib.load(root/'feature.joblib');new=refit(cells,plan['train'],new,cfg['parameters'],seed)
    groups={}
    from scripts.verify_mit_segment_features import condition_key
    for key in plan['train']:groups.setdefault(condition_key(cells[key]),[]).append(key)
    required={k:v for k,v in groups.items() if len(v)>=6};assert set(new['local'])==set(required)
    new['local']={k:refit(cells,ids,new['local'][k],cfg['parameters'],seed) for k,ids in required.items()}
    cache={}
    for key,v in cells.items():
        c,y=targets(v);q=clean(v);a=old_predict(old,q,c);b=replay(new,q,c)
        poison=dict(q,soh=np.inf,label_cycle=np.inf,reference_capacity_ah=np.inf,cycle_life=np.inf,
            cycle=np.r_[q['cycle'],10000],raw=np.vstack([q['raw'],np.full((1,51),np.inf)]))
        np.testing.assert_allclose(a,old_predict(old,poison,c),atol=1e-12,rtol=0)
        poison['raw'][:,[38,39]]=np.inf;np.testing.assert_allclose(b,actual_new(new,poison,c),atol=1e-12,rtol=0)
        for pred,bundle in ((old_predict,old),(actual_new,new)):
            try:pred(bundle,dict(q,physical_family='WRONG'),c)
            except ValueError:pass
            else:raise AssertionError('family gate')
        cache[key]=(c,y,a,b)
    scores=[];count=0
    for i,w in enumerate(cfg['weights']):
        dest=root/f'candidate_{i}';vr=[]
        for role,ids in [('training',plan['train']),('validation',plan['validation'])]:
            stored=json.loads((dest/(role+'_metrics.json')).read_text());assert [r['cell_id'] for r in stored]==ids
            with np.load(dest/(role+'_predictions.npz'),allow_pickle=False) as arrays:
                assert set(arrays.files)=={k+'__100__'+f for k in ids for f in ('cycle','truth','prediction')}
                for j,key in enumerate(ids):
                    c,y,a,b=cache[key];p=(1-w)*a+w*b;stem=key+'__100__'
                    np.testing.assert_array_equal(c,arrays[stem+'cycle']);np.testing.assert_array_equal(y,arrays[stem+'truth']);np.testing.assert_allclose(p,arrays[stem+'prediction'],atol=1e-12,rtol=0)
                    r=dict(**row(key,c,y,p),start_error_pp=float(100*abs(p[0]-y[0])));close(r,stored[j]);count+=1
                    if role=='validation':vr.append(r)
        scores.append(dict(candidate=i,new_weight=w,score=selection_score(vr),**{f:float(np.mean([r[f] for r in vr])) for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp','start_error_pp')}))
    best=min(scores,key=lambda r:(r['score'],r['candidate']));control=scores[0]
    acceptable=best['candidate']>0 and best['score']<control['score'] and best['mape_percent']<=control['mape_percent']+.1 and best['endpoint_pp']<=control['endpoint_pp'] and best['drop_error_pp']<=control['drop_error_pp']
    selected=best if acceptable else control;saved=json.loads((root/'selection.json').read_text())
    close(best,saved['best_score_candidate']);close(selected,saved['selected']);close(control,saved['control'])
    for a,b in zip(scores,saved['candidates']):close(a,b)
    assert len(scores)==len(saved['candidates']) and acceptable==saved['accepted_new'] and not saved['outer_loaded']
    p=json.loads((root/'progress.json').read_text());assert p['complete'] and p['curves']==count and p['new_component_fits']==1+len(required)
    save_json(root/'independent_verification.json',dict(passed=True,independent_component_refits=1+len(required),curves=count,
        frozen_control_exact_state_passed=True,label_future_poison_passed=True,q_poison_new_component_only=True,
        family_gate_passed=True,source_sha256=sha(__file__),state_verifier_sha256=sha('scripts/verify_model_state_portable.py'),outer_loaded=False,goal_complete=False))
    print(json.dumps(dict(passed=True,curves=count,component_refits=1+len(required))),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
