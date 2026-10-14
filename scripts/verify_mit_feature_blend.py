"""Independent component inference and arithmetic for every constant mixture."""
import argparse,json
from pathlib import Path
import joblib,numpy as np
from scripts.verify_mit_segment_features import replay as new_replay,targets
from battery_smetimes.tail_selection import predict as old_predict
from battery_smetimes.mit_segment_features import predict as actual_new
from battery_smetimes.curve_fidelity import selection_score
from scripts.train_curve_fidelity import load_subset
from scripts.train_crossfit_drift_smoke import clean
from scripts.verify_xjtu_segment_shape_smoke import row
from scripts.verify_nasa_robust_inner import close
from scripts.prepare_same_type_forecast import sha,save_json

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);root=ap.parse_args().run
    cfg=json.loads((root/'config.json').read_text());parent=Path(cfg['parent']);pcfg=json.loads((parent/'config.json').read_text());plan=cfg['plan']
    assert cfg['weights']==[0.,.25,.5,.75,1.] and cfg['new_fits']==0 and not cfg['outer_loaded']
    assert sha(parent/'config.json')==cfg['parent_config_sha256'] and sha(parent/'independent_verification.json')==cfg['parent_proof_sha256']
    assert json.loads((parent/'independent_verification.json').read_text())['passed']
    assert plan==pcfg['plan'] and cfg['seed']==pcfg['seed']
    for f,d in cfg['source_sha256'].items():assert sha(f)==d==sha(root/'source_snapshot'/Path(f).name)
    for f,d in cfg['component_sha256'].items():assert sha(f)==d
    cells=load_subset(plan['train']+plan['validation']);assert not set(cells)&set(plan['test'])
    assert pcfg['prepared_sha256']=={k:v['prepared_sha256'] for k,v in cells.items()}
    old=joblib.load(parent/'candidate_0/model.joblib');new=joblib.load(parent/'candidate_2/model.joblib')
    cache={}
    for key,v in cells.items():
        c,y=targets(v);q=clean(v);a=old_predict(old,q,c);b=new_replay(new,q,c)
        poison=dict(q,soh=np.inf,label_cycle=np.inf,reference_capacity_ah=np.inf,cycle_life=np.inf,
            cycle=np.r_[q['cycle'],10000],raw=np.vstack([q['raw'],np.full((1,51),np.inf)]))
        np.testing.assert_allclose(a,old_predict(old,poison,c),atol=1e-12,rtol=0)
        poison['raw'][:,[38,39]]=np.inf
        np.testing.assert_allclose(b,actual_new(new,poison,c),atol=1e-12,rtol=0)
        for predictor,bundle in ((old_predict,old),(actual_new,new)):
            try:predictor(bundle,dict(q,physical_family='WRONG'),c)
            except ValueError:pass
            else:raise AssertionError('family isolation')
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
    best=min(scores,key=lambda r:(r['score'],r['candidate']));old=scores[0]
    expand=best['candidate']>0 and best['score']<old['score'] and best['mape_percent']<=old['mape_percent']+.1 and best['endpoint_pp']<=old['endpoint_pp'] and best['drop_error_pp']<=old['drop_error_pp']
    saved=json.loads((root/'selection.json').read_text());close(best,saved['selected']);close(old,saved['control'])
    for a,b in zip(scores,saved['candidates']):close(a,b)
    assert len(scores)==len(saved['candidates']) and expand==saved['expand']
    assert count==json.loads((root/'progress.json').read_text())['curves']
    save_json(root/'independent_verification.json',dict(passed=True,curves=count,new_fits=0,
        inherited_parent_refits=18,component_inference_and_numeric_replay=True,label_future_poison_passed=True,
        q_poison_checked_new_component_only=True,family_gate_passed=True,expand=expand,
        source_sha256=sha(__file__),outer_loaded=False,goal_complete=False))
    print(json.dumps(dict(passed=True,curves=count,expand=expand)),flush=True)

if __name__=='__main__':main()
