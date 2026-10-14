"""Prespecified constant blending of independently verified inner components."""
import json,shutil,logging
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from battery_smetimes.curve_fidelity import selection_score
from scripts.train_nasa_robust_inner import detail
from scripts.prepare_same_type_forecast import sha,save_json

PARENT=Path('experiments/mit_segment_features/20261006T141941_849519Z')
WEIGHTS=[0.,.25,.5,.75,1.]

def main():
    cfg=json.loads((PARENT/'config.json').read_text());plan=cfg['plan']
    proof=json.loads((PARENT/'independent_verification.json').read_text());assert proof['passed'] and proof['independent_component_refits']==18 and not proof['outer_loaded']
    assert cfg['candidates'][2]==dict(block='V',context='segments') and plan['name']=='MIT_fold1' and cfg['seed']==20261005
    root=Path('experiments/mit_feature_blend')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');root.mkdir(parents=True);print('RUN='+str(root.resolve()),flush=True)
    snap=root/'source_snapshot';snap.mkdir();files=[Path(__file__),Path('docs/soh_mit_feature_blend_smoke_20261006.md')]
    for f in files:shutil.copy2(f,snap/f.name)
    dependency=[PARENT/f'candidate_{i}'/n for i in (0,2) for n in ('model.joblib','training_predictions.npz','validation_predictions.npz')]
    save_json(root/'config.json',dict(parent=str(PARENT.resolve()),parent_config_sha256=sha(PARENT/'config.json'),parent_proof_sha256=sha(PARENT/'independent_verification.json'),
        component_sha256={str(f):sha(f) for f in dependency},plan=plan,seed=cfg['seed'],weights=WEIGHTS,
        source_sha256={str(f):sha(f) for f in files},outer_loaded=False,new_fits=0,legacy_fixed_window_integrals_in_control=True))
    logging.basicConfig(level=logging.INFO,handlers=[logging.FileHandler(root/'screen.log',encoding='utf-8'),logging.StreamHandler()]);scores=[];count=0
    for i,w in enumerate(WEIGHTS):
        dest=root/f'candidate_{i}';dest.mkdir();vr=[]
        for role,ids in [('training',plan['train']),('validation',plan['validation'])]:
            rows=[];out={}
            with np.load(PARENT/f'candidate_0/{role}_predictions.npz') as a,np.load(PARENT/f'candidate_2/{role}_predictions.npz') as b:
                for key in ids:
                    s=key+'__100__';c=a[s+'cycle'];y=a[s+'truth'];np.testing.assert_array_equal(c,b[s+'cycle']);np.testing.assert_array_equal(y,b[s+'truth'])
                    p=(1-w)*a[s+'prediction']+w*b[s+'prediction']
                    rows.append(dict(cell_id=key,**detail(c,y,p),start_error_pp=float(100*abs(p[0]-y[0]))))
                    out.update({s+f:v for f,v in [('cycle',c),('truth',y),('prediction',p)]});count+=1
            save_json(dest/(role+'_metrics.json'),rows);np.savez_compressed(dest/(role+'_predictions.npz'),**out)
            if role=='validation':vr=rows
        r=dict(candidate=i,new_weight=w,score=selection_score(vr),**{f:float(np.mean([r[f] for r in vr])) for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp','start_error_pp')})
        scores.append(r);logging.info('SAVED %s',json.dumps(r))
    best=min(scores,key=lambda r:(r['score'],r['candidate']));old=scores[0]
    expand=best['candidate']>0 and best['score']<old['score'] and best['mape_percent']<=old['mape_percent']+.1 and best['endpoint_pp']<=old['endpoint_pp'] and best['drop_error_pp']<=old['drop_error_pp']
    save_json(root/'selection.json',dict(selected=best,control=old,candidates=scores,expand=expand,outer_loaded=False,goal_complete=False))
    save_json(root/'progress.json',dict(complete=True,new_fits=0,curves=count,outer_loaded=False,goal_complete=False))
    print(json.dumps(dict(selected=best,expand=expand,curves=count)),flush=True)

if __name__=='__main__':main()
