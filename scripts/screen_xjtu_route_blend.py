"""Fixed feature/model routing ablation; no new fits or outer queries."""
import json, shutil, logging
from pathlib import Path
from datetime import datetime, timezone
import joblib, numpy as np
from battery_smetimes.xjtu_segment_features import predict
from battery_smetimes.curve_fidelity import targets, selection_score, condition_key
from scripts.train_curve_fidelity import load_subset
from scripts.train_crossfit_drift_smoke import clean
from scripts.train_nasa_robust_inner import detail
from scripts.prepare_same_type_forecast import sha, save_json

PARENT=Path('experiments/xjtu_segment_full/20261006T132530_948247Z/XJTU_fold1/20261005')
WEIGHTS=[1.,0.,.5]

def route(bundle,query,future,weight):
    if weight not in WEIGHTS: raise ValueError('unregistered weight')
    global_only=dict(bundle,local={})
    global_p=predict(global_only,query,future)
    local_p=predict(bundle,query,future)
    return (1-weight)*global_p+weight*local_p

def decision(scores):
    best=min(scores,key=lambda r:(r['score'],r['candidate']))
    control=scores[0]
    expand=best['candidate']!=0 and best['score']<control['score'] and best['mape_percent']<=control['mape_percent']+.1 and best['endpoint_pp']<=control['endpoint_pp'] and best['drop_error_pp']<=control['drop_error_pp']
    return dict(selected=best,control=control,candidates=scores,expand=expand,outer_loaded=False,goal_complete=False)

def main():
    cc=json.loads((PARENT/'config.json').read_text());plan=cc['plan']
    proof=json.loads((PARENT/'independent_verification.json').read_text());assert proof['passed'] and not proof['outer_loaded']
    sel=json.loads((PARENT/'selection.json').read_text());assert sel['selected']['candidate']==6
    model_file=PARENT/'candidate_6/model.joblib';bundle=joblib.load(model_file)
    assert bundle['parameters']==dict(block='VIT',context='segments')
    cells=load_subset(plan['train']+plan['validation']); assert not set(cells)&set(plan['test'])
    root=Path('experiments/xjtu_route_blend')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');root.mkdir(parents=True)
    print('RUN='+str(root.resolve()),flush=True);snap=root/'source_snapshot';snap.mkdir()
    files=[Path(__file__),Path('docs/soh_xjtu_routing_smoke_20261006.md'),Path('battery_smetimes/xjtu_segment_features.py')]
    for f in files:shutil.copy2(f,snap/f.name)
    save_json(root/'config.json',dict(parent=str(PARENT.resolve()),plan=plan,seed=cc['seed'],weights=WEIGHTS,
        model_file=str(model_file.resolve()),model_sha256=sha(model_file),parent_config_sha256=sha(PARENT/'config.json'),
        parent_proof_sha256=sha(PARENT/'independent_verification.json'),source_sha256={str(f):sha(f) for f in files},
        prepared_sha256={k:c['prepared_sha256'] for k,c in cells.items()},new_fits=0,outer_loaded=False))
    logging.basicConfig(level=logging.INFO,handlers=[logging.FileHandler(root/'screen.log',encoding='utf-8'),logging.StreamHandler()])
    scores=[];count=0
    for i,w in enumerate(WEIGHTS):
        dest=root/f'candidate_{i}';dest.mkdir();vr=[]
        for role,ids in [('training',plan['train']),('validation',plan['validation'])]:
            rows=[];a={}
            for key in ids:
                c,y=targets(cells[key]);p=route(bundle,clean(cells[key]),c,w)
                rows.append(dict(cell_id=key,**detail(c,y,p),start_error_pp=float(100*abs(p[0]-y[0])),has_local=condition_key(cells[key]) in bundle['local']))
                a.update({key+'__100__'+f:v for f,v in [('cycle',c),('truth',y),('prediction',p)]});count+=1
            save_json(dest/(role+'_metrics.json'),rows);np.savez_compressed(dest/(role+'_predictions.npz'),**a)
            if role=='validation':vr=rows
        r=dict(candidate=i,local_weight=w,score=selection_score(vr),**{f:float(np.mean([r[f] for r in vr])) for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp','start_error_pp')})
        scores.append(r);logging.info('SAVED %s',json.dumps(r))
    result=decision(scores);save_json(root/'selection.json',result);save_json(root/'progress.json',dict(complete=True,new_fits=0,curves=count,outer_loaded=False,goal_complete=False))
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
