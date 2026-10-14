"""Three preregistered loss-weight arms, fixed training-side fold only."""
import json,shutil,logging,time,traceback
from pathlib import Path
from datetime import datetime,timezone
import joblib,numpy as np
from threadpoolctl import threadpool_limits
from battery_smetimes.mit_tail_weights import fit,predict
from battery_smetimes.curve_fidelity import targets,selection_score
from scripts.train_curve_fidelity import load_subset
from scripts.train_crossfit_drift_smoke import clean
from scripts.train_nasa_robust_inner import detail
from scripts.prepare_same_type_forecast import sha,save_json

PARENT=Path('experiments/mit_feature_blend_full/20261006T143017_054601Z')
CANDIDATES=[dict(block='V',context='segments',scheme=s) for s in ('ramp10','ramp30','terminal10')]


def main():
    parent_cfg=json.loads((PARENT/'config.json').read_text())
    proof=json.loads((PARENT/'independent_verification.json').read_text())
    assert proof['passed'] and not proof['outer_loaded']
    plan=next(p for p in parent_cfg['plans'] if p['name']=='MIT_fold1');seed=20261005
    child=PARENT/plan['name']/str(seed);sel=json.loads((child/'selection.json').read_text())
    cells=load_subset(plan['train']+plan['validation']);assert not set(cells)&set(plan['test'])
    root=Path('experiments/mit_tail_weight_smoke')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    root.mkdir(parents=True);print('RUN='+str(root.resolve()),flush=True)
    snap=root/'source_snapshot';snap.mkdir()
    files=[Path(__file__),Path('battery_smetimes/mit_tail_weights.py'),Path('battery_smetimes/mit_segment_features.py'),
        Path('docs/soh_mit_tail_weight_smoke_20261006.md'),Path('scripts/train_curve_fidelity.py'),
        Path('scripts/train_crossfit_drift_smoke.py'),Path('battery_smetimes/curve_fidelity.py')]
    for f in files:shutil.copy2(f,snap/f.name)
    save_json(root/'config.json',dict(parent=str(PARENT.resolve()),parent_config_sha256=sha(PARENT/'config.json'),
        parent_proof_sha256=sha(PARENT/'independent_verification.json'),plan=plan,seed=seed,candidates=CANDIDATES,
        selected_control=sel['selected'],control_selection_sha256=sha(child/'selection.json'),
        source_sha256={str(f):sha(f) for f in files},prepared_sha256={k:v['prepared_sha256'] for k,v in cells.items()},
        outer_loaded=False,smoke=True,goal_complete=False))
    logging.basicConfig(level=logging.INFO,handlers=[logging.FileHandler(root/'training.log',encoding='utf-8'),logging.StreamHandler()])
    start=time.perf_counter();scores=[];count=0;components=0
    try:
        for i,parameters in enumerate(CANDIDATES):
            out=root/f'candidate_{i}';out.mkdir()
            bundle=fit(cells,plan['train'],parameters,seed);components+=1+len(bundle['local'])
            joblib.dump(bundle,out/'model.joblib')
            for role,ids in [('training',plan['train']),('validation',plan['validation'])]:
                rows=[];arrays={}
                for key in ids:
                    c,y=targets(cells[key]);p=predict(bundle,clean(cells[key]),c)
                    rows.append(dict(cell_id=key,**detail(c,y,p),start_error_pp=float(100*abs(p[0]-y[0]))))
                    arrays.update({key+'__100__'+f:v for f,v in [('cycle',c),('truth',y),('prediction',p)]});count+=1
                save_json(out/(role+'_metrics.json'),rows);np.savez_compressed(out/(role+'_predictions.npz'),**arrays)
                if role=='validation':vr=rows
            r=dict(candidate=i,parameters=parameters,score=selection_score(vr),**{f:float(np.mean([r[f] for r in vr])) for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp','start_error_pp')})
            scores.append(r);logging.info('SAVED %s',json.dumps(r))
        best=min(scores,key=lambda r:(r['score'],r['candidate']));old=sel['selected']
        expand=best['score']<old['score'] and best['mape_percent']<=old['mape_percent']+.1 and best['endpoint_pp']<=old['endpoint_pp'] and best['drop_error_pp']<=old['drop_error_pp']
        save_json(root/'selection.json',dict(selected=best,control=old,candidates=scores,expand=expand,outer_loaded=False,goal_complete=False))
        save_json(root/'progress.json',dict(complete=True,curves=count,component_fits=components,seconds=time.perf_counter()-start,outer_loaded=False))
        print(json.dumps(dict(selected=best,expand=expand,curves=count,component_fits=components)),flush=True)
    except BaseException:save_json(root/'failure.json',dict(traceback=traceback.format_exc()));raise


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
