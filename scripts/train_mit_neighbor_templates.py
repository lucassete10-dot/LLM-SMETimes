"""Fixed two-neighbor-count inner pilot, shared TRAIN template construction."""
import json,shutil,time,logging
from pathlib import Path
from datetime import datetime,timezone
import joblib,numpy as np
from threadpoolctl import threadpool_limits
from battery_smetimes.mit_neighbor_templates import fit,predict,routing_record
from battery_smetimes.curve_fidelity import targets,selection_score
from scripts.train_curve_fidelity import load_subset
from scripts.train_crossfit_drift_smoke import clean
from scripts.train_nasa_robust_inner import detail
from scripts.prepare_same_type_forecast import sha,save_json
from scripts.verify_mit_fixed_origin_dynamics import serial_trees

PARENT=Path('experiments/mit_fixed_origin_dynamics/20261007T053307_927245Z')


def main():
    pc=json.loads((PARENT/'config.json').read_text());assert json.loads((PARENT/'independent_verification.json').read_text())['passed']
    plan=pc['plan'];of=PARENT/'candidate_0/model.joblib';frozen=joblib.load(of)['frozen'];serial_trees(frozen)
    cells=load_subset(plan['train']+plan['validation']);assert not set(cells)&set(plan['test'])
    root=Path('experiments/mit_neighbor_templates')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    root.mkdir(parents=True);print('RUN='+str(root.resolve()),flush=True);snap=root/'source_snapshot';snap.mkdir()
    files=[Path(__file__),Path('battery_smetimes/mit_neighbor_templates.py'),Path('battery_smetimes/mit_segment_features.py'),
        Path('battery_smetimes/knee_forecast.py'),Path('docs/soh_mit_neighbor_templates_20261007.md')]
    for f in files:shutil.copy2(f,snap/f.name)
    save_json(root/'config.json',dict(parent=str(PARENT.resolve()),parent_config_sha256=sha(PARENT/'config.json'),
        parent_proof_sha256=sha(PARENT/'independent_verification.json'),plan=plan,seed=pc['seed'],neighbor_counts=[3,7],control=pc['control'],
        origin_model_file=str(of.resolve()),origin_model_sha256=sha(of),source_sha256={str(f):sha(f) for f in files},
        prepared_sha256={k:v['prepared_sha256'] for k,v in cells.items()},outer_loaded=False,training_role='resubstitution_with_self_neighbor'))
    logging.basicConfig(level=logging.INFO,handlers=[logging.FileHandler(root/'training.log',encoding='utf-8'),logging.StreamHandler()])
    start=time.perf_counter();base=fit(cells,plan['train'],frozen);scores=[];count=0
    for i,n in enumerate([3,7]):
        b=dict(base,neighbors=n);out=root/f'candidate_{i}';out.mkdir();joblib.dump(b,out/'model.joblib')
        for role,ids in [('training',plan['train']),('validation',plan['validation'])]:
            rows=[];arrays={};routes=[]
            for k in ids:
                c,y=targets(cells[k]);q=clean(cells[k]);p=predict(b,q,c)
                rows.append(dict(cell_id=k,**detail(c,y,p),start_error_pp=float(100*abs(p[0]-y[0]))));count+=1
                routes.append(dict(cell_id=k,**routing_record(b,q,c)))
                arrays.update({k+'__100__'+field:v for field,v in [('cycle',c),('truth',y),('prediction',p)]})
            save_json(out/(role+'_metrics.json'),rows);save_json(out/(role+'_routes.json'),routes)
            np.savez_compressed(out/(role+'_predictions.npz'),**arrays)
            if role=='validation':vr=rows
        r=dict(candidate=i,neighbors=n,score=selection_score(vr),
            **{field:float(np.mean([r[field] for r in vr])) for field in ('mape_percent','late_mape','endpoint_pp','drop_error_pp','start_error_pp')})
        scores.append(r);logging.info('SAVED %s',json.dumps(r))
    best=min(scores,key=lambda r:(r['score'],r['candidate']));old=pc['control']
    expand=best['score']<old['score'] and best['mape_percent']<=old['mape_percent']+.1 and best['endpoint_pp']<=old['endpoint_pp'] and best['drop_error_pp']<=old['drop_error_pp']
    save_json(root/'selection.json',dict(candidates=scores,selected=best,control=old,expand=expand,outer_loaded=False,goal_complete=False))
    save_json(root/'progress.json',dict(complete=True,train_label_shape_fits=83,shared_feature_statistics_fits=1,curves=count,seconds=time.perf_counter()-start,outer_loaded=False))
    print(json.dumps(dict(candidates=scores,expand=expand,curves=count)),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
