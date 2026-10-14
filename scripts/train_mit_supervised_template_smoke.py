"""Two fixed original-inner SOH candidates; no outer queries."""
import json,shutil,time,logging
from pathlib import Path
from datetime import datetime,timezone
import joblib,numpy as np
from threadpoolctl import threadpool_limits
from battery_smetimes.mit_supervised_templates import fit,predict,routing_record
from battery_smetimes.curve_fidelity import targets,selection_score
from scripts.train_curve_fidelity import load_subset
from scripts.train_crossfit_drift_smoke import clean
from scripts.train_nasa_robust_inner import detail
from scripts.verify_mit_fixed_origin_dynamics import serial_trees
from scripts.prepare_same_type_forecast import sha,save_json

PARENT=Path('experiments/mit_neighbor_templates/20261007T054345_631824Z')
SCREEN=Path('experiments/mit_supervised_template_route/20261007T055323_515698Z')


def main():
    pc=json.loads((PARENT/'config.json').read_text());assert json.loads((PARENT/'independent_verification.json').read_text())['passed']
    assert json.loads((SCREEN/'independent_verification.json').read_text())['passed']
    plan=pc['plan'];cf=PARENT/'candidate_0/model.joblib';cache=joblib.load(cf);serial_trees(cache['frozen'])
    cells=load_subset(plan['train']+plan['validation']);assert not set(cells)&set(plan['test'])
    root=Path('experiments/mit_supervised_template_smoke')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    root.mkdir(parents=True);print('RUN='+str(root.resolve()),flush=True);snap=root/'source_snapshot';snap.mkdir()
    files=[Path(__file__),Path('battery_smetimes/mit_supervised_templates.py'),Path('battery_smetimes/mit_pace_forecast.py'),
        Path('battery_smetimes/mit_template_tangent.py'),Path('battery_smetimes/mit_segment_features.py'),Path('docs/soh_mit_supervised_template_smoke_20261007.md')]
    for f in files:shutil.copy2(f,snap/f.name)
    deps=[PARENT/'config.json',PARENT/'independent_verification.json',cf,SCREEN/'config.json',SCREEN/'independent_verification.json']
    save_json(root/'config.json',dict(parent=str(PARENT.resolve()),screen=str(SCREEN.resolve()),input_sha256={str(f.resolve()):sha(f) for f in deps},
        cache_file=str(cf.resolve()),plan=plan,seed=pc['seed'],alphas=[10.,100.],control=pc['control'],
        source_sha256={str(f):sha(f) for f in files},prepared_sha256={k:v['prepared_sha256'] for k,v in cells.items()},outer_loaded=False))
    logging.basicConfig(level=logging.INFO,handlers=[logging.FileHandler(root/'training.log',encoding='utf-8'),logging.StreamHandler()])
    start=time.perf_counter();scores=[];count=0
    for i,alpha in enumerate([10.,100.]):
        b=fit(cache,cells,alpha);out=root/f'candidate_{i}';out.mkdir();joblib.dump(b,out/'model.joblib')
        for role,ids in [('training',plan['train']),('validation',plan['validation'])]:
            rows=[];routes=[];arrays={}
            for k in ids:
                c,y=targets(cells[k]);q=clean(cells[k]);p=predict(b,q,c)
                rows.append(dict(cell_id=k,**detail(c,y,p),start_error_pp=float(100*abs(p[0]-y[0]))));count+=1
                routes.append(dict(cell_id=k,**routing_record(b,q)))
                arrays.update({k+'__100__'+field:v for field,v in [('cycle',c),('truth',y),('prediction',p)]})
            save_json(out/(role+'_metrics.json'),rows);save_json(out/(role+'_routes.json'),routes);np.savez_compressed(out/(role+'_predictions.npz'),**arrays)
            if role=='validation':vr=rows
        s=dict(candidate=i,alpha=alpha,score=selection_score(vr),
            **{field:float(np.mean([r[field] for r in vr])) for field in ('mape_percent','late_mape','endpoint_pp','drop_error_pp','start_error_pp')})
        scores.append(s);logging.info('SAVED %s',json.dumps(s))
    best=min(scores,key=lambda r:(r['score'],r['candidate']));old=pc['control']
    expand=best['score']<old['score'] and best['mape_percent']<=old['mape_percent']+.1 and best['endpoint_pp']<=old['endpoint_pp'] and best['drop_error_pp']<=old['drop_error_pp']
    save_json(root/'selection.json',dict(candidates=scores,selected=best,control=old,expand=expand,outer_loaded=False,goal_complete=False))
    save_json(root/'progress.json',dict(complete=True,new_pace_heads=2,new_template_fits=0,curves=count,seconds=time.perf_counter()-start,outer_loaded=False))
    print(json.dumps(dict(candidates=scores,expand=expand,curves=count)),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
