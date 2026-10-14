"""Evaluate only two preregistered frozen TRAIN models, no new fitting."""
import json,shutil
from pathlib import Path
from datetime import datetime,timezone
import joblib,numpy as np,torch
from threadpoolctl import threadpool_limits
from battery_smetimes.mit_fixed_origin_dynamics import predict
from battery_smetimes.curve_fidelity import targets,selection_score
from scripts.train_curve_fidelity import load_subset
from scripts.train_crossfit_drift_smoke import clean
from scripts.train_nasa_robust_inner import detail
from scripts.prepare_same_type_forecast import sha,save_json
from scripts.verify_mit_fixed_origin_dynamics import serial_trees

PARENT=Path('experiments/mit_dynamics_convergence/20261007T053616_282833Z')


def main():
    pc=json.loads((PARENT/'config.json').read_text());assert json.loads((PARENT/'independent_verification.json').read_text())['passed']
    oldroot=Path(pc['parent']);oldcfg=json.loads((oldroot/'config.json').read_text());control=oldcfg['control'];plan=pc['plan']
    cells=load_subset(plan['validation']);assert set(cells)==set(plan['validation']) and not set(cells)&set(plan['train']+plan['test'])
    root=Path('experiments/mit_1600_locked_validation')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    root.mkdir(parents=True);print('RUN='+str(root.resolve()),flush=True);snap=root/'source_snapshot';snap.mkdir()
    files=[Path(__file__),Path('docs/soh_mit_1600_locked_validation_20261007.md'),Path('battery_smetimes/mit_fixed_origin_dynamics.py')]
    for f in files:shutil.copy2(f,snap/f.name)
    models=[PARENT/f'candidate_{i}'/'model.joblib' for i in range(2)]
    save_json(root/'config.json',dict(parent=str(PARENT.resolve()),parent_config_sha256=sha(PARENT/'config.json'),
        parent_proof_sha256=sha(PARENT/'independent_verification.json'),old_config_file=str((oldroot/'config.json').resolve()),
        old_config_sha256=sha(oldroot/'config.json'),plan=plan,seed=pc['seed'],model_files={str(f.resolve()):sha(f) for f in models},
        source_sha256={str(f):sha(f) for f in files},prepared_sha256={k:v['prepared_sha256'] for k,v in cells.items()},control=control,
        new_fits=0,outer_loaded=False))
    scores=[]
    for i,f in enumerate(models):
        b=joblib.load(f);serial_trees(b['frozen']);assert b['train_ids']==plan['train'] and b['epochs']==1600
        out=root/f'candidate_{i}';out.mkdir();rows=[];arrays={}
        for k in plan['validation']:
            c,y=targets(cells[k]);p=predict(b,clean(cells[k]),c)
            rows.append(dict(cell_id=k,**detail(c,y,p),start_error_pp=float(100*abs(p[0]-y[0]))))
            arrays.update({k+'__100__'+field:v for field,v in [('cycle',c),('truth',y),('prediction',p)]})
        save_json(out/'validation_metrics.json',rows);np.savez_compressed(out/'validation_predictions.npz',**arrays)
        scores.append(dict(candidate=i,width=b['width'],score=selection_score(rows),
            **{field:float(np.mean([r[field] for r in rows])) for field in ('mape_percent','late_mape','endpoint_pp','drop_error_pp','start_error_pp')}))
    best=min(scores,key=lambda r:(r['score'],r['candidate']))
    expand=best['score']<control['score'] and best['mape_percent']<=control['mape_percent']+.1 and best['endpoint_pp']<=control['endpoint_pp'] and best['drop_error_pp']<=control['drop_error_pp']
    save_json(root/'selection.json',dict(candidates=scores,selected=best,control=control,expand=expand,goal_complete=False,outer_loaded=False))
    save_json(root/'progress.json',dict(complete=True,new_fits=0,curves=48,outer_loaded=False))
    print(json.dumps(dict(candidates=scores,expand=expand)),flush=True)


if __name__=='__main__':
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):main()
