"""Two fixed shape-only fits with immutable prior origin evidence."""
import json,shutil,time,logging
from pathlib import Path
from datetime import datetime,timezone
import joblib,numpy as np,torch
from threadpoolctl import threadpool_limits
from battery_smetimes.mit_fixed_origin_dynamics import fit,predict
from battery_smetimes.curve_fidelity import targets,selection_score
from scripts.train_curve_fidelity import load_subset
from scripts.train_crossfit_drift_smoke import clean
from scripts.train_nasa_robust_inner import detail
from scripts.prepare_same_type_forecast import sha,save_json

PARENT=Path('experiments/mit_feature_blend_full/20261006T143017_054601Z')


def main():
    pc=json.loads((PARENT/'config.json').read_text());proof=json.loads((PARENT/'independent_verification.json').read_text())
    assert proof['passed'] and not proof['outer_loaded']
    plan=next(p for p in pc['plans'] if p['name']=='MIT_fold1');seed=20261005
    child=PARENT/plan['name']/str(seed);sf=child/'selection.json';control=json.loads(sf.read_text())['selected']
    files0=[sf,child/'control.joblib',child/'feature.joblib']
    frozen=dict(control=joblib.load(files0[1]),feature=joblib.load(files0[2]),new_weight=control['new_weight'])
    cells=load_subset(plan['train']+plan['validation']);assert not set(cells)&set(plan['test'])
    root=Path('experiments/mit_fixed_origin_dynamics')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    root.mkdir(parents=True);print('RUN='+str(root.resolve()),flush=True);snap=root/'source_snapshot';snap.mkdir()
    files=[Path(__file__),Path('battery_smetimes/mit_fixed_origin_dynamics.py'),Path('battery_smetimes/monotone_network.py'),
        Path('battery_smetimes/mit_segment_features.py'),Path('docs/soh_mit_fixed_origin_dynamics_20261007.md')]
    for f in files:shutil.copy2(f,snap/f.name)
    save_json(root/'config.json',dict(parent=str(PARENT.resolve()),parent_config_sha256=sha(PARENT/'config.json'),
        parent_proof_sha256=sha(PARENT/'independent_verification.json'),plan=plan,seed=seed,widths=[16,32],epochs=400,
        control=control,origin_files={str(f.resolve()):sha(f) for f in files0},
        source_sha256={str(f):sha(f) for f in files},prepared_sha256={k:v['prepared_sha256'] for k,v in cells.items()},
        outer_loaded=False,torch_version=torch.__version__,device='cpu',torch_threads=1))
    logging.basicConfig(level=logging.INFO,handlers=[logging.FileHandler(root/'training.log',encoding='utf-8'),logging.StreamHandler()])
    start=time.perf_counter();scores=[];curves=0
    for i,width in enumerate([16,32]):
        out=root/f'candidate_{i}';out.mkdir();b=fit(cells,plan['train'],width,seed,frozen)
        joblib.dump(b,out/'model.joblib');save_json(out/'epoch_loss.json',b['history'])
        for role,ids in [('training',plan['train']),('validation',plan['validation'])]:
            rows=[];arrays={}
            for k in ids:
                c,y=targets(cells[k]);p=predict(b,clean(cells[k]),c)
                rows.append(dict(cell_id=k,**detail(c,y,p),start_error_pp=float(100*abs(p[0]-y[0]))));curves+=1
                arrays.update({k+'__100__'+f:v for f,v in [('cycle',c),('truth',y),('prediction',p)]})
            save_json(out/(role+'_metrics.json'),rows);np.savez_compressed(out/(role+'_predictions.npz'),**arrays)
            if role=='validation':vr=rows
        r=dict(candidate=i,width=width,shape_parameter_count=b['parameter_count'],score=selection_score(vr),
            **{f:float(np.mean([r[f] for r in vr])) for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp','start_error_pp')})
        scores.append(r);logging.info('SAVED %s',json.dumps(r))
    best=min(scores,key=lambda r:(r['score'],r['candidate']))
    expand=best['score']<control['score'] and best['mape_percent']<=control['mape_percent']+.1 and best['endpoint_pp']<=control['endpoint_pp'] and best['drop_error_pp']<=control['drop_error_pp']
    save_json(root/'selection.json',dict(selected=best,control=control,candidates=scores,expand=expand,outer_loaded=False,goal_complete=False))
    save_json(root/'progress.json',dict(complete=True,new_fits=2,reused_origin=True,curves=curves,seconds=time.perf_counter()-start,outer_loaded=False))
    print(json.dumps(dict(selected=best,expand=expand,curves=curves)),flush=True)


if __name__=='__main__':
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):main()
