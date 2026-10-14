"""Fixed TRAIN-only 1600-epoch replay; no validation selection."""
import json,shutil,time,logging
from pathlib import Path
from datetime import datetime,timezone
import joblib,numpy as np,torch
from threadpoolctl import threadpool_limits
from battery_smetimes.mit_fixed_origin_dynamics import fit,predict
from battery_smetimes.curve_fidelity import targets
from scripts.train_curve_fidelity import load_subset
from scripts.train_crossfit_drift_smoke import clean
from scripts.train_nasa_robust_inner import detail
from scripts.prepare_same_type_forecast import sha,save_json
from scripts.verify_mit_fixed_origin_dynamics import serial_trees

PARENT=Path('experiments/mit_fixed_origin_dynamics/20261007T053307_927245Z')
FIELDS=('mape_percent','late_mape','endpoint_pp','drop_error_pp','start_error_pp')


def main():
    cfg=json.loads((PARENT/'config.json').read_text());proof=json.loads((PARENT/'independent_verification.json').read_text());assert proof['passed']
    plan=cfg['plan'];cells=load_subset(plan['train'])
    assert set(cells)==set(plan['train']) and not set(cells)&set(plan['validation']+plan['test'])
    root=Path('experiments/mit_dynamics_convergence')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    root.mkdir(parents=True);print('RUN='+str(root.resolve()),flush=True);snap=root/'source_snapshot';snap.mkdir()
    files=[Path(__file__),Path('battery_smetimes/mit_fixed_origin_dynamics.py'),Path('battery_smetimes/monotone_network.py'),
        Path('battery_smetimes/mit_segment_features.py'),Path('scripts/verify_mit_fixed_origin_dynamics.py'),
        Path('docs/soh_mit_dynamics_convergence_20261007.md')]
    for f in files:shutil.copy2(f,snap/f.name)
    dependencies={str((PARENT/f'candidate_{i}'/name).resolve()):sha(PARENT/f'candidate_{i}'/name)
        for i in range(2) for name in ('model.joblib','training_metrics.json','epoch_loss.json')}
    save_json(root/'config.json',dict(parent=str(PARENT.resolve()),parent_config_sha256=sha(PARENT/'config.json'),
        parent_proof_sha256=sha(PARENT/'independent_verification.json'),parent_artifacts=dependencies,plan=plan,seed=cfg['seed'],
        widths=cfg['widths'],epochs=1600,old_epochs=400,plateau_window=100,plateau_relative_drop_threshold=.01,
        source_sha256={str(f):sha(f) for f in files},prepared_sha256={k:v['prepared_sha256'] for k,v in cells.items()},
        validation_loaded=False,outer_loaded=False,device='cpu',torch_threads=1,torch_version=torch.__version__))
    logging.basicConfig(level=logging.INFO,handlers=[logging.FileHandler(root/'training.log',encoding='utf-8'),logging.StreamHandler()])
    start=time.perf_counter();reports=[];curves=0
    for i,width in enumerate(cfg['widths']):
        old=joblib.load(PARENT/f'candidate_{i}'/'model.joblib');serial_trees(old['frozen'])
        b=fit(cells,plan['train'],width,cfg['seed'],old['frozen'],1600)
        np.testing.assert_array_equal(b['history'][:400],old['history'])
        out=root/f'candidate_{i}';out.mkdir();joblib.dump(b,out/'model.joblib');save_json(out/'epoch_loss.json',b['history'])
        rows=[];arrays={}
        for k in plan['train']:
            c,y=targets(cells[k]);p=predict(b,clean(cells[k]),c)
            rows.append(dict(cell_id=k,**detail(c,y,p),start_error_pp=float(100*abs(p[0]-y[0]))));curves+=1
            arrays.update({k+'__100__'+f:v for f,v in [('cycle',c),('truth',y),('prediction',p)]})
        save_json(out/'training_metrics.json',rows);np.savez_compressed(out/'training_predictions.npz',**arrays)
        oldrows=json.loads((PARENT/f'candidate_{i}'/'training_metrics.json').read_text())
        prior=float(np.mean(b['history'][-200:-100]));last=float(np.mean(b['history'][-100:]));drop=(prior-last)/prior
        report=dict(candidate=i,width=width,old_training={f:float(np.mean([r[f] for r in oldrows])) for f in FIELDS},
            new_training={f:float(np.mean([r[f] for r in rows])) for f in FIELDS},
            epoch400_loss=b['history'][399],epoch1600_loss=b['history'][-1],prior100_loss=prior,last100_loss=last,
            last_relative_drop=drop,plateau_by_fixed_rule=bool(drop<=.01),first400_loss_exact=True)
        reports.append(report);logging.info('SAVED %s',json.dumps(report))
    save_json(root/'summary.json',dict(reports=reports,new_predictive_fits=2,curves=curves,seconds=time.perf_counter()-start,
        validation_loaded=False,outer_loaded=False,adopted=False,goal_complete=False,
        interpretation='TRAIN-only optimizer adequacy diagnostic, not unseen-cell accuracy or validation-based epoch selection.'))
    print(json.dumps(dict(curves=curves,adopted=False,reports=reports)),flush=True)


if __name__=='__main__':
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):main()
