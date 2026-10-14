"""TRAIN-only representation diagnosis, explicitly not early-signal prediction."""
import json,shutil,logging,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from battery_smetimes.curve_fidelity import shape_parameters
from battery_smetimes.knee_forecast import fit_knee,trajectory
from scripts.train_tail_boost import load_cells,NCA
from scripts.train_nasa_robust_inner import detail
from scripts.prepare_same_type_forecast import sha,save_json

PARENT=Path('experiments/tail_boost/20261005T034431_681399Z')


def main():
    pc=json.loads((PARENT/'config.json').read_text());plan=next(p for p in pc['plans'] if p['name']=='NCA_fold1')
    # Loader accepts a plan; empty validation/test ensure only TRAIN pickles opened.
    cells=load_cells(dict(plan,validation=[],test=[]))
    assert set(cells)==set(plan['train']) and not set(cells)&set(plan['validation']+plan['test'])
    root=Path('experiments/nca_training_shape_capacity')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');root.mkdir(parents=True)
    print('RUN='+str(root.resolve()),flush=True);snap=root/'source_snapshot';snap.mkdir()
    files=[Path(__file__),Path('battery_smetimes/curve_fidelity.py'),Path('battery_smetimes/knee_forecast.py')]
    for f in files:shutil.copy2(f,snap/f.name)
    index=json.loads((NCA/'index.json').read_text())
    save_json(root/'config.json',dict(parent=str(PARENT.resolve()),parent_config_sha256=sha(PARENT/'config.json'),plan=plan,
        loaded_ids=list(cells),prepared_sha256={k:index['cells'][k]['sha256'] for k in cells},nca_index_sha256=sha(NCA/'index.json'),
        source_sha256={str(f):sha(f) for f in files},candidates=['linear','quadratic','knee'],
        role='TRAIN label-fitted representation audit, not model forecast',outer_loaded=False,validation_loaded=False,
        deterministic=True,shape_fits=117,predictive_fits=0))
    logging.basicConfig(level=logging.INFO,handlers=[logging.FileHandler(root/'diagnosis.log',encoding='utf-8'),logging.StreamHandler()])
    rows=[];arrays={};started=time.perf_counter()
    for k,cell in cells.items():
        c,y=cell['cycle'],cell['truth'];h=(c-100)/100
        for mode in ('linear','quadratic','knee'):
            if mode=='knee':a=fit_knee(c,y);p=trajectory(a,c)
            else:
                a=shape_parameters(c,y,1 if mode=='linear' else 2)
                p=a[0]-a[1]*h-(a[2]*h*h if mode=='quadratic' else 0)
            rows.append(dict(cell_id=k,mode=mode,coefficients=a.tolist(),**detail(c,y,p),
                fitted_with_own_training_future_labels=True,not_forecast=True))
            arrays.update({k+'__'+mode+'__'+f:v for f,v in [('cycle',c),('truth',y),('reconstruction',p),('coefficients',a)]})
        logging.info('SAVED train-cell=%s',k)
    summary={m:{f:float(np.mean([r[f] for r in rows if r['mode']==m])) for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp')}
        for m in ('linear','quadratic','knee')}
    knee=[r for r in rows if r['mode']=='knee'];summary['training_knee_coverage']=dict(
        count=len(knee),active_acceleration=int(sum(r['coefficients'][2]>1e-6 for r in knee)),
        knee_inside_observed_future=int(sum(100+100*r['coefficients'][3]<cells[r['cell_id']]['cycle'][-1] and r['coefficients'][2]>1e-6 for r in knee)),
        fitted_knee_source_cycle=[100+100*r['coefficients'][3] for r in knee])
    save_json(root/'rows.json',rows);np.savez_compressed(root/'reconstructions.npz',**arrays)
    save_json(root/'summary.json',dict(arms=summary,seconds=time.perf_counter()-started,shape_fits=117,predictive_fits=0,
        outer_loaded=False,validation_loaded=False,goal_complete=False,
        warning='Own TRAIN future-label fits quantify representation only; not forecast accuracy or a deployable predictor.'))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
