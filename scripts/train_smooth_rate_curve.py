"""Frozen small-scale comparison on the preceding protocol experiment's cells."""
from copy import deepcopy
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import traceback
import joblib
import numpy as np
from battery_smetimes import smooth_rate_curve as smooth
from battery_smetimes import condition_length as previous
from scripts.train_continuous_length_pilot import load_nominal,target,evaluate
from scripts.prepare_same_type_forecast import sha,save_json

BASE=Path('experiments/condition_length/20261008T020929_340690Z')


def main():
    cfg=json.loads((BASE/'config.json').read_text());plan=cfg['plan'];origin=100;seed=cfg['seeds'][0]
    train=json.loads((BASE/'train_coverage.json').read_text())['used']
    val=json.loads((BASE/'validation_coverage.json').read_text())['used']
    root=Path('experiments/smooth_rate_curve')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    root.mkdir(parents=True);snap=root/'source_snapshot';snap.mkdir()
    files=[Path(__file__),Path('battery_smetimes/smooth_rate_curve.py'),Path('battery_smetimes/continuous_length.py'),Path('scripts/train_continuous_length_pilot.py')]
    for f in files:shutil.copy2(f,snap/f.name)
    save_json(root/'config.json',dict(plan=plan,used_train=train,validation=val,origin=origin,seed=seed,
        strengths=[.001,.01],routes=['global','local'],source_sha256={str(f):sha(f) for f in files},
        base_run=str(BASE),base_config_sha256=sha(BASE/'config.json'),outer_loaded=False,independent_test=False,
        rationale='Fixed positive rate-increment basis avoids nonlinear parameter averaging, flat end-value extension and unbounded quadratic slope.',
        limitation='Acceleration-only long-term trend; no guarantee for recovery/deceleration. Single-seed development screen.'))
    print('RUN='+str(root.resolve()),flush=True);done=0;allrows=[]
    try:
        cells,_=load_nominal(plan);assert not set(cells)&set(plan['test'])
        targets={k:target(c,100) for k,c in cells.items()}
        for strength in (.001,.01):
            for route in ('global','local'):
                name=f'{route}_{strength}';folder=root/name;folder.mkdir()
                b=smooth.fit(cells,train,targets,origin,seed,strength,route=='local')
                joblib.dump(b,folder/'model.joblib');saved=joblib.load(folder/'model.joblib');rows=[];arrays={}
                for k in val:
                    c,y=targets[k];p=smooth.predict(b,cells[k],c)
                    np.testing.assert_allclose(p,smooth.predict(saved,cells[k],c),rtol=0,atol=1e-12)
                    poison=deepcopy(cells[k]);poison['raw'][poison['cycle']>origin]=np.inf
                    poison['soh']=np.full_like(poison['soh'],np.inf);poison['reference_capacity_ah']=np.inf;poison['cycle_life']=np.inf
                    np.testing.assert_allclose(p,smooth.predict(b,poison,c),rtol=0,atol=1e-12)
                    grid=np.linspace(c[0],c[-1],500);dense=smooth.predict(b,cells[k],grid)
                    assert np.max(np.diff(dense))<=1e-10 and np.max(np.diff(dense,n=2))<=1e-9
                    try:smooth.predict(b,dict(cells[k],physical_family='wrong'),c)
                    except ValueError:pass
                    else:raise AssertionError('family gate')
                    rows.append(dict(cell_id=k,**evaluate(c,y,p)))
                    arrays.update({k+'__'+n:v for n,v in [('cycle',c),('truth',y),('prediction',p)]})
                save_json(folder/'metrics.json',rows);np.savez_compressed(folder/'validation_predictions.npz',**arrays)
                save_json(folder/'config.json',dict(candidate=name,model_sha256=sha(folder/'model.joblib'),used_train=train,validation=val))
                allrows.extend(dict(candidate=name,origin=origin,seed=seed,**r) for r in rows);done+=1
                save_json(root/'all_rows.json',allrows);save_json(root/'progress.json',dict(completed_fits=done,terminal=False))
                print(name+' MAPE='+str(np.mean([r['mape_percent'] for r in rows]))+' lateMAEpp='+str(np.mean([r['late20_mae_pp'] for r in rows])),flush=True)
        save_json(root/'progress.json',dict(completed_fits=done,terminal=True,success=True,goal_complete=False))
    except BaseException:
        save_json(root/'failure.json',dict(traceback=traceback.format_exc()));save_json(root/'progress.json',dict(completed_fits=done,terminal=True,success=False));raise


if __name__=='__main__':main()
