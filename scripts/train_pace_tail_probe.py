"""Paired full-curve probe: original point output vs causal pace tail."""
from copy import deepcopy
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import traceback
import joblib
import numpy as np
from battery_smetimes import pace_tail
from scripts.train_continuous_length_pilot import load_nominal,target,evaluate
from scripts.prepare_same_type_forecast import save_json,sha


def main():
    base=Path('experiments/condition_length/20261008T020929_340690Z')
    plan=json.loads((base/'config.json').read_text())['plan']
    val=json.loads((base/'validation_coverage.json').read_text())['used']
    cells,_=load_nominal(plan);assert not set(cells)&set(plan['test'])
    train=plan['train']; targets={cid:target(cell,100) for cid,cell in cells.items()}
    root=Path('experiments/pace_tail_probe')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');root.mkdir(parents=True)
    files=[Path(__file__),Path('battery_smetimes/pace_tail.py'),Path('battery_smetimes/condition_length.py'),Path('battery_smetimes/continuous_length.py'),Path('scripts/train_continuous_length_pilot.py')]
    (root/'source_snapshot').mkdir()
    for p in files:shutil.copy2(p,root/'source_snapshot'/p.name)
    save_json(root/'config.json',dict(plan=plan,train=train,validation=val,origins=[75,100],seeds=[20261008],
        common_future_cutoff=100,pace_window=[101,150],source_sha256={str(p):sha(p) for p in files},
        outer_loaded=False,independent_test=False,goal_complete=False,
        hypothesis='Replace plateau outside known-protocol training support using early predicted near-term pace; no test endpoint input.',
        limitation='Near-term pace may underpredict future acceleration. Joining is continuous but not necessarily derivative-continuous.'))
    print('RUN='+str(root.resolve()),flush=True);rows=[];done=0
    try:
        for origin in (75,100):
            b=pace_tail.fit(cells,train,targets,origin,20261008)
            folder=root/str(origin);folder.mkdir();modelpath=folder/'model.joblib';joblib.dump(b,modelpath)
            reloaded=joblib.load(modelpath)
            for extend in (False,True):
                name='pace_tail' if extend else 'paired_local_point';arrays={};batch=[]
                for cid in val:
                    c,y=targets[cid];cell=cells[cid];p=pace_tail.predict(b,cell,c,extend)
                    np.testing.assert_allclose(p,pace_tail.predict(reloaded,cell,c,extend),atol=1e-12,rtol=0)
                    poison=deepcopy(cell);poison['raw'][poison['cycle']>origin]=np.inf
                    poison['soh']=np.inf;poison['reference_capacity_ah']=np.inf;poison['cycle_life']=np.inf
                    np.testing.assert_allclose(p,pace_tail.predict(b,poison,c,extend),atol=1e-12,rtol=0)
                    row=dict(cell_id=cid,candidate=name,origin=origin,seed=20261008,**evaluate(c,y,p));batch.append(row);rows.append(row)
                    arrays.update({cid+'__'+k:v for k,v in [('cycle',c),('truth',y),('prediction',p)]})
                np.savez_compressed(folder/(name+'_predictions.npz'),**arrays)
                save_json(folder/(name+'_metrics.json'),batch)
                print(f'N={origin} {name}: MAPE={np.mean([r["mape_percent"] for r in batch]):.3f}, lateMAEpp={np.mean([r["late20_mae_pp"] for r in batch]):.3f}',flush=True)
            done+=1;save_json(root/'all_rows.json',rows)
            save_json(root/'progress.json',dict(completed_fits=done,terminal=False))
        save_json(root/'progress.json',dict(completed_fits=done,terminal=True,success=True,goal_complete=False))
    except BaseException:
        save_json(root/'failure.json',dict(traceback=traceback.format_exc()))
        save_json(root/'progress.json',dict(completed_fits=done,terminal=True,success=False));raise


if __name__=='__main__':main()
