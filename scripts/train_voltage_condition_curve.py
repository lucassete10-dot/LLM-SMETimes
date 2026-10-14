"""Full-curve voltage ablation motivated by training-cell-only rate diagnostics."""
from copy import deepcopy
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import joblib
import numpy as np
from battery_smetimes import continuous_length as point
from battery_smetimes import condition_length as local
from scripts.train_continuous_length_pilot import load_nominal,target,evaluate
from scripts.prepare_same_type_forecast import sha,save_json

BASE=Path('experiments/condition_length/20261008T020929_340690Z')


def bundle(cells,ids,targets,origin,seed,separate):
    b=local.fit(cells,ids,targets,origin,'point',seed,3,False)
    b['base']=point.fit(cells,ids,origin,'point','V',1,seed,targets)
    groups={}
    for k in ids:groups.setdefault(local.key(cells[k]),[]).append(k)
    b['local']={g:bundle(cells,ks,targets,origin,seed,False) for g,ks in groups.items() if separate and len(ks)>=3}
    return b


def main():
    previous=json.loads((BASE/'config.json').read_text());plan=previous['plan'];seed=previous['seeds'][0]
    train=json.loads((BASE/'train_coverage.json').read_text())['used'];val=json.loads((BASE/'validation_coverage.json').read_text())['used']
    root=Path('experiments/voltage_condition_curve')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');root.mkdir(parents=True)
    files=[Path(__file__),Path('battery_smetimes/continuous_length.py'),Path('battery_smetimes/condition_length.py'),Path('scripts/train_continuous_length_pilot.py')]
    snap=root/'source_snapshot';snap.mkdir()
    for f in files:shutil.copy2(f,snap/f.name)
    save_json(root/'config.json',dict(plan=plan,origins=[50,100],seeds=[seed],source_sha256={str(f):sha(f) for f in files},
        input_block='V',diagnostic_basis='training-only fixed-window rate ablation',
        base_run=str(BASE),outer_loaded=False,independent_test=False,goal_complete=False))
    cells,_=load_nominal(plan);assert not set(cells)&set(plan['test']);targets={k:target(c,100) for k,c in cells.items()}
    rowsall=[];done=0;print('RUN='+str(root.resolve()),flush=True)
    for origin in (50,100):
        for name,separate in [('global_V',False),('local_V',True)]:
            b=bundle(cells,train,targets,origin,seed,separate);folder=root/str(origin)/str(seed)/name;folder.mkdir(parents=True)
            joblib.dump(b,folder/'model.joblib');rows=[];arrays={}
            for k in val:
                c,y=targets[k];p=local.predict(b,cells[k],c);child,matched=local.route(b,cells[k])
                poison=deepcopy(cells[k]);poison['raw'][poison['cycle']>origin]=np.inf;poison['soh']=np.inf;poison['reference_capacity_ah']=np.inf;poison['cycle_life']=np.inf
                np.testing.assert_allclose(p,local.predict(b,poison,c),rtol=0,atol=1e-12)
                rows.append(dict(cell_id=k,used_local=matched,route_train_count=len(child['train_ids']),**evaluate(c,y,p)))
                arrays.update({k+'__'+n:v for n,v in [('cycle',c),('truth',y),('prediction',p)]})
            rec=dict(candidate=name,origin=origin,seed=seed,used_train=train,validation=val,model_sha256=sha(folder/'model.joblib'))
            save_json(folder/'config.json',rec);save_json(folder/'metrics.json',rows);np.savez_compressed(folder/'validation_predictions.npz',**arrays)
            rowsall.extend(dict(**rec,**r) for r in rows);done+=1
            save_json(root/'all_rows.json',rowsall);save_json(root/'progress.json',dict(completed_fits=done,terminal=False))
            print(f'N={origin} {name} MAPE={np.mean([r["mape_percent"] for r in rows]):.3f} lateMAEpp={np.mean([r["late20_mae_pp"] for r in rows]):.3f}',flush=True)
    save_json(root/'progress.json',dict(completed_fits=done,terminal=True,success=True,goal_complete=False))


if __name__=='__main__':main()
