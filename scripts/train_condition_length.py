"""XJTU protocol-separated input-length development experiment."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from battery_smetimes import continuous_length as point
from battery_smetimes import integrated_length as rate
from battery_smetimes import condition_length as local
from scripts.train_continuous_length_pilot import load_nominal,target,evaluate
from scripts.train_curve_fidelity import AUDIT
from scripts.prepare_same_type_forecast import sha,save_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--five-seeds',action='store_true');args=ap.parse_args()
    plan=next(p for p in json.loads((AUDIT/'splits.json').read_text())['plans'] if p['name']=='XJTU_fold1')
    origins=[50,100];seeds=list(range(20261008,20261013)) if args.five_seeds else [20261008]
    root=Path('experiments/condition_length')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');root.mkdir(parents=True)
    files=[Path(__file__),Path('battery_smetimes/condition_length.py'),Path('battery_smetimes/continuous_length.py'),
           Path('battery_smetimes/integrated_length.py'),Path('scripts/train_continuous_length_pilot.py'),Path('battery_smetimes/curve_fidelity.py')]
    save_json(root/'config.json',dict(plan=plan,origins=origins,seeds=seeds,minimum_local_cells=3,
        candidates=['global_point','local_point','local_rate'],source_sha256={str(f):sha(f) for f in files},
        outer_loaded=False,independent_test=False,label_definition='nominal',goal_complete=False))
    snap=root/'source_snapshot';snap.mkdir()
    for f in files:shutil.copy2(f,snap/f.name)
    shutil.copy2('docs/soh_condition_length_20261008.md',root/'registration.md')
    cells,_=load_nominal(plan);assert not set(cells)&set(plan['test']);targets={k:target(c,100) for k,c in cells.items()}
    available={}
    for role in ('train','validation'):
        available[role]=[];excluded=[]
        for k in plan[role]:
            try:
                if len(targets[k][0])<3:raise ValueError('less than three future labels')
                for n in origins:point.features(cells[k],n)
                local.key(cells[k]);available[role].append(k)
            except ValueError as e:excluded.append(dict(cell_id=k,reason=str(e),not_bad_cell=True))
        save_json(root/(role+'_coverage.json'),dict(used=available[role],excluded=excluded))
    allrows=[];done=0;print('RUN='+str(root.resolve()),flush=True)
    for origin in origins:
        for seed in seeds:
            for candidate in ['global_point','local_point','local_rate']:
                b=local.fit(cells,available['train'],targets,origin,'rate' if candidate=='local_rate' else 'point',seed,3,candidate!='global_point')
                folder=root/str(origin)/str(seed)/candidate;folder.mkdir(parents=True);joblib.dump(b,folder/'model.joblib');b2=joblib.load(folder/'model.joblib')
                rows=[];arrays={}
                for k in available['validation']:
                    cell=cells[k];c,y=targets[k];p=local.predict(b,cell,c);child,matched=local.route(b,cell)
                    np.testing.assert_allclose(p,local.predict(b2,cell,c),rtol=0,atol=1e-12)
                    poison=deepcopy(cell);poison['raw'][poison['cycle']>origin]=np.inf
                    poison['soh']=np.full_like(poison['soh'],np.inf);poison['reference_capacity_ah']=np.inf;poison['cycle_life']=np.inf
                    np.testing.assert_allclose(p,local.predict(b,poison,c),rtol=0,atol=1e-12)
                    try:local.predict(b,dict(cell,physical_family='wrong'),c)
                    except ValueError:pass
                    else:raise AssertionError('family gate')
                    if candidate=='local_rate':assert max(np.diff(local.predict(b,cell,np.linspace(c[0],c[-1],400))))<=1e-10
                    rows.append(dict(cell_id=k,used_local=matched,route_train_count=len(child['train_ids']),**evaluate(c,y,p)))
                    arrays.update({k+'__'+n:v for n,v in [('cycle',c),('truth',y),('prediction',p)]})
                rec=dict(candidate=candidate,origin=origin,seed=seed,used_train=available['train'],validation=available['validation'],model_sha256=sha(folder/'model.joblib'))
                save_json(folder/'config.json',rec);save_json(folder/'metrics.json',rows);np.savez_compressed(folder/'validation_predictions.npz',**arrays)
                allrows.extend(dict(**rec,**r) for r in rows);done+=1;save_json(root/'all_rows.json',allrows)
                save_json(root/'progress.json',dict(completed_fits=done,terminal=False))
                print(f'FIT N={origin} seed={seed} {candidate} MAPE={np.mean([r["mape_percent"] for r in rows]):.3f} screen={np.mean([r["screen_pass"] for r in rows]):.3f}',flush=True)
    save_json(root/'progress.json',dict(completed_fits=done,terminal=True,success=True,goal_complete=False));print('COMPLETE '+str(root.resolve()),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
