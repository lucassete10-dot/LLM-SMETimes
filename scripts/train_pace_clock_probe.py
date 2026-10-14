"""Paired phase-clock experiment across three available physical cohorts."""
from copy import deepcopy
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil,traceback
import joblib
import numpy as np
from battery_smetimes import pace_clock,continuous_length as point
from scripts.train_continuous_length_pilot import load_nominal,target,evaluate
from scripts.prepare_same_type_forecast import sha,save_json


def main():
    old=Path('experiments/continuous_length/20261008T014005_004506Z/config.json')
    plans=json.loads(old.read_text())['plans']
    root=Path('experiments/pace_clock_probe')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');root.mkdir(parents=True)
    files=[Path(__file__),Path('battery_smetimes/pace_clock.py'),Path('battery_smetimes/continuous_length.py'),Path('scripts/train_continuous_length_pilot.py')]
    (root/'source_snapshot').mkdir()
    for p in files:shutil.copy2(p,root/'source_snapshot'/p.name)
    save_json(root/'config.json',dict(plans=plans,origins=[75,100],seeds=[20261008],pace_window=[101,150],
        reference_rate='train group median',factor_bounds=[.25,4],rate_floor=.0005,
        source_sha256={str(p):sha(p) for p in files},outer_loaded=False,independent_test=False,
        hypothesis='Normalize training time by individual train-supervised early-future pace, infer query clock from prefix signals only.',
        limitation='Near-term pace does not necessarily identify late knee; bounded clock and point head may still create platforms.',goal_complete=False))
    print('RUN='+str(root.resolve()),flush=True);rows=[];coverage=[];done=0
    try:
        for plan in plans:
            cells,_=load_nominal(plan);assert not set(cells)&set(plan['test'])
            assert not set(plan['train'])&set(plan['validation'])
            targets={k:target(v,100) for k,v in cells.items()};roles={}
            for role in ('train','validation'):
                used=[];excluded=[]
                for cid in plan[role]:
                    try:
                        if len(targets[cid][0])<2:raise ValueError('insufficient future labels')
                        for origin in (75,100):point.features(cells[cid],origin)
                        used.append(cid)
                    except ValueError as e:excluded.append(dict(cell_id=cid,reason=str(e),not_bad_cell=True))
                roles[role]=used;coverage.append(dict(fold=plan['name'],role=role,used=used,excluded=excluded))
            save_json(root/'coverage.json',coverage)
            for origin in (75,100):
                for name in ('point','pace_clock'):
                    b=point.fit(cells,roles['train'],origin,'point','VIT',1,20261008,targets) if name=='point' else pace_clock.fit(cells,roles['train'],targets,origin,20261008)
                    fn=point.predict if name=='point' else pace_clock.predict
                    folder=root/plan['name']/str(origin)/name;folder.mkdir(parents=True);joblib.dump(b,folder/'model.joblib')
                    replay=joblib.load(folder/'model.joblib');arrays={};batch=[]
                    for cid in roles['validation']:
                        c,y=targets[cid];p=fn(b,cells[cid],c)
                        np.testing.assert_allclose(p,fn(replay,cells[cid],c),rtol=0,atol=1e-12)
                        poisoned=deepcopy(cells[cid])
                        if 'early_cycle' in poisoned:
                            poisoned['early_raw'][poisoned['early_cycle']>origin]=np.inf;poisoned['truth']=np.inf;poisoned['cycle']=np.array([np.inf])
                        else:
                            poisoned['raw'][poisoned['cycle']>origin]=np.inf;poisoned['soh']=np.inf;poisoned['reference_capacity_ah']=np.inf
                        poisoned['cycle_life']=np.inf
                        np.testing.assert_allclose(p,fn(b,poisoned,c),rtol=0,atol=1e-12)
                        row=dict(dataset=plan['dataset'],fold=plan['name'],origin=origin,candidate=name,seed=20261008,cell_id=cid,**evaluate(c,y,p));batch.append(row);rows.append(row)
                        arrays.update({cid+'__'+k:v for k,v in [('cycle',c),('truth',y),('prediction',p)]})
                    save_json(folder/'metrics.json',batch);np.savez_compressed(folder/'predictions.npz',**arrays)
                    save_json(folder/'config.json',dict(train=roles['train'],validation=roles['validation'],model_sha256=sha(folder/'model.joblib')))
                    done+=1;save_json(root/'all_rows.json',rows);save_json(root/'progress.json',dict(completed_fits=done,terminal=False))
                    print(f'{plan["dataset"]} N={origin} {name} MAPE={np.mean([r["mape_percent"] for r in batch]):.3f} lateMAEpp={np.mean([r["late20_mae_pp"] for r in batch]):.3f}',flush=True)
        save_json(root/'progress.json',dict(completed_fits=done,terminal=True,success=True,goal_complete=False))
    except BaseException:
        save_json(root/'failure.json',dict(traceback=traceback.format_exc()));save_json(root/'progress.json',dict(completed_fits=done,terminal=True,success=False));raise


if __name__=='__main__':main()
