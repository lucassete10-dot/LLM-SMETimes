"""Five-seed paired confirmation of the NCA short-history clock candidate."""
from copy import deepcopy
from datetime import datetime,timezone
import json,shutil,time,traceback
from pathlib import Path
import joblib
import numpy as np
from battery_smetimes import pace_clock,continuous_length as point
from scripts.train_continuous_length_pilot import load_nominal,target,evaluate
from scripts.prepare_same_type_forecast import sha,save_json


def main():
    pilot=Path('experiments/pace_clock_probe/20261008T063432_818706Z')
    assert json.loads((pilot/'verification.json').read_text())['passed']
    plan=next(p for p in json.loads((pilot/'config.json').read_text())['plans'] if p['dataset']=='Zhu2022_NCA')
    coverage=json.loads((pilot/'coverage.json').read_text())
    train=next(c['used'] for c in coverage if c['fold']==plan['name'] and c['role']=='train')
    val=next(c['used'] for c in coverage if c['fold']==plan['name'] and c['role']=='validation')
    root=Path('experiments/nca_clock_stability')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');root.mkdir(parents=True)
    files=[Path(__file__),Path('battery_smetimes/pace_clock.py'),Path('battery_smetimes/continuous_length.py'),Path('scripts/train_continuous_length_pilot.py')]
    (root/'source_snapshot').mkdir()
    for p in files:shutil.copy2(p,root/'source_snapshot'/p.name)
    seeds=list(range(20261008,20261013))
    save_json(root/'config.json',dict(plans=[plan],origins=[75,100],seeds=seeds,train=train,validation=val,
        pilot=str(pilot),common_cutoff=100,pace_window=[101,150],factor_bounds=[.25,4],rate_floor=.0005,
        source_sha256={str(p):sha(p) for p in files},outer_loaded=False,independent_test=False,
        limitation='One development split, five seeds; not full-fold or optimal-length evidence.',goal_complete=False))
    cells,_=load_nominal(plan);assert not set(cells)&set(plan['test']);targets={k:target(v,100) for k,v in cells.items()}
    start=time.perf_counter();rows=[];done=0;print('RUN='+str(root.resolve()),flush=True)
    try:
        for origin in (75,100):
            for seed in seeds:
                for name in ('point','pace_clock'):
                    b=point.fit(cells,train,origin,'point','VIT',1,seed,targets) if name=='point' else pace_clock.fit(cells,train,targets,origin,seed)
                    fn=point.predict if name=='point' else pace_clock.predict
                    folder=root/plan['name']/str(origin)/str(seed)/name;folder.mkdir(parents=True)
                    path=folder/'model.joblib';joblib.dump(b,path);replay=joblib.load(path);batch=[];arrays={}
                    for cid in val:
                        c,y=targets[cid];p=fn(b,cells[cid],c)
                        np.testing.assert_allclose(p,fn(replay,cells[cid],c),atol=1e-12,rtol=0)
                        poison=deepcopy(cells[cid]);poison['early_raw'][poison['early_cycle']>origin]=np.inf
                        poison['truth']=np.inf;poison['cycle']=np.array([np.inf]);poison['cycle_life']=np.inf
                        np.testing.assert_allclose(p,fn(b,poison,c),atol=1e-12,rtol=0)
                        row=dict(dataset=plan['dataset'],fold=plan['name'],origin=origin,candidate=name,seed=seed,cell_id=cid,**evaluate(c,y,p));batch.append(row);rows.append(row)
                        arrays.update({cid+'__'+k:v for k,v in [('cycle',c),('truth',y),('prediction',p)]})
                    save_json(folder/'config.json',dict(train=train,validation=val,model_sha256=sha(path),seed=seed))
                    save_json(folder/'metrics.json',batch);np.savez_compressed(folder/'predictions.npz',**arrays)
                    done+=1;save_json(root/'all_rows.json',rows);save_json(root/'progress.json',dict(completed_fits=done,terminal=False,seconds=time.perf_counter()-start))
                    print(f'N={origin} seed={seed} {name} MAPE={np.mean([r["mape_percent"] for r in batch]):.3f}',flush=True)
        save_json(root/'progress.json',dict(completed_fits=done,terminal=True,success=True,seconds=time.perf_counter()-start,goal_complete=False))
    except BaseException:
        save_json(root/'failure.json',dict(traceback=traceback.format_exc()));save_json(root/'progress.json',dict(completed_fits=done,terminal=True,success=False));raise


if __name__=='__main__':main()
