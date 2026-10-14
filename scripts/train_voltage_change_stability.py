"""Five-seed paired feature confirmation, retaining every development failure."""
from copy import deepcopy
from datetime import datetime,timezone
import json,logging,shutil,time,traceback
from pathlib import Path
import joblib
import numpy as np
from battery_smetimes import voltage_change as rich,continuous_length as point
from scripts.train_continuous_length_pilot import load_nominal,target,evaluate
from scripts.prepare_same_type_forecast import sha,save_json


def main():
    pilot=Path('experiments/voltage_change_probe/20261008T065227_466334Z')
    assert json.loads((pilot/'verification.json').read_text())['passed']
    old=json.loads((pilot/'config.json').read_text());plans=[p for p in old['plans'] if p['dataset']!='Zhu2022_NCA']
    coverage=json.loads((pilot/'coverage.json').read_text())
    root=Path('experiments/voltage_change_stability')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');root.mkdir(parents=True)
    sources=[Path(__file__),Path('battery_smetimes/voltage_change.py'),Path('battery_smetimes/continuous_length.py'),Path('scripts/train_continuous_length_pilot.py')]
    (root/'source_snapshot').mkdir()
    for p in sources:shutil.copy2(p,root/'source_snapshot'/p.name)
    seeds=list(range(20261008,20261013))
    save_json(root/'config.json',dict(plans=plans,origin=100,seeds=seeds,hust_index=old['hust_index'],
        hust_index_sha256=old['hust_index_sha256'],source_sha256={str(p):sha(p) for p in sources},
        source_probe=str(pilot),coverage=coverage,budget=dict(trees=80,minimum_leaf=1,max_features=.8),
        outer_loaded=False,independent_test=False,goal_complete=False,
        limitation='One development split per cohort; five seeds do not replace multiple cell splits.'))
    log=logging.getLogger('voltage_change_stability');log.setLevel(logging.INFO)
    fh=logging.FileHandler(root/'training.log',encoding='utf-8');fh.setFormatter(logging.Formatter('%(asctime)s %(message)s'));log.addHandler(fh)
    log.info('START %s',root);print('RUN='+str(root.resolve()),flush=True)
    done=0;allrows=[];start=time.perf_counter()
    try:
        for plan in plans:
            if plan['dataset']=='HUST':
                path=Path(old['hust_index']);assert sha(path)==old['hust_index_sha256'];index=json.loads(path.read_text())
                records={r['cell_id']:r for r in index['cells']};cells={}
                for cid in plan['train']+plan['validation']:
                    r=records[cid];path=Path(r['path']);assert sha(path)==r['sha256']
                    with np.load(path,allow_pickle=False) as a:cells[cid]={n:a[n].copy() for n in a.files}
                    cells[cid]['physical_family']=r['physical_family']
            else:cells,_=load_nominal(plan)
            assert not set(cells)&set(plan['test'])
            targets={k:target(v,100) for k,v in cells.items()}
            roles={role:next(r['used'] for r in coverage if r['fold']==plan['name'] and r['role']==role) for role in ('train','validation')}
            for seed in seeds:
                for name,module in [('point',point),('voltage_change',rich)]:
                    t=time.perf_counter()
                    b=point.fit(cells,roles['train'],100,'point','VIT',1,seed,targets) if name=='point' else rich.fit(cells,roles['train'],targets,100,seed)
                    folder=root/plan['name']/str(seed)/name;folder.mkdir(parents=True);path=folder/'model.joblib';joblib.dump(b,path);replay=joblib.load(path)
                    rows=[];arrays={};infer=[]
                    for cid in roles['validation']:
                        c,y=targets[cid];ti=time.perf_counter();p=module.predict(b,cells[cid],c);infer.append(time.perf_counter()-ti)
                        np.testing.assert_allclose(p,module.predict(replay,cells[cid],c),rtol=0,atol=1e-12)
                        poison=deepcopy(cells[cid])
                        if 'early_cycle' in poison:
                            poison['early_raw'][poison['early_cycle']>100]=np.inf;poison['truth']=np.inf;poison['cycle']=np.array([np.inf])
                        else:
                            poison['raw'][poison['cycle']>100]=np.inf;poison['soh']=np.inf;poison['reference_capacity_ah']=np.inf
                        poison['cycle_life']=np.inf
                        np.testing.assert_allclose(p,module.predict(b,poison,c),rtol=0,atol=1e-12)
                        row=dict(dataset=plan['dataset'],fold=plan['name'],candidate=name,origin=100,seed=seed,cell_id=cid,**evaluate(c,y,p));rows.append(row);allrows.append(row)
                        arrays.update({cid+'__'+k:v for k,v in [('cycle',c),('truth',y),('prediction',p)]})
                    save_json(folder/'config.json',dict(train=roles['train'],validation=roles['validation'],seed=seed,model_sha256=sha(path),
                        seconds=time.perf_counter()-t,serialized_bytes=path.stat().st_size,tree_nodes=sum(tree.tree_.node_count for tree in b['model'].estimators_),
                        inference_seconds_per_full_curve_mean=float(np.mean(infer)),inference_timing_scope='CPU full curve including features, not MCU benchmark'))
                    save_json(folder/'metrics.json',rows);np.savez_compressed(folder/'predictions.npz',**arrays)
                    done+=1;save_json(root/'all_rows.json',allrows);save_json(root/'progress.json',dict(completed_fits=done,planned_fits=30,terminal=False,seconds=time.perf_counter()-start))
                    message=f'{plan["dataset"]} seed={seed} {name}: MAPE={np.mean([r["mape_percent"] for r in rows]):.3f}, lateMAEpp={np.mean([r["late20_mae_pp"] for r in rows]):.3f}'
                    log.info(message);print(message,flush=True)
        log.info('COMPLETE fits=%d',done)
        save_json(root/'progress.json',dict(completed_fits=done,planned_fits=30,terminal=True,success=True,seconds=time.perf_counter()-start,goal_complete=False))
    except BaseException:
        log.exception('FAILED');save_json(root/'failure.json',dict(traceback=traceback.format_exc()));save_json(root/'progress.json',dict(completed_fits=done,terminal=True,success=False));raise
    finally:fh.close();log.removeHandler(fh)


if __name__=='__main__':main()
