"""Paired four-cohort prefix-feature ablation with unchanged model budget."""
from copy import deepcopy
from datetime import datetime,timezone
import json,shutil,time,traceback
from pathlib import Path
import joblib
import numpy as np
from battery_smetimes import voltage_change as rich,continuous_length as point
from scripts.train_continuous_length_pilot import load_nominal,target,evaluate
from scripts.prepare_same_type_forecast import sha,save_json


def main():
    plans=json.loads(Path('experiments/pace_clock_probe/20261008T063432_818706Z/config.json').read_text())['plans']
    hustpath=Path('experiments/hust_length_smoke/20261008T015438_626064Z/config.json');hcfg=json.loads(hustpath.read_text())
    plans.append(dict(name='HUST_fold1',dataset='HUST',train=hcfg['train'],validation=hcfg['validation'],test=hcfg['test']))
    root=Path('experiments/voltage_change_probe')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');root.mkdir(parents=True)
    files=[Path(__file__),Path('battery_smetimes/voltage_change.py'),Path('battery_smetimes/continuous_length.py'),Path('scripts/train_continuous_length_pilot.py')]
    (root/'source_snapshot').mkdir()
    for p in files:shutil.copy2(p,root/'source_snapshot'/p.name)
    save_json(root/'config.json',dict(plans=plans,origin=100,seeds=[20261008],hust_source_config=str(hustpath),hust_index=hcfg['source_index'],
        source_sha256={str(p):sha(p) for p in files},hust_index_sha256=hcfg['index_sha256'],
        budget=dict(trees=80,minimum_leaf=1,max_features=.8),outer_loaded=False,independent_test=False,
        feature_contract='Partial fixed-time voltage grid differences only; NOT full Q(V) and no capacity input.',goal_complete=False))
    done=0;allrows=[];coverage=[];print('RUN='+str(root.resolve()),flush=True)
    try:
        for plan in plans:
            if plan['dataset']=='HUST':
                indexpath=Path(hcfg['source_index']);assert sha(indexpath)==hcfg['index_sha256'];index=json.loads(indexpath.read_text())
                records={r['cell_id']:r for r in index['cells']};cells={}
                for cid in plan['train']+plan['validation']:
                    r=records[cid];path=Path(r['path']);assert sha(path)==r['sha256']
                    with np.load(path,allow_pickle=False) as a:cells[cid]={n:a[n].copy() for n in a.files}
                    cells[cid]['physical_family']=r['physical_family']
            else:cells,_=load_nominal(plan)
            assert not set(cells)&set(plan['test'])
            targets={k:target(v,100) for k,v in cells.items()};roles={}
            for role in ('train','validation'):
                used=[];excluded=[]
                for cid in plan[role]:
                    try:
                        rich.features(cells[cid],100)
                        if len(targets[cid][0])<2:raise ValueError('insufficient future labels')
                        used.append(cid)
                    except ValueError as e:excluded.append(dict(cell_id=cid,reason=str(e),not_bad_cell=True))
                roles[role]=used;coverage.append(dict(fold=plan['name'],role=role,used=used,excluded=excluded))
            save_json(root/'coverage.json',coverage)
            for name,module in [('point',point),('voltage_change',rich)]:
                t=time.perf_counter()
                b=point.fit(cells,roles['train'],100,'point','VIT',1,20261008,targets) if name=='point' else rich.fit(cells,roles['train'],targets,100,20261008)
                folder=root/plan['name']/name;folder.mkdir(parents=True);path=folder/'model.joblib';joblib.dump(b,path);replay=joblib.load(path);rows=[];arrays={}
                for cid in roles['validation']:
                    c,y=targets[cid];p=module.predict(b,cells[cid],c)
                    np.testing.assert_allclose(p,module.predict(replay,cells[cid],c),rtol=0,atol=1e-12)
                    poison=deepcopy(cells[cid])
                    if 'early_cycle' in poison:
                        poison['early_raw'][poison['early_cycle']>100]=np.inf;poison['truth']=np.inf;poison['cycle']=np.array([np.inf])
                    else:
                        poison['raw'][poison['cycle']>100]=np.inf;poison['soh']=np.inf;poison['reference_capacity_ah']=np.inf
                    poison['cycle_life']=np.inf
                    np.testing.assert_allclose(p,module.predict(b,poison,c),rtol=0,atol=1e-12)
                    row=dict(dataset=plan['dataset'],fold=plan['name'],candidate=name,origin=100,seed=20261008,cell_id=cid,**evaluate(c,y,p));rows.append(row);allrows.append(row)
                    arrays.update({cid+'__'+k:v for k,v in [('cycle',c),('truth',y),('prediction',p)]})
                save_json(folder/'config.json',dict(train=roles['train'],validation=roles['validation'],model_sha256=sha(path),seconds=time.perf_counter()-t))
                save_json(folder/'metrics.json',rows);np.savez_compressed(folder/'predictions.npz',**arrays)
                done+=1;save_json(root/'all_rows.json',allrows);save_json(root/'progress.json',dict(completed_fits=done,terminal=False))
                print(f'{plan["dataset"]} {name}: MAPE={np.mean([r["mape_percent"] for r in rows]):.3f}, lateMAEpp={np.mean([r["late20_mae_pp"] for r in rows]):.3f}',flush=True)
        save_json(root/'progress.json',dict(completed_fits=done,terminal=True,success=True,goal_complete=False))
    except BaseException:
        save_json(root/'failure.json',dict(traceback=traceback.format_exc()));save_json(root/'progress.json',dict(completed_fits=done,terminal=True,success=False));raise


if __name__=='__main__':main()
