"""HUST prefix-length search: operational and common-after200 evaluation."""
from copy import deepcopy
import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import shutil
import time
import traceback
import joblib
import numpy as np
from sklearn.model_selection import KFold
from battery_smetimes import continuous_length as point, voltage_change as rich, voltage_interval_features as physics
from scripts.train_continuous_length_pilot import target, evaluate
from scripts.prepare_same_type_forecast import sha, save_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cv-fold',type=int,choices=range(1,6))
    parser.add_argument('--seed',type=int,default=20261008)
    parser.add_argument('--physics-index',type=Path)
    parser.add_argument('--origins',type=int,nargs='+',default=[10,20,30,50,75,100,150,200])
    args=parser.parse_args()
    original=json.loads(Path('experiments/hust_length_smoke/20261008T015438_626064Z/config.json').read_text())
    development=sorted(original['train']+original['validation'])
    if args.cv_fold:
        train_idx,val_idx=list(KFold(5,shuffle=True,random_state=20261008).split(development))[args.cv_fold-1]
        original['train']=[development[i] for i in train_idx]
        original['validation']=[development[i] for i in val_idx]
    indexpath=Path(original['source_index']); assert sha(indexpath)==original['index_sha256']
    index=json.loads(indexpath.read_text()); records={r['cell_id']:r for r in index['cells']}
    ids=original['train']+original['validation']; cells={}
    assert not set(ids)&set(original['test'])
    for key in ids:
        rec=records[key]; path=Path(rec['path']); assert sha(path)==rec['sha256']
        with np.load(path,allow_pickle=False) as a: cells[key]={k:a[k].copy() for k in a.files}
        cells[key]['physical_family']=rec['physical_family']
    origins=args.origins
    if args.physics_index:
        intervals=json.loads(args.physics_index.read_text())
        assert intervals['feature_source_sha256']==sha(Path('battery_smetimes/voltage_interval_features.py'))
        assert intervals['outer_loaded'] is False and intervals['full_capacity_or_label_inputs'] is False
        interval_records={r['cell_id']:r for r in intervals['cells']}
        for key in ids:
            rec=interval_records[key]; assert sha(Path(rec['path']))==rec['sha256']
            with np.load(rec['path'],allow_pickle=False) as a:
                cells[key].update({k:a[k].copy() for k in a.files})
    candidate_module=physics if args.physics_index else rich
    candidate_name='voltage_intervals' if args.physics_index else 'voltage_change'
    root=Path('experiments/hust_voltage_intervals' if args.physics_index else 'experiments/hust_length_cv' if args.cv_fold else 'experiments/hust_length_pair')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    root.mkdir(parents=True); (root/'source_snapshot').mkdir()
    files=[Path(__file__),Path('battery_smetimes/continuous_length.py'),Path('battery_smetimes/voltage_change.py'),Path('scripts/train_continuous_length_pilot.py')]
    if args.physics_index: files.append(Path('battery_smetimes/voltage_interval_features.py'))
    for p in files: shutil.copy2(p,root/'source_snapshot'/p.name)
    cfg=dict(dataset='HUST',train=original['train'],validation=original['validation'],test=original['test'],
        origins=origins,seeds=[args.seed],candidates=['point',candidate_name],source_index=str(indexpath),
        physics_index=str(args.physics_index) if args.physics_index else None,
        physics_index_sha256=sha(args.physics_index) if args.physics_index else None,
        cv_fold=args.cv_fold,cv_split_seed=20261008,development_cells=development,
        index_sha256=sha(indexpath),source_sha256={str(p):sha(p) for p in files},common_cutoff=200,
        training_targets_after='origin',outer_loaded=False,independent_test=False,goal_complete=False,
        length_rule=dict(numeric_mape_goal=6,late_mae_pp=2,stage_pp=3,max_reversal_fraction=.1,
            near_best_mape_pp=.25,near_best_late_mae_pp=.5,near_best_stage_pp=.5,
            scope='Exploratory one-split recommendation only; both operational and common horizons must qualify. No final knee/platform acceptance.'),
        limitation='One split and one seed; do not attribute changes to N without common-future and coverage checks.')
    save_json(root/'config.json',cfg)
    coverage=[]
    for role in ('train','validation'):
        used=[]; excluded=[]
        for key in cfg[role]:
            try:
                for origin in origins: candidate_module.features(cells[key],origin)
                if len(target(cells[key],200)[0])<2: raise ValueError('insufficient common labels')
                used.append(key)
            except ValueError as e: excluded.append(dict(cell_id=key,reason=str(e),not_bad_cell=True))
        coverage.append(dict(role=role,used=used,excluded=excluded))
    save_json(root/'coverage.json',coverage)
    roles={r['role']:r['used'] for r in coverage}
    logger=logging.getLogger('hust_length_pair'); logger.setLevel(logging.INFO)
    handler=logging.FileHandler(root/'training.log',encoding='utf-8'); logger.addHandler(handler)
    done=0; rows=[]; start=time.perf_counter(); print('RUN='+str(root.resolve()),flush=True)
    try:
        for origin in origins:
            targets={k:target(cells[k],origin) for k in ids}
            for name,module in [('point',point),(candidate_name,candidate_module)]:
                begin=time.perf_counter()
                b=(point.fit(cells,roles['train'],origin,'point','VIT',1,args.seed,targets) if name=='point' else
                   candidate_module.fit(cells,roles['train'],targets,origin,args.seed))
                folder=root/str(origin)/name; folder.mkdir(parents=True)
                path=folder/'model.joblib'; joblib.dump(b,path); replay=joblib.load(path)
                arrays={}; batch=[]
                for cid in roles['validation']:
                    c,y=targets[cid]; p=module.predict(b,cells[cid],c)
                    np.testing.assert_allclose(p,module.predict(replay,cells[cid],c),rtol=0,atol=1e-12)
                    poison=deepcopy(cells[cid]); poison['early_raw'][poison['early_cycle']>origin]=np.inf
                    poison['truth']=np.inf; poison['cycle']=np.array([np.inf]); poison['cycle_life']=np.inf
                    if args.physics_index:
                        poison['physics_raw'][poison['physics_cycle']>origin]=np.inf
                    np.testing.assert_allclose(p,module.predict(b,poison,c),rtol=0,atol=1e-12)
                    try: module.predict(b,dict(cells[cid],physical_family='wrong'),c)
                    except ValueError: pass
                    else: raise AssertionError('family gate')
                    arrays.update({cid+'__'+k:v for k,v in [('cycle',c),('truth',y),('prediction',p)]})
                    for horizon,mask in [('operational',np.ones(len(c),bool)),('common',c>200)]:
                        rec=dict(cell_id=cid,origin=origin,candidate=name,seed=args.seed,horizon=horizon,**evaluate(c[mask],y[mask],p[mask]))
                        rows.append(rec); batch.append(rec)
                    for distance in (50,100,200):
                        mask=c<=origin+distance
                        if mask.sum()>=2:
                            rec=dict(cell_id=cid,origin=origin,candidate=name,seed=args.seed,horizon=f'fixed_{distance}',
                                     full_horizon_covered=bool(c[-1]>=origin+distance),**evaluate(c[mask],y[mask],p[mask]))
                            rows.append(rec); batch.append(rec)
                np.savez_compressed(folder/'predictions.npz',**arrays); save_json(folder/'metrics.json',batch)
                save_json(folder/'config.json',dict(train=roles['train'],validation=roles['validation'],model_sha256=sha(path),seconds=time.perf_counter()-begin))
                done+=1; save_json(root/'all_rows.json',rows)
                save_json(root/'progress.json',dict(completed_fits=done,planned_fits=2*len(origins),terminal=False))
                selected=[r for r in batch if r['horizon']=='common']
                message=f'N={origin} {name} common MAPE={np.mean([r["mape_percent"] for r in selected]):.3f}, lateMAEpp={np.mean([r["late20_mae_pp"] for r in selected]):.3f}'
                logger.info(message); print(message,flush=True)
        save_json(root/'progress.json',dict(completed_fits=done,planned_fits=2*len(origins),terminal=True,success=True,seconds=time.perf_counter()-start,goal_complete=False))
    except BaseException:
        logger.exception('FAILED'); save_json(root/'failure.json',dict(traceback=traceback.format_exc()))
        save_json(root/'progress.json',dict(completed_fits=done,terminal=True,success=False)); raise
    finally:
        handler.close(); logger.removeHandler(handler)


if __name__=='__main__': main()
