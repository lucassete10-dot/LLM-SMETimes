"""Fixed-reserve MIT/XJTU whole-cell CV with real operational target coverage."""
import argparse,json,logging,shutil,time,traceback
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import joblib,numpy as np
from sklearn.model_selection import KFold
from battery_smetimes import continuous_length as model
from scripts.train_continuous_length_pilot import load_nominal,target,evaluate
from scripts.train_curve_fidelity import AUDIT
from scripts.prepare_same_type_forecast import sha,save_json

ORIGINS=[10,20,30,50,75,100]


def load(plan,signal_index=None):
    cells=load_nominal(plan)[0]
    if signal_index:
        records={r['cell_id']:r for r in json.loads(Path(signal_index).read_text())['cells']}
        for cid,cell in cells.items():
            rec=records[cid];assert sha(Path(rec['path']))==rec['sha256']
            with np.load(rec['path'],allow_pickle=False) as data:
                cell['cycle']=data['signal_cycle'].copy();cell['raw']=data['signal_raw'].copy()
    return cells


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',choices=['MIT','XJTU'],required=True)
    parser.add_argument('--seed',type=int,default=20261008)
    parser.add_argument('--signal-index',type=Path)
    parser.add_argument('--origins',type=int,nargs='+')
    parser.add_argument('--cv-fold',type=int,choices=range(1,6))
    parser.add_argument('--sparse',action='store_true')
    parser.add_argument('--arms',choices=['available','matched'],nargs='+',default=['available','matched'])
    args=parser.parse_args()
    from battery_smetimes import sparse_history_point
    backend=sparse_history_point if args.sparse else model
    origins=args.origins or (ORIGINS+[150,200] if args.signal_index else ORIGINS)
    if args.signal_index:
        signal_config=json.loads(args.signal_index.read_text())
        assert args.dataset=='XJTU' and signal_config['dataset']==args.dataset
        assert max(origins)<=signal_config['max_source_cycle'] and signal_config['outer_loaded'] is False
    else:assert max(origins)<=100
    splitpath=AUDIT/'splits.json';registry_path=AUDIT/'cell_registry.json'
    original=next(p for p in json.loads(splitpath.read_text())['plans'] if p['name']==args.dataset+'_fold1')
    dev=sorted(original['train']+original['validation']);reserve=original['test']
    registry=json.loads(registry_path.read_text())['cells']
    folds=[]
    for number,(train_idx,val_idx) in enumerate(KFold(5,shuffle=True,random_state=20261008).split(dev),1):
        folds.append(dict(name=args.dataset+'_fixedCV'+str(number),dataset=args.dataset,
            train=[dev[i] for i in train_idx],validation=[dev[i] for i in val_idx],test=reserve))
    if args.cv_fold:folds=[folds[args.cv_fold-1]]
    root=Path('experiments/core_length_cv')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');root.mkdir(parents=True)
    (root/'source_snapshot').mkdir()
    sources=[Path(__file__),Path('battery_smetimes/continuous_length.py'),Path('scripts/train_continuous_length_pilot.py'),
        Path('scripts/train_curve_fidelity.py'),Path('scripts/train_prefix_length_screen.py'),Path('prepare_mit.py'),Path('prepare_xjtu.py')]
    for path in sources:shutil.copy2(path,root/'source_snapshot'/path.name)
    if args.sparse:
        path=Path('battery_smetimes/sparse_history_point.py');sources.append(path);shutil.copy2(path,root/'source_snapshot'/path.name)
    auditcfg=json.loads((AUDIT/'config.json').read_text());integrity=Path(auditcfg['config']['integrity_audit'])
    cfg=dict(dataset=args.dataset,plans=folds,origins=origins,seeds=[args.seed],arms=args.arms,
        signal_index=str(args.signal_index) if args.signal_index else None,
        signal_index_sha256=sha(args.signal_index) if args.signal_index else None,
        feature_backend='sparse_history' if args.sparse else 'original',cv_fold=args.cv_fold,
        development=dev,reserve=reserve,split_seed=20261008,registry=str(registry_path),registry_sha256=sha(registry_path),
        splits=str(splitpath),splits_sha256=sha(splitpath),integrity=str(integrity),integrity_sha256=sha(integrity),
        data_sha256={cid:registry[cid]['prepared_sha256'] for cid in dev},source_sha256={str(p):sha(p) for p in sources},
        common_cutoff=max(100,max(origins)),training_targets_after='origin',outer_loaded=False,independent_test=False,goal_complete=False,
        limitation='One seed exploration. Reserve has prior historical exposure; not an independent confirmation set. Available arm varies input-data coverage; matched arm controls cell coverage.',
        length_rule=dict(mape=6,late_mae_pp=2,stage_pp=3,reversal_fraction=.1,near_best_mape_pp=.25,
            near_best_late_pp=.5,near_best_stage_pp=.5,scope='Exploratory bands only, not verified knee/platform acceptance.'))
    save_json(root/'config.json',cfg)
    log=logging.getLogger('core_length');log.setLevel(logging.INFO);handler=logging.FileHandler(root/'training.log',encoding='utf-8');log.addHandler(handler)
    rows=[];coverage=[];done=0;begin=time.perf_counter();print('RUN='+str(root.resolve()),flush=True)
    try:
        for plan in folds:
            cells=load(plan,args.signal_index);assert set(cells)==set(plan['train']+plan['validation']) and not set(cells)&set(reserve)
            assert len({c['physical_family'] for c in cells.values()})==1
            eligibility={};reasons={}
            for n in origins:
                good=set();bad=[]
                for cid,cell in cells.items():
                    try:
                        backend.features(cell,n)
                        if len(target(cell,n)[0])<2:raise ValueError('insufficient future labels')
                        good.add(cid)
                    except ValueError as exc:bad.append(dict(cell_id=cid,reason=str(exc),not_bad_cell=True))
                eligibility[n]=good;reasons[n]=bad
            common=set.intersection(*eligibility.values())
            for n in origins:
                targets={cid:target(cell,n) for cid,cell in cells.items()}
                for arm in cfg['arms']:
                    allowed=eligibility[n] if arm=='available' else common
                    train=[cid for cid in plan['train'] if cid in allowed];val=[cid for cid in plan['validation'] if cid in allowed]
                    coverage.append(dict(fold=plan['name'],origin=n,arm=arm,train=train,validation=val,
                        exclusions=reasons[n],matched_cells=sorted(common),original_train=len(plan['train']),original_validation=len(plan['validation'])))
                    save_json(root/'coverage.json',coverage)
                    if len(train)<3 or not val:continue
                    folder=root/plan['name']/str(n)/arm;folder.mkdir(parents=True)
                    b=backend.fit(cells,train,n,'point','VIT',1,args.seed,targets);path=folder/'model.joblib';joblib.dump(b,path)
                    replay=joblib.load(path);arrays={};batch=[]
                    for cid in val:
                        c,y=targets[cid];p=backend.predict(b,cells[cid],c)
                        np.testing.assert_allclose(p,backend.predict(replay,cells[cid],c),rtol=0,atol=1e-12)
                        poison=deepcopy(cells[cid]);poison['raw'][poison['cycle']>n]=np.inf
                        poison['soh']=np.inf;poison['label_cycle']=np.array([np.inf]);poison['reference_capacity_ah']=np.inf
                        np.testing.assert_allclose(p,backend.predict(b,poison,c),rtol=0,atol=1e-12)
                        try:backend.predict(b,dict(cells[cid],physical_family='wrong'),c)
                        except ValueError:pass
                        else:raise AssertionError('family gate')
                        arrays.update({cid+'__'+k:v for k,v in [('cycle',c),('truth',y),('prediction',p)]})
                        horizons=[('operational',np.ones(len(c),bool))]
                        if np.sum(c>cfg['common_cutoff'])>=2:horizons.append(('common',c>cfg['common_cutoff']))
                        horizons.extend((f'fixed_{distance}',c<=n+distance) for distance in [50,100,200] if np.sum(c<=n+distance)>=2)
                        for horizon,mask in horizons:
                            rec=dict(cell_id=cid,fold=plan['name'],origin=n,arm=arm,seed=args.seed,horizon=horizon,
                                matched_all_lengths=cid in common,**evaluate(c[mask],y[mask],p[mask]))
                            if horizon.startswith('fixed_'):rec['full_horizon_covered']=bool(c[-1]>=n+int(horizon.split('_')[1]))
                            batch.append(rec);rows.append(rec)
                    np.savez_compressed(folder/'predictions.npz',**arrays);save_json(folder/'metrics.json',batch)
                    save_json(folder/'config.json',dict(train=train,validation=val,model_sha256=sha(path)))
                    done+=1;save_json(root/'all_rows.json',rows);save_json(root/'progress.json',dict(terminal=False,completed_fits=done,planned_fits=len(cfg['arms'])*len(folds)*len(origins)))
                    rr=[r for r in batch if r['horizon']=='common'];message=f'{plan["name"]} N={n} {arm} train={len(train)} val={len(val)} MAPE={np.mean([r["mape_percent"] for r in rr]):.3f} late={np.mean([r["late20_mae_pp"] for r in rr]):.3f}'
                    print(message,flush=True);log.info(message)
        save_json(root/'progress.json',dict(terminal=True,success=True,completed_fits=done,planned_fits=len(cfg['arms'])*len(folds)*len(origins),seconds=time.perf_counter()-begin,goal_complete=False))
    except BaseException:
        log.exception('FAILED');save_json(root/'failure.json',dict(traceback=traceback.format_exc()))
        save_json(root/'progress.json',dict(terminal=True,success=False,completed_fits=done));raise
    finally:handler.close();log.removeHandler(handler)


if __name__=='__main__':main()
