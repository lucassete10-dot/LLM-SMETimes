"""HUST-only cell-split input-length pilot; sealed outer fold is not loaded."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import joblib
import numpy as np
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits
from battery_smetimes import continuous_length as point
from battery_smetimes import integrated_length as rate
from scripts.train_continuous_length_pilot import target,evaluate
from scripts.prepare_same_type_forecast import sha,save_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('index',type=Path);args=ap.parse_args()
    index=json.loads(args.index.read_text());records={r['cell_id']:r for r in index['cells']};keys=sorted(records)
    folds=[np.array(keys)[test].tolist() for _,test in KFold(5,shuffle=True,random_state=20261008).split(keys)]
    test=folds[0];validation=folds[1];train=[k for k in keys if k not in test+validation]
    assert not set(train)&set(validation+test) and not set(validation)&set(test)
    root=Path('experiments/hust_length_smoke')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');root.mkdir(parents=True)
    origins=[30,50,100];candidates=[('point',0,1),('point',0,4),('rate',25,3)];seed=20261008
    sources=[Path(__file__),Path('battery_smetimes/continuous_length.py'),Path('battery_smetimes/integrated_length.py'),
             Path('scripts/prepare_hust_public.py'),Path('scripts/train_continuous_length_pilot.py')]
    save_json(root/'config.json',dict(dataset='HUST',train=train,validation=validation,test=test,folds=folds,
        seed=seed,origins=origins,candidates=candidates,source_index=str(args.index.resolve()),index_sha256=sha(args.index),
        source_sha256={str(f):sha(f) for f in sources},outer_loaded=False,independent_test=False,
        same_protocol_claim=False,label_definition='nominal',goal_complete=False))
    snap=root/'source_snapshot';snap.mkdir()
    for f in sources:shutil.copy2(f,snap/f.name)
    shutil.copy2('docs/soh_hust_smoke_20261008.md',root/'registration.md')
    cells={}
    for k in train+validation:
        r=records[k];assert sha(Path(r['path']))==r['sha256']
        with np.load(r['path'],allow_pickle=False) as a:cells[k]={n:a[n].copy() for n in a.files}
        cells[k]['physical_family']=r['physical_family']
        for n in origins:point.features(cells[k],n)
    assert not set(cells)&set(test)
    targets={k:target(c,100) for k,c in cells.items()};allrows=[];done=0
    print('RUN='+str(root.resolve()),flush=True)
    for origin in origins:
        anchor=point.fit(cells,train,origin,'point','VIT',1,seed,targets)
        for kind,window,leaf in candidates:
            folder=root/str(origin)/f'{kind}_w{window}_leaf{leaf}';folder.mkdir(parents=True)
            b=anchor if kind=='point' and leaf==1 else (point.fit(cells,train,origin,'point','VIT',leaf,seed,targets)
                if kind=='point' else rate.fit(cells,train,targets,origin,'VIT',window,leaf,seed,anchor))
            pred=point.predict if kind=='point' else rate.predict;joblib.dump(b,folder/'model.joblib');b2=joblib.load(folder/'model.joblib')
            rows=[];arrays={}
            for k in validation:
                c,y=targets[k];p=pred(b,cells[k],c)
                np.testing.assert_allclose(p,pred(b2,cells[k],c),rtol=0,atol=1e-12)
                poison=deepcopy(cells[k]);poison['early_raw'][poison['early_cycle']>origin]=np.inf
                poison['truth']=np.full_like(poison['truth'],np.inf);poison['cycle']=np.array([np.inf]);poison['cycle_life']=np.inf
                np.testing.assert_allclose(p,pred(b,poison,c),rtol=0,atol=1e-12)
                try:pred(b,dict(cells[k],physical_family='wrong'),c)
                except ValueError:pass
                else:raise AssertionError('family gate failed')
                if kind=='rate':assert max(np.diff(pred(b,cells[k],np.linspace(c[0],c[-1],400))))<=1e-10
                rows.append(dict(cell_id=k,**evaluate(c,y,p)))
                arrays.update({k+'__'+n:a for n,a in [('cycle',c),('truth',y),('prediction',p)]})
            rec=dict(dataset='HUST',origin=origin,kind=kind,window=window,leaf=leaf,seed=seed,
                     used_train=train,validation=validation,model_sha256=sha(folder/'model.joblib'),outer_loaded=False)
            save_json(folder/'config.json',rec);save_json(folder/'metrics.json',rows)
            np.savez_compressed(folder/'validation_predictions.npz',**arrays);allrows.extend(dict(**rec,**r) for r in rows)
            save_json(root/'all_rows.json',allrows);done+=1;save_json(root/'progress.json',dict(completed_fits=done,terminal=False))
            print(f'FIT N={origin} {kind} leaf={leaf} MAPE={np.mean([r["mape_percent"] for r in rows]):.3f} screen={np.mean([r["screen_pass"] for r in rows]):.3f}',flush=True)
    save_json(root/'progress.json',dict(completed_fits=done,terminal=True,success=True,goal_complete=False))
    print('COMPLETE '+str(root.resolve()),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
