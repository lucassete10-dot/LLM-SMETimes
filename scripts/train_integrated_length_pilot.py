"""Validation-only paired point-versus-integrated-rate experiment."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import time
import traceback
import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from battery_smetimes import continuous_length as point
from battery_smetimes import integrated_length as rate
from scripts.train_continuous_length_pilot import load_nominal,target,evaluate,NCA_AUDIT
from scripts.train_curve_fidelity import AUDIT
from scripts.train_nca_curve_fidelity import OLD
from scripts.prepare_same_type_forecast import sha,save_json

CANDIDATES=[('point',0,1),('rate',10,1),('rate',25,1),('rate',25,3)]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset');ap.add_argument('--origins',default='30,50,100')
    args=ap.parse_args();origins=list(map(int,args.origins.split(',')));assert all(0<n<=100 for n in origins)
    plans=json.loads((AUDIT/'splits.json').read_text())['plans']
    plans += [dict(p,dataset='Zhu2022_NCA') for p in json.loads((OLD/'config.json').read_text())['plans']]
    plans=[p for p in plans if p['dataset'] in ('MIT','XJTU','Zhu2022_NCA') and p['name'].endswith('fold1')]
    if args.dataset:plans=[p for p in plans if p['dataset']==args.dataset]
    assert plans;seed=20261008;cutoff=max(100,max(origins))
    root=Path('experiments/integrated_length')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');root.mkdir(parents=True)
    files=[Path(__file__),Path('battery_smetimes/integrated_length.py'),Path('battery_smetimes/continuous_length.py'),
           Path('battery_smetimes/xjtu_integrated_rate_v2.py'),Path('scripts/train_continuous_length_pilot.py'),
           Path('scripts/train_tail_boost.py'),Path('scripts/train_curve_fidelity.py'),Path('scripts/train_prefix_length_screen.py'),
           Path('prepare_mit.py'),Path('prepare_xjtu.py')]
    snap=root/'source_snapshot';snap.mkdir()
    for f in files:shutil.copy2(f,snap/f.name)
    shutil.copy2('docs/soh_integrated_length_pilot_20261008.md',root/'registration.md')
    save_json(root/'config.json',dict(plans=plans,origins=origins,seed=seed,candidates=CANDIDATES,common_cutoff=cutoff,
        source_sha256={str(f):sha(f) for f in files},nca_audit_sha256=sha(NCA_AUDIT),
        outer_loaded=False,independent_test=False,label_definition='nominal',goal_complete=False))
    print('RUN='+str(root.resolve()),flush=True);allrows=[];coverage=[];done=0;started=time.perf_counter()
    try:
        for plan in plans:
            cells,conversion=load_nominal(plan);assert not set(cells)&set(plan['test'])
            targets={k:target(c,cutoff) for k,c in cells.items()};available={}
            for role in ('train','validation'):
                available[role]=[];excluded=[]
                for k in plan[role]:
                    try:
                        if len(targets[k][0])<3:raise ValueError('less than three future labels')
                        for n in origins:point.features(cells[k],n)
                        available[role].append(k)
                    except ValueError as e:excluded.append(dict(cell_id=k,reason=str(e),not_bad_cell=True))
                coverage.append(dict(fold=plan['name'],role=role,matched_cells=available[role],excluded=excluded))
            assert not set(available['train'])&set(available['validation'])
            save_json(root/'coverage.json',coverage);save_json(root/(plan['name']+'_label_conversion.json'),conversion)
            for origin in origins:
                anchor=point.fit(cells,available['train'],origin,'point','VIT',1,seed,targets)
                for kind,window,leaf in CANDIDATES:
                    t=time.perf_counter();folder=root/plan['name']/str(origin)/str(seed)/f'{kind}_w{window}_leaf{leaf}';folder.mkdir(parents=True)
                    b=anchor if kind=='point' else rate.fit(cells,available['train'],targets,origin,'VIT',window,leaf,seed,anchor)
                    pred=point.predict if kind=='point' else rate.predict;joblib.dump(b,folder/'model.joblib')
                    if kind=='rate':
                        x,_,_=rate.training_arrays(cells,b['train_ids'],targets,origin,'VIT',window)
                        imp=SimpleImputer(strategy='median',keep_empty_features=True);z=imp.fit_transform(x);sc=StandardScaler().fit(z)
                        np.testing.assert_allclose(imp.statistics_,b['imputer'].statistics_,rtol=0,atol=1e-12)
                        np.testing.assert_allclose(sc.mean_,b['scaler'].mean_,rtol=0,atol=1e-12)
                    rows=[];arrays={};reloaded=joblib.load(folder/'model.joblib')
                    for k in available['validation']:
                        cell=cells[k];c,y=targets[k];p=pred(b,cell,c)
                        np.testing.assert_allclose(p,pred(reloaded,cell,c),rtol=0,atol=1e-12)
                        poison=deepcopy(cell)
                        if 'early_cycle' in cell:
                            poison['early_raw'][poison['early_cycle']>origin]=np.inf;poison['truth']=np.full_like(cell['truth'],np.inf)
                            poison['cycle']=np.array([np.inf])
                        else:
                            poison['raw'][poison['cycle']>origin]=np.inf;poison['soh']=np.full_like(cell['soh'],np.inf)
                            poison['reference_capacity_ah']=np.inf
                        poison['cycle_life']=np.inf
                        np.testing.assert_allclose(p,pred(b,poison,c),rtol=0,atol=1e-12)
                        try:pred(b,dict(cell,physical_family='wrong'),c)
                        except ValueError:pass
                        else:raise AssertionError('physical gate failed')
                        if kind=='rate':assert np.max(np.diff(pred(b,cell,np.linspace(c[0],c[-1],400))))<=1e-10
                        row=dict(cell_id=k,**evaluate(c,y,p));rows.append(row)
                        arrays.update({k+'__'+name:a for name,a in [('cycle',c),('truth',y),('prediction',p)]})
                    save_json(folder/'metrics.json',rows);np.savez_compressed(folder/'validation_predictions.npz',**arrays)
                    rec=dict(fold=plan['name'],dataset=plan['dataset'],origin=origin,seed=seed,kind=kind,window=window,leaf=leaf,
                             model_sha256=sha(folder/'model.joblib'),used_train=available['train'],validation=available['validation'],
                             seconds=time.perf_counter()-t,outer_loaded=False,replay_poison_family_checked=True)
                    save_json(folder/'config.json',rec);allrows.extend(dict(**rec,**r) for r in rows);done+=1
                    save_json(root/'all_rows.json',allrows);save_json(root/'progress.json',dict(completed_fits=done,terminal=False))
                    print(f'FIT {plan["name"]} N={origin} {kind} w={window} leaf={leaf} MAPE={np.mean([r["mape_percent"] for r in rows]):.3f} screen={np.mean([r["screen_pass"] for r in rows]):.3f}',flush=True)
        save_json(root/'progress.json',dict(completed_fits=done,terminal=True,success=True,seconds=time.perf_counter()-started,goal_complete=False))
        print('COMPLETE '+str(root.resolve()),flush=True)
    except BaseException:
        save_json(root/'failure.json',dict(traceback=traceback.format_exc()));save_json(root/'progress.json',dict(completed_fits=done,terminal=True,success=False));raise


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
