"""Train-only nominal-SOH continuous curve and history-length comparison."""
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
from threadpoolctl import threadpool_limits
from battery_smetimes.continuous_length import features,fit,predict
from scripts.train_tail_boost import load_cells
from scripts.train_curve_fidelity import AUDIT
from scripts.train_nca_curve_fidelity import OLD
from scripts.train_prefix_length_screen import metrics as diagnostic_metrics
from scripts.prepare_same_type_forecast import sha,save_json

CANDIDATES=[('point','VIT',1),('continuous','VIT',1),('continuous','VIT',3),('continuous','V',1),('continuous','condition',1)]
NCA_AUDIT=Path('data/interim/zhu2022_nca/20261004T161215_905258Z/audit.json')


def load_nominal(plan):
    cells=load_cells(plan);conversion={}
    nca_refs={r['cell_id']:r for r in json.loads(NCA_AUDIT.read_text())['cells']} if plan['dataset']=='Zhu2022_NCA' else {}
    for key,c in cells.items():
        if 'early_cycle' in c:
            factor=nca_refs[key]['initial_capacity_mAh']/3500
            c['truth']=np.asarray(c['truth'],float)*factor
        else:
            nominal={'MIT':1.1,'XJTU':2.0}[plan['dataset']]
            factor=c['reference_capacity_ah']/nominal
            c['soh']=np.asarray(c['soh'],float)*factor
        conversion[key]=dict(initial_to_nominal_factor=float(factor),label_definition='measured capacity / rated capacity',reference_used_as_input=False)
    return cells,conversion


def target(cell,cutoff):
    c,y=(cell['cycle'],cell['truth']) if 'early_cycle' in cell else (cell['label_cycle'],cell['soh'])
    c,y=np.asarray(c,float),np.asarray(y,float);take=(c>cutoff)&np.isfinite(y)&(y>0)
    return c[take],y[take]


def evaluate(c,y,p):
    row=diagnostic_metrics(c,y,p);late=c>=c[0]+.8*(c[-1]-c[0])
    row['late20_mae_pp']=float(np.mean(abs(y[late]-p[late]))*100)
    row['screen_pass']=bool(row['mape_percent']<=6 and row['mae_soh']<=.01 and row['late20_mae_pp']<=1.5
        and row['stage_error_pp']<=2 and row['fade_normalized_mae']<=.25
        and not row['acceleration_direction_reversed'] and not row['short_coverage'])
    row['knee_gate_verified']=False
    row['selection_score']=float(100*row['mae_soh']+row['late20_mae_pp']+row['stage_error_pp']
        +row['fade_normalized_mae']+10*row['acceleration_direction_reversed'])
    return row


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset');ap.add_argument('--origins',default='10,20,30,50,75,100')
    ap.add_argument('--full',action='store_true');ap.add_argument('--five-seeds',action='store_true');args=ap.parse_args()
    origins=[int(x) for x in args.origins.split(',')];assert all(0<x<=100 for x in origins)
    plans=json.loads((AUDIT/'splits.json').read_text())['plans']
    plans+=[dict(p,dataset='Zhu2022_NCA') for p in json.loads((OLD/'config.json').read_text())['plans']]
    plans=[p for p in plans if p['dataset'] in ('MIT','XJTU','Zhu2022_NCA')]
    if args.dataset:plans=[p for p in plans if p['dataset']==args.dataset]
    if not args.full:plans=[p for p in plans if p['name'].endswith('fold1')]
    assert plans
    seeds=list(range(20261008,20261013)) if args.five_seeds else [20261008]
    root=Path('experiments/continuous_length')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    root.mkdir(parents=True);print('RUN='+str(root.resolve()),flush=True)
    files=[Path(__file__),Path('battery_smetimes/continuous_length.py'),Path('scripts/train_tail_boost.py'),
        Path('scripts/train_curve_fidelity.py'),Path('scripts/train_prefix_length_screen.py'),Path('prepare_mit.py'),
        Path('prepare_xjtu.py'),Path('battery_smetimes/forecast.py'),Path('battery_smetimes/research_baselines.py')]
    snap=root/'source_snapshot';snap.mkdir()
    for f in files:shutil.copy2(f,snap/f.name)
    shutil.copy2('docs/soh_continuous_length_pilot_20261008.md',root/'registration.md')
    save_json(root/'config.json',dict(plans=plans,origins=origins,seeds=seeds,candidates=CANDIDATES,
        source_sha256={str(f):sha(f) for f in files},nca_audit_sha256=sha(NCA_AUDIT),
        outer_loaded=False,full=args.full,common_cutoff=max(100,max(origins)),label_definition='nominal',
        registration_sha256=sha(root/'registration.md'),independent_test=False,final_goal_complete=False))
    start=time.perf_counter();completed=0;allrows=[];coverage=[]
    try:
        for plan in plans:
            assert not set(plan['train'])&set(plan['validation'])
            assert not set(plan['train']+plan['validation'])&set(plan['test'])
            cells,conversion=load_nominal(plan);assert not set(cells)&set(plan['test'])
            targets={k:target(c,max(100,max(origins))) for k,c in cells.items()}
            available={}
            for role in ('train','validation'):
                eligible=[];excluded=[]
                for key in plan[role]:
                    try:
                        if len(targets[key][0])<2:raise ValueError('insufficient future labels')
                        for origin in origins:features(cells[key],origin)
                        eligible.append(key)
                    except ValueError as e:excluded.append(dict(cell_id=key,reason=str(e),not_bad_cell=True))
                available[role]=eligible
                coverage.append(dict(fold=plan['name'],role=role,original=len(plan[role]),matched_cells=eligible,excluded=excluded))
            assert len(available['train'])>=3 and available['validation']
            save_json(root/'coverage.json',coverage)
            save_json(root/(plan['name']+'_label_conversion.json'),conversion)
            for origin in origins:
                for seed in seeds:
                    for kind,block,leaf in CANDIDATES:
                        t=time.perf_counter();folder=root/plan['name']/str(origin)/str(seed)/f'{kind}_{block}_leaf{leaf}'
                        folder.mkdir(parents=True)
                        b=fit(cells,available['train'],origin,kind,block,leaf,seed,targets)
                        joblib.dump(b,folder/'model.joblib');rows=[];arrays={}
                        for key in available['validation']:
                            cell=cells[key];c,y=targets[key];p=predict(b,cell,c)
                            poison=deepcopy(cell)
                            if 'early_cycle' in cell:
                                poison['early_raw'][poison['early_cycle']>origin]=np.inf
                                poison['truth']=np.full_like(cell['truth'],np.inf);poison['cycle']=np.array([np.inf])
                            else:
                                poison['raw'][poison['cycle']>origin]=np.inf
                                poison['soh']=np.full_like(cell['soh'],np.inf);poison['reference_capacity_ah']=np.inf
                            poison['cycle_life']=np.inf
                            np.testing.assert_allclose(p,predict(b,poison,c),rtol=0,atol=1e-12)
                            try:predict(b,dict(cell,physical_family='wrong'),c)
                            except ValueError:pass
                            else:raise AssertionError('family gate')
                            grid=np.linspace(c[0],c[-1],400);dense=predict(b,cell,grid)
                            if kind=='continuous':
                                assert np.max(np.diff(dense))<=1e-10
                                assert np.max(np.diff(dense,n=2))<=1e-9
                            row=dict(cell_id=key,**evaluate(c,y,p));rows.append(row)
                            arrays.update({key+'__'+n:a for n,a in [('cycle',c),('truth',y),('prediction',p)]})
                        save_json(folder/'metrics.json',rows);np.savez_compressed(folder/'validation_predictions.npz',**arrays)
                        record=dict(fold=plan['name'],dataset=plan['dataset'],origin=origin,seed=seed,kind=kind,block=block,leaf=leaf,
                            model_sha256=sha(folder/'model.joblib'),used_train=available['train'],validation=available['validation'],
                            seconds=time.perf_counter()-t,input_contract='visible V/I/T only plus known conditions',
                            poison_passed=True,dense_constraints_checked=kind=='continuous',outer_loaded=False)
                        save_json(folder/'config.json',record)
                        for row in rows:allrows.append(dict(**record,**row))
                        completed+=1
                        save_json(root/'progress.json',dict(completed_fits=completed,terminal=False,seconds=time.perf_counter()-start))
                        print(f'FIT {plan["name"]} N={origin} {kind}/{block}/leaf{leaf} MAPE={np.mean([r["mape_percent"] for r in rows]):.3f} screen={np.mean([r["screen_pass"] for r in rows]):.3f}',flush=True)
        save_json(root/'all_rows.json',allrows)
        save_json(root/'progress.json',dict(completed_fits=completed,terminal=True,success=True,seconds=time.perf_counter()-start,goal_complete=False))
        print('COMPLETE '+str(root.resolve()),flush=True)
    except BaseException:
        save_json(root/'failure.json',dict(traceback=traceback.format_exc()))
        save_json(root/'progress.json',dict(completed_fits=completed,terminal=True,success=False));raise


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
