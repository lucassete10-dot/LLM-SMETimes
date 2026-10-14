"""TRAIN-internal whole-cell validation, no original validation/test loading."""
import json,shutil,time,logging
from pathlib import Path
from datetime import datetime,timezone
import joblib,numpy as np
from threadpoolctl import threadpool_limits
from battery_smetimes.xjtu_personal_shape import target,fit,predict,coefficients,basis
from battery_smetimes.curve_fidelity import targets,selection_score
from scripts.train_curve_fidelity import load_subset
from scripts.train_crossfit_drift_smoke import clean
from scripts.train_nasa_robust_inner import detail
from scripts.prepare_same_type_forecast import sha,save_json

PARENT=Path('experiments/xjtu_oof_decline_gain/20261007T060106_225247Z')


def cell_score(r):return float(np.mean([r[f] for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp')]))


def main():
    pc=json.loads((PARENT/'config.json').read_text());plan=pc['plan'];splits=json.loads((PARENT/'oof_splits.json').read_text())
    assert json.loads((PARENT/'independent_verification.json').read_text())['passed']
    cells=load_subset(plan['train']);assert len(cells)==31 and not set(cells)&set(plan['validation']+plan['test'])
    root=Path('experiments/xjtu_personal_shape_learnability')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    root.mkdir(parents=True);snap=root/'source_snapshot';snap.mkdir();print('RUN='+str(root.resolve()),flush=True)
    files=[Path(__file__),Path('battery_smetimes/xjtu_personal_shape.py'),Path('battery_smetimes/xjtu_segment_features.py'),Path('docs/soh_xjtu_personal_shape_learnability_20261007.md')]
    for f in files:shutil.copy2(f,snap/f.name)
    deps=[PARENT/'config.json',PARENT/'independent_verification.json',PARENT/'oof_splits.json']
    save_json(root/'config.json',dict(plan=plan,seed=pc['seed'],splits=splits,alphas=[None,10,100],input_sha256={str(f.resolve()):sha(f) for f in deps},
        source_sha256={str(f):sha(f) for f in files},prepared_sha256={k:v['prepared_sha256'] for k,v in cells.items()},validation_loaded=False,outer_loaded=False))
    logging.basicConfig(level=logging.INFO,handlers=[logging.FileHandler(root/'training.log',encoding='utf-8'),logging.StreamHandler()]);start=time.perf_counter()
    shapes={};oracle=[];oa={}
    for k in plan['train']:
        c,y=targets(cells[k]);shapes[k]=target(c,y);p=basis(c)@shapes[k];oracle.append(dict(cell_id=k,**detail(c,y,p)))
        oa.update({k+'__'+f:v for f,v in [('cycle',c),('truth',y),('prediction',p),('coefficients',shapes[k])]})
    np.savez_compressed(root/'oracle.npz',**oa);save_json(root/'oracle_metrics.json',oracle)
    results=[];allrows=[];curves=0
    for j,alpha in enumerate([None,10,100]):
        out=root/f'candidate_{j}';out.mkdir();rows=[];arrays={}
        for s in splits:
            assert not set(s['train'])&set(s['held']) and set(s['train']+s['held'])==set(plan['train'])
            b=fit(cells,s['train'],shapes,alpha);joblib.dump(b,out/f"fold_{s['index']}.joblib")
            for k in s['held']:
                c,y=targets(cells[k]);q=clean(cells[k]);p=predict(b,q,c);a=coefficients(b,q)
                rows.append(dict(cell_id=k,fold=s['index'],**detail(c,y,p),start_error_pp=float(100*abs(p[0]-y[0])),coefficient_mae=float(np.mean(abs(a-shapes[k])))));curves+=1
                arrays.update({k+'__'+f:v for f,v in [('cycle',c),('truth',y),('prediction',p),('coefficients',a)]})
        np.savez_compressed(out/'predictions.npz',**arrays);save_json(out/'metrics.json',rows);allrows.append(rows)
        summary=dict(candidate=j,alpha=alpha,score=selection_score(rows),**{f:float(np.mean([r[f] for r in rows])) for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp','coefficient_mae')})
        results.append(summary);logging.info('SAVED %s',json.dumps(summary))
    old=results[0]
    for i,s in enumerate(results):
        s['improved_cells']=sum(cell_score(r)<cell_score(c) for r,c in zip(allrows[i],allrows[0]))
        s['passes_diagnostic_gate']=i>0 and s['score']<old['score'] and s['mape_percent']<=old['mape_percent']+.1 and s['endpoint_pp']<=old['endpoint_pp'] and s['drop_error_pp']<=old['drop_error_pp'] and s['improved_cells']/31>=.6
    save_json(root/'selection.json',dict(candidates=results,proceed_original_inner=any(s['passes_diagnostic_gate'] for s in results),validation_loaded=False,outer_loaded=False,goal_complete=False))
    save_json(root/'progress.json',dict(complete=True,new_ridge_fits=10,mean_controls=5,own_label_shape_fits=31,held_curves=curves,seconds=time.perf_counter()-start))
    print(json.dumps(results),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
