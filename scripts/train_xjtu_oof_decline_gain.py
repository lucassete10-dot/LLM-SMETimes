"""Fixed 5-fold TRAIN OOF scalar calibration against frozen inner model."""
import json,shutil,time,logging
from pathlib import Path
from datetime import datetime,timezone
import joblib,numpy as np
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits
from battery_smetimes.xjtu_segment_features import fit,predict as base_predict
from battery_smetimes.xjtu_decline_gain import gain,predict
from battery_smetimes.curve_fidelity import targets,selection_score
from scripts.train_curve_fidelity import load_subset
from scripts.train_crossfit_drift_smoke import clean
from scripts.train_nasa_robust_inner import detail
from scripts.verify_mit_fixed_origin_dynamics import serial_trees
from scripts.prepare_same_type_forecast import sha,save_json

PARENT=Path('experiments/xjtu_segment_full/20261006T132530_948247Z/XJTU_fold1/20261005')


def main():
    pc=json.loads((PARENT/'config.json').read_text());sel=json.loads((PARENT/'selection.json').read_text())
    assert json.loads((PARENT/'independent_verification.json').read_text())['passed'] and sel['selected']['candidate']==6
    plan=pc['plan'];seed=pc['seed'];mf=PARENT/'candidate_6/model.joblib';base=joblib.load(mf);serial_trees(base)
    cells=load_subset(plan['train']+plan['validation']);assert not set(cells)&set(plan['test'])
    root=Path('experiments/xjtu_oof_decline_gain')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    root.mkdir(parents=True);print('RUN='+str(root.resolve()),flush=True);snap=root/'source_snapshot';snap.mkdir()
    files=[Path(__file__),Path('battery_smetimes/xjtu_decline_gain.py'),Path('battery_smetimes/xjtu_segment_features.py'),Path('docs/soh_xjtu_oof_decline_gain_20261007.md')]
    for f in files:shutil.copy2(f,snap/f.name)
    deps=[PARENT/'config.json',PARENT/'selection.json',PARENT/'independent_verification.json',mf]
    save_json(root/'config.json',dict(parent=str(PARENT.resolve()),input_sha256={str(f.resolve()):sha(f) for f in deps},model_file=str(mf.resolve()),
        plan=plan,seed=seed,penalties=[.001,.01],parameters=base['parameters'],source_sha256={str(f):sha(f) for f in files},
        prepared_sha256={k:v['prepared_sha256'] for k,v in cells.items()},outer_loaded=False))
    logging.basicConfig(level=logging.INFO,handlers=[logging.FileHandler(root/'training.log',encoding='utf-8'),logging.StreamHandler()])
    start=time.perf_counter();oof=[];splits=[];components=0;ids=plan['train']
    for i,(tr,va) in enumerate(KFold(5,shuffle=True,random_state=seed).split(ids)):
        train=[ids[j] for j in tr];held=[ids[j] for j in va];assert not set(train)&set(held)
        b=fit(cells,train,base['parameters'],seed);serial_trees(b);components+=1+len(b['local'])
        folder=root/f'oof_{i}';folder.mkdir();joblib.dump(b,folder/'model.joblib');arrays={}
        for k in held:
            c,y=targets(cells[k]);p=base_predict(b,clean(cells[k]),c)
            oof.append(dict(cell_id=k,cycle=c,truth=y,prediction=p))
            arrays.update({k+'__'+f:v for f,v in [('cycle',c),('truth',y),('prediction',p)]})
        np.savez_compressed(folder/'predictions.npz',**arrays);splits.append(dict(index=i,train=train,held=held))
    save_json(root/'oof_splits.json',splits);scores=[];curves=0
    for i,penalty in enumerate([None,.001,.01]):
        g,stats=(1.,None) if penalty is None else gain(oof,penalty)
        b=dict(base=base,gain=g,statistics=stats,penalty=penalty,train_ids=ids);out=root/f'candidate_{i}';out.mkdir();joblib.dump(b,out/'model.joblib')
        for role,keys in [('training',ids),('validation',plan['validation'])]:
            rows=[];arrays={}
            for k in keys:
                c,y=targets(cells[k]);p=predict(b,clean(cells[k]),c)
                rows.append(dict(cell_id=k,**detail(c,y,p),start_error_pp=float(100*abs(p[0]-y[0]))));curves+=1
                arrays.update({k+'__100__'+f:v for f,v in [('cycle',c),('truth',y),('prediction',p)]})
            save_json(out/(role+'_metrics.json'),rows);np.savez_compressed(out/(role+'_predictions.npz'),**arrays)
            if role=='validation':vr=rows
        s=dict(candidate=i,penalty=penalty,gain=g,score=selection_score(vr),
            **{f:float(np.mean([r[f] for r in vr])) for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp','start_error_pp')})
        scores.append(s);logging.info('SAVED %s',json.dumps(s))
    best=min(scores,key=lambda r:(r['score'],r['candidate']));old=scores[0]
    expand=best['candidate']>0 and best['score']<old['score'] and best['mape_percent']<=old['mape_percent']+.1 and best['endpoint_pp']<=old['endpoint_pp'] and best['drop_error_pp']<=old['drop_error_pp']
    save_json(root/'selection.json',dict(candidates=scores,selected=best,control=old,expand=expand,outer_loaded=False,goal_complete=False))
    save_json(root/'progress.json',dict(complete=True,oof_component_fits=components,oof_cells=31,curves=curves,seconds=time.perf_counter()-start,outer_loaded=False))
    print(json.dumps(dict(candidates=scores,expand=expand,curves=curves,components=components)),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
