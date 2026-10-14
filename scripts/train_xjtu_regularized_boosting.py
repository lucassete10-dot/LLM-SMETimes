"""Two prespecified boosting losses against frozen direct-tree baseline."""
import json,shutil,time,logging
from pathlib import Path
from datetime import datetime,timezone
import joblib,numpy as np
from threadpoolctl import threadpool_limits
from battery_smetimes.xjtu_regularized_boosting import fit,predict
from battery_smetimes.xjtu_segment_features import predict as base_predict
from battery_smetimes.curve_fidelity import targets,selection_score
from scripts.train_curve_fidelity import load_subset
from scripts.train_crossfit_drift_smoke import clean
from scripts.train_nasa_robust_inner import detail
from scripts.verify_mit_fixed_origin_dynamics import serial_trees
from scripts.prepare_same_type_forecast import sha,save_json

PARENT=Path('experiments/xjtu_oof_decline_gain/20261007T060106_225247Z')


def main():
    pc=json.loads((PARENT/'config.json').read_text());assert json.loads((PARENT/'independent_verification.json').read_text())['passed']
    plan=pc['plan'];base=joblib.load(pc['model_file']);serial_trees(base)
    cells=load_subset(plan['train']+plan['validation']);assert not set(cells)&set(plan['test'])
    root=Path('experiments/xjtu_regularized_boosting')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');root.mkdir(parents=True);snap=root/'source_snapshot';snap.mkdir();print('RUN='+str(root.resolve()),flush=True)
    files=[Path(__file__),Path('battery_smetimes/xjtu_regularized_boosting.py'),Path('battery_smetimes/xjtu_segment_features.py'),Path('docs/soh_xjtu_regularized_boosting_20261007.md')]
    for f in files:shutil.copy2(f,snap/f.name)
    deps=[PARENT/'config.json',PARENT/'independent_verification.json',Path(pc['model_file'])]
    save_json(root/'config.json',dict(parent=str(PARENT.resolve()),plan=plan,seed=pc['seed'],model_file=pc['model_file'],losses=['squared_error','absolute_error'],
        input_sha256={str(f.resolve()):sha(f) for f in deps},source_sha256={str(f):sha(f) for f in files},prepared_sha256={k:v['prepared_sha256'] for k,v in cells.items()},outer_loaded=False))
    logging.basicConfig(level=logging.INFO,handlers=[logging.FileHandler(root/'training.log',encoding='utf-8'),logging.StreamHandler()]);start=time.perf_counter();scores=[];curves=0;components=0
    for i,loss in enumerate([None,'squared_error','absolute_error']):
        b=base if loss is None else fit(cells,plan['train'],loss,pc['seed'])
        if loss is not None:components+=1+len(b['local'])
        out=root/f'candidate_{i}';out.mkdir();joblib.dump(b,out/'model.joblib')
        for role,ids in [('training',plan['train']),('validation',plan['validation'])]:
            rows=[];arrays={}
            for k in ids:
                c,y=targets(cells[k]);p=(base_predict if loss is None else predict)(b,clean(cells[k]),c)
                rows.append(dict(cell_id=k,**detail(c,y,p),start_error_pp=float(100*abs(p[0]-y[0]))));curves+=1
                arrays.update({k+'__100__'+f:v for f,v in [('cycle',c),('truth',y),('prediction',p)]})
            save_json(out/(role+'_metrics.json'),rows);np.savez_compressed(out/(role+'_predictions.npz'),**arrays)
            if role=='validation':vr=rows
        s=dict(candidate=i,loss=loss,score=selection_score(vr),**{f:float(np.mean([r[f] for r in vr])) for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp','start_error_pp')});scores.append(s);logging.info('SAVED %s',json.dumps(s))
    best=min(scores,key=lambda r:(r['score'],r['candidate']));old=scores[0];expand=best['candidate']>0 and best['score']<old['score'] and best['mape_percent']<=old['mape_percent']+.1 and best['endpoint_pp']<=old['endpoint_pp'] and best['drop_error_pp']<=old['drop_error_pp']
    save_json(root/'selection.json',dict(candidates=scores,selected=best,control=old,expand=expand,outer_loaded=False,goal_complete=False));save_json(root/'progress.json',dict(complete=True,new_component_fits=components,curves=curves,seconds=time.perf_counter()-start));print(json.dumps(dict(candidates=scores,expand=expand)),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
