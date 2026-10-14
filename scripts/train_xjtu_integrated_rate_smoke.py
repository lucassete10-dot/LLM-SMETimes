"""Four fixed rate-label candidates, preserving independent segment anchor."""
import json,logging,shutil,time,traceback
from pathlib import Path
from datetime import datetime,timezone
import joblib,numpy as np
from threadpoolctl import threadpool_limits
from battery_smetimes.xjtu_integrated_rate import fit_rates,predict
from battery_smetimes.xjtu_segment_features import predict as control_predict
from battery_smetimes.curve_fidelity import targets,selection_score
from scripts.train_curve_fidelity import load_subset
from scripts.train_crossfit_drift_smoke import clean
from scripts.train_nasa_robust_inner import detail
from scripts.prepare_same_type_forecast import sha,save_json

PARENT=Path('experiments/xjtu_segment_full/20261006T132530_948247Z/XJTU_fold1/20261005')

def main():
    cc=json.loads((PARENT/'config.json').read_text());plan=cc['plan'];seed=cc['seed']
    assert json.loads((PARENT/'independent_verification.json').read_text())['passed']
    assert json.loads((PARENT/'selection.json').read_text())['selected']['candidate']==6
    model_file=PARENT/'candidate_6/model.joblib';anchor=joblib.load(model_file)
    cells=load_subset(plan['train']+plan['validation']);assert not set(cells)&set(plan['test'])
    specs=[None]+[dict(window=w,leaf=l) for w in (10,25) for l in (1,2)]
    root=Path('experiments/xjtu_integrated_rate')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');root.mkdir(parents=True)
    print('RUN='+str(root.resolve()),flush=True);snap=root/'source_snapshot';snap.mkdir()
    files=[Path(__file__),Path('docs/soh_xjtu_integrated_rate_smoke_20261006.md'),*[Path('battery_smetimes')/(n+'.py') for n in ('xjtu_integrated_rate','xjtu_segment_features','curve_fidelity')],Path('scripts/train_curve_fidelity.py')]
    for f in files:shutil.copy2(f,snap/f.name)
    save_json(root/'config.json',dict(parent=str(PARENT.resolve()),parent_config_sha256=sha(PARENT/'config.json'),parent_proof_sha256=sha(PARENT/'independent_verification.json'),model_file=str(model_file.resolve()),model_sha256=sha(model_file),plan=plan,seed=seed,candidates=specs,prepared_sha256={k:c['prepared_sha256'] for k,c in cells.items()},source_sha256={str(f):sha(f) for f in files},outer_loaded=False,smoke=True))
    logging.basicConfig(level=logging.INFO,handlers=[logging.FileHandler(root/'training.log',encoding='utf-8'),logging.StreamHandler()])
    scores=[];count=fits=0;start=time.perf_counter()
    try:
        for i,p in enumerate(specs):
            dest=root/f'candidate_{i}';dest.mkdir()
            if not i:b=anchor
            else:
                r=fit_rates(cells,plan['train'],p,seed);fits+=1+len(r['local'])
                b=dict(anchor=anchor,rate=r,family=r['family'])
            joblib.dump(b,dest/'model.joblib');vr=[]
            for role,ids in [('training',plan['train']),('validation',plan['validation'])]:
                rows=[];arrays={}
                for key in ids:
                    c,y=targets(cells[key]);query=clean(cells[key]);pred=control_predict(b,query,c) if not i else predict(b,query,c)
                    rows.append(dict(cell_id=key,**detail(c,y,pred),start_error_pp=float(100*abs(pred[0]-y[0]))))
                    arrays.update({key+'__100__'+f:v for f,v in [('cycle',c),('truth',y),('prediction',pred)]});count+=1
                save_json(dest/(role+'_metrics.json'),rows);np.savez_compressed(dest/(role+'_predictions.npz'),**arrays)
                if role=='validation':vr=rows
            r=dict(candidate=i,parameters=p,score=selection_score(vr),**{f:float(np.mean([r[f] for r in vr])) for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp','start_error_pp')});scores.append(r);logging.info('SAVED %s',json.dumps(r))
        best=min(scores,key=lambda r:(r['score'],r['candidate']));old=scores[0]
        expand=best['candidate']>0 and best['score']<old['score'] and best['mape_percent']<=old['mape_percent']+.1 and best['endpoint_pp']<=old['endpoint_pp'] and best['drop_error_pp']<=old['drop_error_pp']
        save_json(root/'selection.json',dict(selected=best,control=old,candidates=scores,expand=expand,outer_loaded=False,goal_complete=False))
        save_json(root/'progress.json',dict(complete=True,component_fits=fits,curves=count,seconds=time.perf_counter()-start,outer_loaded=False,goal_complete=False))
        print(json.dumps(dict(selected=best,expand=expand,component_fits=fits,curves=count)),flush=True)
    except BaseException:save_json(root/'failure.json',dict(traceback=traceback.format_exc()));raise

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
