"""Frozen-fold training-side six-arm feature experiment."""
import json,logging,shutil,time,traceback
from pathlib import Path
from datetime import datetime,timezone
import numpy as np,joblib
from threadpoolctl import threadpool_limits
from battery_smetimes.xjtu_segment_features import fit,predict
from battery_smetimes.tail_selection import choose,predict as old_predict
from battery_smetimes.curve_fidelity import targets,selection_score
from scripts.train_curve_fidelity import load_subset
from scripts.train_crossfit_drift_smoke import clean
from scripts.train_nasa_robust_inner import detail
from scripts.prepare_same_type_forecast import sha,save_json

def main():
    parent=Path('experiments/tail_boost/20261005T034452_066902Z');pcfg=json.loads((parent/'config.json').read_text());plan=next(p for p in pcfg['plans'] if p['name']=='XJTU_fold1');seed=20261005
    specs=[None]+[dict(block=b,context=c) for b in ('V','VI','VIT') for c in ('basic','segments')]
    cells=load_subset(plan['train']+plan['validation']);assert not set(cells)&set(plan['test']);choice=choose(parent,pcfg,plan,seed,cells)
    root=Path('experiments/xjtu_segment_features')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');root.mkdir(parents=True);print('RUN='+str(root.resolve()),flush=True);snap=root/'source_snapshot';snap.mkdir()
    files=[Path(__file__),Path('battery_smetimes/xjtu_segment_features.py'),Path('prepare_xjtu.py'),Path('docs/soh_xjtu_segment_features_20261006.md'),Path('battery_smetimes/tail_selection.py'),Path('battery_smetimes/tail_boost.py'),Path('scripts/train_curve_fidelity.py'),Path('scripts/train_crossfit_drift_smoke.py')]
    for f in files:shutil.copy2(f,snap/f.name)
    save_json(root/'config.json',dict(plan=plan,seed=seed,candidates=specs,parent=str(parent.resolve()),parent_config_sha256=sha(parent/'config.json'),control_choice=choice,source_sha256={str(f):sha(f) for f in files},prepared_sha256={k:v['prepared_sha256'] for k,v in cells.items()},outer_loaded=False,smoke=True))
    selected=choice['selected'];selection_file=Path(selected['selection_file']);base=selection_file.parent/f"candidate_{selected['candidate']}";old=joblib.load(base/'model.joblib');wrapper=dict(mode=selected['mode'],inner=old,family=cells[plan['train'][0]]['physical_family'])
    logging.basicConfig(level=logging.INFO,handlers=[logging.FileHandler(root/'training.log',encoding='utf-8'),logging.StreamHandler()]);scores=[];allrows=[];started=time.perf_counter()
    try:
        for i,p in enumerate(specs):
            out=root/f'candidate_{i}';out.mkdir();bundle=wrapper if not i else fit(cells,plan['train'],p,seed)
            joblib.dump(bundle,out/'model.joblib');vr=[]
            for role,ids in [('training',plan['train']),('validation',plan['validation'])]:
                rows=[];arrays={}
                for key in ids:
                    c,y=targets(cells[key]);q=clean(cells[key]);pred=old_predict(bundle,q,c) if not i else predict(bundle,q,c)
                    r=dict(cell_id=key,candidate=i,role=role,**detail(c,y,pred),start_error_pp=float(100*abs(pred[0]-y[0])))
                    rows.append(r);allrows.append(r);arrays.update({key+'__100__'+f:v for f,v in [('cycle',c),('truth',y),('prediction',pred)]})
                save_json(out/(role+'_metrics.json'),rows);np.savez_compressed(out/(role+'_predictions.npz'),**arrays)
                if role=='validation':vr=rows
            summary=dict(candidate=i,parameters=p,score=selection_score(vr),**{f:float(np.mean([r[f] for r in vr])) for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp','start_error_pp')});scores.append(summary);save_json(out/'summary.json',summary);logging.info('SAVED candidate %s score %.4f MAPE %.4f',i,summary['score'],summary['mape_percent'])
        best=min(scores,key=lambda r:(r['score'],r['candidate']));control=scores[0]
        expand=best['candidate']>0 and best['score']<control['score'] and best['mape_percent']<=control['mape_percent']+.1 and best['endpoint_pp']<=control['endpoint_pp'] and best['drop_error_pp']<=control['drop_error_pp']
        save_json(root/'selection.json',dict(candidates=scores,selected=best,control=control,expand=expand,outer_loaded=False,goal_complete=False));save_json(root/'comprehensive_curve_metrics.json',dict(rows=allrows,count=len(allrows)));save_json(root/'progress.json',dict(complete=True,seconds=time.perf_counter()-started,new_candidate_fits=6,curves=len(allrows),outer_loaded=False,goal_complete=False));print(json.dumps(dict(selected=best,control=control,expand=expand)),flush=True)
    except BaseException:save_json(root/'failure.json',dict(traceback=traceback.format_exc()));raise

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
