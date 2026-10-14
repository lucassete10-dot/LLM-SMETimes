"""Two fixed cross-fitted pace arms; inner fold only and saved full curves."""
import json,shutil,logging,time,traceback
from pathlib import Path
from datetime import datetime,timezone
import joblib,numpy as np
from threadpoolctl import threadpool_limits
from battery_smetimes.mit_pace_forecast import fit,predict,pace_predict,pace_label
from battery_smetimes.curve_fidelity import targets,selection_score
from scripts.train_curve_fidelity import load_subset
from scripts.train_crossfit_drift_smoke import clean
from scripts.train_nasa_robust_inner import detail
from scripts.prepare_same_type_forecast import sha,save_json

PARENT=Path('experiments/mit_feature_blend_full/20261006T143017_054601Z')


def main():
    pc=json.loads((PARENT/'config.json').read_text());proof=json.loads((PARENT/'independent_verification.json').read_text())
    assert proof['passed'] and not proof['outer_loaded']
    plan=next(p for p in pc['plans'] if p['name']=='MIT_fold1');seed=20261005
    child=PARENT/plan['name']/str(seed);control=json.loads((child/'selection.json').read_text())['selected']
    cells=load_subset(plan['train']+plan['validation']);assert not set(cells)&set(plan['test'])
    root=Path('experiments/mit_pace_smoke')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    root.mkdir(parents=True);print('RUN='+str(root.resolve()),flush=True);snap=root/'source_snapshot';snap.mkdir()
    files=[Path(__file__),Path('battery_smetimes/mit_pace_forecast.py'),Path('battery_smetimes/mit_segment_features.py'),
        Path('docs/soh_mit_pace_smoke_20261006.md'),Path('scripts/train_curve_fidelity.py'),Path('scripts/train_crossfit_drift_smoke.py')]
    for f in files:shutil.copy2(f,snap/f.name)
    save_json(root/'config.json',dict(parent=str(PARENT.resolve()),parent_proof_sha256=sha(PARENT/'independent_verification.json'),
        plan=plan,seed=seed,alphas=[10.,100.],control=control,control_selection_sha256=sha(child/'selection.json'),
        source_sha256={str(f):sha(f) for f in files},prepared_sha256={k:v['prepared_sha256'] for k,v in cells.items()},outer_loaded=False,smoke=True))
    logging.basicConfig(level=logging.INFO,handlers=[logging.FileHandler(root/'training.log',encoding='utf-8'),logging.StreamHandler()])
    save_json(root/'training_pace_supervision.json',[dict(cell_id=k,pace_label=pace_label(cells[k])) for k in plan['train']])
    scores=[];count=0;start=time.perf_counter()
    try:
        for i,alpha in enumerate([10.,100.]):
            out=root/f'candidate_{i}';out.mkdir();bundle=fit(cells,plan['train'],alpha,seed);joblib.dump(bundle,out/'model.joblib')
            save_json(out/'oof_split.json',[dict(train=f['train'],held=f['held']) for f in bundle['oof_folds']])
            save_json(out/'oof_paces.json',bundle['oof_paces'])
            for role,ids in [('training',plan['train']),('validation',plan['validation'])]:
                rows=[];arrays={}
                for key in ids:
                    c,y=targets(cells[key]);q=clean(cells[key]);p=predict(bundle,q,c)
                    rows.append(dict(cell_id=key,**detail(c,y,p),start_error_pp=float(100*abs(p[0]-y[0])),predicted_pace=pace_predict(bundle['pace_head'],q)))
                    arrays.update({key+'__100__'+f:v for f,v in [('cycle',c),('truth',y),('prediction',p)]});count+=1
                save_json(out/(role+'_metrics.json'),rows);np.savez_compressed(out/(role+'_predictions.npz'),**arrays)
                if role=='validation':vr=rows
            r=dict(candidate=i,alpha=alpha,score=selection_score(vr),**{f:float(np.mean([r[f] for r in vr])) for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp','start_error_pp')})
            scores.append(r);logging.info('SAVED %s',json.dumps(r))
        best=min(scores,key=lambda r:(r['score'],r['candidate']))
        expand=best['score']<control['score'] and best['mape_percent']<=control['mape_percent']+.1 and best['endpoint_pp']<=control['endpoint_pp'] and best['drop_error_pp']<=control['drop_error_pp']
        save_json(root/'selection.json',dict(selected=best,control=control,candidates=scores,expand=expand,outer_loaded=False,goal_complete=False))
        save_json(root/'progress.json',dict(complete=True,component_fits=14,curves=count,seconds=time.perf_counter()-start,outer_loaded=False))
        print(json.dumps(dict(selected=best,expand=expand,curves=count)),flush=True)
    except BaseException:save_json(root/'failure.json',dict(traceback=traceback.format_exc()));raise


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
