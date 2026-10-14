"""Rate-space weights aligned with SOH-space residual; small inner smoke."""
import copy,json,logging,shutil,time
from pathlib import Path
from datetime import datetime,timezone
import joblib,numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from threadpoolctl import threadpool_limits
from battery_smetimes.tail_boost import labels
from battery_smetimes.curve_fidelity import selection_score
from scripts.train_hgb_rate_smoke import SOURCE,supervision,evaluate
from scripts.train_curve_fidelity import load_subset
from scripts.prepare_same_type_forecast import sha,save_json

def aligned_weights(cells,ids,power):
    if power not in (1,2):raise ValueError('unregistered power')
    ws=[]
    for key in ids:
        c,_=labels(cells[key]);idx=np.unique(np.linspace(0,len(c)-1,min(96,len(c))).round().astype(int))
        w=np.linspace(1,3,len(idx))*(1+(c[idx]-100)/100)**power
        if not np.isfinite(w).all() or np.any(w<=0):raise ValueError('invalid weights')
        ws.append(w/w.sum())
    w=np.concatenate(ws);return w*(len(w)/w.sum())

def main():
    cfg=json.loads((SOURCE/'config.json').read_text());plan=cfg['plans'][0];seed=cfg['seeds'][0]
    assert json.loads((SOURCE/'independent_verification.json').read_text())['passed']
    inner=SOURCE/plan['name']/str(seed)/'inner';base=joblib.load(inner/'components.joblib')['rate'];assert base['train_ids']==plan['train']
    root=Path('experiments/rate_loss_alignment')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');root.mkdir(parents=True);print('RUN='+str(root.resolve()),flush=True)
    snapshot=root/'source_snapshot';snapshot.mkdir();files=[Path(__file__),Path('scripts/train_hgb_rate_smoke.py'),Path('docs/soh_rate_loss_alignment_20261005.md'),*Path('battery_smetimes').glob('*.py'),*[Path('scripts')/(n+'.py') for n in ('train_curve_fidelity','train_crossfit_drift_smoke','prepare_same_type_forecast')]]
    for f in files:shutil.copy2(f,snapshot/f.name)
    candidates=[dict(power=p,leaf=l) for p in (1,2) for l in (1,3)]
    save_json(root/'config.json',dict(plan=plan,seed=seed,candidates=candidates,source_run=str(SOURCE.resolve()),source_components_sha256=sha(inner/'components.joblib'),source_selection_sha256=sha(inner/'selection.json'),prepared_sha256={k:cfg['prepared_sha256'][k] for k in plan['train']+plan['validation']},source_sha256={str(f):sha(f) for f in files},outer_loaded=False))
    logging.basicConfig(level=logging.INFO,handlers=[logging.FileHandler(root/'training.log',encoding='utf-8'),logging.StreamHandler()]);start=time.perf_counter()
    cells=load_subset(plan['train']+plan['validation']);assert not set(cells)&set(plan['test']);x,y,_=supervision(cells,plan['train'],base['training_anchors']);xx=base['imputer'].transform(x);scores=[]
    for i,params in enumerate(candidates):
        folder=root/f'candidate_{i}';folder.mkdir();b=copy.deepcopy(base);w=aligned_weights(cells,plan['train'],params['power'])
        b['model']=ExtraTreesRegressor(n_estimators=128,min_samples_leaf=params['leaf'],max_depth=None,random_state=seed,n_jobs=2).fit(xx,y,sample_weight=w)
        b['rate_loss_parameters']=params;joblib.dump(b,folder/'model.joblib');np.savez_compressed(folder/'supervision.npz',target=y,weight=w)
        evaluate(b,cells,plan['train'],folder,'training');rows=evaluate(b,cells,plan['validation'],folder,'validation')
        score=dict(candidate=i,score=selection_score(rows),endpoint_pp=float(np.mean([r['endpoint_pp'] for r in rows])),drop_error_pp=float(np.mean([r['drop_error_pp'] for r in rows])),mape_percent=float(np.mean([r['mape_percent'] for r in rows])));scores.append(score);logging.info('SAVED %s',score)
    selected=min(scores,key=lambda r:(r['score'],r['candidate']));controls={};si=json.loads((inner/'selection.json').read_text())['selected']['candidate']
    for name,index in [('primary',0),('selected_blend',si)]:
        file=inner/f'candidate_{index}'/'validation_metrics.json';rows=json.loads(file.read_text());assert [r['cell_id'] for r in rows]==plan['validation']
        controls[name]=dict(score=selection_score(rows),endpoint_pp=float(np.mean([r['endpoint_pp'] for r in rows])),drop_error_pp=float(np.mean([r['drop_error_pp'] for r in rows])),file=str(file.resolve()),sha256=sha(file))
    screen=all(selected['score']<v['score'] and selected['endpoint_pp']<=v['endpoint_pp'] and selected['drop_error_pp']<=v['drop_error_pp'] for v in controls.values())
    save_json(root/'selection.json',dict(selected=selected,candidates=scores,controls=controls,expand_fivefold_smoke=screen,outer_loaded=False,goal_complete=False));save_json(root/'progress.json',dict(training_complete=True,seconds=time.perf_counter()-start,outer_loaded=False,goal_complete=False));print(json.dumps(dict(selected=selected,screen=screen)),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
