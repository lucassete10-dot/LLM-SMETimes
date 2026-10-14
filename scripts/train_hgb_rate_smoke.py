"""Fixed-budget training-only fade-rate HGB; frozen train-only anchor."""
import copy,json,logging,shutil,time
from pathlib import Path
from datetime import datetime,timezone
import joblib,numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from battery_smetimes.tail_boost import labels,train_arrays
from battery_smetimes.fade_rate import predict
from battery_smetimes.curve_fidelity import curve_metrics,selection_score
from scripts.train_curve_fidelity import load_subset
from scripts.train_crossfit_drift_smoke import clean
from scripts.prepare_same_type_forecast import save_json,sha

SOURCE=Path('experiments/rate_blend_formal/20261005T094223_349456Z')

def supervision(cells,ids,anchors):
    x,y,w=train_arrays(cells,ids,3);cycle=[];a=[]
    for key,anchor in zip(ids,anchors):
        c,_=labels(cells[key]);idx=np.unique(np.linspace(0,len(c)-1,min(96,len(c))).round().astype(int))
        cycle.extend(c[idx]);a.extend([anchor]*len(idx))
    return x,(np.asarray(a)-y)/(1+(np.asarray(cycle)-100)/100),w

def evaluate(bundle,cells,ids,folder,role):
    arrays={};rows=[]
    for key in ids:
        cycle,y=labels(cells[key]);cell=clean(cells[key]);p=predict(bundle,cell,cycle)
        poison=dict(cell,soh=np.inf,cycle_life=np.inf,cycle=np.r_[cell['cycle'],101],raw=np.vstack([cell['raw'],np.full((1,cell['raw'].shape[1]),np.inf)]))
        np.testing.assert_allclose(p,predict(bundle,poison,cycle),atol=1e-12,rtol=0)
        try:predict(bundle,dict(cell,physical_family='wrong'),cycle)
        except ValueError:pass
        else:raise AssertionError('family gate')
        rows.append(dict(cell_id=key,**curve_metrics(cycle,y,p)))
        arrays.update({key+'__100__'+f:v for f,v in [('cycle',cycle),('truth',y),('prediction',p)]})
    np.savez_compressed(folder/(role+'_predictions.npz'),**arrays);save_json(folder/(role+'_metrics.json'),rows)
    return rows

def main():
    cfg=json.loads((SOURCE/'config.json').read_text());plan=cfg['plans'][0];seed=cfg['seeds'][0]
    assert json.loads((SOURCE/'independent_verification.json').read_text())['passed']
    inner=SOURCE/plan['name']/str(seed)/'inner';base=joblib.load(inner/'components.joblib')['rate']
    assert base['train_ids']==plan['train']
    root=Path('experiments/hgb_rate_smoke')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');root.mkdir(parents=True)
    print('RUN='+str(root.resolve()),flush=True);snapshot=root/'source_snapshot';snapshot.mkdir()
    files=[Path(__file__),Path('docs/soh_hgb_rate_smoke_20261005.md'),*Path('battery_smetimes').glob('*.py'),*[Path('scripts')/(n+'.py') for n in ('train_curve_fidelity','train_crossfit_drift_smoke','prepare_same_type_forecast')]]
    for f in files:shutil.copy2(f,snapshot/f.name)
    candidates=[dict(max_leaf_nodes=l,l2_regularization=r) for l in (15,31) for r in (.1,1.)]
    save_json(root/'config.json',dict(plan=plan,seed=seed,candidates=candidates,source_run=str(SOURCE.resolve()),source_components_sha256=sha(inner/'components.joblib'),source_selection_sha256=sha(inner/'selection.json'),prepared_sha256={k:cfg['prepared_sha256'][k] for k in plan['train']+plan['validation']},source_sha256={str(f):sha(f) for f in files},outer_loaded=False))
    logging.basicConfig(level=logging.INFO,handlers=[logging.FileHandler(root/'training.log',encoding='utf-8'),logging.StreamHandler()]);start=time.perf_counter()
    cells=load_subset(plan['train']+plan['validation']);assert not set(cells)&set(plan['test'])
    x,y,w=supervision(cells,plan['train'],base['training_anchors']);xx=base['imputer'].transform(x);scores=[]
    for i,params in enumerate(candidates):
        folder=root/f'candidate_{i}';folder.mkdir();b=copy.deepcopy(base)
        b['model']=HistGradientBoostingRegressor(**params,learning_rate=.05,max_iter=120,min_samples_leaf=20,max_bins=64,early_stopping=False,random_state=seed).fit(xx,y,sample_weight=w)
        joblib.dump(b,folder/'model.joblib');evaluate(b,cells,plan['train'],folder,'training');rows=evaluate(b,cells,plan['validation'],folder,'validation')
        score=dict(candidate=i,score=selection_score(rows),endpoint_pp=float(np.mean([r['endpoint_pp'] for r in rows])),drop_error_pp=float(np.mean([r['drop_error_pp'] for r in rows])),mape_percent=float(np.mean([r['mape_percent'] for r in rows])))
        scores.append(score);logging.info('SAVED %s',score)
    selected=min(scores,key=lambda r:(r['score'],r['candidate']));controls={}
    si=json.loads((inner/'selection.json').read_text())['selected']['candidate']
    for name,index in [('primary',0),('selected_blend',si)]:
        file=inner/f'candidate_{index}'/'validation_metrics.json';rows=json.loads(file.read_text());assert [r['cell_id'] for r in rows]==plan['validation']
        controls[name]=dict(score=selection_score(rows),endpoint_pp=float(np.mean([r['endpoint_pp'] for r in rows])),drop_error_pp=float(np.mean([r['drop_error_pp'] for r in rows])),file=str(file.resolve()),sha256=sha(file))
    screen=all(selected['score']<v['score'] and selected['endpoint_pp']<=v['endpoint_pp'] and selected['drop_error_pp']<=v['drop_error_pp'] for v in controls.values())
    save_json(root/'selection.json',dict(selected=selected,candidates=scores,controls=controls,expand_fivefold_smoke=screen,outer_loaded=False,goal_complete=False))
    save_json(root/'progress.json',dict(training_complete=True,seconds=time.perf_counter()-start,outer_loaded=False,goal_complete=False));print(json.dumps(dict(selected=selected,controls=controls,screen=screen)),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
