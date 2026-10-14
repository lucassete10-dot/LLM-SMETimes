"""NASA fourfold training-side calibration smoke; outer never loaded."""
import json,logging,shutil,time,traceback
from pathlib import Path
from datetime import datetime,timezone
import joblib,numpy as np
from threadpoolctl import threadpool_limits
from battery_smetimes.nasa_cycle_calibration import fit,predict,early_health,supervision
from battery_smetimes.research_baselines import fit_baseline,predict_baseline
from battery_smetimes.physical_types import load_registry
from battery_smetimes.curve_fidelity import curve_metrics,selection_score
from battery_smetimes.tail_boost import labels
from scripts.train_curve_fidelity import load_subset,AUDIT
from scripts.train_crossfit_drift_smoke import clean
from scripts.prepare_same_type_forecast import save_json,sha


def main():
    source=Path('experiments/nasa_signal_trajectory/ridge/20261004T165029_391663Z');scfg=json.loads((source/'config.json').read_text());seed=20261005;plans=scfg['plans'];registry=load_registry(AUDIT/'source_snapshot/battery_physical_registry.json');specs=[dict(mode=m,alpha=a,window=w) for m in ('absolute','delta') for a in (.1,10.,100.) for w in (30,60)]
    root=Path('experiments/nasa_cycle_calibration')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');root.mkdir(parents=True);print('RUN='+str(root.resolve()),flush=True);snapshot=root/'source_snapshot';snapshot.mkdir();files=[Path(__file__),Path('battery_smetimes/nasa_cycle_calibration.py'),Path('docs/soh_nasa_cycle_calibration_20261005.md'),*[Path('battery_smetimes')/(n+'.py') for n in ('research_baselines','forecast','physical_types')],Path('scripts/train_curve_fidelity.py')]
    for f in files:shutil.copy2(f,snapshot/f.name)
    provenance={}
    for plan in plans:
        file=source/plan['name']/str(seed)/'selection.json';provenance[plan['name']]=dict(file=str(file.resolve()),sha256=sha(file),parameters=json.loads(file.read_text())['selected']['parameters'])
    save_json(root/'config.json',dict(dataset='NASA',plans=plans,seed=seed,candidates=specs,control=provenance,source_config_sha256=sha(source/'config.json'),source_run=str(source.resolve()),source_sha256={str(f):sha(f) for f in files},registry_sha256=sha(AUDIT/'cell_registry.json'),physical_registry_sha256=sha(AUDIT/'source_snapshot/battery_physical_registry.json'),original_test_loaded=False,goal_complete=False))
    logging.basicConfig(level=logging.INFO,handlers=[logging.FileHandler(root/'training.log',encoding='utf-8'),logging.StreamHandler()]);start=time.perf_counter();records=[]
    try:
        for plan in plans:
            ids=plan['train'];cells=load_subset(ids+plan['validation']);assert set(cells)==set(ids+plan['validation']) and not set(cells)&set(plan['test']);folder=root/plan['name'];folder.mkdir();family=cells[ids[0]]['physical_family'];control=fit_baseline(cells,ids,family,registry,dict(origins_source_cycles=[100]),'ridge',provenance[plan['name']]['parameters'],seed);models=[control]+[fit(cells,ids,p) for p in specs];results=[]
            save_json(folder/'data_sha256.json',{k:cells[k]['prepared_sha256'] for k in cells})
            for i,bundle in enumerate(models):
                dest=folder/f'candidate_{i}';dest.mkdir();joblib.dump(bundle,dest/'model.joblib');allrows={}
                if i:
                    x,y,w,members=supervision(cells,ids,bundle['parameters']['mode']);np.savez_compressed(dest/'supervision.npz',features=x,truth=y,weight=w);save_json(dest/'members.json',members)
                for role,keys in [('training',ids),('validation',plan['validation'])]:
                    arrays={};rows=[];early={}
                    for key in keys:
                        c,y=labels(cells[key]);query=clean(cells[key]);p=predict(bundle,query,c) if i else predict_baseline(bundle,query,100,c)
                        poisoned=dict(query,soh=np.inf,label_cycle=np.inf,cycle_life=np.inf,cycle=np.r_[query['cycle'],101],raw=np.vstack([query['raw'],np.full((1,query['raw'].shape[1]),np.inf)]));other=predict(bundle,poisoned,c) if i else predict_baseline(bundle,poisoned,100,c);np.testing.assert_allclose(p,other,rtol=0,atol=1e-12)
                        rows.append(dict(cell_id=key,**curve_metrics(c,y,p)));arrays.update({key+'__100__'+f:v for f,v in [('cycle',c),('truth',y),('prediction',p)]})
                        if i:
                            ec,ey=early_health(bundle,query);early.update({key+'__'+f:v for f,v in [('cycle',ec),('predicted_health',ey)]})
                    np.savez_compressed(dest/(role+'_predictions.npz'),**arrays);save_json(dest/(role+'_metrics.json'),rows)
                    if i:np.savez_compressed(dest/(role+'_early_health.npz'),**early)
                    allrows[role]=rows
                rows=allrows['validation'];results.append(dict(candidate=i,parameters=bundle['parameters'],score=selection_score(rows),**{f:float(np.mean([r[f] for r in rows])) for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp')}));logging.info('SAVED %s candidate%s score%.6f',plan['name'],i,results[-1]['score'])
            chosen=min(results,key=lambda r:(r['score'],r['candidate']));record=dict(fold=plan['name'],selected=chosen,control=results[0],candidates=results);records.append(record);save_json(folder/'selection.json',record)
        fields=('score','mape_percent','endpoint_pp','drop_error_pp');new={f:float(np.mean([r['selected'][f] for r in records])) for f in fields};old={f:float(np.mean([r['control'][f] for r in records])) for f in fields};wins=sum(r['selected']['score']<r['control']['score'] for r in records);expand=wins>=3 and new['endpoint_pp']<=old['endpoint_pp'] and new['drop_error_pp']<=old['drop_error_pp'] and new['mape_percent']<=old['mape_percent']+.1
        save_json(root/'summary.json',dict(records=records,new=new,control=old,wins=wins,expand=expand,outer_loaded=False,goal_complete=False));save_json(root/'progress.json',dict(training_complete=True,seconds=time.perf_counter()-start,outer_loaded=False,goal_complete=False));print(json.dumps(dict(new=new,control=old,wins=wins,expand=expand)),flush=True)
    except BaseException:save_json(root/'failure.json',dict(traceback=traceback.format_exc()));raise


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
