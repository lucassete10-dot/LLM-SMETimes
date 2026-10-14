"""Lock same-anchor slope-blend inner choices before frozen outer refits."""
import json,logging,shutil,time,traceback
from pathlib import Path
from datetime import datetime,timezone
import joblib,numpy as np
from threadpoolctl import threadpool_limits
from battery_smetimes.nasa_cycle_calibration import fit as calibration_fit
from battery_smetimes.nasa_slope_blend import predict as blend_predict
from battery_smetimes.research_baselines import fit_baseline,predict_baseline
from battery_smetimes.physical_types import load_registry
from battery_smetimes.tail_boost import labels
from scripts.train_curve_fidelity import load_subset,AUDIT
from scripts.train_crossfit_drift_smoke import clean
from scripts.prepare_same_type_forecast import sha,save_json
from scripts.audit_tail_numeric_gates import metric,aggregate,gates


def predict(bundle,cell,c):
    if bundle['calibration'] is None:return predict_baseline(bundle['primary'],cell,100,c)
    return blend_predict(bundle['primary'],bundle['calibration'],cell,c,bundle['weight'])


def main():
    screened=Path('experiments/nasa_slope_blend/20261005T162823_305584Z');proof=json.loads((screened/'independent_verification.json').read_text());assert proof['passed'] and proof['expand'];scfg=json.loads((screened/'config.json').read_text());parent=Path(json.loads((Path(scfg['parent'])/'config.json').read_text())['parent']);pcfg=json.loads((parent/'config.json').read_text());plans=scfg['plans'];seeds=list(range(20261005,20261010));baseline=Path(pcfg['source_run']);registry=load_registry(AUDIT/'source_snapshot/battery_physical_registry.json');root=Path('experiments/nasa_slope_blend_formal')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');root.mkdir(parents=True);print('RUN='+str(root.resolve()),flush=True);snapshot=root/'source_snapshot';snapshot.mkdir();files=[Path(__file__),Path('docs/soh_nasa_slope_blend_formal_20261006.md'),*[Path('battery_smetimes')/(n+'.py') for n in ('nasa_cycle_calibration','nasa_slope_blend','research_baselines','forecast','physical_types')],Path('scripts/train_curve_fidelity.py')]
    for f in files:shutil.copy2(f,snapshot/f.name)
    save_json(root/'config.json',dict(dataset='NASA',plans=plans,seeds=seeds,screened=str(screened.resolve()),screen_config_sha256=sha(screened/'config.json'),screen_verification_sha256=sha(screened/'independent_verification.json'),parent=str(parent.resolve()),parent_config_sha256=sha(parent/'config.json'),baseline=str(baseline.resolve()),baseline_config_sha256=sha(baseline/'config.json'),source_sha256={str(f):sha(f) for f in files},prepared_registry_sha256=sha(AUDIT/'cell_registry.json'),deterministic_repetitions=True,development_cv=True,goal_complete=False));logging.basicConfig(level=logging.INFO,handlers=[logging.FileHandler(root/'training.log',encoding='utf-8'),logging.StreamHandler()]);start=time.perf_counter();locks=[]
    try:
        for plan in plans:
            selection_file=screened/plan['name']/'selection.json';selection=json.loads(selection_file.read_text())['selected'];index=selection['candidate'];spec=scfg['candidates'][index];assert spec['arm'] in ('control','blend');control=pcfg['control'][plan['name']]['parameters'];calibration=spec['calibration'];weight=spec.get('weight',0.)
            for seed in seeds:
                oldselection=baseline/plan['name']/str(seed if seed<=20261008 else 20261008)/'selection.json';assert json.loads(oldselection.read_text())['selected']['parameters']==control
                locks.append(dict(fold=plan['name'],seed=seed,candidate=index,control_parameters=control,calibration_parameters=calibration,weight=weight,selection_sha256=sha(selection_file),robust_summary_sha256=sha(screened/'summary.json')))
        assert len(locks)==20;save_json(root/'inner_lock.json',dict(outer_loaded=False,selections=locks));locksha=sha(root/'inner_lock.json');rows=[];oldrows=[]
        for plan in plans:
            ids=plan['train']+plan['validation'];cells=load_subset(ids+plan['test']);folder=root/plan['name'];folder.mkdir();save_json(folder/'data_sha256.json',{k:c['prepared_sha256'] for k,c in cells.items()})
            for seed in seeds:
                assert sha(root/'inner_lock.json')==locksha;lock=next(r for r in locks if r['fold']==plan['name'] and r['seed']==seed);dest=folder/str(seed);dest.mkdir();primary=fit_baseline(cells,ids,cells[ids[0]]['physical_family'],registry,dict(origins_source_cycles=[100]),'ridge',lock['control_parameters'],seed);calibration=calibration_fit(cells,ids,lock['calibration_parameters']) if lock['calibration_parameters'] else None;bundle=dict(primary=primary,calibration=calibration,weight=lock['weight'],train_ids=ids);joblib.dump(bundle,dest/'model.joblib')
                for role,keys in [('training',ids),('test',plan['test'])]:
                    arrays={};rr=[]
                    for key in keys:
                        query=clean(cells[key]);c,y=labels(cells[key]);p=predict(bundle,query,c);r=dict(cell_id=key,seed=seed,**metric(c,y,p));rr.append(r);arrays.update({key+'__100__'+f:v for f,v in [('cycle',c),('truth',y),('prediction',p)]});poison=dict(query,soh=np.inf,reference_capacity_ah=np.inf,cycle_life=np.inf,cycle=np.r_[query['cycle'],101],raw=np.vstack([query['raw'],np.full((1,query['raw'].shape[1]),np.inf)]));np.testing.assert_allclose(p,predict(bundle,poison,c),rtol=0,atol=1e-12)
                    save_json(dest/(role+'_metrics.json'),rr);np.savez_compressed(dest/(role+'_predictions.npz'),**arrays)
                    if role=='test':rows.extend(rr)
                logging.info('SAVED %s seed%s',plan['name'],seed)
            for seed in range(20261004,20261009):
                with np.load(baseline/plan['name']/str(seed)/'predictions.npz') as a:
                    for key in plan['test']:
                        c,y,p=[a[key+'__100__'+f] for f in ('cycle','truth','prediction')];cx,cy=labels(cells[key]);np.testing.assert_array_equal(c,cx);np.testing.assert_array_equal(y,cy);oldrows.append(dict(cell_id=key,seed=seed,**metric(c,y,p)))
        new,bycell=aggregate(rows);old,_=aggregate(oldrows);check=gates(new,old);save_json(root/'by_cell_seed.json',rows);save_json(root/'by_cell.json',bycell);save_json(root/'summary.json',dict(new=new,baseline=old,gates=check,numeric_gates_passed=all(check.values()),goal_complete=False));save_json(root/'progress.json',dict(training_complete=True,models=20,inner_lock_sha256=locksha,seconds=time.perf_counter()-start,goal_complete=False));print(json.dumps(dict(new=new,gates=check)),flush=True)
    except BaseException:save_json(root/'failure.json',dict(traceback=traceback.format_exc()));raise


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
