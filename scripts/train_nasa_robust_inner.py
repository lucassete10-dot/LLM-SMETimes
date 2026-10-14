"""Fixed NASA calibration budget, rotating all three training-side cells."""
import json,logging,shutil,time,traceback
from pathlib import Path
from datetime import datetime,timezone
import joblib,numpy as np
from threadpoolctl import threadpool_limits
from battery_smetimes.nasa_cycle_calibration import fit as calfit,predict as calpredict
from battery_smetimes.research_baselines import fit_baseline,predict_baseline
from battery_smetimes.physical_types import load_registry
from battery_smetimes.tail_boost import labels
from scripts.train_curve_fidelity import load_subset,AUDIT
from scripts.train_crossfit_drift_smoke import clean
from scripts.prepare_same_type_forecast import sha,save_json
from scripts.audit_rate_blend_metrics import metrics
from scripts.verify_nasa_cycle_calibration import score


def specifications():
    grid=[dict(mode=m,alpha=a,window=w) for m in ('absolute','delta') for a in (.1,10.,100.) for w in (30,60)]
    return [dict(arm='control',calibration=None)]+[dict(arm=arm,calibration=p) for arm in ('direct','level') for p in grid]


def predict(primary,calibration,arm,query,c):
    if arm=='direct':return calpredict(calibration,query,c)
    base=predict_baseline(primary,query,100,c)
    if arm=='control':return base
    if arm!='level':raise ValueError('unknown arm')
    return base+calpredict(calibration,query,[101.])[0]-predict_baseline(primary,query,100,[101.])[0]


def detail(c,y,p):
    r=metrics(c,y,p);e=p-y;ss=float(np.sum((y-y.mean())**2))
    return dict(**r,mae_soh=float(np.mean(abs(e))),r2=1-float(e@e)/ss if ss else None,r2_undefined=not bool(ss),points=len(c),future_observation_span=float(c[-1]-c[0]),short_coverage=bool(len(c)<10 or c[-1]-c[0]<100))


def decisions(records,single):
    choices=[]
    for i in range(25):
        rr=[r for r in records if r['candidate']==i];assert len(rr)==3
        choices.append(dict(candidate=i,score=score(rr),**{f:float(np.mean([r[f] for r in rr])) for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp')}))
    chosen=min(choices,key=lambda r:(r['score'],r['candidate']));jackknife=[]
    for held in sorted({r['cell_id'] for r in records}):
        ranks=[(score([r for r in records if r['candidate']==i and r['cell_id']!=held]),i) for i in range(25)]
        jackknife.append(dict(omitted_validation_cell=held,candidate=min(ranks)[1]))
    return dict(selected=chosen,control=choices[0],single_validation_choice=choices[single],candidates=choices,selector_jackknife=jackknife)


def summarize(records):
    fields=('score','mape_percent','endpoint_pp','drop_error_pp');summary={name:{f:float(np.mean([r[name][f] for r in records])) for f in fields} for name in ('selected','control','single_validation_choice')};new,old=summary['selected'],summary['control'];wins=sum(r['selected']['score']<r['control']['score'] for r in records)
    expand=wins>=3 and new['mape_percent']<=old['mape_percent']+.1 and new['endpoint_pp']<=old['endpoint_pp']+1e-10 and new['drop_error_pp']<=old['drop_error_pp']+1e-10
    return dict(**summary,wins=wins,expand=expand,outer_loaded=False,goal_complete=False)


def main():
    parent=Path('experiments/nasa_cycle_calibration/20261005T154730_956753Z');level=Path('experiments/nasa_calibrated_level/20261005T155241_501407Z');pcfg=json.loads((parent/'config.json').read_text());assert json.loads((parent/'independent_verification.json').read_text())['passed'];assert json.loads((level/'independent_verification.json').read_text())['passed'];plans=pcfg['plans'];specs=specifications();seed=20261005;registry=load_registry(AUDIT/'source_snapshot/battery_physical_registry.json')
    root=Path('experiments/nasa_robust_inner')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');root.mkdir(parents=True);print('RUN='+str(root.resolve()),flush=True);snapshot=root/'source_snapshot';snapshot.mkdir();files=[Path(__file__),Path('docs/soh_nasa_robust_inner_20261006.md'),*[Path('battery_smetimes')/(n+'.py') for n in ('nasa_cycle_calibration','research_baselines','forecast','physical_types')],Path('scripts/train_curve_fidelity.py'),Path('scripts/train_crossfit_drift_smoke.py'),Path('scripts/audit_rate_blend_metrics.py'),Path('scripts/verify_nasa_cycle_calibration.py')]
    for f in files:shutil.copy2(f,snapshot/f.name)
    save_json(root/'config.json',dict(dataset='NASA',plans=plans,seed=seed,candidates=specs,control=pcfg['control'],parent=str(parent.resolve()),parent_config_sha256=sha(parent/'config.json'),level_parent=str(level.resolve()),level_parent_config_sha256=sha(level/'config.json'),source_sha256={str(f):sha(f) for f in files},registry_sha256=sha(AUDIT/'cell_registry.json'),physical_registry_sha256=sha(AUDIT/'source_snapshot/battery_physical_registry.json'),outer_loaded=False,deterministic=True,goal_complete=False));logging.basicConfig(level=logging.INFO,handlers=[logging.FileHandler(root/'training.log',encoding='utf-8'),logging.StreamHandler()]);start=time.perf_counter();records=[];allrows=[]
    try:
        for plan in plans:
            ids=plan['train']+plan['validation'];cells=load_subset(ids);assert set(cells)==set(ids) and not set(cells)&set(plan['test']);folder=root/plan['name'];folder.mkdir();save_json(folder/'data_sha256.json',{k:cells[k]['prepared_sha256'] for k in ids});vr=[]
            for held in ids:
                train=[k for k in ids if k!=held];dest=folder/('hold_'+held);dest.mkdir();save_json(dest/'split.json',dict(train=train,validation=[held],excluded_outer=plan['test']));primary=fit_baseline(cells,train,cells[train[0]]['physical_family'],registry,dict(origins_source_cycles=[100]),'ridge',pcfg['control'][plan['name']]['parameters'],seed);joblib.dump(primary,dest/'primary.joblib');calibrators=[calfit(cells,train,spec['calibration']) for spec in specs[1:13]]
                for i,model in enumerate(calibrators,1):joblib.dump(model,dest/f'calibration_{i}.joblib')
                for i,spec in enumerate(specs):
                    calibration=None if i==0 else calibrators[(i-1)%12];out=dest/f'candidate_{i}';out.mkdir()
                    for role,keys in [('training',train),('validation',[held])]:
                        rr=[];arrays={}
                        for key in keys:
                            c,y=labels(cells[key]);query=clean(cells[key]);p=predict(primary,calibration,spec['arm'],query,c);poison=dict(query,soh=np.inf,label_cycle=np.inf,reference_capacity_ah=np.inf,cycle_life=np.inf,cycle=np.r_[query['cycle'],101],raw=np.vstack([query['raw'],np.full((1,query['raw'].shape[1]),np.inf)]));np.testing.assert_allclose(p,predict(primary,calibration,spec['arm'],poison,c),rtol=0,atol=1e-12)
                            r=dict(cell_id=key,seed=seed,candidate=i,fold=plan['name'],heldout=held,role=role,**detail(c,y,p));rr.append(r);allrows.append(r);arrays.update({key+'__100__'+f:v for f,v in [('cycle',c),('truth',y),('prediction',p)]})
                        save_json(out/(role+'_metrics.json'),rr);np.savez_compressed(out/(role+'_predictions.npz'),**arrays)
                        if role=='validation':vr.extend(rr)
                    if held in plan['validation']:
                        reference=parent if spec['arm']=='direct' else level;ci=i if i<=12 else i-12
                        with np.load(reference/plan['name']/f'candidate_{ci}/validation_predictions.npz') as a:np.testing.assert_allclose(predict(primary,calibration,spec['arm'],clean(cells[held]),a[held+'__100__cycle']),a[held+'__100__prediction'],rtol=0,atol=1e-12)
                logging.info('SAVED %s held%s models13 curves75',plan['name'],held)
            single=json.loads((level/plan['name']/'selection.json').read_text())['selected']['candidate'];single=single+12 if single else 0;r=dict(fold=plan['name'],**decisions(vr,single));records.append(r);save_json(folder/'selection.json',r)
        save_json(root/'summary.json',dict(records=records,**summarize(records)));save_json(root/'comprehensive_curve_metrics.json',dict(rows=allrows,count=len(allrows),outer_loaded=False));save_json(root/'progress.json',dict(training_complete=True,component_models=156,curves=len(allrows),seconds=time.perf_counter()-start,outer_loaded=False,goal_complete=False));print(json.dumps(summarize(records)),flush=True)
    except BaseException:save_json(root/'failure.json',dict(traceback=traceback.format_exc()));raise


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
