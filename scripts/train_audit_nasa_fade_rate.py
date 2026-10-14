"""Fixed NASA fourfold rate experiment plus independent target/model audit."""
import json,logging,time,shutil
from pathlib import Path
from datetime import datetime,timezone
import joblib,numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from threadpoolctl import threadpool_limits
from battery_smetimes.fade_rate import fit,predict
from battery_smetimes.tail_boost import labels,inputs
from battery_smetimes.research_baselines import predict_baseline
from battery_smetimes.curve_fidelity import curve_metrics,selection_score
from scripts.train_crossfit_drift_smoke import clean
from scripts.train_tail_boost import load_cells
from scripts.prepare_same_type_forecast import save_json,sha
from scripts.audit_tail_numeric_gates import metric
from scripts.train_curve_fidelity import AUDIT


def main():
    smoke=Path('experiments/fade_rate/20261005T083158_483446Z');assert json.loads((smoke/'independent_verification.json').read_text())['passed']
    source=Path('experiments/nasa_signal_trajectory/ridge/20261004T165029_391663Z');scfg=json.loads((source/'config.json').read_text());plans=scfg['plans'];seed=20261005
    assert len(plans)==4 and all(p['dataset']=='NASA' for p in plans)
    root=Path('experiments/nasa_fade_rate_fourfold')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');root.mkdir(parents=True);snap=root/'source_snapshot';snap.mkdir()
    files=[Path(__file__),*[Path('battery_smetimes')/(n+'.py') for n in ('fade_rate','conditional_drift','crossfit_drift','tail_boost','research_baselines','forecast')],*[Path('scripts')/(n+'.py') for n in ('train_tail_boost','train_curve_fidelity','prepare_zhu2022_nca','train_zhu2022_nca')]]
    for f in files:shutil.copy2(f,snap/f.name)
    provenance={}
    for plan in plans:
        selection=source/plan['name']/str(seed)/'selection.json';record=json.loads(selection.read_text());idx=scfg['candidates'].index(record['selected']['parameters']);model=selection.parent/('candidate_'+str(idx))/'model.joblib'
        provenance[plan['name']]=dict(selection_file=str(selection.resolve()),selection_sha256=sha(selection),model_file=str(model.resolve()),model_sha256=sha(model),historical_parameters=record['selected']['parameters'])
    registry=json.loads((AUDIT/'cell_registry.json').read_text())['cells'];members={k for p in plans for k in p['train']+p['validation']}
    prepared={k:registry[k]['prepared_sha256'] for k in sorted(members)};assert len(prepared)==4
    save_json(root/'config.json',dict(plans=plans,seed=seed,parameters=dict(learned_anchor=False),source_run=str(source.resolve()),source_config_sha256=sha(source/'config.json'),provenance=provenance,source_sha256={f.name:sha(f) for f in files},preregistration_sha256=sha('docs/soh_nasa_rate_fourfold_20261005.md'),prepared_sha256=prepared,prepared_registry_sha256=sha(AUDIT/'cell_registry.json'),outer_loaded=False))
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(root/'training_audit.log',encoding='utf-8'),logging.StreamHandler()]);reports=[];checked=[];start=time.perf_counter()
    for plan in plans:
        cells=load_cells(plan);assert set(cells)==set(plan['train']+plan['validation']) and not set(cells)&set(plan['test'])
        assert len(plan['train'])==2 and len(plan['validation'])==1
        folder=root/plan['name'];folder.mkdir();logging.info('FIT %s',plan['name'])
        # Membership records only: no parameter fit is performed for fixed a=1.
        records=[dict(cell_id=k) for k in plan['train']];b=fit(cells,plan['train'],False,seed,records);joblib.dump(b,folder/'model.joblib')
        original=joblib.load(provenance[plan['name']]['model_file']);assert original['train_ids']==original['used_train_ids']==plan['train']
        # Independent training matrix, per-cell equal tail weights and median.
        xx=[];rr=[];ww=[]
        for key in plan['train']:
            c,y=labels(cells[key]);idx=np.unique(np.linspace(0,len(c)-1,min(96,len(c))).round().astype(int))
            xx.append(inputs(clean(cells[key]),c[idx]));r=(1-y[idx])/(1+(c[idx]-100)/100)
            np.testing.assert_allclose(r,100*(1-y[idx])/c[idx],atol=1e-14,rtol=0);rr.append(r);w=np.linspace(1,3,len(idx));ww.append(w/w.sum())
        z=np.concatenate(xx);r=np.concatenate(rr);w=np.concatenate(ww);w*=len(w)/w.sum();median=[]
        for col in z.T:
            values=col[np.isfinite(col)];median.append(np.median(values) if len(values) else 0.)
        median=np.asarray(median);np.testing.assert_allclose(median,b['imputer'].statistics_,atol=1e-12,rtol=0)
        z=np.where(np.isfinite(z),z,median);replay=ExtraTreesRegressor(n_estimators=128,min_samples_leaf=1,max_depth=None,random_state=seed,n_jobs=2).fit(z,r,sample_weight=w)
        new_rows=[];old_rows=[];train_rows=[];arrays={}
        for key in plan['train']+plan['validation']:
            cc=clean(cells[key]);x,y=labels(cells[key]);p=predict(b,cc,x);old=predict_baseline(original,cc,100,x)
            independent=1-x/100*replay.predict(b['imputer'].transform(inputs(cc,x)))
            np.testing.assert_allclose(p,independent,atol=1e-12,rtol=0)
            values=metric(x,y,p);display=curve_metrics(x,y,p)
            for field,value in values.items():np.testing.assert_allclose(value,display[field],atol=1e-10,rtol=0)
            poison=dict(cc,cycle=np.r_[cc['cycle'],101],raw=np.vstack([cc['raw'],np.full((1,cc['raw'].shape[1]),np.inf)]),soh=np.inf,cycle_life=np.inf)
            np.testing.assert_allclose(p,predict(b,poison,x),atol=1e-12,rtol=0);np.testing.assert_allclose(old,predict_baseline(original,poison,100,x),atol=1e-12,rtol=0)
            for fn in (lambda c:predict(b,c,x),lambda c:predict_baseline(original,c,100,x)):
                try:fn(dict(cc,physical_family='different'))
                except ValueError:pass
                else:raise AssertionError('family gate failed')
            arrays.update({key+'__100__'+field:a for field,a in [('cycle',x),('truth',y),('prediction',p),('strong_ridge',old)]})
            if key in plan['validation']:
                new_rows.append(dict(cell_id=key,**display));old_rows.append(dict(cell_id=key,**curve_metrics(x,y,old)))
            else:train_rows.append(dict(cell_id=key,**display))
            checked.append(dict(fold=plan['name'],cell_id=key,role='validation' if key in plan['validation'] else 'train',points=len(x)))
        np.savez_compressed(folder/'predictions.npz',**arrays);save_json(folder/'training_metrics.json',train_rows);save_json(folder/'validation_metrics.json',new_rows);save_json(folder/'strong_ridge_validation_metrics.json',old_rows)
        for name,rows in [('rate',new_rows),('strong_ridge',old_rows)]:
            report=dict(fold=plan['name'],model=name,score=selection_score(rows),**{f:float(np.mean([v[f] for v in rows])) for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp')});reports.append(report);logging.info('SAVED %s',report)
        save_json(root/'summary.json',reports)
    fields=('score','mape_percent','late_mape','endpoint_pp','drop_error_pp');means={name:{f:float(np.mean([r[f] for r in reports if r['model']==name])) for f in fields} for name in ('rate','strong_ridge')}
    wins=sum(next(r['score'] for r in reports if r['fold']==p['name'] and r['model']=='rate')<next(r['score'] for r in reports if r['fold']==p['name'] and r['model']=='strong_ridge') for p in plans)
    decision=wins>=3 and means['rate']['score']<means['strong_ridge']['score'] and all(means['rate'][f]<=means['strong_ridge'][f] for f in ('endpoint_pp','drop_error_pp'))
    save_json(root/'verification.json',dict(passed=True,checked=checked,models_retrained=4,outer_loaded=False,seconds=time.perf_counter()-start,goal_complete=False))
    save_json(root/'screening_decision.json',dict(means=means,better_folds=wins,folds=4,expand_formal=bool(decision),goal_complete=False,aggregation='equal-cell/fold, one seed, development internal validation'))
    print('OUTPUT='+str(root.resolve()),flush=True);print(json.dumps(dict(means=means,better_folds=wins,expand_formal=bool(decision))),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
