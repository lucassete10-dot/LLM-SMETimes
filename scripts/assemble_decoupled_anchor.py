"""Freeze and audit four signal-anchor assemblies; no new fitting or outer data."""
import json,logging,shutil,time
from datetime import datetime,timezone
from pathlib import Path
import joblib,numpy as np
from threadpoolctl import threadpool_limits
from battery_smetimes.decoupled_anchor import make,predict
from battery_smetimes.pooled_anchor import anchor
from battery_smetimes.fade_rate import predict as fade
from battery_smetimes.tail_boost import labels
from battery_smetimes.curve_fidelity import curve_metrics,selection_score
from scripts.train_curve_fidelity import load_subset
from scripts.train_crossfit_drift_smoke import clean
from scripts.prepare_same_type_forecast import save_json,sha
from scripts.audit_tail_numeric_gates import metric


def main():
    pooled=Path('experiments/pooled_anchor/20261005T090310_249870Z');pcfg=json.loads((pooled/'config.json').read_text());assert json.loads((pooled/'independent_verification.json').read_text())['passed']
    rate=Path(pcfg['source_run']);assert sha(rate/'config.json')==pcfg['source_config_sha256']
    reference=Path('experiments/nasa_fade_rate_fourfold/20261005T085132_456474Z');assert json.loads((reference/'verification.json').read_text())['passed'];refcfg=json.loads((reference/'config.json').read_text());assert refcfg['plans']==pcfg['plans']
    root=Path('experiments/decoupled_anchor')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');root.mkdir(parents=True);snap=root/'source_snapshot';snap.mkdir()
    files=[Path(__file__),*[Path('battery_smetimes')/(n+'.py') for n in ('decoupled_anchor','pooled_anchor','fade_rate','tail_boost','forecast','research_baselines')]]
    for f in files:shutil.copy2(f,snap/f.name)
    save_json(root/'config.json',dict(plans=pcfg['plans'],seed=pcfg['seed'],candidates=pcfg['candidates'],prepared_sha256=pcfg['prepared_sha256'],pooled_run=str(pooled.resolve()),pooled_config_sha256=sha(pooled/'config.json'),rate_run=str(rate.resolve()),rate_config_sha256=sha(rate/'config.json'),reference_run=str(reference.resolve()),reference_summary_sha256=sha(reference/'summary.json'),source_sha256={f.name:sha(f) for f in files},preregistration_sha256=sha('docs/soh_decoupled_anchor_stage_20261005.md'),outer_loaded=False,new_models_trained=0))
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(root/'assembly.log',encoding='utf-8'),logging.StreamHandler()]);reports=[];checked=[];start=time.perf_counter()
    for plan in pcfg['plans']:
        cells=load_subset(plan['train']+plan['validation']);assert not set(cells)&set(plan['test'])
        rate_file=rate/plan['name']/str(pcfg['seed'])/'inner'/'model.joblib';assert sha(rate_file)==pcfg['control_sha256'][plan['name']];base=joblib.load(rate_file);assert base['train_ids']==plan['train']
        for index,params in enumerate(pcfg['candidates']):
            folder=root/plan['name']/('candidate_'+str(index));folder.mkdir(parents=True);logging.info('ASSEMBLE %s candidate=%s',plan['name'],index)
            pooled_file=None if params is None else pooled/plan['name']/('candidate_'+str(index))/'model.joblib'
            b=base if params is None else make(joblib.load(pooled_file),base)
            save_json(folder/'model.json',dict(rate_file=str(rate_file.resolve()),rate_sha256=sha(rate_file),pooled_file=str(pooled_file.resolve()) if pooled_file else None,pooled_sha256=sha(pooled_file) if pooled_file else None,formula='anchor+rate(cycle)-rate(101)' if params else 'rate(cycle)',train_ids=plan['train'],family=base['family']))
            fn=fade if params is None else predict;validation=[]
            for role,ids in [('training',plan['train']),('validation',plan['validation'])]:
                rows=[];arrays={};control_file=pooled/plan['name']/'candidate_0'/(role+'_predictions.npz')
                with np.load(control_file) as original:
                    for key in ids:
                        c=clean(cells[key]);x,y=labels(cells[key]);p=fn(b,c,x);d=fade(base,c,x)
                        np.testing.assert_array_equal(x,original[key+'__100__cycle']);np.testing.assert_array_equal(y,original[key+'__100__truth']);np.testing.assert_allclose(d,original[key+'__100__prediction'],atol=1e-12,rtol=0)
                        if params is not None:
                            independent=anchor(joblib.load(pooled_file),c)+original[key+'__100__prediction']-fade(joblib.load(rate_file),c,np.array([101.]))[0]
                            np.testing.assert_allclose(p,independent,atol=1e-12,rtol=0)
                        np.testing.assert_allclose(p[0]-p[-1],d[0]-d[-1],atol=1e-12,rtol=0)
                        values=curve_metrics(x,y,p);manual=metric(x,y,p)
                        for field,value in manual.items():np.testing.assert_allclose(value,values[field],atol=1e-10,rtol=0)
                        poison=dict(c,cycle=np.r_[c['cycle'],101],raw=np.vstack([c['raw'],np.full((1,c['raw'].shape[1]),np.inf)]),soh=np.inf,cycle_life=np.inf)
                        np.testing.assert_allclose(p,fn(b,poison,x),atol=1e-12,rtol=0)
                        try:fn(b,dict(c,physical_family='different'),x)
                        except ValueError:pass
                        else:raise AssertionError('family gate failed')
                        rows.append(dict(cell_id=key,**values));arrays.update({key+'__100__'+f:a for f,a in [('cycle',x),('truth',y),('prediction',p)]});checked.append(dict(fold=plan['name'],candidate=index,role=role,cell_id=key,source_predictions_sha256=sha(control_file)))
                save_json(folder/(role+'_metrics.json'),rows);np.savez_compressed(folder/(role+'_predictions.npz'),**arrays)
                if role=='validation':validation=rows
            report=dict(fold=plan['name'],candidate=index,parameters=params,score=selection_score(validation),**{f:float(np.mean([r[f] for r in validation])) for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp')});reports.append(report);save_json(root/'summary.json',reports);logging.info('SAVED %s',report)
    refs=json.loads((reference/'summary.json').read_text());selected=[];primary=[]
    for plan in pcfg['plans']:
        selected.append(min([r for r in reports if r['fold']==plan['name'] and r['candidate']!=0],key=lambda r:r['score']));primary.append(next(r for r in refs if r['fold']==plan['name'] and r['model']=='strong_ridge'))
        save_json(root/plan['name']/'selection.json',dict(selected=selected[-1],used_only_validation=True,outer_loaded=False))
    fields=('score','mape_percent','late_mape','endpoint_pp','drop_error_pp');means={name:{f:float(np.mean([r[f] for r in rr])) for f in fields} for name,rr in [('new',selected),('strong_ridge',primary)]}
    wins=sum(n['score']<o['score'] for n,o in zip(selected,primary));passed=wins>=3 and means['new']['score']<means['strong_ridge']['score'] and all(means['new'][f]<=means['strong_ridge'][f] for f in ('endpoint_pp','drop_error_pp'))
    assert len(checked)==60;save_json(root/'assembly_audit.json',dict(passed=True,checked=checked,drop_preserved=True,source_sha256=sha(__file__),seconds=time.perf_counter()-start,goal_complete=False))
    save_json(root/'screening_decision.json',dict(means=means,selected=selected,better_folds=wins,expand_formal=bool(passed),aggregation='one-seed selected internal validation; not an outer evaluation',goal_complete=False))
    print('OUTPUT='+str(root.resolve()),flush=True);print(json.dumps(dict(means=means,better_folds=wins,expand_formal=bool(passed))),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
