"""MIT five-seed inner selection locks before any outer access."""
import json,logging,shutil,time,traceback
from pathlib import Path
from datetime import datetime,timezone
import joblib,numpy as np
from threadpoolctl import threadpool_limits
from battery_smetimes.rate_blend import make
from battery_smetimes.fade_rate import fit as rate_fit,predict as rate_predict
from battery_smetimes.tail_selection import choose,fit as primary_fit,predict as primary_predict
from battery_smetimes.warm_monotone import derive_prior
from battery_smetimes.tail_boost import labels
from battery_smetimes.curve_fidelity import curve_metrics,selection_score
from scripts.train_curve_fidelity import load_subset
from scripts.train_crossfit_drift_smoke import clean
from scripts.prepare_same_type_forecast import save_json,sha
from scripts.audit_tail_numeric_gates import metric,aggregate,gates


def components(parts,cells,ids,folder,role):
    arrays={}
    for key in ids:
        cell=clean(cells[key]);cycle,truth=labels(cells[key]);p=primary_predict(parts['primary'],cell,cycle);q=rate_predict(parts['rate'],cell,cycle)
        poison=dict(cell,cycle=np.r_[cell['cycle'],101],raw=np.vstack([cell['raw'],np.full((1,cell['raw'].shape[1]),np.inf)]),soh=np.inf,cycle_life=np.inf)
        for fn,b,value in [(primary_predict,parts['primary'],p),(rate_predict,parts['rate'],q)]:
            np.testing.assert_allclose(value,fn(b,poison,cycle),atol=1e-12,rtol=0)
            try:fn(b,dict(cell,physical_family='different'),cycle)
            except ValueError:pass
            else:raise AssertionError('family gate failed')
        arrays.update({key+'__100__'+f:a for f,a in [('cycle',cycle),('truth',truth),('primary',p),('rate',q)]})
    np.savez_compressed(folder/(role+'_components.npz'),**arrays);return arrays


def evaluate(arrays,ids,weight,folder,role):
    rows=[];output={}
    for key in ids:
        cycle,truth,p,q=[arrays[key+'__100__'+f] for f in ('cycle','truth','primary','rate')];prediction=(1-weight)*p+weight*q
        rows.append(dict(cell_id=key,**curve_metrics(cycle,truth,prediction)));output.update({key+'__100__'+f:a for f,a in [('cycle',cycle),('truth',truth),('prediction',prediction)]})
    save_json(folder/(role+'_metrics.json'),rows);np.savez_compressed(folder/(role+'_predictions.npz'),**output);return rows


def main():
    screened=Path('experiments/rate_blend_mit/20261005T093421_556509Z');scfg=json.loads((screened/'config.json').read_text())
    assert json.loads((screened/'independent_verification.json').read_text())['passed'] and json.loads((screened/'screening_decision.json').read_text())['expand_formal']
    route=Path(scfg['primary_run']);pcfg=json.loads((route/'config.json').read_text());plans=scfg['plans'];seeds=list(range(20261005,20261010));weights=scfg['weights']
    baseline=Path('experiments/stm32_compact_trees/runs/20261004T074429_087457Z');current=Path('experiments/tail_boost_outer/20261005T041231_340759Z')
    root=Path('experiments/rate_blend_formal')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');root.mkdir(parents=True);snap=root/'source_snapshot';snap.mkdir();print('RUN='+str(root.resolve()),flush=True)
    files=[Path(__file__),*[Path('battery_smetimes')/(n+'.py') for n in ('rate_blend','fade_rate','warm_monotone','monotone_network','conditional_drift','crossfit_drift','tail_selection','tail_boost','curve_fidelity','forecast','research_baselines','physical_types')],*[Path('scripts')/(n+'.py') for n in ('train_tail_boost','train_curve_fidelity','train_crossfit_drift_smoke','train_nca_curve_fidelity','prepare_zhu2022_nca','train_zhu2022_nca')]]
    for f in files:shutil.copy2(f,snap/f.name)
    save_json(root/'config.json',dict(dataset='MIT',plans=plans,seeds=seeds,weights=weights,prepared_sha256=scfg['prepared_sha256'],screened_run=str(screened.resolve()),screened_config_sha256=sha(screened/'config.json'),primary_run=str(route.resolve()),primary_config_sha256=sha(route/'config.json'),baseline=str(baseline.resolve()),baseline_config_sha256=sha(baseline/'config.json'),current_primary=str(current.resolve()),current_primary_config_sha256=sha(current/'config.json'),source_sha256={f.name:sha(f) for f in files},preregistration_sha256=sha('docs/soh_rate_blend_formal_20261005.md'),policy='all 25 inner selections locked before any outer cell load',development_cv=True))
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(root/'training.log',encoding='utf-8'),logging.StreamHandler()]);start=time.perf_counter();locks=[]
    try:
        for plan in plans:
            cells=load_subset(plan['train']+plan['validation']);assert not set(cells)&set(plan['test']);params=derive_prior(cells,plan['train'])[1]
            for seed in seeds:
                folder=root/plan['name']/str(seed)/'inner';folder.mkdir(parents=True);logging.info('INNER %s seed=%s',plan['name'],seed)
                decision=choose(route,pcfg,plan,seed,cells);sel=decision['selected'];pf=Path(sel['selection_file']).parent/f"candidate_{sel['candidate']}"/'model.joblib';rate=rate_fit(cells,plan['train'],True,seed,params)
                primary=dict(mode=sel['mode'],inner=joblib.load(pf),family=rate['family'],train_ids=plan['train']);assert primary['inner']['train_ids']==plan['train'];parts=dict(primary=primary,rate=rate);make(primary,rate,0)
                joblib.dump(parts,folder/'components.joblib');save_json(folder/'training_parameters.json',params);save_json(folder/'provenance.json',dict(decision=decision,primary_file=str(pf.resolve()),primary_sha256=sha(pf)))
                train=components(parts,cells,plan['train'],folder,'training');val=components(parts,cells,plan['validation'],folder,'validation');scores=[]
                for index,w in enumerate(weights):
                    dest=folder/f'candidate_{index}';dest.mkdir();evaluate(train,plan['train'],w,dest,'training');rr=evaluate(val,plan['validation'],w,dest,'validation')
                    scores.append(dict(candidate=index,weight=w,score=selection_score(rr),validation_sha256=sha(dest/'validation_predictions.npz')))
                selected=min(scores,key=lambda r:(r['score'],r['weight']));save_json(folder/'selection.json',dict(selected=selected,candidates=scores,used_only_validation=True))
                locks.append(dict(fold=plan['name'],seed=seed,selection_file=str((folder/'selection.json').resolve()),selection_sha256=sha(folder/'selection.json'),components_sha256=sha(folder/'components.joblib'),parameter_sha256=sha(folder/'training_parameters.json'),selected=selected,primary_choice=sel))
                save_json(root/'progress.json',dict(inner_models_complete=len(locks),training_complete=False,outer_loaded=False));logging.info('INNER SAVED weight=%s score=%s',selected['weight'],selected['score'])
        assert len(locks)==25;save_json(root/'inner_lock.json',dict(outer_loaded=False,selections=locks));lock_sha=sha(root/'inner_lock.json');save_json(root/'progress.json',dict(inner_complete=True,inner_lock_sha256=lock_sha,training_complete=False))
        rows=[];old_rows=[];current_rows=[]
        for plan in plans:
            ids=plan['train']+plan['validation'];cells=load_subset(ids+plan['test']);params=derive_prior(cells,ids)[1]
            for seed in seeds:
                assert sha(root/'inner_lock.json')==lock_sha;lock=next(r for r in locks if r['fold']==plan['name'] and r['seed']==seed);folder=root/plan['name']/str(seed)/'outer';folder.mkdir();logging.info('OUTER %s seed=%s weight=%s',plan['name'],seed,lock['selected']['weight'])
                primary=primary_fit(cells,ids,lock['primary_choice'],seed);rate=rate_fit(cells,ids,True,seed,params);b=make(primary,rate,lock['selected']['weight']);joblib.dump(b,folder/'model.joblib');save_json(folder/'training_parameters.json',params)
                train=components(b,cells,ids,folder,'training');test=components(b,cells,plan['test'],folder,'test');evaluate(train,ids,b['weight'],folder,'training');evaluate(test,plan['test'],b['weight'],folder,'test')
                for key in plan['test']:
                    x,y,p,q=[test[key+'__100__'+f] for f in ('cycle','truth','primary','rate')];rows.append(dict(cell_id=key,seed=seed,**metric(x,y,(1-b['weight'])*p+b['weight']*q)))
                logging.info('OUTER SAVED %s seed=%s',plan['name'],seed)
            for destination,path,seeds_old,name in [(old_rows,baseline,range(20261004,20261009),'test/predictions.npz'),(current_rows,current,seeds,'test_predictions.npz')]:
                for seed in seeds_old:
                    with np.load(path/plan['name']/str(seed)/name) as a:
                        for key in plan['test']:
                            x,y,p=[a[key+'__100__'+f] for f in ('cycle','truth','prediction')];cx,cy=labels(cells[key]);np.testing.assert_array_equal(x,cx);np.testing.assert_array_equal(y,cy);destination.append(dict(cell_id=key,seed=seed,**metric(x,y,p)))
        new,cell_rows=aggregate(rows);old,_=aggregate(old_rows);primary,_=aggregate(current_rows);checks=gates(new,old)
        save_json(root/'by_cell_seed.json',rows);save_json(root/'by_cell.json',cell_rows);save_json(root/'summary.json',dict(new=new,baseline=old,current_primary=primary,gates=checks,numeric_gates_passed=all(checks.values()),goal_complete=False))
        save_json(root/'progress.json',dict(training_complete=True,inner_complete=True,inner_lock_sha256=lock_sha,outer_complete=True,seconds=time.perf_counter()-start,goal_complete=False));print('OUTPUT='+str(root.resolve()),flush=True);print(json.dumps(dict(mape=new['mape_percent'],endpoint=new['endpoint_pp'],drop=new['drop_error_pp'],joint=new['joint_pass_fraction'],gates=checks)),flush=True)
    except BaseException:save_json(root/'failure.json',dict(traceback=traceback.format_exc()));raise


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
