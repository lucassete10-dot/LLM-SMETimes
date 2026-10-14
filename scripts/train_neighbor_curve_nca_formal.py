"""Frozen inner locks precede any outer load; training-only curve references."""
import json,logging,shutil,time,traceback
from pathlib import Path
from datetime import datetime,timezone
import joblib,numpy as np
from threadpoolctl import threadpool_limits
from battery_smetimes.neighbor_curve import fit as neighbor_fit,predict as neighbor_predict
from battery_smetimes.tail_selection import choose,fit as primary_fit,predict as primary_predict
from battery_smetimes.tail_boost import labels
from battery_smetimes.curve_fidelity import curve_metrics,selection_score
from scripts.train_tail_boost import load_cells
from scripts.train_crossfit_drift_smoke import clean
from scripts.prepare_same_type_forecast import sha,save_json
from scripts.audit_tail_numeric_gates import metric,aggregate,gates

def components(parts,cells,ids,folder,role,exclude_self):
    arrays={}
    for key in ids:
        c,y=labels(cells[key]);query=clean(cells[key]);base=primary_predict(parts['primary'],query,c)
        values={k:neighbor_predict(parts['analog'],query,c,k,exclude=key if exclude_self else None) for k in (1,3)}
        poison=dict(query,truth=np.inf,soh=np.inf,cycle_life=np.inf,early_cycle=np.r_[query['early_cycle'],101],early_raw=np.vstack([query['early_raw'],np.full((1,query['early_raw'].shape[1]),np.inf)]))
        for k in (1,3):
            np.testing.assert_allclose(values[k],neighbor_predict(parts['analog'],poison,c,k,exclude=key if exclude_self else None),atol=1e-12,rtol=0)
        np.testing.assert_allclose(base,primary_predict(parts['primary'],poison,c),atol=1e-12,rtol=0)
        for fn,b in [(neighbor_predict,parts['analog']),(primary_predict,parts['primary'])]:
            try:
                if fn==neighbor_predict:fn(b,dict(query,physical_family='wrong'),c,1)
                else:fn(b,dict(query,physical_family='wrong'),c)
            except ValueError:pass
            else:raise AssertionError('family gate failed')
        arrays.update({key+'__100__'+f:v for f,v in [('cycle',c),('truth',y),('primary',base),('neighbor1',values[1]),('neighbor3',values[3])]})
    np.savez_compressed(folder/(role+'_components.npz'),**arrays);return arrays

def evaluate(arrays,ids,params,folder,role):
    rows=[];out={}
    for key in ids:
        prefix=key+'__100__';c,y,p,q=[arrays[prefix+f] for f in ('cycle','truth','primary','neighbor'+str(params['k']))];forecast=(1-params['weight'])*p+params['weight']*q
        rows.append(dict(cell_id=key,**curve_metrics(c,y,forecast)));out.update({prefix+f:v for f,v in [('cycle',c),('truth',y),('prediction',forecast)]})
    np.savez_compressed(folder/(role+'_predictions.npz'),**out);save_json(folder/(role+'_metrics.json'),rows);return rows

def main():
    screen=Path('experiments/neighbor_curve_nca_fivefold/20261005T141626_356022Z')
    assert json.loads((screen/'independent_verification.json').read_text())['passed'] and json.loads((screen/'independent_screening_verification.json').read_text())['expand_formal']
    route=Path('experiments/tail_boost/20261005T034431_681399Z');cfg=json.loads((route/'config.json').read_text());plans=cfg['plans'];seeds=list(range(20261005,20261010));candidates=json.loads((screen/'config.json').read_text())['candidates']
    current=Path('experiments/tail_boost_outer/20261005T040409_096593Z');baseline=Path(json.loads((current/'config.json').read_text())['baseline'])
    root=Path('experiments/neighbor_curve_nca_formal')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');root.mkdir(parents=True);print('RUN='+str(root.resolve()),flush=True);snapshot=root/'source_snapshot';snapshot.mkdir()
    files=[Path(__file__),Path('docs/soh_neighbor_curve_nca_formal_20261005.md'),*Path('battery_smetimes').glob('*.py'),*[Path('scripts')/(n+'.py') for n in ('train_tail_boost','train_curve_fidelity','train_crossfit_drift_smoke','train_nca_curve_fidelity','prepare_zhu2022_nca','train_zhu2022_nca','prepare_same_type_forecast','audit_tail_numeric_gates')]]
    for f in files:shutil.copy2(f,snapshot/f.name)
    save_json(root/'config.json',dict(dataset='Zhu2022_NCA',plans=plans,seeds=seeds,candidates=candidates,training_source=str(route.resolve()),training_config_sha256=sha(route/'config.json'),screened_run=str(screen.resolve()),screened_config_sha256=sha(screen/'config.json'),baseline=str(baseline.resolve()),current_primary=str(current.resolve()),current_config_sha256=sha(current/'config.json'),prepared_sha256=cfg['prepared_sha256'],source_sha256={str(f):sha(f) for f in files},development_cv=True))
    logging.basicConfig(level=logging.INFO,handlers=[logging.FileHandler(root/'training.log',encoding='utf-8'),logging.StreamHandler()]);start=time.perf_counter();locks=[]
    try:
        for plan in plans:
            cells=load_cells(plan);assert not set(cells)&set(plan['test']);analog=neighbor_fit(cells,plan['train'])
            for seed in seeds:
                folder=root/plan['name']/str(seed)/'inner';folder.mkdir(parents=True);decision=choose(route,cfg,plan,seed,cells);sel=decision['selected'];file=Path(sel['selection_file']).parent/f"candidate_{sel['candidate']}"/'model.joblib';primary=dict(mode=sel['mode'],inner=joblib.load(file),family=analog['family'],train_ids=plan['train']);assert primary['inner']['train_ids']==plan['train'];parts=dict(analog=analog,primary=primary)
                joblib.dump(parts,folder/'components.joblib');save_json(folder/'provenance.json',dict(decision=decision,primary_file=str(file.resolve()),primary_sha256=sha(file)))
                train=components(parts,cells,plan['train'],folder,'training',True);val=components(parts,cells,plan['validation'],folder,'validation',False);scores=[]
                for i,params in enumerate(candidates):
                    dest=folder/f'candidate_{i}';dest.mkdir();evaluate(train,plan['train'],params,dest,'training');rows=evaluate(val,plan['validation'],params,dest,'validation');scores.append(dict(candidate=i,score=selection_score(rows),parameters=params))
                selected=min(scores,key=lambda r:(r['score'],r['candidate']));save_json(folder/'selection.json',dict(selected=selected,candidates=scores,used_only_validation=True));locks.append(dict(fold=plan['name'],seed=seed,selected=selected,primary_choice=sel,components_sha256=sha(folder/'components.joblib'),selection_sha256=sha(folder/'selection.json')))
                save_json(root/'progress.json',dict(inner_complete_count=len(locks),outer_loaded=False,training_complete=False));logging.info('INNER %s seed=%s selected=%s',plan['name'],seed,selected)
        assert len(locks)==25;save_json(root/'inner_lock.json',dict(outer_loaded=False,selections=locks));lock_sha=sha(root/'inner_lock.json');rows=[];oldrows=[];currentrows=[]
        for plan in plans:
            ids=plan['train']+plan['validation'];cells=load_cells(dict(plan,train=ids+plan['test'],validation=[]));analog=neighbor_fit(cells,ids)
            for seed in seeds:
                assert sha(root/'inner_lock.json')==lock_sha;lock=next(r for r in locks if r['fold']==plan['name'] and r['seed']==seed);folder=root/plan['name']/str(seed)/'outer';folder.mkdir();primary=primary_fit(cells,ids,lock['primary_choice'],seed);parts=dict(analog=analog,primary=primary,parameters=lock['selected']['parameters']);joblib.dump(parts,folder/'model.joblib')
                train=components(parts,cells,ids,folder,'training',True);test=components(parts,cells,plan['test'],folder,'test',False);evaluate(train,ids,parts['parameters'],folder,'training');evaluate(test,plan['test'],parts['parameters'],folder,'test')
                with np.load(folder/'test_predictions.npz') as a:
                    for key in plan['test']:rows.append(dict(cell_id=key,seed=seed,**metric(*[a[key+'__100__'+f] for f in ('cycle','truth','prediction')])))
                save_json(root/'progress.json',dict(inner_complete_count=25,outer_complete_count=len(rows)//11,outer_loaded=True,inner_lock_sha256=lock_sha,training_complete=False));logging.info('OUTER %s seed=%s saved',plan['name'],seed)
            for dest,path,ss,name in [(oldrows,baseline,range(20261004,20261009),'predictions.npz'),(currentrows,current,seeds,'test_predictions.npz')]:
                for seed in ss:
                    with np.load(path/plan['name']/str(seed)/name) as a:
                        for key in plan['test']:
                            c,y,p=[a[key+'__100__'+f] for f in ('cycle','truth','prediction')];cx,cy=labels(cells[key]);np.testing.assert_array_equal(c,cx);np.testing.assert_array_equal(y,cy);dest.append(dict(cell_id=key,seed=seed,**metric(c,y,p)))
        new,bycell=aggregate(rows);old,_=aggregate(oldrows);primary,_=aggregate(currentrows);checks=gates(new,old);save_json(root/'by_cell_seed.json',rows);save_json(root/'by_cell.json',bycell);save_json(root/'summary.json',dict(new=new,baseline=old,current_primary=primary,gates=checks,numeric_gates_passed=all(checks.values()),goal_complete=False));save_json(root/'progress.json',dict(training_complete=True,inner_complete_count=25,outer_complete_count=25,inner_lock_sha256=lock_sha,seconds=time.perf_counter()-start,goal_complete=False));print(json.dumps(dict(new=new,gates=checks)),flush=True)
    except BaseException:save_json(root/'failure.json',dict(traceback=traceback.format_exc()));raise

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
