"""Training-only inner CV choices locked before outer evaluation."""
import json,logging,subprocess,sys,time,traceback,shutil
from pathlib import Path
from datetime import datetime,timezone
import numpy as np,joblib
from threadpoolctl import threadpool_limits
from battery_smetimes.neighbor_curve import fit as reference_fit
from battery_smetimes.tail_selection import fit as primary_fit
from scripts.train_tail_boost import load_cells
from scripts.train_neighbor_curve_nca_formal import components,evaluate
from scripts.prepare_same_type_forecast import sha,save_json
from scripts.audit_tail_numeric_gates import metric,aggregate,gates

def baseline_config_path(model_path):
    """Historical predictor subdirectories share their parent run configuration."""
    model_path=Path(model_path)
    for path in (model_path/'config.json',model_path.parent/'config.json'):
        if path.is_file():
            return path
    raise FileNotFoundError(f'No baseline run configuration for {model_path}')

def main():
    source=Path('experiments/neighbor_nested_cv_allfold/20261005T144008_361904Z');assert json.loads((source/'independent_verification.json').read_text())['expand_formal']
    route=Path('experiments/tail_boost/20261005T034431_681399Z');original=json.loads((route/'config.json').read_text());plans=original['plans'];seeds=list(range(20261005,20261010));current=Path('experiments/tail_boost_outer/20261005T040409_096593Z');baseline=Path(json.loads((current/'config.json').read_text())['baseline'])
    root=Path('experiments/neighbor_nested_formal')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');root.mkdir(parents=True);print('RUN='+str(root.resolve()),flush=True);snapshot=root/'source_snapshot';snapshot.mkdir()
    files=[Path(__file__),Path('docs/soh_neighbor_nested_formal_20261005.md'),*Path('battery_smetimes').glob('*.py'),*[Path('scripts')/(n+'.py') for n in ('train_neighbor_nested_cv_seed','verify_neighbor_nested_cv_seed','verify_neighbor_curve_nca_formal','train_neighbor_curve_nca_formal','train_tail_boost','train_curve_fidelity','train_crossfit_drift_smoke','train_nca_curve_fidelity','train_zhu2022_nca','prepare_zhu2022_nca','prepare_same_type_forecast','audit_tail_numeric_gates','audit_rate_blend_metrics')]]
    for f in files:shutil.copy2(f,snapshot/f.name)
    choice=dict(mode='tail',parameters=dict(model='hist_gradient_boosting',max_leaves=15,tail_weight=1,local=False));candidates=[dict(k=1,weight=0.)]+[dict(k=k,weight=w) for k in (1,3) for w in (.25,.5,1.)]
    baseline_config=baseline_config_path(baseline)
    save_json(root/'config.json',dict(dataset='Zhu2022_NCA',plans=plans,seeds=seeds,split_seed=20261005,candidates=candidates,primary_choice=choice,baseline=str(baseline.resolve()),baseline_config_path=str(baseline_config.resolve()),baseline_config_sha256=sha(baseline_config),current_primary=str(current.resolve()),current_config_sha256=sha(current/'config.json'),source_sha256={str(f):sha(f) for f in files},prepared_sha256=original['prepared_sha256'],training_source=str(route.resolve()),training_config_sha256=sha(route/'config.json'),screened_source=str(source.resolve()),screened_sha256=sha(source/'independent_verification.json'),development_cv=True))
    logging.basicConfig(level=logging.INFO,handlers=[logging.FileHandler(root/'training.log',encoding='utf-8'),logging.StreamHandler()]);start=time.perf_counter();locks=[]
    try:
        for i,plan in enumerate(plans):
            for seed in seeds:
                folder=root/plan['name']/str(seed);folder.mkdir(parents=True);logging.info('INNER START %s seed%s',plan['name'],seed)
                process=subprocess.run([sys.executable,'-X','utf8','-m','scripts.train_neighbor_nested_cv_seed','--parent-fold',str(i),'--seed',str(seed)],capture_output=True,text=True,encoding='utf-8');save_json(folder/'inner_training_process.json',dict(exit_code=process.returncode,stdout=process.stdout,stderr=process.stderr))
                if process.returncode:raise RuntimeError(process.stderr)
                paths=[s[4:] for s in process.stdout.splitlines() if s.startswith('RUN=')];assert len(paths)==1;child=Path(paths[0]);verify=subprocess.run([sys.executable,'-X','utf8','-m','scripts.verify_neighbor_nested_cv_seed','--run',str(child)],capture_output=True,text=True,encoding='utf-8');save_json(folder/'inner_verification_process.json',dict(exit_code=verify.returncode,stdout=verify.stdout,stderr=verify.stderr))
                if verify.returncode:raise RuntimeError(verify.stderr)
                proof=json.loads((child/'independent_verification.json').read_text());assert proof['passed'];selection=json.loads((child/'selection.json').read_text());selected=selection['selected'];params=candidates[selected['candidate']]
                locks.append(dict(fold=plan['name'],seed=seed,inner_run=str(child.resolve()),inner_config_sha256=sha(child/'config.json'),verification_sha256=sha(child/'independent_verification.json'),selection_sha256=sha(child/'selection.json'),selected=selected,parameters=params,component_sha256={p.name:sha(child/p.name/'components.joblib') for p in child.iterdir() if p.is_dir() and p.name.startswith('inner')}));save_json(root/'progress.json',dict(inner_complete_count=len(locks),outer_loaded=False,training_complete=False));logging.info('INNER VERIFIED %s seed%s chosen%s',plan['name'],seed,params)
        assert len(locks)==25;save_json(root/'inner_lock.json',dict(outer_loaded=False,selections=locks));lock_sha=sha(root/'inner_lock.json');rows=[];controlrows=[];oldrows=[];currentrows=[];outer_count=0
        for plan in plans:
            ids=plan['train']+plan['validation'];cells=load_cells(dict(plan,train=ids+plan['test'],validation=[]));reference=reference_fit(cells,ids)
            for seed in seeds:
                assert sha(root/'inner_lock.json')==lock_sha;lock=next(r for r in locks if r['fold']==plan['name'] and r['seed']==seed);folder=root/plan['name']/str(seed)/'outer';folder.mkdir();parts=dict(analog=reference,primary=primary_fit(cells,ids,choice,seed),parameters=lock['parameters']);joblib.dump(parts,folder/'model.joblib');tr=components(parts,cells,ids,folder,'training',True);te=components(parts,cells,plan['test'],folder,'test',False);evaluate(tr,ids,parts['parameters'],folder,'training');evaluate(te,plan['test'],parts['parameters'],folder,'test')
                with np.load(folder/'test_predictions.npz') as a:
                    for key in plan['test']:
                        c,y,p=[a[key+'__100__'+f] for f in ('cycle','truth','prediction')];rows.append(dict(cell_id=key,seed=seed,**metric(c,y,p)));controlrows.append(dict(cell_id=key,seed=seed,**metric(c,y,te[key+'__100__primary'])))
                outer_count+=1;save_json(root/'progress.json',dict(inner_complete_count=25,outer_complete_count=outer_count,outer_loaded=True,inner_lock_sha256=lock_sha,training_complete=False));logging.info('OUTER SAVED %s seed%s',plan['name'],seed)
            for dest,path,ss,suffix in [(oldrows,baseline,range(20261004,20261009),'predictions.npz'),(currentrows,current,seeds,'test_predictions.npz')]:
                for seed in ss:
                    with np.load(path/plan['name']/str(seed)/suffix) as a:
                        for key in plan['test']:
                            c,y,p=[a[key+'__100__'+f] for f in ('cycle','truth','prediction')];from battery_smetimes.tail_boost import labels
                            cx,cy=labels(cells[key]);np.testing.assert_array_equal(c,cx);np.testing.assert_array_equal(y,cy);dest.append(dict(cell_id=key,seed=seed,**metric(c,y,p)))
        new,bycell=aggregate(rows);control,_=aggregate(controlrows);old,_=aggregate(oldrows);primary,_=aggregate(currentrows);check=gates(new,old);save_json(root/'by_cell_seed.json',rows);save_json(root/'by_cell.json',bycell);save_json(root/'summary.json',dict(new=new,fixed_hgb_control=control,baseline=old,current_primary=primary,gates=check,numeric_gates_passed=all(check.values()),goal_complete=False));save_json(root/'progress.json',dict(training_complete=True,inner_complete_count=25,outer_complete_count=25,inner_lock_sha256=lock_sha,seconds=time.perf_counter()-start,goal_complete=False));print(json.dumps(dict(new=new,gates=check)),flush=True)
    except BaseException:save_json(root/'failure.json',dict(traceback=traceback.format_exc()));raise

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
