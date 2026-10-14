"""NASA decoupled-anchor inner selection locked before all outer predictions."""
import json,logging,shutil,time
from pathlib import Path
from datetime import datetime,timezone
import joblib,numpy as np
from threadpoolctl import threadpool_limits
from battery_smetimes.pooled_anchor import fit as pooled_fit
from battery_smetimes.fade_rate import fit as rate_fit
from battery_smetimes.decoupled_anchor import make,predict
from battery_smetimes.curve_fidelity import selection_score
from battery_smetimes.tail_boost import labels
from scripts.train_nasa_rate_formal import evaluate
from scripts.train_curve_fidelity import load_subset
from scripts.prepare_same_type_forecast import save_json,sha
from scripts.audit_tail_numeric_gates import metric,aggregate,gates


def main():
    source=Path('experiments/decoupled_anchor/20261005T091119_090689Z');old=json.loads((source/'config.json').read_text())
    assert json.loads((source/'independent_verification.json').read_text())['passed'] and json.loads((source/'screening_decision.json').read_text())['expand_formal']
    plans=old['plans'];params=old['candidates'][1:];seeds=list(range(20261005,20261010));baseline=Path('experiments/nasa_signal_trajectory/ridge/20261004T165029_391663Z')
    root=Path('experiments/decoupled_anchor_formal')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');root.mkdir(parents=True);snap=root/'source_snapshot';snap.mkdir();print('RUN='+str(root.resolve()),flush=True)
    files=[Path(__file__),*[Path('battery_smetimes')/(n+'.py') for n in ('decoupled_anchor','pooled_anchor','fade_rate','conditional_drift','crossfit_drift','tail_boost','forecast','research_baselines')],*[Path('scripts')/(n+'.py') for n in ('train_nasa_rate_formal','train_curve_fidelity','prepare_zhu2022_nca','train_zhu2022_nca')]]
    for f in files:shutil.copy2(f,snap/f.name)
    save_json(root/'config.json',dict(plans=plans,seeds=seeds,candidates=params,prepared_sha256=old['prepared_sha256'],source_run=str(source.resolve()),source_config_sha256=sha(source/'config.json'),baseline=str(baseline.resolve()),baseline_config_sha256=sha(baseline/'config.json'),source_sha256={f.name:sha(f) for f in files},preregistration_sha256=sha('docs/soh_decoupled_anchor_formal_20261005.md'),policy='all inner selections SHA locked before any outer access',development_cv=True))
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(root/'training.log',encoding='utf-8'),logging.StreamHandler()]);start=time.perf_counter();locks=[]
    for plan in plans:
        cells=load_subset(plan['train']+plan['validation']);assert not set(cells)&set(plan['test'])
        for seed in seeds:
            folder=root/plan['name']/str(seed)/'inner';folder.mkdir(parents=True);scores=[]
            rate=rate_fit(cells,plan['train'],False,seed,[dict(cell_id=k) for k in plan['train']])
            for index,p in enumerate(params):
                dest=folder/('candidate_'+str(index));dest.mkdir();logging.info('INNER %s seed=%s candidate=%s',plan['name'],seed,index)
                bundle=make(pooled_fit(cells,plan['train'],seed=seed,**p),rate);joblib.dump(bundle,dest/'model.joblib')
                evaluate(bundle,cells,plan['train'],dest,'training',predictor=predict);rows=evaluate(bundle,cells,plan['validation'],dest,'validation',predictor=predict)
                scores.append(dict(candidate=index,parameters=p,score=selection_score(rows),model_sha256=sha(dest/'model.joblib'),validation_sha256=sha(dest/'validation_predictions.npz')))
            selected=min(scores,key=lambda r:r['score']);save_json(folder/'selection.json',dict(selected=selected,candidates=scores,used_only_validation=True))
            locks.append(dict(fold=plan['name'],seed=seed,selection_file=str((folder/'selection.json').resolve()),selection_sha256=sha(folder/'selection.json'),selected=selected))
    assert len(locks)==20;save_json(root/'inner_lock.json',dict(outer_loaded=False,selections=locks));lock_sha=sha(root/'inner_lock.json');save_json(root/'progress.json',dict(inner_complete=True,inner_lock_sha256=lock_sha,training_complete=False))
    rows=[];old_rows=[]
    for plan in plans:
        ids=plan['train']+plan['validation'];assert not set(ids)&set(plan['test']);cells=load_subset(ids+plan['test'])
        for seed in seeds:
            assert sha(root/'inner_lock.json')==lock_sha;sel=next(r for r in locks if r['fold']==plan['name'] and r['seed']==seed)['selected'];folder=root/plan['name']/str(seed)/'outer';folder.mkdir();logging.info('OUTER %s seed=%s params=%s',plan['name'],seed,sel['parameters'])
            rate=rate_fit(cells,ids,False,seed,[dict(cell_id=k) for k in ids]);bundle=make(pooled_fit(cells,ids,seed=seed,**sel['parameters']),rate);joblib.dump(bundle,folder/'model.joblib')
            evaluate(bundle,cells,ids,folder,'training',predictor=predict);evaluate(bundle,cells,plan['test'],folder,'test',predictor=predict)
            with np.load(folder/'test_predictions.npz') as arrays:
                for key in plan['test']:
                    x,y,p=[arrays[key+'__100__'+f] for f in ('cycle','truth','prediction')];rows.append(dict(cell_id=key,seed=seed,**metric(x,y,p)))
        for seed in range(20261004,20261009):
            with np.load(baseline/plan['name']/str(seed)/'predictions.npz') as arrays:
                for key in plan['test']:
                    x,y,p=[arrays[key+'__100__'+f] for f in ('cycle','truth','prediction')];cx,cy=labels(cells[key]);np.testing.assert_array_equal(x,cx);np.testing.assert_array_equal(y,cy);old_rows.append(dict(cell_id=key,seed=seed,**metric(x,y,p)))
    new,cells=aggregate(rows);old,_=aggregate(old_rows);checks=gates(new,old);save_json(root/'by_cell_seed.json',rows);save_json(root/'by_cell.json',cells)
    save_json(root/'summary.json',dict(new=new,baseline=old,gates=checks,numeric_gates_passed=all(checks.values()),goal_complete=False));save_json(root/'progress.json',dict(training_complete=True,inner_complete=True,inner_lock_sha256=lock_sha,outer_complete=True,seconds=time.perf_counter()-start,goal_complete=False))
    print('OUTPUT='+str(root.resolve()),flush=True);print(json.dumps(dict(mape=new['mape_percent'],endpoint=new['endpoint_pp'],drop=new['drop_error_pp'],joint=new['joint_pass_fraction'],gates=checks)),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
