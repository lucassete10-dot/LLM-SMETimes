"""Lock 25 inner choices before any outer load, refit and save every curve."""
import argparse,json,logging,shutil,time,traceback
from pathlib import Path
from datetime import datetime,timezone
import joblib,numpy as np
from threadpoolctl import threadpool_limits
from battery_smetimes.xjtu_segment_features import fit as segment_fit,predict as segment_predict
from battery_smetimes.tail_selection import fit as legacy_fit,predict as legacy_predict
from battery_smetimes.curve_fidelity import targets
from scripts.train_curve_fidelity import load_subset
from scripts.train_crossfit_drift_smoke import clean
from scripts.train_nasa_robust_inner import detail
from scripts.prepare_same_type_forecast import save_json,sha
from scripts.audit_tail_numeric_gates import aggregate,gates

BASELINE=Path('experiments/stm32_compact_trees/runs/20261004T074429_087457Z')
PRIMARY=Path('experiments/tail_boost_outer/20261005T035307_281056Z')

def predict(bundle,cell,future):
    return segment_predict(bundle['inner'],cell,future) if bundle['mode']=='segment' else legacy_predict(bundle,cell,future)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--inner',type=Path,required=True);source=ap.parse_args().inner
    cfg=json.loads((source/'config.json').read_text());proof=json.loads((source/'independent_verification.json').read_text());assert proof['passed'] and proof['verified_tasks']==25 and proof['outer_loaded'] is False
    assert proof['source_sha256']==sha('scripts/verify_xjtu_segment_full.py') and proof['child_verifier_sha256']==sha('scripts/verify_xjtu_segment_fold_state.py')
    for f,d in cfg['source_sha256'].items():assert sha(f)==d==sha(source/'source_snapshot'/Path(f).name)
    records=[]
    for t in cfg['tasks']:
        child=Path(t['run']);ccfg=json.loads((child/'config.json').read_text());sel=json.loads((child/'selection.json').read_text());cproof=json.loads((child/'independent_verification.json').read_text());assert cproof['passed'] and cproof['outer_loaded'] is False
        stored=next(r for r in proof['records'] if r['fold']==t['fold'] and r['seed']==t['seed']);assert stored['selection_sha256']==sha(child/'selection.json') and stored['proof_sha256']==sha(child/'independent_verification.json')
        records.append(dict(fold=t['fold'],seed=t['seed'],choice=sel['selected'],legacy_choice=ccfg['control_choice']['selected'],selection_sha256=sha(child/'selection.json'),config_sha256=sha(child/'config.json'),proof_sha256=sha(child/'independent_verification.json')))
    root=Path('experiments/xjtu_segment_outer')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');root.mkdir(parents=True);print('RUN='+str(root.resolve()),flush=True);snap=root/'source_snapshot';snap.mkdir()
    files=[Path(__file__),Path('battery_smetimes/xjtu_segment_features.py'),Path('battery_smetimes/tail_selection.py'),Path('battery_smetimes/tail_boost.py'),Path('scripts/train_curve_fidelity.py'),Path('scripts/train_crossfit_drift_smoke.py'),Path('scripts/train_nasa_robust_inner.py'),Path('scripts/audit_tail_numeric_gates.py'),Path('docs/soh_xjtu_segment_outer_20261006.md')]
    for f in files:shutil.copy2(f,snap/f.name)
    save_json(root/'choice_lock.json',dict(records=records,locked_at_utc=datetime.now(timezone.utc).isoformat(),outer_loaded=False))
    save_json(root/'config.json',dict(dataset='XJTU',plans=cfg['plans'],seeds=cfg['seeds'],inner=str(source.resolve()),inner_config_sha256=sha(source/'config.json'),inner_proof_sha256=sha(source/'independent_verification.json'),choice_lock_sha256=sha(root/'choice_lock.json'),source_sha256={str(f):sha(f) for f in files},baseline=str(BASELINE.resolve()),baseline_config_sha256=sha(BASELINE/'config.json'),current_primary=str(PRIMARY.resolve()),current_primary_config_sha256=sha(PRIMARY/'config.json'),not_blind=True,origin=100))
    logging.basicConfig(level=logging.INFO,handlers=[logging.FileHandler(root/'training.log',encoding='utf-8'),logging.StreamHandler()]);allrows=[];outer=[];start=time.perf_counter();done=[]
    try:
        for plan in cfg['plans']:
            ids=plan['train']+plan['validation'];assert not set(ids)&set(plan['test'])
            for seed in cfg['seeds']:
                rec=next(r for r in records if r['fold']==plan['name'] and r['seed']==seed);train=load_subset(ids);out=root/plan['name']/str(seed);out.mkdir(parents=True);started=datetime.now(timezone.utc).isoformat()
                if rec['choice']['candidate']:
                    params=rec['choice']['parameters'];bundle=dict(mode='segment',inner=segment_fit(train,ids,params,seed),family=train[ids[0]]['physical_family'],train_ids=ids,parameters=params)
                else:bundle=legacy_fit(train,ids,rec['legacy_choice'],seed)
                joblib.dump(bundle,out/'model.joblib');finished=datetime.now(timezone.utc).isoformat();save_json(out/'model_timing.json',dict(fit_started_utc=started,fit_finished_utc=finished,model_saved_before_test=True))
                # The model is fixed and saved before loading the held-out cells.
                test=load_subset(plan['test']);loaded=datetime.now(timezone.utc).isoformat();save_json(out/'evaluation_timing.json',dict(test_loaded_at_utc=loaded));assert finished<=loaded
                cells=dict(train,**test);save_json(out/'data_sha256.json',{k:c['prepared_sha256'] for k,c in cells.items()})
                for role,members in [('training',plan['train']),('validation',plan['validation']),('test',plan['test'])]:
                    arrays={};rows=[]
                    for key in members:
                        c,y=targets(cells[key]);p=predict(bundle,clean(cells[key]),c);r=dict(cell_id=key,fold=plan['name'],seed=seed,role=role,**detail(c,y,p),start_error_pp=float(100*abs(p[0]-y[0])))
                        allrows.append(r);rows.append(r);arrays.update({key+'__100__'+f:v for f,v in [('cycle',c),('truth',y),('prediction',p)]})
                        if role=='test':outer.append(r)
                    save_json(out/(role+'_metrics.json'),rows);np.savez_compressed(out/(role+'_predictions.npz'),**arrays)
                done.append(dict(fold=plan['name'],seed=seed));save_json(root/'progress.json',dict(training_complete=len(done)==25,tasks=done,seconds=time.perf_counter()-start,curves=len(allrows),goal_complete=False));logging.info('SAVED %s seed %s outer %s',plan['name'],seed,len(outer))
        summaries={};by_cell=None
        new,by_cell=aggregate(outer);summaries['new']=new
        for name,path,seeds in [('baseline',BASELINE,range(20261004,20261009)),('current_primary',PRIMARY,cfg['seeds'])]:
            rr=[]
            for plan in cfg['plans']:
                for seed in seeds:
                    file=path/plan['name']/str(seed)/('test/predictions.npz' if name=='baseline' else 'test_predictions.npz')
                    with np.load(file) as a:
                        for key in plan['test']:
                            c,y,p=[a[key+'__100__'+f] for f in ('cycle','truth','prediction')];rr.append(dict(cell_id=key,seed=seed,**detail(c,y,p)))
            summaries[name]=aggregate(rr)[0]
        checks=gates(new,summaries['baseline']);save_json(root/'summary.json',dict(**summaries,gates=checks,numeric_gates_passed=all(checks.values()),goal_complete=False));save_json(root/'by_cell.json',by_cell);save_json(root/'comprehensive_curve_metrics.json',dict(rows=allrows,count=len(allrows)));save_json(root/'failed_cells.json',[c for c in by_cell if not c['joint_pass'] or c['severe_flat_failure']]);print(json.dumps(dict(new=new,gates=checks)),flush=True)
    except BaseException:save_json(root/'failure.json',dict(traceback=traceback.format_exc()));raise

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
