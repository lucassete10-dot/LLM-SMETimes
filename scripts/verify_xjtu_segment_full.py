"""Independent per-fold refits before any outer lock/evaluation."""
import argparse,json,subprocess,sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
from scripts.prepare_same_type_forecast import save_json,sha

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);root=ap.parse_args().run
    cfg=json.loads((root/'config.json').read_text());progress=json.loads((root/'progress.json').read_text())
    assert progress['training_complete'] and progress['completed']==25 and all(t['exit_code']==0 for t in progress['tasks'])
    assert cfg['seeds']==list(range(20261005,20261010)) and len(cfg['plans'])==5 and len(cfg['tasks'])==25
    for f,d in cfg['source_sha256'].items():assert sha(f)==d==sha(root/'source_snapshot'/Path(f).name)
    def verify(task):
        log=root/(task['fold']+'_'+str(task['seed'])+'_verification.log')
        with log.open('w',encoding='utf-8') as handle:
            result=subprocess.run([sys.executable,'-X','utf8','-m','scripts.verify_xjtu_segment_fold_state','--run',task['run']],stdout=handle,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        return dict(**task,exit_code=result.returncode,verification_log=str(log.resolve()))
    started=time.perf_counter();done=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(verify,t) for t in cfg['tasks']]
        for f in as_completed(futures):
            r=f.result();done.append(r);save_json(root/'verification_progress.json',dict(complete=len(done)==25,tasks=done,seconds=time.perf_counter()-started,goal_complete=False));print('VERIFIED '+r['fold']+' '+str(r['seed'])+' exit '+str(r['exit_code'])+' '+str(len(done))+'/25',flush=True)
    assert all(r['exit_code']==0 for r in done),'Independent verifier failed; inspect verification logs'
    records=[];curves=fits=0;tasks=cfg['tasks']
    for t in tasks:
        child=Path(t['run']);ccfg=json.loads((child/'config.json').read_text());proof=json.loads((child/'independent_verification.json').read_text());sel=json.loads((child/'selection.json').read_text());assert proof['passed'] and proof['outer_loaded'] is False
        assert ccfg['seed']==t['seed'] and ccfg['plan']==next(p for p in cfg['plans'] if p['name']==t['fold'])
        curves+=proof['curves'];fits+=proof['independent_component_refits'];records.append(dict(fold=t['fold'],seed=t['seed'],selected=sel['selected'],control=sel['control'],selection_sha256=sha(child/'selection.json'),proof_sha256=sha(child/'independent_verification.json')))
    fields=('score','mape_percent','late_mape','endpoint_pp','drop_error_pp','start_error_pp')
    # Fold observations overlap in validation membership; these are descriptive
    # inner comparisons, not 25 independent test samples or cohort accuracy.
    summary={name:{f:dict(mean=float(np.mean([r[name][f] for r in records])),sd=float(np.std([r[name][f] for r in records],ddof=1))) for f in fields} for name in ('selected','control')}
    save_json(root/'independent_verification.json',dict(passed=True,verified_tasks=25,curves=curves,independent_component_refits=fits,records=records,inner_descriptive_summary=summary,not_outer_results=True,source_sha256=sha(__file__),child_verifier_sha256=sha('scripts/verify_xjtu_segment_fold_state.py'),outer_loaded=False,goal_complete=False))
    print(json.dumps(dict(passed=True,tasks=25,curves=curves,component_refits=fits,summary=summary)),flush=True)

if __name__=='__main__':main()
