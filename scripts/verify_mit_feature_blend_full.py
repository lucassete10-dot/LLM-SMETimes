"""All-task isolated audits and preregistered inner expansion decision."""
import argparse,json,subprocess,sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
from scripts.prepare_same_type_forecast import sha,save_json

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);root=ap.parse_args().run
    cfg=json.loads((root/'config.json').read_text());p=json.loads((root/'progress.json').read_text())
    assert p['training_complete'] and p['completed']==25 and all(t['exit_code']==0 for t in p['tasks'])
    assert cfg['seeds']==list(range(20261005,20261010)) and len(cfg['tasks'])==25 and not cfg['outer_loaded']
    for f,d in cfg['source_sha256'].items():assert sha(f)==d==sha(root/'source_snapshot'/Path(f).name)
    def verify(task):
        log=root/(task['fold']+'_'+str(task['seed'])+'_verification.log')
        with log.open('w',encoding='utf-8') as h:
            result=subprocess.run([sys.executable,'-X','utf8','-m','scripts.verify_mit_feature_blend_fold','--run',task['run']],stdout=h,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        return dict(**task,exit_code=result.returncode,verification_log=str(log.resolve()))
    start=time.perf_counter();done=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(verify,t) for t in cfg['tasks']]
        for f in as_completed(futures):
            r=f.result();done.append(r);save_json(root/'verification_progress.json',dict(complete=len(done)==25,tasks=done,seconds=time.perf_counter()-start,goal_complete=False))
            print('VERIFIED '+r['fold']+' '+str(r['seed'])+' exit '+str(r['exit_code'])+' '+str(len(done))+'/25',flush=True)
    assert all(r['exit_code']==0 for r in done),'Independent audit failed; inspect logs'
    records=[];curves=fits=0
    for t in cfg['tasks']:
        child=Path(t['run']);cc=json.loads((child/'config.json').read_text());proof=json.loads((child/'independent_verification.json').read_text());sel=json.loads((child/'selection.json').read_text())
        assert proof['passed'] and not proof['outer_loaded'] and cc['seed']==t['seed']
        assert cc['plan']==next(p for p in cfg['plans'] if p['name']==t['fold'])
        curves+=proof['curves'];fits+=proof['independent_component_refits']
        records.append(dict(fold=t['fold'],seed=t['seed'],selected=sel['selected'],control=sel['control'],selection_sha256=sha(child/'selection.json'),proof_sha256=sha(child/'independent_verification.json')))
    fields=('score','mape_percent','late_mape','endpoint_pp','drop_error_pp','start_error_pp')
    summary={name:{f:dict(mean=float(np.mean([r[name][f] for r in records])),sd=float(np.std([r[name][f] for r in records],ddof=1))) for f in fields} for name in ('selected','control')}
    folds=[]
    for p in cfg['plans']:
        rr=[r for r in records if r['fold']==p['name']];assert len(rr)==5
        folds.append(dict(fold=p['name'],selected_score=float(np.mean([r['selected']['score'] for r in rr])),control_score=float(np.mean([r['control']['score'] for r in rr]))))
    wins=sum(f['selected_score']<f['control_score'] for f in folds);new,old=summary['selected'],summary['control']
    expand=wins>=3 and new['mape_percent']['mean']<=old['mape_percent']['mean']+.1 and new['endpoint_pp']['mean']<=old['endpoint_pp']['mean']+1e-10 and new['drop_error_pp']['mean']<=old['drop_error_pp']['mean']+1e-10
    save_json(root/'independent_verification.json',dict(passed=True,verified_tasks=25,curves=curves,independent_component_refits=fits,records=records,inner_descriptive_summary=summary,folds=folds,fold_score_wins=wins,expand_outer=expand,
        source_sha256=sha(__file__),fold_verifier_sha256=sha('scripts/verify_mit_feature_blend_fold.py'),state_verifier_sha256=sha('scripts/verify_model_state_portable.py'),not_independent_cell_test_samples=True,outer_loaded=False,goal_complete=False))
    print(json.dumps(dict(passed=True,tasks=25,curves=curves,refits=fits,wins=wins,expand=expand,summary=summary)),flush=True)

if __name__=='__main__':main()
