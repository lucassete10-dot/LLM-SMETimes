"""Two-process isolated-fold expansion, preserving every child log/output."""
import json,subprocess,sys,shutil,time,traceback
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor,as_completed
from scripts.prepare_same_type_forecast import sha,save_json

def main():
    smoke=Path('experiments/xjtu_segment_features/20261006T132030_214341Z')
    assert json.loads((smoke/'independent_verification.json').read_text())['passed']
    assert json.loads((smoke/'selection.json').read_text())['expand']
    parent=Path('experiments/tail_boost/20261005T034452_066902Z');pcfg=json.loads((parent/'config.json').read_text());plans=pcfg['plans'];seeds=list(range(20261005,20261010))
    assert len(plans)==5 and all(p['dataset']=='XJTU' for p in plans)
    root=Path('experiments/xjtu_segment_full')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');root.mkdir(parents=True);print('RUN='+str(root.resolve()),flush=True);snap=root/'source_snapshot';snap.mkdir()
    files=[Path(__file__),Path('scripts/train_xjtu_segment_fold.py'),Path('scripts/verify_xjtu_segment_fold.py'),Path('battery_smetimes/xjtu_segment_features.py'),Path('docs/soh_xjtu_segment_full_20261006.md')]
    for f in files:shutil.copy2(f,snap/f.name)
    tasks=[dict(fold=p['name'],seed=s,run=str((root/p['name']/str(s)).resolve())) for p in plans for s in seeds]
    save_json(root/'config.json',dict(plans=plans,seeds=seeds,tasks=tasks,source_sha256={str(f):sha(f) for f in files},parent=str(parent.resolve()),parent_config_sha256=sha(parent/'config.json'),smoke=str(smoke.resolve()),smoke_proof_sha256=sha(smoke/'independent_verification.json'),max_workers=2,outer_loaded=False,goal_complete=False))
    started=time.perf_counter();done=[]
    def train(task):
        log=root/(task['fold']+'_'+str(task['seed'])+'_process.log')
        cmd=[sys.executable,'-X','utf8','-m','scripts.train_xjtu_segment_fold','--fold',task['fold'],'--seed',str(task['seed']),'--output',task['run']]
        with log.open('w',encoding='utf-8') as handle:
            result=subprocess.run(cmd,stdout=handle,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        return dict(**task,exit_code=result.returncode,process_log=str(log.resolve()))
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(train,t) for t in tasks]
            for f in as_completed(futures):
                r=f.result();done.append(r);save_json(root/'progress.json',dict(training_complete=len(done)==25,completed=len(done),tasks=done,seconds=time.perf_counter()-started,outer_loaded=False,goal_complete=False));print('FINISHED '+r['fold']+' seed '+str(r['seed'])+' exit '+str(r['exit_code'])+' '+str(len(done))+'/25',flush=True)
                if r['exit_code']!=0:raise RuntimeError('child training failed; inspect '+r['process_log'])
        choices=[]
        for task in tasks:
            run=Path(task['run']);selection=json.loads((run/'selection.json').read_text());choices.append(dict(fold=task['fold'],seed=task['seed'],selected=selection['selected'],control=selection['control'],selection_sha256=sha(run/'selection.json')))
        save_json(root/'training_summary.json',dict(choices=choices,independent_verification_pending=True,outer_loaded=False,goal_complete=False));print('TRAINING_COMPLETE '+str(root.resolve()),flush=True)
    except BaseException:save_json(root/'failure.json',dict(traceback=traceback.format_exc(),finished_tasks=done));raise

if __name__=='__main__':main()
