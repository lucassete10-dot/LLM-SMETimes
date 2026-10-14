"""Two isolated subprocesses; do not read outer cells during inner expansion."""
import json,subprocess,sys,shutil,time,traceback
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor,as_completed
from scripts.prepare_same_type_forecast import sha,save_json

def main():
    screen=Path('experiments/mit_feature_blend/20261006T142639_303287Z')
    assert json.loads((screen/'independent_verification.json').read_text())['passed'] and json.loads((screen/'selection.json').read_text())['expand']
    parent=Path('experiments/tail_boost/20261005T034441_870182Z');pcfg=json.loads((parent/'config.json').read_text());plans=pcfg['plans'];seeds=list(range(20261005,20261010))
    assert len(plans)==5 and all(p['dataset']=='MIT' for p in plans)
    root=Path('experiments/mit_feature_blend_full')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');root.mkdir(parents=True);print('RUN='+str(root.resolve()),flush=True);snap=root/'source_snapshot';snap.mkdir()
    files=[Path(__file__),Path('scripts/train_mit_feature_blend_fold.py'),Path('battery_smetimes/mit_segment_features.py'),Path('docs/soh_mit_feature_blend_full_20261006.md')]
    for f in files:shutil.copy2(f,snap/f.name)
    tasks=[dict(fold=p['name'],seed=s,run=str((root/p['name']/str(s)).resolve())) for p in plans for s in seeds]
    save_json(root/'config.json',dict(plans=plans,seeds=seeds,tasks=tasks,source_sha256={str(f):sha(f) for f in files},parent=str(parent.resolve()),parent_config_sha256=sha(parent/'config.json'),screen=str(screen.resolve()),screen_proof_sha256=sha(screen/'independent_verification.json'),max_workers=2,outer_loaded=False,goal_complete=False))
    started=time.perf_counter();done=[]
    def train(task):
        log=root/(task['fold']+'_'+str(task['seed'])+'_process.log')
        with log.open('w',encoding='utf-8') as f:
            result=subprocess.run([sys.executable,'-X','utf8','-m','scripts.train_mit_feature_blend_fold','--fold',task['fold'],'--seed',str(task['seed']),'--output',task['run']],stdout=f,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        return dict(**task,exit_code=result.returncode,process_log=str(log.resolve()))
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(train,t) for t in tasks]
            for f in as_completed(futures):
                r=f.result();done.append(r);save_json(root/'progress.json',dict(training_complete=len(done)==25,completed=len(done),tasks=done,seconds=time.perf_counter()-started,outer_loaded=False,goal_complete=False))
                print('FINISHED '+r['fold']+' seed '+str(r['seed'])+' exit '+str(r['exit_code'])+' '+str(len(done))+'/25',flush=True)
                if r['exit_code']!=0:raise RuntimeError('child failure; inspect '+r['process_log'])
        choices=[]
        for t in tasks:
            folder=Path(t['run']);s=json.loads((folder/'selection.json').read_text());choices.append(dict(fold=t['fold'],seed=t['seed'],selected=s['selected'],control=s['control'],selection_sha256=sha(folder/'selection.json')))
        save_json(root/'training_summary.json',dict(choices=choices,independent_verification_pending=True,outer_loaded=False,goal_complete=False))
        print('TRAINING_COMPLETE '+str(root.resolve()),flush=True)
    except BaseException:save_json(root/'failure.json',dict(traceback=traceback.format_exc(),finished_tasks=done));raise

if __name__=='__main__':main()
