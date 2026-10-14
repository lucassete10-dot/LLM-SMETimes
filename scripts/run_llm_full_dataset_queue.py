"""Durable sequential GPU queue with persisted, MVP-first task ordering.

Only validation selects checkpoints. Accuracy failures do not delete cells or
stop later datasets; execution/isolation failures stop the queue for repair.
"""
import argparse,json,os,subprocess,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
from scripts.prepare_same_type_forecast import save_json
from scripts.audit_llm_curve_repair import fingerprint

BASE=Path('configs/llm_curve_full_v1.json')
P0=Path('experiments/llm_curve_repair/P0_20261011T072421_376067Z')
P1=Path('experiments/llm_curve_repair/tiny_query_rate_local_20261011T072510_043572Z')
AUTH=Path('experiments/llm_curve_full/user_training_direction_20261011.json')


def build_queue(root,existing):
    (root/'settings').mkdir(parents=True,exist_ok=True)
    settings=json.loads(BASE.read_text(encoding='utf-8'));core=json.loads((Path(settings['frozen_plan'])/'splits.json').read_text(encoding='utf-8'))['plans']
    pointer=json.loads(Path('data/interim/llm_external/latest.json').read_text(encoding='utf-8'));external=Path(pointer['folder'])/'index.json'
    more=json.loads(external.read_text(encoding='utf-8'))['plans'];tasks=[]
    seeds=settings.get('run_seeds',[20260930,20260931,20260932])
    for dataset in ('MIT','XJTU','HUST','UConn_ISU_ILCC','Zhu2022_NCA'):
        plans=[p for p in core+more if p['dataset']==dataset]
        for plan in plans:
            for seed in seeds:
                cfg={**settings,'fold':plan['name'],'seed':seed,'dataset':dataset}
                if dataset not in ('MIT','XJTU'):cfg.update(unified_external_index=str(external.resolve()),unified_external_index_sha256=fingerprint(external))
                file=root/'settings'/(plan['name']+'_'+str(seed)+'.json');save_json(file,cfg)
                task={'dataset':dataset,'fold':plan['name'],'seed':seed,'settings':str(file.resolve()),'state':'pending',
                    'role_counts':{k:len(plan[k]) for k in ('train','validation','test')}}
                if existing and dataset=='MIT' and plan['name']=='MIT_fold1' and seed==20260930:
                    task.update(run=str(existing.resolve()),state='training_attached')
                tasks.append(task)
    return {'tasks':tasks,'sequence':['MIT','XJTU','HUST','UConn_ISU_ILCC','Zhu2022_NCA'],
        'seeds':seeds,'origin':100,'backbone':'models/llm/llama-3.2-1b-instruct',
        'source_sha256':{str(p):fingerprint(p) for p in (Path(__file__),BASE,AUTH)},
        'P1_strict_passed':False,'authorization':json.loads(AUTH.read_text(encoding='utf-8')),
        'independent_confirmation_certified':False,'goal_complete':False}


def command(root,index,arguments,on_run=None):
    log=root/'logs'/f'{index:03d}_{arguments[1].split(".")[-1]}.log';log.parent.mkdir(exist_ok=True)
    if log.exists():log=log.with_name(log.stem+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')+'.log')
    with log.open('w',encoding='utf-8') as output:
        process=subprocess.Popen([sys.executable,*arguments],cwd=Path.cwd(),stdout=output,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONUTF8':'1'})
        save_json(root/'active_process.json',{'pid':process.pid,'command':arguments,'log':str(log.resolve())})
        announced=False
        while process.poll() is None:
            if on_run is not None and not announced:
                output.flush()
                lines=log.read_text(encoding='utf-8').splitlines()
                run=next((line[4:] for line in lines if line.startswith('RUN=')),None)
                if run:on_run(run);announced=True
            time.sleep(2)
    if process.returncode:raise RuntimeError(f'command exited {process.returncode}; see {log}')
    return log


def wait_existing(path):
    while True:
        status=json.loads((path/'status.json').read_text(encoding='utf-8'))
        if status['state']=='failed':raise RuntimeError('attached training failed: '+str(path))
        if status['state']=='complete' and (path/'result.json').exists():return
        pid=status.get('pid')
        if os.name=='nt' and pid:
            # Query process membership using the Windows API; never launch a
            # duplicate GPU worker when an attached process is still alive.
            import ctypes
            handle=ctypes.windll.kernel32.OpenProcess(0x1000,False,pid)
            if not handle:raise RuntimeError('attached trainer is no longer running and has no complete result')
            ctypes.windll.kernel32.CloseHandle(handle)
        time.sleep(3)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path);ap.add_argument('--attach-first-run',type=Path);ap.add_argument('--prepare-only',action='store_true');args=ap.parse_args()
    root=args.root or Path('experiments/llm_curve_full')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    root.mkdir(parents=True,exist_ok=True)
    queue=json.loads((root/'queue.json').read_text(encoding='utf-8')) if (root/'queue.json').exists() else build_queue(root,args.attach_first_run)
    save_json(root/'queue.json',queue);save_json(Path('experiments/llm_curve_full/latest.json'),{'folder':str(root.resolve())})
    if args.prepare_only:
        save_json(root/'status.json',{'state':'queued','planned':len(queue['tasks']),'goal_complete':False})
        print(json.dumps({'queue':str(root.resolve()),'runs':len(queue['tasks']),'datasets':queue['sequence']},indent=2));return
    lock=(Path('experiments/llm_curve_full')/'gpu_queue.lock').open('a+b')
    lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
    if os.name=='nt':
        import msvcrt
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    save_json(root/'status.json',{'state':'running','pid':os.getpid(),'completed':sum(t['state']=='complete' for t in queue['tasks']),'planned':len(queue['tasks']),'goal_complete':False})
    def finish_dataset(dataset,index):
        if queue.get('optimization_required'):
            save_json(root/'status.json',{'state':'running','pid':os.getpid(),'dataset':dataset,'phase':'validation_algorithm_optimization',
                'completed':sum(t['state']=='complete' for t in queue['tasks']),'planned':len(queue['tasks']),'goal_complete':False})
            from scripts.optimize_llm_dataset_validation import run_optimization
            run_optimization(root,queue,dataset,command,P0,P1,AUTH)
        command(root,index,['-m','scripts.report_llm_full_dataset','--queue',str(root),'--dataset',dataset])
        optimized=root/'optimization'/dataset/'state.json'
        if optimized.exists() and json.loads(optimized.read_text(encoding='utf-8')).get('formal_tasks'):
            command(root,index,['-m','scripts.report_llm_full_dataset','--queue',str(root),'--dataset',dataset,'--variant','optimized'])
    def dataset_finished(dataset):
        return all(t['state']=='complete' for t in queue['tasks'] if t['dataset']==dataset)
    def report_mvp(dataset,index):
        if queue.get('schedule_mode')!='mvp_first' or dataset in queue.get('mvp_reports',{}):return
        if not any(t['state']=='complete' for t in queue['tasks'] if t['dataset']==dataset):return
        command(root,index,['-m','scripts.report_llm_full_dataset','--queue',str(root),'--dataset',dataset,'--progress-only','--figure-roles','test'])
        queue.setdefault('mvp_reports',{})[dataset]={'created_utc':datetime.now(timezone.utc).isoformat(),'single_seed_is_not_full_dataset_completion':True}
        save_json(root/'queue.json',queue)
    try:
        for index,task in enumerate(queue['tasks']):
            if task['state']=='complete':
                report_mvp(task['dataset'],index)
                next_dataset=queue['tasks'][index+1]['dataset'] if index+1<len(queue['tasks']) else None
                if next_dataset!=task['dataset'] and dataset_finished(task['dataset']):finish_dataset(task['dataset'],index)
                continue
            save_json(root/'status.json',{'state':'running','pid':os.getpid(),'dataset':task['dataset'],'fold':task['fold'],'seed':task['seed'],
                'completed':sum(t['state']=='complete' for t in queue['tasks']),'planned':len(queue['tasks']),'goal_complete':False})
            if 'run' in task:
                existing=Path(task['run']);saved=json.loads((existing/'status.json').read_text(encoding='utf-8'))
                alive=False
                if os.name=='nt' and saved.get('pid'):
                    import ctypes
                    handle=ctypes.windll.kernel32.OpenProcess(0x1000,False,saved['pid'])
                    alive=bool(handle)
                    if handle:ctypes.windll.kernel32.CloseHandle(handle)
                if saved.get('state') in ('starting','training') and not alive:
                    if not (existing/'resume.pt').exists():raise RuntimeError('interrupted trainer has no saved checkpoint: '+str(existing))
                    source=json.loads((existing/'source_manifest.json').read_text(encoding='utf-8'))['sha256']
                    if any(fingerprint(Path(file))!=sha for file,sha in source.items()):raise RuntimeError('resume source changed; preserve checkpoint and stop for review')
                    diagnostic=existing/'gradient_diagnostics.json'
                    if diagnostic.exists():
                        import shutil
                        backup=existing/('gradient_diagnostics_before_resume_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')+'.json')
                        shutil.copy2(diagnostic,backup)
                    task['resumed_from_saved_checkpoint']=True;save_json(root/'queue.json',queue)
                    command(root,index,['-m','scripts.train_llm_curve_repair','--p0',str(P0),'--settings',task['settings'],
                        '--architecture','query_rate_local','--mode','development','--p1-evidence',str(P1),
                        '--full-training-authorization',str(AUTH),'--resume',task['run']])
                else:wait_existing(existing)
            else:
                task['state']='training';save_json(root/'queue.json',queue)
                def announce(run):
                    task['run']=run;save_json(root/'queue.json',queue)
                log=command(root,index,['-m','scripts.train_llm_curve_repair','--p0',str(P0),'--settings',task['settings'],
                    '--architecture','query_rate_local','--mode','development','--p1-evidence',str(P1),'--full-training-authorization',str(AUTH)],on_run=announce)
                lines=log.read_text(encoding='utf-8').splitlines();run=next(line[4:] for line in lines if line.startswith('RUN='))
                task['run']=run
            task['state']='verifying';save_json(root/'queue.json',queue)
            command(root,index,['-m','scripts.evaluate_llm_curve_repair','--run',task['run'],'--roles','training','validation','test'])
            result=json.loads((Path(task['run'])/'result.json').read_text(encoding='utf-8'));accept=json.loads((Path(task['run'])/'strict_acceptance.json').read_text(encoding='utf-8'))
            task.update(state='complete',model_sha256=result['model_sha256'],selected_epoch=result['selected_epoch'],
                acceptance={k:{f:v[f] for f in ('cells_requested','strict_passes','all_requested_evaluable_and_passed')} for k,v in accept.items()})
            save_json(root/'queue.json',queue)
            report_mvp(task['dataset'],index)
            next_dataset=queue['tasks'][index+1]['dataset'] if index+1<len(queue['tasks']) else None
            if next_dataset!=task['dataset'] and dataset_finished(task['dataset']):
                finish_dataset(task['dataset'],index)
        final_reports={}
        for dataset in queue['sequence']:
            optimized=root/'reports'/dataset/'optimized'/'summary.json'
            report=json.loads((optimized if optimized.exists() else root/'reports'/dataset/'summary.json').read_text(encoding='utf-8'))
            final_reports[dataset]=report['roles']['test']
        save_json(root/'status.json',{'state':'complete','completed':len(queue['tasks']),'planned':len(queue['tasks']),
            'training_and_evaluation_complete':True,'optimization_workflow_complete':queue.get('optimization_required',False),
            'primary_mape_target_percent':json.loads(Path('configs/llm_curve_reporting_practical_v1.json').read_text(encoding='utf-8'))['numeric_thresholds']['mape_percent'],
            'mape_target_all_requested_test_instances':all(r['mape_target_passes']==r['requested_cell_seed_instances'] for r in final_reports.values()),
            'strict_reference_accuracy_success':all(t['acceptance']['test']['all_requested_evaluable_and_passed'] for t in queue['tasks']),
            'final_test_reports':final_reports,'goal_complete':False,
            'note':'Current registered reporting threshold used; accuracy failures, coverage, independent-confirmation limitations and visual-review status remain explicitly reported.'})
    except BaseException:
        save_json(root/'failure.json',{'traceback':traceback.format_exc()});save_json(root/'status.json',{'state':'failed','pid':os.getpid(),
            'completed':sum(t['state']=='complete' for t in queue['tasks']),'goal_complete':False});raise


if __name__=='__main__':main()
