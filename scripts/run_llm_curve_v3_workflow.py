"""Durable v3 comparisons/confirmation/coverage after the five-dataset R0 round."""
import argparse
import ctypes
import json
import os
import time
import traceback
from pathlib import Path
import numpy as np
from scripts.prepare_same_type_forecast import save_json
from scripts.audit_llm_curve_repair import fingerprint
from scripts.run_llm_full_dataset_queue import command, wait_existing, P0, P1, AUTH
from scripts.run_llm_curve_v3_r0 import read
from battery_smetimes.llm_curve_quality_v3 import curve_metrics_v3,validation_score
from battery_smetimes.llm_v3_workflow import new_dataset_state,next_round_robin_action


def alive(pid):
    if os.name!='nt':raise RuntimeError('this queue requires Windows process checks')
    handle=ctypes.windll.kernel32.OpenProcess(0x1000,False,pid or 0)
    if handle:ctypes.windll.kernel32.CloseHandle(handle)
    return bool(handle)


def evidence(run):
    policy=read(run/'validation_policy.json');rows={}
    with np.load(run/'validation_audited_predictions.npz',allow_pickle=False) as data:
        for cid in policy['cell_ids']:
            prefix=f'{cid}__{policy["origin"]}__'
            rows[cid]=curve_metrics_v3(*(data[prefix+k] for k in ('cycle','truth','prediction')))
    result={'policy':policy,'by_cell':rows,'score':validation_score(rows,policy),
        'model_sha256':fingerprint(run/'model.pt'),
        'predictions_sha256':fingerprint(run/'validation_audited_predictions.npz')}
    save_json(run/'v3_validation_evidence.json',result)
    return result


def noise_evidence(run,validation):
    cfg=read(run/'config.json');maes=[];noisy=[]
    with np.load(run/'training_audited_predictions.npz',allow_pickle=False) as data:
        for cid in cfg['train_ids']:
            prefix=f'{cid}__{cfg["origin"]}__'
            if prefix+'cycle' not in data:continue
            x,y,p=[data[prefix+k] for k in ('cycle','truth','prediction')]
            maes.append(curve_metrics_v3(x,y,p)['mae_pp'])
            # Require dense observations; sparse sampling cannot establish noise.
            noisy.append(bool(len(y)>=10 and np.median(np.diff(x))<=2 and
                              50*np.mean(np.abs(np.diff(y.astype(float),n=2)))>=.1))
    val_mae=np.mean([r['mae_pp'] for r in validation['by_cell'].values()])
    return {'training_validation_mae_ratio':float(val_mae/max(np.mean(maes),1e-12)) if maes else 0.,
            'noisy_training_cell_fraction':float(np.mean(noisy)) if noisy else 0.,
            'rule':'VAL/TRAIN mean MAE>=2 AND >=20% dense TRAIN curves have half mean absolute second difference>=0.1pp',
            'training_cells_numeric':len(maes),'training_requested':len(cfg['train_ids']),
            'test_used':False}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--prepare-only',action='store_true');args=ap.parse_args();root=args.root
    registration=root/'workflow_registration.json'
    if not registration.exists():
        save_json(registration,{'version':'curve_quality_v3','goal':'docs/LLM_SMETimes_round_robin_curve_goal_v3_20261013.md',
            'sources_sha256':{str(p):fingerprint(p) for p in (Path(__file__),Path('battery_smetimes/llm_v3_workflow.py'))},
            'noise_rule':'ratio>=2 AND noisy dense training cell fraction>=0.2; half mean abs second difference>=0.1pp',
            'maximum_fits':91,'formal_slots':51,'test_selection':False})
    reg=read(registration)
    if any(fingerprint(Path(p))!=h for p,h in reg['sources_sha256'].items()):
        raise ValueError('registered workflow source changed')
    if args.prepare_only:print(json.dumps(reg,indent=2));return
    try:
        while True:
            q=read(root/'R0_queue.json')
            if all(t['state']=='complete' for t in q['tasks']):break
            status=read(root/'status.json')
            if status['state']=='failed':raise RuntimeError('R0 controller failed')
            if not alive(status.get('pid')):raise RuntimeError('R0 controller stopped; inspect/resume before proceeding')
            save_json(root/'workflow_status.json',{'state':'waiting_R0','pid':os.getpid(),'goal_complete':False})
            time.sleep(3)
        # Allow the R0 controller to release the single global GPU lock.
        while alive(read(root/'status.json').get('pid')):time.sleep(2)
        gpu_lock=Path('experiments/llm_curve_full/gpu_queue.lock').open('a+b')
        gpu_lock.seek(0);gpu_lock.write(b'0');gpu_lock.flush();gpu_lock.seek(0)
        import msvcrt
        msvcrt.locking(gpu_lock.fileno(),msvcrt.LK_NBLCK,1)
        statefile=root/'workflow.json'
        if statefile.exists():workflow=read(statefile)
        else:
            legacy=read('experiments/llm_curve_full/20261011_MIT_then_other_datasets/queue.json')
            workflow={'sequence':q['sequence'],'datasets':{},'last_dataset_index':-1,
                      'registration_sha256':fingerprint(registration),'goal_complete':False}
            for task in q['tasks']:
                dataset=task['dataset'];run=Path(task['run']);ev=evidence(run)
                folds=list(dict.fromkeys(t['fold'] for t in legacy['tasks'] if t['dataset']==dataset))
                ds=new_dataset_state(dataset,folds,ev,str(run),noise_evidence(run,ev))
                ds['templates']={t['fold']:t for t in legacy['tasks'] if t['dataset']==dataset and t['seed']==20260930}
                ds['R0_settings']=read(task['settings']);workflow['datasets'][dataset]=ds
            save_json(statefile,workflow)
        if workflow['registration_sha256']!=fingerprint(registration):raise ValueError('workflow registration drift')
        while True:
            action=next_round_robin_action(workflow);save_json(statefile,workflow)
            if action is None:break
            ds=workflow['datasets'][action['dataset']]
            lockpath=root/'selection_locks'/f'{action["dataset"]}.json'
            if ds['selection_lock']:
                if lockpath.exists():
                    if read(lockpath)!=ds['selection_lock']:raise ValueError('selection lock changed')
                else:
                    if any((Path(t['run'])/'test_audited_predictions.npz').exists() for t in ds['trials'].values() if t.get('run')):
                        raise ValueError('candidate TEST exists before validation lock')
                    lockpath.parent.mkdir(exist_ok=True);save_json(lockpath,ds['selection_lock'])
            save_json(root/'workflow_status.json',{'state':'running','pid':os.getpid(),'action':action,
                'registered_fits':sum(len(d['trials']) for d in workflow['datasets'].values()),'goal_complete':False})
            index=100+sum(len(d['trials']) for d in workflow['datasets'].values())
            if action['kind']=='fit':
                trial=ds['trials'].setdefault(action['key'],{'state':'pending'})
                if sum(len(d['trials']) for d in workflow['datasets'].values())>91:raise ValueError('fit budget exceeded')
                cfg={**ds['R0_settings'],**action['override'],'fold':action['fold'],'seed':action['seed'],
                    'max_tiny_epochs':action['epochs'],'validation_min_epochs':40 if action['epochs']==120 else 60,
                    'validation_patience':4 if action['epochs']==120 else 6}
                settings=root/'settings'/(action['dataset']+'_'+action['key'].replace(':','_')+'.json')
                if settings.exists() and read(settings)!=cfg:raise ValueError('trial settings drift')
                save_json(settings,cfg);trial['settings']=str(settings.resolve());save_json(statefile,workflow)
                arguments=['-m','scripts.train_llm_curve_v3','--p0',str(P0),'--settings',str(settings),
                    '--architecture','query_rate_local','--mode','development','--p1-evidence',str(P1),
                    '--full-training-authorization',str(AUTH)]
                if trial.get('run'):
                    run=Path(trial['run']);status=read(run/'status.json')
                    if status['state']!='complete':
                        if alive(status.get('pid')):wait_existing(run)
                        else:
                            hashes=read(run/'source_manifest.json')['sha256']
                            if not (run/'resume.pt').exists() or any(fingerprint(Path(p))!=h for p,h in hashes.items()):
                                raise ValueError('unsafe v3 resume')
                            command(root,index,arguments+['--resume',str(run)])
                else:
                    def announce(run):trial.update(run=run,state='training');save_json(statefile,workflow)
                    log=command(root,index,arguments,on_run=announce)
                    if not trial.get('run'):
                        trial['run']=next(line[4:] for line in log.read_text(encoding='utf-8').splitlines() if line.startswith('RUN='))
                    run=Path(trial['run'])
                if ds['phase']!='formal' and (run/'test_audited_predictions.npz').exists():
                    raise ValueError('premature candidate TEST artifact')
                command(root,index,['-m','scripts.evaluate_llm_curve_repair','--run',str(run),'--roles','training','validation'])
                proof=read(run/'repair_verification.json')
                if not proof['passed'] or proof['model_sha256']!=fingerprint(run/'model.pt'):raise ValueError('replay failed')
                if cfg.get('numeric_lora_rank',0):
                    if not any(value>0 for rec in read(run/'gradient_diagnostics.json')
                               for key,value in rec['gradient_norms'].items() if key.endswith('.lora_B')):
                        raise ValueError('LoRA gradient missing')
                trial.update(state='complete',evidence=evidence(run))
            else:
                if not lockpath.exists():raise ValueError('TEST requires persisted lock')
                trial=ds['trials'][action['key']];run=Path(trial['run'])
                command(root,index,['-m','scripts.evaluate_llm_curve_repair','--run',str(run),
                                     '--roles','training','validation','test'])
                proof=read(run/'repair_verification.json')
                if not proof['passed'] or proof['model_sha256']!=trial['evidence']['model_sha256']:
                    raise ValueError('formal replay failed')
                ds['formal_slots'][action['key']]={'test_verified':True,'run':str(run),
                    'model_sha256':proof['model_sha256'],'selection_lock_sha256':fingerprint(lockpath)}
            workflow['last_dataset_index']=action['dataset_index'];save_json(statefile,workflow)
        formal=sum(len(d['formal_slots']) for d in workflow['datasets'].values())
        if formal!=51:raise ValueError('formal coverage differs from full goal')
        save_json(root/'workflow_status.json',{'state':'training_evaluation_complete','formal_slots':formal,
            'remaining':['independent audits','all-role figures and raw metrics','visual review','final goal audit'],
            'goal_complete':False})
    except BaseException:
        save_json(root/'workflow_failure.json',{'traceback':traceback.format_exc(),'goal_complete':False})
        save_json(root/'workflow_status.json',{'state':'failed','pid':os.getpid(),'goal_complete':False});raise


if __name__=='__main__':main()
