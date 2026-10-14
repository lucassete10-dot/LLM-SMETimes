"""First v3 round: finish preserved legacy work, then rotate five R0 trials.

This stage never claims the complete goal is finished. Comparisons,
confirmation, formal coverage and delivery remain explicit subsequent phases.
"""
import argparse
import json
import os
from pathlib import Path
from scripts.prepare_same_type_forecast import save_json
from scripts.audit_llm_curve_repair import fingerprint
from scripts.run_llm_full_dataset_queue import command, wait_existing, P0, P1, AUTH
from battery_smetimes.llm_curve_quality_v3 import curve_metrics_v3, validation_score
import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    args = ap.parse_args()
    root = args.root
    q = read(root/'R0_queue.json')
    lock = Path('experiments/llm_curve_full/gpu_queue.lock').open('a+b')
    lock.seek(0); lock.write(b'0'); lock.flush(); lock.seek(0)
    if os.name == 'nt':
        import msvcrt
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    try:
        migration = read('experiments/llm_curve_v3/migration_latest.json')
        legacy = Path(migration['legacy_run'])
        if not (root/'legacy_finished.json').exists():
            save_json(root/'status.json', {'state':'waiting_legacy_finish', 'pid':os.getpid(),
                      'run':str(legacy), 'goal_complete':False})
            wait_existing(legacy)
            command(root, 0, ['-m','scripts.evaluate_llm_curve_repair','--run',str(legacy),
                             '--roles','training','validation','test'])
            proof = read(legacy/'repair_verification.json')
            result = read(legacy/'result.json')
            if not proof['passed'] or proof['model_sha256'] != fingerprint(legacy/'model.pt'):
                raise ValueError('legacy replay verification failed')
            oldroot = Path('experiments/llm_curve_full/20261011_MIT_then_other_datasets')
            old = read(oldroot/'queue.json')
            for task in old['tasks']:
                if task.get('run') and Path(task['run']).resolve() == legacy.resolve():
                    task.update(state='complete', selected_epoch=result['selected_epoch'],
                                model_sha256=result['model_sha256'])
                    task['acceptance'] = {role:{k:record[k] for k in
                        ('cells_requested','strict_passes','all_requested_evaluable_and_passed')}
                        for role,record in read(legacy/'strict_acceptance.json').items()}
                elif task['state'] == 'pending':
                    task.update(state='superseded_v3', replacement_goal=q['full_goal'])
            save_json(oldroot/'queue.json', old)
            save_json(oldroot/'status.json', {'state':'migrated_to_v3',
                'completed':sum(t['state']=='complete' for t in old['tasks']),
                'new_root':str(root.resolve()), 'goal_complete':False})
            save_json(root/'legacy_finished.json', {'run':str(legacy),
                      'model_sha256':result['model_sha256'], 'passed':True})
        for i, task in enumerate(q['tasks'], 1):
            if task['state'] == 'complete':
                continue
            save_json(root/'status.json', {'state':'running', 'pid':os.getpid(),
                'dataset':task['dataset'],'phase':'R0','completed_R0':sum(t['state']=='complete' for t in q['tasks']),
                'planned_R0':5,'formal_planned':51,'goal_complete':False})
            if task.get('run'):
                run = Path(task['run'])
                status = read(run/'status.json')
                if status['state'] != 'complete':
                    # A restart must verify unchanged source before resuming.
                    source = read(run/'source_manifest.json')['sha256']
                    if any(fingerprint(Path(f)) != h for f,h in source.items()):
                        raise ValueError('v3 resume source drift')
                    if os.name == 'nt' and status.get('pid'):
                        import ctypes
                        handle = ctypes.windll.kernel32.OpenProcess(0x1000,False,status['pid'])
                        if handle:
                            ctypes.windll.kernel32.CloseHandle(handle)
                            wait_existing(run)
                        else:
                            command(root,i,['-m','scripts.train_llm_curve_v3','--p0',str(P0),
                                '--settings',task['settings'],'--architecture','query_rate_local',
                                '--mode','development','--p1-evidence',str(P1),
                                '--full-training-authorization',str(AUTH),'--resume',str(run)])
                    else:
                        raise RuntimeError('cannot safely establish v3 trainer liveness')
            else:
                task['state']='training'; save_json(root/'R0_queue.json',q)
                def announce(run):
                    task['run']=run; save_json(root/'R0_queue.json',q)
                log=command(root,i,['-m','scripts.train_llm_curve_v3','--p0',str(P0),
                    '--settings',task['settings'],'--architecture','query_rate_local',
                    '--mode','development','--p1-evidence',str(P1),
                    '--full-training-authorization',str(AUTH)],on_run=announce)
                if not task.get('run'):
                    task['run']=next(line[4:] for line in log.read_text(encoding='utf-8').splitlines()
                                     if line.startswith('RUN='))
                run=Path(task['run'])
            task['state']='verifying';save_json(root/'R0_queue.json',q)
            command(root,i,['-m','scripts.evaluate_llm_curve_repair','--run',str(run),
                            '--roles','training','validation'])
            proof=read(run/'repair_verification.json')
            if not proof['passed'] or proof['role_scope'] != ['training','validation']:
                raise ValueError('R0 must verify TRAIN/VAL only')
            policy=read(run/'validation_policy.json')
            metrics={}
            with np.load(run/'validation_audited_predictions.npz',allow_pickle=False) as data:
                for cid in policy['cell_ids']:
                    prefix=f'{cid}__{policy["origin"]}__'
                    metrics[cid]=curve_metrics_v3(*(data[prefix+k] for k in ('cycle','truth','prediction')))
            score=validation_score(metrics,policy)
            save_json(run/'v3_validation_evidence.json',{'score':score,'by_cell':metrics,
                'policy':policy,'model_sha256':fingerprint(run/'model.pt'),
                'source_predictions_sha256':fingerprint(run/'validation_audited_predictions.npz'),
                'no_candidate_test_selection':True,'goal_complete':False})
            task.update(state='complete',score=score,model_sha256=fingerprint(run/'model.pt'),
                        selected_epoch=read(run/'result.json')['selected_epoch'])
            save_json(root/'R0_queue.json',q)
        save_json(root/'status.json',{'state':'R0_round_complete','pid':os.getpid(),
            'completed_R0':5,'remaining_phases':q['remaining_phases'],'goal_complete':False})
    except BaseException:
        import traceback
        save_json(root/'failure.json',{'traceback':traceback.format_exc(),'goal_complete':False})
        save_json(root/'status.json',{'state':'failed','pid':os.getpid(),'goal_complete':False})
        raise


if __name__ == '__main__':
    main()
