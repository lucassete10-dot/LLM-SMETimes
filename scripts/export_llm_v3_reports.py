"""Export only verified formal v3 slots, preserving frozen coverage and roles."""
import argparse,json,subprocess,sys
from pathlib import Path
from scripts.prepare_same_type_forecast import save_json
from scripts.audit_llm_curve_repair import fingerprint
from scripts.run_llm_curve_v3_r0 import read
from battery_smetimes.llm_v3_workflow import SEEDS


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--dataset',required=True);ap.add_argument('--progress-only',action='store_true')
    ap.add_argument('--prepare-only',action='store_true');args=ap.parse_args()
    root=args.root;workflow=read(root/'workflow.json');state=workflow['datasets'][args.dataset]
    if not state.get('selection_lock'):raise ValueError('formal reports require validation selection lock')
    lock=root/'selection_locks'/f'{args.dataset}.json'
    if read(lock)!=state['selection_lock']:raise ValueError('selection lock differs')
    tasks=[]
    for fold in state['folds']:
        parent=state['templates'][fold]
        for seed in SEEDS:
            key=f'{state["winner"]}:{fold}:{seed}:240';slot=state['formal_slots'].get(key)
            task={'dataset':args.dataset,'fold':fold,'seed':seed,'role_counts':parent['role_counts'],'state':'pending'}
            if slot and slot.get('test_verified'):
                run=Path(slot['run']);proof=read(run/'repair_verification.json')
                if (not proof['passed'] or set(proof['role_scope'])!={'training','validation','test'} or
                    proof['model_sha256']!=fingerprint(run/'model.pt') or
                    slot['selection_lock_sha256']!=fingerprint(lock)):
                    raise ValueError('formal reporting evidence invalid')
                task.update(state='complete',run=str(run),model_sha256=proof['model_sha256'])
            tasks.append(task)
    if not args.progress_only and any(t['state']!='complete' for t in tasks):raise ValueError('dataset formal coverage incomplete')
    if not any(t['state']=='complete' for t in tasks):raise ValueError('no verified formal predictions yet')
    bridge=root/'report_bridges'/args.dataset;bridge.mkdir(parents=True,exist_ok=True)
    save_json(bridge/'queue.json',{'tasks':tasks,'seeds':list(SEEDS),'origin':100,'sequence':[args.dataset],
        'selection_lock_sha256':fingerprint(lock),'goal_complete':False})
    if args.prepare_only:print(json.dumps({'bridge':str(bridge),'tasks':len(tasks)}));return
    cohort=f'completed_{sum(t["state"]=="complete" for t in tasks):03d}'
    for role in ('training','validation','test'):
        out=root/'reports'/args.dataset/cohort/role
        arguments=[sys.executable,'-m','scripts.report_llm_full_dataset','--queue',str(bridge),
            '--dataset',args.dataset,'--figure-roles',role,'--output-dir',str(out),'--quality-v3']
        if args.progress_only:arguments+=['--progress-only']
        subprocess.run(arguments,check=True)
    save_json(root/'reports'/args.dataset/cohort/'delivery_manifest.json',{'formal_tasks':tasks,
        'roles':['training','validation','test'],'visual_review':'pending',
        'source_predictions_in_original_run_folders':True,'goal_complete':False})


if __name__=='__main__':main()
