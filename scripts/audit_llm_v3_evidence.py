"""Audit completed real v3 trials without inventing training or TEST coverage."""
import argparse,json
from datetime import datetime,timezone
from pathlib import Path
import torch
from threadpoolctl import threadpool_limits
from scripts.audit_llm_completed_run_evidence import audit_run
from scripts.audit_llm_curve_repair import load_frozen,fingerprint
from scripts.prepare_same_type_forecast import save_json
from battery_smetimes.llm_v3_selection_audit import audit_selection


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args()
    root=args.root;read=lambda p:json.loads(p.read_text(encoding='utf-8'));q=read(root/'R0_queue.json')
    tasks=[dict(t,expected_roles=['training','validation']) for t in q['tasks'] if t['state']=='complete']
    if (root/'workflow.json').exists():
        w=read(root/'workflow.json')
        for dataset,state in w['datasets'].items():
            for key,trial in state['trials'].items():
                if trial['state']!='complete':continue
                cfg=read(Path(trial['run'])/'config.json')
                formal=state['formal_slots'].get(key,{}).get('test_verified',False)
                if formal:
                    lock=root/'selection_locks'/f'{dataset}.json'
                    if not lock.exists() or fingerprint(lock)!=state['formal_slots'][key]['selection_lock_sha256']:
                        raise ValueError('formal selection lock missing or changed')
                tasks.append({'run':trial['run'],'dataset':dataset,'fold':cfg['fold'],'seed':cfg['seed'],
                    'model_sha256':trial['evidence']['model_sha256'],
                    'expected_roles':['training','validation','test'] if formal else ['training','validation']})
    unique={}
    for task in tasks:
        key=str(Path(task['run']).resolve())
        if key not in unique or len(task['expected_roles'])>len(unique[key]['expected_roles']):unique[key]=task
    records=[];failures=[];cache={};torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        for task in unique.values():
            try:
                run=Path(task['run']);cfg=read(run/'config.json')
                if cfg.get('selection_version')!='curve_quality_v3':raise ValueError('legacy run in v3 cohort')
                if 'test' not in task['expected_roles'] and (run/'test_audited_predictions.npz').exists():
                    raise ValueError('premature TEST prediction')
                record=audit_run(task,cache,expected_roles=task['expected_roles'])
                _,cells,plan,_,_=load_frozen(cfg)
                record['v3_selection']=audit_selection(run,cells,plan)
                records.append(record);print(json.dumps({'run':str(run),'passed':True}),flush=True)
            except Exception as exc:failures.append({'run':task['run'],'error':str(exc)})
    report={'utc':datetime.now(timezone.utc).isoformat(),'completed_unique_runs':len(unique),
        'audited_runs':len(records),'records':records,'failures':failures,
        'passed':bool(unique and not failures and len(records)==len(unique)),
        'scope':'Completed v3 runs and actually evaluated roles only; no accuracy or full-goal completion claim',
        'source_sha256':fingerprint(Path(__file__)),'goal_complete':False}
    save_json(root/'completed_v3_evidence_audit.json',report)
    print(json.dumps({k:report[k] for k in ('completed_unique_runs','audited_runs','passed','failures')}),flush=True)
    if failures:raise ValueError('v3 evidence audit failed')


if __name__=='__main__':main()
