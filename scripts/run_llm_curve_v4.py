"""One GPU, five-dataset diagnosis and bounded v4 candidates/confirmation.

All selection is based on TRAIN/VAL; TEST requires a persisted selection lock.
Historical models are read-only and are reused only at a matched budget/seed.
"""
import argparse
import json
import os
import traceback
from pathlib import Path

import numpy as np

from battery_smetimes.llm_v4_workflow import SEEDS, OVERRIDES, screen_winner, accepted_confirmation
from battery_smetimes.llm_v3_workflow import summarize_validation
from battery_smetimes.llm_curve_quality_v3 import curve_metrics_v3, validation_score
from scripts.prepare_same_type_forecast import save_json
from scripts.audit_llm_curve_repair import fingerprint, load_frozen
from scripts.run_llm_full_dataset_queue import command, P0, P1, AUTH, wait_existing
from scripts.run_llm_curve_v3_workflow import alive

PREVIOUS = Path('experiments/llm_curve_v3/20261013_round_robin')
GOAL = Path('docs/LLM_SMETimes_curve_optimization_v4_20261014.md')


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def validation_evidence(run):
    run = Path(run)
    proof = read(run/'repair_verification.json')
    if not proof['passed'] or proof['model_sha256']!=fingerprint(run/'model.pt'):
        raise ValueError('model/replay evidence mismatch')
    if not set(('training','validation')) <= set(proof['role_scope']):
        raise ValueError('training and validation must be audited')
    policy = read(run/'validation_policy.json'); rows = {}
    with np.load(run/'validation_audited_predictions.npz', allow_pickle=False) as data:
        for cid in policy['cell_ids']:
            prefix = f'{cid}__{policy["origin"]}__'
            rows[cid] = curve_metrics_v3(*(data[prefix+k] for k in ('cycle','truth','prediction')))
    return {'policy':policy, 'by_cell':rows, 'score':validation_score(rows,policy),
            'model_sha256':fingerprint(run/'model.pt'),
            'predictions_sha256':fingerprint(run/'validation_audited_predictions.npz')}


def task_key(dataset, variant, fold, seed, epochs):
    return f'{dataset}_{variant}_{fold}_{seed}_{epochs}'


def append_task(q, dataset, variant, fold, seed, epochs, phase):
    key = task_key(dataset,variant,fold,seed,epochs)
    if any(t['key']==key for t in q['tasks']): return
    q['tasks'].append(dict(key=key,dataset=dataset,variant=variant,fold=fold,seed=seed,
                           epochs=epochs,phase=phase,state='pending'))
    if len(q['tasks'])>61: raise ValueError('v4 fit budget exceeded')


def prepare(root):
    root.mkdir(parents=True,exist_ok=True)
    for name in ('settings','logs','diagnostics','selection_locks','reports'):
        (root/name).mkdir(exist_ok=True)
    sources = [Path(__file__),GOAL,Path('scripts/train_llm_curve_v4.py'),Path('scripts/diagnose_llm_curve_v4.py'),
               Path('battery_smetimes/llm_curve_objective_v4.py'),Path('battery_smetimes/llm_v4_workflow.py'),
               Path('battery_smetimes/llm_query_rate.py'),Path('battery_smetimes/llm_local_recovery.py'),
               Path('battery_smetimes/research_llm.py'),Path('battery_smetimes/research_sequences.py'),
               Path('scripts/evaluate_llm_curve_repair.py'),Path('scripts/train_llm_curve_repair.py')]
    registration = dict(version='curve_optimization_v4',goal=str(GOAL.resolve()),maximum_new_fits=61,
                        overrides=OVERRIDES,seeds=list(SEEDS),test_used_for_selection=False,
                        sources_sha256={str(p):fingerprint(p) for p in sources},
                        previous_workflow_sha256=fingerprint(PREVIOUS/'workflow.json'),
                        screen_epochs=120,confirmation_epochs=240)
    target = root/'registration.json'
    if target.exists():
        if read(target)!=registration: raise ValueError('registered sources or plan changed')
    else: save_json(target,registration)
    save_json(Path('experiments/llm_curve_v4/latest.json'),{'folder':str(root.resolve())})
    statefile = root/'queue.json'
    if statefile.exists(): return read(statefile)
    old = read(PREVIOUS/'workflow.json'); r0 = read(PREVIOUS/'R0_queue.json')
    q = dict(version='curve_optimization_v4',sequence=old['sequence'],phase='diagnosis',datasets={},tasks=[],
             registration_sha256=fingerprint(target),execution_complete=False,accuracy_goal_reached=False)
    for dataset in q['sequence']:
        task = next(t for t in r0['tasks'] if t['dataset']==dataset)
        ds = old['datasets'][dataset]
        confirmations = [ds['trials'][f'R0:{ds["folds"][0]}:{seed}:240']['run'] for seed in SEEDS]
        q['datasets'][dataset] = dict(folds=ds['folds'],R0_settings=read(task['settings']),
                                     R0_run=task['run'],R0_evidence=validation_evidence(task['run']),
                                     R0_confirm_runs=confirmations,
                                     R0_confirm_evidence=[validation_evidence(r) for r in confirmations],
                                     diagnostic_complete=False,screen_winner=None,winner=None,
                                     old_formal_slots=ds['formal_slots'])
    for variant in OVERRIDES:
        for dataset in q['sequence']:
            append_task(q,dataset,variant,q['datasets'][dataset]['folds'][0],SEEDS[0],120,'screen')
    save_json(statefile,q); return q


def candidate_evidence(q, dataset, variant, seed, epochs):
    fold = q['datasets'][dataset]['folds'][0]
    return next(t['evidence'] for t in q['tasks'] if t['key']==task_key(dataset,variant,fold,seed,epochs))


def write_progress(root, q):
    lines = ['# v4 运行进度','',f'阶段：{q["phase"]}；新训练完成 {sum(t["state"]=="complete" for t in q["tasks"])}/{len(q["tasks"])}（最大61）。',
             '', '主标准：MAPE≤3%，最终3种子；当前单种子筛选不能称为3种子成功。', '',
             '| 数据集 | 诊断 | 120轮筛选候选 | 三种子后采纳 |','|---|---|---|---|']
    for name in q['sequence']:
        ds = q['datasets'][name]
        lines.append(f'| {name} | {"完成" if ds["diagnostic_complete"] else "待运行"} | {ds["screen_winner"] or "尚无"} | {ds["winner"] or "待确认"} |')
    lines += ['', '验证集对照：', '', '| 数据集 | 候选 | S_val | R0 S_val | MAPE≤3% | 最差MAE pp |', '|---|---|---:|---:|---:|---:|']
    for task in q['tasks']:
        if task['state']=='complete' and task['phase']=='screen':
            c = summarize_validation(task['evidence']); b = summarize_validation(q['datasets'][task['dataset']]['R0_evidence'])
            lines.append(f'| {task["dataset"]} | {task["variant"]} | {c["score"]:.4f} | {b["score"]:.4f} | {c["mape_passes"]}/{c["cells"]} | {c["worst_mae_pp"]:.3f} |')
    lines += ['', '起点、尾部、阶段指标与原始预测存于各运行目录。筛选不读取TEST；实际生成图的视觉检查尚需完成。']
    (root/'progress.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def export_selected(root,q):
    indexlines=['# v4 五数据集结果入口','',
                '候选选择依据为VAL；TEST均是已查看数据的开发评估。图集使用原始标签，MAPE≤3%与曲线参考指标分开报告。','',
                '| 数据集 | 采纳配置 | 结果入口 |','|---|---|---|']
    for index,dataset in enumerate(q['sequence'],200):
        ds=q['datasets'][dataset]
        if ds['winner']=='R0':
            count=len(ds['old_formal_slots'])
            out=PREVIOUS/'reports'/dataset/f'completed_{count:03d}'
            for role in ('training','validation','test'):
                if not (out/role/'all_cells.pdf').exists():raise ValueError('preserved report missing')
        else:
            selected=[t for t in q['tasks'] if t['dataset']==dataset and t['variant']==ds['winner'] and t['epochs']==240]
            if len(selected)!=len(ds['folds'])*3 or not all(t.get('test_verified') for t in selected):
                raise ValueError('adopted model formal coverage incomplete')
            out=root/'reports'/dataset/'selected'
            bridge=out/'source_bridge';bridge.mkdir(parents=True,exist_ok=True)
            tasks=[]
            for t in selected:
                _,_,plan,_,_=load_frozen(read(Path(t['run'])/'config.json'))
                tasks.append(dict(dataset=dataset,fold=t['fold'],seed=t['seed'],state='complete',run=t['run'],
                                  model_sha256=t['evidence']['model_sha256'],
                                  role_counts={r:len(plan[r]) for r in ('train','validation','test')}))
            save_json(bridge/'queue.json',{'tasks':tasks,'seeds':list(SEEDS),'origin':100,'test_used_for_selection':False})
            for role in ('training','validation','test'):
                command(root,index,['-m','scripts.report_llm_full_dataset','--queue',str(bridge),'--dataset',dataset,
                                    '--figure-roles',role,'--available-roles','training','validation','test',
                                    '--quality-v3','--output-dir',str(out/role)])
        links=' / '.join(f'[{r} PDF]({(out/r/"all_cells.pdf").resolve().as_posix()})' for r in ('training','validation','test'))
        indexlines.append(f'| {dataset} | {ds["winner"]} | {links} |')
        ds['selected_report_root']=str(out.resolve())
    indexlines += ['', '新候选图集的视觉检查状态为pending；已有R0图集沿用v3完成的检查。全部达到参考曲线精度尚未认证。',
                   '',f'[同预算VAL比较与进度]({(root/"progress.md").resolve().as_posix()})']
    (root/'results.md').write_text('\n'.join(indexlines)+'\n',encoding='utf-8')


def execute_fit(root, q, task, index):
    ds = q['datasets'][task['dataset']]
    cfg = {**ds['R0_settings'],**OVERRIDES[task['variant']], 'fold':task['fold'],'seed':task['seed'],
           'max_tiny_epochs':task['epochs'],'validation_min_epochs':40 if task['epochs']==120 else 60,
           'validation_patience':4 if task['epochs']==120 else 6,
           'experiment_version':'curve_optimization_v4','objective_file':str(GOAL.resolve())}
    settings = root/'settings'/(task['key']+'.json')
    if settings.exists() and read(settings)!=cfg: raise ValueError('trial setting drift')
    save_json(settings,cfg);task['settings']=str(settings.resolve())
    arguments = ['-m','scripts.train_llm_curve_v4','--p0',str(P0),'--settings',str(settings),
                 '--architecture','query_rate_local','--mode','development','--p1-evidence',str(P1),
                 '--full-training-authorization',str(AUTH)]
    def persist(): save_json(root/'queue.json',q)
    if task.get('run'):
        run = Path(task['run']); status = read(run/'status.json')
        if status['state']!='complete':
            if alive(status.get('pid')): wait_existing(run)
            else:
                if not (run/'resume.pt').exists(): raise ValueError('no safe resumable checkpoint')
                if any(fingerprint(Path(p))!=h for p,h in read(run/'source_manifest.json')['sha256'].items()):
                    raise ValueError('resume source changed')
                command(root,index,arguments+['--resume',str(run)])
    else:
        def announce(run): task.update(run=run,state='training');persist()
        log = command(root,index,arguments,on_run=announce)
        if not task.get('run'):
            task['run'] = next(line[4:] for line in log.read_text(encoding='utf-8').splitlines() if line.startswith('RUN='))
        run = Path(task['run'])
    if (run/'test_audited_predictions.npz').exists() and task['phase']!='formal':
        # Confirmations can be reused formal slots only after a persisted lock.
        if not (root/'selection_locks'/f'{task["dataset"]}.json').exists():
            raise ValueError('premature candidate TEST')
    command(root,index,['-m','scripts.evaluate_llm_curve_repair','--run',str(run),'--roles','training','validation'])
    task.update(state='complete',evidence=validation_evidence(run),selected_epoch=read(run/'result.json')['selected_epoch'])
    persist()
    # Actual predictions, scientific plotter; visual review remains pending.
    if task['phase'] in ('screen','confirm'):
        command(root,index,['-m','scripts.report_llm_v3_trial','--run',str(run),'--out',str(root/'reports'/task['key']),
                            '--roles','training','validation'])
    task['train_validation_report_complete']=task['phase'] in ('screen','confirm'); persist()


def main():
    ap = argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--prepare-only',action='store_true')
    args = ap.parse_args();root = args.root; q = prepare(root)
    write_progress(root,q)
    if args.prepare_only:
        print(json.dumps({'root':str(root.resolve()),'screen_fits':len(q['tasks']),'maximum_new_fits':61}));return
    # Lock before both GPU diagnostics and any training process.
    lock = Path('experiments/llm_curve_full/gpu_queue.lock').open('a+b')
    lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
    import msvcrt
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    try:
        for index,dataset in enumerate(q['sequence']):
            ds = q['datasets'][dataset]
            if ds['diagnostic_complete']:continue
            save_json(root/'status.json',dict(state='running',pid=os.getpid(),phase='diagnosis',dataset=dataset))
            out = root/'diagnostics'/dataset
            if not (out/'diagnosis.json').exists():
                command(root,index,['-m','scripts.diagnose_llm_curve_v4','--run',ds['R0_run'],'--out',str(out)])
            diagnosis = read(out/'diagnosis.json')
            if diagnosis['model_sha256']!=ds['R0_evidence']['model_sha256'] or diagnosis['test_used']:
                raise ValueError('diagnosis model/role mismatch')
            ds['diagnostic_complete']=True;save_json(root/'queue.json',q);write_progress(root,q)
        if q['phase']=='diagnosis': q['phase']='screen'
        while q['phase']!='complete':
            save_json(root/'queue.json',q);write_progress(root,q)
            pending = next((t for t in q['tasks'] if t['state']!='complete' or
                            (t['phase'] in ('screen','confirm') and not t.get('train_validation_report_complete'))),None)
            if pending:
                index = 10+q['tasks'].index(pending)
                save_json(root/'status.json',dict(state='running',pid=os.getpid(),phase=q['phase'],task=pending['key'],
                                                  completed=sum(t['state']=='complete' for t in q['tasks']),planned=len(q['tasks'])))
                execute_fit(root,q,pending,index);continue
            if q['phase']=='screen':
                for dataset in q['sequence']:
                    ds=q['datasets'][dataset]
                    candidates={v:candidate_evidence(q,dataset,v,SEEDS[0],120) for v in OVERRIDES}
                    if any(e['policy']!=ds['R0_evidence']['policy'] for e in candidates.values()):
                        raise ValueError('screen physical membership/policy drift')
                    ds['screen_winner']=screen_winner(candidates,ds['R0_evidence'])
                for seed in SEEDS:
                    for dataset in q['sequence']:
                        ds=q['datasets'][dataset]
                        if ds['screen_winner']:append_task(q,dataset,ds['screen_winner'],ds['folds'][0],seed,240,'confirm')
                q['phase']='confirm'
            elif q['phase']=='confirm':
                for dataset in q['sequence']:
                    ds=q['datasets'][dataset];v=ds['screen_winner']
                    accepted=bool(v and accepted_confirmation([candidate_evidence(q,dataset,v,s,240) for s in SEEDS],ds['R0_confirm_evidence']))
                    ds['winner']=v if accepted else 'R0'
                    chosen=[candidate_evidence(q,dataset,v,s,240) for s in SEEDS] if accepted else ds['R0_confirm_evidence']
                    lockpath=root/'selection_locks'/f'{dataset}.json'
                    selection={'dataset':dataset,'winner':ds['winner'],'seeds':list(SEEDS),'validation_evidence':chosen,
                               'test_used_for_selection':False,'test_scope':'previously viewed development evaluation',
                               'reference_models':ds['R0_confirm_runs'],'registration_sha256':q['registration_sha256']}
                    if lockpath.exists() and read(lockpath)!=selection: raise ValueError('selection lock drift')
                    save_json(lockpath,selection)
                for fold_index in range(1,max(len(ds['folds']) for ds in q['datasets'].values())):
                    for seed in SEEDS:
                        for dataset in q['sequence']:
                            ds=q['datasets'][dataset]
                            if ds['winner']!='R0' and fold_index<len(ds['folds']):
                                append_task(q,dataset,ds['winner'],ds['folds'][fold_index],seed,240,'formal')
                q['phase']='formal'
            elif q['phase']=='formal':
                for index,task in enumerate(q['tasks'],100):
                    ds=q['datasets'][task['dataset']]
                    if task['epochs']!=240 or ds['winner']!=task['variant'] or task.get('test_verified'):continue
                    lockpath=root/'selection_locks'/f'{task["dataset"]}.json'
                    if not lockpath.exists():raise ValueError('TEST needs prior persisted validation lock')
                    command(root,index,['-m','scripts.evaluate_llm_curve_repair','--run',task['run'], '--roles','training','validation','test'])
                    proof=read(Path(task['run'])/'repair_verification.json')
                    if not proof['passed'] or proof['model_sha256']!=task['evidence']['model_sha256']:raise ValueError('formal replay failed')
                    task.update(test_verified=True,selection_lock_sha256=fingerprint(lockpath));save_json(root/'queue.json',q)
                q['phase']='complete'
            else:raise ValueError('unknown execution phase')
        export_selected(root,q)
        q['training_evaluation_complete']=True
        q['execution_complete']=False
        save_json(root/'queue.json',q);write_progress(root,q)
        save_json(root/'status.json',dict(state='bounded_training_evaluation_complete',pid=os.getpid(),
                                          new_fits=len(q['tasks']),remaining=['new-candidate visual review and final execution audit'],
                                          universal_accuracy_achieved=False))
    except BaseException:
        save_json(root/'failure.json',dict(traceback=traceback.format_exc()))
        save_json(root/'status.json',dict(state='failed',pid=os.getpid()));raise


if __name__=='__main__':main()
