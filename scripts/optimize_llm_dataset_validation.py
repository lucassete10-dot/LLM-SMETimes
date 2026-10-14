"""Finite validation-only LLM+SMETimes optimization within the GPU queue."""
import ctypes,json,os,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from scripts.prepare_same_type_forecast import save_json
from scripts.audit_llm_curve_repair import fingerprint
from scripts.report_llm_optimization import write_optimization_report
from battery_smetimes.llm_reporting import finite_metric


def validation_summary(metrics,plan):
    rows=list(metrics.values())
    required=set(plan['score_weights'])|set(plan['numeric_thresholds'])
    missing_stage=[cid for cid,r in metrics.items() if r.get('stage_drop_max_pp') is None and r.get('stage_metrics_available') is False]
    if not rows or any(any(not finite_metric(r,k) and not (k=='stage_drop_max_pp' and cid in missing_stage) for k in required) for cid,r in metrics.items()):raise ValueError('incomplete validation metric coverage')
    # Legacy 'evaluable' includes a dense-sampling shape requirement. Sparse
    # RPT curves with valid numeric errors remain eligible for numeric ranking.
    return {'cells':len(rows),'ranking_eligible':not missing_stage,'stage_unavailable_cells':missing_stage,
        'score':float(np.mean([sum(r[k]*w for k,w in plan['score_weights'].items()) for r in rows])) if not missing_stage else None,
        'worst_mae_pp':max(r['mae_pp'] for r in rows),'mape_passes':sum(r['mape_percent']<=plan['numeric_thresholds']['mape_percent'] for r in rows),
        'practical_numeric_passes':sum(all(finite_metric(r,k) and r[k]<=limit for k,limit in plan['numeric_thresholds'].items()) for r in rows)}


def passes_screen(candidate,baseline,plan):
    if not candidate.get('ranking_eligible',True) or not baseline.get('ranking_eligible',True):return False
    if candidate['cells']!=baseline['cells']:return False
    return (candidate['score']<=(1-plan['screen_min_relative_improvement'])*baseline['score']
        and candidate['worst_mae_pp']<=plan['worst_val_mae_max_ratio']*baseline['worst_mae_pp']
        and candidate['mape_passes']>=baseline['mape_passes'])


def confirmed(candidates,baselines,plan):
    if len(candidates)!=3 or len(baselines)!=3:return False
    if any(not c.get('ranking_eligible',True) for c in candidates+baselines):return False
    improvement=[c['score']<b['score'] for c,b in zip(candidates,baselines)]
    return (sum(improvement)>=plan['confirmation_min_improved_seed_count']
        and np.mean([c['score'] for c in candidates])<=(1-plan['confirmation_min_mean_relative_improvement'])*np.mean([b['score'] for b in baselines])
        and all(c['cells']==b['cells'] and c['worst_mae_pp']<=plan['worst_val_mae_max_ratio']*b['worst_mae_pp'] and c['mape_passes']>=b['mape_passes'] for c,b in zip(candidates,baselines)))


def run_optimization(root,queue,dataset,command,P0,P1,AUTH):
    root=Path(root);folder=root/'optimization'/dataset;folder.mkdir(parents=True,exist_ok=True)
    plan_path=Path(queue['optimization_plan']);plan=json.loads(plan_path.read_text(encoding='utf-8'))
    state_path=folder/'state.json'
    state=json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {'state':'pending','dataset':dataset,'screening':{},'confirmation':{},'formal_tasks':[],'plan_sha256':fingerprint(plan_path),'test_used_for_selection':False}
    if state['plan_sha256']!=fingerprint(plan_path):raise ValueError('optimization registration changed after start')
    if state['state']=='complete':return
    bases=[t for t in queue['tasks'] if t['dataset']==dataset]
    if not bases or any(t['state']!='complete' for t in bases):raise ValueError('baseline dataset is not completely evaluated')
    first_fold=bases[0]['fold'];reference=next(t for t in bases if t['fold']==first_fold and t['seed']==plan['screen_seed'])
    def metric(task):
        # This is the only accuracy file consumed for algorithm selection.
        rows=json.loads((Path(task['run'])/'selected_metrics.json').read_text(encoding='utf-8'))
        if len(rows)!=task['role_counts']['validation']:raise ValueError('validation denominator changed')
        return validation_summary(rows,plan)
    baseline=metric(reference);state['baseline_validation']=baseline
    task_index=1000+plan['datasets'].index(dataset)*100
    def train(slot,parent,override,index,roles):
        if slot.get('state')=='complete':return
        cfg=json.loads(Path(parent['settings']).read_text(encoding='utf-8'));cfg.update(override)
        cfg['description']='Preregistered validation-only single-factor optimization: '+str(override)
        config=folder/(slot['name']+'.json');save_json(config,cfg)
        slot.update(dataset=dataset,fold=parent['fold'],seed=parent['seed'],settings=str(config.resolve()),role_counts=parent['role_counts'])
        arguments=['-m','scripts.train_llm_curve_repair','--p0',str(P0),'--settings',str(config),
            '--architecture','query_rate_local','--mode','development','--p1-evidence',str(P1),'--full-training-authorization',str(AUTH)]
        if slot.get('run'):
            run=Path(slot['run']);saved=json.loads((run/'status.json').read_text(encoding='utf-8'))
            while saved.get('state') in ('starting','training'):
                handle=ctypes.windll.kernel32.OpenProcess(0x1000,False,saved.get('pid',0)) if os.name=='nt' else None
                if not handle:break
                ctypes.windll.kernel32.CloseHandle(handle);time.sleep(3);saved=json.loads((run/'status.json').read_text(encoding='utf-8'))
            if saved.get('state')=='failed':raise RuntimeError('optimization training failed: '+str(run))
            if saved.get('state')!='complete':
                hashes=json.loads((run/'source_manifest.json').read_text(encoding='utf-8'))['sha256']
                if not (run/'resume.pt').exists() or any(fingerprint(Path(p))!=h for p,h in hashes.items()):raise ValueError('unsafe optimization resume')
                command(root,index,arguments+['--resume',str(run)])
        else:
            def announce(run):slot['run']=run;slot['state']='training';save_json(state_path,state)
            log=command(root,index,arguments,on_run=announce)
            slot['run']=next(line[4:] for line in log.read_text(encoding='utf-8').splitlines() if line.startswith('RUN='))
        command(root,index,['-m','scripts.evaluate_llm_curve_repair','--run',slot['run'],'--roles',*roles])
        if cfg.get('numeric_lora_rank',0):
            diagnostics=json.loads((Path(slot['run'])/'gradient_diagnostics.json').read_text(encoding='utf-8'))
            adapted=any(value>0 for record in diagnostics for key,value in record['gradient_norms'].items() if key.endswith('.lora_B'))
            if not adapted:raise ValueError('numeric LoRA candidate has no verified nonzero adaptation gradient')
            slot['numeric_lora_gradient_verified']=True
        slot.update(state='complete',validation=metric(slot),audited_roles=roles)
        save_json(state_path,state)
    state['state']='screening';save_json(state_path,state)
    for index,candidate in enumerate(plan['screen_candidates']):
        slot=state['screening'].setdefault(candidate['name'],{'name':candidate['name'],'override':candidate['override']})
        train(slot,reference,candidate['override'],task_index+index,['training','validation'])
        slot['screen_passed']=passes_screen(slot['validation'],baseline,plan);save_json(state_path,state)
        write_optimization_report(folder,state,plan)
    eligible=[s for s in state['screening'].values() if s['screen_passed']]
    if not eligible:
        if not baseline.get('ranking_eligible',True):
            state.update(state='complete',winner=None,optimization_accuracy_improved=False,conclusion='All four candidates evaluated; the preregistered stage-inclusive score is unavailable on sparse validation cells. No winner can be confirmed; retain baseline and report numeric errors and missing coverage without fabricating stage measurements.')
            save_json(state_path,state);write_optimization_report(folder,state,plan);return
        state.update(state='complete',winner=None,conclusion='No candidate met preregistered validation improvement; baseline retained.',optimization_accuracy_improved=False)
        save_json(state_path,state);write_optimization_report(folder,state,plan);return
    winner=min(eligible,key=lambda s:s['validation']['score']);state['provisional_winner']=winner['name']
    state['state']='confirming';save_json(state_path,state);candidates=[winner['validation']];baselines=[baseline]
    for index,seed in enumerate(plan['confirmation_seeds']):
        parent=next(t for t in bases if t['fold']==first_fold and t['seed']==seed)
        slot=state['confirmation'].setdefault(str(seed),{'name':winner['name']+'_'+str(seed)})
        train(slot,parent,winner['override'],task_index+10+index,['training','validation'])
        candidates.append(slot['validation']);baselines.append(metric(parent))
    if not confirmed(candidates,baselines,plan):
        state.update(state='complete',winner=None,conclusion='Validation screening winner did not replicate; baseline retained.',optimization_accuracy_improved=False)
        save_json(state_path,state);write_optimization_report(folder,state,plan);return
    # Lock selection before requesting any TEST forecast from candidate runs.
    state.update(state='formal_replication',winner=winner['name'],winner_override=winner['override'],optimization_accuracy_improved=True,
        locked_validation_evidence={'candidate':candidates,'baseline':baselines})
    lock_path=folder/'selection_lock.json'
    selected=[winner,*[state['confirmation'][str(seed)] for seed in plan['confirmation_seeds']]]
    evidence=[{'seed':slot['seed'],'run':slot['run'],'validation_metrics_sha256':fingerprint(Path(slot['run'])/'selected_metrics.json'),
        'model_sha256':fingerprint(Path(slot['run'])/'model.pt'),'validation':slot['validation']} for slot in selected]
    baseline_evidence=[]
    for seed,value in zip([plan['screen_seed'],*plan['confirmation_seeds']],baselines):
        task=next(t for t in bases if t['fold']==first_fold and t['seed']==seed)
        baseline_evidence.append({'seed':seed,'run':task['run'],'validation_metrics_sha256':fingerprint(Path(task['run'])/'selected_metrics.json'),
            'model_sha256':fingerprint(Path(task['run'])/'model.pt'),'validation':value})
    if lock_path.exists():
        lock=json.loads(lock_path.read_text(encoding='utf-8'))
        if lock['winner']!=winner['name'] or lock['plan_sha256']!=state['plan_sha256'] or lock['candidate_evidence']!=evidence or lock['baseline_evidence']!=baseline_evidence:
            raise ValueError('locked validation choice or its model evidence changed')
    else:
        if any((Path(slot['run'])/'test_audited_predictions.npz').exists() for slot in selected):raise ValueError('candidate TEST exists before selection lock')
        save_json(lock_path,{'timestamp_utc':datetime.now(timezone.utc).isoformat(),'winner':winner['name'],'override':winner['override'],
            'plan_sha256':state['plan_sha256'],'candidate_evidence':evidence,'baseline_evidence':baseline_evidence,
            'selection_scope':'validation only','TEST_requested_before_lock':False})
    state['selection_lock_sha256']=fingerprint(lock_path)
    save_json(state_path,state)
    for index,parent in enumerate(bases):
        slot=next((s for s in state['formal_tasks'] if s['fold']==parent['fold'] and s['seed']==parent['seed']),None)
        if slot is None:
            existing=winner if parent['fold']==first_fold and parent['seed']==plan['screen_seed'] else state['confirmation'].get(str(parent['seed'])) if parent['fold']==first_fold else None
            slot={'name':winner['name']+'_'+parent['fold']+'_'+str(parent['seed']),'dataset':dataset,'fold':parent['fold'],'seed':parent['seed']}
            if existing:slot.update(run=existing['run'],settings=existing['settings'],role_counts=existing['role_counts'])
            state['formal_tasks'].append(slot);save_json(state_path,state)
        train(slot,parent,winner['override'],task_index+20+index,['training','validation','test'])
    state.update(state='complete',conclusion='Confirmed validation improvement replicated on every original fold/seed; TEST accuracy reported separately, without changing selection.')
    save_json(state_path,state)
    write_optimization_report(folder,state,plan)
