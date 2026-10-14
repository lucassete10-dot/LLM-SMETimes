"""Finite v3 experiment decisions. Only validation evidence enters selection."""
import math
import numpy as np
from battery_smetimes.llm_curve_quality_v3 import reference_pass, validation_score
from battery_smetimes.llm_reporting import finite_metric

SEEDS = (20260930, 20260931, 20260932)
OVERRIDES = {'R0': {}, 'A': {'local_readout':'query'},
             'B': {'numeric_lora_rank':4}, 'C': {'local_recovery_max_soh':.005},
             'D': {'point_huber_beta_pp':.5}}


def summarize_validation(evidence):
    rows = evidence['by_cell']
    score = validation_score(rows, evidence['policy'])
    if not all(finite_metric(r,'mape_percent') and finite_metric(r,'mae_pp') for r in rows.values()):
        raise ValueError('invalid validation coverage')
    return {'score':score, 'cells':len(rows), 'mape_passes':sum(r['mape_percent']<=3 for r in rows.values()),
            'worst_mae_pp':max(r['mae_pp'] for r in rows.values())}


def protection(candidate, reference):
    return (candidate['cells']==reference['cells'] and
            candidate['mape_passes']>=reference['mape_passes'] and
            candidate['worst_mae_pp']<=1.1*reference['worst_mae_pp'])


def screen_pass(candidate, reference):
    return candidate['score']<reference['score'] and candidate['score']<=.95*reference['score'] and protection(candidate, reference)


def confirmation_pass(candidates, references):
    if len(candidates)!=3 or len(references)!=3:
        raise ValueError('confirmation requires three matched seeds')
    return (sum(c['score']<r['score'] for c,r in zip(candidates,references))>=2 and
            np.mean([c['score'] for c in candidates])<=.95*np.mean([r['score'] for r in references]) and
            all(protection(c,r) for c,r in zip(candidates,references)))


def work_goal(evidences):
    """One seed is preliminary; three matched seeds establish stability."""
    if len(evidences) not in (1,3):
        raise ValueError('goal evidence requires one or three seeds')
    ids=set(evidences[0]['policy']['cell_ids'])
    if any(set(e['by_cell'])!=ids for e in evidences):
        raise ValueError('goal membership differs between seeds')
    primary=quality=0
    for cid in ids:
        rows=[e['by_cell'][cid] for e in evidences]
        if any(not finite_metric(r,'mape_percent') or r['mape_percent']>10 for r in rows):
            return False
        primary+=all(r['mape_percent']<=3 for r in rows)
        quality+=(all(r['mape_percent']<=3 for r in rows) and
                  sum(reference_pass(r) for r in rows)>=(1 if len(rows)==1 else 2))
    return primary>=math.ceil(.9*len(ids)) and quality>=math.ceil(.8*len(ids))


def trial_key(config, fold, seed, epochs):
    return f'{config}:{fold}:{seed}:{epochs}'


def new_dataset_state(dataset, folds, r0_evidence, r0_run, noise_evidence):
    if not folds or len(folds)!=len(set(folds)):
        raise ValueError('invalid frozen folds')
    summarize_validation(r0_evidence)
    # The trigger is fixed before any conditional candidate is trained.
    enabled=bool(noise_evidence.get('training_validation_mae_ratio',0)>=2 and
                 noise_evidence.get('noisy_training_cell_fraction',0)>=.2)
    key=trial_key('R0',folds[0],SEEDS[0],120)
    return {'dataset':dataset,'folds':list(folds),'phase':'screen',
        'trials':{key:{'state':'complete','run':r0_run,'evidence':r0_evidence}},
        'screen_order':['A','B']+(['C','D'] if enabled else []),
        'conditional_noise_evidence':noise_evidence,'conditional_candidates_enabled':enabled,
        'early_R0_confirmation_attempted':False,'winner':None,'selection_lock':None,
        'formal_slots':{},'goal_complete':False}


def plan_dataset_step(state):
    """Return one fit/audit request or None. Completed fits are always reused."""
    first=state['folds'][0]
    def need(config,fold,seed,epochs):
        key=trial_key(config,fold,seed,epochs)
        trial=state['trials'].get(key)
        if trial is None or trial['state']!='complete':
            return {'kind':'fit','key':key,'config':config,'override':OVERRIDES[config],
                    'fold':fold,'seed':seed,'epochs':epochs,'roles':['training','validation']}
        return None
    def evidence(config,seed,epochs):
        return state['trials'][trial_key(config,first,seed,epochs)]['evidence']
    for _ in range(12):
        phase=state['phase']
        if phase=='complete': return None
        if phase=='screen':
            base=evidence('R0',SEEDS[0],120)
            if not state['early_R0_confirmation_attempted'] and work_goal([base]):
                state['early_R0_confirmation_attempted']=True
                state['phase']='confirm_R0_early';continue
            for config in state['screen_order']:
                request=need(config,first,SEEDS[0],120)
                if request:return request
            summary=summarize_validation(base)
            eligible=[config for config in state['screen_order'] if screen_pass(
                summarize_validation(evidence(config,SEEDS[0],120)),summary)]
            state['screen_winner']=min(eligible,key=lambda c:(summarize_validation(
                evidence(c,SEEDS[0],120))['score'],state['screen_order'].index(c))) if eligible else None
            state['phase']='confirm';continue
        if phase in ('confirm','confirm_R0_early'):
            for seed in SEEDS:
                request=need('R0',first,seed,240)
                if request:return request
            refs=[evidence('R0',seed,240) for seed in SEEDS]
            if phase=='confirm_R0_early' and not work_goal(refs):
                state['phase']='screen';continue
            candidate=state.get('screen_winner') if phase=='confirm' else None
            if candidate:
                for seed in SEEDS:
                    request=need(candidate,first,seed,240)
                    if request:return request
                accepted=confirmation_pass([summarize_validation(evidence(candidate,s,240)) for s in SEEDS],
                                           [summarize_validation(e) for e in refs])
                state['confirmation_improved']=bool(accepted)
            else: accepted=False
            state['winner']=candidate if accepted else 'R0'
            chosen=[evidence(state['winner'],seed,240) for seed in SEEDS]
            state['selection_lock']={'winner':state['winner'],'seeds':list(SEEDS),
                'validation_evidence':chosen,'test_used_for_selection':False,
                'curve_work_goal_reached':work_goal(chosen)}
            state['phase']='formal';continue
        if phase=='formal':
            if state['selection_lock'] is None:raise ValueError('formal evaluation requires selection lock')
            for fold in state['folds']:
                for seed in SEEDS:
                    request=need(state['winner'],fold,seed,240)
                    if request:return request
                    key=trial_key(state['winner'],fold,seed,240)
                    if not state['formal_slots'].get(key,{}).get('test_verified'):
                        return {'kind':'evaluate_test','key':key,'fold':fold,'seed':seed,
                                'roles':['training','validation','test']}
            state['phase']='complete';state['training_and_evaluation_complete']=True
            return None
        raise ValueError('unknown v3 phase: '+phase)
    raise RuntimeError('v3 phase transition loop')


def next_round_robin_action(workflow):
    sequence=workflow['sequence'];start=(workflow.get('last_dataset_index',-1)+1)%len(sequence)
    for offset in range(len(sequence)):
        index=(start+offset)%len(sequence);dataset=sequence[index]
        action=plan_dataset_step(workflow['datasets'][dataset])
        if action:
            return {**action,'dataset':dataset,'dataset_index':index}
    return None
