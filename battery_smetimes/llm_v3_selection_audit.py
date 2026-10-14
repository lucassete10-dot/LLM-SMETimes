"""Read-only arithmetic/identity check of a v3 selected validation checkpoint."""
import json
from pathlib import Path
import numpy as np
from battery_smetimes.llm_curve_quality_v3 import (
    freeze_validation_policy, curve_metrics_v3, validation_score)


def audit_selection(run, cells, plan):
    run=Path(run)
    read=lambda name:json.loads((run/name).read_text(encoding='utf-8'))
    cfg=read('config.json');policy=read('validation_policy.json')
    if cfg.get('selection_version')!='curve_quality_v3':raise ValueError('not a v3 selection run')
    expected=freeze_validation_policy(cells,plan['validation'],cfg['origin'])
    if policy!=expected or cfg.get('validation_selection_policy')!=expected:
        raise ValueError('frozen validation policy changed')
    metrics={}
    with np.load(run/'selected_predictions.npz',allow_pickle=False) as data:
        for cid in policy['cell_ids']:
            prefix=f'{cid}__{cfg["origin"]}__'
            metrics[cid]=curve_metrics_v3(*(data[prefix+k] for k in ('cycle','truth','prediction')))
    selected=read('selected_metrics.json')
    if set(selected)!=set(metrics):raise ValueError('selected validation membership changed')
    for cid,actual in metrics.items():
        for key,value in actual.items():
            if value is None:
                if selected[cid].get(key) is not None:raise ValueError('missing stage replaced by value')
            elif isinstance(value,float):
                np.testing.assert_allclose(selected[cid][key],value,rtol=0,atol=1e-10)
    score=validation_score(metrics,policy);selection=read('selection.json');result=read('result.json')
    np.testing.assert_allclose(selection['score'],[score],rtol=0,atol=1e-10)
    if selection['scope']!='development validation' or selection['epoch']!=result['selected_epoch']:
        raise ValueError('selected scope/epoch differs')
    history=[r for r in read('training_history.json') if 'validation_quality_score' in r]
    if not history:raise ValueError('missing validation score history')
    best=min(history,key=lambda r:(r['validation_quality_score'],r['epoch']))
    if best['epoch']!=selection['epoch']:raise ValueError('selected epoch is not earliest minimum validation score')
    np.testing.assert_allclose(best['validation_quality_score'],score,rtol=0,atol=1e-10)
    return {'validation_score':score,'selected_epoch':selection['epoch'],
            'frozen_validation_policy_verified':True,'selected_raw_arithmetic_verified':True,
            'earliest_minimum_score_verified':True,'stage_evidence_complete':policy['stage_evidence_complete']}
