"""Frozen-validation curve selection for the user's v3 reference goal."""
import math
import numpy as np
from battery_smetimes.llm_reporting import raw_curve_numeric_metrics, finite_metric

SCALES = {'mae_pp': 2., 'tail_mae_pp': 3., 'start_error_pp': 1.,
          'endpoint_error_pp': 3., 'stage_drop_max_pp': 3.}
QUALITY = {'mape_percent': 1.5, 'late_mape_percent': 3., 'start_error_pp': 1.,
           'endpoint_error_pp': 3., 'stage_drop_max_pp': 3.}


def curve_metrics_v3(x, y, p):
    result = raw_curve_numeric_metrics(x, y, p)
    x, y, p = [np.asarray(v, dtype=np.float64) for v in (x, y, p)]
    late = x >= x[0] + .8 * (x[-1] - x[0])
    result['late_mape_percent'] = float(100 * np.mean(np.abs(p[late]-y[late])/y[late]))
    return result


def freeze_validation_policy(cells, ids, origin):
    """Choose metric availability from observed sampling, before predictions."""
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('empty or duplicate validation membership')
    missing_stage = []
    for cid in ids:
        cell = cells[cid]
        take = (cell['label_cycle'] > origin) & np.isfinite(cell['soh']) & (cell['soh'] > 0)
        x = np.asarray(cell['label_cycle'][take], dtype=float)
        if len(x) < 2 or np.any(np.diff(x) <= 0):
            raise ValueError('validation has insufficient numeric coverage: ' + cid)
        span = x[-1]-x[0]
        if any(np.count_nonzero((x >= x[0]+j*span/3) &
                               (x <= x[0]+(j+1)*span/3)) < 3 for j in range(3)):
            missing_stage.append(cid)
    active = [key for key in SCALES if key != 'stage_drop_max_pp' or not missing_stage]
    return {'version': 'curve_quality_v3', 'cell_ids': list(ids), 'origin': origin,
            'active_metrics': active, 'scales': SCALES, 'stage_unavailable_cells': missing_stage,
            'stage_evidence_complete': not missing_stage, 'worst_fraction': .2, 'worst_weight': .25}


def validation_score(metrics, policy):
    if set(metrics) != set(policy['cell_ids']):
        raise ValueError('validation prediction membership differs from frozen policy')
    scores = []
    for cid in policy['cell_ids']:
        row = metrics[cid]
        if not all(finite_metric(row, key) for key in policy['active_metrics']):
            raise ValueError('missing or nonfinite validation metric: ' + cid)
        scores.append(float(np.mean([row[k]/policy['scales'][k] for k in policy['active_metrics']])))
    worst_n = max(1, math.ceil(.2*len(scores)))
    return float(np.mean(scores) + .25*np.mean(sorted(scores)[-worst_n:]))


def reference_pass(row):
    return all(finite_metric(row, key) and row[key] <= limit for key, limit in QUALITY.items())


def evaluate_v3(bundle, cells, ids, policy, legacy_evaluate):
    report, arrays, metrics, _ = legacy_evaluate(bundle, cells, ids)
    origin = policy['origin']
    for cid in ids:
        prefix = f'{cid}__{origin}__'
        metrics[cid].update(curve_metrics_v3(*(arrays[prefix+k] for k in ('cycle','truth','prediction'))))
    return report, arrays, metrics, (validation_score(metrics, policy),)
