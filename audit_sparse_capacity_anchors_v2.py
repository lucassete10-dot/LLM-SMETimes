"""Measure sensitivity of the conditional online SOH route to anchor cadence.

At a prediction point, the latest *previously completed* capacity diagnostic
may be used. A current diagnostic is never used to predict itself; if it is
scheduled as an anchor, it becomes available only after that prediction.
This is not the strict short-prefix, signal-only benchmark.
"""

import json
from pathlib import Path

import numpy as np

from battery_v2_features import load_manifest


OUT = Path('battery_models_v2/sparse_capacity_anchor_audit.json')
CADENCES = (1, 5, 10, 25, 50, 100)


def predict_with_sparse_anchors(cycle, capacity, reference_count, reference,
                                cadence):
    cycle = np.asarray(cycle, dtype=int)
    capacity = np.asarray(capacity, dtype=float)
    if (len(cycle) != len(capacity) or not 1 <= reference_count < len(cycle)
            or np.any(np.diff(cycle) <= 0) or cadence < 1):
        raise ValueError('invalid ordered diagnostics or anchor cadence')
    anchor_index = reference_count - 1
    prediction = np.empty(len(cycle) - reference_count, dtype=float)
    anchor_age = np.empty_like(prediction, dtype=int)
    updates = 0
    for output_index, index in enumerate(range(reference_count, len(cycle))):
        # The current cycle's capacity becomes known only after its estimate.
        prediction[output_index] = capacity[anchor_index] / reference
        anchor_age[output_index] = cycle[index] - cycle[anchor_index]
        if anchor_age[output_index] >= cadence:
            anchor_index = index
            updates += 1
    return prediction, anchor_age, updates


def score_cell(record):
    with np.load(record['labels_file']) as archive:
        cycle = np.asarray(archive['cycle'], dtype=int)
        capacity = np.asarray(archive['capacity_ah'], dtype=float)
        target = np.asarray(archive['soh'], dtype=float)
        available = np.asarray(archive['soh_available'], dtype=bool)
    reference_count = int(record['reference_measurement_count'])
    if not np.array_equal(np.flatnonzero(available),
                          np.arange(reference_count, len(cycle))):
        raise ValueError(f"{record['cell_id']}: unsupported availability mask")
    scores = {}
    for cadence in CADENCES:
        estimate, age, updates = predict_with_sparse_anchors(
            cycle, capacity, reference_count,
            float(record['reference_capacity_ah']), cadence)
        true = target[available]
        scores[str(cadence)] = {
            'mape_percent': float(np.mean(100 * np.abs(estimate - true) / true)),
            'anchor_updates': updates,
            'median_anchor_age_cycles': float(np.median(age)),
            'maximum_anchor_age_cycles': int(age.max()),
        }
    return {'cell_id': record['cell_id'], 'group': record['group'],
            'scored_observations': int(available.sum()), 'cadences': scores}


def summarize(rows):
    return {str(cadence): {
        'cells': len(rows),
        'cell_macro_mape_percent': float(np.mean([
            r['cadences'][str(cadence)]['mape_percent'] for r in rows])),
        'cells_below_0_5_percent': sum(
            r['cadences'][str(cadence)]['mape_percent'] <= .5 for r in rows),
        'total_anchor_updates': sum(
            r['cadences'][str(cadence)]['anchor_updates'] for r in rows),
    } for cadence in CADENCES}


def main():
    manifest = load_manifest()
    domains = {}
    for domain in ('NASA', 'CALCE', 'MIT', 'XJTU'):
        rows = [score_cell(record) for record in manifest['cells']
                if record['dataset'] == domain]
        domains[domain] = {
            'summary': summarize(rows),
            'by_group': {group: summarize([r for r in rows if r['group'] == group])
                         for group in sorted({r['group'] for r in rows})},
            'by_cell': rows,
        }
        print(domain, {key: round(value['cell_macro_mape_percent'], 3)
                       for key, value in domains[domain]['summary'].items()},
              flush=True)
    report = {
        'status': 'causal conditional-mode development diagnostic; no model training',
        'anchor_definition': 'completed full-discharge capacity at or after '
                             'the requested minimum cycle interval',
        'current_cycle_never_used_for_own_prediction': True,
        'warning': 'requires previous completed-capacity measurements and is '
                   'not interchangeable with the signal-only prefix score',
        'cadence_cycles': CADENCES,
        'domains': domains,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('WROTE', OUT)


if __name__ == '__main__':
    main()
