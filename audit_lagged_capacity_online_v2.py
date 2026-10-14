"""Causal previous-full-discharge SOH persistence, as a distinct online mode.

This is an observation-availability diagnostic, not a signal-prefix model.
At the start of each eligible discharge it uses the capacity measured during
the preceding *completed* diagnostic, never the current or future capacity.
It is deployable only when the preceding complete discharge was actually
measured and its capacity was available at the prediction time.
"""

import json
from pathlib import Path

import numpy as np

from battery_v2_features import load_manifest


OUT = Path('battery_models_v2/lagged_capacity_online_audit.json')


def score_cell(record):
    with np.load(record['labels_file']) as archive:
        cycle = np.asarray(archive['cycle'], dtype=int)
        capacity = np.asarray(archive['capacity_ah'], dtype=float)
        target = np.asarray(archive['soh'], dtype=float)
        eligible = np.asarray(archive['soh_available'], dtype=bool)
    reference_count = int(record['reference_measurement_count'])
    reference = float(record['reference_capacity_ah'])
    if not np.all(np.flatnonzero(eligible) >= reference_count):
        raise ValueError(f"{record['cell_id']}: calibration row eligible")
    row = np.flatnonzero(eligible)
    previous = row - 1
    if not np.all(previous >= 0) or not np.all(cycle[previous] < cycle[row]):
        raise ValueError(f"{record['cell_id']}: previous capacity not causal")
    prediction = capacity[previous] / reference
    error = 100 * np.abs(prediction - target[row]) / target[row]
    gap = cycle[row] - cycle[previous]
    return {
        'cell_id': record['cell_id'],
        'group': record['group'],
        'scored_observations': int(len(row)),
        'mape_percent': float(error.mean()),
        'mae_soh': float(np.mean(np.abs(prediction - target[row]))),
        'consecutive_cycle_rows': int(np.sum(gap == 1)),
        'consecutive_cycle_mape_percent': float(error[gap == 1].mean())
        if np.any(gap == 1) else None,
        'gapped_cycle_rows': int(np.sum(gap > 1)),
        'gapped_cycle_mape_percent': float(error[gap > 1].mean())
        if np.any(gap > 1) else None,
        'first_prediction_cycle': int(cycle[row[0]]),
        'first_previous_diagnostic_cycle': int(cycle[previous[0]]),
        'first_prediction_ape_percent': float(error[0]),
        'maximum_observation_gap_cycles': int(gap.max()),
    }


def summarize(rows):
    return {
        'cells': len(rows),
        'cell_macro_mape_percent': float(np.mean([r['mape_percent'] for r in rows])),
        'worst_cell_mape_percent': float(max(r['mape_percent'] for r in rows)),
        'cells_below_0_5_percent': sum(r['mape_percent'] <= .5 for r in rows),
        'scored_observations': sum(r['scored_observations'] for r in rows),
        'gapped_cycle_rows': sum(r['gapped_cycle_rows'] for r in rows),
    }


def main():
    manifest = load_manifest()
    domains = {}
    for domain in ('NASA', 'CALCE', 'MIT', 'XJTU'):
        rows = [score_cell(record) for record in manifest['cells']
                if record['dataset'] == domain]
        domains[domain] = {
            **summarize(rows),
            'by_group': {group: summarize([r for r in rows if r['group'] == group])
                         for group in sorted({r['group'] for r in rows})},
            'by_cell': rows,
        }
    report = {
        'status': 'causal diagnostic, not trained model or independent test',
        'inference_input': 'previous completed full-discharge measured capacity '
                           'and previously established per-cell reference',
        'observation_requirement': 'all immediately preceding accepted '
                                   'diagnostics must have been measured before inference',
        'warning': 'not comparable to short-prefix signal-only model when prior '
                   'complete-discharge capacity is unavailable in deployment',
        'domains': domains,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    for domain, scored in domains.items():
        print(domain, 'cells', scored['cells'], 'macro MAPE',
              round(scored['cell_macro_mape_percent'], 4), '%',
              'cells <=0.5%', scored['cells_below_0_5_percent'],
              'gapped rows', scored['gapped_cycle_rows'], flush=True)
    print('WROTE', OUT)


if __name__ == '__main__':
    main()
