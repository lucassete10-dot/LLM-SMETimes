"""Read-only audit of XJTU feature selection and stage semantics.

Checks whether target-capacity filtering actually removed otherwise usable
diagnostics, and whether stage 0/2 have expected charge/discharge behavior.
No prepared file or frozen model is modified.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from prepare_xjtu import SOURCE, diagnostic, stage_features


OUT = Path('battery_models_v2/xjtu_feature_selection_audit.json')
PROCESSED = Path(r'D:\LLM\XJTU\processed')


def stage_summary(cycle, stage):
    time = np.asarray(cycle['relative_time_min'], dtype='float64').reshape(-1)
    cuts = np.flatnonzero(time == 0)
    if len(cuts) <= stage:
        return None
    start = cuts[stage]
    stop = cuts[stage + 1] if stage + 1 < len(cuts) else len(time)
    current = np.asarray(cycle['current_A'], dtype='float64').reshape(-1)[start:stop]
    voltage = np.asarray(cycle['voltage_V'], dtype='float64').reshape(-1)[start:stop]
    if not len(current) or not np.isfinite(current).all() or not np.isfinite(voltage).all():
        return None
    return float(np.median(current)), float(voltage[-1] - voltage[0])


def audit_cell(path, batch):
    mat = loadmat(path, simplify_cells=True)
    cycles = mat['data']
    capacity = np.asarray(mat['summary']['discharge_capacity_Ah'], dtype='float64').reshape(-1)
    if len(cycles) != len(capacity):
        raise ValueError(f'{path}: capacity/cycle mismatch')
    with np.load(PROCESSED / f'b{batch}_{path.stem}.npz') as prepared:
        accepted = set(prepared['cycle'].astype(int))
    counts = Counter()
    measurements = defaultdict(list)
    qualified = set()
    rejected_capacity = []
    for index, cycle in enumerate(cycles, 1):
        counts['source_cycles'] += 1
        if not diagnostic(batch, cycle['description']):
            continue
        counts['protocol_diagnostics'] += 1
        if stage_features(cycle, 0) is None or stage_features(cycle, 2) is None:
            counts['signal_incomplete'] += 1
            continue
        counts['signal_complete'] += 1
        q = capacity[index - 1]
        if not np.isfinite(q) or not .8 < q < 2.4:
            counts['capacity_rejected'] += 1
            rejected_capacity.append({'cycle': index, 'capacity_ah':
                                      float(q) if np.isfinite(q) else None})
            continue
        qualified.add(index)
        charge = stage_summary(cycle, 0)
        discharge = stage_summary(cycle, 2)
        if charge is None or discharge is None:
            raise ValueError(f'{path}: accepted stage summary missing at cycle {index}')
        measurements['stage0_median_current_a'].append(charge[0])
        measurements['stage2_median_current_a'].append(discharge[0])
        measurements['stage0_voltage_change_v'].append(charge[1])
        measurements['stage2_voltage_change_v'].append(discharge[1])
    if qualified != accepted:
        raise ValueError(f'{path}: audited vs prepared cycle mismatch')
    counts['accepted'] = len(accepted)
    return {'cell_id': f'b{batch}_{path.stem}', 'counts': dict(counts),
            'capacity_rejected_rows': rejected_capacity,
            'stage_medians': {key: float(np.median(values)) for key, values in measurements.items()},
            'stage_sign_fractions': {
                'stage0_current_positive': float(np.mean(np.asarray(measurements[
                    'stage0_median_current_a']) > 0)),
                'stage2_current_negative': float(np.mean(np.asarray(measurements[
                    'stage2_median_current_a']) < 0)),
                'stage0_voltage_increases': float(np.mean(np.asarray(measurements[
                    'stage0_voltage_change_v']) > 0)),
                'stage2_voltage_decreases': float(np.mean(np.asarray(measurements[
                    'stage2_voltage_change_v']) < 0)),
            }}


def main():
    batches = {}
    for batch in range(1, 7):
        cells = [audit_cell(path, batch) for path in sorted((SOURCE / f'Batch-{batch}').glob('*.mat'))]
        if len(cells) != (15 if batch == 2 else 8):
            raise ValueError(f'Batch {batch}: unexpected source cell count')
        totals = {name: sum(cell['counts'].get(name, 0) for cell in cells)
                  for name in ('source_cycles', 'protocol_diagnostics', 'signal_incomplete',
                               'signal_complete', 'capacity_rejected', 'accepted')}
        batches[f'batch{batch}'] = {'cells': len(cells), 'totals': totals,
                                   'min_stage_sign_fractions': {key: min(
                                       cell['stage_sign_fractions'][key] for cell in cells)
                                       for key in cells[0]['stage_sign_fractions']},
                                   'cells_detail': cells}
        print('batch', batch, totals, batches[f'batch{batch}']['min_stage_sign_fractions'],
              flush=True)
    report = {'purpose': 'read-only XJTU selection and stage audit',
              'capacity_filter': 'prepared sample included only when 0.8 < target Ah < 2.4',
              'batches': batches}
    OUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Wrote', OUT)


if __name__ == '__main__':
    main()
