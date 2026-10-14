"""Fixed age-weighted blend of two whole-strategy OOF XJTU EOL routes.

Only existing held-out predictions are combined. The weight depends on
observed source-cycle age, never on true lifetime or validation capacity.
The rule was motivated by development-fold results, so the same folds cannot
serve as an independent confirmation of a selected model.
"""

import json
from pathlib import Path

import numpy as np


MANIFEST = Path('battery_data_v2/manifest.json')
DIRECT = Path('battery_models_v2/xjtu_throughput_rul_challenger/development_oof_throughput_full.npz')
FRACTION = Path('battery_models_v2/xjtu_fractional_life_challenger/development_oof_endpoint.npz')
OUT = Path('battery_models_v2/xjtu_age_blend_challenger/development_cv.json')
LANDMARKS = (50, 100, 200, 300, 500)
AGE_SCALE = 100.0


def blend_endpoints(cycle, direct_rul, fractional_endpoint, age_scale=AGE_SCALE):
    cycle = np.asarray(cycle, dtype=float)
    direct = cycle + np.asarray(direct_rul, dtype=float)
    fractional = np.asarray(fractional_endpoint, dtype=float)
    if not (cycle.shape == direct.shape == fractional.shape) or age_scale <= 0:
        raise ValueError('aligned predictions and positive age scale required')
    weight = cycle/(cycle+age_scale)
    return np.maximum(cycle, (1-weight)*direct + weight*fractional)


def read_oof(path, field):
    with np.load(path) as source:
        ids = source['cell_id'].astype(str)
        cycles = source['cycle'].astype(int)
        value = source[field].astype(float)
    return {(key, int(cycle)): float(v) for key, cycle, v in zip(ids, cycles, value)}


def main():
    records = [row for row in json.loads(MANIFEST.read_text(encoding='utf-8'))['cells']
               if row['dataset'] == 'XJTU']
    direct = read_oof(DIRECT, 'estimate')
    fraction = read_oof(FRACTION, 'endpoint')
    if set(direct) != set(fraction):
        raise ValueError('component OOF prediction rows do not align')
    by_cell = []
    for record in records:
        key = record['cell_id']
        endpoint = record['endpoint_cycle']
        cycles = np.array(sorted(cycle for cell_id, cycle in direct if cell_id == key))
        if not len(cycles):
            continue
        source_direct = np.array([direct[key, int(cycle)] for cycle in cycles])
        source_fraction = np.array([fraction[key, int(cycle)] for cycle in cycles])
        source_blend = blend_endpoints(cycles, source_direct, source_fraction)
        origin = int(cycles[0])
        if origin >= endpoint:
            continue
        row = {'cell_id': key, 'group': record['group'],
               'origin_cycle': origin, 'true_endpoint': endpoint,
               'landmarks': {}}
        for name, prediction in (('direct', cycles+source_direct),
                                 ('fraction', source_fraction),
                                 ('blend', source_blend)):
            row[f'{name}_origin_ape_percent'] = float(100*abs(
                prediction[0]-endpoint)/endpoint)
            row[f'{name}_all_cycle_mae'] = float(np.mean(abs(prediction-endpoint)))
        for landmark in LANDMARKS:
            if endpoint <= landmark:
                continue
            ix = np.flatnonzero(cycles <= landmark)
            if not len(ix):
                continue
            i = int(ix[-1])
            row['landmarks'][str(landmark)] = {
                'last_observed_cycle': int(cycles[i]),
                'direct_ape_percent': float(100*abs(
                    cycles[i]+source_direct[i]-endpoint)/endpoint),
                'fraction_ape_percent': float(100*abs(
                    source_fraction[i]-endpoint)/endpoint),
                'blend_ape_percent': float(100*abs(
                    source_blend[i]-endpoint)/endpoint)}
        by_cell.append(row)
    summary = {'scored_cells': len(by_cell)}
    for metric in ('origin_ape_percent', 'all_cycle_mae'):
        summary[metric] = {name: float(np.mean([
            row[f'{name}_{metric}'] for row in by_cell]))
            for name in ('direct', 'fraction', 'blend')}
    summary['landmarks'] = {}
    for landmark in LANDMARKS:
        subset = [row['landmarks'][str(landmark)] for row in by_cell
                  if str(landmark) in row['landmarks']]
        summary['landmarks'][str(landmark)] = {
            'scored_cells': len(subset),
            **{name: float(np.mean([r[f'{name}_ape_percent'] for r in subset]))
               if subset else None for name in ('direct', 'fraction', 'blend')}}
    report = {'status': 'unpromoted development-only route combination; not independent test',
              'age_scale_cycles': AGE_SCALE,
              'rule': 'fraction_weight = observed_cycle/(observed_cycle+age_scale)',
              'input': 'the two existing whole-strategy OOF 600s-prefix-and-usage predictions',
              'endpoint_warning': 'source XJTU EOL, not exact per-cell 80% crossing',
              'summary': summary, 'by_cell': by_cell}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('SUMMARY', summary)
    print('WROTE', OUT)


if __name__ == '__main__':
    main()
