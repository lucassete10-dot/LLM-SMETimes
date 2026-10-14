"""Read-only landmark audit of stored whole-strategy-held-out RUL predictions.

For each landmark, only a diagnostic already observed at or before that
source cycle is used. Cells already at EOL are excluded from the at-risk set.
This is not training, calibration, or independent testing.
"""

import json
from pathlib import Path

import numpy as np


MANIFEST = Path('battery_data_v2/manifest.json')
BASELINE = Path('battery_models_v2/rul/xjtu/development_oof_predictions.npz')
THROUGHPUT = Path('battery_models_v2/xjtu_throughput_rul_challenger/development_oof_throughput_full.npz')
OUT = Path('battery_models_v2/xjtu_landmark_rul_audit.json')
LANDMARKS = (50, 100, 200, 300, 500)


def read_predictions(path, field):
    with np.load(path) as source:
        ids = source['cell_id'].astype(str)
        cycles = source['cycle'].astype(int)
        estimates = source[field].astype(float)
    by_cell = {}
    for key, cycle, estimate in zip(ids, cycles, estimates):
        by_cell.setdefault(key, []).append((int(cycle), float(estimate)))
    return {key: sorted(rows) for key, rows in by_cell.items()}


def at_landmark(rows, landmark):
    valid = [(cycle, rul) for cycle, rul in rows if cycle <= landmark]
    return valid[-1] if valid else None


def main():
    records = [row for row in json.loads(MANIFEST.read_text(encoding='utf-8'))['cells']
               if row['dataset'] == 'XJTU']
    baseline = read_predictions(BASELINE, 'estimate')
    throughput = read_predictions(THROUGHPUT, 'estimate')
    report = {'status': 'read-only development OOF landmark audit, not independent test',
              'landmarks': {}}
    for landmark in LANDMARKS:
        at_risk = [row for row in records if row['endpoint_cycle'] > landmark]
        scored = []
        for row in at_risk:
            key = row['cell_id']
            if key not in baseline or key not in throughput:
                continue
            pair = at_landmark(baseline[key], landmark)
            extended = at_landmark(throughput[key], landmark)
            if pair is None or extended is None:
                continue
            if pair[0] != extended[0]:
                raise ValueError(f'{key}: baseline and throughput cycles differ')
            cycle = pair[0]
            endpoint = row['endpoint_cycle']
            scored.append({'cell_id': key, 'group': row['group'],
                           'last_observed_cycle': cycle, 'true_endpoint_cycle': endpoint,
                           'baseline_ape_percent': float(100*abs(cycle+pair[1]-endpoint)/endpoint),
                           'throughput_ape_percent': float(100*abs(
                               cycle+extended[1]-endpoint)/endpoint)})
        groups = sorted({row['group'] for row in at_risk})
        report['landmarks'][str(landmark)] = {
            'at_risk_cells': len(at_risk), 'scored_cells': len(scored),
            'baseline_cell_macro_mape_percent': float(np.mean([
                row['baseline_ape_percent'] for row in scored])) if scored else None,
            'throughput_cell_macro_mape_percent': float(np.mean([
                row['throughput_ape_percent'] for row in scored])) if scored else None,
            'by_group': {group: {
                'at_risk_cells': sum(row['group'] == group for row in at_risk),
                'scored_cells': sum(row['group'] == group for row in scored),
                'baseline_mape_percent': float(np.mean([
                    row['baseline_ape_percent'] for row in scored
                    if row['group'] == group])) if any(
                        row['group'] == group for row in scored) else None,
                'throughput_mape_percent': float(np.mean([
                    row['throughput_ape_percent'] for row in scored
                    if row['group'] == group])) if any(
                        row['group'] == group for row in scored) else None}
                for group in groups},
            'by_cell': scored}
        print(landmark, 'coverage', len(scored), '/', len(at_risk),
              'baseline', report['landmarks'][str(landmark)]['baseline_cell_macro_mape_percent'],
              'throughput', report['landmarks'][str(landmark)]['throughput_cell_macro_mape_percent'])
    OUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('WROTE', OUT)


if __name__ == '__main__':
    main()
