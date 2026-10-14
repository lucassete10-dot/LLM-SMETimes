"""MIT cross-batch SOH ablation adding known charging-policy rates.

The two commanded charge-rate steps are metadata known before the current
discharge. They are constant for a cell and do not use endpoint or capacity.
The same whole-batch folds and age-model algorithm are used for comparison.
"""

import json
from pathlib import Path

import numpy as np

from battery_v2_features import load_cell, load_manifest
from evaluate_paper_inspired_soh_v2 import fit_gated
from evaluate_soh_age_signal_v2 import age_matrix
from train_soh_v2 import outer_splits


ROOT = Path('battery_models_v2/soh_age_signal_challenger')
BASE = ROOT / 'mit_development_oof_predictions.npz'
OUT = ROOT / 'mit_policy_soh_development_cv.json'


def policy_matrix(cell, record):
    steps = record.get('charge_rate_steps_c')
    if not steps or not 1 <= len(steps) <= 2:
        raise ValueError(f'{cell.cell_id}: missing or invalid charging policy')
    first = float(steps[0])
    second = float(steps[-1])
    policy = np.tile([first, second, second-first], (len(cell.cycle), 1))
    return np.column_stack((age_matrix(cell), policy))


def main():
    records = [row for row in load_manifest()['cells'] if row['dataset'] == 'MIT']
    cells = {row['cell_id']: load_cell(row) for row in records}
    matrices = {row['cell_id']: policy_matrix(cells[row['cell_id']], row)
                for row in records}
    with np.load(BASE) as file:
        baseline = {(str(key), int(cycle)): float(pred)
                    for key, cycle, pred in zip(file['cell_id'], file['cycle'],
                                                file['gated_with_age'])}
    folds = []
    rows = []
    for group, training, validation in outer_splits(records, 'MIT'):
        imputer, model = fit_gated(cells, training, matrices)
        for key in validation:
            cell = cells[key]
            target = cell.soh[cell.soh_available]
            candidate = model.predict(imputer.transform(matrices[key]))[cell.soh_available]
            original = np.array([baseline[key, int(cycle)] for cycle in cell.cycle])[
                cell.soh_available]
            rows.append({'cell_id': key, 'held_out_batch': group,
                         'baseline_mape_percent': float(100*np.mean(
                             np.abs(original-target)/target)),
                         'candidate_mape_percent': float(100*np.mean(
                             np.abs(candidate-target)/target))})
        group_rows = [row for row in rows if row['held_out_batch'] == group]
        folds.append({'held_out_batch': group, 'scored_cells': len(group_rows),
                      'baseline_mape_percent': float(np.mean([
                          row['baseline_mape_percent'] for row in group_rows])),
                      'candidate_mape_percent': float(np.mean([
                          row['candidate_mape_percent'] for row in group_rows]))})
        print(group, folds[-1]['baseline_mape_percent'],
              folds[-1]['candidate_mape_percent'], flush=True)
    report = {'status': 'unpromoted MIT whole-batch charging-policy SOH ablation',
              'feature': 'declared charge-rate steps, current/past 300 s signal, observed cycle',
              'summary': {'scored_cells': len(rows),
                          'baseline_mape_percent': float(np.mean([
                              row['baseline_mape_percent'] for row in rows])),
                          'candidate_mape_percent': float(np.mean([
                              row['candidate_mape_percent'] for row in rows]))},
              'folds': folds, 'by_cell': rows}
    OUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('SUMMARY', report['summary'])


if __name__ == '__main__':
    main()
