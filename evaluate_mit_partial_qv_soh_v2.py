"""MIT whole-batch online SOH ablation with reconstructed 300 s Q(V) samples.

All new values are computed from the existing 11 voltage/current samples at
0..300 s of each discharge. No full-cycle capacity, SOH, or future-cycle
measurement enters the feature matrix. Reference is the observed initial
diagnostic of the same cell.
"""

import json
import warnings
from pathlib import Path

import numpy as np

from battery_v2_features import load_cell, load_manifest
from evaluate_paper_inspired_soh_v2 import fit_gated
from evaluate_soh_age_signal_v2 import age_matrix
from train_soh_v2 import outer_splits


OUT = Path('battery_models_v2/soh_age_signal_challenger')
BASE = OUT / 'mit_development_oof_predictions.npz'
GRID = np.linspace(3.18, 3.50, 9)


def qv_from_300s_sample(raw_row):
    raw_row = np.asarray(raw_row, dtype=float)
    if len(raw_row) != 51:
        raise ValueError('MIT enhanced prefix must have 51 columns')
    voltage, current = raw_row[:11], raw_row[11:22]
    if not (np.isfinite(voltage).all() and np.isfinite(current).all()):
        return np.full(len(GRID), np.nan)
    increments = .5*(np.abs(current[1:])+np.abs(current[:-1]))*30/3600
    capacity = np.r_[0., np.cumsum(increments)]
    order = np.argsort(voltage)
    v, q = voltage[order], capacity[order]
    unique, first, count = np.unique(v, return_index=True, return_counts=True)
    averaged = np.add.reduceat(q, first)/count
    return np.interp(GRID, unique, averaged, left=np.nan, right=np.nan)


def augmented_matrix(cell):
    qv = np.stack([qv_from_300s_sample(row) for row in cell.raw])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        reference = np.nanmedian(qv[:cell.reference_count], axis=0)
    return np.column_stack((age_matrix(cell), qv, qv-reference))


def main():
    records = [row for row in load_manifest()['cells'] if row['dataset'] == 'MIT']
    cells = {row['cell_id']: load_cell(row) for row in records}
    matrices = {key: augmented_matrix(cell) for key, cell in cells.items()}
    with np.load(BASE) as source:
        base = {(str(key), int(cycle)): float(value)
                for key, cycle, value in zip(source['cell_id'], source['cycle'],
                                             source['gated_with_age'])}
    folds = []
    oof = {}
    for group, train, validation in outer_splits(records, 'MIT'):
        bundle = fit_gated(cells, train, matrices)
        rows = []
        for key in validation:
            cell = cells[key]
            candidate = bundle[1].predict(bundle[0].transform(matrices[key]))
            oof[key] = candidate
            baseline = np.array([base[key, int(cycle)] for cycle in cell.cycle])
            keep = cell.soh_available
            target = cell.soh[keep]
            rows.append({'cell_id': key,
                         'baseline_mape_percent': float(100*np.mean(
                             abs(baseline[keep]-target)/target)),
                         'candidate_mape_percent': float(100*np.mean(
                             abs(candidate[keep]-target)/target))})
        folds.append({'held_out': group, 'training_cells': len(train),
                      'scored_cells': len(rows),
                      'baseline_mape_percent': float(np.mean([
                          row['baseline_mape_percent'] for row in rows])),
                      'candidate_mape_percent': float(np.mean([
                          row['candidate_mape_percent'] for row in rows])),
                      'by_cell': rows})
        print(group, 'baseline', round(folds[-1]['baseline_mape_percent'], 3),
              'candidate', round(folds[-1]['candidate_mape_percent'], 3), flush=True)
    rows = [row for fold in folds for row in fold['by_cell']]
    report = {'status': 'unpromoted MIT whole-batch 300 s Q(V)-feature SOH development ablation',
              'grid_voltage_v': GRID.tolist(),
              'input': 'current and prior 300 s discharge voltage/current only',
              'full_qv_feature_finite_fraction': float(np.mean(np.isfinite(
                  np.concatenate([matrices[key][:, -2*len(GRID):]
                                  for key in cells])))),
              'summary': {'scored_cells': len(rows),
                          'baseline_mape_percent': float(np.mean([
                              row['baseline_mape_percent'] for row in rows])),
                          'candidate_mape_percent': float(np.mean([
                              row['candidate_mape_percent'] for row in rows]))},
              'folds': folds}
    (OUT / 'mit_partial_qv_soh_development_cv.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    ids, cycle, target, estimate = [], [], [], []
    for key, cell in cells.items():
        keep = cell.soh_available
        ids.extend([key]*int(keep.sum()))
        cycle.extend(cell.cycle[keep].tolist())
        target.extend(cell.soh[keep].tolist())
        estimate.extend(oof[key][keep].tolist())
    np.savez_compressed(OUT / 'mit_partial_qv_soh_development_oof.npz',
                        cell_id=np.asarray(ids), cycle=np.asarray(cycle),
                        target=np.asarray(target), estimate=np.asarray(estimate))
    print('SUMMARY', report['summary'], 'finite',
          report['full_qv_feature_finite_fraction'])


if __name__ == '__main__':
    main()
