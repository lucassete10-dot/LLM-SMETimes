"""MIT early-life ΔQ(V) ablation from only the first 300 s of discharges.

Observed cycles 10 and 100 are used; the full cycle capacity or any later
time sample is excluded. The common voltage grid is fixed from the read-only
coverage audit before fitting any labels. This is a 300 s prefix route, not
the full-discharge Severson feature.
"""

import json
from pathlib import Path

import h5py
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from audit_mit_partial_qv_v2 import partial_trace
from battery_v2_features import load_manifest
from evaluate_mit_deltaq_v2 import OUT
from train_soh_v2 import outer_splits


VOLTAGE_GRID = np.linspace(3.15, 3.53, 77)
ALPHA = 1.0
BASE = OUT / 'deltaq_stats_cv.json'
TARGET = OUT / 'partial_deltaq_cv.json'


def capacity_on_voltage_grid(voltage, capacity, voltage_grid=VOLTAGE_GRID):
    voltage = np.asarray(voltage, dtype=float)
    capacity = np.asarray(capacity, dtype=float)
    if voltage.shape != capacity.shape or len(voltage) < 3:
        raise ValueError('invalid partial Q/V trace')
    if not (np.isfinite(voltage).all() and np.isfinite(capacity).all()):
        raise ValueError('nonfinite partial Q/V trace')
    voltage_grid = np.asarray(voltage_grid, dtype=float)
    if voltage_grid.ndim != 1 or len(voltage_grid) < 2 or np.any(
            np.diff(voltage_grid) <= 0):
        raise ValueError('voltage grid must be strictly increasing')
    if voltage.min() > voltage_grid[0] or voltage.max() < voltage_grid[-1]:
        raise ValueError('partial trace does not cover fixed common voltage grid')
    order = np.argsort(voltage)
    v, q = voltage[order], (capacity-capacity[0])[order]
    unique, first, count = np.unique(v, return_index=True, return_counts=True)
    averaged = np.add.reduceat(q, first)/count
    return np.interp(voltage_grid, unique, averaged)


def partial_statistics(early, later, voltage_grid=VOLTAGE_GRID):
    first = capacity_on_voltage_grid(*early, voltage_grid=voltage_grid)
    second = capacity_on_voltage_grid(*later, voltage_grid=voltage_grid)
    change = second-first
    variance = np.var(change)
    if variance <= 0:
        raise ValueError('nonpositive partial ΔQ variance')
    return np.array([np.log10(variance), np.min(change), np.mean(change)],
                    dtype=float)


def load_statistics(records, duration_s=300, voltage_grid=VOLTAGE_GRID):
    result = {}
    for group in sorted({row['group'] for row in records}):
        subset = [row for row in records if row['group'] == group]
        with h5py.File(subset[0]['measurement_source'], 'r') as mat:
            for row in subset:
                index = int(row['cell_id'].split('c', 1)[1])
                early = partial_trace(mat, index, 10, duration_s)
                later = partial_trace(mat, index, 100, duration_s)
                if early is None or later is None:
                    continue
                result[row['cell_id']] = partial_statistics(
                    early, later, voltage_grid=voltage_grid)
    return result


def main():
    records = [row for row in load_manifest()['cells'] if row['dataset'] == 'MIT']
    by_id = {row['cell_id']: row for row in records}
    features = load_statistics(records)
    full = json.loads(BASE.read_text(encoding='utf-8'))
    previous = {row['cell_id']: row for fold in full['folds'] for row in fold['by_cell']}
    folds = []
    for group, train, validation in outer_splits(records, 'MIT'):
        training = [key for key in train if key in features and
                    by_id[key]['endpoint_cycle'] is not None]
        models = {}
        for name, width in (('partial_variance', 1),
                            ('partial_variance_min', 2),
                            ('partial_variance_min_mean', 3)):
            model = make_pipeline(StandardScaler(), Ridge(alpha=ALPHA))
            model.fit(np.stack([features[key][:width] for key in training]),
                      np.log([by_id[key]['endpoint_cycle'] for key in training]))
            models[name] = (model, width)
        rows = []
        for key in validation:
            if key not in features or key not in previous:
                continue
            true = by_id[key]['endpoint_cycle']
            estimates = {name: float(np.clip(np.exp(model.predict(
                         features[key][:width][None, :])[0]), 101, 2500))
                         for name, (model, width) in models.items()}
            rows.append({'cell_id': key, 'true_endpoint_cycle': true,
                         'ape_percent': {
                             'full_variance_min': previous[key]['ape_percent']['variance_min'],
                             **{name: float(100*abs(value-true)/true)
                                for name, value in estimates.items()}},
                         'predicted_endpoint_cycle': estimates})
        names = ('full_variance_min', 'partial_variance',
                 'partial_variance_min', 'partial_variance_min_mean')
        folds.append({'held_out': group, 'training_event_cells': len(training),
                      'scored_cells': len(rows),
                      'mape_percent': {name: float(np.mean([r['ape_percent'][name]
                                                          for r in rows]))
                                       for name in names}, 'by_cell': rows})
        print(group, folds[-1]['mape_percent'], flush=True)
    rows = [row for fold in folds for row in fold['by_cell']]
    summary = {'scored_cells': len(rows), 'prefix_available_cells': len(features),
               'mape_percent': {name: float(np.mean([row['ape_percent'][name]
                                                   for row in rows]))
                                for name in names}}
    report = {'status': 'unpromoted MIT whole-batch 300 s prefix ΔQ(V) ablation',
              'measurement': 'first 300 s of cycles 10 and 100, no full-cycle capacity',
              'voltage_grid_v': [float(VOLTAGE_GRID[0]), float(VOLTAGE_GRID[-1]),
                                 len(VOLTAGE_GRID)],
              'endpoint_warning': 'source 0.88 Ah, not per-cell exact 80%',
              'summary': summary, 'folds': folds}
    TARGET.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('SUMMARY', summary)


if __name__ == '__main__':
    main()
