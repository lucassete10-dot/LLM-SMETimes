"""Training-only feature-distance diagnostic for leave-strategy-out XJTU risk.

This is an explanatory diagnostic, not a certified out-of-distribution detector.
No held-out group participates in its imputer or scaler fit.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

from battery_v2_features import load_all
from train_soh_v2 import outer_splits


OUT = Path('battery_models_v2/xjtu_shift_diagnostic.json')


def initial_signature(cell):
    return np.nanmedian(cell.raw[:min(20, len(cell.raw))], axis=0)


def scaled_distances(cells, train, val):
    train_x = np.stack([initial_signature(cells[key]) for key in train])
    val_x = np.stack([initial_signature(cells[key]) for key in val])
    imputer = SimpleImputer(strategy='median', keep_empty_features=True).fit(train_x)
    scaler = RobustScaler(quantile_range=(10, 90)).fit(imputer.transform(train_x))
    scaler.scale_ = np.maximum(scaler.scale_, .02)
    reference = np.clip(scaler.transform(imputer.transform(train_x)), -10, 10)
    candidate = np.clip(scaler.transform(imputer.transform(val_x)), -10, 10)
    cross = np.linalg.norm(candidate[:, None, :]-reference[None, :, :], axis=2)
    self_distance = np.linalg.norm(reference[:, None, :]-reference[None, :, :], axis=2)
    np.fill_diagonal(self_distance, np.inf)
    within = np.min(self_distance, axis=1)
    unseen = np.min(cross, axis=1)
    return within, unseen


def main():
    manifest, cells = load_all()
    rul_folds = {row['held_out']: row for row in json.loads(
        Path('battery_models_v2/rul/xjtu/group_cv.json').read_text(encoding='utf-8'))}
    rows = []
    for group, train, val in outer_splits(manifest['cells'], 'XJTU'):
        within, unseen = scaled_distances(cells, train, val)
        typical = float(np.quantile(within, .95))
        rows.append({'held_out': group, 'training_cells': len(train), 'validation_cells': len(val),
                     'training_nearest_neighbor_p95': typical,
                     'validation_nearest_neighbor_median': float(np.median(unseen)),
                     'validation_nearest_neighbor_max': float(np.max(unseen)),
                     'validation_to_training_p95_ratio': float(np.median(unseen)/typical),
                     'validation_cell_distances': {key: float(dist) for key, dist in zip(val, unseen)},
                     'direct_rul_macro_mae_cycles':
                         rul_folds[group]['direct']['cell_macro_mae_cycles']})
    OUT.write_text(json.dumps({'input': 'first 20 raw signal observations only',
                               'method': 'leave-strategy-out train-only imputation/robust scaling; Euclidean nearest training cell',
                               'interpretation': 'descriptive feature shift, not calibrated abstention threshold',
                               'folds': rows}, indent=2), encoding='utf-8')
    for row in rows:
        print(row['held_out'], round(row['validation_nearest_neighbor_median'], 2),
              round(row['validation_to_training_p95_ratio'], 2),
              round(row['direct_rul_macro_mae_cycles'], 1), flush=True)


if __name__ == '__main__':
    main()
