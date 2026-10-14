"""MIT diagnostic-assisted early-life RUL at a fixed checkpoint.

Requires complete discharge-capacity diagnostics at prior checkpoints. This
is *not* the capacity-free online RUL route. The forecast target is the source
0.88-Ah life label, not a harmonized per-cell 80%-SOH crossing. Every outer
fold holds out a full MIT batch, and no post-checkpoint capacity is an input.
"""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from battery_v2_features import load_all
from evaluate_mit_policy_rul_v2 import policy_values
from train_soh_v2 import outer_splits


OUT = Path('battery_models_v2/mit_checkpoint_rul_challenger')
CHECKPOINTS = (50, 100)
RIDGE_ALPHA = 100


def diagnostic_features(cell, record, checkpoint):
    """Return features available by checkpoint, or None when a diagnostic is absent."""
    requested = np.arange(10, checkpoint + 1, 10, dtype='int64')
    indices = np.searchsorted(cell.cycle, requested)
    if (np.any(indices >= len(cell.cycle)) or
            not np.array_equal(cell.cycle[indices], requested) or
            not np.all(cell.soh_available[indices])):
        return None
    health = cell.soh[indices].astype('float64')
    voltage_change = cell.delta[indices[-1], :11].astype('float64')
    policy = policy_values(record).astype('float64')
    return np.r_[health, health-health[0], health[-1]-health[0],
                 voltage_change, policy]


def evaluate_checkpoint(records, cells, checkpoint):
    available = {record['cell_id']: diagnostic_features(cells[record['cell_id']],
                 record, checkpoint) for record in records
                 if cells[record['cell_id']].endpoint_cycle is not None}
    available = {key: value for key, value in available.items() if value is not None}
    folds = []
    for held, train_ids, val_ids in outer_splits(records, 'MIT'):
        train = [key for key in train_ids if key in available]
        val = [key for key in val_ids if key in available]
        model = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                              Ridge(alpha=RIDGE_ALPHA))
        model.fit(np.stack([available[key] for key in train]),
                  np.asarray([cells[key].endpoint_cycle for key in train]))
        prediction = model.predict(np.stack([available[key] for key in val]))
        rows = []
        for key, endpoint in zip(val, prediction):
            target = cells[key].endpoint_cycle
            rows.append({'cell_id': key, 'true_endpoint_cycle': target,
                         'predicted_endpoint_cycle': float(endpoint),
                         'absolute_percentage_error': float(100 * abs(endpoint-target)/target),
                         'absolute_error_cycles': float(abs(endpoint-target))})
        folds.append({'held_out': held, 'training_cells': len(train),
                      'scored_cells': len(val), 'mape_percent': float(np.mean(
                          [row['absolute_percentage_error'] for row in rows])),
                      'mae_cycles': float(np.mean([row['absolute_error_cycles'] for row in rows])),
                      'by_cell': rows})
        print(checkpoint, held, len(val), round(folds[-1]['mape_percent'], 2), flush=True)
    return {'checkpoint_cycle': checkpoint, 'scored_cells': sum(f['scored_cells'] for f in folds),
            'equal_batch_mape_percent': float(np.mean([f['mape_percent'] for f in folds])),
            'equal_cell_mape_percent': float(np.mean([
                row['absolute_percentage_error'] for fold in folds for row in fold['by_cell']])),
            'folds': folds, 'features': 'observed SOH at 10-cycle diagnostic intervals through '
                                      'checkpoint, 300s voltage delta at checkpoint, known charge policy'}


def main():
    manifest, all_cells = load_all()
    records = [row for row in manifest['cells'] if row['dataset'] == 'MIT']
    cells = {row['cell_id']: all_cells[row['cell_id']] for row in records}
    results = {str(checkpoint): evaluate_checkpoint(records, cells, checkpoint)
               for checkpoint in CHECKPOINTS}
    OUT.mkdir(parents=True, exist_ok=True)
    for checkpoint in CHECKPOINTS:
        x, y, ids = [], [], []
        for record in records:
            key = record['cell_id']
            if cells[key].endpoint_cycle is None:
                continue
            feature = diagnostic_features(cells[key], record, checkpoint)
            if feature is not None:
                x.append(feature)
                y.append(cells[key].endpoint_cycle)
                ids.append(key)
        model = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                              Ridge(alpha=RIDGE_ALPHA))
        model.fit(np.stack(x), np.asarray(y))
        joblib.dump({'model': model, 'checkpoint_cycle': checkpoint,
                     'training_cell_ids': ids, 'requires_measured_capacity': True,
                     'endpoint_rule': 'MIT source 0.88 Ah', 'data_contract_version': 2},
                    OUT / f'mit_diagnostic_cycle_{checkpoint}_candidate.joblib')
    report = {'status': 'development whole-batch CV; not online and not independent test',
              'selection_warning': 'features and regularization selected after exploratory CV',
              'target_warning': 'MIT source 0.88 Ah is not exact 80% of each cell initial capacity',
              'censored_cells_excluded': 12, 'checkpoints': results}
    (OUT / 'development_cv.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print({key: round(value['equal_batch_mape_percent'], 2) for key, value in results.items()})


if __name__ == '__main__':
    main()
