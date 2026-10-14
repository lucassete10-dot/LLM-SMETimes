"""MIT early RUL at cycle 50 from *predicted* SOH and signal history.

Training-cell SOH inputs are cross-fitted by batch inside each outer fold.
Validation cells use an SOH model fit only on outer training batches. Actual
capacity/SOH is never an RUL inference input. Target remains MIT source
0.88-Ah life, pending the user's choice about exact per-cell 80% SOH.
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
from evaluate_paper_inspired_rul_v2 import baseline_oof
from train_rul_v2 import oof_soh
from train_soh_v2 import fit as fit_soh, outer_splits, predict as predict_soh


OUT = Path('battery_models_v2/mit_checkpoint_signal_rul_challenger')
CHECKPOINT = 50
RIDGE_ALPHA = 100


def feature(cell, record, soh_prediction, checkpoint=CHECKPOINT):
    requested = np.arange(10, checkpoint+1, 10, dtype='int64')
    indices = np.searchsorted(cell.cycle, requested)
    if np.any(indices >= len(cell.cycle)) or not np.array_equal(cell.cycle[indices], requested):
        return None
    health = np.asarray(soh_prediction, dtype='float64')[indices]
    voltage_change = cell.delta[indices[-1], :11].astype('float64')
    policy = policy_values(record).astype('float64')
    return np.r_[health, health-health[0], health[-1]-health[0],
                 voltage_change, policy]


def fit_endpoint(x, y):
    model = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                          Ridge(alpha=RIDGE_ALPHA))
    model.fit(np.stack(x), np.asarray(y))
    return model


def evaluate(save_model=False):
    manifest, all_cells = load_all()
    records = [r for r in manifest['cells'] if r['dataset'] == 'MIT']
    cells = {r['cell_id']: all_cells[r['cell_id']] for r in records}
    record_by_id = {r['cell_id']: r for r in records}
    baseline = baseline_oof('MIT', cells)
    folds = []
    for group, train, val in outer_splits(records, 'MIT'):
        inner = oof_soh(cells, records, train)
        train_features = {}
        for key in train:
            if cells[key].endpoint_cycle is not None:
                train_features[key] = feature(cells[key], record_by_id[key], inner[key])
        train_features = {k: x for k, x in train_features.items() if x is not None}
        model = fit_endpoint(list(train_features.values()),
                             [cells[key].endpoint_cycle for key in train_features])
        soh_model = fit_soh(cells, train, 'extra_trees_history')
        rows = []
        for key in val:
            cell = cells[key]
            if cell.endpoint_cycle is None:
                continue
            x = feature(cell, record_by_id[key], predict_soh(soh_model, cell))
            if x is None:
                continue
            predicted = float(model.predict(x[None, :])[0])
            at = np.flatnonzero(cell.cycle == CHECKPOINT)
            if not len(at) or not np.isfinite(baseline[key][at[0]]):
                raise ValueError(f'{key}: missing baseline at checkpoint')
            baseline_eol = float(CHECKPOINT+baseline[key][at[0]])
            rows.append({'cell_id': key, 'true_endpoint_cycle': cell.endpoint_cycle,
                         'predicted_endpoint_cycle': predicted,
                         'baseline_endpoint_cycle': baseline_eol,
                         'candidate_ape_percent': float(100*abs(predicted-cell.endpoint_cycle)/
                                                         cell.endpoint_cycle),
                         'baseline_ape_percent': float(100*abs(baseline_eol-cell.endpoint_cycle)/
                                                        cell.endpoint_cycle)})
        folds.append({'held_out': group, 'training_endpoint_cells': len(train_features),
                      'scored_cells': len(rows),
                      'candidate_mape_percent': float(np.mean([
                          row['candidate_ape_percent'] for row in rows])),
                      'baseline_mape_percent': float(np.mean([
                          row['baseline_ape_percent'] for row in rows])),
                      'by_cell': rows})
        print(group, 'n', len(rows), 'baseline', round(folds[-1]['baseline_mape_percent'], 2),
              'candidate', round(folds[-1]['candidate_mape_percent'], 2), flush=True)
    all_rows = [row for fold in folds for row in fold['by_cell']]
    report = {'status': 'development whole-batch CV, not independent test',
              'input': 'estimated SOH from capacity-free discharge prefixes through cycle 50, '
                       'voltage delta, and known charge policy',
              'endpoint_warning': 'MIT source 0.88 Ah, not exact per-cell 80% SOH',
              'summary': {'scored_cells': len(all_rows),
                          'candidate_cell_macro_mape_percent': float(np.mean([
                              row['candidate_ape_percent'] for row in all_rows])),
                          'baseline_cell_macro_mape_percent': float(np.mean([
                              row['baseline_ape_percent'] for row in all_rows])),
                          'candidate_equal_batch_mape_percent': float(np.mean([
                              fold['candidate_mape_percent'] for fold in folds])),
                          'baseline_equal_batch_mape_percent': float(np.mean([
                              fold['baseline_mape_percent'] for fold in folds]))},
              'folds': folds}
    if save_model:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / 'development_cv.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        ids = list(cells)
        inner = oof_soh(cells, records, ids)
        x = {key: feature(cells[key], record_by_id[key], inner[key])
             for key in ids if cells[key].endpoint_cycle is not None}
        x = {key: val for key, val in x.items() if val is not None}
        endpoint = fit_endpoint(list(x.values()), [cells[key].endpoint_cycle for key in x])
        soh = fit_soh(cells, ids, 'extra_trees_history')
        joblib.dump({'soh_model': soh, 'endpoint_model': endpoint,
                     'training_cell_ids': list(x), 'checkpoint_cycle': CHECKPOINT,
                     'endpoint_rule': 'MIT source 0.88 Ah'},
                    OUT / 'mit_signal_checkpoint_candidate.joblib')
    return report


if __name__ == '__main__':
    result = evaluate()
    print('SUMMARY', result['summary'], flush=True)
