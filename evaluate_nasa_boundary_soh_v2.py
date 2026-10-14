"""NASA SOH boundary-extrapolation challenger (development CV only).

The tree baseline interpolates well, but cannot continue below the lowest
SOH represented by training cells. A ridge model of discharge-prefix changes
is used only when the tree prediction touches that boundary. Both models and
the boundary are fitted from the training cells in each outer fold.
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
from train_soh_v2 import fit as fit_tree, outer_splits, predict as predict_tree


OUT = Path('battery_models_v2/nasa_boundary_soh_challenger')
BOUNDARY_QUANTILE = .01
BOUNDARY_WIDTH = .005
RIDGE_ALPHA = .3


def boundary_adjustment(tree_prediction, linear_prediction, training_floor):
    """Only correct downward at the bottom of the tree's training range."""
    tree = np.asarray(tree_prediction, dtype='float64')
    linear = np.asarray(linear_prediction, dtype='float64')
    weight = np.clip((training_floor + BOUNDARY_WIDTH - tree) /
                     BOUNDARY_WIDTH, 0, 1)
    return tree + weight * np.minimum(linear - tree, 0)


def fit(cells, ids):
    x = np.concatenate([cells[key].delta[cells[key].soh_available] for key in ids])
    y = np.concatenate([cells[key].soh[cells[key].soh_available] for key in ids])
    linear = make_pipeline(SimpleImputer(strategy='median'),
                           StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
    linear.fit(x, y)
    return {'tree': fit_tree(cells, ids, 'extra_trees_history'),
            'linear': linear, 'training_floor': float(np.quantile(y, BOUNDARY_QUANTILE)),
            'training_cell_ids': list(ids), 'data_contract_version': 2}


def predict(bundle, cell):
    tree = predict_tree(bundle['tree'], cell)
    linear = bundle['linear'].predict(cell.delta)
    return boundary_adjustment(tree, linear, bundle['training_floor'])


def mape(cell, prediction):
    keep = cell.soh_available
    return float(100 * np.mean(np.abs(prediction[keep] - cell.soh[keep]) /
                               cell.soh[keep]))


def evaluate(save_oof=False):
    manifest, cells = load_all()
    records = [row for row in manifest['cells'] if row['dataset'] == 'NASA']
    folds = []
    oof = {}
    for held, train_ids, val_ids in outer_splits(records, 'NASA'):
        if set(train_ids) & set(val_ids):
            raise ValueError('training/validation overlap')
        bundle = fit(cells, train_ids)
        rows = []
        for key in val_ids:
            cell = cells[key]
            tree = predict_tree(bundle['tree'], cell)
            estimate = predict(bundle, cell)
            oof[key] = estimate
            rows.append({'cell_id': key, 'baseline_mape_percent': mape(cell, tree),
                         'challenger_mape_percent': mape(cell, estimate),
                         'training_floor': bundle['training_floor'],
                         'corrected_observations': int(np.count_nonzero(
                             np.abs(estimate-tree) > 1e-9))})
        folds.append({'held_out': held, 'training_cells': train_ids, 'rows': rows})
    if save_oof:
        ids = [row['cell_id'] for row in records]
        if set(oof) != set(ids):
            raise ValueError('incomplete NASA boundary OOF predictions')
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(OUT / 'development_oof_predictions.npz',
                            cell_id=np.concatenate([np.full(len(cells[key].cycle), key)
                                                     for key in ids]),
                            cycle=np.concatenate([cells[key].cycle for key in ids]),
                            estimate=np.concatenate([oof[key] for key in ids]),
                            soh_available=np.concatenate([cells[key].soh_available for key in ids]))
    all_rows = [row for fold in folds for row in fold['rows']]
    return {'status': 'development leave-one-cell-out; parameters selected using these folds',
            'not_independent_test': True,
            'input': 'first 1000 seconds of each discharge, referenced to first discharge; '
                     'no measured capacity in inference features',
            'parameters': {'ridge_alpha': RIDGE_ALPHA,
                           'boundary_quantile': BOUNDARY_QUANTILE,
                           'boundary_width_soh': BOUNDARY_WIDTH},
            'cells': len(all_rows), 'folds': folds,
            'baseline_cell_macro_mape_percent': float(np.mean(
                [row['baseline_mape_percent'] for row in all_rows])),
            'challenger_cell_macro_mape_percent': float(np.mean(
                [row['challenger_mape_percent'] for row in all_rows])),
            'worst_cell_mape_percent': float(max(
                row['challenger_mape_percent'] for row in all_rows))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = evaluate(save_oof=True)
    (OUT / 'development_cv.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    manifest, cells = load_all()
    ids = [row['cell_id'] for row in manifest['cells'] if row['dataset'] == 'NASA']
    joblib.dump(fit(cells, ids), OUT / 'nasa_candidate.joblib')
    print(json.dumps({'baseline_mape_percent': report['baseline_cell_macro_mape_percent'],
                      'candidate_mape_percent': report['challenger_cell_macro_mape_percent'],
                      'worst_cell_mape_percent': report['worst_cell_mape_percent']}, indent=2))


if __name__ == '__main__':
    main()
