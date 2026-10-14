"""MIT whole-batch SOH ablation: robust-scaled ridge on causal signal+age.

Target SOH is used only for training labels/scoring. Input contains the
current/past 300 s discharge prefix and known cycle age. This tests whether
a smooth extrapolating map transfers better than the tree under batch shift.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler

from battery_v2_features import load_cell, load_manifest
from evaluate_soh_age_signal_v2 import age_matrix
from train_soh_v2 import outer_splits


OUT = Path('battery_models_v2/soh_age_signal_challenger')
BASE = OUT / 'mit_development_cv.json'
ALPHA = 100.0


def fit(cells, ids, matrices):
    parts = [[], [], []]
    for key in ids:
        cell = cells[key]
        keep = cell.soh_available
        if not keep.any():
            continue
        weight = np.where(cell.capacity_review_flag[keep], .25, 1.).astype(float)
        weight /= weight.sum()*len(ids)
        for target, value in zip(parts, (matrices[key][keep], cell.soh[keep], weight)):
            target.append(value)
    x, y, weights = map(np.concatenate, parts)
    imputer = SimpleImputer(strategy='median', keep_empty_features=True).fit(x)
    transformed = imputer.transform(x)
    scaler = RobustScaler(quantile_range=(10, 90)).fit(transformed)
    scaler.scale_ = np.maximum(scaler.scale_, .02)
    normalized = np.clip(scaler.transform(transformed), -6, 6)
    model = Ridge(alpha=ALPHA)
    model.fit(normalized, y, sample_weight=weights*len(x))
    return imputer, scaler, model


def predict(bundle, x):
    imputer, scaler, model = bundle
    normalized = np.clip(scaler.transform(imputer.transform(x)), -6, 6)
    return model.predict(normalized)


def main():
    records = [row for row in load_manifest()['cells'] if row['dataset'] == 'MIT']
    cells = {row['cell_id']: load_cell(row) for row in records}
    matrices = {key: age_matrix(cell) for key, cell in cells.items()}
    baseline = json.loads(BASE.read_text(encoding='utf-8'))
    base_by_fold = {fold['held_out']: fold for fold in baseline['folds']}
    folds = []
    oof = {}
    for group, train, validation in outer_splits(records, 'MIT'):
        bundle = fit(cells, train, matrices)
        rows = []
        for key in validation:
            cell = cells[key]
            prediction = predict(bundle, matrices[key])
            oof[key] = prediction
            keep = cell.soh_available
            mape = float(100*np.mean(abs(prediction[keep]-cell.soh[keep])/
                                     cell.soh[keep]))
            rows.append({'cell_id': key, 'mape_percent': mape,
                         'signed_bias_soh': float(np.mean(prediction[keep]-
                                                         cell.soh[keep]))})
        baseline_mape = base_by_fold[group]['metrics'][
            'gated_original']['cell_macro_mape_percent']
        tree_age_mape = base_by_fold[group]['metrics'][
            'gated_with_age']['cell_macro_mape_percent']
        folds.append({'held_out': group, 'training_cells': len(train),
                      'scored_cells': len(rows),
                      'mape_percent': {
                          'gated_original': baseline_mape,
                          'gated_tree_with_age': tree_age_mape,
                          'ridge_with_age': float(np.mean([row['mape_percent']
                                                          for row in rows]))},
                      'by_cell': rows})
        print(group, folds[-1]['mape_percent'], flush=True)
    rows = [row for fold in folds for row in fold['by_cell']]
    summary = {'scored_cells': len(rows),
               'mape_percent': {
                   'gated_original': baseline['summary']['gated_original'][
                       'cell_macro_mape_percent'],
                   'gated_tree_with_age': baseline['summary']['gated_with_age'][
                       'cell_macro_mape_percent'],
                   'ridge_with_age': float(np.mean([row['mape_percent']
                                                   for row in rows]))}}
    report = {'status': 'unpromoted MIT whole-batch SOH development ablation',
              'input': 'causal 300 s discharge signal + observed cycle age',
              'model': f'robust-scaled ridge, alpha={ALPHA}, cell-balanced weights',
              'summary': summary, 'folds': folds}
    (OUT / 'mit_ridge_age_development_cv.json').write_text(json.dumps(report, indent=2),
                                                           encoding='utf-8')
    ids, cycle, target, estimate = [], [], [], []
    for key, cell in cells.items():
        keep = cell.soh_available
        ids.extend([key]*int(keep.sum()))
        cycle.extend(cell.cycle[keep].tolist())
        target.extend(cell.soh[keep].tolist())
        estimate.extend(oof[key][keep].tolist())
    np.savez_compressed(OUT / 'mit_ridge_age_development_oof.npz',
                        cell_id=np.asarray(ids), cycle=np.asarray(cycle),
                        target=np.asarray(target), estimate=np.asarray(estimate))
    print('SUMMARY', summary)


if __name__ == '__main__':
    main()
