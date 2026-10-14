"""Cell/batch-held-out SOH training on battery data contract v2."""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler

from battery_v2_features import load_all


OUT = Path('battery_models_v2/soh')
CANDIDATES = {'ridge_delta': 'delta', 'extra_trees_history': 'signal_history'}


def training_arrays(cells, ids, variant):
    features, labels, weights = [], [], []
    for key in ids:
        cell = cells[key]
        keep = cell.soh_available
        if not keep.any():
            continue
        x = cell.matrix(variant)[keep]
        y = cell.soh[keep]
        w = np.where(cell.capacity_review_flag[keep], .25, 1.).astype('float64')
        w = w / w.sum() / len(ids)
        features.append(x)
        labels.append(y)
        weights.append(w)
    return np.concatenate(features), np.concatenate(labels), np.concatenate(weights)


def fit(cells, ids, name):
    variant = CANDIDATES[name]
    x, y, weights = training_arrays(cells, ids, variant)
    imputer = SimpleImputer(strategy='median', keep_empty_features=True).fit(x)
    x = imputer.transform(x)
    if name == 'ridge_delta':
        scaler = RobustScaler(quantile_range=(10, 90)).fit(x)
        scaler.scale_ = np.maximum(scaler.scale_, .02)
        x = np.clip(scaler.transform(x), -6, 6)
        model = Ridge(alpha=10).fit(x, y, sample_weight=weights*len(x))
    else:
        scaler = None
        model = ExtraTreesRegressor(n_estimators=160, min_samples_leaf=5,
                                    max_features=.8, random_state=2026, n_jobs=-1)
        model.fit(x, y, sample_weight=weights)
    return {'name': name, 'variant': variant, 'imputer': imputer,
            'scaler': scaler, 'model': model, 'training_cell_ids': list(ids),
            'data_contract_version': 2}


def predict(bundle, cell):
    x = bundle['imputer'].transform(cell.matrix(bundle['variant']))
    if bundle['scaler'] is not None:
        x = np.clip(bundle['scaler'].transform(x), -6, 6)
    return bundle['model'].predict(x)


def score(cells, predictions, ids):
    rows = []
    for key in ids:
        cell = cells[key]
        mask = cell.soh_available
        actual = cell.soh[mask]
        estimate = predictions[key][mask]
        error = np.abs(estimate - actual)
        progress = cell.cycle[mask] / cell.endpoint_cycle if cell.endpoint_cycle else None
        phases = ({'early': progress < .33,
                   'middle': (progress >= .33) & (progress < .67),
                   'late': progress >= .67} if progress is not None else {})
        rows.append({'cell_id': key, 'mae': float(np.mean(error)),
                     'signed_bias': float(np.mean(estimate-actual)),
                     'rmse': float(np.sqrt(np.mean((estimate-actual)**2))),
                     'n': int(mask.sum()), 'max_absolute_error': float(np.max(error)),
                     'absolute_error_p90': float(np.quantile(error, .9)),
                     'by_phase': {phase: {'n': int(part.sum()),
                                          'mae': float(np.mean(error[part])) if part.any() else None}
                                  for phase, part in phases.items()}})
    phase_scores = {phase: [row['by_phase'][phase]['mae'] for row in rows
                            if row['by_phase'].get(phase, {}).get('mae') is not None]
                    for phase in ('early', 'middle', 'late')}
    return {'cell_macro_mae': float(np.mean([row['mae'] for row in rows])),
            'cell_macro_signed_bias': float(np.mean([row['signed_bias'] for row in rows])),
            'worst_cell_mae': float(np.max([row['mae'] for row in rows])),
            'cells': len(rows), 'observations': sum(row['n'] for row in rows),
            'phase_macro_mae': {phase: float(np.mean(values)) if values else None
                                for phase, values in phase_scores.items()},
            'by_cell': rows}


def outer_splits(records, domain):
    rows = [row for row in records if row['dataset'] == domain]
    groups = sorted({row['group'] for row in rows})
    ids = [row['cell_id'] for row in rows]
    if domain in ('MIT', 'XJTU'):
        return [(group, [row['cell_id'] for row in rows if row['group'] != group],
                 [row['cell_id'] for row in rows if row['group'] == group]) for group in groups]
    return [(key, [other for other in ids if other != key], [key]) for key in ids]


def cell_holdout(records, domain):
    rows = [row for row in records if row['dataset'] == domain]
    groups = sorted({row['group'] for row in rows})
    if len(groups) < 2:
        return None
    holdout = [key for group in groups for key in sorted(
        row['cell_id'] for row in rows if row['group'] == group)[-2:]]
    train = [row['cell_id'] for row in rows if row['cell_id'] not in holdout]
    return train, holdout


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest, cells = load_all()
    records = manifest['cells']
    report = {'contract_version': 2, 'validation': 'whole-cell or whole-group development CV',
              'domains': {}}
    for domain in ('NASA', 'CALCE', 'MIT', 'XJTU'):
        folder = OUT / domain.lower()
        folder.mkdir(parents=True, exist_ok=True)
        folds = []
        oof_predictions = {candidate: {} for candidate in CANDIDATES}
        oof_splits = {}
        for name, train_ids, val_ids in outer_splits(records, domain):
            fold = {'held_out': name, 'train_cells': len(train_ids), 'validation_cells': len(val_ids)}
            for candidate in CANDIDATES:
                fitted = fit(cells, train_ids, candidate)
                predictions = {key: predict(fitted, cells[key]) for key in val_ids}
                oof_predictions[candidate].update(predictions)
                fold[candidate] = score(cells, predictions, val_ids)
            oof_splits.update({key: name for key in val_ids})
            folds.append(fold)
            print(domain, name, {key: round(fold[key]['cell_macro_mae'], 5)
                                  for key in CANDIDATES}, flush=True)
        (folder / 'group_cv.json').write_text(json.dumps(folds, indent=2), encoding='utf-8')
        means = {candidate: float(np.mean([fold[candidate]['cell_macro_mae']
                                             for fold in folds])) for candidate in CANDIDATES}
        selected = min(means, key=means.get)
        ids = [row['cell_id'] for row in records if row['dataset'] == domain]
        target, estimate, split, cell_id, cycle = [], [], [], [], []
        for key in ids:
            cell = cells[key]
            keep = cell.soh_available
            n = int(keep.sum())
            target.append(cell.soh[keep])
            estimate.append(oof_predictions[selected][key][keep])
            split.append(np.full(n, oof_splits[key]))
            cell_id.append(np.full(n, key))
            cycle.append(cell.cycle[keep])
        np.savez_compressed(folder / 'development_oof_predictions.npz',
                            target=np.concatenate(target), estimate=np.concatenate(estimate),
                            held_out=np.concatenate(split), cell_id=np.concatenate(cell_id),
                            cycle=np.concatenate(cycle))
        for candidate in CANDIDATES:
            joblib.dump(fit(cells, ids, candidate), folder / f'{candidate}.joblib')
        in_domain = None
        split = cell_holdout(records, domain)
        if split:
            train_ids, val_ids = split
            fitted = fit(cells, train_ids, selected)
            in_domain = score(cells, {key: predict(fitted, cells[key]) for key in val_ids}, val_ids)
            in_domain['validation_cell_ids'] = val_ids
        info = {'dataset': domain, 'cells': len(ids), 'selected_model': selected,
                'group_cv_macro_mae': means, 'same_group_cell_holdout': in_domain,
                'training_cells': ids,
                'reference_rule': sorted({row['reference_rule'] for row in records
                                          if row['dataset'] == domain}),
                'input': 'observed signal features only; train-cell imputation and scaling; '
                         'calibration cycles excluded from fit'}
        (folder / 'training_manifest.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
        report['domains'][domain] = {key: value for key, value in info.items()
                                     if key != 'training_cells'}
        print('SELECTED', domain, selected, means, flush=True)
    (OUT / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
