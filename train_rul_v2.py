"""RUL development CV with source endpoint masks and leakage-free SOH stacking.

The direct route uses observed signal history and age. The assisted route uses
SOH estimates fitted out of fold on training cells/groups, never measured SOH.
Neither route uses the eventual end-of-life cycle as an input feature.
"""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer

from battery_v2_features import load_all
from train_soh_v2 import fit as fit_soh, predict as predict_soh, outer_splits


OUT = Path('battery_models_v2/rul')
SOH_MODEL = 'extra_trees_history'
MIN_HISTORY = 15


def eligible(cell):
    index = np.arange(len(cell.cycle))
    return cell.rul_available & (index >= max(MIN_HISTORY, cell.reference_count))


def signal_features(cell):
    age = cell.cycle.astype('float32')
    x = cell.matrix('signal_history')
    return np.column_stack((x, age, np.log1p(age))).astype('float32')


def assisted_features(cell, soh_pred):
    soh_pred = np.asarray(soh_pred, dtype='float32')
    # All windows end at the prediction cycle; negative slopes mean aging.
    lag = soh_pred[np.maximum(0, np.arange(len(soh_pred))-20)]
    slope = (soh_pred-lag) / np.maximum(1, np.arange(len(soh_pred)) -
                                     np.maximum(0, np.arange(len(soh_pred))-20))
    smooth = np.array([np.median(soh_pred[max(0, i-4):i+1])
                       for i in range(len(soh_pred))], dtype='float32')
    return np.column_stack((signal_features(cell), soh_pred, smooth, slope))


def oof_soh(cells, records, ids):
    """For each training row, predict SOH from other cells/strategies only."""
    by_group = {}
    for row in records:
        if row['cell_id'] in ids:
            by_group.setdefault(row['group'], []).append(row['cell_id'])
    splits = list(by_group.values()) if len(by_group) > 1 else [[key] for key in ids]
    result = {}
    for hold in splits:
        train = [key for key in ids if key not in hold]
        if not train:
            raise ValueError('SOH cross-fitting needs at least two cells')
        model = fit_soh(cells, train, SOH_MODEL)
        result.update({key: predict_soh(model, cells[key]) for key in hold})
    if set(result) != set(ids):
        raise ValueError('incomplete SOH cross-fitting')
    return result


def fit_rul(cells, ids, kind, soh_predictions=None):
    x_parts, y_parts, w_parts = [], [], []
    for key in ids:
        cell = cells[key]
        keep = eligible(cell)
        if not keep.any():
            continue
        if kind == 'assisted':
            x = assisted_features(cell, soh_predictions[key])[keep]
        else:
            x = signal_features(cell)[keep]
        y = cell.rul_cycles[keep]
        weight = np.where(cell.capacity_review_flag[keep], .25, 1.).astype('float64')
        # Each cell has equal total weight, regardless of diagnostic density.
        weight /= weight.sum()
        x_parts.append(x[::5])
        y_parts.append(y[::5])
        w_parts.append(weight[::5] / len(ids))
    if not x_parts:
        raise ValueError('no observed RUL endpoints in training fold')
    x, y, weights = map(np.concatenate, (x_parts, y_parts, w_parts))
    imputer = SimpleImputer(strategy='median', keep_empty_features=True).fit(x)
    model = ExtraTreesRegressor(n_estimators=120, min_samples_leaf=5,
                                max_features=.8, random_state=2026, n_jobs=-1)
    model.fit(imputer.transform(x), np.log1p(y), sample_weight=weights)
    return {'kind': kind, 'imputer': imputer, 'model': model,
            'soh_model_name': SOH_MODEL if kind == 'assisted' else None,
            'training_cell_ids': ids, 'data_contract_version': 2}


def predict_rul(model, cell, soh_pred=None):
    x = assisted_features(cell, soh_pred) if model['kind'] == 'assisted' else signal_features(cell)
    return np.maximum(0, np.expm1(model['model'].predict(model['imputer'].transform(x))))


def assess(cells, ids, predictions, baseline_life):
    by_cell = []
    for key in ids:
        cell = cells[key]
        keep = eligible(cell)
        if not keep.any():
            continue
        actual = cell.rul_cycles[keep]
        estimate = predictions[key][keep]
        naive = np.maximum(0, baseline_life-cell.cycle[keep])
        endpoint = cell.endpoint_cycle
        progress = cell.cycle[keep] / endpoint
        bins = {'early': progress < .33, 'middle': (progress >= .33) & (progress < .67),
                'late': progress >= .67}
        row = {'cell_id': key, 'n': int(keep.sum()),
               'mae_cycles': float(np.mean(np.abs(estimate-actual))),
               'signed_bias_cycles': float(np.mean(estimate-actual)),
               'baseline_mae_cycles': float(np.mean(np.abs(naive-actual))),
               'absolute_error_p90_cycles': float(np.quantile(np.abs(estimate-actual), .9)),
               'by_phase': {phase: {'n': int(mask.sum()),
                                   'mae_cycles': float(np.mean(np.abs(estimate[mask]-actual[mask])))
                                   if mask.any() else None}
                            for phase, mask in bins.items()}}
        by_cell.append(row)
    phases = {phase: [row['by_phase'][phase]['mae_cycles'] for row in by_cell
                      if row['by_phase'][phase]['mae_cycles'] is not None]
              for phase in ('early', 'middle', 'late')}
    return {'cells': len(by_cell), 'observations': sum(row['n'] for row in by_cell),
            'cell_macro_mae_cycles': float(np.mean([row['mae_cycles'] for row in by_cell])),
            'cell_macro_signed_bias_cycles': float(np.mean(
                [row['signed_bias_cycles'] for row in by_cell])),
            'median_life_baseline_macro_mae_cycles': float(np.mean(
                [row['baseline_mae_cycles'] for row in by_cell])),
            'worst_cell_mae_cycles': float(max(row['mae_cycles'] for row in by_cell)),
            'phase_macro_mae_cycles': {phase: float(np.mean(values)) if values else None
                                       for phase, values in phases.items()},
            'by_cell': by_cell}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest, cells = load_all()
    records = manifest['cells']
    summary = {'data_contract_version': 2,
               'evaluation': 'whole-cell NASA/CALCE; whole-batch MIT; whole-strategy XJTU',
               'domains': {}}
    for domain in ('NASA', 'CALCE', 'MIT', 'XJTU'):
        folder = OUT / domain.lower()
        folder.mkdir(parents=True, exist_ok=True)
        folds = []
        oof_predictions = {'direct': {}, 'assisted': {}}
        oof_splits = {}
        for name, training, validation in outer_splits(records, domain):
            observed_train = [key for key in training if cells[key].endpoint_cycle is not None]
            observed_val = [key for key in validation if cells[key].endpoint_cycle is not None]
            base = float(np.median([cells[key].endpoint_cycle for key in observed_train]))
            fold = {'held_out': name, 'train_cells': len(observed_train),
                    'validation_cells': len(observed_val), 'censored_validation_cells':
                    len(validation)-len(observed_val)}
            direct = fit_rul(cells, observed_train, 'direct')
            direct_predictions = {key: predict_rul(direct, cells[key]) for key in observed_val}
            oof_predictions['direct'].update(direct_predictions)
            fold['direct'] = assess(cells, observed_val, direct_predictions, base)
            oof = oof_soh(cells, records, observed_train)
            assisted = fit_rul(cells, observed_train, 'assisted', oof)
            soh = fit_soh(cells, observed_train, SOH_MODEL)
            assisted_predictions = {key: predict_rul(assisted, cells[key],
                                    predict_soh(soh, cells[key])) for key in observed_val}
            oof_predictions['assisted'].update(assisted_predictions)
            fold['assisted'] = assess(cells, observed_val, assisted_predictions, base)
            oof_splits.update({key: name for key in observed_val})
            folds.append(fold)
            print(domain, name, {kind: round(fold[kind]['cell_macro_mae_cycles'], 1)
                                 for kind in ('direct', 'assisted')}, flush=True)
        (folder / 'group_cv.json').write_text(json.dumps(folds, indent=2), encoding='utf-8')
        means = {kind: float(np.mean([fold[kind]['cell_macro_mae_cycles'] for fold in folds]))
                 for kind in ('direct', 'assisted')}
        selected = min(means, key=means.get)
        ids = [row['cell_id'] for row in records if row['dataset'] == domain and
               cells[row['cell_id']].endpoint_cycle is not None]
        target, estimate, split, cell_id, cycle, endpoint = [], [], [], [], [], []
        for key in ids:
            cell = cells[key]
            keep = eligible(cell)
            n = int(keep.sum())
            target.append(cell.rul_cycles[keep])
            estimate.append(oof_predictions[selected][key][keep])
            split.append(np.full(n, oof_splits[key]))
            cell_id.append(np.full(n, key))
            cycle.append(cell.cycle[keep])
            endpoint.append(np.full(n, cell.endpoint_cycle))
        np.savez_compressed(folder / 'development_oof_predictions.npz',
                            target=np.concatenate(target), estimate=np.concatenate(estimate),
                            held_out=np.concatenate(split), cell_id=np.concatenate(cell_id),
                            cycle=np.concatenate(cycle), endpoint=np.concatenate(endpoint))
        direct = fit_rul(cells, ids, 'direct')
        joblib.dump(direct, folder / 'direct.joblib')
        oof = oof_soh(cells, records, ids)
        assisted = fit_rul(cells, ids, 'assisted', oof)
        joblib.dump(assisted, folder / 'assisted.joblib')
        info = {'dataset': domain, 'selected_model': selected, 'group_cv_macro_mae_cycles': means,
                'observed_endpoint_cells': len(ids),
                'censored_cells': len([row for row in records if row['dataset'] == domain])-len(ids),
                'training_cell_ids': ids,
                'input': 'causal signal history, observed age; assisted adds out-of-fold SOH predictions',
                'censoring': 'censored cells excluded from regression targets; no inferred end-of-life',
                'baseline': 'median endpoint cycle of training cells'}
        (folder / 'training_manifest.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
        summary['domains'][domain] = {key: value for key, value in info.items()
                                      if key != 'training_cell_ids'}
        print('SELECTED', domain, selected, means, flush=True)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
