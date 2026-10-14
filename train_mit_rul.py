"""MIT cycle-life/RUL baseline with explicit endpoint and censoring audit.

RUL is defined against the paper's 0.88 Ah threshold. Cells stopped above
0.885 Ah are treated as censored and never assigned a fake end-of-life label.
"""
import json
from pathlib import Path

import h5py
import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer

from battery_mit_crossbatch import load_batch
from battery_mit_validation import load_mit
from download_mit_batches import FILES, ROOT
from train_mit_all import load_data


OUT = Path('battery_final_models/mit_rul')
SOURCE1 = ROOT / '2017-05-12_batchdata_updated_struct_errorcorrect.mat'
THRESHOLD_AH = .88
END_TOLERANCE_AH = .005


def endpoints(groups):
    decisions = {}
    for batch in (1, 2, 3):
        path = SOURCE1 if batch == 1 else ROOT / FILES[batch][0]
        with h5py.File(path, 'r') as mat:
            source = mat['batch']
            for key in groups[f'batch{batch}']:
                index = int(key.split('c')[1])
                summary = mat[source['summary'][index, 0]]
                q_last = float(np.asarray(summary['QDischarge']).reshape(-1)[-1])
                raw_eol = float(np.asarray(mat[source['cycle_life'][index, 0]]).reshape(-1)[0])
                eol = int(raw_eol) if np.isfinite(raw_eol) else None
                observed = eol is not None and q_last <= THRESHOLD_AH + END_TOLERANCE_AH
                decisions[key] = {'eol_cycle': eol if observed else None,
                                  'last_capacity_ah': q_last,
                                  'endpoint_near_threshold': observed,
                                  'source_cycle_life': eol}
    return decisions


def trajectory_features(battery):
    x = battery.features.astype('float32')
    cycle = battery.discharge_number.astype('float32')
    n = len(x)
    # Causal history only: previous observed discharge curves, no capacity labels.
    smoothed = np.stack([x[max(0, i-7):i+1].mean(0) for i in range(n)])
    anchor = x[:, 10]
    lag = anchor - anchor[np.maximum(0, np.arange(n)-20)]
    return np.column_stack([x, smoothed, cycle, np.log1p(cycle), lag]).astype('float32')


def fit(batteries, endpoint, ids, log_target):
    features, labels, weights = [], [], []
    for key in ids:
        b = batteries[key]
        cycle = b.discharge_number
        keep = np.arange(15, len(cycle), 5)
        remaining = endpoint[key]['eol_cycle'] - cycle[keep]
        valid = remaining > 0
        if not valid.any():
            continue
        features.append(trajectory_features(b)[keep[valid]])
        y = remaining[valid].astype('float32')
        labels.append(np.log1p(y) if log_target else y)
        weights.append(np.full(len(y), 1/len(y)))
    x = np.concatenate(features)
    y = np.concatenate(labels)
    w = np.concatenate(weights)
    imputer = SimpleImputer(strategy='median', keep_empty_features=True).fit(x)
    model = ExtraTreesRegressor(n_estimators=200, min_samples_leaf=5,
                                max_features=.8, random_state=2026, n_jobs=-1)
    model.fit(imputer.transform(x), y, sample_weight=w)
    return imputer, model


def predict(bundle, batteries, key, log_target):
    imputer, model = bundle
    value = model.predict(imputer.transform(trajectory_features(batteries[key])))
    return np.maximum(0, np.expm1(value) if log_target else value)


def assess(bundle, batteries, endpoint, ids, log_target, baseline_life):
    by_cell = {}
    for key in ids:
        b = batteries[key]
        truth = endpoint[key]['eol_cycle'] - b.discharge_number
        mask = (np.arange(len(truth)) >= 15) & (truth > 0)
        pred = predict(bundle, batteries, key, log_target)
        naive = np.maximum(0, baseline_life - b.discharge_number)
        by_cell[key] = {'observations': int(mask.sum()),
                        'model_mae_cycles': float(np.mean(np.abs(pred[mask] - truth[mask]))),
                        'median_life_baseline_mae_cycles': float(np.mean(np.abs(naive[mask] - truth[mask]))),
                        'early_model_mae_cycles': float(np.mean(np.abs(pred[mask & (b.discharge_number <= 100)]
                                                                - truth[mask & (b.discharge_number <= 100)])))
                            if (mask & (b.discharge_number <= 100)).any() else None}
    return {'cells': len(by_cell), 'macro_mae_cycles': float(np.mean([x['model_mae_cycles'] for x in by_cell.values()])),
            'median_life_baseline_macro_mae_cycles': float(np.mean([
                x['median_life_baseline_mae_cycles'] for x in by_cell.values()])),
            'by_cell': by_cell}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    batteries, groups, _ = load_data()
    decision = endpoints(groups)
    (OUT / 'endpoint_audit.json').write_text(json.dumps(decision, indent=2))
    eligible = {group: [key for key in ids if decision[key]['endpoint_near_threshold']]
                for group, ids in groups.items()}
    folds = []
    for group, validation in eligible.items():
        train_ids = [key for name, ids in eligible.items() if name != group for key in ids]
        base_life = float(np.median([decision[key]['eol_cycle'] for key in train_ids]))
        fold = {'held_out_batch': group, 'train_cells': len(train_ids), 'validation_cells': len(validation),
                'median_training_life': base_life}
        for name, log_target in [('direct', False), ('log_target', True)]:
            fitted = fit(batteries, decision, train_ids, log_target)
            fold[name] = assess(fitted, batteries, decision, validation, log_target, base_life)
        folds.append(fold)
        print('RUL FOLD', group, {name: fold[name]['macro_mae_cycles']
                                 for name in ('direct', 'log_target')}, flush=True)
    (OUT / 'internal_leave_batch_out.json').write_text(json.dumps(folds, indent=2))
    scores = {name: float(np.mean([fold[name]['macro_mae_cycles'] for fold in folds]))
              for name in ('direct', 'log_target')}
    chosen = min(scores, key=scores.get)
    all_ids = [key for ids in eligible.values() for key in ids]
    imputer, model = fit(batteries, decision, all_ids, chosen == 'log_target')
    joblib.dump({'imputer': imputer, 'model': model, 'log_target': chosen == 'log_target'},
                OUT / 'rul_extra_trees.joblib')
    report = {'status': 'trained_waiting_for_external_test', 'task': 'MIT RUL to 0.88 Ah',
              'endpoint_rule': 'source cycle_life accepted only if last capacity <=0.885 Ah',
              'eligible_cells_by_batch': {name: len(ids) for name, ids in eligible.items()},
              'censored_cells_by_batch': {name: sorted(set(groups[name])-set(ids)) for name, ids in eligible.items()},
              'total_training_cells': len(all_ids), 'chosen_target': chosen,
              'internal_macro_mae_cycles_by_variant': scores,
              'input': 'current and previous discharge prefix features plus observed cycle number; no capacity labels as input',
              'feature_dimension': 105, 'training_cell_ids': all_ids}
    (OUT / 'training_manifest.json').write_text(json.dumps(report, indent=2))
    print('RUL TRAINED', {'cells': len(all_ids), 'chosen': chosen, 'scores': scores}, flush=True)


if __name__ == '__main__':
    main()
