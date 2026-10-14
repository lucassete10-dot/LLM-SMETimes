"""Refit MIT SOH models on all usable cells from original Batches 1/2/3.

Leave-one-batch-out scores are development diagnostics; every eligible cell is
used in the final fit. The user's future external test remains untouched.
"""
import copy
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy.ndimage import median_filter
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer

from battery_mit_validation import NumericPrep, load_mit, metrics_by_cell
from battery_mit_crossbatch import load_batch
from battery_mvp import set_seed
from battery_smetimes_regularized import RegularizedSMETimes
from prepare_mit import ENHANCED_NAMES


OUT = Path('battery_final_models/mit_all_batches')
SEEDS = [2026, 2027, 2028]


def load_data():
    _, rich1 = load_mit()
    _, rich2, source2 = load_batch(2)
    _, rich3, source3 = load_batch(3)
    batteries = {**rich1, **rich2, **rich3}
    groups = {'batch1': list(rich1), 'batch2': list(rich2), 'batch3': list(rich3)}
    if len(batteries) != 134 or sum(map(len, groups.values())) != 134:
        raise ValueError('Unexpected cell count or duplicate cell identifier')
    return batteries, groups, {2: source2, 3: source3}


def tree_fit(batteries, ids):
    x = np.concatenate([batteries[key].features for key in ids])
    y = np.concatenate([batteries[key].soh for key in ids])
    imputer = SimpleImputer(strategy='median', keep_empty_features=True).fit(x)
    weights = np.concatenate([np.full(len(batteries[key].soh), 1/len(batteries[key].soh))
                              for key in ids])
    model = ExtraTreesRegressor(n_estimators=200, min_samples_leaf=5, max_features=.8,
                                random_state=2026, n_jobs=-1)
    model.fit(imputer.transform(x), y, sample_weight=weights)
    return imputer, model


def make_numeric_data(prep, batteries, ids):
    return {key: prep.transform(batteries[key]) for key in ids}


def numeric_fit(data, batteries, train_ids, seed, epochs, validation_ids=None, patience=10):
    set_seed(seed)
    model = RegularizedSMETimes(False, feature_dim=51)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.03)
    samples, weights = [], []
    outlier_count = 0
    for key in train_ids:
        idx = np.arange(0, len(data[key][0]), 3)
        samples.append([part[idx] for part in data[key]])
        labels = batteries[key].soh
        bad = np.abs(labels - median_filter(labels, size=9, mode='nearest')) > .04
        outlier_count += int(bad[idx].sum())
        weight = np.where(bad[idx], .25, 1.).astype('float32')
        weights.append(torch.from_numpy(weight / weight.sum() / len(train_ids)))
    x, history, semantic, base, y = [torch.cat([sample[j] for sample in samples])
                                     for j in range(5)]
    weight = torch.cat(weights)
    best = None
    best_epoch = 0
    best_score = float('inf')
    stale = 0
    trace = []
    for epoch in range(1, epochs + 1):
        model.train()
        prediction, reconstructed, _ = model(x + .015*torch.randn_like(x), history, semantic, base)
        error = prediction - y
        absolute = error.abs()
        per = torch.where(absolute <= .025, .5*error.square(), .025*(absolute-.0125))
        loss = (per*weight).sum() + .0002*(reconstructed-x).square().mean() + .01*(prediction-base).square().mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        if validation_ids:
            model.eval()
            with torch.no_grad():
                scores = []
                for key in validation_ids:
                    pred = model(*data[key][:4])[0]
                    scores.append(float((pred[15:] - data[key][4][15:]).abs().mean()))
            score = float(np.mean(scores))
            trace.append({'epoch': epoch, 'validation_macro_mae': score})
            if score < best_score - 1e-6:
                best, best_epoch, best_score, stale = copy.deepcopy(model.state_dict()), epoch, score, 0
            else:
                stale += 1
            if stale >= patience:
                break
    if validation_ids:
        model.load_state_dict(best)
    else:
        best_epoch = epochs
    model.eval()
    return model, best_epoch, trace, outlier_count


def score(batteries, ids, arrays):
    preds = {key: {name: values[key] for name, values in arrays.items()} for key in ids}
    return metrics_by_cell(batteries, preds, ids)['macro']


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    batteries, groups, manifests = load_data()
    folds = []
    selected_epochs = []
    for held_out, val_ids in groups.items():
        train_ids = [key for name, keys in groups.items() if name != held_out for key in keys]
        inner_validation = [key for name, keys in groups.items() if name != held_out
                            for key in keys[-max(2, len(keys)//8):]]
        inner_train = [key for key in train_ids if key not in inner_validation]
        inner_prep = NumericPrep(robust=True).fit(batteries, inner_train)
        inner_data = make_numeric_data(inner_prep, batteries, inner_train + inner_validation)
        _, epoch, trace, _ = numeric_fit(inner_data, batteries, inner_train, 2026,
                                         epochs=60, validation_ids=inner_validation)
        imputer, tree = tree_fit(batteries, train_ids)
        tree_pred = {key: tree.predict(imputer.transform(batteries[key].features)) for key in val_ids}
        prep = NumericPrep(robust=True).fit(batteries, train_ids)
        data = make_numeric_data(prep, batteries, train_ids + val_ids)
        ridge_pred = {key: data[key][3].numpy() for key in val_ids}
        numeric, _, _, downweighted = numeric_fit(
            data, batteries, train_ids, 2026, epochs=epoch)
        with torch.no_grad():
            numeric_pred = {key: numeric(*data[key][:4])[0].numpy() for key in val_ids}
        metric = score(batteries, val_ids,
                       {'ridge': ridge_pred, 'extra_trees': tree_pred, 'numeric_smetimes': numeric_pred})
        folds.append({'held_out_batch': held_out, 'train_cells': len(train_ids),
                      'validation_cells': len(val_ids), 'inner_validation_cells': inner_validation,
                      'selected_epoch': epoch,
                      'downweighted_train_cycles': downweighted,
                      'macro': metric, 'training_trace': trace})
        selected_epochs.append(epoch)
        print('FOLD', held_out, json.dumps(metric), 'epoch', epoch, flush=True)
    (OUT / 'internal_leave_batch_out.json').write_text(json.dumps(folds, indent=2))

    all_ids = [key for ids in groups.values() for key in ids]
    imputer, tree = tree_fit(batteries, all_ids)
    joblib.dump({'imputer': imputer, 'model': tree}, OUT / 'extra_trees.joblib')
    prep = NumericPrep(robust=True).fit(batteries, all_ids)
    joblib.dump(vars(prep), OUT / 'numeric_preprocessing.joblib')
    joblib.dump({'imputer': prep.imputer, 'scaler': prep.scaler,
                 'ridge': prep.ridge, 'feature_clip': [-6, 6]}, OUT / 'soh_ridge.joblib')
    data = make_numeric_data(prep, batteries, all_ids)
    epochs = max(1, int(np.median(selected_epochs)))
    checkpoints = []
    for seed in SEEDS:
        model, _, _, downweighted = numeric_fit(data, batteries, all_ids, seed, epochs)
        name = f'numeric_smetimes_{seed}.pt'
        torch.save({'state_dict': model.state_dict(), 'feature_dim': 51,
                    'seed': seed, 'epochs': epochs, 'training_cells': all_ids}, OUT / name)
        checkpoints.append(name)
        print('FINAL NUMERIC', seed, 'epochs', epochs, flush=True)
    report = {'status': 'trained_waiting_for_external_test', 'dataset': 'MIT Severson original MAT batches 1/2/3',
              'n_cells': len(all_ids), 'n_cycles': int(sum(len(batteries[key].soh) for key in all_ids)),
              'batches': {name: len(ids) for name, ids in groups.items()},
              'batch2_continuations_excluded': [7, 8, 9, 15, 16], 'batch3_bad_collection_excluded': ['b3c37'],
              'target': 'current discharge capacity / first valid-cycle discharge capacity',
              'input': 'first 300 seconds main discharge V/I/T, 51 enhanced features relative to first valid cycle',
              'feature_names': ENHANCED_NAMES,
              'final_numeric_epochs_from_internal_cv': epochs, 'numeric_checkpoints': checkpoints,
              'recommended_soh_model': 'soh_ridge.joblib (lowest nested-CV batch-average MAE)',
              'internal_cv_file': 'internal_leave_batch_out.json',
              'source_manifests': {'batch1': r'D:\LLM\MIT\processed\manifest.json',
                                   'batch2': manifests[2]['source'], 'batch3': manifests[3]['source']}}
    (OUT / 'training_manifest.json').write_text(json.dumps(report, indent=2))
    print('TRAINED', json.dumps({'cells': report['n_cells'], 'cycles': report['n_cycles'],
                                 'epochs': epochs}), flush=True)


if __name__ == '__main__':
    main()
