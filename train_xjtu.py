"""Train XJTU SOH and RUL models with held-out-strategy diagnostics.

Outer validation holds out all cells from one strategy. Epoch choice for the
small numerical SMETimes derivative uses only cells in the outer training set.
All 55 cells are used for the final models; external testing is still pending.
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

from battery_mit_validation import NumericPrep
from battery_mvp import set_seed
from battery_raw_mvp import RawBattery
from battery_smetimes_regularized import RegularizedSMETimes


SOURCE = Path(r'D:\LLM\XJTU\processed')
OUT = Path('battery_final_models/xjtu_all_batches')


def load_data():
    source = json.loads((SOURCE / 'manifest.json').read_text(encoding='utf-8'))
    batteries, groups, lives = {}, {i: [] for i in range(1, 7)}, {}
    for record in source['cells']:
        key = record['id']
        with np.load(SOURCE / f'{key}.npz') as data:
            absolute = data['features'].astype('float32')
            delta = absolute - absolute[0]
            # History is causal and uses observed signals only; no capacity labels.
            history = np.stack([delta[max(0, i-4):i+1].mean(0)
                                for i in range(len(delta))]).astype('float32')
            x = np.concatenate([absolute, delta, history], axis=1)
            cycle = data['cycle'].astype('int32')
            capacity = data['capacity'].astype('float32')
            soh = data['soh'].astype('float32')
            life = int(data['cycle_life'])
        batteries[key] = RawBattery(key, cycle, cycle, x, capacity, soh,
                                    str(SOURCE / f'{key}.npz'))
        groups[record['batch']].append(key)
        lives[key] = life
    if len(batteries) != 55:
        raise ValueError(f'Expected 55 XJTU cells, got {len(batteries)}')
    return source, batteries, groups, lives


def stack(batteries, ids, task, lives=None):
    x, y, weights = [], [], []
    for key in ids:
        battery = batteries[key]
        if task == 'soh':
            feature = battery.features
            target = battery.soh
        else:
            cycle = battery.discharge_number.astype('float32')
            life = lives[key]
            extra = np.stack([cycle, np.log1p(cycle), cycle / 1000], axis=1)
            feature = np.c_[battery.features, extra]
            target = (life - cycle).astype('float32')
        x.append(feature)
        y.append(target)
        weights.append(np.full(len(target), 1 / len(target) / len(ids)))
    return np.concatenate(x), np.concatenate(y), np.concatenate(weights)


def fit_tree(batteries, ids, task, lives=None):
    x, y, weight = stack(batteries, ids, task, lives)
    imputer = SimpleImputer(strategy='median', keep_empty_features=True).fit(x)
    model = ExtraTreesRegressor(n_estimators=160, min_samples_leaf=5,
                                max_features=.8, random_state=2026, n_jobs=-1)
    model.fit(imputer.transform(x), y, sample_weight=weight)
    return {'imputer': imputer, 'model': model}


def predict_tree(fitted, battery, task):
    if task == 'soh':
        x = battery.features
    else:
        cycle = battery.discharge_number.astype('float32')
        x = np.c_[battery.features, cycle, np.log1p(cycle), cycle / 1000]
    prediction = fitted['model'].predict(fitted['imputer'].transform(x))
    return np.clip(prediction, 0, None) if task == 'rul' else prediction


def numeric_fit(data, batteries, ids, epochs, seed, val_ids=None, patience=8):
    set_seed(seed)
    dim = data[ids[0]][0].shape[1]
    model = RegularizedSMETimes(False, feature_dim=dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.03)
    examples, weights = [], []
    for key in ids:
        part = data[key]
        indices = np.arange(0, len(part[0]), 2)
        examples.append([array[indices] for array in part])
        labels = batteries[key].soh
        outliers = np.abs(labels - median_filter(labels, size=7, mode='nearest')) > .04
        weight = np.where(outliers[indices], .25, 1.).astype('float32')
        weights.append(torch.from_numpy(weight / weight.sum() / len(ids)))
    x, history, semantic, base, y = [torch.cat([example[j] for example in examples])
                                     for j in range(5)]
    weight = torch.cat(weights)
    best, best_epoch, best_score, stale = None, 0, float('inf'), 0
    for epoch in range(1, epochs + 1):
        model.train()
        pred, reconstruction, _ = model(x + .01*torch.randn_like(x), history, semantic, base)
        error = (pred - y).abs()
        loss = (torch.where(error <= .025, .5*error.square(), .025*(error-.0125))*weight).sum()
        loss = loss + .0002*(reconstruction-x).square().mean() + .01*(pred-base).square().mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        if val_ids:
            model.eval()
            with torch.no_grad():
                score = float(np.mean([np.mean(np.abs(
                    model(*data[key][:4])[0].numpy() - batteries[key].soh))
                    for key in val_ids]))
            if score < best_score - 1e-6:
                best, best_epoch, best_score, stale = copy.deepcopy(model.state_dict()), epoch, score, 0
            else:
                stale += 1
            if stale >= patience:
                break
    if val_ids:
        model.load_state_dict(best)
    else:
        best_epoch = epochs
    model.eval()
    return model, best_epoch


def inner_epoch(batteries, train_ids, groups):
    validation = [keys[0] for keys in groups.values() if keys and keys[0] in train_ids]
    inner_train = [key for key in train_ids if key not in validation]
    prep = NumericPrep(robust=True).fit(batteries, inner_train)
    data = {key: prep.transform(batteries[key]) for key in inner_train + validation}
    _, epoch = numeric_fit(data, batteries, inner_train, 35, 2026, validation)
    return epoch


def metrics(batteries, lives, ids, predictions, task):
    by_cell = {}
    for key in ids:
        battery = batteries[key]
        truth = battery.soh if task == 'soh' else lives[key] - battery.discharge_number
        estimate = np.asarray(predictions[key])
        by_cell[key] = {'mae': float(np.mean(np.abs(estimate-truth))),
                        'rmse': float(np.sqrt(np.mean((estimate-truth)**2))),
                        'n': int(len(truth))}
    return {'macro_mae': float(np.mean([row['mae'] for row in by_cell.values()])),
            'macro_rmse': float(np.mean([row['rmse'] for row in by_cell.values()])),
            'cells': len(ids), 'observations': sum(row['n'] for row in by_cell.values())}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    source, batteries, groups, lives = load_data()
    folds, epochs = [], []
    for held_batch, val_ids in groups.items():
        train_ids = [key for batch, keys in groups.items() if batch != held_batch for key in keys]
        tree_soh = fit_tree(batteries, train_ids, 'soh')
        tree_rul = fit_tree(batteries, train_ids, 'rul', lives)
        tree_soh_pred = {key: predict_tree(tree_soh, batteries[key], 'soh') for key in val_ids}
        tree_rul_pred = {key: predict_tree(tree_rul, batteries[key], 'rul') for key in val_ids}
        epoch = inner_epoch(batteries, train_ids, groups)
        epochs.append(epoch)
        prep = NumericPrep(robust=True).fit(batteries, train_ids)
        data = {key: prep.transform(batteries[key]) for key in train_ids + val_ids}
        numeric, _ = numeric_fit(data, batteries, train_ids, epoch, 2026)
        with torch.no_grad():
            numeric_pred = {key: numeric(*data[key][:4])[0].numpy() for key in val_ids}
        fold = {'held_out_batch': held_batch, 'train_cells': len(train_ids),
                'validation_cells': len(val_ids), 'inner_selected_epochs': epoch,
                'soh_extra_trees': metrics(batteries, lives, val_ids, tree_soh_pred, 'soh'),
                'soh_numeric_smetimes': metrics(batteries, lives, val_ids, numeric_pred, 'soh'),
                'rul_extra_trees': metrics(batteries, lives, val_ids, tree_rul_pred, 'rul')}
        folds.append(fold)
        print('FOLD', held_batch, json.dumps(fold), flush=True)
    (OUT / 'internal_leave_batch_out.json').write_text(json.dumps(folds, indent=2), encoding='utf-8')

    # A separate development diagnostic for new cells under known strategies.
    # The held-out whole cells are included only after this check, in final fit.
    cell_val_ids = [key for keys in groups.values() for key in keys[-2:]]
    cell_train_ids = [key for keys in groups.values() for key in keys[:-2]]
    cell_soh = fit_tree(batteries, cell_train_ids, 'soh')
    cell_rul = fit_tree(batteries, cell_train_ids, 'rul', lives)
    cell_soh_pred = {key: predict_tree(cell_soh, batteries[key], 'soh') for key in cell_val_ids}
    cell_rul_pred = {key: predict_tree(cell_rul, batteries[key], 'rul') for key in cell_val_ids}
    cell_report = {
        'validation_design': 'last two cell IDs of each strategy; no cycles from those cells in fit',
        'training_cells': cell_train_ids, 'validation_cells': cell_val_ids,
        'soh_extra_trees': metrics(batteries, lives, cell_val_ids, cell_soh_pred, 'soh'),
        'rul_extra_trees': metrics(batteries, lives, cell_val_ids, cell_rul_pred, 'rul'),
        'by_batch': {str(batch): {
            'soh_extra_trees': metrics(batteries, lives, keys[-2:], cell_soh_pred, 'soh'),
            'rul_extra_trees': metrics(batteries, lives, keys[-2:], cell_rul_pred, 'rul')}
            for batch, keys in groups.items()}}
    (OUT / 'internal_cell_holdout.json').write_text(json.dumps(cell_report, indent=2), encoding='utf-8')
    print('CELL HOLDOUT', json.dumps({task: cell_report[task] for task in
                                      ('soh_extra_trees', 'rul_extra_trees')}), flush=True)

    all_ids = [key for keys in groups.values() for key in keys]
    joblib.dump(fit_tree(batteries, all_ids, 'soh'), OUT / 'soh_extra_trees.joblib')
    joblib.dump(fit_tree(batteries, all_ids, 'rul', lives), OUT / 'rul_extra_trees.joblib')
    prep = NumericPrep(robust=True).fit(batteries, all_ids)
    joblib.dump(vars(prep), OUT / 'soh_numeric_preprocessing.joblib')
    data = {key: prep.transform(batteries[key]) for key in all_ids}
    final_epochs = max(1, int(np.median(epochs)))
    checkpoints = []
    for seed in (2026, 2027, 2028):
        model, _ = numeric_fit(data, batteries, all_ids, final_epochs, seed)
        name = f'soh_numeric_smetimes_{seed}.pt'
        torch.save({'state_dict': model.state_dict(), 'feature_dim': batteries[all_ids[0]].features.shape[1],
                    'training_cells': all_ids, 'epochs': final_epochs, 'seed': seed}, OUT / name)
        checkpoints.append(name)
    manifest = {'status': 'trained_waiting_for_external_test',
                'dataset': 'XJTU original MAT, 55 cells, six strategies',
                'source': source['source'], 'cells': len(all_ids),
                'diagnostics': int(sum(len(batteries[key].soh) for key in all_ids)),
                'batches': {str(batch): len(keys) for batch, keys in groups.items()},
                'soh_target': source['label'], 'rul_target': 'source cycle_life minus observed cycle number',
                'input': source['input'] + '; absolute, first-reference delta and causal 5-observation mean',
                'base_feature_names': source['features'],
                'recommended_soh_model': 'soh_extra_trees.joblib',
                'numeric_smetimes_status': 'experimental; large held-out-strategy error, not default',
                'recommended_rul_model': 'rul_extra_trees.joblib (development baseline; external test required)',
                'final_numeric_epochs_from_inner_cv': final_epochs,
                'numeric_checkpoints': checkpoints,
                'internal_validation': 'leave-one-strategy-out and whole-cell holdout; inner cells select numeric epochs',
                'source_manifest': str(SOURCE / 'manifest.json')}
    (OUT / 'training_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('TRAINED', json.dumps({'cells': len(all_ids), 'diagnostics': manifest['diagnostics'],
                                 'epochs': final_epochs}), flush=True)


if __name__ == '__main__':
    main()
