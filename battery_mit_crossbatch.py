"""Frozen Batch 1 checkpoints evaluated on independent Batch 2/3 cells.

No fitting, feature selection, threshold tuning, or epoch selection occurs here.
"""
import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from battery_mit_validation import (NumericPrep, battery, metrics_by_cell,
                                    plot_examples, save_predictions)
from battery_smetimes_regularized import RegularizedSMETimes
from prepare_mit import COMMON_NAMES, ENHANCED_NAMES


PROCESSED = Path(r'D:\LLM\MIT\processed')
CHECKPOINTS = Path('battery_mit_results/mit_cell_holdout')
OUTPUT = Path('battery_mit_results/crossbatch_frozen_batch1')


def load_batch(number):
    folder = PROCESSED / f'batch{number}'
    manifest = json.loads((folder / 'manifest.json').read_text())
    common, enhanced = {}, {}
    # Original authors' LoadData.m removes original Batch 3 cell 38
    # (zero-based b3c37) for a documented data-collection problem.
    excluded = {'b3c37'} if number == 3 else set()
    for key in manifest['saved_cells']:
        if key in excluded:
            continue
        with np.load(folder / f'{key}.npz') as data:
            if list(data['common_names']) != COMMON_NAMES or list(data['enhanced_names']) != ENHANCED_NAMES:
                raise ValueError(f'Schema mismatch in {key}')
            cycle, capacity = data['cycle'], data['capacity']
            common[key] = battery(key, data['common'], capacity, cycle, folder / f'{key}.npz')
            enhanced[key] = battery(key, data['enhanced'], capacity, cycle, folder / f'{key}.npz')
    return common, enhanced, manifest


def load_prep(name):
    prep = NumericPrep()
    vars(prep).update(joblib.load(CHECKPOINTS / name))
    return prep


def neural_predictions(name, prep, batteries):
    ids = list(batteries)
    data = {key: prep.transform(batteries[key]) for key in ids}
    runs = []
    seeds = [2026] if name == 'enhanced_mse' else [2026, 2027, 2028]
    with torch.no_grad():
        for seed in seeds:
            saved = torch.load(CHECKPOINTS / f'{name}_{seed}.pt', map_location='cpu', weights_only=False)
            model = RegularizedSMETimes(False, feature_dim=saved['feature_dim'])
            model.load_state_dict(saved['state_dict'])
            model.eval()
            runs.append({key: model(*data[key][:4])[0].numpy() for key in ids})
    return {key: np.mean([run[key] for run in runs], axis=0) for key in ids}


def evaluate(number):
    common, enhanced, manifest = load_batch(number)
    if number == 2 and any(f'b2c{i}' in common for i in (7, 8, 9, 15, 16)):
        raise ValueError('Batch 1 continuation cell would leak into Batch 2 test')
    if set(common) != set(enhanced):
        raise ValueError('Common/enhanced cell mismatch')
    folder = OUTPUT / f'batch{number}'
    folder.mkdir(parents=True, exist_ok=True)
    common_prep = load_prep('common_preprocessing.joblib')
    rich_prep = load_prep('enhanced_preprocessing.joblib')
    forest = joblib.load(CHECKPOINTS / 'common_random_forest.joblib')
    trees = joblib.load(CHECKPOINTS / 'enhanced_extra_trees.joblib')
    ids = list(common)
    preds = {}
    for key in ids:
        base = common_prep.transform(common[key])
        preds[key] = {
            'constant_one': np.ones(len(common[key].soh), dtype='float32'),
            'common_ridge': base[3].numpy(),
            'common_rf': forest.predict(common[key].features),
            'enhanced_extra_trees': trees.predict(rich_prep.imputer.transform(enhanced[key].features)),
        }
    for name, prep, batteries in [('common_numeric', common_prep, common),
                                   ('enhanced_mse', rich_prep, enhanced),
                                   ('enhanced_robust', rich_prep, enhanced)]:
        values = neural_predictions(name, prep, batteries)
        for key in ids:
            preds[key][name] = values[key]
        print(f'Batch {number} {name} evaluated', flush=True)
    results = metrics_by_cell(common, preds, ids)
    subgroups = {}
    for label, condition in [('below_0p9', lambda y: y < .9),
                              ('at_least_0p9', lambda y: y >= .9)]:
        y = np.concatenate([common[key].soh[15:] for key in ids])
        mask = condition(y)
        subgroups[label] = {'observations': int(mask.sum()),
            'mae': {name: float(np.mean(np.abs(np.concatenate([preds[key][name][15:] for key in ids])[mask] - y[mask])))
                    if mask.any() else None for name in preds[ids[0]]}}
    results['soh_subgroups'] = subgroups
    results['source_manifest'] = str(PROCESSED / f'batch{number}' / 'manifest.json')
    results['author_excluded_bad_collection_cells'] = ['b3c37'] if number == 3 else []
    results['frozen_checkpoints'] = str(CHECKPOINTS)
    results['protocol'] = 'No Batch 2/3 data used to fit Batch 1 preprocessors or models'
    (folder / 'metrics.json').write_text(json.dumps(results, indent=2))
    save_predictions(folder / 'predictions.csv', common, preds, ids)
    plot_examples(folder / 'examples.png', common, preds, ids,
                  f'MIT Batch {number}: frozen Batch 1 models, first 300 s')
    print('BATCH', number, json.dumps({'cells': results['cells'],
           'observations': results['observations'], 'macro': results['macro'],
           'subgroups': subgroups, 'audit': manifest['excluded_reasons']}), flush=True)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('batch', type=int, choices=[2, 3])
    args = parser.parse_args()
    torch.set_num_threads(4)
    evaluate(args.batch)
