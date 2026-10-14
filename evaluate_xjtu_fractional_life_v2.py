"""XJTU RUL target ablation: predict fraction of source lifetime consumed.

Whole-strategy development CV with the same 600-second throughput-aligned
diagnostic and past usage features as the prior XJTU challenger. A fraction
target can in principle imply a life beyond the longest training cell, unlike
direct tree regression of RUL. It must be tested, not assumed to generalize.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer

from battery_v2_features import load_cell, load_manifest
from evaluate_xjtu_combined_rul_v2 import matrix
from evaluate_xjtu_throughput_rul_v2 import (LONG_ROOT, USAGE,
                                              throughput_matrix)
from train_rul_v2 import eligible
from train_soh_v2 import outer_splits


OUT = Path('battery_models_v2/xjtu_fractional_life_challenger')
LANDMARKS = (50, 100, 200, 300, 500)
MAX_LIFE = 2500


def fit_fraction(cells, ids, features):
    parts = [[], [], []]
    active = []
    for key in ids:
        cell = cells[key]
        keep = eligible(cell)
        if not keep.any() or cell.endpoint_cycle is None:
            continue
        active.append(key)
        x = features[key][keep][::5]
        progress = (cell.cycle[keep]/cell.endpoint_cycle)[::5]
        weight = np.where(cell.capacity_review_flag[keep], .25, 1.).astype(float)
        weight = (weight/weight.sum())[::5]
        for target, value in zip(parts, (x, progress, weight)):
            target.append(value)
    if len(active) < 3:
        raise ValueError('too few observed endpoint cells')
    x, y, weight = map(np.concatenate, parts)
    weight /= len(active)
    imputer = SimpleImputer(strategy='median', keep_empty_features=True).fit(x)
    model = ExtraTreesRegressor(n_estimators=120, min_samples_leaf=8,
                                max_features=.8, random_state=2026, n_jobs=-1)
    model.fit(imputer.transform(x), y, sample_weight=weight)
    return imputer, model


def predict_endpoint(bundle, cell, features):
    imputer, model = bundle
    fraction = np.clip(model.predict(imputer.transform(features)), .01, 1.)
    return np.clip(cell.cycle/fraction, cell.cycle, MAX_LIFE)


def score_cell(cell, endpoint_prediction):
    origin = max(15, cell.reference_count)
    if origin >= len(cell.cycle) or cell.cycle[origin] >= cell.endpoint_cycle:
        return None
    result = {'cell_id': cell.cell_id, 'group': cell.group,
              'origin_cycle': int(cell.cycle[origin]),
              'origin_eol_ape_percent': float(100*abs(
                  endpoint_prediction[origin]-cell.endpoint_cycle)/cell.endpoint_cycle),
              'origin_predicted_endpoint_cycle': float(endpoint_prediction[origin]),
              'landmarks': {}}
    for landmark in LANDMARKS:
        if cell.endpoint_cycle <= landmark:
            continue
        candidates = np.flatnonzero(cell.cycle <= landmark)
        if not len(candidates) or candidates[-1] < origin:
            continue
        index = int(candidates[-1])
        result['landmarks'][str(landmark)] = {
            'observed_cycle': int(cell.cycle[index]),
            'eol_ape_percent': float(100*abs(
                endpoint_prediction[index]-cell.endpoint_cycle)/cell.endpoint_cycle)}
    return result


def main():
    manifest = load_manifest()
    records = [row for row in manifest['cells'] if row['dataset'] == 'XJTU']
    cells = {row['cell_id']: load_cell(row) for row in records}
    features = {}
    with np.load(LONG_ROOT / 'features.npz') as long_file, np.load(USAGE) as usage_file:
        for key, cell in cells.items():
            if not (np.array_equal(long_file[f'{key}_cycle'], cell.cycle) and
                    np.array_equal(usage_file[f'{key}_cycle'], cell.cycle)):
                raise ValueError(f'{key}: prefix/usage cycle mismatch')
            raw = throughput_matrix(long_file[f'{key}_600s'])
            features[key] = matrix(cell, raw, usage_file[f'{key}_usage'])
    folds = []
    heldout_predictions = {}
    for group, train, val in outer_splits(records, 'XJTU'):
        model = fit_fraction(cells, train, features)
        scored = []
        for key in val:
            endpoint_prediction = predict_endpoint(model, cells[key], features[key])
            heldout_predictions[key] = endpoint_prediction
            result = score_cell(cells[key], endpoint_prediction)
            if result is not None:
                scored.append(result)
        folds.append({'held_out': group, 'scored_cells': len(scored),
                      'origin_mape_percent': float(np.mean([
                          row['origin_eol_ape_percent'] for row in scored])),
                      'by_cell': scored})
        print(group, 'origin EOL MAPE', round(folds[-1]['origin_mape_percent'], 2),
              'cells', len(scored), flush=True)
    scored = [row for fold in folds for row in fold['by_cell']]
    landmarks = {}
    for landmark in LANDMARKS:
        rows = [row['landmarks'][str(landmark)]['eol_ape_percent'] for row in scored
                if str(landmark) in row['landmarks']]
        landmarks[str(landmark)] = {'scored_cells': len(rows),
                                    'eol_mape_percent': float(np.mean(rows)) if rows else None}
    report = {'status': 'unpromoted whole-strategy development CV; not independent test',
              'input': '600-second charge/discharge prefix and past usage controls; '
                       'no measured capacity',
              'target': 'source-defined endpoint via fraction cycle/endpoint',
              'endpoint_warning': 'not exact first per-cell 80% crossing',
              'summary': {'scored_cells': len(scored),
                          'origin_eol_mape_percent': float(np.mean([
                              row['origin_eol_ape_percent'] for row in scored])),
                          'landmarks': landmarks},
              'folds': folds}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'development_cv.json'
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    row_id, row_cycle, row_endpoint = [], [], []
    for key, cell in cells.items():
        keep = eligible(cell)
        row_id.extend([key]*int(keep.sum()))
        row_cycle.extend(cell.cycle[keep].tolist())
        row_endpoint.extend(heldout_predictions[key][keep].tolist())
    np.savez_compressed(OUT / 'development_oof_endpoint.npz',
                        cell_id=np.asarray(row_id), cycle=np.asarray(row_cycle),
                        endpoint=np.asarray(row_endpoint))
    print('SUMMARY', report['summary'])
    print('WROTE', path)


if __name__ == '__main__':
    main()
