"""Development-only XJTU 300s charge/discharge SOH ablation.

Use the same whole-strategy folds, ExtraTrees estimator, per-cell weights and
capacity-label masks as the existing 120s/300s experiments. The original
300s features contain 41 charge columns followed by 41 discharge columns.
No new model is selected or deployed from this post-hoc diagnostic.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from battery_v2_features import load_all
from evaluate_xjtu_long_prefix_v2 import fit, matrix
from train_soh_v2 import outer_splits, score


ROOT = Path('battery_models_v2/xjtu_long_prefix')
OUTPUT = ROOT / 'stage_ablation.json'


def stage_views(raw):
    raw = np.asarray(raw, dtype='float32')
    if raw.ndim != 2 or raw.shape[1] != 82:
        raise ValueError('XJTU 300s signal must have 82 columns')
    return {'charge_300s': raw[:, :41], 'discharge_300s': raw[:, 41:]}


def evaluate():
    manifest, all_cells = load_all()
    cells = {key: cell for key, cell in all_cells.items() if cell.dataset == 'XJTU'}
    records = [row for row in manifest['cells'] if row['dataset'] == 'XJTU']
    baseline = {row['held_out']: row['extra_trees_history'] for row in json.loads(
        Path('battery_models_v2/soh/xjtu/group_cv.json').read_text(encoding='utf-8'))}
    prior_300 = {row['held_out']: row['300s'] for row in json.loads(
        (ROOT / 'group_cv.json').read_text(encoding='utf-8'))['folds']}
    views = {'charge_300s': {}, 'discharge_300s': {}}
    with np.load(ROOT / 'features.npz', allow_pickle=False) as archive:
        for row in records:
            key = row['cell_id']
            if not np.array_equal(archive[f'{key}_cycle'], cells[key].cycle):
                raise ValueError(f'{key}: 300s feature/capacity cycle mismatch')
            for name, raw in stage_views(archive[f'{key}_300s']).items():
                views[name][key] = matrix(raw, cells[key].reference_count)
    total_diagnostics = sum(len(cell.cycle) for cell in cells.values())
    if len(cells) != 55 or total_diagnostics != 10389:
        raise ValueError('XJTU development cohort differs from audited 55-cell corpus')
    folds = []
    for group, train_ids, val_ids in outer_splits(manifest['cells'], 'XJTU'):
        if group not in baseline or group not in prior_300:
            raise ValueError(f'{group}: missing stored baseline fold')
        fold = {'held_out': group, 'baseline_120s': baseline[group],
                'full_300s': prior_300[group]}
        for name, by_cell in views.items():
            imputer, model = fit(cells, by_cell, train_ids)
            predictions = {key: model.predict(imputer.transform(by_cell[key]))
                           for key in val_ids}
            fold[name] = score(cells, predictions, val_ids)
        folds.append(fold)
        print(group, {name: round(fold[name]['cell_macro_mae'], 5)
                      for name in ('baseline_120s', 'full_300s', *views)}, flush=True)
    names = ('baseline_120s', 'full_300s', *views)
    summary = {name: {
        'equal_fold_macro_mae': float(np.mean([row[name]['cell_macro_mae']
                                              for row in folds])),
        'worst_cell_mae': float(max(cell['mae'] for row in folds
                                    for cell in row[name]['by_cell'])),
        'folds_improved_vs_120s': sum(row[name]['cell_macro_mae'] <
                                      row['baseline_120s']['cell_macro_mae']
                                      for row in folds),
        'folds_improved_vs_full_300s': sum(row[name]['cell_macro_mae'] <
                                          row['full_300s']['cell_macro_mae']
                                          for row in folds),
    } for name in names}
    return {'status': 'post-hoc development ablation; no model selected or deployed',
            'data': {'cells': len(cells), 'diagnostics': total_diagnostics},
            'evaluation': 'six whole-strategy holdouts; same fit and score as 300s candidate',
            'warning': 'using the same development folds for exploration is not independent evidence',
            'summary': summary, 'folds': folds}


def assert_reproduced(saved, current, path='report'):
    """Accept floating-point roundoff, but no substantive metric drift."""
    if isinstance(saved, dict) and isinstance(current, dict):
        if saved.keys() != current.keys():
            raise ValueError(f'{path}: keys differ')
        for key in saved:
            assert_reproduced(saved[key], current[key], f'{path}.{key}')
    elif isinstance(saved, list) and isinstance(current, list):
        if len(saved) != len(current):
            raise ValueError(f'{path}: list length differs')
        for index, (left, right) in enumerate(zip(saved, current)):
            assert_reproduced(left, right, f'{path}[{index}]')
    elif (isinstance(saved, (int, float)) and not isinstance(saved, bool) and
          isinstance(current, (int, float)) and not isinstance(current, bool)):
        if not np.isclose(saved, current, atol=1e-10, rtol=1e-10):
            raise ValueError(f'{path}: metric differs: {saved} versus {current}')
    elif saved != current:
        raise ValueError(f'{path}: value differs: {saved} versus {current}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true',
                        help='retrain the six folds and compare to the saved diagnostic')
    args = parser.parse_args()
    result = evaluate()
    if args.verify:
        assert_reproduced(json.loads(OUTPUT.read_text(encoding='utf-8')), result)
        print('REPRODUCED', OUTPUT, flush=True)
    else:
        if OUTPUT.exists():
            raise FileExistsError(OUTPUT)
        OUTPUT.write_text(json.dumps(result, indent=2), encoding='utf-8')
        print('WROTE', OUTPUT, flush=True)
