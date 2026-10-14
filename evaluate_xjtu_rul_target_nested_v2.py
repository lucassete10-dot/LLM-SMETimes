"""Nested leave-strategy XJTU RUL target-transform selection.

For each untouched outer strategy, choose log1p or raw-cycle target using only
inner held-out strategies from the outer training set. Outer scores reuse the
already-computed, identical-fit target-transform audit. This family was still
proposed after development inspection, so the result is not independent.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from battery_v2_features import load_all
from evaluate_xjtu_rul_target_v2 import fit_raw_target, score_summary
from train_rul_v2 import assess, fit_rul, predict_rul, signal_features
from train_soh_v2 import outer_splits


SOURCE = Path('battery_models_v2/rul/xjtu/target_transform_diagnostic.json')
OUT = Path('battery_models_v2/rul/xjtu/target_transform_nested_cv.json')
TARGETS = ('log1p_target', 'raw_cycle_target')


def choose_target(inner_folds):
    means = {name: float(np.mean([fold[name]['cell_macro_mae_cycles']
                                  for fold in inner_folds])) for name in TARGETS}
    # The already-selected log target wins an exact tie; avoid needless churn.
    selected = min(TARGETS, key=lambda name: (means[name], TARGETS.index(name)))
    return selected, means


def evaluate():
    manifest, all_cells = load_all()
    cells = {key: cell for key, cell in all_cells.items() if cell.dataset == 'XJTU'}
    prior = {fold['held_out']: fold for fold in json.loads(
        SOURCE.read_text(encoding='utf-8'))['folds']}
    outer_folds = []
    for outer, outer_train, outer_val in outer_splits(manifest['cells'], 'XJTU'):
        training_records = [row for row in manifest['cells']
                            if row['dataset'] == 'XJTU' and row['cell_id'] in outer_train]
        inner = []
        for group, train, val in outer_splits(training_records, 'XJTU'):
            median_life = float(np.median([cells[key].endpoint_cycle for key in train]))
            log_model = fit_rul(cells, train, 'direct')
            log_pred = {key: predict_rul(log_model, cells[key]) for key in val}
            raw_imputer, raw_model = fit_raw_target(cells, train)
            raw_pred = {key: np.maximum(0, raw_model.predict(
                raw_imputer.transform(signal_features(cells[key])))) for key in val}
            inner.append({'held_out': group,
                          'log1p_target': assess(cells, val, log_pred, median_life),
                          'raw_cycle_target': assess(cells, val, raw_pred, median_life)})
        selected, means = choose_target(inner)
        if outer not in prior or {
                row['cell_id'] for row in prior[outer][selected]['by_cell']} != set(outer_val):
            raise ValueError(f'{outer}: stored outer target-transform scores mismatch')
        outer_folds.append({'held_out': outer, 'selected_target': selected,
                            'inner_equal_fold_mae_cycles': means,
                            'inner_folds': inner,
                            'nested_selected': prior[outer][selected],
                            'log1p_target': prior[outer]['log1p_target'],
                            'raw_cycle_target': prior[outer]['raw_cycle_target']})
        print(outer, 'inner', {name: round(value, 1) for name, value in means.items()},
              'selected', selected, 'outer_mae',
              round(prior[outer][selected]['cell_macro_mae_cycles'], 1), flush=True)
    summary = {name: score_summary(outer_folds, name)
               for name in ('log1p_target', 'raw_cycle_target', 'nested_selected')}
    summary['outer_folds_selecting_raw'] = sum(
        fold['selected_target'] == 'raw_cycle_target' for fold in outer_folds)
    summary['nested_improved_outer_folds_vs_log'] = sum(
        fold['nested_selected']['cell_macro_mae_cycles'] <
        fold['log1p_target']['cell_macro_mae_cycles'] for fold in outer_folds)
    return {'status': 'nested development CV only; no RUL model replaced',
            'selection': 'five inner whole-strategy folds within each of six outer holdouts',
            'warning': 'target-transform family was considered after earlier development '
                       'errors; nested result is not an independent deployment guarantee',
            'summary': summary, 'folds': outer_folds}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    current = evaluate()
    if args.verify:
        stored = json.loads(OUT.read_text(encoding='utf-8'))
        for name in ('log1p_target', 'raw_cycle_target', 'nested_selected'):
            if not np.isclose(current['summary'][name]['equal_fold_macro_mae_cycles'],
                              stored['summary'][name]['equal_fold_macro_mae_cycles'],
                              atol=1e-8, rtol=1e-10):
                raise ValueError(f'{name}: nested score differs')
        if [fold['selected_target'] for fold in current['folds']] != [
                fold['selected_target'] for fold in stored['folds']]:
            raise ValueError('nested selections differ')
        print('REPRODUCED', OUT)
    else:
        if OUT.exists():
            raise FileExistsError(OUT)
        OUT.write_text(json.dumps(current, indent=2), encoding='utf-8')
        print('WROTE', OUT)
