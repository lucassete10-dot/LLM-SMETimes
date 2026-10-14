"""Nested leave-strategy-out check of absolute/full SOH model blending.

The blend family was motivated by development results, so even nested CV is
still a post-hoc development diagnostic, not an independent test. Every outer
prediction is trained without that strategy; the blend weight is selected
using only inner folds from the outer training strategies.
"""

import json
from pathlib import Path

import numpy as np

from battery_v2_features import load_all
from evaluate_relative_soh_features_v2 import fit_view, predict_view
from train_soh_v2 import fit as fit_soh, outer_splits, predict as predict_soh, score


OUT = Path('battery_models_v2/xjtu_nested_blend_diagnostic.json')
WEIGHTS = (0., .25, .5, .75, 1.)  # weight on raw-only model


def blend(full, raw, weight):
    if not 0 <= weight <= 1 or set(full) != set(raw):
        raise ValueError('invalid blend inputs')
    return {key: (1 - weight) * np.asarray(full[key]) + weight * np.asarray(raw[key])
            for key in full}


def predict_pair(cells, training, validation):
    full_model = fit_soh(cells, training, 'extra_trees_history')
    raw_model = fit_view(cells, training, 'raw_only')
    full = {key: predict_soh(full_model, cells[key]) for key in validation}
    raw = {key: predict_view(raw_model, cells[key], 'raw_only') for key in validation}
    return full, raw


def select_inner_weight(cells, records, outer_training):
    rows = [row for row in records if row['cell_id'] in outer_training]
    groups = sorted({row['group'] for row in rows})
    if len(groups) < 2:
        raise ValueError('nested selection requires two training strategies')
    inner = []
    for group in groups:
        hold = [row['cell_id'] for row in rows if row['group'] == group]
        train = [key for key in outer_training if key not in hold]
        full, raw = predict_pair(cells, train, hold)
        inner.append({'held_out': group, 'mae_by_weight': {
            str(w): score(cells, blend(full, raw, w), hold)['cell_macro_mae']
            for w in WEIGHTS}})
    means = {w: float(np.mean([row['mae_by_weight'][str(w)] for row in inner]))
             for w in WEIGHTS}
    selected = min(WEIGHTS, key=lambda w: (means[w], abs(w - .5)))
    return selected, means, inner


def main():
    manifest, cells = load_all()
    folds = []
    for group, training, validation in outer_splits(manifest['cells'], 'XJTU'):
        selected, inner_means, inner = select_inner_weight(cells, manifest['cells'], training)
        full, raw = predict_pair(cells, training, validation)
        scores = {str(w): score(cells, blend(full, raw, w), validation)
                  for w in (0., .5, selected)}
        folds.append({'held_out': group, 'selected_raw_weight': selected,
                      'inner_equal_fold_mae_by_weight': {str(w): value for w, value in inner_means.items()},
                      'inner_folds': inner, 'outer_scores': scores})
        print(group, 'selected', selected, 'baseline',
              round(scores['0.0']['cell_macro_mae'], 5), 'nested',
              round(scores[str(selected)]['cell_macro_mae'], 5), flush=True)
    def summary(which):
        values = [row['outer_scores'][str(which(row))] for row in folds]
        return {'equal_fold_macro_mae': float(np.mean([v['cell_macro_mae'] for v in values])),
                'equal_cell_macro_mae': float(np.mean([cell['mae'] for v in values
                                                       for cell in v['by_cell']])),
                'worst_cell_mae': float(np.max([cell['mae'] for v in values
                                                for cell in v['by_cell']])),
                'improved_outer_folds_vs_baseline': sum(
                    v['cell_macro_mae'] < row['outer_scores']['0.0']['cell_macro_mae']
                    for v, row in zip(values, folds))}
    report = {'status': 'post-hoc nested development diagnostic; not independent or deployed',
              'weight_grid': WEIGHTS,
              'selection': 'equal-inner-strategy macro MAE; no outer strategy labels used',
              'summary': {'baseline': summary(lambda row: 0.),
                          'fixed_half': summary(lambda row: .5),
                          'nested_selected': summary(lambda row: row['selected_raw_weight'])},
              'folds': folds}
    OUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Wrote', OUT)


if __name__ == '__main__':
    main()
