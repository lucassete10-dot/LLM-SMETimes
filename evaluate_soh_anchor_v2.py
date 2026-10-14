"""Development-only, label-free initial-anchor correction for SOH predictions.

The SOH reference is defined from the first one/five diagnostics. This test
asks whether making the model's predictions agree with that known definition
helps on unseen cells/strategies. No held-out capacity is read by the correction.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from battery_v2_features import load_all
from train_soh_v2 import fit, outer_splits, predict, score


OUT = Path('battery_models_v2/soh_anchor_diagnostic.json')
ALPHAS = (0., .5, 1.)


def anchored_prediction(prediction, reference_count, alpha):
    prediction = np.asarray(prediction, dtype='float64')
    if prediction.ndim != 1 or not 1 <= reference_count <= len(prediction):
        raise ValueError('invalid prediction/reference length')
    if not np.isfinite(prediction[:reference_count]).all() or not 0 <= alpha <= 1:
        raise ValueError('nonfinite reference prediction or invalid strength')
    reference_prediction = float(np.median(prediction[:reference_count]))
    return prediction + alpha * (1. - reference_prediction)


def evaluate_domain(manifest, cells, domain):
    name = 'extra_trees_history'
    folds = []
    for group, train_ids, val_ids in outer_splits(manifest['cells'], domain):
        fitted = fit(cells, train_ids, name)
        base = {key: predict(fitted, cells[key]) for key in val_ids}
        corrections = {
            str(alpha): {key: anchored_prediction(base[key], cells[key].reference_count, alpha)
                         for key in val_ids}
            for alpha in ALPHAS
        }
        anchor_offsets = {key: float(1. - np.median(base[key][:cells[key].reference_count]))
                          for key in val_ids}
        folds.append({
            'held_out': group,
            'cells': len(val_ids),
            'reference_prediction_offset_by_cell': anchor_offsets,
            'strengths': {strength: score(cells, values, val_ids)
                          for strength, values in corrections.items()},
        })
        print(domain, group, {a: round(folds[-1]['strengths'][str(a)]['cell_macro_mae'], 5)
                              for a in ALPHAS}, flush=True)
    summary = {str(alpha): {
        'equal_fold_macro_mae': float(np.mean([
            fold['strengths'][str(alpha)]['cell_macro_mae'] for fold in folds])),
        'equal_cell_macro_mae': float(np.mean([
            row['mae'] for fold in folds
            for row in fold['strengths'][str(alpha)]['by_cell']])),
        'worst_cell_mae': float(np.max([
            row['mae'] for fold in folds
            for row in fold['strengths'][str(alpha)]['by_cell']])),
        'folds_improved_vs_uncorrected': sum(
            fold['strengths'][str(alpha)]['cell_macro_mae'] <
            fold['strengths']['0.0']['cell_macro_mae'] for fold in folds),
    } for alpha in ALPHAS}
    return {'domain': domain, 'model': name,
            'method': 'prediction + strength * (1 - median(predicted reference diagnostics))',
            'correction_inputs': 'predictions on first reference diagnostics only; no capacity labels',
            'validation': 'whole-cell or whole-group outer folds; diagnostic only',
            'summary': summary, 'folds': folds}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domains', nargs='+', choices=('NASA', 'CALCE', 'MIT', 'XJTU'),
                        default=['NASA', 'CALCE', 'MIT', 'XJTU'])
    args = parser.parse_args()
    manifest, cells = load_all()
    report = {'status': 'development diagnostic; not selected or calibrated',
              'domains': {domain: evaluate_domain(manifest, cells, domain)
                          for domain in args.domains}}
    OUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Wrote', OUT)


if __name__ == '__main__':
    main()
