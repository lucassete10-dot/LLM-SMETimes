"""Causal endpoint-stability diagnostic on frozen development OOF RUL output.

For a fixed observed end-of-life event, cycle + RUL is the predicted event
cycle. This experiment smooths that quantity using only current/past held-out
predictions, then checks both prediction jumps and accuracy. No model is fit.
"""

import json
from pathlib import Path

import numpy as np


OUT = Path('battery_models_v2/rul_temporal_stability_diagnostic.json')
MODES = ('raw', 'last5_records', 'last15_records', 'last50_cycles')


def smooth_endpoint(cycle, rul, mode):
    cycle = np.asarray(cycle, dtype='int64')
    rul = np.asarray(rul, dtype='float64')
    if cycle.ndim != 1 or len(cycle) != len(rul) or not len(cycle):
        raise ValueError('cycle/RUL must be aligned nonempty vectors')
    if np.any(np.diff(cycle) <= 0) or not np.isfinite(rul).all():
        raise ValueError('cycle must increase and RUL must be finite')
    endpoint = cycle + rul
    if mode == 'raw':
        return endpoint
    if mode not in MODES:
        raise ValueError(f'unknown smoothing mode: {mode}')
    result = np.empty(len(cycle), dtype='float64')
    for index in range(len(cycle)):
        if mode == 'last5_records':
            start = max(0, index - 4)
        elif mode == 'last15_records':
            start = max(0, index - 14)
        else:
            start = int(np.searchsorted(cycle, cycle[index] - 50))
        result[index] = np.median(endpoint[start:index + 1])
    return result


def score_cell(cycle, target, estimate, endpoint, mode):
    event_estimate = smooth_endpoint(cycle, estimate, mode)
    residual = np.maximum(0, event_estimate - cycle) - target
    progress = cycle / endpoint
    jumps = np.abs(np.diff(event_estimate))
    spans = np.diff(cycle)
    return {'n': int(len(cycle)),
            'mae_cycles': float(np.mean(np.abs(residual))),
            'signed_bias_cycles': float(np.mean(residual)),
            'early_mae_cycles': float(np.mean(np.abs(residual[progress < .33])))
            if np.any(progress < .33) else None,
            'middle_mae_cycles': float(np.mean(np.abs(residual[
                (progress >= .33) & (progress < .67)])))
            if np.any((progress >= .33) & (progress < .67)) else None,
            'late_mae_cycles': float(np.mean(np.abs(residual[progress >= .67])))
            if np.any(progress >= .67) else None,
            'median_endpoint_jump_per_observation': float(np.median(jumps))
            if len(jumps) else None,
            'p90_endpoint_jump_per_observation': float(np.quantile(jumps, .9))
            if len(jumps) else None,
            'median_endpoint_jump_per_cycle': float(np.median(jumps / spans))
            if len(jumps) else None}


def evaluate_domain(domain):
    path = Path('battery_models_v2/rul') / domain.lower() / 'development_oof_predictions.npz'
    with np.load(path) as data:
        rows = {key: data[key].copy() for key in data.files}
    cells = {}
    for key in np.unique(rows['cell_id']):
        where = np.flatnonzero(rows['cell_id'] == key)
        order = where[np.argsort(rows['cycle'][where])]
        cycle = rows['cycle'][order]
        endpoint = rows['endpoint'][order]
        if len(np.unique(endpoint)) != 1 or not np.allclose(rows['target'][order],
                                                              endpoint - cycle, atol=1e-4):
            raise ValueError(f'{key}: RUL endpoint/target mismatch')
        cells[str(key)] = {'held_out': str(rows['held_out'][order[0]]),
                           'cycle': cycle, 'target': rows['target'][order],
                           'estimate': rows['estimate'][order],
                           'endpoint': int(endpoint[0])}
    cell_scores = {key: {mode: score_cell(cell['cycle'], cell['target'],
                                          cell['estimate'], cell['endpoint'], mode)
                         for mode in MODES}
                   for key, cell in cells.items()}
    groups = sorted({cell['held_out'] for cell in cells.values()})
    summary = {}
    for mode in MODES:
        group_mae = [np.mean([cell_scores[key][mode]['mae_cycles'] for key, cell in cells.items()
                              if cell['held_out'] == group]) for group in groups]
        values = [cell_scores[key][mode] for key in cells]
        early = [value['early_mae_cycles'] for value in values
                 if value['early_mae_cycles'] is not None]
        jumps = [value['median_endpoint_jump_per_observation'] for value in values
                 if value['median_endpoint_jump_per_observation'] is not None]
        summary[mode] = {
            'equal_fold_macro_mae_cycles': float(np.mean(group_mae)),
            'equal_cell_macro_mae_cycles': float(np.mean([v['mae_cycles'] for v in values])),
            'worst_cell_mae_cycles': float(np.max([v['mae_cycles'] for v in values])),
            'early_phase_cell_macro_mae_cycles': float(np.mean(early)) if early else None,
            'median_cell_median_endpoint_jump_cycles': float(np.median(jumps)) if jumps else None,
            'folds_improved_vs_raw': int(sum(group_mae[i] < np.mean([
                cell_scores[key]['raw']['mae_cycles'] for key, cell in cells.items()
                if cell['held_out'] == group]) for i, group in enumerate(groups))),
        }
    return {'cells': len(cells), 'groups': groups, 'summary': summary,
            'by_cell': {key: {'held_out': cells[key]['held_out'], **cell_scores[key]}
                        for key in cells}}


def main():
    domains = ('NASA', 'CALCE', 'MIT', 'XJTU')
    report = {'status': 'development diagnostic only; no frozen inference change',
              'method': 'causal median of predicted endpoint = cycle + RUL',
              'domains': {domain: evaluate_domain(domain) for domain in domains}}
    OUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    for domain in domains:
        print(domain, {mode: round(values['equal_fold_macro_mae_cycles'], 2)
                       for mode, values in report['domains'][domain]['summary'].items()},
              flush=True)
    print('Wrote', OUT)


if __name__ == '__main__':
    main()
