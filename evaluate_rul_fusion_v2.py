"""Compare fixed direct/trajectory blends on aligned whole-group OOF rows."""

import json
from pathlib import Path

import numpy as np


ROOT = Path('battery_models_v2')
WEIGHTS = (0., .1, .25, .5, 1.)


def load_aligned(domain):
    direct_path = ROOT / 'rul' / domain.lower() / 'development_oof_predictions.npz'
    trajectory_path = ROOT / 'trajectory' / domain.lower() / 'development_oof_rul.npz'
    with np.load(direct_path) as a, np.load(trajectory_path) as b:
        for field in ('target', 'held_out', 'cell_id', 'cycle', 'endpoint'):
            np.testing.assert_array_equal(a[field], b[field], err_msg=f'{domain} {field} misaligned')
        return {key: a[key].copy() for key in ('target', 'held_out', 'cell_id', 'cycle')}, \
            a['estimate'].copy(), b['estimate'].copy()


def score(reference, estimate):
    target, splits, cells = reference['target'], reference['held_out'], reference['cell_id']
    folds = []
    for group in np.unique(splits):
        mask = splits == group
        by_cell = [float(np.mean(np.abs(estimate[mask & (cells == key)]-
                                        target[mask & (cells == key)])))
                   for key in np.unique(cells[mask])]
        folds.append({'held_out': str(group), 'cell_macro_mae_cycles': float(np.mean(by_cell)),
                      'worst_cell_mae_cycles': float(np.max(by_cell))})
    return {'group_macro_mae_cycles': float(np.mean([x['cell_macro_mae_cycles'] for x in folds])),
            'worst_group_mae_cycles': float(np.max([x['cell_macro_mae_cycles'] for x in folds])),
            'folds': folds}


def main():
    output = {'data_contract_version': 2,
              'warning': 'blend weights compared on development folds; no independent selection proof',
              'domains': {}}
    for domain in ('NASA', 'CALCE', 'MIT', 'XJTU'):
        reference, direct, trajectory = load_aligned(domain)
        metrics = {}
        for weight in WEIGHTS:
            candidate = (1-weight)*direct + weight*trajectory
            metrics[str(weight)] = score(reference, candidate)
        output['domains'][domain] = metrics
        print(domain, {weight: (round(value['group_macro_mae_cycles'], 1),
                                round(value['worst_group_mae_cycles'], 1))
                       for weight, value in metrics.items()}, flush=True)
    (ROOT / 'rul_fusion_diagnostic.json').write_text(json.dumps(output, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
