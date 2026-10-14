"""OOF test of trajectories reconciled with the selected direct/assisted RUL.

No model is retrained or changed. This reconstructs a smooth path from the
held-out current SOH, held-out selected RUL, and held-out local-rate RUL. The
path crosses the domain endpoint threshold at the selected RUL, subject to the
same future-operating-strategy assumption. Compare 25/50/100-cycle SOH errors
before considering it for inference.
"""

import json
from pathlib import Path

import numpy as np

from battery_v2_features import load_all
from train_trajectory_v2 import future_pairs, threshold_soh


ROOT = Path('battery_models_v2')
HORIZONS = (25, 50, 100)


def anchored_soh(current_soh, threshold, selected_rul, local_rul, horizon):
    s = np.asarray(current_soh, dtype='float64')
    r = np.asarray(selected_rul, dtype='float64')
    l = np.asarray(local_rul, dtype='float64')
    h = np.asarray(horizon, dtype='float64')
    d = np.maximum(0, s-threshold)
    safe_r = np.maximum(r, 1e-6)
    local_rate = np.where(l >= 2500, 0., d/np.maximum(l, 1e-6))
    initial_rate = np.clip(local_rate, 0, 2*d/safe_r)
    at_or_before = s-initial_rate*h-(d-initial_rate*safe_r)*(h/safe_r)**2
    end_rate = np.maximum(0, 2*d/safe_r-initial_rate)
    after = threshold-end_rate*(h-safe_r)
    result = np.where(h <= safe_r, at_or_before, after)
    result = np.where((d <= 0) | (r <= 0), threshold, result)
    return np.maximum(0, result)


def get_domain_arrays(domain):
    folder = domain.lower()
    with np.load(ROOT / 'soh' / folder / 'development_oof_predictions.npz') as soh, \
         np.load(ROOT / 'rul' / folder / 'development_oof_predictions.npz') as selected, \
         np.load(ROOT / 'trajectory' / folder / 'development_oof_rul.npz') as trajectory:
        for field in ('target', 'held_out', 'cell_id', 'cycle', 'endpoint'):
            np.testing.assert_array_equal(selected[field], trajectory[field],
                                          err_msg=f'{domain}: RUL artifact misalignment')
        soh_by_cell = {}
        for key in np.unique(soh['cell_id']):
            keep = soh['cell_id'] == key
            soh_by_cell[str(key)] = (soh['cycle'][keep].copy(), soh['estimate'][keep].copy())
        rul_by_cell = {}
        for key in np.unique(selected['cell_id']):
            keep = selected['cell_id'] == key
            rul_by_cell[str(key)] = (selected['cycle'][keep].copy(),
                                     selected['estimate'][keep].copy(),
                                     trajectory['estimate'][keep].copy())
    return soh_by_cell, rul_by_cell


def score_domain(cells, domain):
    soh_by_cell, rul_by_cell = get_domain_arrays(domain)
    rows = []
    for key, (rul_cycle, selected_rul, local_rul) in rul_by_cell.items():
        cell = cells[key]
        soh_cycle, soh_estimate = soh_by_cell[key]
        soh_position = np.searchsorted(soh_cycle, rul_cycle)
        np.testing.assert_array_equal(soh_cycle[soh_position], rul_cycle)
        current_soh = soh_estimate[soh_position]
        threshold = threshold_soh(cell)
        for horizon in HORIZONS:
            current, future = future_pairs(cell, horizon)
            if not len(current):
                continue
            target_cycle = cell.cycle[current]
            position = np.searchsorted(rul_cycle, target_cycle)
            if np.any(position >= len(rul_cycle)) or not np.array_equal(rul_cycle[position], target_cycle):
                raise ValueError(f'{key}: missing held-out RUL at a forecast origin')
            delta = cell.cycle[future]-cell.cycle[current]
            true_soh = cell.soh[future]
            s = current_soh[position]
            r = selected_rul[position]
            l = local_rul[position]
            d = np.maximum(0, s-threshold)
            flat = s
            local = s-np.where(l >= 2500, 0., d/np.maximum(l, 1e-6))*delta
            linear = s-d*delta/np.maximum(r, 1e-6)
            anchored = anchored_soh(s, threshold, r, l, delta)
            rows.append({'cell_id': key, 'group': cell.group, 'horizon': horizon,
                         'observations': len(current),
                         'mae': {name: float(np.mean(np.abs(value-true_soh)))
                                 for name, value in (('flat', flat), ('local_rate', local),
                                                     ('direct_linear', linear),
                                                     ('anchored', anchored))}})
    summary = {}
    for horizon in HORIZONS:
        cohort = [row for row in rows if row['horizon'] == horizon]
        groups = sorted({row['group'] for row in cohort})
        per_group = {group: {method: float(np.mean([row['mae'][method] for row in cohort
                                                   if row['group'] == group]))
                             for method in ('flat', 'local_rate', 'direct_linear', 'anchored')}
                     for group in groups}
        summary[str(horizon)] = {'cells': len(cohort), 'groups': len(groups),
            'group_macro_mae': {method: float(np.mean([per_group[group][method]
                                                      for group in groups]))
                                for method in ('flat', 'local_rate', 'direct_linear', 'anchored')},
            'by_group': per_group}
    return {'domain': domain, 'horizons': summary, 'by_cell': rows}


def main():
    _, cells = load_all()
    report = {'data_contract_version': 2,
              'warning': 'development OOF diagnostic; no external validation or model replacement',
              'method': 'smooth trajectory anchored at selected RUL threshold crossing',
              'domains': {}}
    for domain in ('NASA', 'CALCE', 'MIT', 'XJTU'):
        result = score_domain(cells, domain)
        report['domains'][domain] = result
        print(domain, {horizon: {key: round(value, 4) for key, value in
                               row['group_macro_mae'].items()}
                       for horizon, row in result['horizons'].items()}, flush=True)
    (ROOT / 'anchored_trajectory_diagnostic.json').write_text(json.dumps(report, indent=2),
                                                               encoding='utf-8')


if __name__ == '__main__':
    main()
