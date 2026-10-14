"""Describe held-out XJTU SOH errors and selected diagnostic-signal changes.

This is a post-hoc development audit, not model selection or a new test set.
Capacity is read only to score and describe targets; no model is fitted.
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from battery_v2_features import load_all
from prepare_xjtu import FEATURE_NAMES


OOF = Path('battery_models_v2/soh/xjtu/development_oof_predictions.npz')
OUT = Path('battery_models_v2/xjtu_soh_residual_audit.json')
SIGNALS = ('charge_voltage_V_120s', 'discharge_voltage_V_120s',
           'discharge_voltage_drop_v', 'discharge_current_mean_a')


def median(values):
    return float(np.median(values))


def audit():
    manifest, cells = load_all()
    records = {row['cell_id']: row for row in manifest['cells'] if row['dataset'] == 'XJTU'}
    with np.load(OOF) as data:
        oof = {name: data[name].copy() for name in data.files}
    by_cell = []
    for key, record in records.items():
        cell = cells[key]
        select = oof['cell_id'] == key
        observed_cycle = oof['cycle'][select]
        expected_cycle = cell.cycle[cell.soh_available]
        if not np.array_equal(observed_cycle, expected_cycle):
            raise ValueError(f'{key}: OOF cycle alignment mismatch')
        truth = oof['target'][select]
        prediction = oof['estimate'][select]
        if not np.allclose(truth, cell.soh[cell.soh_available]):
            raise ValueError(f'{key}: OOF SOH target alignment mismatch')
        split = np.unique(oof['held_out'][select])
        if len(split) != 1 or split[0] != record['group']:
            raise ValueError(f'{key}: OOF group alignment mismatch')
        n = len(truth)
        third = max(1, n // 3)
        signal = {}
        for name in SIGNALS:
            values = cell.raw[:, FEATURE_NAMES.index(name)].astype('float64')
            initial = median(values[:cell.reference_count])
            final = median(values[-min(5, len(values)):])
            rho = spearmanr(values[cell.soh_available], truth).statistic
            signal[name] = {
                'first_reference_median': initial,
                'last_five_median': final,
                'last_minus_reference': final - initial,
                'spearman_with_soh': None if not np.isfinite(rho) else float(rho),
            }
        by_cell.append({
            'cell_id': key, 'group': record['group'],
            'diagnostics': len(cell.cycle),
            'median_source_cycle_gap': median(np.diff(cell.cycle)),
            'first_scored_soh': float(truth[0]),
            'last_scored_soh': float(truth[-1]),
            'last_predicted_soh': float(prediction[-1]),
            'first_third_bias': float(np.mean(prediction[:third] - truth[:third])),
            'last_third_bias': float(np.mean(prediction[-third:] - truth[-third:])),
            'mae': float(np.mean(np.abs(prediction - truth))),
            'signals': signal,
        })
    groups = {}
    for group in sorted({row['group'] for row in by_cell}):
        rows = [row for row in by_cell if row['group'] == group]
        groups[group] = {
            'cells': len(rows),
            'median_diagnostics': median([row['diagnostics'] for row in rows]),
            'median_source_cycle_gap': median([row['median_source_cycle_gap'] for row in rows]),
            'median_first_scored_soh': median([row['first_scored_soh'] for row in rows]),
            'median_last_scored_soh': median([row['last_scored_soh'] for row in rows]),
            'median_last_predicted_soh': median([row['last_predicted_soh'] for row in rows]),
            'cell_macro_first_third_bias': float(np.mean([row['first_third_bias'] for row in rows])),
            'cell_macro_last_third_bias': float(np.mean([row['last_third_bias'] for row in rows])),
            'cell_macro_mae': float(np.mean([row['mae'] for row in rows])),
            'signals': {name: {
                'median_last_minus_reference': median(
                    [row['signals'][name]['last_minus_reference'] for row in rows]),
                'median_within_cell_spearman_soh': median(
                    [row['signals'][name]['spearman_with_soh'] for row in rows
                     if row['signals'][name]['spearman_with_soh'] is not None]),
            } for name in SIGNALS},
        }
    return {
        'status': 'post-hoc development diagnostic; no model or label changes',
        'note': 'first/last thirds are by diagnostic observation count, not equal cycle durations',
        'groups': groups, 'cells': by_cell,
    }


if __name__ == '__main__':
    result = audit()
    OUT.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('WROTE', OUT)
    for group, row in result['groups'].items():
        print(group, 'n', row['median_diagnostics'], 'gap', row['median_source_cycle_gap'],
              'MAE', round(row['cell_macro_mae'], 4),
              'bias early/late', round(row['cell_macro_first_third_bias'], 4),
              round(row['cell_macro_last_third_bias'], 4))
