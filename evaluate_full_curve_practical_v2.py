"""Practical full-remaining-curve audit from one held-out early-life origin.

Uses saved whole-cell/batch out-of-fold SOH, RUL, and local-rate predictions.
The true endpoint chooses retrospective 20%/50% checkpoints for evaluation
only; it is never passed to a predictor. Scores every available future SOH
measurement up to the observed endpoint, rather than only 50/100 cycles.
MIT and XJTU use their source-declared endpoint conventions; they are not
exactly a per-cell 80%-of-reference SOH threshold.
"""

import json
from pathlib import Path

import numpy as np

from battery_v2_features import load_all
from evaluate_anchored_trajectory_v2 import anchored_soh, get_domain_arrays
from train_trajectory_v2 import threshold_soh


OUT = Path('battery_models_v2/full_curve_practical_audit.json')
CHECKPOINTS = ('first_available', .2, .5)
METHODS = ('flat', 'local_linear', 'half_local_rate', 'bounded_half_local_rate',
           'direct_linear', 'anchored', 'fixed_half')


def project(current_soh, threshold, rul, local_rul, horizons):
    """Capacity-free forecasts anchored at one observed origin."""
    d = max(0, float(current_soh)-float(threshold))
    r = max(float(rul), 1e-6)
    local_rate = 0. if local_rul >= 2500 else d/max(float(local_rul), 1e-6)
    flat = np.full(len(horizons), current_soh, dtype='float64')
    local = current_soh-local_rate*horizons
    direct = current_soh-d*horizons/r
    anchored = anchored_soh(current_soh, threshold, rul, local_rul, horizons)
    half_local = .5*flat+.5*local
    return {'flat': flat, 'local_linear': local,
            'half_local_rate': half_local,
            'bounded_half_local_rate': np.clip(half_local, 0, current_soh),
            'direct_linear': direct,
            'anchored': anchored, 'fixed_half': .5*local+.5*direct}


def score_cell(cell, soh_by_cell, rul_by_cell, fraction):
    if cell.endpoint_cycle is None:
        return None
    soh_cycle, soh_estimate = soh_by_cell
    rul_cycle, selected_rul, local_rul = rul_by_cell
    if fraction == 'first_available':
        origin_position = 0
    else:
        candidate = np.flatnonzero(rul_cycle <= fraction*cell.endpoint_cycle)
        if not len(candidate):
            return None
        origin_position = int(candidate[-1])
    origin_cycle = int(rul_cycle[origin_position])
    origin_index = int(np.searchsorted(cell.cycle, origin_cycle))
    if origin_index >= len(cell.cycle) or cell.cycle[origin_index] != origin_cycle:
        raise ValueError(f'{cell.cell_id}: origin cycle mismatch')
    soh_position = int(np.searchsorted(soh_cycle, origin_cycle))
    if soh_position >= len(soh_cycle) or soh_cycle[soh_position] != origin_cycle:
        raise ValueError(f'{cell.cell_id}: OOF SOH cycle mismatch')
    future = np.flatnonzero((cell.cycle > origin_cycle) &
                            (cell.cycle <= cell.endpoint_cycle) & cell.soh_available)
    if len(future) < 2:
        return None
    horizon = cell.cycle[future]-origin_cycle
    truth = cell.soh[future]
    predicted = project(soh_estimate[soh_position], threshold_soh(cell),
                        selected_rul[origin_position], local_rul[origin_position], horizon)
    metrics = {}
    for name, curve in predicted.items():
        if not np.all(np.isfinite(curve)):
            raise ValueError(f'{cell.cell_id}: invalid curve from {name}')
        metrics[name] = {
            'curve_mape_percent': float(100*np.mean(np.abs(curve-truth)/truth)),
            'curve_mae_soh': float(np.mean(np.abs(curve-truth))),
            'last_point_error_soh': float(curve[-1]-truth[-1]),
            'upward_steps': int(np.count_nonzero(np.diff(curve) > 1e-9)),
            'below_zero_points': int(np.count_nonzero(curve < 0)),
            'above_origin_points': int(np.count_nonzero(curve >
                                                         soh_estimate[soh_position]+1e-9))}
    threshold = threshold_soh(cell)
    current = float(soh_estimate[soh_position])
    remaining = max(0, current-threshold)
    predicted_eol = origin_cycle + float(selected_rul[origin_position])
    local_remaining = float(local_rul[origin_position])
    direct_remaining = float(selected_rul[origin_position])
    local_rate = (remaining/local_remaining if 0 < local_remaining < 2500
                  else 0.)
    direct_rate = (remaining/max(direct_remaining, 1e-6) if remaining > 0 else 0.)
    local_eol = origin_cycle + remaining/local_rate if remaining > 0 and local_rate > 0 else None
    half_local_eol = (origin_cycle + remaining/(.5*local_rate)
                      if remaining > 0 and local_rate > 0 else None)
    fixed_half_rate = .5*(local_rate+direct_rate)
    fixed_half_eol = (origin_cycle + remaining/fixed_half_rate
                      if remaining > 0 and fixed_half_rate > 0 else None)
    def endpoint_error(predicted):
        return (float(100*abs(predicted-cell.endpoint_cycle)/cell.endpoint_cycle)
                if predicted is not None else None)
    return {'cell_id': cell.cell_id, 'group': cell.group,
            'origin_cycle': origin_cycle, 'actual_eol_cycle': cell.endpoint_cycle,
            'future_points': len(future),
            'selected_rul_eol_ape_percent': endpoint_error(predicted_eol),
            'local_eol_ape_percent': endpoint_error(local_eol),
            'half_local_eol_ape_percent': endpoint_error(half_local_eol),
            'fixed_half_eol_ape_percent': endpoint_error(fixed_half_eol),
            'metrics': metrics}


def evaluate():
    _, cells = load_all()
    domains = {}
    for domain in ('NASA', 'CALCE', 'MIT', 'XJTU'):
        soh_by_cell, rul_by_cell = get_domain_arrays(domain)
        domain_cells = [cells[key] for key in rul_by_cell]
        checkpoint_reports = {}
        for fraction in CHECKPOINTS:
            rows = [score_cell(cell, soh_by_cell[cell.cell_id],
                               rul_by_cell[cell.cell_id], fraction)
                    for cell in domain_cells]
            rows = [row for row in rows if row is not None]
            groups = sorted({row['group'] for row in rows})
            group_macro = {name: float(np.mean([
                np.mean([row['metrics'][name]['curve_mape_percent'] for row in rows
                         if row['group'] == group]) for group in groups]))
                for name in METHODS} if groups else {}
            checkpoint_reports[str(fraction)] = {
                'eligible_cells': len(rows), 'endpoint_cells': len(domain_cells),
                'groups_with_coverage': len(groups),
                'equal_cell_curve_mape_percent': {name: float(np.mean([
                    row['metrics'][name]['curve_mape_percent'] for row in rows]))
                    for name in METHODS} if rows else {},
                'equal_group_curve_mape_percent': group_macro,
                'selected_rul_eol_mape_percent': float(np.mean([
                    row['selected_rul_eol_ape_percent'] for row in rows])) if rows else None,
                'curve_derived_eol': {method: {
                    'scored_cells': sum(row[field] is not None for row in rows),
                    'mape_percent': float(np.mean([row[field] for row in rows
                                                  if row[field] is not None]))
                    if any(row[field] is not None for row in rows) else None}
                    for method, field in (
                        ('local_linear', 'local_eol_ape_percent'),
                        ('half_local_rate', 'half_local_eol_ape_percent'),
                        ('bounded_half_local_rate', 'half_local_eol_ape_percent'),
                        ('fixed_half', 'fixed_half_eol_ape_percent'),
                        ('direct_linear', 'selected_rul_eol_ape_percent'),
                        ('anchored', 'selected_rul_eol_ape_percent'))},
                'by_cell': rows}
        domains[domain] = checkpoint_reports
        print(domain, {fraction: {'cells': result['eligible_cells'],
                                  'curve_mape': {key: round(value, 2) for key, value in
                                                 result['equal_cell_curve_mape_percent'].items()}}
                       for fraction, result in checkpoint_reports.items()}, flush=True)
    return {'status': 'development held-out full-remaining-curve audit; no model promotion',
            'input': 'saved OOF SOH, RUL, and local-rate predictions only',
            'future_assumption': 'current degradation regime persists',
            'checkpoint_rule': 'first eligible observation or last eligible observation '
                               'at or before 20%/50% of actual endpoint; fractions chosen '
                               'retrospectively for evaluation only',
            'domains': domains}


if __name__ == '__main__':
    report = evaluate()
    OUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('WROTE', OUT)
