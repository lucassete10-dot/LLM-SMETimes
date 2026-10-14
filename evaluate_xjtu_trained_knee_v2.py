"""Train-only XJTU pre-EOL knee shape atop the age-blended RUL route.

One curvature coefficient is fitted from measured trajectories of the outer
training strategies. Held-out SOH and RUL remain the existing whole-strategy
OOF predictions. The curve crosses the same predicted endpoint and reduces
to the prior linear route after that endpoint, avoiding cubic runaway.
"""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from audit_xjtu_blended_joint_curve_v2 import SOH, coherent_curve
from battery_v2_features import load_cell, load_manifest
from evaluate_xjtu_age_blend_v2 import (DIRECT, FRACTION, blend_endpoints,
                                         read_oof)
from train_soh_v2 import outer_splits
from train_trajectory_v2 import threshold_soh


OUT = Path('battery_models_v2/xjtu_trained_knee_challenger/development_cv.json')


def knee_curve(current, threshold, remaining, horizons, curvature):
    if not 0 <= curvature <= 1:
        raise ValueError('curvature must lie in [0, 1]')
    current = float(current)
    drop = max(0., current-float(threshold))
    x = np.asarray(horizons, dtype=float)/max(float(remaining), 1e-6)
    baseline = current-drop*x
    bend = drop*curvature*np.clip(x, 0, 1)*np.maximum(0, 1-x)
    return np.clip(baseline+bend, 0, current)


def power_curve(current, threshold, remaining, horizons, power):
    if not 1 <= power <= 4:
        raise ValueError('power must lie in [1, 4]')
    current = float(current)
    drop = max(0., current-float(threshold))
    x = np.asarray(horizons, dtype=float)/max(float(remaining), 1e-6)
    fade = np.where(x <= 1, x**power, x)
    return np.clip(current-drop*fade, 0, current)


def fit_power(cells, train_ids):
    trajectories = []
    for key in train_ids:
        cell = cells[key]
        origin = max(15, cell.reference_count)
        if origin >= len(cell.cycle) or cell.endpoint_cycle <= cell.cycle[origin]:
            continue
        if not cell.soh_available[origin]:
            continue
        future = np.flatnonzero((cell.cycle > cell.cycle[origin]) &
                                (cell.cycle <= cell.endpoint_cycle) &
                                cell.soh_available)
        if len(future) < 2:
            continue
        current = float(cell.soh[origin])
        drop = max(0., current-threshold_soh(cell))
        x = (cell.cycle[future]-cell.cycle[origin]) / (
            cell.endpoint_cycle-cell.cycle[origin])
        trajectories.append((current, drop, x, cell.soh[future]))
    if len(trajectories) < 3:
        raise ValueError('insufficient training trajectories for power fit')
    def loss(power):
        return float(np.mean([np.mean((current-drop*x**power-truth)**2)
                              for current, drop, x, truth in trajectories]))
    result = minimize_scalar(loss, bounds=(1, 4), method='bounded',
                             options={'xatol': .01})
    if not result.success:
        raise RuntimeError('training-fold power fit failed')
    return float(result.x), len(trajectories)


def fit_curvature(cells, train_ids):
    numerators, denominators = [], []
    for key in train_ids:
        cell = cells[key]
        origin = max(15, cell.reference_count)
        if origin >= len(cell.cycle) or cell.endpoint_cycle <= cell.cycle[origin]:
            continue
        if not cell.soh_available[origin]:
            continue
        future = np.flatnonzero((cell.cycle > cell.cycle[origin]) &
                                (cell.cycle <= cell.endpoint_cycle) &
                                cell.soh_available)
        if len(future) < 2:
            continue
        current = float(cell.soh[origin])
        drop = max(0., current-threshold_soh(cell))
        x = (cell.cycle[future]-cell.cycle[origin]) / (
            cell.endpoint_cycle-cell.cycle[origin])
        linear = current-drop*x
        bend_unit = drop*x*(1-x)
        numerators.append(float(np.mean(bend_unit*(cell.soh[future]-linear))))
        denominators.append(float(np.mean(bend_unit**2)))
    if len(numerators) < 3 or sum(denominators) < 1e-12:
        raise ValueError('insufficient training trajectories for knee fit')
    return float(np.clip(sum(numerators)/sum(denominators), 0, 1)), len(numerators)


def main():
    records = [row for row in load_manifest()['cells'] if row['dataset'] == 'XJTU']
    cells = {row['cell_id']: load_cell(row) for row in records}
    direct = read_oof(DIRECT, 'estimate')
    fraction = read_oof(FRACTION, 'endpoint')
    with np.load(SOH) as source:
        soh = {(str(key), int(cycle)): float(value)
               for key, cycle, value in zip(source['cell_id'], source['cycle'],
                                             source['fusion'])}
    folds = []
    for held, train, validation in outer_splits(records, 'XJTU'):
        curvature, training_cells = fit_curvature(cells, train)
        power, power_training_cells = fit_power(cells, train)
        if training_cells != power_training_cells:
            raise ValueError('shape fits use different training cohorts')
        scored = []
        for key in validation:
            cell = cells[key]
            origin = max(15, cell.reference_count)
            if origin >= len(cell.cycle):
                continue
            cycle = int(cell.cycle[origin])
            if cycle >= cell.endpoint_cycle or (key, cycle) not in soh:
                continue
            future = np.flatnonzero((cell.cycle > cycle) &
                                    (cell.cycle <= cell.endpoint_cycle) &
                                    cell.soh_available)
            if len(future) < 2:
                continue
            direct_rul = direct[key, cycle]
            fraction_eol = fraction[key, cycle]
            blended_eol = float(blend_endpoints(np.array([cycle]),
                                 np.array([direct_rul]),
                                 np.array([fraction_eol]))[0])
            horizon = cell.cycle[future]-cycle
            current = soh[key, cycle]
            threshold = threshold_soh(cell)
            linear = coherent_curve(current, threshold, blended_eol-cycle, horizon)
            curved = knee_curve(current, threshold, blended_eol-cycle, horizon,
                                curvature)
            powered = power_curve(current, threshold, blended_eol-cycle, horizon,
                                  power)
            direct_powered = power_curve(current, threshold, direct_rul,
                                         horizon, power)
            endpoint_oracle = power_curve(current, threshold,
                                          cell.endpoint_cycle-cycle, horizon, power)
            current_oracle = power_curve(cell.soh[origin], threshold,
                                         blended_eol-cycle, horizon, power)
            both_oracle = power_curve(cell.soh[origin], threshold,
                                      cell.endpoint_cycle-cycle, horizon, power)
            truth = cell.soh[future]
            scored.append({'cell_id': key, 'group': cell.group,
                           'origin_cycle': cycle, 'future_points': len(future),
                           'linear_curve_mape_percent': float(100*np.mean(
                               abs(linear-truth)/truth)),
                           'trained_knee_curve_mape_percent': float(100*np.mean(
                               abs(curved-truth)/truth)),
                           'trained_power_curve_mape_percent': float(100*np.mean(
                               abs(powered-truth)/truth)),
                           'direct_power_curve_mape_percent': float(100*np.mean(
                               abs(direct_powered-truth)/truth)),
                           'endpoint_oracle_curve_mape_percent': float(100*np.mean(
                               abs(endpoint_oracle-truth)/truth)),
                           'current_oracle_curve_mape_percent': float(100*np.mean(
                               abs(current_oracle-truth)/truth)),
                           'both_oracle_curve_mape_percent': float(100*np.mean(
                               abs(both_oracle-truth)/truth)),
                           'eol_ape_percent': float(100*abs(
                               blended_eol-cell.endpoint_cycle)/cell.endpoint_cycle),
                           'direct_eol_ape_percent': float(100*abs(
                               cycle+direct_rul-cell.endpoint_cycle)/cell.endpoint_cycle),
                           'curvature': curvature, 'power': power})
        folds.append({'held_out': held, 'training_cells_with_trajectory': training_cells,
                      'curvature': curvature, 'power': power,
                      'scored_cells': len(scored),
                      'linear_mape_percent': float(np.mean([
                          row['linear_curve_mape_percent'] for row in scored])),
                      'trained_knee_mape_percent': float(np.mean([
                          row['trained_knee_curve_mape_percent'] for row in scored])),
                      'trained_power_mape_percent': float(np.mean([
                          row['trained_power_curve_mape_percent'] for row in scored])),
                      'by_cell': scored})
        print(held, 'k', round(curvature, 3), 'p', round(power, 2), 'linear',
              round(folds[-1]['linear_mape_percent'], 2), 'knee',
              round(folds[-1]['trained_knee_mape_percent'], 2), 'power',
              round(folds[-1]['trained_power_mape_percent'], 2), flush=True)
    rows = [row for fold in folds for row in fold['by_cell']]
    report = {'status': 'unpromoted whole-strategy development shape challenger',
              'method': 'one train-fold-fitted pre-EOL quadratic bend; linear after crossing',
              'input': 'held-out fusion SOH and age-blended source EOL',
              'endpoint_warning': 'source XJTU EOL, not exact per-cell 80% crossing',
              'summary': {'scored_cells': len(rows),
                          'linear_curve_mape_percent': float(np.mean([
                              row['linear_curve_mape_percent'] for row in rows])),
                          'trained_knee_curve_mape_percent': float(np.mean([
                              row['trained_knee_curve_mape_percent'] for row in rows])),
                          'trained_power_curve_mape_percent': float(np.mean([
                              row['trained_power_curve_mape_percent'] for row in rows])),
                          'direct_power_curve_mape_percent': float(np.mean([
                              row['direct_power_curve_mape_percent'] for row in rows])),
                          'endpoint_oracle_curve_mape_percent': float(np.mean([
                              row['endpoint_oracle_curve_mape_percent'] for row in rows])),
                          'current_oracle_curve_mape_percent': float(np.mean([
                              row['current_oracle_curve_mape_percent'] for row in rows])),
                          'both_oracle_curve_mape_percent': float(np.mean([
                              row['both_oracle_curve_mape_percent'] for row in rows])),
                          'eol_mape_percent': float(np.mean([
                              row['eol_ape_percent'] for row in rows])),
                          'direct_eol_mape_percent': float(np.mean([
                              row['direct_eol_ape_percent'] for row in rows]))},
              'folds': folds}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('SUMMARY', report['summary'])
    print('WROTE', OUT)


if __name__ == '__main__':
    main()
