"""Matched MIT Batch-2 exact80 full-curve audit of cycle-100 EOL routes.

Power shape is fitted on the two training batches. Current SOH is a held-out
300 s signal estimate. Source EOL is predicted from observed 750 s prefixes
or complete discharges; true exact80 SOH appears only in scoring.
"""

import json
from pathlib import Path

import numpy as np

from battery_v2_features import load_cell, load_manifest
from evaluate_xjtu_trained_knee_v2 import fit_power, power_curve
from train_soh_v2 import outer_splits
from train_trajectory_v2 import threshold_soh


ROOT = Path('battery_models_v2/mit_qv_challenger')
EXACT = Path('battery_models_v2/exact80_target_audit.json')
SOH = Path('battery_models_v2/paper_inspired_soh_challenger/mit_development_oof_predictions.npz')
ROUTES = {
    'prefix_750s_two_stat': ROOT / 'prefix_750s_qv_cv.json',
    'prefix_750s_multicycle': ROOT / 'prefix_750s_multicycle_cv.json',
    'prefix_750s_censored': ROOT / 'prefix_750s_censored_cv.json',
    'full_discharge_paper': ROOT / 'paper_discharge_cv.json',
}
OUT = ROOT / 'prefix_750s_multicycle_curve_audit.json'
ORIGIN = 100
TARGET_SOH = .8


def power_exact80_crossing(current, source_threshold, origin, source_endpoint,
                           power):
    if current <= TARGET_SOH:
        return float(origin)
    drop = max(0., current-source_threshold)
    if drop <= 0 or source_endpoint <= origin:
        return None
    required_fade = (current-TARGET_SOH)/drop
    fraction = (required_fade**(1/power) if required_fade <= 1
                else required_fade)
    return float(origin+(source_endpoint-origin)*fraction)


def route_predictions(path, name):
    report = json.loads(path.read_text(encoding='utf-8'))
    by_id = {}
    for fold in report['folds']:
        for row in fold['by_cell']:
            predicted = (row['predicted_endpoint_cycle']['plus_capacity_all']
                         if name == 'full_discharge_paper'
                         else row['predicted_source_endpoint_cycle'])
            by_id[row['cell_id']] = float(predicted)
    return by_id


def main():
    records = [row for row in load_manifest()['cells'] if row['dataset'] == 'MIT']
    cells = {row['cell_id']: load_cell(row) for row in records}
    events = {row['cell_id']: row['first_confirmed_exact80_cycle']
              for row in json.loads(EXACT.read_text(encoding='utf-8'))[
                  'domains']['MIT']['by_cell']}
    predictions = {name: route_predictions(path, name)
                   for name, path in ROUTES.items()}
    with np.load(SOH) as source:
        current_oof = {(str(key), int(cycle)): float(value)
                       for key, cycle, value in zip(source['cell_id'],
                                                     source['cycle'], source['gated'])}
    rows = []
    powers = {}
    for held, train, validation in outer_splits(records, 'MIT'):
        power, trained = fit_power(cells, [key for key in train
                                          if cells[key].endpoint_cycle is not None])
        powers[held] = {'power': power, 'training_cells': trained}
        for key in validation:
            event = events[key]
            if event is None or event <= ORIGIN:
                continue
            cell = cells[key]
            i = np.flatnonzero(cell.cycle <= ORIGIN)
            if not len(i):
                continue
            origin = int(cell.cycle[i[-1]])
            current = current_oof[key, origin]
            future = np.flatnonzero((cell.cycle > origin) &
                                    (cell.cycle <= event) & cell.soh_available)
            if len(future) < 2:
                continue
            truth = cell.soh[future]
            horizons = cell.cycle[future]-origin
            scored = {}
            for name, by_id in predictions.items():
                eol = by_id[key]
                curve = power_curve(current, threshold_soh(cell), eol-origin,
                                    horizons, power)
                exact80_prediction = power_exact80_crossing(
                    current, threshold_soh(cell), origin, eol, power)
                scored[name] = {
                    'predicted_source_endpoint_cycle': eol,
                    'curve_implied_exact80_cycle': exact80_prediction,
                    'curve_mape_percent': float(100*np.mean(np.abs(curve-truth)/truth)),
                    'exact80_remaining_rul_ape_percent': float(
                        100*abs(exact80_prediction-event)/(event-origin))
                    if exact80_prediction is not None else None,
                }
            rows.append({'cell_id': key, 'held_out_batch': held,
                         'origin_cycle': origin, 'confirmed_exact80_cycle': event,
                         'future_points': len(future), 'routes': scored})
    report = {
        'status': 'unpromoted MIT exact80 full-curve comparison on observed-event subset',
        'input_warning': 'full_discharge_paper requires complete discharges; prefix routes use 750 s',
        'validation_warning': 'exact80 events only in Batch 2; shape and model choices reused development folds',
        'current_soh': 'held-out 300 s time-gated estimate',
        'curve_shape': 'power fitted on other batches; linear after predicted source crossing',
        'scored_cells': len(rows),
        'scored_batches': sorted({row['held_out_batch'] for row in rows}),
        'powers': powers,
        'summary': {name: {
            'curve_mape_percent': float(np.mean([
                row['routes'][name]['curve_mape_percent'] for row in rows])),
            'exact80_remaining_rul_mape_percent': float(np.mean([
                row['routes'][name]['exact80_remaining_rul_ape_percent']
                for row in rows])),
        } for name in ROUTES},
        'by_cell': rows,
    }
    OUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('scored', len(rows), 'batches', report['scored_batches'])
    print('SUMMARY', report['summary'])
    print('WROTE', OUT)


if __name__ == '__main__':
    main()
