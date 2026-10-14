"""Supplementary within-strategy XJTU battery holdout diagnostic.

Every validation battery is wholly held out, while other batteries under its
known operating strategy remain in training. This tests coverage of known
strategies only; it MUST NOT replace whole-strategy holdout results. Choices
of features and blend rule were informed by prior development on these cells.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from battery_v2_features import load_cell, load_manifest
from evaluate_paper_inspired_rul_v2 import fit as fit_direct, predict as predict_direct
from evaluate_xjtu_age_blend_v2 import blend_endpoints
from evaluate_xjtu_combined_rul_v2 import matrix
from evaluate_xjtu_fractional_life_v2 import fit_fraction, predict_endpoint
from evaluate_xjtu_throughput_rul_v2 import LONG_ROOT, USAGE, throughput_matrix
from evaluate_xjtu_trained_knee_v2 import fit_power, power_curve
from train_soh_v2 import fit as fit_soh, predict as predict_soh
from train_trajectory_v2 import threshold_soh


OUT = Path('battery_models_v2/xjtu_seen_strategy_diagnostic.json')
FOLDS = 4


def assignments(records):
    by_group = defaultdict(list)
    for row in records:
        by_group[row['group']].append(row['cell_id'])
    return {key: index % FOLDS for group in sorted(by_group)
            for index, key in enumerate(sorted(by_group[group]))}


def main():
    records = [row for row in load_manifest()['cells'] if row['dataset'] == 'XJTU']
    cells = {row['cell_id']: load_cell(row) for row in records}
    folds_for = assignments(records)
    features = {}
    with np.load(LONG_ROOT / 'features.npz') as long_file, np.load(USAGE) as usage_file:
        for key, cell in cells.items():
            if not (np.array_equal(long_file[f'{key}_cycle'], cell.cycle) and
                    np.array_equal(usage_file[f'{key}_cycle'], cell.cycle)):
                raise ValueError(f'{key}: prefix/usage cycle mismatch')
            raw = throughput_matrix(long_file[f'{key}_600s'])
            features[key] = matrix(cell, raw, usage_file[f'{key}_usage'])
    folds = []
    for fold_number in range(FOLDS):
        train = [key for key in cells if folds_for[key] != fold_number]
        validation = [key for key in cells if folds_for[key] == fold_number]
        soh_model = fit_soh(cells, train, 'extra_trees_history')
        direct_model = fit_direct(cells, train, features, 'gated_raw')
        fraction_model = fit_fraction(cells, train, features)
        power, fitted_curves = fit_power(cells, train)
        rows = []
        for key in validation:
            cell = cells[key]
            soh = predict_soh(soh_model, cell)
            keep = cell.soh_available
            soh_mape = float(100*np.mean(abs(soh[keep]-cell.soh[keep])/
                                          cell.soh[keep]))
            origin = max(15, cell.reference_count)
            result = {'cell_id': key, 'group': cell.group,
                      'current_soh_mape_percent': soh_mape,
                      'direct_origin_eol_ape_percent': None,
                      'fraction_origin_eol_ape_percent': None,
                      'origin_eol_ape_percent': None,
                      'direct_future_curve_mape_percent': None,
                      'fraction_future_curve_mape_percent': None,
                      'future_curve_mape_percent': None}
            if origin < len(cell.cycle) and cell.cycle[origin] < cell.endpoint_cycle:
                direct_rul = predict_direct(direct_model, features[key], 'gated_raw')
                fraction_eol = predict_endpoint(fraction_model, cell, features[key])
                blended = blend_endpoints(cell.cycle, direct_rul, fraction_eol)
                cycle = int(cell.cycle[origin])
                result['origin_cycle'] = cycle
                result['direct_origin_eol_ape_percent'] = float(100*abs(
                    cycle+direct_rul[origin]-cell.endpoint_cycle)/cell.endpoint_cycle)
                result['fraction_origin_eol_ape_percent'] = float(100*abs(
                    fraction_eol[origin]-cell.endpoint_cycle)/cell.endpoint_cycle)
                result['origin_eol_ape_percent'] = float(100*abs(
                    blended[origin]-cell.endpoint_cycle)/cell.endpoint_cycle)
                future = np.flatnonzero((cell.cycle > cycle) &
                                        (cell.cycle <= cell.endpoint_cycle) &
                                        cell.soh_available)
                if len(future) >= 2:
                    for name, predicted_endpoint in (
                            ('direct', cycle+direct_rul[origin]),
                            ('fraction', fraction_eol[origin]),
                            ('blend', blended[origin])):
                        curve = power_curve(soh[origin], threshold_soh(cell),
                                            predicted_endpoint-cycle,
                                            cell.cycle[future]-cycle, power)
                        field = ('future_curve_mape_percent' if name == 'blend' else
                                 f'{name}_future_curve_mape_percent')
                        result[field] = float(100*np.mean(
                            abs(curve-cell.soh[future])/cell.soh[future]))
            rows.append(result)
        folds.append({'fold': fold_number, 'training_cells': len(train),
                      'validation_cells': len(validation), 'fitted_power': power,
                      'power_training_curves': fitted_curves, 'by_cell': rows})
        print('fold', fold_number, 'current SOH', round(float(np.mean([
            row['current_soh_mape_percent'] for row in rows])), 2),
              'EOL', round(float(np.mean([row['origin_eol_ape_percent'] for row in rows
                                         if row['origin_eol_ape_percent'] is not None])), 2),
              'curve', round(float(np.mean([row['future_curve_mape_percent'] for row in rows
                                           if row['future_curve_mape_percent'] is not None])), 2),
              flush=True)
    rows = [row for fold in folds for row in fold['by_cell']]
    def aggregate(field):
        values = [row[field] for row in rows if row[field] is not None]
        return {'scored_cells': len(values), 'cell_macro_mape_percent': float(np.mean(values))}
    report = {'status': 'supplementary known-strategy whole-cell CV; not independent test',
              'warning': 'does not verify transfer to a wholly unseen XJTU strategy; '
                         'feature and blend choices already informed by development data',
              'input': '120s SOH diagnostic plus 600s RUL diagnostic and prior usage',
              'endpoint': 'source XJTU EOL, not exact first per-cell 80% SOH crossing',
              'summary': {'current_soh': aggregate('current_soh_mape_percent'),
                          'direct_first_origin_eol': aggregate(
                              'direct_origin_eol_ape_percent'),
                          'fraction_first_origin_eol': aggregate(
                              'fraction_origin_eol_ape_percent'),
                          'first_origin_eol': aggregate('origin_eol_ape_percent'),
                          'direct_full_future_curve': aggregate(
                              'direct_future_curve_mape_percent'),
                          'fraction_full_future_curve': aggregate(
                              'fraction_future_curve_mape_percent'),
                          'full_future_curve': aggregate('future_curve_mape_percent')},
              'folds': folds}
    OUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('SUMMARY', report['summary'])
    print('WROTE', OUT)


if __name__ == '__main__':
    main()
