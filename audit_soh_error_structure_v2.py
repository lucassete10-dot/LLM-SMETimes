"""Read-only SOH residual and target-volatility audit across four datasets.

Neighbor interpolation uses FUTURE LABELS and is diagnostic-only; it must
never be reported as a deployable model or a formal noise lower bound.
"""

import json
from pathlib import Path

import numpy as np

from battery_v2_features import load_cell, load_manifest


ROOT = Path('battery_models_v2')
OUT = ROOT / 'soh_error_structure_audit.json'
PREDICTIONS = {
    'NASA': (ROOT / 'nasa_boundary_soh_challenger/development_oof_predictions.npz',
             'estimate'),
    'CALCE': (ROOT / 'paper_inspired_soh_challenger/calce_development_oof_predictions.npz',
              'fusion'),
    'MIT': (ROOT / 'paper_inspired_soh_challenger/mit_development_oof_predictions.npz',
            'gated'),
    'XJTU': (ROOT / 'paper_inspired_soh_challenger/xjtu_development_oof_predictions.npz',
             'fusion'),
}


def local_interpolation_error(cycle, target):
    cycle, target = np.asarray(cycle), np.asarray(target)
    if len(cycle) < 3:
        return None
    gaps = np.diff(cycle)
    median_gap = float(np.median(gaps))
    valid = (gaps[:-1] <= 3*median_gap) & (gaps[1:] <= 3*median_gap)
    if not np.any(valid):
        return None
    fraction = (cycle[1:-1]-cycle[:-2])/(cycle[2:]-cycle[:-2])
    interpolation = target[:-2] + fraction*(target[2:]-target[:-2])
    error = 100*np.abs(interpolation-target[1:-1])/target[1:-1]
    return {'observations': int(np.sum(valid)),
            'mape_percent': float(np.mean(error[valid])),
            'fraction_exceeding_0_5_percent': float(np.mean(error[valid] > .5))}


def main():
    manifest = load_manifest()
    result = {'status': 'post-hoc development diagnostic; no model fitted',
              'neighbor_warning': 'interpolation uses preceding AND future measured SOH; not a deployable predictor or formal noise floor',
              'domains': {}}
    for domain, (source_path, field) in PREDICTIONS.items():
        with np.load(source_path) as source:
            oof = {(str(key), int(cycle)): float(value)
                   for key, cycle, value in zip(source['cell_id'], source['cycle'],
                                                source[field])}
        rows = []
        for record in manifest['cells']:
            if record['dataset'] != domain:
                continue
            cell = load_cell(record)
            keep = cell.soh_available
            cycle, target = cell.cycle[keep], cell.soh[keep]
            estimate = np.array([oof[cell.cell_id, int(number)] for number in cycle])
            if not np.all(np.isfinite(estimate)):
                raise ValueError(f'nonfinite selected held-out SOH: {cell.cell_id}')
            n = len(target)
            third = max(1, n//3)
            error = 100*np.abs(estimate-target)/target
            rows.append({'cell_id': cell.cell_id, 'group': cell.group,
                         'scored_observations': n,
                         'selected_model_mape_percent': float(np.mean(error)),
                         'selected_model_signed_bias_first_third_soh': float(np.mean(
                             estimate[:third]-target[:third])),
                         'selected_model_signed_bias_last_third_soh': float(np.mean(
                             estimate[-third:]-target[-third:])),
                         'selected_model_fraction_exceeding_0_5_percent': float(
                             np.mean(error > .5)),
                         'neighbor_interpolation': local_interpolation_error(cycle, target)})
        groups = {}
        for group in sorted({row['group'] for row in rows}):
            subset = [row for row in rows if row['group'] == group]
            groups[group] = {
                'cells': len(subset),
                'selected_model_cell_macro_mape_percent': float(np.mean([
                    row['selected_model_mape_percent'] for row in subset])),
                'selected_model_first_third_bias_soh': float(np.mean([
                    row['selected_model_signed_bias_first_third_soh'] for row in subset])),
                'selected_model_last_third_bias_soh': float(np.mean([
                    row['selected_model_signed_bias_last_third_soh'] for row in subset])),
                'neighbor_interpolation_cell_macro_mape_percent': float(np.mean([
                    row['neighbor_interpolation']['mape_percent'] for row in subset
                    if row['neighbor_interpolation'] is not None])),
            }
        result['domains'][domain] = {
            'selected_route': field,
            'cells': len(rows),
            'selected_model_cell_macro_mape_percent': float(np.mean([
                row['selected_model_mape_percent'] for row in rows])),
            'neighbor_interpolation_cell_macro_mape_percent': float(np.mean([
                row['neighbor_interpolation']['mape_percent'] for row in rows
                if row['neighbor_interpolation'] is not None])),
            'groups': groups, 'by_cell': rows}
        print(domain, 'model', round(result['domains'][domain][
              'selected_model_cell_macro_mape_percent'], 3), 'future-label neighbor',
              round(result['domains'][domain][
                  'neighbor_interpolation_cell_macro_mape_percent'], 3), flush=True)
    OUT.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('WROTE', OUT)


if __name__ == '__main__':
    main()
