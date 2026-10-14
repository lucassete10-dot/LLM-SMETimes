"""NASA 100-cycle SOH forecast challenger using training-cell rate curvature.

The 50-cycle local-rate model remains unchanged. A multiplicative factor for
100-cycle forecasts is estimated from 50/100-cycle rates on *training cells*
only. Training may use their post-80%-SOH records; validation scores before
the 80% endpoint are reported separately. This is a small development set.
"""

import json
from pathlib import Path

import numpy as np

from battery_v2_features import load_all
from train_rul_v2 import oof_soh
from train_soh_v2 import fit as fit_soh, outer_splits, predict as predict_soh
from train_trajectory_v2 import fit_rate, future_pairs, predict_rate


OUT = Path('battery_models_v2/nasa_rate_decay_challenger/development_cv.json')


def all_future_pairs(cell, horizon):
    current, future = [], []
    for i in range(max(15, cell.reference_count), len(cell.cycle)):
        j = int(np.searchsorted(cell.cycle, cell.cycle[i] + horizon))
        if j >= len(cell.cycle) or cell.cycle[j] - cell.cycle[i] > 1.5*horizon:
            continue
        if not (cell.soh_available[i] and cell.soh_available[j]):
            continue
        current.append(i)
        future.append(j)
    return np.asarray(current, dtype=int), np.asarray(future, dtype=int)


def training_rate_factor(cells, train_ids):
    """Equal-cell robust median of observed 100-vs-50 cycle degradation rates."""
    ratios = []
    for key in train_ids:
        cell = cells[key]
        start100, end100 = all_future_pairs(cell, 100)
        start50, end50 = all_future_pairs(cell, 50)
        match50 = {int(a): int(b) for a, b in zip(start50, end50)}
        local = []
        for i, j in zip(start100, end100):
            if int(i) not in match50:
                continue
            k = match50[int(i)]
            rate50 = (cell.soh[i]-cell.soh[k]) / (cell.cycle[k]-cell.cycle[i])
            rate100 = (cell.soh[i]-cell.soh[j]) / (cell.cycle[j]-cell.cycle[i])
            if rate50 > 1e-5 and rate100 > 0:
                local.append(float(rate100 / rate50))
        if local:
            ratios.append({'cell_id': key, 'pairs': len(local),
                           'median_ratio': float(np.median(local))})
    if len(ratios) < 2:
        raise ValueError('need at least two training cells for rate calibration')
    factor = float(np.clip(np.median([row['median_ratio'] for row in ratios]), .5, 1.2))
    return factor, ratios


def score(cell, soh, rate, factor, before_eol):
    origin, target = (future_pairs(cell, 100) if before_eol
                      else all_future_pairs(cell, 100))
    if not len(origin):
        return None
    dt = cell.cycle[target]-cell.cycle[origin]
    actual = cell.soh[target]
    original = soh[origin]-rate[origin]*dt
    corrected = soh[origin]-rate[origin]*dt*factor
    return {'pairs': len(origin),
            'baseline_mape_percent': float(100*np.mean(np.abs(original-actual)/actual)),
            'corrected_mape_percent': float(100*np.mean(np.abs(corrected-actual)/actual)),
            'baseline_signed_error_soh': float(np.mean(original-actual)),
            'corrected_signed_error_soh': float(np.mean(corrected-actual))}


def evaluate():
    manifest, cells = load_all()
    records = [record for record in manifest['cells'] if record['dataset'] == 'NASA']
    folds = []
    for held, train_ids, val_ids in outer_splits(records, 'NASA'):
        factor, provenance = training_rate_factor(cells, train_ids)
        inner_soh = oof_soh(cells, records, train_ids)
        rate_model = fit_rate(cells, train_ids, inner_soh)
        soh_model = fit_soh(cells, train_ids, 'extra_trees_history')
        rows = []
        for key in val_ids:
            cell = cells[key]
            soh = predict_soh(soh_model, cell)
            rate = predict_rate(rate_model, cell, soh)
            rows.append({'cell_id': key, 'before_eol': score(cell, soh, rate, factor, True),
                         'all_observed': score(cell, soh, rate, factor, False)})
        folds.append({'held_out': held, 'factor_from_training': factor,
                      'training_rate_ratios': provenance, 'validation': rows})
        print(held, 'factor', round(factor, 3),
              'before EOL', rows[0]['before_eol'], flush=True)
    scored = [row['before_eol'] for fold in folds for row in fold['validation']
              if row['before_eol'] is not None]
    all_scored = [row['all_observed'] for fold in folds for row in fold['validation']]
    return {'status': 'development leave-one-cell-out; no independent test',
            'training_source_warning': 'rate curvature uses post-80%-SOH data of training cells',
            'pre_eol_validation_cells': len(scored),
            'pre_eol_baseline_mape_percent': float(np.mean([
                row['baseline_mape_percent'] for row in scored])),
            'pre_eol_corrected_mape_percent': float(np.mean([
                row['corrected_mape_percent'] for row in scored])),
            'all_observed_validation_cells': len(all_scored),
            'all_observed_baseline_mape_percent': float(np.mean([
                row['baseline_mape_percent'] for row in all_scored])),
            'all_observed_corrected_mape_percent': float(np.mean([
                row['corrected_mape_percent'] for row in all_scored])),
            'folds': folds}


if __name__ == '__main__':
    OUT.parent.mkdir(parents=True, exist_ok=True)
    result = evaluate()
    OUT.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('before EOL', result['pre_eol_baseline_mape_percent'],
          result['pre_eol_corrected_mape_percent'])
    print('all observed', result['all_observed_baseline_mape_percent'],
          result['all_observed_corrected_mape_percent'])
