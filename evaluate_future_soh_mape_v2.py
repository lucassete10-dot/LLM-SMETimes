"""Re-evaluate saved trajectory experiment with actual future-SOH MAPE.

Each outer fold retrains the original SOH and local-rate models from training
cells/strategies only. Forecasts use measurements available at the origin and
are scored 50/100 source cycles later where a valid future label exists.
"""

import json
from pathlib import Path

import numpy as np

from battery_v2_features import load_all
from train_rul_v2 import oof_soh
from train_soh_v2 import fit as fit_soh, outer_splits, predict as predict_soh
from train_trajectory_v2 import SOH_MODEL, fit_rate, future_pairs, predict_rate


ROOT = Path('battery_models_v2/trajectory')
OUTPUT = ROOT / 'future_soh_mape.json'
DOMAINS = ('NASA', 'CALCE', 'MIT', 'XJTU')
HORIZONS = (50, 100)


def score_horizon(cells, ids, soh, rate, horizon):
    rows = []
    for key in ids:
        cell = cells[key]
        current, future = future_pairs(cell, horizon)
        if len(current) == 0:
            continue
        dt = cell.cycle[future]-cell.cycle[current]
        estimate = soh[key][current] - rate[key][current]*dt
        true = cell.soh[future]
        if np.any(true <= 0) or not np.isfinite(estimate).all():
            raise ValueError(f'{key}/{horizon}: nonpositive target or invalid forecast')
        error = np.abs(estimate-true)
        rows.append({'cell_id': key, 'pairs': int(len(current)),
                     'mae': float(np.mean(error)),
                     'mape_percent': float(100*np.mean(error/true)),
                     'flat_current_soh_mape_percent': float(100*np.mean(
                         np.abs(soh[key][current]-true)/true))})
    return {'cells': len(rows), 'pairs': sum(row['pairs'] for row in rows),
            'cell_macro_mae': float(np.mean([row['mae'] for row in rows])) if rows else None,
            'cell_macro_mape_percent': float(np.mean([row['mape_percent']
                                                      for row in rows])) if rows else None,
            'flat_current_soh_mape_percent': float(np.mean([
                row['flat_current_soh_mape_percent'] for row in rows])) if rows else None,
            'worst_cell_mape_percent': float(max(row['mape_percent']
                                                  for row in rows)) if rows else None,
            'by_cell': rows}


def evaluate():
    manifest, all_cells = load_all()
    report = {'status': 'development whole-cell/batch rolling-origin future-SOH evaluation',
              'assumption': 'observed operating strategy and learned local rate continue',
              'warning': '50/100-cycle forecasts from each eligible origin are not a '
                         'single early-life forecast to end of life; future MAPE <=6% '
                         'has not yet been assessed for the full remaining curve',
              'domains': {}}
    for domain in DOMAINS:
        records = [row for row in manifest['cells'] if row['dataset'] == domain]
        cells = {row['cell_id']: all_cells[row['cell_id']] for row in records}
        saved_folds = {row['held_out']: row for row in json.loads(
            (ROOT / domain.lower() / 'group_cv.json').read_text(encoding='utf-8'))}
        folds = []
        for held, train_ids, val_ids in outer_splits(records, domain):
            if set(train_ids) & set(val_ids):
                raise ValueError(f'{domain}/{held}: training/validation overlap')
            inner_soh = oof_soh(cells, records, train_ids)
            rate_model = fit_rate(cells, train_ids, inner_soh)
            soh_model = fit_soh(cells, train_ids, SOH_MODEL)
            soh = {key: predict_soh(soh_model, cells[key]) for key in val_ids}
            rate = {key: predict_rate(rate_model, cells[key], soh[key]) for key in val_ids}
            scores = {}
            for horizon in HORIZONS:
                result = score_horizon(cells, val_ids, soh, rate, horizon)
                saved = saved_folds[held][f'forecast_{horizon}']['cell_macro_mae']
                if saved is None:
                    if result['cell_macro_mae'] is not None:
                        raise ValueError(f'{domain}/{held}/{horizon}: unexpected scored cells')
                elif not np.isclose(result['cell_macro_mae'], saved, atol=1e-8):
                    raise ValueError(f'{domain}/{held}/{horizon}: reproduction MAE mismatch')
                scores[str(horizon)] = result
            folds.append({'held_out': held, 'train_cells': len(train_ids),
                          'validation_cells': len(val_ids), 'horizons': scores})
            print(domain, held, {h: round(scores[str(h)]['cell_macro_mape_percent'], 3)
                                  if scores[str(h)]['cells'] else None for h in HORIZONS},
                  flush=True)
        summary = {}
        for horizon in HORIZONS:
            metrics = [fold['horizons'][str(horizon)] for fold in folds
                       if fold['horizons'][str(horizon)]['cells']]
            if not metrics:
                summary[str(horizon)] = None
                continue
            summary[str(horizon)] = {
                'equal_fold_cell_macro_mape_percent': float(np.mean([
                    item['cell_macro_mape_percent'] for item in metrics])),
                'equal_cell_mape_percent': float(np.mean([
                    row['mape_percent'] for item in metrics for row in item['by_cell']])),
                'equal_fold_flat_current_soh_mape_percent': float(np.mean([
                    item['flat_current_soh_mape_percent'] for item in metrics])),
                'worst_cell_mape_percent': float(max(
                    item['worst_cell_mape_percent'] for item in metrics)),
                'scored_folds': len(metrics),
                'scored_cells': sum(item['cells'] for item in metrics),
                'scored_pairs': sum(item['pairs'] for item in metrics),
            }
        report['domains'][domain] = {'summary': summary, 'folds': folds}
    return report


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    result = evaluate()
    OUTPUT.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('WROTE', OUTPUT, flush=True)


if __name__ == '__main__':
    main()
