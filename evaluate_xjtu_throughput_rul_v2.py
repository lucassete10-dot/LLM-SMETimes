"""XJTU RUL from charge-throughput-aligned 600-second signal prefixes.

Voltage, current, and temperature are sampled at the same *observed early*
charge throughput instead of the same elapsed time. The integrated current
up to 0.08 Ah is an input signal, not the full-discharge capacity label.
Whole-strategy validation checks transfer across diagnostic rates.
"""

import json
from pathlib import Path

import joblib
import numpy as np

from battery_v2_features import load_all
from evaluate_paper_inspired_rul_v2 import fit, predict, score
from evaluate_xjtu_combined_rul_v2 import matrix
from evaluate_xjtu_invariant_rul_v2 import invariant_view
from evaluate_xjtu_long_prefix_v2 import ROOT as LONG_ROOT
from train_rul_v2 import eligible
from train_soh_v2 import outer_splits


OUT = Path('battery_models_v2/xjtu_throughput_rul_challenger')
USAGE = Path('battery_models_v2/xjtu_usage_history/features.npz')
Q_GRID = np.linspace(0, .08, 9, dtype='float32')
SOURCE_GRID_SECONDS = np.arange(0, 601, 30, dtype='float64')
STAGE_WIDTH = 3*len(SOURCE_GRID_SECONDS)+8


def stage_at_charge(stage):
    """Interpolate an early stage on cumulative absolute-current Ah."""
    stage = np.asarray(stage, dtype='float64')
    n = len(SOURCE_GRID_SECONDS)
    if stage.shape != (STAGE_WIDTH,):
        raise ValueError('expected one 600-second stage feature vector')
    voltage, current, temperature = stage[:n], stage[n:2*n], stage[2*n:3*n]
    q = np.r_[0, np.cumsum((np.abs(current[1:])+np.abs(current[:-1]))*.5*30/3600)]
    if not np.isfinite(q).all() or q[-1] < Q_GRID[-1]:
        raise ValueError('600-second stage does not reach 0.08 Ah of prefix throughput')
    unique_q, first = np.unique(q, return_index=True)
    v = np.interp(Q_GRID, unique_q, voltage[first])
    i = np.interp(Q_GRID, unique_q, current[first])
    temp = np.interp(Q_GRID, unique_q, temperature[first])
    duration = np.interp(Q_GRID[-1], unique_q, SOURCE_GRID_SECONDS[first])
    return np.r_[v, i, temp, duration, v[-1]-v[0], temp[-1]-temp[0],
                 np.mean(np.abs(i))].astype('float32')


def throughput_matrix(raw600):
    raw600 = np.asarray(raw600, dtype='float32')
    if raw600.ndim != 2 or raw600.shape[1] != 2*STAGE_WIDTH:
        raise ValueError('expected aligned 600-second charge/discharge features')
    return np.stack([np.r_[stage_at_charge(row[:STAGE_WIDTH]),
                            stage_at_charge(row[STAGE_WIDTH:])]
                     for row in raw600]).astype('float32')


def evaluate(save_models=True):
    manifest, all_cells = load_all()
    records = [row for row in manifest['cells'] if row['dataset'] == 'XJTU']
    cells = {row['cell_id']: all_cells[row['cell_id']] for row in records}
    views = {'throughput_full': {}, 'throughput_relative': {}}
    with np.load(LONG_ROOT / 'features.npz') as long_file, np.load(USAGE) as usage_file:
        for key, cell in cells.items():
            if not (np.array_equal(long_file[f'{key}_cycle'], cell.cycle) and
                    np.array_equal(usage_file[f'{key}_cycle'], cell.cycle)):
                raise ValueError(f'{key}: source cycle mismatch')
            raw = throughput_matrix(long_file[f'{key}_600s'])
            usage = usage_file[f'{key}_usage'].astype('float32')
            full = matrix(cell, raw, usage)
            views['throughput_full'][key] = full
            views['throughput_relative'][key] = invariant_view(full, raw.shape[1], False)
    folds = []
    oof = {name: {} for name in views}
    for held, train, val in outer_splits(records, 'XJTU'):
        fold = {'held_out': held}
        for name, features in views.items():
            model = fit(cells, train, features, 'gated_raw')
            estimates = {key: predict(model, features[key], 'gated_raw') for key in val}
            oof[name].update(estimates)
            fold[name] = score(cells, val, estimates)
        folds.append(fold)
        print(held, {name: {'mae': round(fold[name]['cell_macro_mae_cycles'], 1),
                            'early_eol': (round(fold[name]['eol_mape_at_50_percent'], 2)
                                          if fold[name]['cells_at_50'] else None)}
                     for name in views}, flush=True)
    summary = {}
    for name in views:
        by_cell = [row for fold in folds for row in fold[name]['by_cell']]
        at50 = [row['eol_mape_at_50_percent'] for row in by_cell
                if row['eol_mape_at_50_percent'] is not None]
        summary[name] = {'cell_macro_mae_cycles': float(np.mean([
            row['mae_cycles'] for row in by_cell])),
            'eol_mape_at_50_percent': float(np.mean(at50)) if at50 else None,
            'cells_at_50': len(at50),
            'equal_fold_mae_cycles': float(np.mean([
                fold[name]['cell_macro_mae_cycles'] for fold in folds])),
            'worst_cell_mae_cycles': float(max(row['mae_cycles'] for row in by_cell))}
    report = {'status': 'development whole-strategy CV, not independent test',
              'input': 'first 600 seconds of charge and discharge diagnostic stages, '
                       'up to 0.08 Ah early measured throughput; no total discharge capacity',
              'future_strategy_assumption': 'observed usage distribution continues',
              'summary': summary, 'folds': folds}
    if save_models:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / 'development_cv.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        for name, predictions in oof.items():
            ids_by_row, cycles, estimates = [], [], []
            for key, cell in cells.items():
                keep = eligible(cell)
                ids_by_row.extend([key]*int(keep.sum()))
                cycles.extend(cell.cycle[keep].tolist())
                estimates.extend(predictions[key][keep].tolist())
            np.savez_compressed(OUT / f'development_oof_{name}.npz',
                                cell_id=np.asarray(ids_by_row), cycle=np.asarray(cycles),
                                estimate=np.asarray(estimates))
            joblib.dump({'model': fit(cells, list(cells), views[name], 'gated_raw'),
                         'variant': name, 'training_cell_ids': list(cells),
                         'throughput_grid_ah': Q_GRID},
                        OUT / f'xjtu_{name}_candidate.joblib')
        best = min(views, key=lambda name: summary[name]['cell_macro_mae_cycles'])
        (OUT / 'selected_development_variant.txt').write_text(best, encoding='utf-8')
        joblib.dump(joblib.load(OUT / f'xjtu_{best}_candidate.joblib'),
                    OUT / 'xjtu_throughput_candidate.joblib')
    return report


if __name__ == '__main__':
    result = evaluate()
    print('SUMMARY', result['summary'], flush=True)
