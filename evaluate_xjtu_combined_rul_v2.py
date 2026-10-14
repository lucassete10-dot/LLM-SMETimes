"""XJTU RUL ablation: 300-second diagnostics plus past usage conditions.

All feature windows end no later than the prediction cycle. Validation holds
out an entire XJTU charging/discharging strategy. This is exploratory because
unseen strategies can lie far outside the training feature distribution.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np

from battery_v2_features import derive_signal_features, load_all
from evaluate_paper_inspired_rul_v2 import baseline_oof, fit, predict, score
from evaluate_xjtu_long_prefix_v2 import ROOT as LONG_ROOT
from paper_inspired_battery_v2 import time_gated_matrix
from train_rul_v2 import eligible
from train_soh_v2 import outer_splits


OUT = Path('battery_models_v2/xjtu_combined_rul_challenger')
USAGE = Path('battery_models_v2/xjtu_usage_history/features.npz')
KINDS = ('gated_log', 'gated_raw')


def matrix(cell, raw300, usage):
    if len(raw300) != len(cell.cycle) or usage.shape != (len(cell.cycle), 16):
        raise ValueError(f'{cell.cell_id}: unaligned source arrays')
    delta, _ = derive_signal_features(raw300, cell.reference_count)
    view = SimpleNamespace(cycle=cell.cycle, raw=raw300, delta=delta)
    age = cell.cycle.astype('float32')
    return np.column_stack((time_gated_matrix(view), usage, age,
                            np.log1p(age))).astype('float32')


def evaluate():
    manifest, all_cells = load_all()
    records = [record for record in manifest['cells'] if record['dataset'] == 'XJTU']
    cells = {row['cell_id']: all_cells[row['cell_id']] for row in records}
    features = {}
    with np.load(LONG_ROOT / 'features.npz') as long_file, np.load(USAGE) as usage_file:
        for key, cell in cells.items():
            if not (np.array_equal(long_file[f'{key}_cycle'], cell.cycle) and
                    np.array_equal(usage_file[f'{key}_cycle'], cell.cycle)):
                raise ValueError(f'{key}: cycle alignment mismatch')
            features[key] = matrix(cell, long_file[f'{key}_300s'].astype('float32'),
                                   usage_file[f'{key}_usage'].astype('float32'))
    baseline = baseline_oof('XJTU', cells)
    folds = []
    oof_raw = {}
    for held, train, val in outer_splits(records, 'XJTU'):
        row = {'held_out': held, 'baseline': score(cells, val, baseline)}
        for kind in KINDS:
            fitted = fit(cells, train, features, kind)
            estimates = {key: predict(fitted, features[key], kind) for key in val}
            row[kind] = score(cells, val, estimates)
            if kind == 'gated_raw':
                oof_raw.update(estimates)
        folds.append(row)
        print(held, {name: {'mae': round(row[name]['cell_macro_mae_cycles'], 1),
                            'eol_mape_at_50': (round(row[name]['eol_mape_at_50_percent'], 2)
                                               if row[name]['cells_at_50'] else None)}
                     for name in ('baseline', *KINDS)}, flush=True)
    summary = {}
    for kind in ('baseline', *KINDS):
        by_cell = [cell for fold in folds for cell in fold[kind]['by_cell']]
        at50 = [cell['eol_mape_at_50_percent'] for cell in by_cell
                if cell['eol_mape_at_50_percent'] is not None]
        summary[kind] = {
            'cell_macro_mae_cycles': float(np.mean([cell['mae_cycles'] for cell in by_cell])),
            'eol_mape_at_50_percent': float(np.mean(at50)) if at50 else None,
            'cells_at_50': len(at50), 'all_endpoint_cells': len(by_cell),
            'improved_strategy_folds_vs_baseline': sum(
                fold[kind]['cell_macro_mae_cycles'] < fold['baseline']['cell_macro_mae_cycles']
                for fold in folds) if kind != 'baseline' else None}
    return {'status': 'development whole-strategy CV, not independent test',
            'input': '300s full diagnostic prefix plus past source-cycle controls; no capacity',
            'future_strategy_assumption': 'observed control distribution continues',
            'summary': summary, 'folds': folds}, cells, features, oof_raw


if __name__ == '__main__':
    report, cells, features, oof_raw = evaluate()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'development_cv.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    ids_by_row, cycles, estimates = [], [], []
    for key, cell in cells.items():
        keep = eligible(cell)
        ids_by_row.extend([key]*int(keep.sum()))
        cycles.extend(cell.cycle[keep].tolist())
        estimates.extend(oof_raw[key][keep].tolist())
    np.savez_compressed(OUT / 'development_oof_raw_rul.npz',
                        cell_id=np.asarray(ids_by_row), cycle=np.asarray(cycles),
                        estimate=np.asarray(estimates))
    ids = list(cells)
    # Keep this separate from the frozen selected model even if a metric improves.
    joblib.dump({'model': fit(cells, ids, features, 'gated_log'),
                 'training_cell_ids': ids, 'input': report['input']},
                OUT / 'xjtu_combined_log_candidate.joblib')
    joblib.dump({'model': fit(cells, ids, features, 'gated_raw'),
                 'training_cell_ids': ids, 'input': report['input']},
                OUT / 'xjtu_combined_raw_candidate.joblib')
    print('SUMMARY', report['summary'], flush=True)
