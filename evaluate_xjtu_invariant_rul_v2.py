"""XJTU cross-strategy RUL ablation removing absolute diagnostic levels.

This tests whether relative, past-only degradation features transfer better
to an unseen discharge protocol. The six strategy folds are development CV.
"""

import json
from pathlib import Path

import joblib
import numpy as np

from battery_v2_features import load_all
from evaluate_paper_inspired_rul_v2 import fit, predict, score
from evaluate_xjtu_combined_rul_v2 import matrix
from evaluate_xjtu_long_prefix_v2 import ROOT as LONG_ROOT
from train_soh_v2 import outer_splits


OUT = Path('battery_models_v2/xjtu_invariant_rul_challenger')
USAGE = Path('battery_models_v2/xjtu_usage_history/features.npz')
PRIOR = Path('battery_models_v2/xjtu_combined_rul_challenger/development_cv.json')


def invariant_view(full, raw_width, include_usage):
    """Drop raw absolute prefix; optionally drop usage-condition statistics."""
    if full.ndim != 2 or full.shape[1] != 4*raw_width+1+16+2:
        raise ValueError('unexpected combined diagnostic feature layout')
    relative = full[:, raw_width:]
    if include_usage:
        return relative
    return np.column_stack((relative[:, :-18], relative[:, -2:]))


def evaluate(save_candidate=True):
    manifest, all_cells = load_all()
    records = [record for record in manifest['cells'] if record['dataset'] == 'XJTU']
    cells = {row['cell_id']: all_cells[row['cell_id']] for row in records}
    matrices = {'relative_plus_usage': {}, 'relative_only': {}}
    with np.load(LONG_ROOT / 'features.npz') as long_file, np.load(USAGE) as usage_file:
        for key, cell in cells.items():
            if not (np.array_equal(long_file[f'{key}_cycle'], cell.cycle) and
                    np.array_equal(usage_file[f'{key}_cycle'], cell.cycle)):
                raise ValueError(f'{key}: cycle alignment mismatch')
            raw = long_file[f'{key}_300s'].astype('float32')
            usage = usage_file[f'{key}_usage'].astype('float32')
            full = matrix(cell, raw, usage)
            matrices['relative_plus_usage'][key] = invariant_view(full, raw.shape[1], True)
            matrices['relative_only'][key] = invariant_view(full, raw.shape[1], False)
    prior = {row['held_out']: row['gated_raw'] for row in json.loads(
        PRIOR.read_text(encoding='utf-8'))['folds']}
    folds = []
    oof_relative = {}
    for held, train, val in outer_splits(records, 'XJTU'):
        row = {'held_out': held, 'full_300s_usage': prior[held]}
        for name, features in matrices.items():
            model = fit(cells, train, features, 'gated_raw')
            estimate = {key: predict(model, features[key], 'gated_raw') for key in val}
            row[name] = score(cells, val, estimate)
            if name == 'relative_only':
                oof_relative.update(estimate)
        folds.append(row)
        print(held, {key: round(row[key]['cell_macro_mae_cycles'], 1)
                     for key in ('full_300s_usage', *matrices)}, flush=True)
    summary = {}
    for name in ('full_300s_usage', *matrices):
        by_cell = [cell for fold in folds for cell in fold[name]['by_cell']]
        at50 = [cell['eol_mape_at_50_percent'] for cell in by_cell
                if cell['eol_mape_at_50_percent'] is not None]
        summary[name] = {
            'cell_macro_mae_cycles': float(np.mean([cell['mae_cycles'] for cell in by_cell])),
            'eol_mape_at_50_percent': float(np.mean(at50)) if at50 else None,
            'cells_at_50': len(at50),
            'improved_strategy_folds_vs_full': sum(
                fold[name]['cell_macro_mae_cycles'] <
                fold['full_300s_usage']['cell_macro_mae_cycles'] for fold in folds)
                if name != 'full_300s_usage' else None,
        }
    report = {'status': 'development whole-strategy CV, not independent test',
              'input': 'first 300s diagnostic and causal history; no capacity',
              'ablation': 'remove absolute raw levels; optionally remove prior usage statistics',
              'summary': summary, 'folds': folds}
    if save_candidate:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / 'development_cv.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        from train_rul_v2 import eligible
        ids_by_row, cycles, estimates = [], [], []
        for key, cell in cells.items():
            keep = eligible(cell)
            ids_by_row.extend([key]*int(keep.sum()))
            cycles.extend(cell.cycle[keep].tolist())
            estimates.extend(oof_relative[key][keep].tolist())
        np.savez_compressed(OUT / 'development_oof_relative_rul.npz',
                            cell_id=np.asarray(ids_by_row), cycle=np.asarray(cycles),
                            estimate=np.asarray(estimates))
        best = min(matrices, key=lambda name: summary[name]['cell_macro_mae_cycles'])
        joblib.dump({'model': fit(cells, list(cells), matrices[best], 'gated_raw'),
                     'variant': best, 'training_cell_ids': list(cells)},
                    OUT / 'xjtu_invariant_candidate.joblib')
    return report


if __name__ == '__main__':
    result = evaluate()
    print('SUMMARY', result['summary'], flush=True)
