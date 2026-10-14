"""XJTU RUL routing between absolute and relative experts under signal shift.

The route is selected from the first 15 diagnostic signal rows only, before
RUL prediction is eligible. All distance normalization is trained inside the
outer fold. A fixed 5x training-p95 threshold is exploratory, not calibrated.
"""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

from battery_v2_features import load_all
from evaluate_paper_inspired_rul_v2 import score
from train_rul_v2 import eligible
from train_soh_v2 import outer_splits


OUT = Path('battery_models_v2/xjtu_ood_routing_challenger')
FULL = Path('battery_models_v2/xjtu_combined_rul_challenger/development_oof_raw_rul.npz')
RELATIVE = Path('battery_models_v2/xjtu_invariant_rul_challenger/development_oof_relative_rul.npz')
OOD_THRESHOLD = 5.0


def initial_signature(cell):
    if len(cell.raw) < 15:
        raise ValueError('at least 15 diagnostic rows required')
    return np.nanmedian(cell.raw[:15], axis=0)


def fit_distance(cells, train):
    train_x = np.stack([initial_signature(cells[key]) for key in train])
    imputer = SimpleImputer(strategy='median', keep_empty_features=True).fit(train_x)
    scaler = RobustScaler(quantile_range=(10, 90)).fit(imputer.transform(train_x))
    scaler.scale_ = np.maximum(scaler.scale_, .02)
    reference = np.clip(scaler.transform(imputer.transform(train_x)), -10, 10)
    self_distances = np.linalg.norm(reference[:, None, :]-reference[None, :, :], axis=2)
    np.fill_diagonal(self_distances, np.inf)
    p95 = float(np.quantile(np.min(self_distances, axis=1), .95))
    return {'imputer': imputer, 'scaler': scaler, 'reference': reference,
            'nearest_train_p95': p95, 'training_cell_ids': list(train)}


def distance_ratio(calibration, cell):
    candidate = initial_signature(cell)[None, :]
    normalized = np.clip(calibration['scaler'].transform(
        calibration['imputer'].transform(candidate)), -10, 10)
    distance = np.min(np.linalg.norm(
        calibration['reference']-normalized, axis=1))
    return float(distance/max(calibration['nearest_train_p95'], 1e-6))


def load_oof(path, cells):
    prediction = {key: np.full(len(cell.cycle), np.nan)
                  for key, cell in cells.items()}
    with np.load(path) as archive:
        for key, cycle, estimate in zip(archive['cell_id'], archive['cycle'], archive['estimate']):
            key = str(key)
            i = int(np.searchsorted(cells[key].cycle, cycle))
            if i >= len(cells[key].cycle) or cells[key].cycle[i] != cycle:
                raise ValueError(f'{key}: OOF cycle misalignment')
            prediction[key][i] = estimate
    for key, cell in cells.items():
        if not np.isfinite(prediction[key][eligible(cell)]).all():
            raise ValueError(f'{key}: missing OOF RUL prediction')
    return prediction


def evaluate():
    manifest, all_cells = load_all()
    records = [row for row in manifest['cells'] if row['dataset'] == 'XJTU']
    cells = {row['cell_id']: all_cells[row['cell_id']] for row in records}
    full = load_oof(FULL, cells)
    relative = load_oof(RELATIVE, cells)
    folds = []
    for held, train, val in outer_splits(records, 'XJTU'):
        calibration = fit_distance(cells, train)
        ratio = {key: distance_ratio(calibration, cells[key]) for key in val}
        routed = {key: relative[key] if ratio[key] > OOD_THRESHOLD else full[key]
                  for key in val}
        row = {'held_out': held, 'training_distance_p95': calibration['nearest_train_p95'],
               'cell_distance_ratio': ratio,
               'relative_route_cells': [key for key in val if ratio[key] > OOD_THRESHOLD],
               'full': score(cells, val, full),
               'relative': score(cells, val, relative),
               'routed': score(cells, val, routed)}
        folds.append(row)
        print(held, 'relative', len(row['relative_route_cells']), '/', len(val),
              'MAE', {name: round(row[name]['cell_macro_mae_cycles'], 1)
                      for name in ('full', 'relative', 'routed')}, flush=True)
    summary = {}
    for name in ('full', 'relative', 'routed'):
        by_cell = [cell for fold in folds for cell in fold[name]['by_cell']]
        at50 = [cell['eol_mape_at_50_percent'] for cell in by_cell
                if cell['eol_mape_at_50_percent'] is not None]
        summary[name] = {'cell_macro_mae_cycles': float(np.mean([
            cell['mae_cycles'] for cell in by_cell])),
            'eol_mape_at_50_percent': float(np.mean(at50)) if at50 else None,
            'cells_at_50': len(at50),
            'improved_strategy_folds_vs_full': sum(
                fold[name]['cell_macro_mae_cycles'] < fold['full']['cell_macro_mae_cycles']
                for fold in folds) if name != 'full' else None}
    report = {'status': 'development whole-strategy CV; route threshold not independently calibrated',
              'route': 'relative expert if first-15-signal nearest-distance exceeds 5x '
                       'training nearest-neighbor p95',
              'summary': summary, 'folds': folds}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'development_cv.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    joblib.dump({'calibration': fit_distance(cells, list(cells)),
                 'threshold': OOD_THRESHOLD,
                 'full_model': 'battery_models_v2/xjtu_combined_rul_challenger/'
                               'xjtu_combined_raw_candidate.joblib',
                 'relative_model': 'battery_models_v2/xjtu_invariant_rul_challenger/'
                                   'xjtu_invariant_candidate.joblib'},
                OUT / 'router_candidate.joblib')
    return report


if __name__ == '__main__':
    result = evaluate()
    print('SUMMARY', result['summary'], flush=True)
