"""Development ablation: forecast EOL in equivalent full cycles (EFC).

This *usage-assisted* experiment reads prior complete discharge-current
traces. It is not comparable on input requirements to the short-prefix-only
models. Future current traces are used solely to make training EFC targets;
the held-out cell contributes only cycles strictly before its origin.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

from battery_v2_features import load_cell, load_manifest
from evaluate_empirical_curve_v2 import early_features
from train_soh_v2 import outer_splits


AUDIT = Path('battery_models_v2/xjtu_operating_throughput_audit.json')
USAGE = Path('battery_models_v2/xjtu_usage_history/features.npz')
OUT = Path('battery_models_v2/xjtu_efc_life_challenger')
MAX_LIFE = 2500


def build_view(records, rows, cells):
    result = {}
    with np.load(USAGE) as source:
        for record in records:
            key = record['cell_id']
            row = rows[key]
            if not row['origin_admissible']:
                continue
            cell = cells[key]
            origin = max(15, cell.reference_count)
            if int(cell.cycle[origin]) != row['origin_cycle']:
                raise ValueError(f'{key}: origin misalignment')
            if not np.array_equal(source[f'{key}_cycle'], cell.cycle):
                raise ValueError(f'{key}: usage cycle misalignment')
            signal = early_features(cell, origin, 'relative')
            usage = source[f'{key}_usage'][origin].astype(float)
            scalars = np.r_[row['early_mean_ah_per_cycle'],
                            row['early_equivalent_full_cycles'],
                            row['origin_cycle'], usage]
            result[key] = (signal, scalars)
    return result


def encode(train, val, view):
    signal_train = np.stack([view[key][0] for key in train])
    signal_val = np.stack([view[key][0] for key in val])
    imputer = SimpleImputer(strategy='median', keep_empty_features=True).fit(signal_train)
    scaler = RobustScaler(quantile_range=(10, 90)).fit(imputer.transform(signal_train))
    a = np.clip(scaler.transform(imputer.transform(signal_train)), -6, 6)
    b = np.clip(scaler.transform(imputer.transform(signal_val)), -6, 6)
    pca = PCA(n_components=min(8, len(train)-2, a.shape[1])).fit(a)
    scalar_train = np.stack([view[key][1] for key in train])
    scalar_val = np.stack([view[key][1] for key in val])
    scalar_imputer = SimpleImputer(strategy='median', keep_empty_features=True).fit(scalar_train)
    scalar_scaler = RobustScaler(quantile_range=(10, 90)).fit(
        scalar_imputer.transform(scalar_train))
    c = np.clip(scalar_scaler.transform(scalar_imputer.transform(scalar_train)), -8, 8)
    d = np.clip(scalar_scaler.transform(scalar_imputer.transform(scalar_val)), -8, 8)
    return np.column_stack((pca.transform(a), c)), np.column_stack((pca.transform(b), d))


def score_one(row, predicted_efc, predicted_cycle):
    origin = row['origin_cycle']
    usage = row['early_mean_ah_per_cycle']/2
    predicted_from_efc = np.clip(
        origin + max(0, predicted_efc-row['early_equivalent_full_cycles'])/usage,
        origin, MAX_LIFE)
    oracle = np.clip(
        origin + max(0, row['endpoint_equivalent_full_cycles']-
                     row['early_equivalent_full_cycles'])/usage,
        origin, MAX_LIFE)
    true = row['endpoint_cycle']
    return {'cell_id': row['cell_id'], 'group': row['group'],
            'source_endpoint_cycle': true,
            'predicted_efc_endpoint_cycle': float(predicted_from_efc),
            'predicted_direct_endpoint_cycle': float(np.clip(predicted_cycle, origin, MAX_LIFE)),
            'oracle_efc_conversion_cycle': float(oracle),
            'efc_endpoint_ape_percent': float(100*abs(predicted_from_efc-true)/true),
            'direct_endpoint_ape_percent': float(100*abs(np.clip(
                predicted_cycle, origin, MAX_LIFE)-true)/true),
            'oracle_conversion_ape_percent': float(100*abs(oracle-true)/true)}


def main():
    manifest = load_manifest()
    records = [row for row in manifest['cells'] if row['dataset'] == 'XJTU']
    cells = {row['cell_id']: load_cell(row) for row in records}
    audit = json.loads(AUDIT.read_text(encoding='utf-8'))
    rows = {row['cell_id']: row for row in audit['by_cell']}
    if len(rows) != 55:
        raise ValueError('expected audited operating throughput for 55 cells')
    view = build_view(records, rows, cells)
    folds = []
    for group, train_all, val_all in outer_splits(records, 'XJTU'):
        train = [key for key in train_all if key in view]
        val = [key for key in val_all if key in view]
        x_train, x_val = encode(train, val, view)
        labels_efc = np.array([rows[key]['endpoint_equivalent_full_cycles']
                               for key in train])
        labels_cycle = np.array([rows[key]['endpoint_cycle'] for key in train])
        reg_efc = ExtraTreesRegressor(n_estimators=200, min_samples_leaf=2,
                                      max_features=.8, random_state=2026, n_jobs=-1)
        reg_cycle = ExtraTreesRegressor(n_estimators=200, min_samples_leaf=2,
                                        max_features=.8, random_state=2026, n_jobs=-1)
        reg_efc.fit(x_train, labels_efc)
        reg_cycle.fit(x_train, labels_cycle)
        scored = [score_one(rows[key], e, c) for key, e, c in zip(
            val, reg_efc.predict(x_val), reg_cycle.predict(x_val))]
        fold = {'held_out': group, 'training_cells': len(train),
                'scored_cells': len(scored),
                'efc_eol_mape_percent': float(np.mean([
                    row['efc_endpoint_ape_percent'] for row in scored])),
                'direct_eol_mape_percent': float(np.mean([
                    row['direct_endpoint_ape_percent'] for row in scored])),
                'oracle_conversion_mape_percent': float(np.mean([
                    row['oracle_conversion_ape_percent'] for row in scored])),
                'by_cell': scored}
        folds.append(fold)
        print(group, 'EFC', round(fold['efc_eol_mape_percent'], 2),
              'direct', round(fold['direct_eol_mape_percent'], 2),
              'oracle', round(fold['oracle_conversion_mape_percent'], 2), flush=True)
    all_rows = [row for fold in folds for row in fold['by_cell']]
    summary = {'scored_cells': len(all_rows),
               'efc_eol_mape_percent': float(np.mean([
                   row['efc_endpoint_ape_percent'] for row in all_rows])),
               'direct_eol_mape_percent': float(np.mean([
                   row['direct_endpoint_ape_percent'] for row in all_rows])),
               'oracle_conversion_mape_percent': float(np.mean([
                   row['oracle_conversion_ape_percent'] for row in all_rows]))}
    report = {'status': 'development whole-strategy CV; not independent test or selected model',
              'input': 'past completed discharge current traces plus causal 120s diagnostic '
                       'signals; different regime from prefix-only inference',
              'target': 'source-defined XJTU endpoint, not exact per-cell 80% crossing',
              'future_assumption': 'early mean Ah per source cycle continues',
              'oracle_warning': 'oracle uses future EFC label only to test conversion, '
                                'never as a model prediction',
              'summary': summary, 'folds': folds}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'development_cv.json'
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('SUMMARY', summary)
    print('WROTE', path)


if __name__ == '__main__':
    main()
