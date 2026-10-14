"""Check whether a regularized linear life estimator extrapolates across groups."""

import json
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler

from battery_v2_features import load_all
from train_rul_v2 import eligible, signal_features, assess
from train_soh_v2 import outer_splits


def fit(cells, ids, alpha):
    x_parts, y_parts, w_parts = [], [], []
    for key in ids:
        cell = cells[key]
        keep = np.flatnonzero(eligible(cell))[::5]
        if not len(keep):
            continue
        x_parts.append(signal_features(cell)[keep])
        y_parts.append(np.full(len(keep), cell.endpoint_cycle, dtype='float32'))
        w_parts.append(np.full(len(keep), 1/len(keep)/len(ids)))
    x, y, w = map(np.concatenate, (x_parts, y_parts, w_parts))
    imputer = SimpleImputer(strategy='median', keep_empty_features=True).fit(x)
    scaler = RobustScaler(quantile_range=(10, 90)).fit(imputer.transform(x))
    transformed = np.clip(scaler.transform(imputer.transform(x)), -5, 5)
    model = Ridge(alpha=alpha).fit(transformed, y, sample_weight=w*len(y))
    return imputer, scaler, model


def predict(bundle, cell):
    imputer, scaler, model = bundle
    x = np.clip(scaler.transform(imputer.transform(signal_features(cell))), -5, 5)
    life = model.predict(x)
    return np.maximum(0, life-cell.cycle)


def main():
    manifest, cells = load_all()
    for domain in ('MIT', 'XJTU'):
        folds = []
        for group, train, val in outer_splits(manifest['cells'], domain):
            train = [key for key in train if cells[key].endpoint_cycle is not None]
            val = [key for key in val if cells[key].endpoint_cycle is not None]
            base = float(np.median([cells[key].endpoint_cycle for key in train]))
            variants = {}
            for alpha in (100, 1000, 10000):
                model = fit(cells, train, alpha)
                preds = {key: predict(model, cells[key]) for key in val}
                variants[str(alpha)] = assess(cells, val, preds, base)['cell_macro_mae_cycles']
            folds.append({'held_out': group, 'macro_mae_cycles': variants})
            print(domain, group, variants, flush=True)
        folder = Path('battery_models_v2/rul') / domain.lower()
        (folder / 'linear_life_diagnostic.json').write_text(json.dumps(folds, indent=2), encoding='utf-8')
        print('MEANS', domain, {alpha: float(np.mean([row['macro_mae_cycles'][alpha]
                                                    for row in folds])) for alpha in ('100','1000','10000')}, flush=True)


if __name__ == '__main__':
    main()
