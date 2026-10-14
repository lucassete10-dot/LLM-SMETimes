"""Separate prefix-only nominal health level from supervised future capacity loss."""
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from battery_smetimes import continuous_length as point


def estimator(x, y, seed, weights=None):
    imputer = SimpleImputer(strategy='median', keep_empty_features=True)
    z = imputer.fit_transform(x)
    scaler = StandardScaler().fit(z)
    model = ExtraTreesRegressor(n_estimators=40, min_samples_leaf=1,
        max_features=.8, random_state=seed, n_jobs=2).fit(scaler.transform(z), y,
                                                      sample_weight=weights)
    return dict(imputer=imputer, scaler=scaler, model=model)


def reference_level(cell, origin):
    # Training supervision only. Never read this during inference.
    c, y = ((cell['cycle'], cell['truth']) if 'early_cycle' in cell else
            (cell['label_cycle'], cell['soh']))
    c, y = np.asarray(c, float), np.asarray(y, float)
    valid = np.isfinite(c) & np.isfinite(y) & (y > 0)
    c, y = c[valid], y[valid]
    assert np.all(np.diff(c) > 0)
    if len(c) < 2 or c[0] > origin or c[-1] < origin:
        raise ValueError('training labels do not bracket origin')
    return float(np.interp(origin, c, y))


def fit(cells, ids, targets, origin, seed):
    ids = list(ids)
    families = {cells[k]['physical_family'] for k in ids}
    if len(families) != 1:
        raise ValueError('physical families differ')
    rows, losses, weights, levels, prefix = [], [], [], [], []
    for key in ids:
        level = reference_level(cells[key], origin)
        c, y = targets[key]
        ii = np.unique(np.linspace(0, len(c)-1, min(96, len(c))).round().astype(int))
        rows.append(point.point_inputs(cells[key], origin, c[ii], 'VIT'))
        losses.append(y[ii]-level)
        weights.append(np.full(len(ii), 1/len(ii)))
        levels.append(level)
        prefix.append(point.features(cells[key], origin, 'VIT'))
    w = np.concatenate(weights)
    w *= len(w)/w.sum()
    return dict(origin=origin, seed=seed, family=next(iter(families)), train_ids=ids,
                level=estimator(np.stack(prefix), np.array(levels), seed),
                loss=estimator(np.concatenate(rows), np.concatenate(losses), seed+1, w),
                training_levels=dict(zip(ids, levels)), total_trees=80)


def infer(head, x):
    return head['model'].predict(head['scaler'].transform(head['imputer'].transform(x)))


def predict(bundle, cell, cycle):
    if cell['physical_family'] != bundle['family']:
        raise ValueError('physical family mismatch')
    c = np.asarray(cycle, float)
    origin = bundle['origin']
    if c.ndim != 1 or not len(c) or not np.isfinite(c).all() or np.any(c <= origin):
        raise ValueError('invalid future query')
    level = infer(bundle['level'], point.features(cell, origin, 'VIT')[None])[0]
    return level + infer(bundle['loss'], point.point_inputs(cell, origin, c, 'VIT'))
