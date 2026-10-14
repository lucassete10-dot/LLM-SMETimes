"""Preserve jointly learned knee/rate combinations instead of averaging parameters."""
import numpy as np
from battery_smetimes import continuous_length as point


def fit(cells, ids, targets, origin, seed):
    bundle = point.fit(cells,ids,origin,'continuous','VIT',1,seed,targets)
    x = np.stack([point.features(cells[k],origin) for k in ids])
    z = bundle['scaler'].transform(bundle['imputer'].transform(x))
    bundle['training_leaves'] = bundle['model'].apply(z)
    bundle['joint_parameters'] = np.array([r['parameters'] for r in bundle['training_shape_fits']])
    return bundle


def donor_weights(bundle, cell):
    x = point.features(cell,bundle['origin'],'VIT')[None]
    leaves = bundle['model'].apply(bundle['scaler'].transform(bundle['imputer'].transform(x)))[0]
    matches = bundle['training_leaves'] == leaves[None]
    assert np.all(matches.sum(axis=0) > 0)
    weights = (matches/matches.sum(axis=0)).mean(axis=1)
    assert np.all(weights >= 0) and abs(weights.sum()-1) < 1e-12
    return weights


def predict(bundle, cell, cycle):
    if cell['physical_family'] != bundle['family']:
        raise ValueError('physical family mismatch')
    c = np.asarray(cycle,float)
    if c.ndim != 1 or not len(c) or not np.isfinite(c).all() or np.any(c <= bundle['origin']):
        raise ValueError('invalid future')
    w = donor_weights(bundle,cell)
    curves = np.stack([point.basis(c,bundle['origin'],a[3])@a[:3]
                       for a in bundle['joint_parameters']])
    return w@curves
