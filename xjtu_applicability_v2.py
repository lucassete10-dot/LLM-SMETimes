"""Descriptive XJTU early-signal distance to the training population.

This is not a calibrated OOD detector or an accuracy guarantee. Only the
first 20 signal diagnostics are used; no capacity or life labels are read.
"""

import json
from pathlib import Path
import warnings

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler


MANIFEST = Path('battery_data_v2/manifest.json')
REFERENCE = Path('battery_models_v2/xjtu_applicability_reference.npz')


def signature(raw):
    raw = np.asarray(raw, dtype='float32')
    if raw.ndim != 2 or raw.shape[1] != 46 or len(raw) < 6 or np.isinf(raw).any():
        raise ValueError('XJTU applicability needs valid 46-column signal rows')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmedian(raw[:min(20, len(raw))], axis=0)


def build_reference():
    records = [row for row in json.loads(MANIFEST.read_text(encoding='utf-8'))['cells']
               if row['dataset'] == 'XJTU']
    if len(records) != 55:
        raise ValueError('expected 55 XJTU development cells')
    inputs = []
    for row in records:
        with np.load(row['feature_source']) as data:
            inputs.append(signature(data[row['feature_array']]))
    x = np.asarray(inputs, dtype='float32')
    imputer = SimpleImputer(strategy='median', keep_empty_features=True).fit(x)
    filled = imputer.transform(x)
    scaler = RobustScaler(quantile_range=(10, 90)).fit(filled)
    scaler.scale_ = np.maximum(scaler.scale_, .02)
    reference = np.clip(scaler.transform(filled), -10, 10)
    distance = np.linalg.norm(reference[:, None, :] - reference[None, :, :], axis=2)
    np.fill_diagonal(distance, np.inf)
    nearest = np.min(distance, axis=1)
    p95 = float(np.quantile(nearest, .95))
    if not np.isfinite(p95) or p95 <= 0:
        raise ValueError('training-neighbor reference radius is invalid')
    np.savez_compressed(REFERENCE, cell_ids=np.asarray([row['cell_id'] for row in records]),
                        imputer_statistics=imputer.statistics_,
                        scaler_center=scaler.center_, scaler_scale=scaler.scale_,
                        reference_scaled=reference,
                        nearest_training_p95=p95)
    print('WROTE', REFERENCE, 'training-cell nearest-neighbor p95', p95)


def assess(raw, reference_path=REFERENCE):
    raw = np.asarray(raw, dtype='float32')
    if len(raw) < 20:
        return {'status': 'INSUFFICIENT_HISTORY', 'observations_required': 20,
                'distance_ratio': None,
                'warning': 'no applicability claim before 20 diagnostics'}
    vector = signature(raw)
    with np.load(reference_path, allow_pickle=False) as data:
        statistics = data['imputer_statistics']
        center = data['scaler_center']
        scale = data['scaler_scale']
        training = data['reference_scaled']
        typical = float(data['nearest_training_p95'])
    if len(vector) != len(statistics) or training.shape[1] != len(vector):
        raise ValueError('XJTU applicability feature schema mismatch')
    filled = np.where(np.isnan(vector), statistics, vector)
    scaled = np.clip((filled-center)/scale, -10, 10)
    nearest = float(np.min(np.linalg.norm(training-scaled[None, :], axis=1)))
    ratio = nearest/typical
    return {'status': ('OUTSIDE_TYPICAL_TRAINING_NEIGHBORHOOD' if ratio > 1 else
                       'WITHIN_TYPICAL_TRAINING_NEIGHBORHOOD'),
            'distance_ratio': ratio, 'nearest_training_distance': nearest,
            'training_nearest_neighbor_p95': typical,
            'warning': 'descriptive feature shift only; not a calibrated error or abstention threshold'}


if __name__ == '__main__':
    build_reference()
