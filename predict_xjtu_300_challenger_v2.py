"""Label-free SOH inference for the separate XJTU 300-second challenger.

No independent accuracy or calibrated interval is claimed. The selected
120-second SOH/RUL model and its external-test workflow remain unchanged.
"""

from pathlib import Path

import joblib
import numpy as np

from battery_v2_features import make_inference_features
from train_xjtu_300_challenger_v2 import MODEL


def predict_series(raw, cycle, *, model_path=MODEL):
    raw = np.asarray(raw, dtype='float32')
    if (raw.ndim != 2 or raw.shape[1] != 82 or np.isinf(raw).any() or
            not np.isfinite(raw).any(axis=1).all()):
        raise ValueError('expected 82 finite-or-missing 300s XJTU raw columns')
    feature = make_inference_features(raw, cycle, 5)
    if len(feature.cycle) <= 5 or np.any(feature.cycle < 1):
        raise ValueError('at least six diagnostics are required')
    bundle = joblib.load(Path(model_path))
    if (bundle['feature_schema'] != 'xjtu_diagnostic_prefix_300s_v1' or
            bundle['reference_count'] != 5 or bundle['raw_columns'] != 82):
        raise ValueError('saved challenger schema mismatch')
    matrix = feature.matrix(bundle['variant'])
    if matrix.shape[1] != bundle['imputer'].n_features_in_:
        raise ValueError('300s model feature width mismatch')
    soh = np.asarray(bundle['model'].predict(bundle['imputer'].transform(matrix)),
                     dtype='float32')
    soh[:5] = np.nan
    return {'cycle': feature.cycle.copy(), 'soh': soh,
            'model_status': 'DEVELOPMENT CHALLENGER; independent accuracy unknown',
            'feature_schema': bundle['feature_schema'],
            'interval_status': 'NO CALIBRATED INTERVAL',
            'rul_status': 'NOT EVALUATED with this SOH challenger'}
