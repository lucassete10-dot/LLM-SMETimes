"""Label-free inference for the separately frozen XJTU raw-target RUL model."""

from pathlib import Path

import joblib
import numpy as np

from battery_v2_features import make_inference_features
from train_rul_v2 import MIN_HISTORY, signal_features


MODEL = Path('battery_models_v2/rul/xjtu/raw_cycle_target_challenger.joblib')


def predict_series(raw, cycle, model_path=MODEL):
    feature = make_inference_features(raw, cycle, 5)
    if feature.raw.shape[1] != 46:
        raise ValueError('XJTU raw-target RUL challenger needs 46-column 120s signal')
    bundle = joblib.load(model_path)
    if (bundle['dataset'] != 'XJTU' or
            bundle['feature_schema'] != 'xjtu_diagnostic_prefix_120s_v1' or
            bundle['target_transform'] != 'raw_cycle_count' or
            bundle['reference_count'] != 5 or bundle['min_history'] != MIN_HISTORY):
        raise ValueError('raw-target RUL challenger model metadata mismatch')
    x = signal_features(feature)
    if x.shape[1] != bundle['imputer'].n_features_in_:
        raise ValueError('raw-target RUL challenger feature width mismatch')
    estimate = np.maximum(0, bundle['model'].predict(bundle['imputer'].transform(x)))
    estimate = np.asarray(estimate, dtype='float32')
    estimate[:MIN_HISTORY] = np.nan
    return {'cycle': feature.cycle.copy(), 'rul_cycles': estimate,
            'model': 'raw_cycle_target_challenger',
            'interval_status': 'NO CALIBRATED INTERVAL; not selected for default RUL'}
