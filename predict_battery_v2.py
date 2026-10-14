"""Label-free inference from protocol-matched, already extracted signal features.

Use the same feature extractor and column order as training. This API does not
accept capacity or SOH as input. Unknown protocols require a new adapter and
independent validation; a matching column count alone is insufficient.
"""

import json
from pathlib import Path

import joblib
import numpy as np

from battery_v2_features import make_inference_features
from train_rul_v2 import MIN_HISTORY, predict_rul
from train_soh_v2 import predict as predict_soh
from train_trajectory_v2 import predict_rate, predict_trajectory
from xjtu_applicability_v2 import assess as assess_xjtu_applicability


ROOT = Path('battery_models_v2')
REFERENCE_COUNTS = {'NASA': 1, 'CALCE': 1, 'MIT': 1, 'XJTU': 5}
FUTURE_STRATEGY_RELATIONS = {'same', 'changed', 'unknown'}


def predict_series(dataset, raw, cycle, *, model_root=ROOT,
                   future_strategy_relation='unknown'):
    dataset = dataset.upper()
    if dataset not in REFERENCE_COUNTS:
        raise ValueError(f'unsupported protocol: {dataset}')
    if (not isinstance(future_strategy_relation, str) or
            future_strategy_relation not in FUTURE_STRATEGY_RELATIONS):
        raise ValueError('future_strategy_relation must be same, changed, or unknown')
    feature = make_inference_features(raw, cycle, REFERENCE_COUNTS[dataset])
    folder = Path(model_root)
    soh_info = json.loads((folder / 'soh' / dataset.lower() / 'training_manifest.json')
                          .read_text(encoding='utf-8'))
    rul_info = json.loads((folder / 'rul' / dataset.lower() / 'training_manifest.json')
                          .read_text(encoding='utf-8'))
    soh_model = joblib.load(folder / 'soh' / dataset.lower() /
                            f"{soh_info['selected_model']}.joblib")
    rul_model = joblib.load(folder / 'rul' / dataset.lower() /
                            f"{rul_info['selected_model']}.joblib")
    if feature.matrix(soh_model['variant']).shape[1] != soh_model['imputer'].n_features_in_:
        raise ValueError('signal feature schema differs from the trained protocol')
    soh = np.asarray(predict_soh(soh_model, feature), dtype='float32')
    if rul_model['kind'] == 'assisted':
        rul_soh = joblib.load(folder / 'soh' / dataset.lower() / 'extra_trees_history.joblib')
        rul_input = predict_soh(rul_soh, feature)
    else:
        rul_input = None
    rul = np.asarray(predict_rul(rul_model, feature, rul_input), dtype='float32')
    soh[:feature.reference_count] = np.nan
    rul[:max(MIN_HISTORY, feature.reference_count)] = np.nan
    future_cycle = None
    future_soh = None
    if (future_strategy_relation != 'changed' and
            len(feature.cycle) > max(MIN_HISTORY, feature.reference_count)):
        rate_model = joblib.load(folder / 'trajectory' / dataset.lower() / 'rate_model.joblib')
        predicted_rate = predict_rate(rate_model, feature, predict_soh(soh_model, feature))
        horizons = np.array([0, 25, 50, 100], dtype='int64')
        future_cycle = feature.cycle[-1] + horizons
        future_soh = np.maximum(0, predict_trajectory(feature, soh, predicted_rate,
                                                       len(feature.cycle)-1, horizons))
    interval_info = json.loads((folder / 'development_intervals.json').read_text(encoding='utf-8'))
    soh_band = interval_info['tasks']['soh'][dataset]
    rul_band = interval_info['tasks']['rul'][dataset]
    soh_radius = soh_band['development_oof_radius']
    rul_radius = rul_band['development_oof_radius']
    applicability = (assess_xjtu_applicability(feature.raw) if dataset == 'XJTU' else None)
    if future_strategy_relation == 'same':
        future_status = 'CONDITIONAL: assumes the observed operating strategy continues'
        trajectory_status = ('EXPERIMENTAL: assumes unchanged operating strategy and '
                             'constant learned 50-cycle degradation rate; not the source '
                             'of selected RUL prediction')
    elif future_strategy_relation == 'changed':
        future_status = ('NOT APPLICABLE: model has no validated counterfactual for the '
                         'declared future strategy change; numeric RUL is diagnostic only')
        trajectory_status = 'NOT RETURNED: unchanged-strategy trajectory assumption is violated'
    else:
        future_status = ('UNSPECIFIED: RUL is an exploratory same-strategy counterfactual; '
                         'future operating strategy has not been confirmed')
        trajectory_status = ('EXPERIMENTAL SAME-STRATEGY COUNTERFACTUAL: future strategy '
                             'unspecified; do not interpret as an unconditional forecast')
    return {'cycle': feature.cycle.copy(), 'soh': soh, 'rul_cycles': rul,
            'soh_development_band_lower': np.maximum(0, soh-soh_radius),
            'soh_development_band_upper': soh+soh_radius,
            'rul_development_band_lower_cycles': np.maximum(0, rul-rul_radius),
            'rul_development_band_upper_cycles': rul+rul_radius,
            'future_cycle_experimental': future_cycle,
            'future_soh_experimental': future_soh,
            'future_trajectory_status': trajectory_status,
            'future_strategy_relation': future_strategy_relation,
            'rul_future_strategy_status': future_status,
            'soh_model': soh_info['selected_model'],
            'rul_model': rul_info['selected_model'],
            'development_band_design_quantile': .9,
            'interval_coverage_guarantee': None,
            'development_cross_split_coverage': {
                'soh': soh_band['cross_split_observation_coverage'],
                'rul': rul_band['cross_split_observation_coverage']},
            'development_worst_split_coverage': {
                'soh': soh_band['worst_split_coverage'],
                'rul': rul_band['worst_split_coverage']},
            'interval_status': 'NOT CALIBRATED: 90th-percentile development residual '
                               'radius is a design choice, not a 90% coverage guarantee; '
                               'some held-out groups have much lower coverage',
            'feature_applicability': applicability,
            'rul_reliability_status': (
                'NOT VALIDATED FOR NOVEL XJTU STRATEGIES; large held-out strategy errors remain'
                if dataset == 'XJTU' else
                'DEVELOPMENT ESTIMATE; independent validation pending') + '; ' + future_status,
            'warning': 'development models; require protocol-matched independent test'}
