"""Two-stage, frozen-model external battery evaluation.

`predict` reads only label-free signal NPZ files. `score` later reads separate
capacity/endpoint labels and the immutable predictions. Both stages verify the
frozen candidate; neither trains or changes a model. Run from project root.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import tempfile

import numpy as np

from battery_data_contract import confirmed_threshold_event
from battery_v2_features import load_all, load_manifest
from freeze_battery_models_v2 import FREEZE, sha256, verify
from predict_battery_v2 import predict_series
from predict_xjtu_300_challenger_v2 import predict_series as predict_xjtu_300
from predict_xjtu_raw_rul_challenger_v2 import predict_series as predict_xjtu_raw_rul
from train_rul_v2 import MIN_HISTORY


FEATURE_SCHEMAS = {
    'NASA': ('nasa_discharge_prefix_1000s_v1', 44, 'observed discharge count', 1),
    'CALCE': ('calce_step7_prefix_1000s_v1', 29,
              'accepted/discarded discharge segment ordinal', 1),
    'MIT': ('mit_4c_discharge_prefix_300s_v1', 51, 'source cycle number', 1),
    'XJTU': ('xjtu_diagnostic_prefix_120s_v1', 46, 'source cycle number', 5),
}
LABEL_WORDS = ('capacity', 'soh', 'rul', 'endpoint', 'life')
MAX_OBSERVATIONS = 100000
XJTU_300_SCHEMA = 'xjtu_diagnostic_prefix_300s_v1'
XJTU_RAW_RUL_CHALLENGER = 'xjtu_raw_cycle_rul_v1'
DIAGNOSTIC_DISCHARGE_RATES_C = {'CALCE': (1.,), 'MIT': (4.,),
                                'XJTU': (.5, 1.)}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def resolve_file(manifest_path, value):
    path = Path(value)
    return (path if path.is_absolute() else Path(manifest_path).resolve().parent / path).resolve()


def valid_cell_id(cell_id):
    return (isinstance(cell_id, str) and 1 <= len(cell_id) <= 100 and
            cell_id not in ('.', '..') and re.fullmatch(r'[A-Za-z0-9_.-]+', cell_id))


def validate_diagnostic_rate(cell_id, domain, conditions):
    """Reject a stated protocol match that contradicts the training schema."""
    if domain not in DIAGNOSTIC_DISCHARGE_RATES_C:
        return
    rate = conditions['diagnostic_discharge_rate_c']
    expected = DIAGNOSTIC_DISCHARGE_RATES_C[domain]
    if rate is None or not any(np.isclose(rate, value, rtol=0, atol=.05)
                               for value in expected):
        raise ValueError(f'{cell_id}: {domain} diagnostic discharge rate must match '
                         f'one of {expected} C; cannot assert protocol match otherwise')


def signal_fingerprint(cycle, raw, prefix_only=False, raw_only=False):
    count = min(16, len(cycle)) if prefix_only else len(cycle)
    cycle_bytes = b'' if raw_only else np.ascontiguousarray(cycle[:count], dtype='<i8').tobytes()
    raw_bytes = np.ascontiguousarray(raw[:count], dtype='<f4').tobytes()
    return hashlib.sha256(cycle_bytes+raw_bytes).hexdigest()


def read_signal(path, domain, *, feature_schema=None):
    schema, dimension, _, reference_count = FEATURE_SCHEMAS[domain]
    if feature_schema is not None:
        if domain != 'XJTU' or feature_schema != XJTU_300_SCHEMA:
            raise ValueError('unsupported challenger feature schema')
        schema, dimension = XJTU_300_SCHEMA, 82
    if path.suffix.lower() != '.npz':
        raise ValueError(f'{path}: signal adapter must produce NPZ, not raw source file')
    with np.load(path, allow_pickle=False) as source:
        names = set(source.files)
        if any(any(word in name.lower() for word in LABEL_WORDS) for name in names):
            raise ValueError(f'{path}: signal file contains label-like arrays; keep labels separate')
        if names != {'raw_features', 'cycle'}:
            raise ValueError(f'{path}: expected only raw_features and cycle, got {sorted(names)}')
        raw = np.asarray(source['raw_features'], dtype='float32')
        cycle = np.asarray(source['cycle'], dtype='int64')
    if (raw.ndim != 2 or raw.shape[1] != dimension or cycle.ndim != 1 or
            len(raw) != len(cycle) or not reference_count < len(cycle) <= MAX_OBSERVATIONS):
        raise ValueError(f'{path}: invalid {schema} array dimensions')
    if np.any(np.diff(cycle) <= 0) or np.any(cycle < 1):
        raise ValueError(f'{path}: cycles must be positive and strictly increasing')
    if np.isinf(raw).any() or not np.isfinite(raw).any(axis=1).all():
        raise ValueError(f'{path}: infinite or wholly missing signal row')
    return cycle, raw


def validate_signal_manifest(path):
    manifest = read_json(path)
    if manifest.get('schema_version') != 1 or not isinstance(manifest.get('cells'), list):
        raise ValueError('external signal manifest needs schema_version 1 and cells list')
    if not manifest['cells']:
        raise ValueError('external signal manifest has no cells')
    training_ids = {row['cell_id'] for row in load_manifest()['cells']}
    seen = set()
    records = []
    for row in manifest['cells']:
        key = row.get('cell_id')
        domain = row.get('dataset')
        if any(any(word in name.lower() for word in LABEL_WORDS) for name in row):
            raise ValueError(f'{key}: signal manifest must not include capacity/end-of-life labels')
        if not valid_cell_id(key) or key in seen or key in training_ids:
            raise ValueError(f'invalid, duplicate, or training cell ID: {key}')
        seen.add(key)
        if domain not in FEATURE_SCHEMAS:
            raise ValueError(f'{key}: unsupported domain {domain}')
        schema, _, cycle_unit, _ = FEATURE_SCHEMAS[domain]
        if row.get('feature_schema') != schema or row.get('cycle_unit') != cycle_unit:
            raise ValueError(f'{key}: feature schema/cycle unit differs from trained protocol')
        if (not row.get('protocol_description') or
                row.get('diagnostic_protocol_match') is not True or
                row.get('aging_strategy_relation') not in ('known', 'novel')):
            raise ValueError(f'{key}: diagnostic protocol and aging-strategy relation must be declared')
        conditions = row.get('operating_conditions')
        expected_conditions = {'charge_rate_c', 'discharge_rate_c',
                               'diagnostic_discharge_rate_c', 'ambient_temperature_c',
                               'strategy'}
        if not isinstance(conditions, dict) or set(conditions) != expected_conditions:
            raise ValueError(f'{key}: operating_conditions must state rates, temperature, strategy')
        for field in expected_conditions - {'strategy'}:
            value = conditions[field]
            if value is not None and (not isinstance(value, (int, float)) or
                                      not np.isfinite(value) or value < 0):
                raise ValueError(f'{key}: invalid operating condition {field}')
        if not isinstance(conditions['strategy'], str) or not conditions['strategy'].strip():
            raise ValueError(f'{key}: strategy description is required')
        validate_diagnostic_rate(key, domain, conditions)
        future_relation = row.get('future_strategy_relation', 'unknown')
        if future_relation not in ('same', 'changed', 'unknown'):
            raise ValueError(f'{key}: future_strategy_relation must be same, changed, or unknown')
        future_description = row.get('future_strategy_description')
        if future_description is not None and (not isinstance(future_description, str) or
                                               not future_description.strip()):
            raise ValueError(f'{key}: future_strategy_description must be nonempty text')
        if future_relation == 'changed' and future_description is None:
            raise ValueError(f'{key}: changed future strategy requires a description')
        challengers = row.get('optional_challengers', [])
        if (not isinstance(challengers, list) or len(challengers) != len(set(
                str(value) for value in challengers)) or
                any(value != XJTU_RAW_RUL_CHALLENGER for value in challengers) or
                (domain != 'XJTU' and challengers)):
            raise ValueError(f'{key}: unsupported optional challenger')
        source_path = resolve_file(path, row['signal_file'])
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        cycle, raw = read_signal(source_path, domain)
        candidate = {'cell_id': key, 'dataset': domain, 'signal_file': str(source_path),
                        'signal_sha256': sha256(source_path),
                        'feature_fingerprint': signal_fingerprint(cycle, raw),
                        'prefix_fingerprint': signal_fingerprint(cycle, raw, True),
                        'raw_prefix_fingerprint': signal_fingerprint(cycle, raw, True, True),
                        'observations': len(cycle),
                        'protocol_description': row['protocol_description'],
                        'diagnostic_protocol_match': True,
                        'aging_strategy_relation': row['aging_strategy_relation'],
                        'operating_conditions': conditions,
                        'optional_challengers': challengers,
                        'future_strategy_relation': future_relation,
                        'future_strategy_description': future_description}
        if 'challenger_signal_file' in row:
            if domain != 'XJTU':
                raise ValueError(f'{key}: 300s challenger is XJTU-only')
            challenger_path = resolve_file(path, row['challenger_signal_file'])
            if not challenger_path.is_file() or challenger_path == source_path:
                raise ValueError(f'{key}: separate 300s challenger signal file required')
            challenger_cycle, challenger_raw = read_signal(
                challenger_path, domain, feature_schema=XJTU_300_SCHEMA)
            if not np.array_equal(challenger_cycle, cycle):
                raise ValueError(f'{key}: 120s and 300s diagnostic cycles must align exactly')
            candidate.update({
                'challenger_signal_file': str(challenger_path),
                'challenger_signal_sha256': sha256(challenger_path),
                'challenger_feature_schema': XJTU_300_SCHEMA,
                'challenger_raw_prefix_fingerprint': signal_fingerprint(
                    challenger_cycle, challenger_raw, True, True),
            })
        records.append(candidate)
    for field in ('signal_file', 'feature_fingerprint', 'prefix_fingerprint',
                  'raw_prefix_fingerprint'):
        if len({row[field] for row in records}) != len(records):
            raise ValueError(f'external manifest reuses a signal or signal prefix: {field}')
    challenger_paths = [row['challenger_signal_file'] for row in records
                        if 'challenger_signal_file' in row]
    if len(challenger_paths) != len(set(challenger_paths)):
        raise ValueError('external manifest reuses a 300s challenger signal file')
    challenger_prefixes = [row['challenger_raw_prefix_fingerprint'] for row in records
                           if 'challenger_raw_prefix_fingerprint' in row]
    if len(challenger_prefixes) != len(set(challenger_prefixes)):
        raise ValueError('external manifest reuses a 300s challenger signal prefix')
    return manifest, records


def reject_training_overlap(records):
    _, training_cells = load_all()
    full = {signal_fingerprint(cell.cycle, cell.raw): key
            for key, cell in training_cells.items()}
    prefix = {signal_fingerprint(cell.cycle, cell.raw, True): key
              for key, cell in training_cells.items()}
    raw_prefix = {signal_fingerprint(cell.cycle, cell.raw, True, True): key
                  for key, cell in training_cells.items()}
    with np.load(Path('battery_models_v2/xjtu_long_prefix/features.npz')) as archive:
        challenger_prefix = {
            signal_fingerprint(archive[f'{key}_cycle'], archive[f'{key}_300s'], True, True): key
            for key, cell in training_cells.items() if cell.dataset == 'XJTU'}
    for row in records:
        if row['feature_fingerprint'] in full:
            raise ValueError(f"{row['cell_id']}: same signals as training cell "
                             f"{full[row['feature_fingerprint']]}")
        if row['prefix_fingerprint'] in prefix:
            raise ValueError(f"{row['cell_id']}: same first 16 observations as training cell "
                             f"{prefix[row['prefix_fingerprint']]}")
        if row['raw_prefix_fingerprint'] in raw_prefix:
            raise ValueError(f"{row['cell_id']}: same first 16 raw signal rows as training cell "
                             f"{raw_prefix[row['raw_prefix_fingerprint']]}")
        if row.get('challenger_raw_prefix_fingerprint') in challenger_prefix:
            raise ValueError(f"{row['cell_id']}: same first 16 challenger signal rows as "
                             f"training cell {challenger_prefix[row['challenger_raw_prefix_fingerprint']]}")


def verify_blind_signal_hashes(records):
    for row in records:
        if sha256(row['signal_file']) != row['signal_sha256']:
            raise ValueError(f"{row['cell_id']}: signal differs from blind-stage hash")
        if ('challenger_signal_file' in row and
                sha256(row['challenger_signal_file']) != row['challenger_signal_sha256']):
            raise ValueError(f"{row['cell_id']}: challenger signal differs from blind-stage hash")


def predict(manifest_path, output_path):
    verify()
    frozen_hash = sha256(FREEZE)
    manifest_path = Path(manifest_path).resolve()
    manifest_hash = sha256(manifest_path)
    output_path = Path(output_path).resolve()
    if (output_path == Path.cwd().resolve() or
            output_path.is_relative_to(Path('battery_models_v2').resolve()) or
            output_path.is_relative_to(Path('battery_data_v2').resolve())):
        raise ValueError('prediction output must be outside model/data directories')
    if output_path.exists():
        raise FileExistsError(f'prediction output already exists: {output_path}')
    _, records = validate_signal_manifest(manifest_path)
    if sha256(manifest_path) != manifest_hash:
        raise ValueError('signal manifest changed during blind validation')
    reject_training_overlap(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{output_path.name}.staging-',
                                    dir=output_path.parent))
    predictions = staging / 'predictions'
    predictions.mkdir()
    output_rows = []
    for row in records:
        verify_blind_signal_hashes([row])
        cycle, raw = read_signal(Path(row['signal_file']), row['dataset'])
        result = predict_series(row['dataset'], raw, cycle,
                                future_strategy_relation=row['future_strategy_relation'])
        destination = predictions / f"{row['cell_id']}.npz"
        saved = {
            'cycle': result['cycle'], 'soh': result['soh'],
            'rul_cycles': result['rul_cycles'],
            'soh_band_lower': result['soh_development_band_lower'],
            'soh_band_upper': result['soh_development_band_upper'],
            'rul_band_lower': result['rul_development_band_lower_cycles'],
            'rul_band_upper': result['rul_development_band_upper_cycles'],
            'future_cycle': (result['future_cycle_experimental'] if
                             result['future_cycle_experimental'] is not None else
                             np.array([], dtype='int64')),
            'future_soh': (result['future_soh_experimental'] if
                           result['future_soh_experimental'] is not None else
                           np.array([], dtype='float32')),
        }
        if 'challenger_signal_file' in row:
            challenger_cycle, challenger_raw = read_signal(
                Path(row['challenger_signal_file']), 'XJTU',
                feature_schema=XJTU_300_SCHEMA)
            challenger = predict_xjtu_300(challenger_raw, challenger_cycle)
            if not np.array_equal(challenger['cycle'], cycle):
                raise ValueError(f"{row['cell_id']}: challenger prediction cycle mismatch")
            saved['soh_challenger_300'] = challenger['soh']
        if XJTU_RAW_RUL_CHALLENGER in row['optional_challengers']:
            challenger = predict_xjtu_raw_rul(raw, cycle)
            if not np.array_equal(challenger['cycle'], cycle):
                raise ValueError(f"{row['cell_id']}: raw RUL challenger cycle mismatch")
            saved['rul_challenger_raw_target'] = challenger['rul_cycles']
        np.savez_compressed(destination, **saved)
        final_destination = output_path / 'predictions' / f"{row['cell_id']}.npz"
        output_rows.append({**row, 'prediction_file': str(final_destination),
                            'prediction_sha256': sha256(destination),
                            'soh_model': result['soh_model'], 'rul_model': result['rul_model'],
                            'feature_applicability': result['feature_applicability'],
                            'rul_reliability_status': result['rul_reliability_status'],
                            'rul_future_strategy_status': result['rul_future_strategy_status'],
                            'future_trajectory_status': result['future_trajectory_status'],
                            **({'soh_challenger_model': 'extra_trees_300s_challenger'}
                               if 'challenger_signal_file' in row else {}),
                            **({'rul_challenger_model': 'raw_cycle_target_challenger'}
                               if XJTU_RAW_RUL_CHALLENGER in row['optional_challengers']
                               else {})})
        print('PREDICTED', row['cell_id'], row['dataset'], len(cycle), flush=True)
    verify_blind_signal_hashes(records)
    if sha256(manifest_path) != manifest_hash:
        raise ValueError('signal manifest changed during blind prediction')
    verify()
    if sha256(FREEZE) != frozen_hash:
        raise ValueError('freeze manifest changed during blind prediction')
    audit = {'schema_version': 1, 'stage': 'frozen_prediction_before_labels',
             'created_utc': datetime.now(timezone.utc).isoformat(),
             'signal_manifest': str(manifest_path),
             'signal_manifest_sha256': manifest_hash,
             'freeze_manifest_sha256': frozen_hash,
             'cells': output_rows,
             'warning': 'feature-schema and protocol assertions require source-file audit; '
                        'novel aging strategies are out of distribution; '
                        'operating conditions are recorded but not separate frozen-model inputs; '
                        'changed future strategies make RUL diagnostic only; '
                        'development intervals are not calibrated'}
    (staging / 'prediction_manifest.json').write_text(json.dumps(audit, indent=2),
                                                     encoding='utf-8')
    if output_path.exists():
        raise FileExistsError(f'prediction output appeared during staging: {output_path}; '
                              f'incomplete staging retained at {staging}')
    staging.rename(output_path)
    print('SAVED', output_path / 'prediction_manifest.json', flush=True)


def read_labels(path, domain):
    if path.suffix.lower() != '.npz':
        raise ValueError(f'{path}: labels must be NPZ')
    with np.load(path, allow_pickle=False) as source:
        if set(source.files) != {'cycle', 'capacity_ah'}:
            raise ValueError(f'{path}: labels need only cycle and capacity_ah')
        cycle = np.asarray(source['cycle'], dtype='int64')
        capacity = np.asarray(source['capacity_ah'], dtype='float64')
    if (cycle.ndim != 1 or capacity.ndim != 1 or len(cycle) != len(capacity) or
            len(cycle) <= FEATURE_SCHEMAS[domain][3] or np.any(np.diff(cycle) <= 0) or
            not np.isfinite(capacity).all() or np.any(capacity <= 0)):
        raise ValueError(f'{path}: invalid capacity labels')
    return cycle, capacity


def metric_summary(rows, name):
    values = [row[name] for row in rows if row.get(name) is not None]
    return float(np.mean(values)) if values else None


def summarize_scored_cells(rows):
    soh_rows = [row['soh'] for row in rows]
    rul_rows = [row['rul'] for row in rows if row['rul'] is not None]
    return {'cells': len(rows), 'rul_observed_cells': len(rul_rows),
        'soh_cell_macro_mae': metric_summary(soh_rows, 'mae'),
        'soh_cell_macro_signed_bias': metric_summary(soh_rows, 'signed_bias'),
        'soh_worst_cell_mae': max((row['mae'] for row in soh_rows), default=None),
        'soh_cell_macro_band_coverage': metric_summary(soh_rows, 'development_band_coverage'),
        'rul_cell_macro_mae_cycles': metric_summary(rul_rows, 'mae_cycles'),
        'rul_cell_macro_signed_bias_cycles': metric_summary(rul_rows, 'signed_bias_cycles'),
        'rul_worst_cell_mae_cycles': max((row['mae_cycles'] for row in rul_rows), default=None),
        'rul_cell_macro_band_coverage': metric_summary(rul_rows, 'development_band_coverage'),
        'rul_phase_macro_mae_cycles': {phase: metric_summary(
            [row['phase_mae_cycles'] for row in rul_rows], phase)
            for phase in ('early', 'middle', 'late')}}


def paired_soh_summary(rows, include_strata=True):
    """Compare the two frozen XJTU SOH candidates using cells as paired units."""
    paired = sorted(rows, key=lambda row: row['cell_id'])
    baseline = np.asarray([row['soh']['mae'] for row in paired], dtype='float64')
    challenger = np.asarray([row['soh_challenger_300']['mae'] for row in paired],
                            dtype='float64')
    delta = challenger - baseline  # Negative means the 300s challenger is better.
    tolerance = 1e-12
    result = {
        'cells': len(paired),
        'baseline_120s_cell_macro_mae': float(np.mean(baseline)),
        'challenger_300s_cell_macro_mae': float(np.mean(challenger)),
        'mae_delta_challenger_minus_baseline': float(np.mean(delta)),
        'cells_improved': int(np.sum(delta < -tolerance)),
        'cells_worsened': int(np.sum(delta > tolerance)),
        'cells_tied': int(np.sum(np.abs(delta) <= tolerance)),
        'per_cell_mae_delta': {row['cell_id']: float(value)
                               for row, value in zip(paired, delta)},
        'baseline_120s_cell_macro_signed_bias': metric_summary(
            [row['soh'] for row in paired], 'signed_bias'),
        'challenger_300s_cell_macro_signed_bias': metric_summary(
            [row['soh_challenger_300'] for row in paired], 'signed_bias'),
        'baseline_120s_worst_cell_mae': float(np.max(baseline)),
        'challenger_300s_worst_cell_mae': float(np.max(challenger)),
        'cell_bootstrap_95pct_mae_delta_interval': None,
        'cell_bootstrap_status': 'INSUFFICIENT_CELLS',
        'warning': ('Paired 300s SOH challenger only; RUL and intervals use the 120s '
                    'candidate. Cell bootstrap is descriptive, not proof of transfer '
                    'to unseen strategies or an automatic model-selection rule.'),
    }
    if len(paired) >= 8:
        rng = np.random.default_rng(2026)
        indices = rng.integers(0, len(delta), size=(2000, len(delta)))
        means = np.mean(delta[indices], axis=1)
        result['cell_bootstrap_95pct_mae_delta_interval'] = [
            float(bound) for bound in np.quantile(means, [.025, .975])]
        result['cell_bootstrap_status'] = 'DESCRIPTIVE_ONLY'
    if include_strata:
        result['by_aging_strategy_relation'] = {
            relation: paired_soh_summary(
                [row for row in paired if row['aging_strategy_relation'] == relation],
                include_strata=False)
            for relation in ('known', 'novel')
            if any(row['aging_strategy_relation'] == relation for row in paired)}
    return result


def paired_rul_summary(rows, include_strata=True):
    """Compare RUL variants on the same observed-endpoint cells, never cycles."""
    paired = sorted(rows, key=lambda row: row['cell_id'])
    baseline = np.asarray([row['rul']['mae_cycles'] for row in paired], dtype='float64')
    challenger = np.asarray([row['rul_challenger_raw_target']['mae_cycles']
                             for row in paired], dtype='float64')
    delta = challenger-baseline  # Negative favors the raw-target challenger.
    result = {
        'cells': len(paired),
        'baseline_log_target_cell_macro_mae_cycles': float(np.mean(baseline)),
        'challenger_raw_target_cell_macro_mae_cycles': float(np.mean(challenger)),
        'mae_delta_challenger_minus_baseline_cycles': float(np.mean(delta)),
        'cells_improved': int(np.sum(delta < -1e-12)),
        'cells_worsened': int(np.sum(delta > 1e-12)),
        'cells_tied': int(np.sum(np.abs(delta) <= 1e-12)),
        'per_cell_mae_delta_cycles': {row['cell_id']: float(value)
                                      for row, value in zip(paired, delta)},
        'baseline_log_target_cell_macro_signed_bias_cycles': metric_summary(
            [row['rul'] for row in paired], 'signed_bias_cycles'),
        'challenger_raw_target_cell_macro_signed_bias_cycles': metric_summary(
            [row['rul_challenger_raw_target'] for row in paired], 'signed_bias_cycles'),
        'baseline_log_target_worst_cell_mae_cycles': float(np.max(baseline)),
        'challenger_raw_target_worst_cell_mae_cycles': float(np.max(challenger)),
        'phase_cell_macro_mae_cycles': {
            name: {phase: metric_summary(
                [row[field]['phase_mae_cycles'] for row in paired], phase)
                for phase in ('early', 'middle', 'late')}
            for name, field in (('baseline_log_target', 'rul'),
                                ('challenger_raw_target', 'rul_challenger_raw_target'))},
        'cell_bootstrap_95pct_mae_delta_interval_cycles': None,
        'cell_bootstrap_status': 'INSUFFICIENT_CELLS',
        'warning': ('The raw-target RUL challenger has no calibrated interval and '
                    'does not replace the default. Both RUL estimates are diagnostic '
                    'only when the future operating strategy changes. A cell bootstrap '
                    'is descriptive, not a new-strategy transfer guarantee.'),
    }
    if len(paired) >= 8:
        rng = np.random.default_rng(2026)
        indices = rng.integers(0, len(delta), size=(2000, len(delta)))
        means = np.mean(delta[indices], axis=1)
        result['cell_bootstrap_95pct_mae_delta_interval_cycles'] = [
            float(bound) for bound in np.quantile(means, [.025, .975])]
        result['cell_bootstrap_status'] = 'DESCRIPTIVE_ONLY'
    if include_strata:
        result['by_aging_strategy_relation'] = {
            relation: paired_rul_summary([
                row for row in paired if row['aging_strategy_relation'] == relation], False)
            for relation in ('known', 'novel')
            if any(row['aging_strategy_relation'] == relation for row in paired)}
        result['by_future_strategy_relation'] = {
            relation: paired_rul_summary([
                row for row in paired if row['future_strategy_relation'] == relation], False)
            for relation in ('same', 'changed', 'unknown')
            if any(row['future_strategy_relation'] == relation for row in paired)}
    return result


def score_cell(prediction, label_cycle, capacity, domain, endpoint):
    cycle = prediction['cycle']
    ref_count = FEATURE_SCHEMAS[domain][3]
    if not np.array_equal(label_cycle[:ref_count], cycle[:ref_count]):
        raise ValueError('calibration labels must align with the first signal observations')
    positions = np.searchsorted(cycle, label_cycle)
    if np.any(positions >= len(cycle)) or not np.array_equal(cycle[positions], label_cycle):
        raise ValueError('every capacity label must align with a signal observation')
    reference = float(np.median(capacity[:ref_count]))
    target_soh = capacity/reference
    valid_soh = np.arange(len(label_cycle)) >= ref_count
    p = positions[valid_soh]
    for name in ('soh', 'soh_band_lower', 'soh_band_upper'):
        if not np.isfinite(prediction[name][p]).all():
            raise ValueError(f'nonfinite scored SOH {name}')
    error = prediction['soh'][p]-target_soh[valid_soh]
    covered = ((target_soh[valid_soh] >= prediction['soh_band_lower'][p]) &
               (target_soh[valid_soh] <= prediction['soh_band_upper'][p]))
    soh_result = {'observations': int(len(p)), 'mae': float(np.mean(np.abs(error))),
                  'signed_bias': float(np.mean(error)),
                  'rmse': float(np.sqrt(np.mean(error**2))),
                  'worst_absolute_error': float(np.max(np.abs(error))),
                  'development_band_coverage': float(np.mean(covered))}
    rul_result = None
    if endpoint is not None:
        valid_rul = (np.arange(len(cycle)) >= MIN_HISTORY) & (cycle <= endpoint)
        p = np.flatnonzero(valid_rul)
        if len(p):
            for name in ('rul_cycles', 'rul_band_lower', 'rul_band_upper'):
                if not np.isfinite(prediction[name][p]).all():
                    raise ValueError(f'nonfinite scored RUL {name}')
            target_rul = endpoint-cycle[p]
            error = prediction['rul_cycles'][p]-target_rul
            covered = ((target_rul >= prediction['rul_band_lower'][p]) &
                       (target_rul <= prediction['rul_band_upper'][p]))
            progress = cycle[p]/endpoint
            phases = {'early': progress < .33,
                      'middle': (progress >= .33) & (progress < .67),
                      'late': progress >= .67}
            rul_result = {'observations': int(len(p)),
                          'mae_cycles': float(np.mean(np.abs(error))),
                          'signed_bias_cycles': float(np.mean(error)),
                          'worst_absolute_error_cycles': float(np.max(np.abs(error))),
                          'development_band_coverage': float(np.mean(covered)),
                          'phase_mae_cycles': {phase: float(np.mean(np.abs(error[mask])))
                                               if mask.any() else None
                                               for phase, mask in phases.items()}}
    return reference, soh_result, rul_result


def score_soh_challenger(prediction, label_cycle, capacity):
    cycle = prediction['cycle']
    if (prediction['soh_challenger_300'].ndim != 1 or
            len(prediction['soh_challenger_300']) != len(cycle)):
        raise ValueError('300s challenger prediction shape mismatch')
    if not np.array_equal(label_cycle[:5], cycle[:5]):
        raise ValueError('challenger calibration labels must align with first five signals')
    positions = np.searchsorted(cycle, label_cycle)
    if np.any(positions >= len(cycle)) or not np.array_equal(cycle[positions], label_cycle):
        raise ValueError('challenger capacity labels must align with signal cycles')
    estimate = prediction['soh_challenger_300'][positions[5:]]
    if not np.isfinite(estimate).all():
        raise ValueError('nonfinite scored 300s challenger SOH')
    target = capacity[5:] / float(np.median(capacity[:5]))
    error = estimate-target
    return {'observations': len(error), 'mae': float(np.mean(np.abs(error))),
            'signed_bias': float(np.mean(error)),
            'rmse': float(np.sqrt(np.mean(error**2))),
            'worst_absolute_error': float(np.max(np.abs(error))),
            'interval_status': 'NO CALIBRATED INTERVAL'}


def score_rul_challenger(prediction, endpoint):
    cycle = prediction['cycle']
    estimate = prediction['rul_challenger_raw_target']
    if estimate.ndim != 1 or len(estimate) != len(cycle):
        raise ValueError('raw-target RUL challenger prediction shape mismatch')
    if not np.isfinite(estimate[MIN_HISTORY:]).all():
        raise ValueError('nonfinite raw-target RUL challenger after minimum history')
    if endpoint is None:
        return None  # Right-censored cells do not get fabricated RUL targets.
    p = np.flatnonzero((np.arange(len(cycle)) >= MIN_HISTORY) & (cycle <= endpoint))
    if not len(p):
        return None
    target = endpoint-cycle[p]
    error = estimate[p]-target
    progress = cycle[p]/endpoint
    phases = {'early': progress < .33,
              'middle': (progress >= .33) & (progress < .67),
              'late': progress >= .67}
    return {'observations': int(len(p)),
            'mae_cycles': float(np.mean(np.abs(error))),
            'signed_bias_cycles': float(np.mean(error)),
            'worst_absolute_error_cycles': float(np.max(np.abs(error))),
            'phase_mae_cycles': {phase: float(np.mean(np.abs(error[mask])))
                                 if mask.any() else None for phase, mask in phases.items()},
            'interval_status': 'NO CALIBRATED INTERVAL'}


def score(prediction_dir, labels_manifest_path, output_path):
    verify()
    frozen_hash = sha256(FREEZE)
    prediction_dir = Path(prediction_dir).resolve()
    labels_manifest_path = Path(labels_manifest_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    prediction_manifest_path = prediction_dir / 'prediction_manifest.json'
    prediction_manifest_hash = sha256(prediction_manifest_path)
    audit = read_json(prediction_manifest_path)
    if sha256(prediction_manifest_path) != prediction_manifest_hash:
        raise ValueError('blind prediction manifest changed while scoring')
    if audit.get('stage') != 'frozen_prediction_before_labels':
        raise ValueError('prediction directory lacks completed blind-stage manifest')
    if audit['freeze_manifest_sha256'] != sha256(FREEZE):
        raise ValueError('model freeze differs from the blind prediction stage')
    if audit['signal_manifest_sha256'] != sha256(audit['signal_manifest']):
        raise ValueError('signal manifest changed after blind prediction')
    labels_manifest_hash = sha256(labels_manifest_path)
    labels = read_json(labels_manifest_path)
    if sha256(labels_manifest_path) != labels_manifest_hash:
        raise ValueError('label manifest changed while scoring')
    if (labels.get('schema_version') != 1 or labels.get('capacity_unit') != 'Ah' or
            not isinstance(labels.get('cells'), list)):
        raise ValueError('label manifest requires schema_version 1, capacity_unit Ah, cells')
    by_id = {row['cell_id']: row for row in labels['cells']}
    if len(by_id) != len(labels['cells']) or set(by_id) != {row['cell_id'] for row in audit['cells']}:
        raise ValueError('label manifest must cover exactly the blind-predicted cells')
    label_paths = [resolve_file(labels_manifest_path, row['label_file'])
                   for row in labels['cells']]
    if len(set(label_paths)) != len(label_paths):
        raise ValueError('label manifest reuses one capacity file for multiple cells')
    reports = []
    label_hashes = {}
    for row in audit['cells']:
        key, domain = row['cell_id'], row['dataset']
        if sha256(row['signal_file']) != row['signal_sha256'] or \
                sha256(row['prediction_file']) != row['prediction_sha256']:
            raise ValueError(f'{key}: signal or prediction changed after blind stage')
        if ('challenger_signal_file' in row and
                sha256(row['challenger_signal_file']) != row['challenger_signal_sha256']):
            raise ValueError(f'{key}: 300s challenger signal changed after blind stage')
        label = by_id[key]
        if (label.get('complete_comparable_capacity_diagnostics') is not True or
                not label.get('capacity_protocol_description')):
            raise ValueError(f'{key}: capacity labels must be comparable full diagnostics')
        label_file = resolve_file(labels_manifest_path, label['label_file'])
        if label_file.resolve() == Path(row['signal_file']).resolve():
            raise ValueError(f'{key}: labels must be separate from signals')
        if ('challenger_signal_file' in row and
                label_file.resolve() == Path(row['challenger_signal_file']).resolve()):
            raise ValueError(f'{key}: labels must be separate from challenger signals')
        label_hash = sha256(label_file)
        label_cycle, capacity = read_labels(label_file, domain)
        if sha256(label_file) != label_hash:
            raise ValueError(f'{key}: capacity label file changed while scoring')
        label_hashes[label_file] = label_hash
        with np.load(row['prediction_file'], allow_pickle=False) as source:
            prediction = {name: source[name] for name in source.files}
        if sha256(row['prediction_file']) != row['prediction_sha256']:
            raise ValueError(f'{key}: prediction file changed while scoring')
        if domain in ('NASA', 'CALCE'):
            if label.get('endpoint_rule') != 'derived_80pct_sustained':
                raise ValueError(f'{key}: wrong endpoint rule')
            reference = float(np.median(capacity[:FEATURE_SCHEMAS[domain][3]]))
            endpoint = confirmed_threshold_event(label_cycle, capacity, .8*reference)
        elif domain == 'MIT':
            if label.get('endpoint_rule') != 'mit_0p88ah_source':
                raise ValueError(f'{key}: wrong endpoint rule')
            if not isinstance(label.get('endpoint_observed'), bool):
                raise ValueError(f'{key}: MIT endpoint_observed must be explicit boolean')
            observed = label.get('endpoint_observed') is True
            endpoint = int(label['endpoint_cycle']) if observed else None
            if not observed and label.get('endpoint_cycle') is not None:
                raise ValueError(f'{key}: censored cell cannot carry endpoint_cycle')
            if observed and capacity[-1] > .885:
                raise ValueError(f'{key}: MIT endpoint is above 0.885 Ah; mark censored')
        else:
            if label.get('endpoint_rule') != 'xjtu_source_cycle_life':
                raise ValueError(f'{key}: wrong endpoint rule')
            if not isinstance(label.get('endpoint_observed'), bool):
                raise ValueError(f'{key}: XJTU endpoint_observed must be explicit boolean')
            observed = label.get('endpoint_observed') is True
            endpoint = int(label['endpoint_cycle']) if observed else None
            if not observed and label.get('endpoint_cycle') is not None:
                raise ValueError(f'{key}: censored cell cannot carry endpoint_cycle')
        if endpoint is not None and endpoint < int(label_cycle[0]):
            raise ValueError(f'{key}: endpoint precedes observations')
        reference, soh_result, rul_result = score_cell(prediction, label_cycle,
                                                       capacity, domain, endpoint)
        challenger_result = None
        if 'challenger_signal_file' in row:
            if 'soh_challenger_300' not in prediction:
                raise ValueError(f'{key}: missing blind 300s challenger prediction')
            challenger_result = score_soh_challenger(prediction, label_cycle, capacity)
        raw_rul_requested = XJTU_RAW_RUL_CHALLENGER in row.get('optional_challengers', [])
        if raw_rul_requested != ('rul_challenger_raw_target' in prediction):
            raise ValueError(f'{key}: raw-target RUL challenger blind prediction mismatch')
        raw_rul_result = (score_rul_challenger(prediction, endpoint)
                          if raw_rul_requested else None)
        reports.append({'cell_id': key, 'dataset': domain,
                        'aging_strategy_relation': row['aging_strategy_relation'],
                        'future_strategy_relation': row.get('future_strategy_relation', 'unknown'),
                        'feature_applicability': row.get('feature_applicability'),
                        'label_file_sha256': label_hash,
                        'reference_capacity_ah': reference,
                        'endpoint_cycle': endpoint, 'endpoint_observed': endpoint is not None,
                        'soh': soh_result, 'rul': rul_result,
                        **({'soh_challenger_300': challenger_result}
                           if challenger_result is not None else {}),
                        **({'rul_challenger_raw_target': raw_rul_result}
                           if raw_rul_requested else {})})
    per_domain = {}
    for domain in FEATURE_SCHEMAS:
        selected = [row for row in reports if row['dataset'] == domain]
        if not selected:
            continue
        per_domain[domain] = summarize_scored_cells(selected)
        per_domain[domain]['by_aging_strategy_relation'] = {
            relation: summarize_scored_cells([row for row in selected
                                             if row['aging_strategy_relation'] == relation])
            for relation in ('known', 'novel')
            if any(row['aging_strategy_relation'] == relation for row in selected)}
        per_domain[domain]['by_future_strategy_relation'] = {
            relation: summarize_scored_cells([
                row for row in selected if row['future_strategy_relation'] == relation])
            for relation in ('same', 'changed', 'unknown')
            if any(row['future_strategy_relation'] == relation for row in selected)}
        paired = [row for row in selected if 'soh_challenger_300' in row]
        if paired:
            per_domain[domain]['paired_300s_challenger'] = paired_soh_summary(paired)
        raw_rul_requested = [row for row in selected
                             if 'rul_challenger_raw_target' in row]
        if raw_rul_requested:
            observed_pair = [row for row in raw_rul_requested
                             if row['rul'] is not None and
                             row['rul_challenger_raw_target'] is not None]
            per_domain[domain]['raw_target_rul_challenger_requested_cells'] = len(
                raw_rul_requested)
            per_domain[domain]['raw_target_rul_challenger_unscored_cells'] = (
                len(raw_rul_requested)-len(observed_pair))
            if observed_pair:
                per_domain[domain]['paired_raw_target_rul_challenger'] = paired_rul_summary(
                    observed_pair)
        if domain == 'XJTU':
            statuses = sorted({row['feature_applicability']['status'] for row in selected
                               if row.get('feature_applicability') is not None})
            per_domain[domain]['by_feature_applicability'] = {
                status: summarize_scored_cells([
                    row for row in selected
                    if row.get('feature_applicability', {}).get('status') == status])
                for status in statuses}
    if (sha256(prediction_manifest_path) != prediction_manifest_hash or
            sha256(labels_manifest_path) != labels_manifest_hash):
        raise ValueError('prediction or label manifest changed while scoring')
    for path, expected in label_hashes.items():
        if sha256(path) != expected:
            raise ValueError(f'capacity label file changed while scoring: {path}')
    verify_blind_signal_hashes(audit['cells'])
    for row in audit['cells']:
        if sha256(row['prediction_file']) != row['prediction_sha256']:
            raise ValueError(f"{row['cell_id']}: prediction file changed while scoring")
    verify()
    if sha256(FREEZE) != frozen_hash:
        raise ValueError('freeze manifest changed while scoring')
    result = {'schema_version': 1, 'stage': 'first_frozen_external_score',
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'prediction_manifest_sha256': prediction_manifest_hash,
              'labels_manifest_sha256': labels_manifest_hash,
              'freeze_manifest_sha256': frozen_hash,
              'domains': per_domain, 'cells': reports,
              'warning': 'development bands are not calibrated; evaluate observed coverage here; '
                         'RUL under changed future strategies is diagnostic only; '
                         'do not adapt models before preserving this report'}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False,
                                     prefix=f'.{output_path.stem}.staging-', suffix='.tmp',
                                     dir=output_path.parent) as stream:
        staged = Path(stream.name)
        json.dump(result, stream, indent=2)
    if output_path.exists():
        raise FileExistsError(f'score output appeared during staging: {output_path}; '
                              f'incomplete staging retained at {staged}')
    staged.rename(output_path)
    print('SCORED', output_path, flush=True)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    predict_parser = commands.add_parser('predict')
    predict_parser.add_argument('manifest')
    predict_parser.add_argument('output_dir')
    score_parser = commands.add_parser('score')
    score_parser.add_argument('prediction_dir')
    score_parser.add_argument('labels_manifest')
    score_parser.add_argument('output_json')
    args = parser.parse_args()
    if args.command == 'predict':
        predict(args.manifest, args.output_dir)
    else:
        score(args.prediction_dir, args.labels_manifest, args.output_json)


if __name__ == '__main__':
    main()
