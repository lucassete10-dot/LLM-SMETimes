"""Signal-only historical usage features before each XJTU diagnostic.

Only the first 120 seconds of charge/discharge stages of *prior* source cycles
are read. No current diagnostic, full-discharge duration/throughput, voltage
integral, MAT summary, capacity, or lifetime label is used as a feature.
"""

import json
from pathlib import Path
import warnings

import numpy as np
from scipy.io import loadmat


MANIFEST = Path('battery_data_v2/manifest.json')
OUT = Path('battery_models_v2/xjtu_usage_history')
WINDOWS = (20, 100)
SIGNALS = ('charge_start_a', 'discharge_start_abs_a', 'temperature_start_c')
FEATURES_PER_WINDOW = (
    'charge_mean_a', 'charge_std_a', 'charge_p90_a',
    'discharge_mean_a', 'discharge_std_a', 'discharge_p90_a',
    'temperature_mean_c', 'temperature_std_c',
)


def early_stage_values(entry, stage):
    time = np.asarray(entry['relative_time_min'], dtype='float64').reshape(-1)
    cuts = np.flatnonzero(time == 0)
    if len(cuts) <= stage:
        return np.empty(0), np.empty(0)
    begin = cuts[stage]
    end = cuts[stage + 1] if stage + 1 < len(cuts) else len(time)
    relative = 60 * time[begin:end]
    early = np.isfinite(relative) & (relative >= 0) & (relative <= 120)
    current = np.asarray(entry['current_A'], dtype='float64').reshape(-1)[begin:end]
    temperature = np.asarray(entry['temperature_C'], dtype='float64').reshape(-1)[begin:end]
    return current[early], temperature[early]


def cycle_control(entry):
    charge, temperature = early_stage_values(entry, 0)
    discharge, _ = early_stage_values(entry, 2)
    charge = charge[np.isfinite(charge) & (charge > .2)]
    discharge = discharge[np.isfinite(discharge) & (discharge < -.2)]
    temperature = temperature[np.isfinite(temperature)]
    return np.asarray((
        float(np.median(charge)) if len(charge) else np.nan,
        float(np.median(-discharge)) if len(discharge) else np.nan,
        float(np.mean(temperature)) if len(temperature) else np.nan,
    ), dtype='float32')


def window_features(prior):
    if len(prior) == 0:
        return np.full(len(FEATURES_PER_WINDOW), np.nan, dtype='float32')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        charge, discharge, temperature = prior.T
        return np.asarray((
            np.nanmean(charge), np.nanstd(charge), np.nanpercentile(charge, 90),
            np.nanmean(discharge), np.nanstd(discharge), np.nanpercentile(discharge, 90),
            np.nanmean(temperature), np.nanstd(temperature),
        ), dtype='float32')


def history_features(controls, diagnostic_cycles):
    controls = np.asarray(controls, dtype='float32')
    cycle = np.asarray(diagnostic_cycles, dtype='int64')
    if controls.ndim != 2 or controls.shape[1] != 3:
        raise ValueError('expected three per-cycle control signals')
    if np.any(cycle < 1) or np.any(cycle > len(controls)) or np.any(np.diff(cycle) <= 0):
        raise ValueError('diagnostic source cycles are out of order or range')
    result = []
    for number in cycle:
        end = int(number)-1  # current diagnostic is never included
        result.append(np.concatenate([
            window_features(controls[max(0, end-window):end]) for window in WINDOWS]))
    return np.asarray(result, dtype='float32')


def extract():
    records = [row for row in json.loads(MANIFEST.read_text(encoding='utf-8'))['cells']
               if row['dataset'] == 'XJTU']
    if len(records) != 55:
        raise ValueError('expected 55 XJTU cells')
    arrays = {}
    report = []
    for record in records:
        key = record['cell_id']
        with np.load(record['feature_source']) as prepared:
            diagnostic_cycle = prepared['cycle'].astype('int64')
        mat = loadmat(record['measurement_source'], variable_names=['data'],
                      simplify_cells=True)
        if 'data' not in mat or 'summary' in mat:
            raise ValueError(f'{key}: MAT data-only load failed')
        controls = np.stack([cycle_control(entry) for entry in mat['data']])
        usage = history_features(controls, diagnostic_cycle)
        if usage.shape != (len(diagnostic_cycle), 16) or np.isinf(usage).any():
            raise ValueError(f'{key}: historical usage feature contract failed')
        arrays[f'{key}_cycle'] = diagnostic_cycle
        arrays[f'{key}_usage'] = usage
        report.append({'cell_id': key, 'group': record['group'],
                       'source_cycles': len(controls), 'diagnostics': len(diagnostic_cycle),
                       'finite_feature_fraction': float(np.isfinite(usage).mean()),
                       'missing_charge_cycles': int(np.isnan(controls[:, 0]).sum()),
                       'missing_discharge_cycles': int(np.isnan(controls[:, 1]).sum())})
        print('EXTRACTED', key, len(diagnostic_cycle), 'diagnostics', flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / 'features.npz', **arrays)
    (OUT / 'extraction_audit.json').write_text(json.dumps({
        'status': 'development-only, causal prior-cycle control signals; no target MAT summary read',
        'windows_source_cycles': list(WINDOWS),
        'signals': list(SIGNALS),
        'features_per_window': list(FEATURES_PER_WINDOW),
        'cells': report, 'total_diagnostics': sum(row['diagnostics'] for row in report),
    }, indent=2), encoding='utf-8')
    print('WROTE', OUT / 'features.npz', sum(row['diagnostics'] for row in report),
          'diagnostics', flush=True)


if __name__ == '__main__':
    extract()
