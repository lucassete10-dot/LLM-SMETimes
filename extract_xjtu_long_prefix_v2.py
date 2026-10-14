"""Extract candidate XJTU 300/600-second prefixes without capacity labels.

The 120-second implementation is checked against every prepared training row
before the candidate features are saved. This is development-only extraction;
the frozen 120-second inference contract is unchanged.
"""

import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat


MANIFEST = Path('battery_data_v2/manifest.json')
OUT = Path('battery_models_v2/xjtu_long_prefix')
SECONDS = (120, 300, 600)
CHANNELS = ('voltage_V', 'current_A', 'temperature_C')


def stage_features(cycle, stage, seconds):
    grid = np.arange(0, seconds + 1, 30, dtype=float)
    time = np.asarray(cycle['relative_time_min'], dtype=float).reshape(-1)
    cuts = np.flatnonzero(time == 0)
    if len(cuts) <= stage:
        return None
    begin = cuts[stage]
    end = cuts[stage + 1] if stage + 1 < len(cuts) else len(time)
    t = 60 * time[begin:end]
    keep = np.isfinite(t) & (t <= grid[-1])
    if keep.sum() < 90 or t[keep][-1] < grid[-1] - 2:
        return None
    t = t[keep]
    order = np.argsort(t)
    unique, index = np.unique(t[order], return_index=True)
    if len(unique) < 90:
        return None
    arrays = []
    for channel in CHANNELS:
        values = np.asarray(cycle[channel], dtype=float).reshape(-1)[begin:end]
        values = values[keep][order][index]
        if not np.isfinite(values).all():
            return None
        arrays.append(np.interp(grid, unique, values))
    voltage, current, temperature = arrays
    extra = np.array([
        voltage[-1] - voltage[0], np.mean(voltage),
        (voltage[-1] - voltage[0]) / grid[-1],
        np.mean(current), np.std(current),
        temperature[-1] - temperature[0], np.mean(temperature),
        np.trapezoid(np.abs(current), grid) / 3600,
    ])
    return np.r_[np.concatenate(arrays), extra].astype('float32')


def extract():
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    records = [row for row in manifest['cells'] if row['dataset'] == 'XJTU']
    if len(records) != 55:
        raise ValueError('expected 55 XJTU training cells')
    arrays = {}
    report = []
    for record in records:
        key = record['cell_id']
        with np.load(record['feature_source']) as prepared:
            cycle_numbers = prepared['cycle'].astype('int64')
            baseline = prepared['features'].astype('float32')
        source = loadmat(record['measurement_source'], variable_names=['data'],
                         simplify_cells=True)
        if 'summary' in source or 'data' not in source:
            raise ValueError(f'{key}: source is not data-only')
        cycles = source['data']
        extracted = {window: [] for window in SECONDS}
        for number, expected in zip(cycle_numbers, baseline):
            current_cycle = cycles[int(number) - 1]
            for window in SECONDS:
                charge = stage_features(current_cycle, 0, window)
                discharge = stage_features(current_cycle, 2, window)
                if charge is None or discharge is None:
                    raise ValueError(f'{key} source cycle {number}: {window}s prefix unavailable')
                extracted[window].append(np.r_[charge, discharge])
            if not np.array_equal(extracted[120][-1], expected, equal_nan=True):
                raise ValueError(f'{key} source cycle {number}: 120s extractor mismatch')
        arrays[f'{key}_cycle'] = cycle_numbers
        for window in (300, 600):
            matrix = np.asarray(extracted[window], dtype='float32')
            width = 2 * (3 * (window // 30 + 1) + 8)
            if matrix.shape != (len(cycle_numbers), width):
                raise ValueError(f'{key}: incorrect {window}s feature shape')
            arrays[f'{key}_{window}s'] = matrix
        report.append({'cell_id': key, 'group': record['group'],
                       'diagnostics': len(cycle_numbers),
                       'baseline_120s_exact': True})
        print('EXTRACTED', key, len(cycle_numbers), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / 'features.npz', **arrays)
    (OUT / 'extraction_audit.json').write_text(json.dumps({
        'status': 'development-only signal features; no capacity/life MAT summary read',
        'windows_s': list(SECONDS), 'cells': report,
        'total_diagnostics': sum(row['diagnostics'] for row in report),
    }, indent=2), encoding='utf-8')
    print('WROTE', OUT / 'features.npz', len(report), 'cells', flush=True)


if __name__ == '__main__':
    extract()
