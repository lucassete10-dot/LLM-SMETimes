"""Prepare leakage-controlled XJTU diagnostic-cycle examples from original MAT files.

Only discharge/charge signals observed in the first 120 seconds are inputs.
Capacity labels come from comparable full-discharge diagnostic protocols; partial
discharges are never treated as capacity/SOH ground truth.
"""
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat


SOURCE = Path(r'D:\LLM\XJTU\raw\extracted\Battery Dataset')
OUT = Path(r'D:\LLM\XJTU\processed')
GRID = np.arange(0, 121, 30, dtype=float)
CHANNELS = ('voltage_V', 'current_A', 'temperature_C')
STAGE_EXTRA_NAMES = ('voltage_drop_v', 'voltage_mean_v', 'voltage_slope_v_s',
                     'current_mean_a', 'current_std_a', 'temperature_change_c',
                     'temperature_mean_c', 'early_throughput_ah')
FEATURE_NAMES = []
for stage in ('charge', 'discharge'):
    # stage_features() concatenates each stage's 15 sampled signal values and
    # 8 summary values before the next stage. The old names listed both stages'
    # samples first, which mislabeled columns 15–37 without altering training.
    FEATURE_NAMES.extend(f'{stage}_{channel}_{int(second):03d}s'
                         for channel in CHANNELS for second in GRID)
    FEATURE_NAMES.extend(f'{stage}_{item}' for item in STAGE_EXTRA_NAMES)


def diagnostic(batch, description):
    text = str(description).lower()
    if batch in (1, 2):
        return '1c discharge' in text and '[test capacity]' not in text
    if batch == 3:
        return '1c discharge' in text and '[test capacity]' not in text
    return '[test capacity]' in text and '0.2c discharge' not in text


def stage_features(cycle, stage):
    time = np.asarray(cycle['relative_time_min'], dtype=float).reshape(-1)
    cuts = np.flatnonzero(time == 0)
    if len(cuts) <= stage:
        return None
    a = cuts[stage]
    b = cuts[stage + 1] if stage + 1 < len(cuts) else len(time)
    t = 60 * time[a:b]
    keep = np.isfinite(t) & (t <= GRID[-1])
    if keep.sum() < 90 or t[keep][-1] < GRID[-1] - 2:
        return None
    t = t[keep]
    order = np.argsort(t)
    t = t[order]
    unique, ix = np.unique(t, return_index=True)
    if len(unique) < 90:
        return None
    arrays = []
    for channel in CHANNELS:
        values = np.asarray(cycle[channel], dtype=float).reshape(-1)[a:b][keep][order][ix]
        if not np.isfinite(values).all():
            return None
        arrays.append(np.interp(GRID, unique, values))
    v, current, temp = arrays
    prefix = np.concatenate(arrays)
    extra = np.array([
        v[-1] - v[0], np.mean(v), (v[-1] - v[0]) / GRID[-1],
        np.mean(current), np.std(current), temp[-1] - temp[0],
        np.mean(temp), np.trapezoid(np.abs(current), GRID) / 3600,
    ])
    return np.r_[prefix, extra].astype('float32')


def process(path, batch):
    mat = loadmat(path, simplify_cells=True)
    data = mat['data']
    summary = mat['summary']
    capacity = np.asarray(summary['discharge_capacity_Ah'], dtype=float).reshape(-1)
    life = int(np.asarray(summary['cycle_life']).item())
    if len(capacity) != len(data) or life != len(data):
        raise ValueError(f'{path}: inconsistent cycle arrays')
    cycle_numbers, features, capacities, descriptions = [], [], [], []
    for index, cycle in enumerate(data):
        description = str(cycle['description'])
        if not diagnostic(batch, description):
            continue
        charge = stage_features(cycle, 0)
        discharge = stage_features(cycle, 2)
        if charge is None or discharge is None:
            continue
        q = capacity[index]
        if not np.isfinite(q) or not .8 < q < 2.4:
            continue
        cycle_numbers.append(index + 1)
        features.append(np.r_[charge, discharge])
        capacities.append(q)
        descriptions.append(description)
    if len(features) < 8:
        raise ValueError(f'{path}: only {len(features)} usable diagnostics')
    capacity = np.asarray(capacities, dtype='float32')
    # Initial formation and current-rate shifts make the first recorded 0.2C
    # measurement non-comparable with later 1C/0.5C diagnostics.
    reference = float(np.median(capacity[:min(5, len(capacity))]))
    if not 1.5 < reference < 2.2:
        raise ValueError(f'{path}: implausible capacity reference {reference}')
    return dict(cycle=np.asarray(cycle_numbers, dtype='int32'),
                features=np.asarray(features, dtype='float32'), capacity=capacity,
                soh=(capacity / reference).astype('float32'),
                reference_capacity_ah=reference, cycle_life=life,
                descriptions=np.asarray(descriptions),
                source=str(path), batch=batch)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {'source': 'https://zenodo.org/records/10963339',
                'input': 'first 120 seconds of charge and discharge diagnostic cycles',
                'label': 'diagnostic discharge Ah / median first five comparable diagnostic Ah',
                'features': FEATURE_NAMES, 'cells': []}
    for batch in range(1, 7):
        paths = sorted((SOURCE / f'Batch-{batch}').glob('*.mat'))
        expected = 15 if batch == 2 else 8
        if len(paths) != expected:
            raise ValueError(f'Batch {batch}: expected {expected}, found {len(paths)}')
        for path in paths:
            prepared = process(path, batch)
            name = f'b{batch}_{path.stem}'
            np.savez_compressed(OUT / f'{name}.npz', **prepared)
            manifest['cells'].append(dict(id=name, batch=batch, n_diagnostics=len(prepared['cycle']),
                                          first_cycle=int(prepared['cycle'][0]),
                                          last_cycle=int(prepared['cycle'][-1]),
                                          life=prepared['cycle_life'],
                                          reference_capacity_ah=prepared['reference_capacity_ah'],
                                          first_soh=float(prepared['soh'][0]),
                                          last_soh=float(prepared['soh'][-1])))
            print(name, len(prepared['cycle']), prepared['cycle_life'], flush=True)
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('Prepared', len(manifest['cells']), 'cells', flush=True)


if __name__ == '__main__':
    main()
