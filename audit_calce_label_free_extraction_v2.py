"""Read-only CALCE signal-only extraction audit against prepared v2 features.

The old extractor included cumulative discharge capacity for deduplication and
quality checks. This variant reads no capacity column, then checks whether its
signal-only row selection and feature values reproduce all four training cells.
Original Excel workbooks are never modified.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_calce import IDS, OUT as TRAIN_PROCESSED, ROOT as TRAIN_RAW, vi_features


OUT = Path('battery_models_v2/calce_label_free_extraction_audit.json')
SIGNAL_COLS = ['Date_Time', 'Test_Time(s)', 'Step_Time(s)', 'Step_Index',
               'Cycle_Index', 'Current(A)', 'Voltage(V)']


def extract_folder(folder):
    folder = Path(folder)
    files = [folder] if folder.is_file() and folder.suffix.lower() == '.xlsx' else sorted(
        folder.rglob('*.xlsx'))
    if not files:
        raise ValueError(f'no CALCE Excel workbooks under {folder}')
    frames = []
    for index, path in enumerate(files, 1):
        excel = pd.ExcelFile(path)
        sheets = [name for name in excel.sheet_names if name.startswith('Channel')]
        if not sheets:
            raise ValueError(f'{path}: no Channel sheet')
        for sheet in sheets:
            df = pd.read_excel(excel, sheet_name=sheet,
                               usecols=lambda column: column in SIGNAL_COLS)
            if set(SIGNAL_COLS) - set(df):
                raise ValueError(f'{path}: missing required signal columns')
            frames.append(df[(df['Step_Index'] == 7) & (df['Current(A)'] < -.5)].copy())
        excel.close()
        if index % 5 == 0:
            print(folder.name, 'read', index, '/', len(files), 'workbooks', flush=True)
    source = pd.concat(frames, ignore_index=True)
    discharge_rows = len(source)
    source = (source.drop_duplicates(subset=SIGNAL_COLS)
                    .sort_values('Date_Time', kind='stable').reset_index(drop=True))
    cuts = ((source['Cycle_Index'].diff() != 0) |
            (source['Step_Time(s)'].diff() <= 0) |
            (source['Date_Time'].diff().dt.total_seconds() > 120))
    segments = cuts.cumsum()
    numbers, features, rejected = [], [], {}
    for ordinal, (_, segment) in enumerate(source.groupby(segments, sort=False), 1):
        time = segment['Step_Time(s)'].to_numpy(float)
        voltage = segment['Voltage(V)'].to_numpy(float)
        current = segment['Current(A)'].to_numpy(float)
        reason = ''
        if not np.isfinite(np.c_[time, voltage, current]).all():
            reason = 'nonfinite_signal'
        elif time[0] > 60:
            reason = 'missing_start'
        elif time[-1] < 1000:
            reason = 'shorter_than_1000s'
        elif voltage[-1] > 2.85:
            reason = 'incomplete_discharge'
        if reason:
            rejected[ordinal] = reason
            continue
        numbers.append(ordinal)
        features.append(vi_features(time, voltage, current))
    if not features:
        raise ValueError(f'{folder}: no complete CALCE diagnostics')
    cycle = np.asarray(numbers, dtype='int64')
    raw = np.asarray(features, dtype='float32')
    if raw.shape != (len(cycle), 29) or np.any(np.diff(cycle) <= 0):
        raise ValueError('CALCE signal feature/ordinal contract failed')
    return cycle, raw, {'workbooks': len(files), 'discharge_rows': discharge_rows,
                        'duplicate_signal_rows_removed': discharge_rows-len(source),
                        'segments': int(ordinal), 'rejected': rejected}


def main():
    cells = {}
    for key in IDS:
        cycle, raw, detail = extract_folder(TRAIN_RAW / key)
        with np.load(TRAIN_PROCESSED / f'{key}.npz') as prepared:
            expected_cycle = prepared['observation'].astype('int64')
            expected_raw = prepared['raw_features'].astype('float32')
        lookup = {int(value): index for index, value in enumerate(cycle)}
        missing = sorted(set(expected_cycle.tolist()) - set(lookup))
        extra = sorted(set(cycle.tolist()) - set(expected_cycle.tolist()))
        if missing:
            raise ValueError(f'{key}: prepared ordinals absent from signal-only extraction: {missing}')
        aligned = raw[[lookup[int(value)] for value in expected_cycle]]
        same = bool(np.array_equal(aligned, expected_raw, equal_nan=True))
        if not same:
            raise ValueError(f'{key}: shared feature values differ')
        cells[key] = {'prepared_rows': len(expected_cycle),
                      'signal_qualified_rows': len(cycle),
                      'extra_signal_only_ordinals': extra,
                      'feature_values_exact': same,
                      'rejected_reason_counts': {reason: list(detail['rejected'].values()).count(reason)
                                                 for reason in set(detail['rejected'].values())},
                      **{name: detail[name] for name in
                         ('workbooks', 'discharge_rows', 'duplicate_signal_rows_removed', 'segments')}}
        print('MATCH', key, cells[key], flush=True)
    report = {'purpose': 'CALCE read-only signal-only Excel extraction audit',
              'capacity_column_read': False, 'cells': cells}
    OUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Wrote', OUT)


if __name__ == '__main__':
    main()
