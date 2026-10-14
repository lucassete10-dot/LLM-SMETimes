"""Apply the *unchanged* Batch 1 prefix and quality rules to Batch 2/3."""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from download_mit_batches import FILES, ROOT
from prepare_mit import (COMMON_GRID, COMMON_NAMES, ENHANCED_NAMES, GRID,
                         discharge_slice, feature_vector, read_ref)


# Author LoadData.m, one-based [8:10,16:17] in Batch 2 (continuations of Batch 1).
CONTINUATIONS = {7, 8, 9, 15, 16}


def process(batch):
    if batch not in FILES:
        raise ValueError(batch)
    source = ROOT / FILES[batch][0]
    output = Path(r'D:\LLM\MIT\processed') / f'batch{batch}'
    output.mkdir(parents=True, exist_ok=True)
    audit = []
    saved = []
    counts = {'source_cells': 0, 'continuation_cells_excluded': 0,
              'insufficient_cells': 0, 'valid_cycles': 0}
    with h5py.File(source, 'r') as mat:
        struct = mat['batch']
        counts['source_cells'] = len(struct['summary'])
        for index in range(counts['source_cells']):
            name = f'b{batch}c{index}'
            if batch == 2 and index in CONTINUATIONS:
                counts['continuation_cells_excluded'] += 1
                audit.append({'cell': name, 'cycle': None, 'capacity_ah': None,
                              'reason': 'continuation_of_batch1_cell'})
                continue
            summary = mat[struct['summary'][index, 0]]
            cycles = mat[struct['cycles'][index, 0]]
            capacities = np.asarray(summary['QDischarge']).reshape(-1)
            cycle_numbers = np.asarray(summary['cycle']).reshape(-1)
            if len(capacities) != len(cycles['I']):
                raise ValueError(f'{name}: summary/cycle mismatch')
            common, enhanced, records = [], [], []
            for j, capacity in enumerate(capacities):
                q = float(capacity)
                cycle = int(cycle_numbers[j])
                reason = ''
                if not np.isfinite(q) or q <= .2:
                    reason = 'invalid_capacity'
                elif q > 1.25:
                    reason = 'capacity_above_physical_screen_1p25Ah'
                if not reason:
                    current = read_ref(mat, cycles['I'][j, 0])
                    voltage = read_ref(mat, cycles['V'][j, 0])
                    time = read_ref(mat, cycles['t'][j, 0])
                    temp = read_ref(mat, cycles['T'][j, 0])
                    discharge_q = read_ref(mat, cycles['Qd'][j, 0])
                    if not (len(current) == len(voltage) == len(time) == len(temp) == len(discharge_q)):
                        reason = 'invalid_curve'
                    elif not np.isfinite(np.c_[current, voltage, time, discharge_q]).all():
                        reason = 'invalid_curve'
                    elif abs(np.nanmax(discharge_q) - q) > max(.02, .03 * q):
                        reason = 'summary_curve_capacity_mismatch'
                    else:
                        run = discharge_slice(time, current, voltage)
                        if run is None:
                            reason = 'no_300s_main_discharge'
                        else:
                            try:
                                base = feature_vector(time[run], current[run], voltage[run],
                                                      temp[run], COMMON_GRID)
                                rich = feature_vector(time[run], current[run], voltage[run],
                                                      temp[run], GRID)
                            except ValueError:
                                reason = 'bad_prefix'
                if not reason:
                    common.append(base)
                    enhanced.append(rich)
                    records.append((cycle, q))
                audit.append({'cell': name, 'cycle': cycle, 'capacity_ah': q, 'reason': reason})
            if len(records) < 50:
                counts['insufficient_cells'] += 1
                print(name, 'SKIP: only', len(records), 'valid cycles', flush=True)
                continue
            x = np.stack(common)
            z = np.stack(enhanced)
            q = np.asarray([r[1] for r in records], dtype='float32')
            first_rich = np.where(np.isfinite(z[0]), z[0], 0)
            np.savez_compressed(output / f'{name}.npz',
                                common=x-x[0], common_raw=x,
                                enhanced=z-first_rich, enhanced_raw=z,
                                capacity=q, soh=q/q[0],
                                cycle=np.asarray([r[0] for r in records]),
                                common_names=COMMON_NAMES, enhanced_names=ENHANCED_NAMES)
            saved.append(name)
            counts['valid_cycles'] += len(q)
            print(name, 'kept', len(q), 'of', len(capacities), flush=True)
    audit_table = pd.DataFrame(audit)
    audit_table.to_csv(output / 'audit.csv', index=False)
    checksum = hashlib.sha256()
    with source.open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            checksum.update(chunk)
    manifest = {'batch': batch, 'source': str(source), 'sha256': checksum.hexdigest(),
                'input_seconds': 300, 'saved_cells': saved,
                'excluded_reasons': audit_table.loc[audit_table.reason.ne(''), 'reason'].value_counts().to_dict(),
                'counts': counts, 'feature_schema': 'identical to Batch 1 prepare_mit.py',
                'no_test_label_features': True,
                'batch2_continuation_indices_zero_based': sorted(CONTINUATIONS) if batch == 2 else []}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('MANIFEST', json.dumps(manifest), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('batch', type=int, choices=FILES)
    process(parser.parse_args().batch)
