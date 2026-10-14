"""Extend signal-only MIT inputs without changing labels or old prepared files.

Creates a new version, does not update latest.json. Non-MIT records retain their
existing observation budgets. Every original array and the first 100 signal
observations must replay exactly before the manifest is committed.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

import h5py
import numpy as np

from scripts.prepare_same_type_forecast import mit_prefix, save_json, sha
from audit_mit_label_free_extraction_v2 import MIT_SOURCE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--max-origin', type=int, default=200)
    args = parser.parse_args()
    old = json.loads((args.source / 'manifest.json').read_text(encoding='utf-8'))
    previous_limit = old['source_cycle_limit']
    if args.max_origin <= previous_limit:
        raise ValueError('extension must increase the existing source cycle budget')
    folder = args.source.parent / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ_MIT_extended')
    folder.mkdir()
    print('EXTEND', folder.resolve(), flush=True)
    manifest = copy.deepcopy(old)
    handles, audit = {}, []
    save_json(folder / 'config.json', {
        'parent_folder': str(args.source.resolve()),
        'parent_manifest_sha256': sha(args.source / 'manifest.json'),
        'mit_max_source_cycle': args.max_origin,
        'non_mit_source_cycle_limit': previous_limit,
        'purpose': 'label-free MIT signal budget extension; unchanged target contract',
        'source_sha256': {p: sha(p) for p in [__file__, 'scripts/prepare_same_type_forecast.py', 'prepare_mit.py']}})
    try:
        for record in manifest['cells']:
            if sha(record['file']) != record['sha256']:
                raise ValueError('parent cell hash mismatch: ' + record['cell_id'])
            if record['metadata']['dataset'] != 'MIT':
                continue
            cid = record['cell_id']
            batch = int(record['metadata']['group'].replace('batch', ''))
            if batch not in handles:
                handles[batch] = h5py.File(MIT_SOURCE[batch], 'r')
            rows, rejected = mit_prefix(handles[batch], int(cid.split('c')[1]), args.max_origin)
            cycle = np.asarray([row[0] for row in rows], dtype=np.int64)
            raw = np.asarray([row[1] for row in rows], dtype=np.float32)
            with np.load(record['file']) as value:
                arrays = {key: value[key].copy() for key in value.files}
            prefix = cycle <= previous_limit
            np.testing.assert_array_equal(cycle[prefix], arrays['cycle'])
            np.testing.assert_array_equal(raw[prefix], arrays['raw_features'])
            if len(cycle) == 0 or np.any(np.diff(cycle) <= 0):
                raise ValueError('invalid extended signal coordinates')
            arrays['cycle'], arrays['raw_features'] = cycle, raw
            file = folder / (cid + '.npz')
            np.savez_compressed(file, **arrays)
            with np.load(file) as saved:
                for key in arrays:
                    np.testing.assert_array_equal(saved[key], arrays[key])
            record.update(file=str(file.resolve()), sha256=sha(file), signal_observations=len(cycle))
            audit.append({'cell_id': cid, 'prefix_replay_passed': True,
                'labels_unchanged': True, 'observations': len(cycle),
                'added_observations': int(np.sum(cycle > previous_limit)),
                'last_signal_source_cycle': int(cycle[-1]), 'prefix_exclusions': rejected})
            save_json(folder / 'progress.json', {'completed_mit_cells': len(audit)})
            print(cid, 'observations', len(cycle), 'added', audit[-1]['added_observations'], flush=True)
        manifest['source_cycle_limit'] = args.max_origin
        manifest['source_cycle_limits_by_dataset'] = {
            domain: args.max_origin if domain == 'MIT' else previous_limit for domain in old['counts']}
        save_json(folder / 'extension_audit.json', {'passed': True, 'by_cell': audit,
            'scope': 'all MIT old signal prefixes and target arrays unchanged; added V/I/T only'})
        save_json(folder / 'manifest.json', manifest)
        print('COMPLETE', folder.resolve(), flush=True)
    except BaseException:
        save_json(folder / 'failure.json', {'traceback': traceback.format_exc()})
        raise
    finally:
        for handle in handles.values():
            handle.close()


if __name__ == '__main__':
    main()
