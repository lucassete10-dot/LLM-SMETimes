"""Restore full-life TRAIN signals while retaining frozen labels and cell splits."""
import argparse
import copy
import json
from pathlib import Path
import traceback
import zipfile

import h5py
import numpy as np
import pandas as pd
from scipy.io import loadmat

from battery_smetimes.mit_units import mit_prefix_to_amperes
from scripts.audit_llm_curve_repair import load_frozen, fingerprint
from scripts.inspect_hust_public import PrimitiveUnpickler
from scripts.prepare_hust_public import features as hust_features
from scripts.prepare_same_type_forecast import mit_prefix, save_json
from scripts.prepare_zhu2022_nca import completed_groups, signal_features
from prepare_xjtu import diagnostic, stage_features


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_cell(folder, cid, cell, cycles, raw, provenance, exclusions):
    cycles = np.asarray(cycles, dtype=np.float32)
    raw = np.asarray(raw, dtype=np.float32)
    if raw.shape != (len(cycles), cell['raw'].shape[1]) or not len(cycles) or np.any(np.diff(cycles)<=0):
        raise ValueError('Full-life signal shape/order mismatch: '+cid)
    if np.isinf(raw).any():
        raise ValueError('Infinite restored feature: '+cid)
    # Preserve every existing <=100 observation exactly; no training target edits.
    old = cell['cycle']<=100
    selected = cycles<=100
    np.testing.assert_array_equal(cycles[selected], cell['cycle'][old])
    np.testing.assert_allclose(raw[selected], cell['raw'][old], rtol=0, atol=0, equal_nan=True)
    path = folder/(cid.replace('#', '_')+'.npz')
    np.savez_compressed(path, cycle=cycles, raw=raw)
    return dict(cell_id=cid, path=str(path.resolve()), sha256=fingerprint(path),
                observations=len(cycles), first_source_cycle=float(cycles[0]), last_source_cycle=float(cycles[-1]),
                first100_exact_replay=True, labels_unchanged=True, provenance=provenance, exclusions=exclusions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--queue', type=Path, default=Path('experiments/llm_curve_observation_tokens/20261015_observation_tokens/queue.json'))
    parser.add_argument('--output', type=Path, default=Path('data/interim/llm_full_life/20261015_full_life'))
    parser.add_argument('--datasets', nargs='+', default=['MIT','XJTU','HUST','UConn_ISU_ILCC','Zhu2022_NCA'])
    args = parser.parse_args()
    queue = read(args.queue)
    args.output.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        folder = args.output/dataset
        folder.mkdir(exist_ok=True)
        settings = queue['parents'][dataset]['screen_settings']
        _, cells, plan, registry, _ = load_frozen(settings)
        ids = plan['train']
        if (folder/'index.json').exists():
            index = read(folder/'index.json')
            if (index['train_ids']!=ids or index['base_settings']!=settings
                    or index['preparation_source_sha256']!=fingerprint(__file__)):
                raise ValueError('Full-life preparation configuration/source changed')
            for record in index['cells']:
                if fingerprint(record['path']) != record['sha256']:
                    raise ValueError('Existing full-life signal changed')
            print('REUSE', dataset, len(ids), flush=True)
            continue
        records = read(folder/'progress.json').get('cells', []) if (folder/'progress.json').exists() else []
        done = {r['cell_id'] for r in records}
        for record in records:
            if fingerprint(record['path']) != record['sha256']:
                raise ValueError('Partial extracted cell changed')
        handles = {}
        source_hashes = {}

        def source(path):
            path = str(Path(path).resolve())
            if path not in source_hashes:
                source_hashes[path] = fingerprint(path)
            return dict(path=path, sha256=source_hashes[path])

        def commit(cid, cycles, raw, provenance, exclusions):
            record = save_cell(folder, cid, cells[cid], cycles, raw, provenance, exclusions)
            records.append(record)
            done.add(cid)
            save_json(folder/'progress.json', dict(dataset=dataset, completed=len(done), planned=len(ids), cells=records))
            print('CELL', dataset, cid, record['observations'], record['last_source_cycle'], flush=True)

        try:
            if dataset in ('MIT', 'XJTU'):
                registry_cells = read(Path(settings['frozen_plan'])/'cell_registry.json')['cells']
                audit_config = read(Path(settings['frozen_plan'])/'config.json')
                issues = read(audit_config['config']['integrity_audit'])['issues']
                for cid in ids:
                    if cid in done:
                        continue
                    entry = registry_cells[cid]
                    path = Path(entry['raw_source'])
                    provenance = source(path)
                    if dataset=='MIT':
                        if str(path) not in handles:
                            handles[str(path)] = h5py.File(path, 'r')
                        rows, rejected = mit_prefix(handles[str(path)], int(cid.split('c')[1]), 10**9)
                        cycles = np.asarray([r[0] for r in rows])
                        raw = mit_prefix_to_amperes(np.asarray([r[1] for r in rows], dtype=np.float32))
                    else:
                        data = loadmat(path, variable_names=['data'], simplify_cells=True)['data']
                        batch = int(entry['metadata']['group'].replace('batch',''))
                        rows, rejected = [], []
                        for number, row in enumerate(data, 1):
                            if not diagnostic(batch, row['description']):
                                continue
                            # Explicit allowlist; capacity fields cannot enter extraction.
                            signal = {k:row[k] for k in ('relative_time_min','voltage_V','current_A','temperature_C')}
                            charge, discharge = stage_features(signal, 0), stage_features(signal, 2)
                            if charge is None or discharge is None:
                                rejected.append(dict(cycle=number, reason='incomplete finite 120-second diagnostic signals'))
                            else:
                                rows.append((number, np.r_[charge,discharge]))
                        cycles = np.asarray([r[0] for r in rows])
                        raw = np.asarray([r[1] for r in rows], dtype=np.float32)
                    rejected_ids = [r['cycle'] for r in issues if r['cell_id']==cid]
                    keep = ~np.isin(cycles, rejected_ids)
                    commit(cid, cycles[keep], raw[keep], provenance,
                           rejected+[dict(cycle=n,reason='existing frozen integrity exclusion') for n in rejected_ids])
            elif dataset=='HUST':
                archive = Path('data/raw/five_dataset_public/BatteryLife_v12/HUST.zip')
                provenance = source(archive)
                with zipfile.ZipFile(archive) as zipped:
                    for member in zipped.namelist():
                        if not member.endswith('.pkl'):
                            continue
                        # Member stem matches the publisher cell_id; avoid loading held-out waveforms.
                        if Path(member).stem not in set(ids)-done:
                            continue
                        with zipped.open(member) as stream:
                            item = PrimitiveUnpickler(stream).load()
                        cid = item['cell_id']
                        if cid not in ids or cid in done:
                            continue
                        cycles, raw, rejected = [], [], []
                        for row in item['cycle_data']:
                            try:
                                raw.append(hust_features(row)); cycles.append(float(row['cycle_number']))
                            except ValueError as error:
                                rejected.append(dict(cycle=float(row['cycle_number']),reason=str(error)))
                        commit(cid, cycles, raw, {**provenance,'member':member}, rejected)
            elif dataset=='UConn_ISU_ILCC':
                for cid in ids:
                    if cid not in done:
                        commit(cid, cells[cid]['cycle'], cells[cid]['raw'],
                               dict(existing_full_life_sparse_release=True), [])
            elif dataset=='Zhu2022_NCA':
                audit = read('data/interim/zhu2022_nca/20261004T161215_905258Z/audit.json')
                raw_folder = Path(audit['raw_directory'])
                intake = read(raw_folder/'intake_manifest.json')
                archive = raw_folder/intake['file']
                provenance = source(archive)
                if provenance['sha256']!=audit['raw_sha256']:
                    raise ValueError('NCA original archive changed')
                by_id = {r['cell_id']:r for r in audit['cells']}
                with zipfile.ZipFile(archive) as zipped:
                    for cid in ids:
                        if cid in done:
                            continue
                        member = by_id[cid]['member']
                        cycles, raw, rejected = [], [], []
                        with zipped.open(member) as stream:
                            reader = pd.read_csv(stream, usecols=['time/s','Ecell/V','<I>/mA','cycle number'], chunksize=200000)
                            for n, frame in completed_groups(reader):
                                try:
                                    raw.append(signal_features(frame)); cycles.append(float(n))
                                except ValueError as error:
                                    rejected.append(dict(cycle=float(n),reason=str(error)))
                        commit(cid, cycles, raw, {**provenance,'member':member}, rejected)
            else:
                raise ValueError('Unregistered dataset')
            if done != set(ids):
                raise ValueError('Not every TRAIN cell was restored: '+str(set(ids)-done))
            save_json(folder/'index.json', dict(dataset=dataset, train_ids=ids, cells=records,
                base_settings=settings, physical_plan=plan, original_registry=str(registry.resolve()),
                preparation_source_sha256=fingerprint(__file__), heldout_signals_extended=False,
                all_train_cells_retained=True, labels_unchanged=True, complete=True))
            print('COMPLETE',dataset,len(records),flush=True)
        except BaseException:
            save_json(folder/'failure.json', dict(traceback=traceback.format_exc(), completed=len(records)))
            raise
        finally:
            for handle in handles.values():
                handle.close()


if __name__=='__main__':
    main()
