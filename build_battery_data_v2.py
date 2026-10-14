"""Build the first unified, auditable battery SOH/RUL label dataset.

Uses original NASA MAT and already-audited CALCE/MIT/XJTU measurements. It does
not fit models, change legacy datasets, or put target capacity in input features.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat

from battery_data_contract import (SCHEMA_VERSION, LabelSpec, build_labels,
                                   confirmed_threshold_event)


NASA = Path(r'D:\LLM\NASA\raw\extracted')
CALCE = Path(r'D:\LLM\CALCE\processed')
MIT = Path(r'D:\LLM\MIT\processed')
MIT_RAW = Path(r'D:\LLM\MIT\raw')
XJTU = Path(r'D:\LLM\XJTU\processed')
DEFAULT_OUTPUT = Path('battery_data_v2')
MIT_SOURCE = {
    1: MIT_RAW / '2017-05-12_batchdata_updated_struct_errorcorrect.mat',
    2: MIT_RAW / '2017-06-30_batchdata_updated_struct_errorcorrect.mat',
    3: MIT_RAW / '2018-04-12_batchdata_updated_struct_errorcorrect.mat',
}


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            sha.update(block)
    return sha.hexdigest()


def nasa_records():
    paths = sorted(NASA.glob('B*.mat'))
    expected = {'B0005', 'B0006', 'B0007', 'B0018'}
    if {p.stem for p in paths} != expected:
        raise ValueError(f'NASA files differ from the audited four: {[p.stem for p in paths]}')
    for path in paths:
        raw = loadmat(path, simplify_cells=True)[path.stem]['cycle']
        entries = [(index, entry) for index, entry in enumerate(raw, 1)
                   if entry['type'] == 'discharge']
        cycle = np.arange(1, len(entries) + 1)
        source_cycle = np.array([index for index, _ in entries])
        q = np.array([float(np.asarray(entry['data']['Capacity']).item())
                      for _, entry in entries])
        threshold = float(.8 * q[0])
        event = confirmed_threshold_event(cycle, q, threshold)
        spec = LabelSpec(1, 'first observed complete discharge Ah',
                         'derived: three consecutive diagnostics below 80% of first Ah, '
                         'with median of up to 10 subsequent diagnostics below threshold',
                         event, threshold)
        yield 'NASA', path.stem, 'NASA', 'observed discharge count', cycle, q, source_cycle, spec, {
            'measurement_source': str(path), 'measurement_sha256': digest(path),
            'feature_source': None, 'capacity_protocol': 'NASA source discharge capacity',
            'temperature_available': True, 'chemistry': None,
            'source_time_unit': 's', 'source_current_unit': 'A',
            'charge_protocol': None, 'charge_rate_c': None,
            'discharge_rate_c': None, 'diagnostic_discharge_rate_c': None,
            'ambient_temperature_c': None,
            'temperature_context': 'discharge temperature measured; ambient setpoint not verified',
            'first_unconfirmed_threshold_cycle': int(cycle[np.flatnonzero(q <= threshold)[0]])
                if np.any(q <= threshold) else None,
        }


def calce_records():
    paths = sorted(CALCE.glob('CS2_*.npz'))
    expected = {'CS2_35', 'CS2_36', 'CS2_37', 'CS2_38'}
    if {p.stem for p in paths} != expected:
        raise ValueError(f'CALCE files differ from the audited four: {[p.stem for p in paths]}')
    for path in paths:
        with np.load(path) as data:
            cycle = data['observation'].astype('int64')
            q = data['capacity'].astype('float64')
            if len(data['raw_features']) != len(cycle):
                raise ValueError(f'{path}: feature/label row mismatch')
        threshold = float(.8 * q[0])
        event = confirmed_threshold_event(cycle, q, threshold)
        spec = LabelSpec(1, 'first accepted full-discharge Ah from Step 7',
                         'derived: three consecutive diagnostics below 80% of first Ah, '
                         'with median of up to 10 subsequent diagnostics below threshold',
                         event, threshold)
        yield 'CALCE', path.stem, 'CS2', 'accepted/discarded discharge segment ordinal', cycle, q, cycle, spec, {
            'measurement_source': str(path), 'measurement_sha256': digest(path),
            'feature_source': str(path), 'feature_array': 'raw_features',
            'capacity_protocol': 'CALCE Step 7 complete discharge',
            'temperature_available': False, 'chemistry': 'LiCoO2 cathode',
            'chemistry_source': 'https://calce.umd.edu/data',
            'source_time_unit': 's', 'source_current_unit': 'A',
            'charge_protocol': '0.5C CC-CV to 4.2V',
            'charge_rate_c': .5, 'discharge_rate_c': 1.,
            'diagnostic_discharge_rate_c': 1., 'ambient_temperature_c': None,
            'temperature_context': 'not measured in accepted Step 7 files',
            'first_unconfirmed_threshold_cycle': int(cycle[np.flatnonzero(q <= threshold)[0]])
                if np.any(q <= threshold) else None,
            'source_audit': str(CALCE / f'{path.stem}_manifest.json'),
        }


def mit_endpoint(mat, key):
    index = int(key.split('c', 1)[1])
    batch = mat['batch']
    summary = mat[batch['summary'][index, 0]]
    last_q = float(np.asarray(summary['QDischarge']).reshape(-1)[-1])
    life = float(np.asarray(mat[batch['cycle_life'][index, 0]]).reshape(-1)[0])
    # A stopped-early cell is right-censored, even when a source life field is
    # present. The 0.005 Ah tolerance follows the audited MIT endpoint rule.
    event = int(life) if np.isfinite(life) and last_q <= .885 else None
    return event, last_q, life


def mit_records():
    for batch, source in MIT_SOURCE.items():
        folder = MIT if batch == 1 else MIT / f'batch{batch}'
        paths = sorted(folder.glob(f'b{batch}c*.npz'))
        expected = {1: 46, 2: 43, 3: 45}[batch]
        # The author's Batch 3 loader excludes b3c37 for a documented data
        # collection issue; its prepared file remains on disk for provenance.
        if batch == 3:
            if {p.stem for p in paths if p.stem == 'b3c37'} != {'b3c37'}:
                raise ValueError('MIT Batch 3 expected the excluded b3c37 source file')
            paths = [path for path in paths if path.stem != 'b3c37']
        if len(paths) != expected:
            raise ValueError(f'MIT batch {batch}: expected {expected} eligible cells, got {len(paths)}')
        with h5py.File(source, 'r') as mat:
            for path in paths:
                with np.load(path) as data:
                    cycle = data['cycle'].astype('int64')
                    q = data['capacity'].astype('float64')
                    if len(data['enhanced_raw']) != len(cycle):
                        raise ValueError(f'{path}: feature/label row mismatch')
                event, last_q, source_life = mit_endpoint(mat, path.stem)
                index = int(path.stem.split('c', 1)[1])
                policy_ref = mat['batch']['policy_readable'][index, 0]
                charge_policy = ''.join(chr(int(code)) for code in
                                        np.asarray(mat[policy_ref]).reshape(-1))
                spec = LabelSpec(1, 'first accepted 4C discharge Ah',
                                 'source cycle_life to 0.88 Ah, only if last source Ah <=0.885 Ah',
                                 event, .88)
                yield 'MIT', path.stem, f'batch{batch}', 'source cycle number', cycle, q, cycle, spec, {
                    'measurement_source': str(source),
                    'feature_source': str(path), 'feature_sha256': digest(path),
                    'feature_array': 'enhanced_raw',
                    'capacity_protocol': 'main 4C discharge',
                    'temperature_available': True, 'chemistry': 'LFP/graphite',
                    'chemistry_source': 'https://web.mit.edu/braatzgroup/Severson_NatureEnergy_2019.pdf',
                    'source_time_unit': 'min', 'source_current_unit': 'A',
                    'charge_protocol': charge_policy, 'ambient_temperature_c': 30,
                    'charge_rate_c': None,
                    'charge_rate_steps_c': [float(value) for value in
                                            re.findall(r'(\d+(?:\.\d+)?)C', charge_policy)],
                    'discharge_rate_c': 4., 'diagnostic_discharge_rate_c': 4.,
                    'temperature_context': '30 C controlled ambient; cell temperature measured',
                    'source_last_capacity_ah': last_q,
                    'source_cycle_life_field': source_life if np.isfinite(source_life) else None,
                    'source_audit': str(folder / 'manifest.json'),
                }


def xjtu_records():
    manifest = json.loads((XJTU / 'manifest.json').read_text(encoding='utf-8'))
    if len(manifest['cells']) != 55:
        raise ValueError('XJTU manifest must list 55 cells')
    for item in manifest['cells']:
        path = XJTU / f"{item['id']}.npz"
        with np.load(path) as data:
            cycle = data['cycle'].astype('int64')
            q = data['capacity'].astype('float64')
            descriptions = [str(x).lower() for x in data['descriptions']]
            life = int(data['cycle_life'])
            batch = int(data['batch'])
            if len(data['features']) != len(cycle):
                raise ValueError(f'{path}: feature/label row mismatch')
        if batch != item['batch']:
            raise ValueError(f'{path}: batch mismatch')
        if batch <= 3:
            valid = all('1c discharge' in text and '[test capacity]' not in text
                        for text in descriptions)
        else:
            valid = all('[test capacity]' in text and '0.2c discharge' not in text
                        for text in descriptions)
        if not valid:
            raise ValueError(f'{path}: partial or non-comparable discharge in SOH labels')
        if not np.isclose(np.median(q[:5]), item['reference_capacity_ah'], atol=1e-5):
            raise ValueError(f'{path}: reference differs from source preparation')
        spec = LabelSpec(5, 'median first five comparable diagnostic discharge Ah',
                         'original MAT cycle_life; author EOL below 80% of initial capacity',
                         life, None)
        raw = Path(r'D:\LLM\XJTU\raw\extracted\Battery Dataset') / f'Batch-{batch}' / (item['id'].split('_', 1)[1] + '.mat')
        if not raw.exists():
            raise FileNotFoundError(raw)
        protocol = {1: '2C charge/1C full discharge', 2: '3C charge/1C full discharge',
                    3: '2C charge/variable-rate full discharge; 1C diagnostics selected',
                    4: 'variable-rate partial discharge; 1C full diagnostic selected',
                    5: 'random partial discharge; 1C full diagnostic selected',
                    6: 'satellite partial discharge; 0.5C full diagnostic selected'}[batch]
        yield 'XJTU', item['id'], f'batch{batch}', 'source cycle number', cycle, q, cycle, spec, {
            'measurement_source': str(raw), 'feature_source': str(path),
            'feature_sha256': digest(path), 'feature_array': 'features',
            'capacity_protocol': protocol, 'temperature_available': True,
            'chemistry': 'NCM532',
            'chemistry_source': 'https://zenodo.org/records/10963339',
            'source_time_unit': 'min', 'source_current_unit': 'A',
            'charge_protocol': protocol, 'temperature_context': 'room temperature',
            'ambient_temperature_c': None,
            'charge_rate_c': {1: 2., 2: 3., 3: 2., 4: None, 5: None, 6: None}[batch],
            'discharge_rate_c': {1: 1., 2: 1., 3: None, 4: None, 5: None, 6: None}[batch],
            'diagnostic_discharge_rate_c': .5 if batch == 6 else 1.,
            'source_audit': str(XJTU / 'manifest.json'),
        }


def build(output):
    output = Path(output)
    cells_folder = output / 'cells'
    cells_folder.mkdir(parents=True, exist_ok=True)
    records = []
    for producer in (nasa_records, calce_records, mit_records, xjtu_records):
        for dataset, key, group, cycle_unit, cycle, q, source_cycle, spec, provenance in producer():
            arrays, labels = build_labels(cycle, q, source_cycle, spec)
            np.savez_compressed(cells_folder / f'{key}.npz', **arrays)
            records.append({
                'dataset': dataset, 'cell_id': key, 'group': group,
                'cycle_unit': cycle_unit, 'rows': len(cycle),
                'labels_file': str(cells_folder / f'{key}.npz'),
                **labels, **provenance,
            })
    if len({row['cell_id'] for row in records}) != len(records):
        raise ValueError('duplicate global cell identifier')
    expected_files = {f"{row['cell_id']}.npz" for row in records}
    actual_files = {p.name for p in cells_folder.glob('*.npz')}
    if expected_files != actual_files:
        raise ValueError('output has stale or missing cell files')
    expected = {'NASA': 4, 'CALCE': 4, 'MIT': 134, 'XJTU': 55}
    summary = {dataset: {
        'cells': sum(row['dataset'] == dataset for row in records),
        'observations': sum(row['rows'] for row in records if row['dataset'] == dataset),
        'soh_examples': sum(row['soh_examples'] for row in records if row['dataset'] == dataset),
        'rul_examples': sum(row['rul_examples'] for row in records if row['dataset'] == dataset),
        'observed_endpoints': sum(row['endpoint_observed'] for row in records if row['dataset'] == dataset),
    } for dataset in expected}
    if {key: value['cells'] for key, value in summary.items()} != expected:
        raise ValueError(f'cell coverage differs from prior audit: {summary}')
    manifest = {
        'schema_version': SCHEMA_VERSION,
        'purpose': 'capacity/reference/endpoint labels; no model features copied and no models fit',
        'units': {'capacity_ah': 'Ah', 'soh': 'ratio', 'rul_cycles': 'cycle_unit per cell'},
        'signal_unit_contract': {'time_s': 's', 'voltage_v': 'V', 'current_a': 'A',
                                 'temperature_c': 'degC',
                                 'note': 'source time in minutes is converted to seconds by feature adapters; '
                                         'raw signal features are not copied by this label build'},
        'availability': 'calibration rows have NaN SOH/RUL; censored RUL and post-endpoint RUL are NaN',
        'splitting_keys': {'cell_holdout': 'cell_id', 'batch_or_strategy_holdout': 'group',
                           'dataset_holdout': 'dataset'},
        'summary': summary, 'cells': records,
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2,
                                                    allow_nan=False), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print('Wrote', output / 'manifest.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    build(parser.parse_args().output)
