"""Audit label-free SMETimes sequence conversion on all 197 training cells."""

import argparse
import json
from pathlib import Path

import numpy as np

from adapt_nasa_mat_v2 import extract_mat
from battery_smetimes_signal_v2 import CHANNELS, CONTEXT, build_sequence
from battery_v2_features import load_manifest


OUT = Path('battery_models_v2/smetimes_signal_audit.json')


def raw_signal(record):
    if record['dataset'] == 'NASA':
        _, raw = extract_mat(record['measurement_source'], record['cell_id'])
        return raw
    with np.load(record['feature_source'], allow_pickle=False) as archive:
        return archive[record['feature_array']].astype('float32')


def audit():
    records = load_manifest()['cells']
    if len(records) != 197:
        raise ValueError('expected the audited 197-cell training cohort')
    report = {}
    for record in records:
        domain = record['dataset']
        raw = raw_signal(record)
        if len(raw) != record['rows']:
            raise ValueError(f"{record['cell_id']}: source/contract row count differs")
        sequence, context = build_sequence(domain, raw)
        entry = report.setdefault(domain, {'cells': 0, 'diagnostics': 0,
                                           'charge_available_diagnostics': 0,
                                           'temperature_available_diagnostics': 0,
                                           'feature_shape': [10, len(CHANNELS)]})
        entry['cells'] += 1
        entry['diagnostics'] += len(raw)
        entry['charge_available_diagnostics'] += int(np.sum(context[:, 2] > 0))
        entry['temperature_available_diagnostics'] += int(np.sum(context[:, 3] > 0))
        if not np.isfinite(sequence).all() or not np.isfinite(context).all():
            raise ValueError(f"{record['cell_id']}: nonfinite adapted signal")
    return {'status': 'label-free input-availability audit; not a SOH/RUL validation',
            'channels': list(CHANNELS), 'context': list(CONTEXT),
            'slot_order': ['five charge positions', 'five diagnostic discharge positions'],
            'limitations': ['missing charge/temperature values are zero with explicit masks',
                            'five-point resampling discards finer curve detail',
                            'the original SMETimes backbone must still be trained and '
                            'compared in strict whole-cell/group holdouts'],
            'domains': report}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    result = audit()
    if args.verify:
        if result != json.loads(OUT.read_text(encoding='utf-8')):
            raise ValueError('stored SMETimes signal audit differs from current corpus')
        print('VERIFIED', OUT)
    else:
        if OUT.exists():
            raise FileExistsError(OUT)
        OUT.write_text(json.dumps(result, indent=2), encoding='utf-8')
        print('WROTE', OUT)
