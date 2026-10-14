"""Audit which operating conditions are actually documented in contract v2.

This is a data-readiness check, not a fitted model or an inferred replacement
for missing per-cycle use. It deliberately leaves unknown conditions unknown.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from battery_v2_features import load_manifest
from external_test_v2 import DIAGNOSTIC_DISCHARGE_RATES_C


OUTPUT = Path('battery_data_v2/operating_metadata_audit.json')
FIELDS = ('chemistry', 'ambient_temperature_c', 'charge_rate_c',
          'discharge_rate_c', 'diagnostic_discharge_rate_c',
          'charge_protocol', 'charge_rate_steps_c')


def audit(records):
    if len(records) != 197 or len({row['cell_id'] for row in records}) != 197:
        raise ValueError('expected 197 distinct cells in the audited contract')
    report = {}
    for domain in ('NASA', 'CALCE', 'MIT', 'XJTU'):
        rows = [row for row in records if row['dataset'] == domain]
        if not rows:
            raise ValueError(f'{domain}: no records')
        if any(row.get('source_current_unit') != 'A' or
               row.get('source_time_unit') not in ('s', 'min') or
               not row.get('capacity_protocol') or
               row.get('reference_capacity_ah', 0) <= 0 for row in rows):
            raise ValueError(f'{domain}: inconsistent source units, protocol or reference')
        rates = DIAGNOSTIC_DISCHARGE_RATES_C.get(domain)
        if rates is not None and any(
            row.get('diagnostic_discharge_rate_c') is None or
            not any(np.isclose(row['diagnostic_discharge_rate_c'], rate,
                               atol=.05, rtol=0) for rate in rates)
            for row in rows):
            raise ValueError(f'{domain}: training diagnostic rate contradicts input contract')
        groups = {}
        for group in sorted({row['group'] for row in rows}):
            members = [row for row in rows if row['group'] == group]
            groups[group] = {'cells': len(members), 'documented_cells': {
                field: sum(row.get(field) is not None and row.get(field) != []
                           for row in members) for field in FIELDS}}
        report[domain] = {
            'cells': len(rows),
            'groups': groups,
            'source_time_units': sorted({row['source_time_unit'] for row in rows}),
            'source_current_units': sorted({row['source_current_unit'] for row in rows}),
            'diagnostic_discharge_rates_c': sorted({
                row['diagnostic_discharge_rate_c'] for row in rows
                if row.get('diagnostic_discharge_rate_c') is not None}),
            'measured_temperature_cells': sum(row.get('temperature_available') is True
                                              for row in rows),
            'observed_rul_endpoint_cells': sum(row['endpoint_observed'] for row in rows),
            'reference_rules': sorted({row['reference_rule'] for row in rows}),
        }
    return {
        'status': 'source metadata completeness audit; not model inputs',
        'contract_version': 2,
        'cells': len(records),
        'fields': list(FIELDS),
        'domains': report,
        'interpretation': [
            'A documented source unit is not proof that every source adapter was correct; '
            'adapter audits verify feature extraction separately.',
            'Measured cell temperature does not supply an ambient setpoint.',
            'MIT charge policies are multi-step; one scalar charge rate would discard policy details.',
            'XJTU variable/random-use batches lack a single meaningful scalar aging rate; '
            'do not impute one from the held-out batch ID.',
            'NASA nominal diagnostic rate and chemistry are not verified in this contract.',
        ],
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    current = audit(load_manifest()['cells'])
    if args.verify:
        if current != json.loads(OUTPUT.read_text(encoding='utf-8')):
            raise ValueError('stored operating metadata audit differs from current contract')
        print('VERIFIED', OUTPUT)
    else:
        if OUTPUT.exists():
            raise FileExistsError(OUTPUT)
        OUTPUT.write_text(json.dumps(current, indent=2), encoding='utf-8')
        print('WROTE', OUTPUT)
