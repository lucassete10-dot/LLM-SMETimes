"""Audit exact per-cell 80%-SOH MIT endpoint observability.

This does not rewrite the source 0.88-Ah life labels or train a model. It
distinguishes observed exact-80% endpoints from right-censored histories.
"""

import json
from pathlib import Path

import numpy as np

from battery_data_contract import confirmed_threshold_event
from battery_v2_features import load_manifest


OUT = Path('battery_models_v2/mit_exact80_endpoint_audit.json')


def audit():
    manifest = load_manifest()
    rows = []
    for record in manifest['cells']:
        if record['dataset'] != 'MIT':
            continue
        with np.load(record['labels_file']) as labels:
            cycle = labels['cycle'].astype('int64')
            capacity = labels['capacity_ah'].astype('float64')
            available = labels['soh_available'].astype(bool)
        reference = float(record['reference_capacity_ah'])
        threshold = .8*reference
        # The first capacity row calibrates SOH; exclude it from endpoint search.
        event = confirmed_threshold_event(cycle[available], capacity[available], threshold)
        rows.append({'cell_id': record['cell_id'], 'group': record['group'],
                     'reference_capacity_ah': reference,
                     'exact80_threshold_ah': threshold,
                     'exact80_event_cycle': event,
                     'observed_through_cycle': int(cycle[-1]),
                     'source_0p88ah_event_cycle': record['endpoint_cycle'],
                     'source_threshold_as_soh': float(.88/reference),
                     'ever_below_exact80': bool(np.any(capacity[available] <= threshold))})
    by_group = {}
    for group in sorted({row['group'] for row in rows}):
        group_rows = [row for row in rows if row['group'] == group]
        by_group[group] = {'cells': len(group_rows),
                           'observed_exact80_events': sum(
                               row['exact80_event_cycle'] is not None for row in group_rows),
                           'right_censored_exact80': sum(
                               row['exact80_event_cycle'] is None for row in group_rows),
                           'source_0p88ah_events': sum(
                               row['source_0p88ah_event_cycle'] is not None
                               for row in group_rows)}
    return {'status': 'label/coverage audit only; no model or data contract changed',
            'exact80_rule': 'three consecutive accepted diagnostics <= 80% of cell reference; '
                            'median of up to ten subsequent diagnostics <= threshold',
            'cells': len(rows),
            'observed_exact80_events': sum(row['exact80_event_cycle'] is not None for row in rows),
            'right_censored_exact80': sum(row['exact80_event_cycle'] is None for row in rows),
            'ever_below_exact80': sum(row['ever_below_exact80'] for row in rows),
            'source_0p88ah_events': sum(row['source_0p88ah_event_cycle'] is not None
                                       for row in rows),
            'source_threshold_soh_range': [min(row['source_threshold_as_soh'] for row in rows),
                                           max(row['source_threshold_as_soh'] for row in rows)],
            'by_group': by_group, 'by_cell': rows}


if __name__ == '__main__':
    report = audit()
    OUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print({key: report[key] for key in ('cells', 'observed_exact80_events',
                                        'right_censored_exact80', 'source_0p88ah_events',
                                        'source_threshold_soh_range')})
    print(report['by_group'])
