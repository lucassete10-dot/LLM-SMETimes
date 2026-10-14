"""Compare source EOL labels with observed capacity-threshold crossings.

This is a label audit, not model evaluation. In particular, a source EOL
without an observed crossing is not evidence that the battery never crosses:
its trajectory is right-censored at the last available diagnostic.
"""

import json
from pathlib import Path

import numpy as np

from battery_v2_features import load_all
from battery_data_contract import confirmed_threshold_event
from train_trajectory_v2 import threshold_soh


OUT = Path('battery_models_v2/endpoint_curve_consistency_audit.json')


def audit_cell(cell):
    threshold = threshold_soh(cell)
    observed = np.flatnonzero(cell.soh_available)
    crossing = observed[cell.soh[observed] <= threshold]
    first = int(cell.cycle[crossing[0]]) if len(crossing) else None
    confirmed = confirmed_threshold_event(cell.cycle[observed],
                                          cell.soh[observed], threshold)
    endpoint = cell.endpoint_cycle
    at_endpoint = observed[cell.cycle[observed] == endpoint] if endpoint else []
    endpoint_soh = float(cell.soh[at_endpoint[0]]) if len(at_endpoint) else None
    return {'cell_id': cell.cell_id, 'group': cell.group,
            'source_endpoint_cycle': endpoint, 'threshold_soh': float(threshold),
            'first_observed_threshold_cycle': first,
            'first_confirmed_threshold_cycle': confirmed,
            'last_observed_cycle': int(cell.cycle[observed[-1]]) if len(observed) else None,
            'soh_at_source_endpoint': endpoint_soh,
            'source_endpoint_observed_below_threshold':
            endpoint_soh <= threshold if endpoint_soh is not None else None,
            'crossing_minus_source_cycles': first-endpoint
            if first is not None and endpoint is not None else None,
            'confirmed_minus_declared_cycles': confirmed-endpoint
            if confirmed is not None and endpoint is not None else None}


def main():
    manifest, cells = load_all()
    report = {'status': 'label/observability audit, not model validation',
              'domains': {}}
    for domain in ('NASA', 'CALCE', 'MIT', 'XJTU'):
        rows = [audit_cell(cells[r['cell_id']]) for r in manifest['cells']
                if r['dataset'] == domain]
        paired = [r for r in rows if r['confirmed_minus_declared_cycles'] is not None]
        observed_endpoints = [r for r in rows if r['source_endpoint_cycle'] is not None]
        report['domains'][domain] = {
            'cells': len(rows), 'source_endpoint_cells': len(observed_endpoints),
            'endpoint_diagnostic_below_threshold': sum(
                r['source_endpoint_observed_below_threshold'] is True for r in rows),
            'endpoint_diagnostic_above_threshold': sum(
                r['source_endpoint_observed_below_threshold'] is False for r in rows),
            'endpoint_diagnostic_missing': sum(
                r['source_endpoint_observed_below_threshold'] is None for r in rows),
            'observed_threshold_crossings': sum(r['first_observed_threshold_cycle']
                                                is not None for r in rows),
            'confirmed_threshold_crossings': sum(r['first_confirmed_threshold_cycle']
                                                 is not None for r in rows),
            'paired_endpoint_confirmations': len(paired),
            'exact_cycle_matches': sum(r['confirmed_minus_declared_cycles'] == 0
                                       for r in paired),
            'median_abs_crossing_source_gap_cycles': float(np.median([
                abs(r['confirmed_minus_declared_cycles']) for r in paired]))
                if paired else None,
            'by_group': {group: {
                'cells': sum(r['group'] == group for r in rows),
                'observed_crossings': sum(r['group'] == group and
                                          r['first_observed_threshold_cycle'] is not None
                                          for r in rows)}
                for group in sorted({r['group'] for r in rows})},
            'by_cell': rows}
    OUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    for domain, summary in report['domains'].items():
        print(domain, 'source endpoints', summary['source_endpoint_cells'],
              'observed crossings', summary['observed_threshold_crossings'],
              'confirmed', summary['confirmed_threshold_crossings'],
              'paired', summary['paired_endpoint_confirmations'],
              'exact matches', summary['exact_cycle_matches'],
              'endpoint below/above/missing',
              summary['endpoint_diagnostic_below_threshold'],
              summary['endpoint_diagnostic_above_threshold'],
              summary['endpoint_diagnostic_missing'], flush=True)
    print('WROTE', OUT)


if __name__ == '__main__':
    main()
