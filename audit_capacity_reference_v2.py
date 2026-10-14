"""Audit how sensitive v2 SOH references are to early capacity measurements.

This reads saved capacity labels only. The alternative first-five median is a
diagnostic, not a replacement label, inference input, or model-selection result.
"""

import json
from pathlib import Path

import numpy as np

from battery_data_contract import confirmed_threshold_event


MANIFEST = Path('battery_data_v2/manifest.json')
OUTPUT = Path('battery_data_v2/reference_sensitivity_audit.json')


def summarize(rows):
    absolute = np.asarray([abs(row['first5_vs_first_relative']) for row in rows])
    return {
        'cells': len(rows),
        'median_absolute_reference_change': float(np.median(absolute)),
        'p95_absolute_reference_change': float(np.percentile(absolute, 95)),
        'max_absolute_reference_change': float(np.max(absolute)),
        'cells_above_1pct': int(np.sum(absolute > .01)),
        'cells_above_2pct': int(np.sum(absolute > .02)),
        'cells_above_4pct': int(np.sum(absolute > .04)),
    }


def audit(manifest_path=MANIFEST):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    cells = []
    for meta in manifest['cells']:
        with np.load(meta['labels_file']) as data:
            cycle = data['cycle'].astype('int64')
            capacity = data['capacity_ah'].astype('float64')
            soh = data['soh'].astype('float64')
            available = data['soh_available'].astype(bool)
        if len(capacity) < 6 or np.any(~np.isfinite(capacity)):
            raise ValueError(f"{meta['cell_id']}: insufficient finite capacity diagnostics")
        count = int(meta['reference_measurement_count'])
        reference = float(np.median(capacity[:count]))
        if not np.isclose(reference, meta['reference_capacity_ah'], rtol=1e-10):
            raise ValueError(f"{meta['cell_id']}: reference does not match label file")
        expected_available = np.arange(len(capacity)) >= count
        if not np.array_equal(available, expected_available):
            raise ValueError(f"{meta['cell_id']}: calibration availability mismatch")
        if not np.allclose(soh[available], capacity[available] / reference, rtol=1e-10):
            raise ValueError(f"{meta['cell_id']}: SOH normalization mismatch")
        first = float(capacity[0])
        first5 = float(np.median(capacity[:5]))
        row = {
            'dataset': meta['dataset'], 'cell_id': meta['cell_id'],
            'reference_measurement_count': count,
            'reference_capacity_ah': reference,
            'first_capacity_ah': first,
            'first5_median_capacity_ah': first5,
            'first5_vs_first_relative': float(first5 / first - 1),
            'max_first5_deviation_from_median': float(np.max(np.abs(capacity[:5] / first5 - 1))),
        }
        if meta['dataset'] in ('NASA', 'CALCE'):
            original = confirmed_threshold_event(cycle, capacity, .8 * first)
            alternative = confirmed_threshold_event(cycle, capacity, .8 * first5)
            if original != meta['endpoint_cycle']:
                raise ValueError(f"{meta['cell_id']}: derived endpoint mismatch")
            row['first_reference_endpoint_cycle'] = original
            row['first5_reference_endpoint_cycle'] = alternative
            row['endpoint_shift_if_first5_used'] = (
                None if original is None or alternative is None else alternative - original)
        cells.append(row)
    domains = {dataset: summarize([row for row in cells if row['dataset'] == dataset])
               for dataset in ('NASA', 'CALCE', 'MIT', 'XJTU')}
    top = sorted(cells, key=lambda row: abs(row['first5_vs_first_relative']), reverse=True)[:12]
    return {
        'purpose': 'label-only reference sensitivity; no model features or weights changed',
        'interpretation': 'first-five median is hypothetical for NASA/CALCE/MIT and would defer SOH evaluation until after five diagnostics',
        'domains': domains, 'largest_first5_vs_first_changes': top, 'cells': cells,
    }


if __name__ == '__main__':
    result = audit()
    OUTPUT.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('WROTE', OUTPUT)
    print(json.dumps(result['domains'], indent=2))
