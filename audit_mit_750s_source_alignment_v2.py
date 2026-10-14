"""Check 750 s MIT extraction uses the same diagnostic as 300 s baseline."""

import json
from pathlib import Path

import numpy as np

from battery_v2_features import load_manifest


ROOT = Path('battery_models_v2/soh_long_prefix_challenger')
OUT = ROOT / 'source_alignment_audit.json'


def main():
    rows = []
    records = [row for row in load_manifest()['cells'] if row['dataset'] == 'MIT']
    for record in records:
        key = record['cell_id']
        with np.load(record['feature_source']) as old, np.load(ROOT / f'{key}.npz') as new:
            if not np.array_equal(old['cycle'], new['cycle']):
                raise ValueError(f'{key}: source cycle mismatch')
            old_v300 = old['enhanced_raw'][:, 10].astype(float)
            new_v300 = (new['raw'][:, 2]-new['raw'][:, 3]).astype(float)
        valid = np.isfinite(new_v300)
        difference = np.abs(new_v300[valid]-old_v300[valid])
        rows.append({'cell_id': key, 'group': record['group'],
                     'compared_rows': int(valid.sum()),
                     'mean_abs_v300_gap_v': float(np.mean(difference)) if len(difference) else None,
                     'rows_gap_over_0p01_v': int(np.sum(difference > .01)),
                     'rows_gap_over_0p05_v': int(np.sum(difference > .05)),
                     'max_abs_v300_gap_v': float(np.max(difference)) if len(difference) else None})
    report = {'status': 'source diagnostic matching audit; not model accuracy',
              'cells': len(rows),
              'compared_rows': sum(row['compared_rows'] for row in rows),
              'rows_gap_over_0p01_v': sum(row['rows_gap_over_0p01_v'] for row in rows),
              'rows_gap_over_0p05_v': sum(row['rows_gap_over_0p05_v'] for row in rows),
              'worst_cell_max_gap_v': max(row['max_abs_v300_gap_v'] or 0 for row in rows),
              'by_cell': rows}
    OUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print({key: value for key, value in report.items() if key != 'by_cell'})
    print('WROTE', OUT)


if __name__ == '__main__':
    main()
