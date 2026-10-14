"""Check whether fixed charging-voltage windows are observable in XJTU 120-s prefixes.

This is a feature-availability audit, not a model comparison. A window that is
missing for many cycles cannot be substituted silently for the time grid.
"""

import json
from pathlib import Path

import numpy as np

from battery_v2_features import load_manifest


WINDOWS = ((3.55, 3.60), (3.60, 3.65), (3.65, 3.70), (3.70, 3.75))
RELATIVE_WINDOWS = ((.05, .10), (.10, .15), (.15, .20))
OUT = Path('battery_models_v2/xjtu_voltage_window_audit.json')


def main():
    groups = {}
    for row in load_manifest()['cells']:
        if row['dataset'] != 'XJTU':
            continue
        with np.load(row['feature_source']) as source:
            voltage = source['features'][:, :5]
        if not np.isfinite(voltage).all():
            raise ValueError(f"{row['cell_id']}: nonfinite prefix voltage")
        found = {f'{lo:.2f}-{hi:.2f}V': (np.min(voltage, axis=1) <= lo) &
                 (np.max(voltage, axis=1) >= hi) for lo, hi in WINDOWS}
        rise = voltage-voltage[:, :1]
        relative_found = {f'+{lo:.2f}-+{hi:.2f}V': np.max(rise, axis=1) >= hi
                          for lo, hi in RELATIVE_WINDOWS}
        group = groups.setdefault(row['group'], {'cells': 0, 'observations': 0,
                                                  'observed': {key: 0 for key in found},
                                                  'relative_observed': {key: 0 for key in relative_found},
                                                  'monotone_charge_prefixes': 0})
        group['cells'] += 1
        group['observations'] += len(voltage)
        group['monotone_charge_prefixes'] += int(np.all(np.diff(voltage, axis=1) >= 0, axis=1).sum())
        for key, mask in found.items():
            group['observed'][key] += int(mask.sum())
        for key, mask in relative_found.items():
            group['relative_observed'][key] += int(mask.sum())
    for group, values in groups.items():
        values['window_coverage'] = {key: count / values['observations']
                                     for key, count in values['observed'].items()}
        values['relative_window_coverage'] = {key: count / values['observations']
                                              for key, count in values['relative_observed'].items()}
        values['monotone_fraction'] = values['monotone_charge_prefixes']/values['observations']
        print(group, 'absolute', values['window_coverage'],
              'relative', values['relative_window_coverage'], flush=True)
    OUT.write_text(json.dumps({'source': 'existing first-120-s XJTU charge voltage samples',
                               'caveat': 'five time samples only; crossing-time interpolation needs separate validation',
                               'groups': groups}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
