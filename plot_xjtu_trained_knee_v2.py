"""Six held-out XJTU examples with training-fold-fitted knee shape."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from audit_xjtu_blended_joint_curve_v2 import SOH, coherent_curve
from battery_v2_features import load_cell, load_manifest
from evaluate_xjtu_age_blend_v2 import (DIRECT, FRACTION, blend_endpoints,
                                         read_oof)
from evaluate_xjtu_trained_knee_v2 import power_curve
from train_trajectory_v2 import threshold_soh


ROOT = Path('battery_models_v2/xjtu_trained_knee_challenger')
AUDIT = ROOT / 'development_cv.json'
OUT = ROOT / 'joint_curve_examples.png'


def representative(rows):
    chosen = []
    for group in sorted({row['group'] for row in rows}):
        candidates = [row for row in rows if row['group'] == group]
        median = np.median([row['trained_power_curve_mape_percent']
                            for row in candidates])
        chosen.append(min(candidates, key=lambda row: abs(
            row['trained_power_curve_mape_percent']-median)))
    return chosen


def main():
    report = json.loads(AUDIT.read_text(encoding='utf-8'))
    rows = [row for fold in report['folds'] for row in fold['by_cell']]
    records = {row['cell_id']: row for row in load_manifest()['cells']
               if row['dataset'] == 'XJTU'}
    direct = read_oof(DIRECT, 'estimate')
    fraction = read_oof(FRACTION, 'endpoint')
    with np.load(SOH) as source:
        soh = {(str(key), int(cycle)): float(value)
               for key, cycle, value in zip(source['cell_id'], source['cycle'],
                                             source['fusion'])}
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), constrained_layout=True)
    for ax, row in zip(axes.flat, representative(rows)):
        key = row['cell_id']
        cell = load_cell(records[key])
        origin = row['origin_cycle']
        future = np.flatnonzero((cell.cycle > origin) &
                                (cell.cycle <= cell.endpoint_cycle) & cell.soh_available)
        h = cell.cycle[future]-origin
        current = soh[key, origin]
        predicted_eol = float(blend_endpoints(np.array([origin]),
                              np.array([direct[key, origin]]),
                              np.array([fraction[key, origin]]))[0])
        linear = coherent_curve(current, threshold_soh(cell),
                                predicted_eol-origin, h)
        shaped = power_curve(current, threshold_soh(cell),
                             predicted_eol-origin, h, row['power'])
        observed = ((cell.cycle >= origin) & (cell.cycle <= cell.endpoint_cycle) &
                    cell.soh_available)
        ax.plot(cell.cycle[observed], cell.soh[observed], color='black', lw=1.8,
                label='Observed SOH')
        ax.plot(cell.cycle[future], linear, color='#d87622', lw=1.4,
                label='Linear blend')
        ax.plot(cell.cycle[future], shaped, color='#315fa5', lw=1.7,
                label='Train-fitted power curve')
        ax.scatter([origin], [current], color='#315fa5', s=24, zorder=4)
        ax.axhline(threshold_soh(cell), color='#bb4d4d', ls='--', lw=1)
        ax.axvline(cell.endpoint_cycle, color='gray', ls=':', lw=1)
        ax.set_title(f"{row['group']} · {key}\n"
                     f"curve {row['trained_power_curve_mape_percent']:.1f}%, "
                     f"EOL {row['eol_ape_percent']:.1f}%, p={row['power']:.2f}")
        ax.set_xlabel('Source cycle')
        ax.set_ylabel('SOH')
        ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle('XJTU whole-strategy held-out shaped curves (development only)',
                 fontsize=14)
    fig.savefig(OUT, dpi=160)
    print('WROTE', OUT)


if __name__ == '__main__':
    main()
