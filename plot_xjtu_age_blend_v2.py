"""Plot representative held-out XJTU SOH curves for age-blended EOL."""

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
from train_trajectory_v2 import threshold_soh


ROOT = Path('battery_models_v2/xjtu_age_blend_challenger')
AUDIT = ROOT / 'joint_curve_audit.json'
OUT = ROOT / 'joint_curve_examples.png'


def examples(rows):
    chosen = []
    for group in sorted({row['group'] for row in rows}):
        group_rows = [row for row in rows if row['group'] == group]
        median = float(np.median([row['metrics']['blend']['curve_mape_percent']
                                  for row in group_rows]))
        chosen.append(min(group_rows, key=lambda row: abs(
            row['metrics']['blend']['curve_mape_percent']-median)))
    return chosen


def main():
    records = [row for row in load_manifest()['cells'] if row['dataset'] == 'XJTU']
    record_by_id = {row['cell_id']: row for row in records}
    direct = read_oof(DIRECT, 'estimate')
    fraction = read_oof(FRACTION, 'endpoint')
    with np.load(SOH) as source:
        soh = {(str(key), int(cycle)): float(value)
               for key, cycle, value in zip(source['cell_id'], source['cycle'],
                                             source['fusion'])}
    rows = json.loads(AUDIT.read_text(encoding='utf-8'))['by_cell']
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), constrained_layout=True)
    for ax, row in zip(axes.flat, examples(rows)):
        key = row['cell_id']
        cell = load_cell(record_by_id[key])
        origin = row['origin_cycle']
        future = np.flatnonzero((cell.cycle > origin) &
                                (cell.cycle <= cell.endpoint_cycle) & cell.soh_available)
        horizon = cell.cycle[future]-origin
        current = soh[key, origin]
        direct_eol = origin+direct[key, origin]
        fraction_eol = fraction[key, origin]
        blended_eol = float(blend_endpoints(np.array([origin]),
                         np.array([direct[key, origin]]),
                         np.array([fraction_eol]))[0])
        direct_curve = coherent_curve(current, threshold_soh(cell),
                                      direct_eol-origin, horizon)
        blend_curve = coherent_curve(current, threshold_soh(cell),
                                     blended_eol-origin, horizon)
        observed = (cell.cycle >= origin) & (cell.cycle <= cell.endpoint_cycle) & cell.soh_available
        ax.plot(cell.cycle[observed], cell.soh[observed], color='black', lw=1.7,
                label='Observed SOH')
        ax.plot(cell.cycle[future], direct_curve, color='#d87622', lw=1.4,
                label='Direct RUL curve')
        ax.plot(cell.cycle[future], blend_curve, color='#315fa5', lw=1.6,
                label='Age-blended curve')
        ax.scatter([origin], [current], color='#315fa5', s=27, zorder=4)
        ax.axhline(threshold_soh(cell), color='#bb4d4d', ls='--', lw=1)
        ax.axvline(cell.endpoint_cycle, color='gray', ls=':', lw=1)
        ax.set_title(f"{row['group']} · {key}\n"
                     f"blend curve {row['metrics']['blend']['curve_mape_percent']:.1f}%, "
                     f"EOL {row['metrics']['blend']['eol_ape_percent']:.1f}%")
        ax.set_xlabel('Source cycle')
        ax.set_ylabel('SOH')
        ax.grid(alpha=.2)
    axes.flat[0].legend(loc='best', fontsize=8)
    fig.suptitle('XJTU whole-strategy held-out development curves (first eligible origin)',
                 fontsize=14)
    fig.savefig(OUT, dpi=160)
    print('WROTE', OUT)


if __name__ == '__main__':
    main()
