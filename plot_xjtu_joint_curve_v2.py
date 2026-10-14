"""Render median/worst honest XJTU SOH+RUL first-origin development curves."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from audit_xjtu_selected_joint_curve_v2 import ROOT, selected_soh, throughput_rul
from battery_v2_features import load_all
from evaluate_anchored_trajectory_v2 import get_domain_arrays
from evaluate_full_curve_practical_v2 import project
from train_trajectory_v2 import threshold_soh


OUT = ROOT / 'xjtu_joint_curve_examples.png'


def main():
    _, all_cells = load_all()
    cells = {key: cell for key, cell in all_cells.items() if cell.dataset == 'XJTU'}
    _, baseline_rul = get_domain_arrays('XJTU')
    soh = selected_soh(cells)
    rul = throughput_rul(cells, baseline_rul)
    rows = json.loads((ROOT / 'xjtu_selected_joint_curve_audit.json').read_text(
        encoding='utf-8'))['by_cell']
    ordered = sorted(rows, key=lambda row: row[
        'throughput_bounded_direct_curve_mape_percent'])
    chosen = [ordered[len(ordered)//2], ordered[-1]]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for ax, row, role in zip(axes, chosen, ('median', 'worst')):
        key = row['cell_id']
        cell = cells[key]
        cycle, predicted_rul, local_rul = rul[key]
        origin = int(cycle[0])
        sc, se = soh[key]
        present = float(se[np.searchsorted(sc, origin)])
        observed = np.flatnonzero((cell.cycle >= origin) &
                                  (cell.cycle <= cell.endpoint_cycle) & cell.soh_available)
        future = cell.cycle[observed]
        forecast = project(present, threshold_soh(cell), predicted_rul[0], local_rul[0],
                           future-origin)
        direct = np.clip(forecast['direct_linear'], 0, present)
        ax.plot(future, cell.soh[observed], color='black', lw=1.7, label='Measured SOH')
        ax.plot(future, direct, color='#246eac', lw=2,
                label='Throughput RUL + bounded SOH curve')
        ax.plot(future, forecast['bounded_half_local_rate'], color='#e49a23', lw=1.4,
                ls='--', label='Low-MAPE half-local curve (EOL unreliable)')
        ax.scatter([origin], [present], color='#246eac', s=28, zorder=3)
        ax.axhline(threshold_soh(cell), color='#bf3645', ls=':', lw=1,
                   label='Source EOL threshold')
        ax.axvline(cell.endpoint_cycle, color='gray', ls=':', lw=1,
                   label='Actual EOL')
        predicted_eol = origin+float(predicted_rul[0])
        if future[0] <= predicted_eol <= future[-1]:
            ax.axvline(predicted_eol, color='#246eac', ls='--', lw=1,
                       label='Predicted EOL')
        ax.set_title(f'{key} | {role} | curve {row["throughput_bounded_direct_curve_mape_percent"]:.1f}%'
                     f' | EOL {row["throughput_eol_ape_percent"]:.1f}%')
        ax.set_xlabel('Source cycle')
        ax.set_ylabel('SOH')
        ax.grid(alpha=.25)
        ax.legend(fontsize=7)
    fig.suptitle('Held-out XJTU: coherent curve and EOL versus low-MAPE flat-looking curve')
    fig.savefig(OUT, dpi=150)
    plt.close(fig)
    print('WROTE', OUT.resolve())


if __name__ == '__main__':
    main()
