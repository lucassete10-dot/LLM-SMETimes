"""Plot median-error and worst-error held-out full-curve examples per dataset."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from battery_v2_features import load_all
from evaluate_anchored_trajectory_v2 import get_domain_arrays
from evaluate_full_curve_practical_v2 import project
from train_trajectory_v2 import threshold_soh


ROOT = Path('battery_models_v2')
OUT = ROOT / 'full_curve_examples_median_worst.png'
ROUTE = {'NASA': 'local_linear', 'CALCE': 'direct_linear',
         'MIT': 'fixed_half', 'XJTU': 'bounded_half_local_rate'}


def main():
    _, cells = load_all()
    report = json.loads((ROOT / 'full_curve_practical_audit.json').read_text(encoding='utf-8'))
    fig, axes = plt.subplots(4, 2, figsize=(14, 16), constrained_layout=True)
    for row_index, domain in enumerate(ROUTE):
        route = ROUTE[domain]
        cases = report['domains'][domain]['first_available']['by_cell']
        ordered = sorted(cases, key=lambda case: case['metrics'][route]['curve_mape_percent'])
        chosen = [ordered[len(ordered)//2], ordered[-1]]
        soh_by_cell, rul_by_cell = get_domain_arrays(domain)
        for column, case in enumerate(chosen):
            ax = axes[row_index, column]
            key = case['cell_id']
            cell = cells[key]
            origin = case['origin_cycle']
            future = np.flatnonzero((cell.cycle > origin) &
                                    (cell.cycle <= cell.endpoint_cycle) & cell.soh_available)
            soh_cycle, soh_estimate = soh_by_cell[key]
            rul_cycle, rul_estimate, local_rul = rul_by_cell[key]
            s = float(soh_estimate[np.searchsorted(soh_cycle, origin)])
            position = int(np.searchsorted(rul_cycle, origin))
            pred = project(s, threshold_soh(cell), rul_estimate[position],
                           local_rul[position], cell.cycle[future]-origin)[route]
            ax.plot(cell.cycle[future], cell.soh[future], 'k-', lw=1.7, label='Measured SOH')
            ax.plot(cell.cycle[future], pred, color='#dd7022', lw=2,
                    label='Held-out forecast')
            ax.scatter([origin], [s], color='#dd7022', s=22, zorder=3,
                       label='Forecast origin')
            ax.axhline(threshold_soh(cell), color='#be3242', ls='--', lw=1,
                       label='EOL threshold')
            ax.axvline(cell.endpoint_cycle, color='#555555', ls=':', lw=1,
                       label='Actual EOL')
            ax.set_title(f'{domain}  {key}  {"median" if column == 0 else "worst"} '
                         f'({case["metrics"][route]["curve_mape_percent"]:.1f}% curve MAPE)',
                         fontsize=10)
            ax.set_xlabel('Source cycle')
            ax.set_ylabel('SOH')
            ax.grid(alpha=.25)
            ax.legend(fontsize=7, loc='best')
    fig.suptitle('Full remaining SOH curves from the first eligible observation\n'
                 'Development whole-cell/batch held-out predictions; median and worst cells',
                 fontsize=15)
    fig.savefig(OUT, dpi=155)
    print('WROTE', OUT.resolve())


if __name__ == '__main__':
    main()
