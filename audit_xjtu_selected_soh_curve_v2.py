"""Recheck XJTU future curves with the actually selected OOF fusion SOH route."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from battery_v2_features import load_all
from evaluate_anchored_trajectory_v2 import get_domain_arrays
from evaluate_full_curve_practical_v2 import project, score_cell
from train_trajectory_v2 import threshold_soh


ROOT = Path('battery_models_v2')
OUT = ROOT / 'xjtu_selected_soh_curve_audit.json'
FIGURE = ROOT / 'xjtu_selected_soh_curve_examples.png'
ROUTE = 'bounded_half_local_rate'


def selected_soh_arrays(cells):
    path = ROOT / 'paper_inspired_soh_challenger/xjtu_development_oof_predictions.npz'
    with np.load(path) as data:
        rows = {name: data[name].copy() for name in data.files}
    result = {}
    for key in np.unique(rows['cell_id']):
        key = str(key)
        mask = rows['cell_id'] == key
        np.testing.assert_array_equal(rows['cycle'][mask], cells[key].cycle)
        if not np.all(rows['held_out'][mask] == cells[key].group):
            raise ValueError(f'{key}: held-out group mismatch')
        result[key] = (rows['cycle'][mask], rows['fusion'][mask])
    return result


def plot_cases(cells, baseline, selected, rul, cases):
    fig, axes = plt.subplots(1, len(cases), figsize=(7*len(cases), 5),
                             constrained_layout=True)
    if len(cases) == 1:
        axes = [axes]
    for ax, case in zip(axes, cases):
        key = case['cell_id']
        cell = cells[key]
        cycle, direct_rul, local_rul = rul[key]
        origin = int(cycle[0])
        future = np.flatnonzero((cell.cycle >= origin) &
                                (cell.cycle <= cell.endpoint_cycle) & cell.soh_available)
        ax.plot(cell.cycle[future], cell.soh[future], 'k-', label='Measured SOH')
        threshold = threshold_soh(cell)
        for name, estimates, color in (('Old baseline SOH', baseline, '#d97219'),
                                        ('Selected fusion SOH', selected, '#2779b7')):
            sc, se = estimates[key]
            s = float(se[np.searchsorted(sc, origin)])
            grid = cell.cycle[future]
            prediction = project(s, threshold, direct_rul[0], local_rul[0],
                                 grid-origin)[ROUTE]
            ax.plot(grid, prediction, lw=2, color=color, label=name)
            ax.scatter([origin], [s], color=color, s=22, zorder=3)
        ax.axhline(threshold, color='#be3242', ls='--', lw=1, label='Source EOL threshold')
        ax.axvline(cell.endpoint_cycle, color='gray', ls=':', lw=1, label='Actual EOL')
        ax.set_title(f'{key} | held-out {cell.group}')
        ax.set_xlabel('Source cycle')
        ax.set_ylabel('SOH')
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.suptitle('XJTU first-origin full curve: correcting the current-SOH input')
    fig.savefig(FIGURE, dpi=150)
    plt.close(fig)


def evaluate():
    _, cells = load_all()
    baseline, rul = get_domain_arrays('XJTU')
    selected = selected_soh_arrays(cells)
    rows = []
    for key in rul:
        old = score_cell(cells[key], baseline[key], rul[key], 'first_available')
        new = score_cell(cells[key], selected[key], rul[key], 'first_available')
        if old is None or new is None:
            continue
        origin = old['origin_cycle']
        ci = int(np.searchsorted(cells[key].cycle, origin))
        si = int(np.searchsorted(selected[key][0], origin))
        bi = int(np.searchsorted(baseline[key][0], origin))
        truth = float(cells[key].soh[ci])
        rows.append({'cell_id': key, 'group': cells[key].group,
                     'origin_cycle': origin,
                     'baseline_origin_ape_percent': 100*abs(baseline[key][1][bi]-truth)/truth,
                     'selected_origin_ape_percent': 100*abs(selected[key][1][si]-truth)/truth,
                     'baseline_curve_mape_percent': old['metrics'][ROUTE]['curve_mape_percent'],
                     'selected_curve_mape_percent': new['metrics'][ROUTE]['curve_mape_percent'],
                     'eol_ape_percent': new['selected_rul_eol_ape_percent']})
    groups = sorted({r['group'] for r in rows})
    summary = {name: float(np.mean([r[name] for r in rows])) for name in (
        'baseline_origin_ape_percent', 'selected_origin_ape_percent',
        'baseline_curve_mape_percent', 'selected_curve_mape_percent', 'eol_ape_percent')}
    summary['scored_cells'] = len(rows)
    summary['equal_group_curve_mape_percent'] = {
        name: float(np.mean([np.mean([r[name] for r in rows if r['group'] == group])
                             for group in groups]))
        for name in ('baseline_curve_mape_percent', 'selected_curve_mape_percent')}
    report = {'status': 'development whole-strategy held-out diagnostic; not independent test',
              'SOH_route': 'fixed 50/50 raw and time-gated fusion',
              'RUL_route': 'unchanged baseline OOF',
              'future_curve_route': ROUTE,
              'summary': summary,
              'worst_selected_cells': sorted(rows, key=lambda r: r[
                  'selected_curve_mape_percent'], reverse=True)[:5],
              'by_group': {group: {'scored_cells': sum(r['group'] == group for r in rows),
                                   'selected_curve_mape_percent': float(np.mean([
                                       r['selected_curve_mape_percent'] for r in rows
                                       if r['group'] == group]))}
                           for group in groups},
              'by_cell': rows}
    OUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    median = sorted(rows, key=lambda r: r['selected_curve_mape_percent'])[len(rows)//2]
    worst = max(rows, key=lambda r: r['selected_curve_mape_percent'])
    plot_cases(cells, baseline, selected, rul, [median, worst])
    print('XJTU selected-SOH comparison', summary, flush=True)
    print('WROTE', OUT, FIGURE, flush=True)
    return report


if __name__ == '__main__':
    evaluate()
