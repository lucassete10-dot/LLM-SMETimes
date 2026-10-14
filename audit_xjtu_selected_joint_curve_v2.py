"""Audit a single XJTU selected-SOH/selected-RUL route end to end.

The old low-MAPE half-local curve ignores the selected direct RUL. Here the
direct linear route is forced to cross the source EOL threshold at its RUL.
Its accuracy is therefore a more honest joint curve/life diagnostic.
"""

import json
from pathlib import Path

import numpy as np

from battery_v2_features import load_all
from evaluate_anchored_trajectory_v2 import get_domain_arrays
from evaluate_full_curve_practical_v2 import project, score_cell
from train_rul_v2 import eligible
from train_trajectory_v2 import threshold_soh


ROOT = Path('battery_models_v2')
OUT = ROOT / 'xjtu_selected_joint_curve_audit.json'


def selected_soh(cells):
    with np.load(ROOT / 'paper_inspired_soh_challenger/'
                 'xjtu_development_oof_predictions.npz') as archive:
        data = {name: archive[name].copy() for name in archive.files}
    result = {}
    for key, cell in cells.items():
        mask = data['cell_id'] == key
        np.testing.assert_array_equal(data['cycle'][mask], cell.cycle)
        result[key] = (data['cycle'][mask], data['fusion'][mask])
    return result


def selected_rul(cells, baseline_rul):
    with np.load(ROOT / 'paper_inspired_soh_challenger/'
                 'xjtu_rul_development_oof.npz') as archive:
        data = {name: archive[name].copy() for name in archive.files}
    result = {}
    for key, cell in cells.items():
        mask = data['cell_id'] == key
        expected = cell.cycle[eligible(cell)]
        np.testing.assert_array_equal(data['cycle'][mask], expected)
        old_cycles, _, local = baseline_rul[key]
        np.testing.assert_array_equal(old_cycles, expected)
        result[key] = (expected, data['gated_log'][mask], local)
    return result


def throughput_rul(cells, baseline_rul):
    with np.load(ROOT / 'xjtu_throughput_rul_challenger/'
                 'development_oof_throughput_full.npz') as archive:
        data = {name: archive[name].copy() for name in archive.files}
    result = {}
    for key, cell in cells.items():
        mask = data['cell_id'] == key
        old_cycles, _, local = baseline_rul[key]
        np.testing.assert_array_equal(data['cycle'][mask], old_cycles)
        result[key] = (old_cycles, data['estimate'][mask], local)
    return result


def checkpoint_index(cell, cycles, checkpoint):
    if checkpoint == 'first_available':
        chosen = cycles[:1]
    elif checkpoint == 'cycle_50':
        chosen = cycles[cycles == 50][:1]
    else:
        fraction = .2 if checkpoint == 'life_20_percent' else .5
        chosen = cycles[cycles <= fraction*cell.endpoint_cycle][-1:]
    if len(chosen) == 0 or chosen[0] >= cell.endpoint_cycle:
        return None
    return int(np.searchsorted(cycles, chosen[0]))


def checkpoint_audit(cells, old_rul, new_rul):
    result = {}
    for checkpoint in ('first_available', 'cycle_50',
                       'life_20_percent', 'life_50_percent'):
        rows = []
        for key in new_rul:
            cell = cells[key]
            cycles, candidate, _ = new_rul[key]
            i = checkpoint_index(cell, cycles, checkpoint)
            if i is None:
                continue
            old_cycles, old, _ = old_rul[key]
            j = int(np.searchsorted(old_cycles, cycles[i]))
            if j >= len(old_cycles) or old_cycles[j] != cycles[i]:
                raise ValueError(f'{key}: RUL route cycle mismatch')
            rows.append({'cell_id': key, 'group': cell.group,
                         'origin_cycle': int(cycles[i]),
                         'baseline_ape_percent': float(100*abs(cycles[i]+old[j]-
                                                                  cell.endpoint_cycle)/
                                                               cell.endpoint_cycle),
                         'selected_ape_percent': float(100*abs(cycles[i]+candidate[i]-
                                                                  cell.endpoint_cycle)/
                                                               cell.endpoint_cycle)})
        result[checkpoint] = {
            'scored_cells': len(rows),
            'baseline_mape_percent': float(np.mean([r['baseline_ape_percent'] for r in rows])),
            'selected_mape_percent': float(np.mean([r['selected_ape_percent'] for r in rows])),
            'by_group': {group: {'cells': sum(r['group'] == group for r in rows),
                                 'selected_mape_percent': float(np.mean([
                                     r['selected_ape_percent'] for r in rows
                                     if r['group'] == group]))}
                         for group in sorted({r['group'] for r in rows})}}
    return result


def evaluate():
    _, all_cells = load_all()
    cells = {key: cell for key, cell in all_cells.items() if cell.dataset == 'XJTU'}
    _, old_rul = get_domain_arrays('XJTU')
    soh = selected_soh(cells)
    new_rul = selected_rul(cells, old_rul)
    throughput = throughput_rul(cells, old_rul)
    rows = []
    for key in new_rul:
        old = score_cell(cells[key], soh[key], old_rul[key], 'first_available')
        new = score_cell(cells[key], soh[key], new_rul[key], 'first_available')
        aligned = score_cell(cells[key], soh[key], throughput[key], 'first_available')
        if old is None or new is None or aligned is None:
            continue
        cell = cells[key]
        origin = new['origin_cycle']
        sc, se = soh[key]
        current = float(se[np.searchsorted(sc, origin)])
        rc, direct_rul, local_rul = new_rul[key]
        future = np.flatnonzero((cell.cycle > origin) &
                                (cell.cycle <= cell.endpoint_cycle) & cell.soh_available)
        curves = project(current, threshold_soh(cell), direct_rul[0], local_rul[0],
                         cell.cycle[future]-origin)
        truth = cell.soh[future]
        bounded_direct = np.clip(curves['direct_linear'], 0, current)
        bounded_anchored = np.clip(curves['anchored'], 0, current)
        _, throughput_direct_rul, _ = throughput[key]
        throughput_curves = project(current, threshold_soh(cell), throughput_direct_rul[0],
                                    local_rul[0], cell.cycle[future]-origin)
        throughput_bounded_direct = np.clip(throughput_curves['direct_linear'], 0, current)
        throughput_bounded_anchored = np.clip(throughput_curves['anchored'], 0, current)
        rows.append({'cell_id': key, 'group': cells[key].group,
                     'baseline_direct_curve_mape_percent': old['metrics'][
                         'direct_linear']['curve_mape_percent'],
                     'selected_direct_curve_mape_percent': new['metrics'][
                         'direct_linear']['curve_mape_percent'],
                     'selected_anchored_curve_mape_percent': new['metrics'][
                         'anchored']['curve_mape_percent'],
                     'selected_bounded_direct_curve_mape_percent': float(
                         100*np.mean(abs(bounded_direct-truth)/truth)),
                     'selected_bounded_anchored_curve_mape_percent': float(
                         100*np.mean(abs(bounded_anchored-truth)/truth)),
                     'throughput_bounded_direct_curve_mape_percent': float(
                         100*np.mean(abs(throughput_bounded_direct-truth)/truth)),
                     'throughput_bounded_anchored_curve_mape_percent': float(
                         100*np.mean(abs(throughput_bounded_anchored-truth)/truth)),
                     'selected_half_local_curve_mape_percent': new['metrics'][
                         'bounded_half_local_rate']['curve_mape_percent'],
                     'baseline_eol_ape_percent': old['selected_rul_eol_ape_percent'],
                     'selected_eol_ape_percent': new['selected_rul_eol_ape_percent'],
                     'throughput_eol_ape_percent': aligned['selected_rul_eol_ape_percent']})
    summary = {'scored_cells': len(rows)}
    for field in ('baseline_direct_curve_mape_percent',
                  'selected_direct_curve_mape_percent',
                  'selected_anchored_curve_mape_percent',
                  'selected_bounded_direct_curve_mape_percent',
                  'selected_bounded_anchored_curve_mape_percent',
                  'throughput_bounded_direct_curve_mape_percent',
                  'throughput_bounded_anchored_curve_mape_percent',
                  'selected_half_local_curve_mape_percent',
                  'baseline_eol_ape_percent', 'selected_eol_ape_percent',
                  'throughput_eol_ape_percent'):
        summary[field] = float(np.mean([r[field] for r in rows]))
    report = {'status': 'development whole-strategy held-out diagnostic; no promotion',
              'route': 'selected fusion SOH and selected gated-log RUL',
              'warning': 'MIT/XJTU endpoint definitions remain source-specific; '
                         'low half-local curve error does not imply correct EOL crossing',
              'summary': summary,
              'checkpoints': checkpoint_audit(cells, old_rul, new_rul),
              'by_group': {group: {'cells': sum(r['group'] == group for r in rows),
                                   'selected_direct_curve_mape_percent': float(np.mean([
                                       r['selected_direct_curve_mape_percent'] for r in rows
                                       if r['group'] == group]))}
                           for group in sorted({r['group'] for r in rows})},
              'by_cell': rows}
    OUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('JOINT', summary, flush=True)
    print('CHECKPOINTS', {key: (row['scored_cells'], round(row[
        'selected_mape_percent'], 2)) for key, row in report['checkpoints'].items()}, flush=True)
    return report


if __name__ == '__main__':
    evaluate()
