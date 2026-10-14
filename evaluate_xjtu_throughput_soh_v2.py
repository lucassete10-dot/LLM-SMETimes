"""XJTU SOH from early throughput-aligned signals, whole-strategy held out.

The first 600 seconds of each diagnostic discharge/charge stage are sampled
at the same observed early transferred charge (up to 0.08 Ah). No complete
discharge capacity, future signal, or held-out strategy labels enter features.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from battery_v2_features import derive_signal_features, load_all
from evaluate_anchored_trajectory_v2 import get_domain_arrays
from evaluate_full_curve_practical_v2 import score_cell
from evaluate_paper_inspired_soh_v2 import fit_gated, summary
from evaluate_xjtu_long_prefix_v2 import ROOT as LONG_ROOT
from evaluate_xjtu_throughput_rul_v2 import throughput_matrix
from paper_inspired_battery_v2 import time_gated_matrix
from train_soh_v2 import outer_splits


OUT = Path('battery_models_v2/xjtu_throughput_soh_challenger')
SELECTED = Path('battery_models_v2/paper_inspired_soh_challenger/'
                'xjtu_development_oof_predictions.npz')
CURVE_ROUTE = 'bounded_half_local_rate'


def matrix(cell, raw):
    if len(raw) != len(cell.cycle):
        raise ValueError(f'{cell.cell_id}: diagnostic rows mismatch')
    delta, _ = derive_signal_features(raw, cell.reference_count)
    view = SimpleNamespace(cycle=cell.cycle, raw=raw, delta=delta)
    return time_gated_matrix(view)


def selected_oof(cells):
    with np.load(SELECTED) as data:
        rows = {name: data[name].copy() for name in data.files}
    result = {}
    for key in cells:
        mask = rows['cell_id'] == key
        np.testing.assert_array_equal(rows['cycle'][mask], cells[key].cycle)
        if not np.all(rows['held_out'][mask] == cells[key].group):
            raise ValueError(f'{key}: selected SOH fold mismatch')
        result[key] = rows['fusion'][mask]
    return result


def practical_score(cells, predictions, rul):
    rows = []
    for key in rul:
        cell = cells[key]
        result = score_cell(cell, (cell.cycle, predictions[key]), rul[key],
                            'first_available')
        if result is None:
            continue
        origin = result['origin_cycle']
        at = int(np.searchsorted(cell.cycle, origin))
        if not cell.soh_available[at]:
            continue
        rows.append({'cell_id': key, 'group': cell.group,
                     'origin_ape_percent': float(100*abs(predictions[key][at]-cell.soh[at])/
                                                 cell.soh[at]),
                     'curve_mape_percent': result['metrics'][CURVE_ROUTE]['curve_mape_percent']})
    return {'scored_cells': len(rows),
            'origin_cell_macro_mape_percent': float(np.mean([r['origin_ape_percent']
                                                              for r in rows])),
            'curve_cell_macro_mape_percent': float(np.mean([r['curve_mape_percent']
                                                             for r in rows])),
            'worst_curve_mape_percent': float(max(r['curve_mape_percent'] for r in rows)),
            'by_group': {group: {'origin_mape_percent': float(np.mean([
                r['origin_ape_percent'] for r in rows if r['group'] == group])),
                'curve_mape_percent': float(np.mean([
                    r['curve_mape_percent'] for r in rows if r['group'] == group]))}
                for group in sorted({r['group'] for r in rows})},
            'by_cell': rows}


def evaluate():
    manifest, all_cells = load_all()
    records = [r for r in manifest['cells'] if r['dataset'] == 'XJTU']
    cells = {r['cell_id']: all_cells[r['cell_id']] for r in records}
    selected = selected_oof(cells)
    features = {}
    with np.load(LONG_ROOT / 'features.npz') as archive:
        for key, cell in cells.items():
            np.testing.assert_array_equal(archive[f'{key}_cycle'], cell.cycle)
            raw = throughput_matrix(archive[f'{key}_600s'])
            features[key] = matrix(cell, raw)
    prediction = {}
    folds = []
    for held, train, val in outer_splits(records, 'XJTU'):
        imputer, model = fit_gated(cells, train, features)
        candidate = {key: model.predict(imputer.transform(features[key])) for key in val}
        prediction.update(candidate)
        fixed_half = {key: .5*(candidate[key]+selected[key]) for key in val}
        fold = {'held_out': held, 'training_cells': len(train),
                'validation_cells': len(val),
                'selected_fusion': summary(cells, val, selected),
                'throughput_gated': summary(cells, val, candidate),
                'fixed_half': summary(cells, val, fixed_half)}
        folds.append(fold)
        print(held, {name: round(fold[name]['cell_macro_mape_percent'], 2)
                     for name in ('selected_fusion', 'throughput_gated', 'fixed_half')},
              flush=True)
    half = {key: .5*(prediction[key]+selected[key]) for key in cells}
    _, rul = get_domain_arrays('XJTU')
    report = {'status': 'development whole-strategy CV; no independent test or model promotion',
              'input': 'first 600 seconds and 0.08 Ah of each diagnostic stage; no capacity label',
              'selection_warning': 'this window and candidate follow prior development diagnostics',
              'folds': folds,
              'summary': {name: summary(cells, list(cells), values)
                          for name, values in (('selected_fusion', selected),
                                               ('throughput_gated', prediction),
                                               ('fixed_half', half))},
              'practical_curve': {name: practical_score(cells, values, rul)
                                  for name, values in (('selected_fusion', selected),
                                                       ('throughput_gated', prediction),
                                                       ('fixed_half', half))}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'development_cv.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    ids = list(cells)
    np.savez_compressed(OUT / 'development_oof_predictions.npz',
                        cell_id=np.concatenate([np.full(len(cells[k].cycle), k) for k in ids]),
                        cycle=np.concatenate([cells[k].cycle for k in ids]),
                        selected_fusion=np.concatenate([selected[k] for k in ids]),
                        throughput_gated=np.concatenate([prediction[k] for k in ids]),
                        fixed_half=np.concatenate([half[k] for k in ids]))
    print('SUMMARY', {name: round(result['cell_macro_mape_percent'], 2)
                      for name, result in report['summary'].items()}, flush=True)
    print('PRACTICAL', {name: round(result['curve_cell_macro_mape_percent'], 2)
                        for name, result in report['practical_curve'].items()}, flush=True)
    return report


if __name__ == '__main__':
    evaluate()
