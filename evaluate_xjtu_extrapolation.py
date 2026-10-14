"""Development-only physics-style XJTU RUL extrapolation diagnostic."""

import json
from pathlib import Path

import numpy as np

from battery_v2_features import load_all
from train_soh_v2 import fit, predict, outer_splits
from train_rul_v2 import assess


def extrapolate(cell, soh, threshold, window, rate_floor):
    result = np.zeros(len(soh), dtype='float32')
    cycle = cell.cycle
    ref = max(0, cell.reference_count-1)
    for i in range(len(soh)):
        past = max(ref, i-window)
        if i <= ref:
            result[i] = np.nan
            continue
        # Median-filter the causal SOH path to suppress diagnostic spikes.
        now = float(np.median(soh[max(ref, i-4):i+1]))
        earlier = float(np.median(soh[max(ref, past-4):past+1]))
        baseline = float(np.median(soh[ref:min(len(soh), ref+5)]))
        local_rate = (earlier-now) / max(1, cycle[i]-cycle[past])
        global_rate = (baseline-now) / max(1, cycle[i]-cycle[ref])
        rate = max(rate_floor, .5*local_rate + .5*global_rate)
        result[i] = min(2500., max(0., (now-threshold)/rate))
    return result


def main():
    manifest, cells = load_all()
    baseline_folds = {row['held_out']: row for row in json.loads(
        Path('battery_models_v2/rul/xjtu/group_cv.json').read_text())}
    rows = []
    for group, train, val in outer_splits(manifest['cells'], 'XJTU'):
        model = fit(cells, train, 'extra_trees_history')
        threshold = float(np.median([cells[key].soh[-1] for key in train]))
        soh = {key: predict(model, cells[key]) for key in val}
        base = float(np.median([cells[key].endpoint_cycle for key in train]))
        variants = {}
        for window in (25, 50, 100):
            for floor in (.00005, .0001, .0002):
                name = f'w{window}_floor{floor}'
                preds = {key: extrapolate(cells[key], soh[key], threshold, window, floor)
                         for key in val}
                variants[name] = assess(cells, val, preds, base)['cell_macro_mae_cycles']
        rows.append({'held_out': group, 'threshold': threshold,
                     'direct_mae_cycles': baseline_folds[group]['direct']['cell_macro_mae_cycles'],
                     'extrapolation_macro_mae_cycles': variants})
        print(group, 'direct', round(rows[-1]['direct_mae_cycles'], 1),
              'best_extrapolation', min(variants.items(), key=lambda pair: pair[1]), flush=True)
    output = Path('battery_models_v2/rul/xjtu/extrapolation_diagnostic.json')
    output.write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print('GROUP MEANS', {name: round(float(np.mean([row['extrapolation_macro_mae_cycles'][name]
                                                  for row in rows])), 1)
                          for name in rows[0]['extrapolation_macro_mae_cycles']})


if __name__ == '__main__':
    main()
