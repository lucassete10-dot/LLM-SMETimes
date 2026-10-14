"""Controlled label-noise stress test; never changes the MIT source data.

Injects synthetic errors in 3% of training cycles, then compares squared-error
and robust numeric models on untouched validation cells. This is a mechanism
check, not an estimate of natural MIT-data performance.
"""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from battery_mvp import compute_metrics
from battery_mit_validation import NumericPrep, load_mit, make_data, predict_model, robust_fit


def main():
    torch.set_num_threads(4)
    _, original = load_mit()
    out = Path('battery_mit_results/mit_cell_holdout')
    split = json.loads((out / 'split.json').read_text())
    train_ids, val_ids = split['train'], split['validation']
    rng = np.random.default_rng(2026)
    contaminated = dict(original)
    changed = {}
    for key in train_ids:
        battery = original[key]
        candidates = np.arange(15, len(battery.soh))
        picks = rng.choice(candidates, size=max(1, round(.03 * len(candidates))), replace=False)
        y = battery.soh.copy()
        y[picks] += rng.choice([-1., 1.], size=len(picks)) * .15
        contaminated[key] = replace(battery, soh=y)
        changed[key] = len(picks)
    prep = NumericPrep(robust=True).fit(contaminated, train_ids)
    data = make_data(prep, contaminated, train_ids)
    data.update(make_data(prep, original, val_ids))
    report = {'injected_fraction': .03, 'injected_amplitude_soh': .15,
              'injected_train_records': sum(changed.values()), 'validation_cells': val_ids,
              'models': {}}
    for name, robust in [('mse', False), ('huber_and_local_downweight', True)]:
        model, epoch, _, counts = robust_fit(data, contaminated, train_ids, val_ids[0],
                                              2026, epochs=100, patience=12, robust=robust)
        prediction = predict_model(model, data, val_ids)
        scores = {key: compute_metrics(prediction[key][15:], original[key].soh[15:])['mae']
                  for key in val_ids}
        report['models'][name] = {'best_epoch': epoch, 'local_outliers_downweighted': counts['downweighted'],
                                  'macro_val_mae': float(np.mean(list(scores.values()))),
                                  'by_cell_mae': scores}
        print(name, report['models'][name]['macro_val_mae'], flush=True)
    (out / 'synthetic_contamination.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__': main()
