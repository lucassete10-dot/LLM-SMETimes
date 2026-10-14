"""MIT OOF SOH audit of past-only short-window prediction smoothing.

No true SOH or future prediction enters the smoother. Candidate window
widths are fixed in advance at 3/5/11 observed cycles. This remains a
development ablation because the underlying age model was explored here.
"""

import json
from pathlib import Path

import numpy as np


ROOT = Path('battery_models_v2/soh_age_signal_challenger')
SOURCE = ROOT / 'mit_development_oof_predictions.npz'
OUT = ROOT / 'mit_causal_smoothing_audit.json'
WINDOWS = (3, 5, 11)


def past_median(cycle, prediction, width):
    cycle = np.asarray(cycle)
    prediction = np.asarray(prediction, dtype=float)
    if cycle.shape != prediction.shape or np.any(np.diff(cycle) <= 0):
        raise ValueError('unaligned or non-increasing observations')
    if width < 1:
        raise ValueError('window must be positive')
    return np.array([np.median(prediction[(cycle >= c-width+1) & (cycle <= c)])
                     for c in cycle])


def main():
    with np.load(SOURCE) as file:
        data = {name: file[name].copy() for name in file.files}
    rows = []
    for key in np.unique(data['cell_id']):
        keep_cell = data['cell_id'] == key
        cycle = data['cycle'][keep_cell]
        true = data['target'][keep_cell]
        score = data['soh_available'][keep_cell]
        original = data['gated_with_age'][keep_cell]
        predictions = {'original': original}
        predictions.update({f'median_{width}': past_median(cycle, original, width)
                            for width in WINDOWS})
        rows.append({'cell_id': str(key),
                     'mape_percent': {name: float(100*np.mean(
                         abs(value[score]-true[score])/true[score]))
                                      for name, value in predictions.items()}})
    names = ('original', *(f'median_{width}' for width in WINDOWS))
    report = {'status': 'unpromoted MIT whole-batch OOF post-processing ablation',
              'input': 'past and current held-out SOH predictions only',
              'windows_observed_cycles': WINDOWS,
              'summary': {'scored_cells': len(rows),
                          'mape_percent': {name: float(np.mean([
                              row['mape_percent'][name] for row in rows]))
                                           for name in names}},
              'by_cell': rows}
    OUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('SUMMARY', report['summary'])


if __name__ == '__main__':
    main()
