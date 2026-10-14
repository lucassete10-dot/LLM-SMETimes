"""Causal, label-free MIT age-model anchor with a decaying correction.

Uses held-out predictions on the calibration cycles (where reference SOH is
known by definition) but never the target capacity on scored cycles.
The fixed decay windows are development ablations, not a final selection.
"""

import json
from pathlib import Path

import numpy as np

from battery_v2_features import load_manifest


ROOT = Path('battery_models_v2/soh_age_signal_challenger')
SOURCE = ROOT / 'mit_development_oof_predictions.npz'
OUT = ROOT / 'mit_decaying_anchor_cv.json'
DECAY_CYCLES = (25, 50, 100, 200)


def decaying_anchor(cycle, prediction, reference_count, tau):
    cycle = np.asarray(cycle, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if len(cycle) != len(prediction) or not 1 <= reference_count < len(cycle):
        raise ValueError('unaligned prediction or invalid reference count')
    if tau <= 0 or np.any(np.diff(cycle) <= 0):
        raise ValueError('positive tau and increasing cycles required')
    reference_cycle = cycle[reference_count-1]
    offset = 1.-float(np.median(prediction[:reference_count]))
    elapsed = np.maximum(0., cycle-reference_cycle)
    return prediction + offset*np.exp(-elapsed/tau)


def main():
    manifest = load_manifest()
    records = {row['cell_id']: row for row in manifest['cells']
               if row['dataset'] == 'MIT'}
    with np.load(SOURCE) as file:
        data = {name: file[name].copy() for name in file.files}
    rows = []
    for key in sorted(records):
        mask = data['cell_id'] == key
        cycle = data['cycle'][mask]
        target = data['target'][mask]
        eligible = data['soh_available'][mask]
        original = data['gated_with_age'][mask]
        reference_count = records[key]['reference_measurement_count']
        offset = 1.-float(np.median(original[:reference_count]))
        candidates = {'original': original,
                      'constant_anchor': original+offset}
        candidates.update({f'exp_{tau}': decaying_anchor(
            cycle, original, reference_count, tau) for tau in DECAY_CYCLES})
        rows.append({
            'cell_id': key, 'held_out_batch': records[key]['group'],
            'anchor_offset_soh': offset,
            'mape_percent': {name: float(100*np.mean(np.abs(value[eligible]-
                target[eligible])/target[eligible]))
                for name, value in candidates.items()},
        })
    names = ('original', 'constant_anchor', *(f'exp_{tau}' for tau in DECAY_CYCLES))
    by_group = {group: {name: float(np.mean([r['mape_percent'][name]
        for r in rows if r['held_out_batch'] == group])) for name in names}
        for group in sorted({r['held_out_batch'] for r in rows})}
    summary = {name: float(np.mean([r['mape_percent'][name] for r in rows]))
               for name in names}
    report = {
        'status': 'unpromoted MIT whole-batch OOF calibration ablation',
        'input': 'causal 300 s model predictions through current cycle; reference predictions only',
        'summary_cell_macro_mape_percent': summary,
        'held_out_batch_mape_percent': by_group,
        'by_cell': rows,
    }
    OUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('overall', {key: round(value, 3) for key, value in summary.items()})
    for group, values in by_group.items():
        print(group, {key: round(value, 3) for key, value in values.items()})
    print('WROTE', OUT)


if __name__ == '__main__':
    main()
