"""Fixed 50/50 causal SOH ensemble on aligned MIT whole-batch OOF rows."""

import json
from pathlib import Path

import numpy as np

from battery_v2_features import load_manifest


ROOT = Path('battery_models_v2/soh_age_signal_challenger')
BASE = ROOT / 'mit_development_oof_predictions.npz'
CHALLENGER = ROOT / 'mit_histgb_development_oof.npz'
OUT = ROOT / 'mit_histgb_fixed_blend_audit.json'


def main():
    with np.load(BASE) as file:
        base = {name: file[name].copy() for name in file.files}
    with np.load(CHALLENGER) as file:
        challenger = {name: file[name].copy() for name in file.files}
    for field in ('cell_id', 'cycle', 'soh_available'):
        if not np.array_equal(base[field], challenger[field]):
            raise ValueError(f'OOF row alignment mismatch: {field}')
    if not np.allclose(base['target'], challenger['target'], equal_nan=True):
        raise ValueError('OOF targets differ')
    groups = {row['cell_id']: row['group'] for row in load_manifest()['cells']
              if row['dataset'] == 'MIT'}
    target = base['target']
    predictions = {'age_extra_trees': base['gated_with_age'],
                   'robust_histgb': challenger['estimate'],
                   'fixed_half_blend': .5*(base['gated_with_age']+
                                          challenger['estimate'])}
    rows = []
    for key in sorted(groups):
        mask = (base['cell_id'] == key) & base['soh_available']
        if not mask.any():
            continue
        rows.append({'cell_id': key, 'held_out_batch': groups[key],
                     'mape_percent': {name: float(100*np.mean(
                         np.abs(pred[mask]-target[mask])/target[mask]))
                         for name, pred in predictions.items()}})
    names = tuple(predictions)
    report = {
        'status': 'unpromoted MIT whole-batch fixed SOH ensemble diagnostic',
        'input': 'two aligned OOF causal 300 s models; fixed 50/50 average',
        'summary_mape_percent': {name: float(np.mean([
            row['mape_percent'][name] for row in rows])) for name in names},
        'by_batch_mape_percent': {group: {name: float(np.mean([
            row['mape_percent'][name] for row in rows
            if row['held_out_batch'] == group])) for name in names}
            for group in sorted(set(groups.values()))},
        'by_cell': rows,
    }
    OUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('SUMMARY', report['summary_mape_percent'])
    print('BATCHES', report['by_batch_mape_percent'])
    print('WROTE', OUT)


if __name__ == '__main__':
    main()
