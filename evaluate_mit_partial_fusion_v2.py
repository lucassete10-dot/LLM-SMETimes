"""MIT cycle-100 300 s prefix voltage + partial ΔQ(V) fusion ablation."""

import json
from pathlib import Path

import numpy as np

from battery_v2_features import load_cell, load_manifest
from evaluate_mit_inverse_life_v2 import checkpoint_feature, fit_model, predict_endpoint
from evaluate_mit_partial_deltaq_v2 import OUT, load_statistics
from train_soh_v2 import outer_splits


PRIOR = Path('battery_models_v2/mit_inverse_life_challenger/development_cv.json')
PARTIAL = OUT / 'partial_deltaq_cv.json'
TARGET = OUT / 'partial_fusion_cv.json'
CHECKPOINT = 100


def main():
    records = [row for row in load_manifest()['cells'] if row['dataset'] == 'MIT']
    by_id = {row['cell_id']: row for row in records}
    cells = {row['cell_id']: load_cell(row) for row in records}
    partial = load_statistics(records)
    signal = {key: checkpoint_feature(cell, CHECKPOINT)
              for key, cell in cells.items()}
    previous = json.loads(PRIOR.read_text(encoding='utf-8'))
    signal_oof = {row['cell_id']: row for fold in previous['folds'] for row in
                  fold['checkpoints'][str(CHECKPOINT)]['by_cell']}
    partial_oof = json.loads(PARTIAL.read_text(encoding='utf-8'))
    partial_oof = {row['cell_id']: row for fold in partial_oof['folds']
                   for row in fold['by_cell']}
    folds = []
    for group, train, validation in outer_splits(records, 'MIT'):
        training = [key for key in train if key in partial and signal[key] is not None
                    and by_id[key]['endpoint_cycle'] is not None]
        model = fit_model([np.r_[signal[key], partial[key][:2]] for key in training],
                          [by_id[key]['endpoint_cycle'] for key in training], 'log')
        rows = []
        for key in validation:
            if key not in signal_oof or key not in partial_oof:
                continue
            if signal[key] is None or key not in partial:
                continue
            true = by_id[key]['endpoint_cycle']
            observed = signal_oof[key]['observed_cycle']
            estimate = predict_endpoint(model, np.r_[signal[key], partial[key][:2]],
                                        'log', observed)
            rows.append({'cell_id': key, 'true_endpoint_cycle': true,
                         'predicted_endpoint_cycle': estimate,
                         'ape_percent': {
                             'signal_log': signal_oof[key]['ape_percent']['log'],
                             'partial_deltaq': partial_oof[key]['ape_percent'][
                                 'partial_variance_min'],
                             'fused': float(100*abs(estimate-true)/true)}})
        folds.append({'held_out': group, 'training_event_cells': len(training),
                      'scored_cells': len(rows),
                      'mape_percent': {name: float(np.mean([r['ape_percent'][name]
                                                          for r in rows]))
                                       for name in ('signal_log', 'partial_deltaq',
                                                    'fused')},
                      'by_cell': rows})
        print(group, folds[-1]['mape_percent'], flush=True)
    rows = [row for fold in folds for row in fold['by_cell']]
    report = {'status': 'unpromoted MIT whole-batch 300 s prefix fusion ablation',
              'measurement': 'first 300 s of cycles 10/100, no full-cycle capacity',
              'endpoint_warning': 'source 0.88 Ah, not per-cell exact 80%',
              'summary': {'scored_cells': len(rows),
                          'mape_percent': {name: float(np.mean([
                              row['ape_percent'][name] for row in rows]))
                                           for name in ('signal_log', 'partial_deltaq',
                                                        'fused')}},
              'folds': folds}
    TARGET.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('SUMMARY', report['summary'])


if __name__ == '__main__':
    main()
