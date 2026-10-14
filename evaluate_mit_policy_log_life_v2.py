"""MIT whole-batch ablation: known charge policy in the log-life ridge model.

The policy is declared before cycling. Both models use the same checkpoint
batteries and source 0.88-Ah endpoint, with no capacity inputs.
"""

import json
from pathlib import Path

import numpy as np

from battery_v2_features import load_cell, load_manifest
from evaluate_mit_inverse_life_v2 import (CHECKPOINTS, OUT, checkpoint_feature,
                                          fit_model, predict_endpoint)
from evaluate_mit_policy_rul_v2 import policy_values
from train_soh_v2 import outer_splits


BASE = OUT / 'development_cv.json'
TARGET = OUT / 'policy_log_cv.json'


def main():
    records = [row for row in load_manifest()['cells'] if row['dataset'] == 'MIT']
    cells = {row['cell_id']: load_cell(row) for row in records}
    policy = {row['cell_id']: policy_values(row) for row in records}
    signal_only = json.loads(BASE.read_text(encoding='utf-8'))
    old = {(int(checkpoint), row['cell_id']): row
           for fold in signal_only['folds'] for checkpoint in CHECKPOINTS
           for row in fold['checkpoints'][str(checkpoint)]['by_cell']}
    folds = []
    for group, train, validation in outer_splits(records, 'MIT'):
        fold = {'held_out': group, 'checkpoints': {}}
        for checkpoint in CHECKPOINTS:
            training = {key: checkpoint_feature(cells[key], checkpoint)
                        for key in train if cells[key].endpoint_cycle is not None}
            training = {key: x for key, x in training.items() if x is not None}
            model = fit_model([np.r_[x, policy[key]] for key, x in training.items()],
                              [cells[key].endpoint_cycle for key in training], 'log')
            rows = []
            for key in validation:
                if (checkpoint, key) not in old:
                    continue
                cell = cells[key]
                feature = checkpoint_feature(cell, checkpoint)
                if feature is None:
                    raise ValueError(f'missing feature for previously scored {key}')
                reference = old[checkpoint, key]
                predicted = predict_endpoint(model, np.r_[feature, policy[key]],
                                             'log', reference['observed_cycle'])
                actual = cell.endpoint_cycle
                rows.append({'cell_id': key,
                             'observed_cycle': reference['observed_cycle'],
                             'true_endpoint_cycle': actual,
                             'policy_log_endpoint_cycle': predicted,
                             'ape_percent': {
                                 'baseline': reference['ape_percent']['baseline'],
                                 'signal_log': reference['ape_percent']['log'],
                                 'policy_log': float(100*abs(predicted-actual)/actual)}})
            fold['checkpoints'][str(checkpoint)] = {
                'training_event_cells': len(training), 'scored_cells': len(rows),
                'mape_percent': {name: float(np.mean([row['ape_percent'][name]
                                                    for row in rows]))
                                 for name in ('baseline', 'signal_log', 'policy_log')},
                'by_cell': rows}
            print(group, checkpoint,
                  fold['checkpoints'][str(checkpoint)]['mape_percent'], flush=True)
        folds.append(fold)
    summary = {}
    for checkpoint in CHECKPOINTS:
        rows = [row for fold in folds for row in
                fold['checkpoints'][str(checkpoint)]['by_cell']]
        summary[str(checkpoint)] = {
            'scored_cells': len(rows),
            'mape_percent': {name: float(np.mean([row['ape_percent'][name]
                                                for row in rows]))
                             for name in ('baseline', 'signal_log', 'policy_log')}}
    report = {'status': 'unpromoted MIT whole-batch development policy ablation',
              'input': 'causal 300 s discharge voltage and predeclared charge policy',
              'endpoint': 'source 0.88 Ah; censored cells excluded',
              'summary': summary, 'folds': folds}
    TARGET.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('SUMMARY', summary)


if __name__ == '__main__':
    main()
