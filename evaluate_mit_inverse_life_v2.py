"""MIT whole-batch early endpoint ablation with extrapolating linear targets.

One prediction per battery at predeclared checkpoints. Features use only the
first 300 s of each available discharge, never capacity or future cycles.
The inverse-life target tests extrapolation beyond the held-in life range.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from battery_v2_features import load_cell, load_manifest
from evaluate_xjtu_age_blend_v2 import read_oof
from train_soh_v2 import outer_splits


OUT = Path('battery_models_v2/mit_inverse_life_challenger')
BASE = Path('battery_models_v2/rul/mit/development_oof_predictions.npz')
CHECKPOINTS = (20, 50, 100)
ALPHA = 100.0


def checkpoint_feature(cell, checkpoint):
    indices = np.flatnonzero(cell.cycle <= checkpoint)
    if not len(indices):
        return None
    i = int(indices[-1])
    if i < max(15, cell.reference_count) or cell.cycle[i] < checkpoint-2:
        return None
    previous = np.flatnonzero(cell.cycle <= cell.cycle[i]-10)
    if not len(previous):
        return None
    j = int(previous[-1])
    # Voltage samples are columns 0:11; 33:51 are short-prefix summaries.
    voltage = cell.raw[i, :11]
    voltage_change = cell.delta[i, :11]
    recent = (cell.raw[i, :11]-cell.raw[j, :11])/(cell.cycle[i]-cell.cycle[j])
    summaries = cell.raw[i, [33, 34, 35, 44, 45, 46]]
    return np.r_[voltage, voltage_change, recent, summaries]


def fit_model(features, endpoint, target):
    y = np.asarray(endpoint, dtype=float)
    if target == 'inverse':
        y = 1000/y
    elif target == 'log':
        y = np.log(y)
    else:
        raise ValueError('unknown target')
    model = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True),
                          StandardScaler(), Ridge(alpha=ALPHA))
    model.fit(np.stack(features), y)
    return model


def predict_endpoint(model, feature, target, observed_cycle):
    value = float(model.predict(feature[None, :])[0])
    if target == 'inverse':
        endpoint = 1000/max(value, 1000/2500)
    elif target == 'log':
        endpoint = np.exp(value)
    else:
        raise ValueError('unknown target')
    return float(np.clip(endpoint, observed_cycle+1, 2500))


def main():
    records = [row for row in load_manifest()['cells'] if row['dataset'] == 'MIT']
    cells = {row['cell_id']: load_cell(row) for row in records}
    baseline = read_oof(BASE, 'estimate')
    folds = []
    for group, train, validation in outer_splits(records, 'MIT'):
        fold = {'held_out': group, 'checkpoints': {}}
        for checkpoint in CHECKPOINTS:
            training = {key: checkpoint_feature(cells[key], checkpoint)
                        for key in train if cells[key].endpoint_cycle is not None}
            training = {key: x for key, x in training.items() if x is not None}
            models = {target: fit_model(list(training.values()),
                      [cells[key].endpoint_cycle for key in training], target)
                      for target in ('inverse', 'log')}
            rows = []
            for key in validation:
                cell = cells[key]
                if cell.endpoint_cycle is None or cell.endpoint_cycle <= checkpoint:
                    continue
                x = checkpoint_feature(cell, checkpoint)
                if x is None:
                    continue
                observed_cycle = int(cell.cycle[np.flatnonzero(cell.cycle <= checkpoint)[-1]])
                if (key, observed_cycle) not in baseline:
                    raise ValueError(f'missing OOF direct RUL: {key}, {observed_cycle}')
                actual = cell.endpoint_cycle
                estimate = {'baseline': observed_cycle+baseline[key, observed_cycle]}
                estimate.update({name: predict_endpoint(model, x, name, observed_cycle)
                                 for name, model in models.items()})
                rows.append({'cell_id': key, 'true_endpoint_cycle': actual,
                             'observed_cycle': observed_cycle,
                             'predicted_endpoint_cycle': estimate,
                             'ape_percent': {name: float(100*abs(value-actual)/actual)
                                             for name, value in estimate.items()}})
            fold['checkpoints'][str(checkpoint)] = {
                'training_event_cells': len(training), 'scored_cells': len(rows),
                'mape_percent': {name: float(np.mean([r['ape_percent'][name]
                                                    for r in rows]))
                                 for name in ('baseline', 'inverse', 'log')},
                'by_cell': rows}
            print(group, checkpoint, fold['checkpoints'][str(checkpoint)]['mape_percent'],
                  flush=True)
        folds.append(fold)
    summary = {}
    for checkpoint in CHECKPOINTS:
        rows = [row for fold in folds for row in
                fold['checkpoints'][str(checkpoint)]['by_cell']]
        summary[str(checkpoint)] = {
            'scored_cells': len(rows),
            'mape_percent': {name: float(np.mean([row['ape_percent'][name]
                                                for row in rows]))
                             for name in ('baseline', 'inverse', 'log')}}
    report = {'status': 'unpromoted MIT whole-batch development ablation',
              'input': 'capacity-free 300 s discharge voltage signal up to checkpoint',
              'endpoint': 'MIT source 0.88 Ah; 12 right-censored batteries unscored',
              'target_models': ['inverse life', 'log life'],
              'summary': summary, 'folds': folds}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'development_cv.json').write_text(json.dumps(report, indent=2),
                                              encoding='utf-8')
    print('SUMMARY', summary)


if __name__ == '__main__':
    main()
