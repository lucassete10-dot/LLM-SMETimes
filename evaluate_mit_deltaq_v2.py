"""Paper-inspired MIT ΔQ100-10(V) variance early-life ablation.

Uses the source MAT's within-cycle, voltage-interpolated Qdlin vectors at
observed cycles 10 and 100. No future cycle or EOL information enters the
feature. This requires full discharge diagnostics, not an online 300 s prefix.
It is a conceptual reproduction, not the paper's licensed modeling code or
its original train/test split.
"""

import json
from pathlib import Path

import h5py
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from battery_v2_features import load_manifest
from evaluate_mit_inverse_life_v2 import OUT as PRIOR_ROOT
from prepare_mit import read_ref
from train_soh_v2 import outer_splits


OUT = Path('battery_models_v2/mit_qv_challenger')
PRIOR = PRIOR_ROOT / 'development_cv.json'
GRID_LENGTH = 1000
ALPHA = 1.0


def source_qdlin(mat, index, requested):
    struct = mat['batch']
    summary = mat[struct['summary'][index, 0]]
    cycles = mat[struct['cycles'][index, 0]]
    numbers = np.asarray(summary['cycle']).reshape(-1).astype(int)
    match = np.flatnonzero(numbers == requested)
    if len(match) != 1:
        return None
    q = read_ref(mat, cycles['Qdlin'][int(match[0]), 0]).astype(float)
    if q.shape != (GRID_LENGTH,) or not np.isfinite(q).all():
        return None
    return q


def deltaq_variance(q10, q100):
    q10, q100 = np.asarray(q10, dtype=float), np.asarray(q100, dtype=float)
    if q10.shape != (GRID_LENGTH,) or q100.shape != (GRID_LENGTH,):
        raise ValueError('expected 1000-point common-voltage Qdlin vectors')
    if not (np.isfinite(q10).all() and np.isfinite(q100).all()):
        raise ValueError('nonfinite Qdlin vector')
    value = float(np.var(q100-q10))
    if value <= 0:
        raise ValueError('nonpositive ΔQ variance')
    return float(np.log10(value))


def load_features(records):
    features = {}
    groups = {}
    for group in sorted({row['group'] for row in records}):
        subset = [row for row in records if row['group'] == group]
        with h5py.File(subset[0]['measurement_source'], 'r') as mat:
            for row in subset:
                index = int(row['cell_id'].split('c', 1)[1])
                q10 = source_qdlin(mat, index, 10)
                q100 = source_qdlin(mat, index, 100)
                if q10 is None or q100 is None:
                    continue
                features[row['cell_id']] = deltaq_variance(q10, q100)
        groups[group] = {'available_cells': sum(row['cell_id'] in features
                                                for row in subset),
                         'total_cells': len(subset)}
    return features, groups


def main():
    records = [row for row in load_manifest()['cells'] if row['dataset'] == 'MIT']
    record_by_id = {row['cell_id']: row for row in records}
    features, availability = load_features(records)
    prior = json.loads(PRIOR.read_text(encoding='utf-8'))
    previous = {row['cell_id']: row for fold in prior['folds'] for row in
                fold['checkpoints']['100']['by_cell']}
    folds = []
    for group, train, validation in outer_splits(records, 'MIT'):
        training = [key for key in train if key in features and
                    record_by_id[key]['endpoint_cycle'] is not None]
        model = make_pipeline(StandardScaler(), Ridge(alpha=ALPHA))
        model.fit(np.array([[features[key]] for key in training]),
                  np.log([record_by_id[key]['endpoint_cycle'] for key in training]))
        rows = []
        for key in validation:
            if key not in previous or key not in features:
                continue
            true = record_by_id[key]['endpoint_cycle']
            endpoint = float(np.clip(np.exp(model.predict([[features[key]]])[0]),
                                     101, 2500))
            old = previous[key]
            rows.append({'cell_id': key, 'deltaq_log_variance': features[key],
                         'true_endpoint_cycle': true,
                         'predicted_endpoint_cycle': endpoint,
                         'ape_percent': {
                             'baseline': old['ape_percent']['baseline'],
                             'prefix_log': old['ape_percent']['log'],
                             'deltaq_variance': float(100*abs(endpoint-true)/true)}})
        folds.append({'held_out': group, 'training_event_cells': len(training),
                      'scored_cells': len(rows),
                      'mape_percent': {name: float(np.mean([row['ape_percent'][name]
                                                          for row in rows]))
                                       for name in ('baseline', 'prefix_log',
                                                    'deltaq_variance')},
                      'by_cell': rows})
        print(group, folds[-1]['mape_percent'], flush=True)
    rows = [row for fold in folds for row in fold['by_cell']]
    summary = {'scored_cells': len(rows), 'feature_available_cells': len(features),
               'mape_percent': {name: float(np.mean([row['ape_percent'][name]
                                                   for row in rows]))
                                for name in ('baseline', 'prefix_log',
                                             'deltaq_variance')}}
    report = {'status': 'unpromoted whole-batch MIT development ablation',
              'source_feature': 'log10 variance(Qdlin(cycle100)-Qdlin(cycle10))',
              'input_warning': 'complete discharge at cycles 10 and 100; not 300 s online',
              'endpoint_warning': 'MIT source 0.88 Ah, not per-cell exact 80%',
              'reproduction_warning': 'conceptual single-feature replication; original model code is licensed',
              'availability': availability, 'summary': summary, 'folds': folds}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'deltaq_variance_cv.json').write_text(json.dumps(report, indent=2),
                                                encoding='utf-8')
    print('SUMMARY', summary)


if __name__ == '__main__':
    main()
