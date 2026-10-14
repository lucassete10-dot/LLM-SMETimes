"""MIT full-discharge ΔQ(V) small-statistic extension, whole-batch CV.

Tests the paper-motivated minimum alongside log variance, and optionally
mean; all three are computed only from observed cycles 10 and 100.
"""

import json
from pathlib import Path

import h5py
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from battery_v2_features import load_manifest
from evaluate_mit_deltaq_v2 import (ALPHA, OUT, PRIOR, deltaq_variance,
                                    source_qdlin)
from train_soh_v2 import outer_splits


BASE = OUT / 'deltaq_variance_cv.json'
TARGET = OUT / 'deltaq_stats_cv.json'


def deltaq_statistics(q10, q100):
    change = np.asarray(q100, dtype=float)-np.asarray(q10, dtype=float)
    return np.array([deltaq_variance(q10, q100), np.min(change),
                     np.mean(change)], dtype=float)


def load_statistics(records):
    result = {}
    for group in sorted({row['group'] for row in records}):
        subset = [row for row in records if row['group'] == group]
        with h5py.File(subset[0]['measurement_source'], 'r') as mat:
            for row in subset:
                index = int(row['cell_id'].split('c', 1)[1])
                q10 = source_qdlin(mat, index, 10)
                q100 = source_qdlin(mat, index, 100)
                if q10 is not None and q100 is not None:
                    result[row['cell_id']] = deltaq_statistics(q10, q100)
    return result


def main():
    records = [row for row in load_manifest()['cells'] if row['dataset'] == 'MIT']
    by_id = {row['cell_id']: row for row in records}
    features = load_statistics(records)
    base = json.loads(BASE.read_text(encoding='utf-8'))
    old = {row['cell_id']: row for fold in base['folds'] for row in fold['by_cell']}
    folds = []
    for group, train, validation in outer_splits(records, 'MIT'):
        training = [key for key in train if key in features and
                    by_id[key]['endpoint_cycle'] is not None]
        models = {}
        for name, width in (('variance_min', 2), ('variance_min_mean', 3)):
            model = make_pipeline(StandardScaler(), Ridge(alpha=ALPHA))
            model.fit(np.stack([features[key][:width] for key in training]),
                      np.log([by_id[key]['endpoint_cycle'] for key in training]))
            models[name] = (model, width)
        rows = []
        for key in validation:
            if key not in old or key not in features:
                continue
            true = by_id[key]['endpoint_cycle']
            endpoint = {name: float(np.clip(np.exp(model.predict(
                        features[key][:width][None, :])[0]), 101, 2500))
                        for name, (model, width) in models.items()}
            rows.append({'cell_id': key, 'true_endpoint_cycle': true,
                         'predicted_endpoint_cycle': endpoint,
                         'ape_percent': {
                             'variance_only': old[key]['ape_percent']['deltaq_variance'],
                             **{name: float(100*abs(value-true)/true)
                                for name, value in endpoint.items()}}})
        folds.append({'held_out': group, 'training_event_cells': len(training),
                      'scored_cells': len(rows),
                      'mape_percent': {name: float(np.mean([r['ape_percent'][name]
                                                          for r in rows]))
                                       for name in ('variance_only', 'variance_min',
                                                    'variance_min_mean')},
                      'by_cell': rows})
        print(group, folds[-1]['mape_percent'], flush=True)
    rows = [row for fold in folds for row in fold['by_cell']]
    summary = {'scored_cells': len(rows),
               'mape_percent': {name: float(np.mean([row['ape_percent'][name]
                                                   for row in rows]))
                                for name in ('variance_only', 'variance_min',
                                             'variance_min_mean')}}
    report = {'status': 'unpromoted MIT whole-batch development statistic ablation',
              'input_warning': 'full discharge at cycles 10 and 100 required',
              'feature_order': ['log10 variance of ΔQ(V)', 'minimum of ΔQ(V)',
                                'mean of ΔQ(V)'],
              'endpoint_warning': 'source 0.88 Ah, not per-cell exact 80%',
              'summary': summary, 'folds': folds}
    TARGET.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('SUMMARY', summary)


if __name__ == '__main__':
    main()
