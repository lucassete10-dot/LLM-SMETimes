"""MIT whole-batch ablation of paper-listed early discharge features.

Additional features (cycle-2 capacity, cycle-2-to-100 slope and early maximum
gain) require complete observed discharges through cycle 100. They are not
valid inputs for the 300 s prefix-only model. This is development CV only.
"""

import json
from pathlib import Path

import h5py
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from battery_v2_features import load_manifest
from evaluate_mit_deltaq_stats_v2 import deltaq_statistics
from evaluate_mit_deltaq_v2 import OUT, source_qdlin
from train_soh_v2 import outer_splits


BASE = OUT / 'deltaq_stats_cv.json'
TARGET = OUT / 'paper_discharge_cv.json'
ALPHA = 1.0


def capacity_features(summary):
    cycles = np.asarray(summary['cycle']).reshape(-1).astype(int)
    capacity = np.asarray(summary['QDischarge']).reshape(-1).astype(float)
    if len(cycles) != len(capacity):
        raise ValueError('summary cycle/capacity mismatch')
    at2 = np.flatnonzero(cycles == 2)
    if len(at2) != 1 or not (.2 < capacity[at2[0]] < 1.25):
        return None
    keep = (cycles >= 2) & (cycles <= 100) & np.isfinite(capacity) &\
           (capacity > .2) & (capacity < 1.25)
    if keep.sum() < 80:
        return None
    q2 = capacity[at2[0]]
    slope = np.polyfit(cycles[keep], capacity[keep], 1)[0]
    gain = np.max(capacity[keep])-q2
    return np.array([q2, slope, gain], dtype=float)


def load_features(records):
    result = {}
    for group in sorted({row['group'] for row in records}):
        subset = [row for row in records if row['group'] == group]
        with h5py.File(subset[0]['measurement_source'], 'r') as mat:
            struct = mat['batch']
            for row in subset:
                index = int(row['cell_id'].split('c', 1)[1])
                q10 = source_qdlin(mat, index, 10)
                q100 = source_qdlin(mat, index, 100)
                if q10 is None or q100 is None:
                    continue
                extra = capacity_features(mat[struct['summary'][index, 0]])
                if extra is None:
                    continue
                dq = deltaq_statistics(q10, q100)
                result[row['cell_id']] = np.r_[dq[:2], extra]
    return result


def main():
    records = [row for row in load_manifest()['cells'] if row['dataset'] == 'MIT']
    by_id = {row['cell_id']: row for row in records}
    features = load_features(records)
    old_report = json.loads(BASE.read_text(encoding='utf-8'))
    old = {row['cell_id']: row for fold in old_report['folds']
           for row in fold['by_cell']}
    folds = []
    definitions = (('variance_min', (0, 1)),
                   ('plus_capacity_slope', (0, 1, 3)),
                   ('plus_capacity_all', (0, 1, 2, 3, 4)))
    for group, train, validation in outer_splits(records, 'MIT'):
        training = [key for key in train if key in features and
                    by_id[key]['endpoint_cycle'] is not None]
        models = {}
        for name, columns in definitions[1:]:
            model = make_pipeline(StandardScaler(), Ridge(alpha=ALPHA))
            model.fit(np.stack([features[key][list(columns)] for key in training]),
                      np.log([by_id[key]['endpoint_cycle'] for key in training]))
            models[name] = (model, columns)
        rows = []
        for key in validation:
            if key not in features or key not in old:
                continue
            true = by_id[key]['endpoint_cycle']
            predicted = {name: float(np.clip(np.exp(model.predict(
                features[key][list(columns)][None, :])[0]), 101, 2500))
                for name, (model, columns) in models.items()}
            rows.append({'cell_id': key, 'true_endpoint_cycle': true,
                         'predicted_endpoint_cycle': predicted,
                         'ape_percent': {
                             'variance_min': old[key]['ape_percent']['variance_min'],
                             **{name: float(100*abs(value-true)/true)
                                for name, value in predicted.items()}}})
        folds.append({'held_out': group, 'training_event_cells': len(training),
                      'scored_cells': len(rows),
                      'mape_percent': {name: float(np.mean([r['ape_percent'][name]
                                                          for r in rows]))
                                       for name, _ in definitions},
                      'by_cell': rows})
        print(group, folds[-1]['mape_percent'], flush=True)
    rows = [row for fold in folds for row in fold['by_cell']]
    summary = {'feature_available_cells': len(features), 'scored_cells': len(rows),
               'mape_percent': {name: float(np.mean([row['ape_percent'][name]
                                                   for row in rows]))
                                for name, _ in definitions}}
    report = {'status': 'unpromoted MIT whole-batch paper-feature development ablation',
              'feature_order': ['log10 var ΔQ100-10(V)', 'min ΔQ100-10(V)',
                                'cycle-2 full-discharge capacity',
                                'cycle-2-to-100 capacity slope',
                                'max early capacity minus cycle-2 capacity'],
              'input_warning': 'complete observed discharge diagnostics, not online prefix',
              'endpoint_warning': 'source 0.88 Ah, not per-cell exact 80%',
              'summary': summary, 'folds': folds}
    TARGET.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('SUMMARY', summary)


if __name__ == '__main__':
    main()
