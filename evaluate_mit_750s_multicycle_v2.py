"""MIT paper-analog features from observed 750 s prefixes at cycles 10/50/100.

The five features parallel variance/min ΔQ(V), early low-voltage capacity,
capacity trend, and gain. All Q(V) values are integrated only through 750 s
of a discharge, never taken from a complete-cycle capacity label.
"""

import json
from pathlib import Path

import h5py
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from audit_mit_exact80_rul_v2 import score
from audit_mit_partial_qv_v2 import partial_trace
from battery_v2_features import load_manifest
from evaluate_mit_partial_deltaq_v2 import capacity_on_voltage_grid
from train_soh_v2 import outer_splits


ROOT = Path('battery_models_v2/mit_qv_challenger')
BASE = ROOT / 'prefix_750s_qv_cv.json'
TARGETS = Path('battery_models_v2/exact80_target_audit.json')
OUT = ROOT / 'prefix_750s_multicycle_cv.json'
GRID = np.linspace(3.0, 3.53, 107)
CYCLES = (10, 50, 100)
ORIGIN = 100
ALPHA = 1.0


def feature_vector(q10, q50, q100):
    q10, q50, q100 = map(lambda x: np.asarray(x, dtype=float),
                          (q10, q50, q100))
    if q10.shape != GRID.shape or q50.shape != GRID.shape or q100.shape != GRID.shape:
        raise ValueError('unaligned partial Q(V) vectors')
    delta = q100-q10
    variance = float(np.var(delta))
    if variance <= 0 or not np.isfinite(variance):
        raise ValueError('nonpositive partial ΔQ variance')
    low10, low50, low100 = float(q10[0]), float(q50[0]), float(q100[0])
    return np.array([np.log10(variance), np.min(delta), low10,
                     (low100-low10)/(CYCLES[-1]-CYCLES[0]),
                     max(low50, low100)-low10], dtype=float)


def extract_features(records):
    result = {}
    for group in sorted({row['group'] for row in records}):
        subset = [row for row in records if row['group'] == group]
        with h5py.File(subset[0]['measurement_source'], 'r') as mat:
            for row in subset:
                index = int(row['cell_id'].split('c', 1)[1])
                traces = [partial_trace(mat, index, cycle, 750)
                          for cycle in CYCLES]
                if any(trace is None for trace in traces):
                    continue
                q = [capacity_on_voltage_grid(*trace, voltage_grid=GRID)
                     for trace in traces]
                result[row['cell_id']] = feature_vector(*q)
    return result


def main():
    records = [row for row in load_manifest()['cells'] if row['dataset'] == 'MIT']
    by_id = {row['cell_id']: row for row in records}
    features = extract_features(records)
    base = json.loads(BASE.read_text(encoding='utf-8'))
    baseline = {row['cell_id']: row for fold in base['folds']
                for row in fold['by_cell']}
    targets = json.loads(TARGETS.read_text(encoding='utf-8'))['domains']['MIT']
    events = {row['cell_id']: row['first_confirmed_exact80_cycle']
              for row in targets['by_cell']}
    folds = []
    for group, train, validation in outer_splits(records, 'MIT'):
        training = [key for key in train if key in features and
                    by_id[key]['endpoint_cycle'] is not None]
        model = make_pipeline(StandardScaler(), Ridge(alpha=ALPHA))
        model.fit(np.stack([features[key] for key in training]),
                  np.log([by_id[key]['endpoint_cycle'] for key in training]))
        rows = []
        for key in validation:
            if key not in features or key not in baseline:
                continue
            source = int(by_id[key]['endpoint_cycle'])
            predicted = float(np.clip(np.exp(model.predict(
                features[key][None, :])[0]), ORIGIN+1, 2500))
            baseline_predicted = baseline[key]['predicted_source_endpoint_cycle']
            event = events[key]
            rows.append({
                'cell_id': key, 'held_out_batch': group,
                'true_source_endpoint_cycle': source,
                'predicted_source_endpoint_cycle': predicted,
                'source_endpoint_ape_percent': float(100*abs(predicted-source)/source),
                'matched_two_stat_source_ape_percent': float(
                    100*abs(baseline_predicted-source)/source),
                'confirmed_exact80_cycle': event,
                'exact80_remaining_rul_ape_percent': score(event, predicted, ORIGIN)
                if event is not None and event > ORIGIN else None,
                'matched_two_stat_exact80_rul_ape_percent': score(
                    event, baseline_predicted, ORIGIN)
                if event is not None and event > ORIGIN else None,
            })
        folds.append({
            'held_out_batch': group, 'training_event_cells': len(training),
            'scored_cells': len(rows),
            'source_endpoint_mape_percent': float(np.mean([
                row['source_endpoint_ape_percent'] for row in rows])),
            'matched_two_stat_source_mape_percent': float(np.mean([
                row['matched_two_stat_source_ape_percent'] for row in rows])),
            'by_cell': rows,
        })
        print(group, 'two-stat', folds[-1]['matched_two_stat_source_mape_percent'],
              'five-stat', folds[-1]['source_endpoint_mape_percent'], flush=True)
    rows = [row for fold in folds for row in fold['by_cell']]
    exact_rows = [row for row in rows if row['exact80_remaining_rul_ape_percent']
                  is not None]
    report = {
        'status': 'unpromoted MIT whole-batch partial-discharge paper-analog ablation',
        'input': 'observed first 750 s of discharges 10, 50, 100; no full-discharge capacity',
        'exact80_metric_warning': 'source endpoint is reused unchanged as an exact80 point-prediction proxy; not a curve-implied exact80 crossing',
        'feature_order': ['log variance ΔQ100-10(V)', 'min ΔQ100-10(V)',
                          'cycle-10 Q(3.0 V)', 'cycle-10-to-100 Q(3.0 V) slope',
                          'max cycle-50/100 Q(3.0 V) gain over cycle 10'],
        'grid_v': GRID.tolist(), 'feature_available_cells': len(features),
        'source_endpoint_scored_cells': len(rows),
        'source_endpoint_mape_percent': float(np.mean([
            row['source_endpoint_ape_percent'] for row in rows])),
        'matched_two_stat_source_mape_percent': float(np.mean([
            row['matched_two_stat_source_ape_percent'] for row in rows])),
        'exact80_scored_cells': len(exact_rows),
        'exact80_scored_batches': sorted({row['held_out_batch'] for row in exact_rows}),
        'exact80_remaining_rul_mape_percent': float(np.mean([
            row['exact80_remaining_rul_ape_percent'] for row in exact_rows]))
        if exact_rows else None,
        'matched_two_stat_exact80_rul_mape_percent': float(np.mean([
            row['matched_two_stat_exact80_rul_ape_percent'] for row in exact_rows]))
        if exact_rows else None,
        'folds': folds,
    }
    OUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('SUMMARY', {key: value for key, value in report.items()
                      if key.endswith('mape_percent') or key.endswith('cells')})
    print('WROTE', OUT)


if __name__ == '__main__':
    main()
