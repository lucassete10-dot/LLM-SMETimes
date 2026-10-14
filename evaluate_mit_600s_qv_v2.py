"""MIT 600-second observed-prefix ΔQ(V) versus the existing 300-second route.

This is a longer but still partial discharge diagnostic at cycles 10 and 100.
Only outer-training batches' observed source EOL labels fit the ridge model;
exact80 events in the held-out group are read only for scoring.
"""

import json
import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from audit_mit_exact80_rul_v2 import score
from battery_v2_features import load_manifest
from evaluate_mit_partial_deltaq_v2 import ALPHA, load_statistics
from train_soh_v2 import outer_splits


ROOT = Path('battery_models_v2/mit_qv_challenger')
PRIOR = ROOT / 'partial_deltaq_cv.json'
TARGETS = Path('battery_models_v2/exact80_target_audit.json')
ORIGIN = 100
GRID_LOWER_V = {600: 3.08, 750: 3.00}


def main(duration_s=600):
    lower = GRID_LOWER_V[duration_s]
    voltage_grid = np.linspace(lower, 3.53,
                               int(round((3.53-lower)/.005))+1)
    records = [row for row in load_manifest()['cells'] if row['dataset'] == 'MIT']
    by_id = {row['cell_id']: row for row in records}
    features = load_statistics(records, duration_s=duration_s,
                               voltage_grid=voltage_grid)
    prior = json.loads(PRIOR.read_text(encoding='utf-8'))
    old = {row['cell_id']: row for fold in prior['folds']
           for row in fold['by_cell']}
    target_report = json.loads(TARGETS.read_text(encoding='utf-8'))['domains']['MIT']
    exact80 = {row['cell_id']: row['first_confirmed_exact80_cycle']
               for row in target_report['by_cell']}
    folds = []
    for group, train, validation in outer_splits(records, 'MIT'):
        training = [key for key in train if key in features and
                    by_id[key]['endpoint_cycle'] is not None]
        model = make_pipeline(StandardScaler(), Ridge(alpha=ALPHA))
        model.fit(np.stack([features[key][:2] for key in training]),
                  np.log([by_id[key]['endpoint_cycle'] for key in training]))
        rows = []
        for key in validation:
            if (key not in features or key not in old or
                    by_id[key]['endpoint_cycle'] is None):
                continue
            predicted = float(np.clip(np.exp(model.predict(
                features[key][:2][None, :])[0]), ORIGIN+1, 2500))
            source = int(by_id[key]['endpoint_cycle'])
            old_predicted = old[key]['predicted_endpoint_cycle']['partial_variance_min']
            event = exact80[key]
            rows.append({
                'cell_id': key, 'held_out_batch': group,
                'source_endpoint_cycle': source,
                'predicted_source_endpoint_cycle': predicted,
                'source_endpoint_ape_percent': float(100*abs(predicted-source)/source),
                'matched_300s_source_endpoint_ape_percent': float(
                    100*abs(old_predicted-source)/source),
                'confirmed_exact80_cycle': event,
                'exact80_remaining_rul_ape_percent': score(event, predicted, ORIGIN)
                if event is not None and event > ORIGIN else None,
                'matched_300s_exact80_remaining_rul_ape_percent': score(
                    event, old_predicted, ORIGIN)
                if event is not None and event > ORIGIN else None,
            })
        folds.append({
            'held_out_batch': group, 'training_event_cells': len(training),
            'scored_source_endpoint_cells': len(rows),
            'source_endpoint_mape_percent': float(np.mean([
                row['source_endpoint_ape_percent'] for row in rows])) if rows else None,
            'matched_300s_source_endpoint_mape_percent': float(np.mean([
                row['matched_300s_source_endpoint_ape_percent'] for row in rows]))
            if rows else None,
            'by_cell': rows,
        })
        print(group, duration_s, 's training', len(training), 'scored', len(rows),
              '300 s source EOL MAPE', folds[-1]['matched_300s_source_endpoint_mape_percent'],
              f'{duration_s} s', folds[-1]['source_endpoint_mape_percent'], flush=True)
    rows = [row for fold in folds for row in fold['by_cell']]
    event_rows = [row for row in rows if row['exact80_remaining_rul_ape_percent']
                  is not None]
    report = {
        'status': 'unpromoted MIT whole-batch observed-prefix development ablation',
        'input': f'first {duration_s} s of cycles 10 and 100 discharges; within-prefix Qd(V)',
        'duration_s': duration_s,
        'feature': 'log variance and minimum of partial ΔQ100-10(V)',
        'voltage_grid_v': voltage_grid.tolist(),
        'grid_selection_warning': 'lower bound selected from label-free coverage audit of all development cells',
        'label': 'source 0.88 Ah EOL for training; own80 event for separate evaluation only',
        'exact80_metric_warning': 'source endpoint is reused unchanged as an exact80 point-prediction proxy; not a curve-implied exact80 crossing',
        'feature_available_cells': len(features),
        'source_scored_cells': len(rows),
        'source_endpoint_mape_percent': float(np.mean([
            row['source_endpoint_ape_percent'] for row in rows])),
        'matched_300s_source_endpoint_mape_percent': float(np.mean([
            row['matched_300s_source_endpoint_ape_percent'] for row in rows])),
        'exact80_scored_cells': len(event_rows),
        'exact80_scored_batches': sorted({row['held_out_batch'] for row in event_rows}),
        'exact80_remaining_rul_mape_percent': float(np.mean([
            row['exact80_remaining_rul_ape_percent'] for row in event_rows]))
        if event_rows else None,
        'matched_300s_exact80_remaining_rul_mape_percent': float(np.mean([
            row['matched_300s_exact80_remaining_rul_ape_percent']
            for row in event_rows])) if event_rows else None,
        'folds': folds,
    }
    out = ROOT / f'prefix_{duration_s}s_qv_cv.json'
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('SUMMARY', {key: value for key, value in report.items()
                      if key.endswith('mape_percent') or key.endswith('cells')})
    print('WROTE', out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--duration', type=int, choices=tuple(GRID_LOWER_V),
                        default=600)
    main(parser.parse_args().duration)
