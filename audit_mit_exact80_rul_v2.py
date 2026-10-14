"""Score existing MIT whole-batch source-endpoint predictions at exact 80%.

No model is fitted here. Only cells with an observed, confirmed per-cell 80%
event can be scored. The source-endpoint model used complete cycle-10/100
discharges and was selected on development folds, so this is diagnostic only.
"""

import json
from pathlib import Path

import numpy as np


TARGET = Path('battery_models_v2/exact80_target_audit.json')
PREDICTIONS = Path('battery_models_v2/mit_qv_challenger/paper_discharge_cv.json')
OUT = Path('battery_models_v2/mit_qv_challenger/exact80_target_audit.json')
ORIGIN_CYCLE = 100


def score(event_cycle, predicted_endpoint_cycle, origin=ORIGIN_CYCLE):
    actual = event_cycle - origin
    if actual <= 0:
        return None
    predicted = max(0., predicted_endpoint_cycle - origin)
    return float(100 * abs(predicted - actual) / actual)


def main():
    targets = json.loads(TARGET.read_text(encoding='utf-8'))['domains']['MIT']
    by_id = {row['cell_id']: row for row in targets['by_cell']}
    predictions = json.loads(PREDICTIONS.read_text(encoding='utf-8'))
    rows = []
    for fold in predictions['folds']:
        for row in fold['by_cell']:
            actual = by_id[row['cell_id']]['first_confirmed_exact80_cycle']
            if actual is None or actual <= ORIGIN_CYCLE:
                continue
            predicted = row['predicted_endpoint_cycle']['plus_capacity_all']
            rows.append({
                'cell_id': row['cell_id'], 'held_out_batch': fold['held_out'],
                'origin_cycle': ORIGIN_CYCLE,
                'exact80_event_cycle': actual,
                'source_endpoint_cycle': row['true_endpoint_cycle'],
                'source_trained_predicted_endpoint_cycle': predicted,
                'source_endpoint_ape_percent': row['ape_percent']['plus_capacity_all'],
                'exact80_endpoint_ape_percent': float(100*abs(predicted-actual)/actual),
                'exact80_remaining_rul_ape_percent': score(actual, predicted),
                'source_vs_exact80_cycle_gap': row['true_endpoint_cycle']-actual,
            })
    report = {
        'status': 'diagnostic exact80 evaluation of source-endpoint-trained MIT model',
        'input_warning': 'complete discharges through cycle 100, not 300-second online signals',
        'validation_warning': 'exact80 events only in held-out Batch 2; prior model selection on development folds',
        'target_warning': 'model trained for source 0.88 Ah endpoint, not exact80 event',
        'prediction_interpretation': 'source endpoint prediction reused unchanged as exact80 point-prediction proxy; not curve-implied crossing',
        'origin_cycle': ORIGIN_CYCLE,
        'mit_total_cells': targets['cells'],
        'mit_confirmed_exact80_cells': targets['confirmed_event_cells'],
        'scored_cells': len(rows),
        'scored_batches': sorted({r['held_out_batch'] for r in rows}),
        'source_endpoint_mape_percent_on_scored_subset': float(np.mean([
            r['source_endpoint_ape_percent'] for r in rows])) if rows else None,
        'exact80_endpoint_mape_percent': float(np.mean([
            r['exact80_endpoint_ape_percent'] for r in rows])) if rows else None,
        'exact80_remaining_rul_mape_percent': float(np.mean([
            r['exact80_remaining_rul_ape_percent'] for r in rows])) if rows else None,
        'median_abs_source_exact80_gap_cycles': float(np.median(np.abs([
            r['source_vs_exact80_cycle_gap'] for r in rows]))) if rows else None,
        'by_cell': rows,
    }
    OUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('scored', len(rows), 'batches', report['scored_batches'],
          'source endpoint MAPE', report['source_endpoint_mape_percent_on_scored_subset'],
          'exact80 endpoint MAPE', report['exact80_endpoint_mape_percent'],
          'exact80 remaining RUL MAPE', report['exact80_remaining_rul_mape_percent'])
    print('WROTE', OUT)


if __name__ == '__main__':
    main()
