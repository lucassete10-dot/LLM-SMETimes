"""Train-batch-only physical threshold transfer from MIT source EOL to exact80.

The end-of-life rate prior is estimated using only training batches' measured
capacity near their source 0.88 Ah endpoints. A held-out cell contributes
only its initial reference capacity and existing cycle-100 model prediction.
This cannot validate Batch 1/3 exact80 because their events are censored.
"""

import json
from pathlib import Path

import numpy as np

from audit_mit_exact80_rul_v2 import ORIGIN_CYCLE, score
from battery_v2_features import load_manifest


BASE = Path('battery_models_v2/mit_qv_challenger/exact80_target_audit.json')
OUT = Path('battery_models_v2/mit_qv_challenger/exact80_threshold_transfer.json')
SOURCE_THRESHOLD_AH = .88


def terminal_rate(record, window=60):
    endpoint = record['endpoint_cycle']
    if endpoint is None:
        return None
    with np.load(record['labels_file']) as labels:
        cycle = labels['cycle'].astype(float)
        capacity = labels['capacity_ah'].astype(float)
    keep = (cycle <= endpoint) & (cycle >= endpoint-window)
    if keep.sum() < 10:
        return None
    # Positive decline rate; robust to isolated capacity regeneration spikes.
    x, y = cycle[keep], capacity[keep]
    midpoint = np.median(x)
    earlier, later = y[x <= midpoint], y[x > midpoint]
    if not len(earlier) or not len(later):
        return None
    rate = (np.median(earlier)-np.median(later)) / (
        np.median(x[x > midpoint])-np.median(x[x <= midpoint]))
    return float(rate) if np.isfinite(rate) and rate > 0 else None


def transferred_endpoint(predicted_source_cycle, reference_capacity_ah, rate):
    exact80_ah = .8*reference_capacity_ah
    shift = (SOURCE_THRESHOLD_AH-exact80_ah)/rate
    return float(max(ORIGIN_CYCLE + 1, predicted_source_cycle + shift))


def main():
    records = [row for row in load_manifest()['cells'] if row['dataset'] == 'MIT']
    base = json.loads(BASE.read_text(encoding='utf-8'))
    rows = []
    training_rates = {}
    for group in sorted({row['group'] for row in records}):
        rates = [terminal_rate(row) for row in records if row['group'] != group]
        rates = [rate for rate in rates if rate is not None]
        if not rates:
            raise ValueError(f'{group}: no training terminal slopes')
        training_rates[group] = {'training_cells_with_rate': len(rates),
                                 'median_decline_ah_per_cycle': float(np.median(rates))}
    for row in base['by_cell']:
        group = row['held_out_batch']
        record = next(item for item in records if item['cell_id'] == row['cell_id'])
        predicted = transferred_endpoint(
            row['source_trained_predicted_endpoint_cycle'],
            record['reference_capacity_ah'],
            training_rates[group]['median_decline_ah_per_cycle'])
        actual = row['exact80_event_cycle']
        rows.append({
            'cell_id': row['cell_id'], 'held_out_batch': group,
            'predicted_exact80_cycle': predicted,
            'true_exact80_cycle': actual,
            'baseline_remaining_rul_ape_percent': row['exact80_remaining_rul_ape_percent'],
            'transfer_remaining_rul_ape_percent': score(actual, predicted),
        })
    report = {
        'status': 'unpromoted, one-event-batch exact80 threshold-transfer diagnostic',
        'training_rule': 'median positive terminal decline among other batches; 60-cycle window',
        'model_input_warning': 'source endpoint model requires complete discharges through cycle 100',
        'scored_cells': len(rows),
        'scored_batches': sorted({row['held_out_batch'] for row in rows}),
        'training_rates': training_rates,
        'baseline_remaining_rul_mape_percent': float(np.mean([
            row['baseline_remaining_rul_ape_percent'] for row in rows])),
        'transfer_remaining_rul_mape_percent': float(np.mean([
            row['transfer_remaining_rul_ape_percent'] for row in rows])),
        'by_cell': rows,
    }
    OUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('RUL MAPE baseline', report['baseline_remaining_rul_mape_percent'],
          'threshold transfer', report['transfer_remaining_rul_mape_percent'],
          'scored', len(rows), 'batches', report['scored_batches'])
    print('WROTE', OUT)


if __name__ == '__main__':
    main()
