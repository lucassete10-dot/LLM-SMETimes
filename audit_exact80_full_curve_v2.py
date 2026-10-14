"""Evaluate selected coherent OOF curves through confirmed own-80% SOH.

Prediction uses source-endpoint-trained held-out SOH/RUL models and only
features available at the first prediction origin. Future SOH labels are
read solely for scoring. Cells without a confirmed exact80 event remain
right-censored and cannot contribute an exact80 full-curve MAPE.
"""

import json
from pathlib import Path

import numpy as np

from audit_selected_joint_curve_all_v2 import RUL, SOH, read_selected
from battery_v2_features import load_all
from evaluate_anchored_trajectory_v2 import get_domain_arrays
from train_trajectory_v2 import threshold_soh


TARGETS = Path('battery_models_v2/exact80_target_audit.json')
OUT = Path('battery_models_v2/exact80_full_curve_audit.json')
TARGET_SOH = .8


def source_anchored_linear_curve(current, source_threshold, origin,
                                 predicted_source_rul, future_cycles):
    if predicted_source_rul <= 0:
        raise ValueError('predicted source RUL must be positive')
    drop = max(0., current-source_threshold)
    horizons = np.asarray(future_cycles, dtype=float)-origin
    return np.clip(current-drop*horizons/predicted_source_rul, 0, current)


def inferred_exact80_cycle(current, source_threshold, origin,
                            predicted_source_rul):
    if current <= TARGET_SOH:
        return float(origin)
    drop = max(0., current-source_threshold)
    if drop <= 0 or predicted_source_rul <= 0:
        return None
    return float(origin+predicted_source_rul*(current-TARGET_SOH)/drop)


def score_domain(domain, cells, targets):
    soh = read_selected(*SOH[domain], cells)
    _, baseline = get_domain_arrays(domain)
    if RUL[domain]:
        rul = read_selected(*RUL[domain], cells)
    else:
        rul = {key: (cycle, estimate) for key, (cycle, estimate, _)
               in baseline.items()}
    target_by_id = {row['cell_id']: row for row in targets['by_cell']}
    rows = []
    for key, (cycles, predictions) in rul.items():
        event = target_by_id[key]['first_confirmed_exact80_cycle']
        if event is None or not len(cycles):
            continue
        origin = int(cycles[0])
        if event <= origin:
            continue
        sc, values = soh[key]
        at_origin = int(np.searchsorted(sc, origin))
        if at_origin >= len(sc) or sc[at_origin] != origin:
            raise ValueError(f'{key}: OOF SOH/RUL origin mismatch')
        cell = cells[key]
        future = np.flatnonzero((cell.cycle > origin) &
                                (cell.cycle <= event) & cell.soh_available)
        if len(future) < 2:
            continue
        current = float(values[at_origin])
        predicted_rul = max(1., float(predictions[0]))
        forecast = source_anchored_linear_curve(
            current, threshold_soh(cell), origin, predicted_rul,
            cell.cycle[future])
        true = cell.soh[future]
        projected = inferred_exact80_cycle(
            current, threshold_soh(cell), origin, predicted_rul)
        rows.append({
            'cell_id': key, 'group': cell.group, 'origin_cycle': origin,
            'confirmed_exact80_cycle': event, 'future_points': len(future),
            'curve_mape_percent': float(100*np.mean(np.abs(forecast-true)/true)),
            'predicted_exact80_cycle_from_source_curve': projected,
            'exact80_endpoint_ape_percent': float(100*abs(projected-event)/event)
            if projected is not None else None,
            'exact80_remaining_rul_ape_percent': float(
                100*abs(projected-event)/(event-origin))
            if projected is not None else None,
        })
    groups = sorted({row['group'] for row in rows})
    return {
        'total_cells': len(cells),
        'confirmed_exact80_event_cells': targets['confirmed_event_cells'],
        'scored_cells': len(rows),
        'scored_groups': groups,
        'curve_cell_macro_mape_percent': float(np.mean([
            row['curve_mape_percent'] for row in rows])) if rows else None,
        'endpoint_mape_percent': float(np.mean([
            row['exact80_endpoint_ape_percent'] for row in rows
            if row['exact80_endpoint_ape_percent'] is not None])) if rows else None,
        'remaining_rul_mape_percent': float(np.mean([
            row['exact80_remaining_rul_ape_percent'] for row in rows
            if row['exact80_remaining_rul_ape_percent'] is not None])) if rows else None,
        'by_cell': rows,
    }


def main():
    targets = json.loads(TARGETS.read_text(encoding='utf-8'))['domains']
    _, all_cells = load_all()
    report = {
        'status': 'development held-out exact80 full-curve audit; not independent test',
        'input_warning': 'selected source-endpoint-trained SOH/RUL models, not retrained for exact80',
        'target': 'first confirmed per-cell 80% SOH crossing',
        'domains': {},
    }
    for domain in SOH:
        cells = {key: cell for key, cell in all_cells.items()
                 if cell.dataset == domain}
        summary = score_domain(domain, cells, targets[domain])
        report['domains'][domain] = summary
        print(domain, 'scored', summary['scored_cells'], 'groups',
              summary['scored_groups'], 'curve MAPE',
              summary['curve_cell_macro_mape_percent'], 'remaining RUL MAPE',
              summary['remaining_rul_mape_percent'], flush=True)
    OUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('WROTE', OUT)


if __name__ == '__main__':
    main()
