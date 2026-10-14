"""Matched-cohort MIT exact80 RUL comparison across fixed early checkpoints.

All predictions are preexisting whole-batch OOF source-endpoint models. This
script does not refit or tune them, and scores only observed exact80 events.
Separate input classes prevent full-discharge results being called online.
"""

import json
from pathlib import Path

import numpy as np

from audit_mit_exact80_rul_v2 import score


ROOT = Path('battery_models_v2')
TARGETS = ROOT / 'exact80_target_audit.json'
INVERSE = ROOT / 'mit_inverse_life_challenger/development_cv.json'
PARTIAL = ROOT / 'mit_qv_challenger/partial_deltaq_cv.json'
PAPER = ROOT / 'mit_qv_challenger/paper_discharge_cv.json'
OUT = ROOT / 'mit_qv_challenger/exact80_checkpoint_comparison.json'
CHECKPOINTS = (20, 50, 100)


def load_rows(path):
    return json.loads(path.read_text(encoding='utf-8'))['folds']


def route_summary(rows):
    return {'scored_cells': len(rows),
            'remaining_rul_mape_percent': float(np.mean([
                row['remaining_rul_ape_percent'] for row in rows]))
            if rows else None,
            'endpoint_mape_percent': float(np.mean([
                row['endpoint_ape_percent'] for row in rows]))
            if rows else None}


def main():
    target_report = json.loads(TARGETS.read_text(encoding='utf-8'))['domains']['MIT']
    events = {row['cell_id']: row['first_confirmed_exact80_cycle']
              for row in target_report['by_cell']}
    by_checkpoint = {}
    for checkpoint in CHECKPOINTS:
        routes = {}
        for fold in load_rows(INVERSE):
            for row in fold['checkpoints'][str(checkpoint)]['by_cell']:
                event = events[row['cell_id']]
                origin = row['observed_cycle']
                if event is None or event <= origin:
                    continue
                for name, predicted in row['predicted_endpoint_cycle'].items():
                    routes.setdefault(name, []).append({
                        'cell_id': row['cell_id'], 'held_out_batch': fold['held_out'],
                        'origin_cycle': origin,
                        'remaining_rul_ape_percent': score(event, predicted, origin),
                        'endpoint_ape_percent': float(100*abs(predicted-event)/event),
                    })
        if checkpoint == 100:
            for fold in load_rows(PARTIAL):
                for row in fold['by_cell']:
                    event = events[row['cell_id']]
                    if event is None or event <= checkpoint:
                        continue
                    for name, predicted in row['predicted_endpoint_cycle'].items():
                        routes.setdefault('partial_qv_'+name, []).append({
                            'cell_id': row['cell_id'], 'held_out_batch': fold['held_out'],
                            'origin_cycle': checkpoint,
                            'remaining_rul_ape_percent': score(event, predicted, checkpoint),
                            'endpoint_ape_percent': float(100*abs(predicted-event)/event),
                        })
            for fold in load_rows(PAPER):
                for row in fold['by_cell']:
                    event = events[row['cell_id']]
                    if event is None or event <= checkpoint:
                        continue
                    predicted = row['predicted_endpoint_cycle']['plus_capacity_all']
                    routes.setdefault('full_discharge_paper_features', []).append({
                        'cell_id': row['cell_id'], 'held_out_batch': fold['held_out'],
                        'origin_cycle': checkpoint,
                        'remaining_rul_ape_percent': score(event, predicted, checkpoint),
                        'endpoint_ape_percent': float(100*abs(predicted-event)/event),
                    })
        by_checkpoint[str(checkpoint)] = {
            'routes': {name: route_summary(values) for name, values in routes.items()},
            'by_route_cell': routes,
        }
        for name, values in routes.items():
            if len(values) != target_report['confirmed_event_cells']:
                raise ValueError(f'{checkpoint}/{name}: unmatched exact80 cohort')
        print(checkpoint, {name: round(route_summary(values)[
            'remaining_rul_mape_percent'], 2) for name, values in routes.items()},
            flush=True)
    report = {
        'status': 'unpromoted exact80 MIT matched-cohort checkpoint diagnostic',
        'cohort': '43 observed exact80 events; all Batch 2',
        'input_classes': {
            'baseline': 'current/past 300 s signal',
            'inverse': '300 s voltage signal at each checkpoint',
            'log': '300 s voltage signal at each checkpoint',
            'partial_qv_*': '300 s signals at cycles 10 and 100',
            'full_discharge_paper_features': 'complete observed discharges through cycle 100',
        },
        'model_target_warning': 'all compared models trained to source 0.88 Ah endpoint',
        'exact80_metric_warning': 'source endpoint point predictions are reused unchanged as exact80 proxies; coherent curve crossing can have a different error',
        'checkpoints': by_checkpoint,
    }
    OUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('WROTE', OUT)


if __name__ == '__main__':
    main()
