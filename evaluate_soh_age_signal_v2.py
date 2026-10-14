"""Whole-batch SOH ablation: causal cycle age added to time-gated signal.

No measured capacity, SOH, or future endpoint is an inference feature. The
existing signal model and fixed raw/gated fusion are compared on identical
held-out cells. This is development-only feature exploration.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from battery_v2_features import load_cell, load_manifest
from evaluate_paper_inspired_soh_v2 import fit_gated, summary
from evaluate_relative_soh_features_v2 import fit_view, predict_view
from evaluate_soh_anchor_v2 import anchored_prediction
from paper_inspired_battery_v2 import time_gated_matrix
from train_soh_v2 import outer_splits


OUT = Path('battery_models_v2/soh_age_signal_challenger')
BASE = Path('battery_models_v2/paper_inspired_soh_challenger')


def age_matrix(cell):
    age = cell.cycle.astype(np.float32)
    return np.column_stack((time_gated_matrix(cell), age, np.log1p(age)))


def evaluate(domain):
    records = [row for row in load_manifest()['cells'] if row['dataset'] == domain]
    cells = {row['cell_id']: load_cell(row) for row in records}
    matrices = {key: age_matrix(cell) for key, cell in cells.items()}
    source_path = BASE / f'{domain.lower()}_development_oof_predictions.npz'
    if source_path.exists():
        with np.load(source_path) as source:
            original = {(str(key), int(cycle)): (float(gated), float(raw))
                        for key, cycle, gated, raw in zip(source['cell_id'],
                            source['cycle'], source['gated'], source['raw'])}
    else:
        original = None
        ungated_matrices = {key: time_gated_matrix(cell)
                            for key, cell in cells.items()}
    folds = []
    heldout_age = {}
    for group, training, validation in outer_splits(records, domain):
        bundle = fit_gated(cells, training, matrices)
        if original is None:
            base_bundle = fit_gated(cells, training, ungated_matrices)
            raw_bundle = fit_view(cells, training, 'raw_only')
        candidate = {}
        baseline = {}
        for key in validation:
            age_gated = bundle[1].predict(bundle[0].transform(matrices[key]))
            if original is None:
                original_gated = base_bundle[1].predict(base_bundle[0].transform(
                    ungated_matrices[key]))
                original_raw = predict_view(raw_bundle, cells[key], 'raw_only')
            else:
                original_gated = np.array([original[key, int(cycle)][0]
                                           for cycle in cells[key].cycle])
                original_raw = np.array([original[key, int(cycle)][1]
                                         for cycle in cells[key].cycle])
            candidate[key] = age_gated
            heldout_age[key] = age_gated
            baseline[key] = (original_gated, original_raw)
        predictions = {
            'gated_original': {key: value[0] for key, value in baseline.items()},
            'gated_with_age': candidate,
            'fusion_original': {key: .5*(value[0]+value[1])
                                for key, value in baseline.items()},
            'fusion_with_age': {key: .5*(candidate[key]+baseline[key][1])
                                for key in validation},
            'gated_age_anchor_half': {key: anchored_prediction(
                candidate[key], cells[key].reference_count, .5)
                for key in validation},
            'gated_age_anchor_full': {key: anchored_prediction(
                candidate[key], cells[key].reference_count, 1.)
                for key in validation},
        }
        scored = {name: summary(cells, validation, estimate)
                  for name, estimate in predictions.items()}
        folds.append({'held_out': group, 'training_cells': len(training),
                      'scored_cells': len(validation), 'metrics': scored})
        print(domain, group, {name: round(value['cell_macro_mape_percent'], 3)
                              for name, value in scored.items()}, flush=True)
    scored = {name: {'cell_macro_mape_percent': float(np.mean([
        cell['mape_percent'] for fold in folds
        for cell in fold['metrics'][name]['by_cell']]))}
        for name in ('gated_original', 'gated_with_age',
                     'fusion_original', 'fusion_with_age',
                     'gated_age_anchor_half', 'gated_age_anchor_full')}
    report = {'status': 'unpromoted whole-batch causal-age SOH development ablation',
              'domain': domain, 'feature': 'observed cycle number and log1p(cycle)',
              'summary': scored, 'folds': folds}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'{domain.lower()}_development_cv.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    ids = [row['cell_id'] for row in records]
    np.savez_compressed(
        OUT / f'{domain.lower()}_development_oof_predictions.npz',
        cell_id=np.concatenate([np.full(len(cells[key].cycle), key) for key in ids]),
        cycle=np.concatenate([cells[key].cycle for key in ids]),
        target=np.concatenate([cells[key].soh for key in ids]),
        soh_available=np.concatenate([cells[key].soh_available for key in ids]),
        gated_with_age=np.concatenate([heldout_age[key] for key in ids]),
        gated_age_anchor_half=np.concatenate([anchored_prediction(
            heldout_age[key], cells[key].reference_count, .5) for key in ids]),
        gated_age_anchor_full=np.concatenate([anchored_prediction(
            heldout_age[key], cells[key].reference_count, 1.) for key in ids]),
    )
    print('SUMMARY', domain, scored, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--domains', nargs='+', choices=('NASA', 'CALCE', 'MIT', 'XJTU'),
                        default=['MIT', 'XJTU'])
    for selected_domain in parser.parse_args().domains:
        evaluate(selected_domain)
