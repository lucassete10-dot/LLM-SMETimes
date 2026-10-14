"""MIT whole-batch current SOH from observed 750 s discharge signals.

At a cycle with a 750-second discharge prefix, use additional voltage,
temperature, and within-prefix charge-at-voltage features. If discharge ends
before 750 s, retain the strictly held-out 300-second age-model prediction.
No current full-cycle capacity or future-cycle measurement is an input.
"""

import json
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from battery_v2_features import load_cell, load_manifest
from evaluate_paper_inspired_soh_v2 import fit_gated
from evaluate_soh_age_signal_v2 import age_matrix
from paper_inspired_battery_v2 import time_gated_matrix
from train_soh_v2 import outer_splits


ROOT = Path('battery_models_v2/soh_long_prefix_challenger')
BASE = Path('battery_models_v2/soh_age_signal_challenger/mit_development_oof_predictions.npz')
OUT = ROOT / 'mit_750s_current_soh_cv.json'
OOF = ROOT / 'mit_750s_current_soh_oof.npz'


def long_matrix(cell, long_raw):
    long_raw = np.asarray(long_raw, dtype='float32')
    if long_raw.ndim != 2 or len(long_raw) != len(cell.cycle):
        raise ValueError('long-prefix matrix and cycles must align')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        anchor = np.nanmedian(long_raw[:cell.reference_count], axis=0)
    long_delta = long_raw-anchor
    view = SimpleNamespace(cycle=cell.cycle, raw=long_raw, delta=long_delta)
    return np.column_stack((age_matrix(cell), time_gated_matrix(view)))


def cell_score(target, estimate, keep):
    if not keep.any():
        return None
    return float(100*np.mean(np.abs(estimate[keep]-target[keep])/target[keep]))


def main():
    records = [row for row in load_manifest()['cells'] if row['dataset'] == 'MIT']
    cells = {row['cell_id']: load_cell(row) for row in records}
    matrices, available = {}, {}
    for row in records:
        key = row['cell_id']
        with np.load(ROOT / f'{key}.npz') as source:
            if not np.array_equal(source['cycle'], cells[key].cycle):
                raise ValueError(f'{key}: long-prefix cycle mismatch')
            raw = source['raw'].astype('float32')
        available[key] = np.isfinite(raw).any(axis=1)
        matrices[key] = long_matrix(cells[key], raw)
    with np.load(BASE) as archive:
        baseline = {(str(key), int(cycle)): float(value)
                    for key, cycle, value in zip(archive['cell_id'],
                                                  archive['cycle'],
                                                  archive['gated_with_age'])}
    folds = []
    heldout = {}
    for group, training, validation in outer_splits(records, 'MIT'):
        imputer, model = fit_gated(cells, training, matrices)
        rows = []
        for key in validation:
            cell = cells[key]
            predicted = model.predict(imputer.transform(matrices[key]))
            original = np.array([baseline[key, int(cycle)] for cycle in cell.cycle])
            avail = available[key]
            candidate = np.where(avail, predicted, original)
            fusion = np.where(avail, .5*(predicted+original), original)
            heldout[key] = {'candidate': candidate, 'fusion': fusion}
            keep = cell.soh_available
            rows.append({
                'cell_id': key,
                'scored_observations': int(keep.sum()),
                'long_prefix_available_fraction': float(np.mean(avail[keep])),
                'baseline_mape_percent': cell_score(cell.soh, original, keep),
                'candidate_mape_percent': cell_score(cell.soh, candidate, keep),
                'fixed_fusion_mape_percent': cell_score(cell.soh, fusion, keep),
                'baseline_available_mape_percent': cell_score(
                    cell.soh, original, keep & avail),
                'candidate_available_mape_percent': cell_score(
                    cell.soh, candidate, keep & avail),
            })
        folds.append({
            'held_out_batch': group, 'training_cells': len(training),
            'scored_cells': len(rows),
            'baseline_mape_percent': float(np.mean([
                row['baseline_mape_percent'] for row in rows])),
            'candidate_mape_percent': float(np.mean([
                row['candidate_mape_percent'] for row in rows])),
            'fixed_fusion_mape_percent': float(np.mean([
                row['fixed_fusion_mape_percent'] for row in rows])),
            'by_cell': rows,
        })
        print(group, 'baseline', folds[-1]['baseline_mape_percent'],
              'candidate', folds[-1]['candidate_mape_percent'],
              'fusion', folds[-1]['fixed_fusion_mape_percent'], flush=True)
    rows = [row for fold in folds for row in fold['by_cell']]
    report = {
        'status': 'unpromoted MIT whole-batch 750 s current-SOH development ablation',
        'input': 'current/past 750 s discharge signals where available; otherwise 300 s baseline',
        'warning': 'prediction time is 750 s into discharge or its earlier end, not 300 s',
        'summary': {'scored_cells': len(rows),
                    'observation_weighted_750s_availability': float(np.average(
                        [row['long_prefix_available_fraction'] for row in rows],
                        weights=[row['scored_observations'] for row in rows])),
                    'baseline_mape_percent': float(np.mean([
                        row['baseline_mape_percent'] for row in rows])),
                    'candidate_mape_percent': float(np.mean([
                        row['candidate_mape_percent'] for row in rows])),
                    'fixed_fusion_mape_percent': float(np.mean([
                        row['fixed_fusion_mape_percent'] for row in rows]))},
        'folds': folds,
    }
    OUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    ids = [row['cell_id'] for row in records]
    np.savez_compressed(OOF,
                        cell_id=np.concatenate([np.full(len(cells[key].cycle), key)
                                                for key in ids]),
                        cycle=np.concatenate([cells[key].cycle for key in ids]),
                        target=np.concatenate([cells[key].soh for key in ids]),
                        soh_available=np.concatenate([cells[key].soh_available for key in ids]),
                        long_available=np.concatenate([available[key] for key in ids]),
                        candidate=np.concatenate([heldout[key]['candidate'] for key in ids]),
                        fixed_fusion=np.concatenate([heldout[key]['fusion'] for key in ids]))
    print('SUMMARY', report['summary'])
    print('WROTE', OUT, OOF)


if __name__ == '__main__':
    main()
