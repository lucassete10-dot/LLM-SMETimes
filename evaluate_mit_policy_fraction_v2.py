"""MIT whole-batch fraction-of-life ablation with declared charge policy.

Policies are available before cycling and do not include capacity or life
labels. This tests whether the fraction target can use operating-condition
features that hurt the earlier direct-RUL route. No test-batch fitting.
"""

import json
from pathlib import Path

import numpy as np

from battery_v2_features import load_cell, load_manifest
from evaluate_mit_fractional_life_v2 import LANDMARKS, score_cell
from evaluate_mit_policy_rul_v2 import policy_values
from evaluate_paper_inspired_rul_v2 import features as signal_features
from evaluate_xjtu_fractional_life_v2 import fit_fraction, predict_endpoint
from train_rul_v2 import eligible
from train_soh_v2 import outer_splits


OUT = Path('battery_models_v2/mit_fractional_life_challenger')


def main():
    records = [row for row in load_manifest()['cells'] if row['dataset'] == 'MIT']
    cells = {row['cell_id']: load_cell(row) for row in records}
    policy = {row['cell_id']: policy_values(row) for row in records}
    features = {key: np.column_stack((signal_features(cell),
                np.tile(policy[key], (len(cell.cycle), 1))))
                for key, cell in cells.items()}
    folds = []
    heldout = {}
    for group, train_all, validation_all in outer_splits(records, 'MIT'):
        train = [key for key in train_all if cells[key].endpoint_cycle is not None]
        val = [key for key in validation_all if cells[key].endpoint_cycle is not None]
        model = fit_fraction(cells, train, features)
        rows = []
        for key in val:
            endpoint = predict_endpoint(model, cells[key], features[key])
            heldout[key] = endpoint
            # Reuse the exact origin/landmark eligibility as signal-only.
            result = score_cell(cells[key], np.zeros(len(endpoint)), endpoint)
            if result is not None:
                rows.append({'cell_id': key, 'group': group,
                             'origin_ape_percent': result['fraction_origin_ape_percent'],
                             'landmarks': {landmark: value['fraction_ape_percent']
                                           for landmark, value in result['landmarks'].items()}})
        folds.append({'held_out': group, 'scored_cells': len(rows),
                      'origin_mape_percent': float(np.mean([
                          row['origin_ape_percent'] for row in rows])),
                      'by_cell': rows})
        print(group, 'n', len(rows), 'policy fraction EOL',
              round(folds[-1]['origin_mape_percent'], 2), flush=True)
    rows = [row for fold in folds for row in fold['by_cell']]
    landmarks = {str(landmark): {
        'scored_cells': sum(str(landmark) in row['landmarks'] for row in rows),
        'eol_mape_percent': float(np.mean([
            row['landmarks'][str(landmark)] for row in rows
            if str(landmark) in row['landmarks']]))}
        for landmark in LANDMARKS}
    report = {'status': 'unpromoted MIT whole-batch development ablation',
              'input': 'capacity-free discharge signal plus declared fast-charge policy',
              'endpoint': 'source 0.88 Ah; 12 right-censored cells excluded',
              'summary': {'scored_cells': len(rows),
                          'origin_eol_mape_percent': float(np.mean([
                              row['origin_ape_percent'] for row in rows])),
                          'landmarks': landmarks},
              'folds': folds}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'policy_fraction_cv.json').write_text(json.dumps(report, indent=2),
                                                 encoding='utf-8')
    ids, cycles, values = [], [], []
    for key, endpoint in heldout.items():
        keep = eligible(cells[key])
        ids.extend([key]*int(keep.sum()))
        cycles.extend(cells[key].cycle[keep].tolist())
        values.extend(endpoint[keep].tolist())
    np.savez_compressed(OUT / 'policy_fraction_oof.npz',
                        cell_id=np.asarray(ids), cycle=np.asarray(cycles),
                        endpoint=np.asarray(values))
    print('SUMMARY', report['summary'])


if __name__ == '__main__':
    main()
