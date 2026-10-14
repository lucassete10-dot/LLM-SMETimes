"""MIT whole-batch RUL ablation: direct remaining cycles vs life fraction.

Only capacity-free current and past discharge-prefix features are inputs.
The source endpoint is 0.88 Ah for observed events, not exact per-cell 80%
SOH. Censored batteries are excluded from event-regression training/scoring;
this is a limitation, not an assertion that their endpoints are known.
"""

import json
from pathlib import Path

import numpy as np

from battery_v2_features import load_cell, load_manifest
from evaluate_paper_inspired_rul_v2 import (features as signal_features,
                                             fit as fit_direct,
                                             predict as predict_direct)
from evaluate_xjtu_fractional_life_v2 import fit_fraction, predict_endpoint
from train_rul_v2 import eligible
from train_soh_v2 import outer_splits


OUT = Path('battery_models_v2/mit_fractional_life_challenger')
LANDMARKS = (50, 100, 200, 500)


def score_cell(cell, direct_rul, fractional_endpoint):
    origin = max(15, cell.reference_count)
    if (cell.endpoint_cycle is None or origin >= len(cell.cycle) or
            cell.cycle[origin] >= cell.endpoint_cycle):
        return None
    cycle = int(cell.cycle[origin])
    def error(endpoint):
        return float(100*abs(endpoint-cell.endpoint_cycle)/cell.endpoint_cycle)
    row = {'cell_id': cell.cell_id, 'group': cell.group,
           'origin_cycle': cycle, 'source_endpoint_cycle': cell.endpoint_cycle,
           'direct_origin_ape_percent': error(cycle+direct_rul[origin]),
           'fraction_origin_ape_percent': error(fractional_endpoint[origin]),
           'landmarks': {}}
    for landmark in LANDMARKS:
        if cell.endpoint_cycle <= landmark:
            continue
        candidates = np.flatnonzero(cell.cycle <= landmark)
        if not len(candidates) or candidates[-1] < origin:
            continue
        i = int(candidates[-1])
        row['landmarks'][str(landmark)] = {
            'observed_cycle': int(cell.cycle[i]),
            'direct_ape_percent': error(cell.cycle[i]+direct_rul[i]),
            'fraction_ape_percent': error(fractional_endpoint[i])}
    return row


def main():
    records = [row for row in load_manifest()['cells'] if row['dataset'] == 'MIT']
    cells = {row['cell_id']: load_cell(row) for row in records}
    features = {key: signal_features(cell) for key, cell in cells.items()}
    folds = []
    oof = {}
    for group, train_all, validation_all in outer_splits(records, 'MIT'):
        train = [key for key in train_all if cells[key].endpoint_cycle is not None]
        validation = [key for key in validation_all if cells[key].endpoint_cycle is not None]
        direct_model = fit_direct(cells, train, features, 'gated_raw')
        fraction_model = fit_fraction(cells, train, features)
        scored = []
        for key in validation:
            cell = cells[key]
            direct = predict_direct(direct_model, features[key], 'gated_raw')
            fraction = predict_endpoint(fraction_model, cell, features[key])
            oof[key] = (direct, fraction)
            row = score_cell(cell, direct, fraction)
            if row is not None:
                scored.append(row)
        folds.append({'held_out': group, 'training_endpoint_cells': len(train),
                      'validation_endpoint_cells': len(validation),
                      'scored_cells': len(scored),
                      'direct_origin_mape_percent': float(np.mean([
                          row['direct_origin_ape_percent'] for row in scored])),
                      'fraction_origin_mape_percent': float(np.mean([
                          row['fraction_origin_ape_percent'] for row in scored])),
                      'by_cell': scored})
        print(group, 'n', len(scored), 'direct', round(folds[-1][
              'direct_origin_mape_percent'], 2), 'fraction', round(folds[-1][
              'fraction_origin_mape_percent'], 2), flush=True)
    scored = [row for fold in folds for row in fold['by_cell']]
    landmarks = {}
    for landmark in LANDMARKS:
        subset = [row['landmarks'][str(landmark)] for row in scored
                  if str(landmark) in row['landmarks']]
        landmarks[str(landmark)] = {
            'scored_cells': len(subset),
            'direct_mape_percent': float(np.mean([
                row['direct_ape_percent'] for row in subset])) if subset else None,
            'fraction_mape_percent': float(np.mean([
                row['fraction_ape_percent'] for row in subset])) if subset else None}
    report = {'status': 'unpromoted whole-MIT-batch development CV; not independent test',
              'input': 'capacity-free current/past discharge-prefix signal and source cycle',
              'endpoint': 'source 0.88 Ah with observed events only; 12 censored cells excluded',
              'summary': {'scored_cells': len(scored),
                          'direct_origin_mape_percent': float(np.mean([
                              row['direct_origin_ape_percent'] for row in scored])),
                          'fraction_origin_mape_percent': float(np.mean([
                              row['fraction_origin_ape_percent'] for row in scored])),
                          'landmarks': landmarks},
              'folds': folds}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'development_cv.json').write_text(json.dumps(report, indent=2),
                                              encoding='utf-8')
    ids, cycles, direct_values, fraction_values = [], [], [], []
    for key, (direct, fraction) in oof.items():
        keep = eligible(cells[key])
        ids.extend([key]*int(keep.sum()))
        cycles.extend(cells[key].cycle[keep].tolist())
        direct_values.extend(direct[keep].tolist())
        fraction_values.extend(fraction[keep].tolist())
    np.savez_compressed(OUT / 'development_oof_endpoint.npz',
                        cell_id=np.asarray(ids), cycle=np.asarray(cycles),
                        direct_rul=np.asarray(direct_values),
                        fractional_endpoint=np.asarray(fraction_values))
    print('SUMMARY', report['summary'])


if __name__ == '__main__':
    main()
