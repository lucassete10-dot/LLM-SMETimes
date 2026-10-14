"""Compare XJTU RUL candidates at fixed and realistic early checkpoints."""

import json
from pathlib import Path

import numpy as np

from battery_v2_features import load_all
from train_rul_v2 import eligible


ROOT = Path('battery_models_v2')
OUT = ROOT / 'xjtu_throughput_rul_challenger/checkpoint_audit.json'
PREDICTIONS = {
    'baseline': (ROOT / 'rul/xjtu/development_oof_predictions.npz', 'estimate'),
    'selected_gated_log': (ROOT / 'paper_inspired_soh_challenger/'
                           'xjtu_rul_development_oof.npz', 'gated_log'),
    'prior_300s_usage': (ROOT / 'xjtu_combined_rul_challenger/'
                         'development_oof_raw_rul.npz', 'estimate'),
    'throughput_full': (ROOT / 'xjtu_throughput_rul_challenger/'
                        'development_oof_throughput_full.npz', 'estimate'),
    'throughput_relative': (ROOT / 'xjtu_throughput_rul_challenger/'
                            'development_oof_throughput_relative.npz', 'estimate'),
    'relative_invariant': (ROOT / 'xjtu_invariant_rul_challenger/'
                           'development_oof_relative_rul.npz', 'estimate'),
}
CHECKPOINTS = ('first_available', 'cycle_50', 'life_20_percent', 'life_50_percent')


def read_oof(path, cells, field='estimate'):
    result = {key: np.full(len(cell.cycle), np.nan) for key, cell in cells.items()}
    with np.load(path) as archive:
        for key, cycle, estimate in zip(archive['cell_id'], archive['cycle'], archive[field]):
            key = str(key)
            i = int(np.searchsorted(cells[key].cycle, cycle))
            if i >= len(cells[key].cycle) or cells[key].cycle[i] != cycle:
                raise ValueError(f'{key}: OOF prediction cycle mismatch')
            result[key][i] = estimate
    for key, cell in cells.items():
        if not np.isfinite(result[key][eligible(cell)]).all():
            raise ValueError(f'{key}: incomplete OOF predictions')
    return result


def origin(cell, checkpoint):
    available = np.flatnonzero(eligible(cell))
    if checkpoint == 'first_available':
        selected = available[:1]
    elif checkpoint == 'cycle_50':
        selected = available[cell.cycle[available] == 50][:1]
    else:
        fraction = .2 if checkpoint == 'life_20_percent' else .5
        selected = available[cell.cycle[available] <= fraction*cell.endpoint_cycle][-1:]
    return int(selected[0]) if len(selected) and cell.cycle[selected[0]] < cell.endpoint_cycle else None


def main():
    _, all_cells = load_all()
    cells = {key: cell for key, cell in all_cells.items() if cell.dataset == 'XJTU'}
    estimates = {name: read_oof(path, cells, field)
                 for name, (path, field) in PREDICTIONS.items()}
    checkpoints = {}
    for checkpoint in CHECKPOINTS:
        by_model = {}
        for name, predictions in estimates.items():
            rows = []
            for key, cell in cells.items():
                i = origin(cell, checkpoint)
                if i is None:
                    continue
                predicted_eol = cell.cycle[i] + predictions[key][i]
                rows.append({'cell_id': key, 'group': cell.group,
                             'origin_cycle': int(cell.cycle[i]),
                             'eol_ape_percent': float(100*abs(predicted_eol-cell.endpoint_cycle)/
                                                      cell.endpoint_cycle)})
            groups = sorted({row['group'] for row in rows})
            by_model[name] = {'scored_cells': len(rows), 'endpoint_cells': len(cells),
                              'cell_macro_mape_percent': float(np.mean([
                                  row['eol_ape_percent'] for row in rows])),
                              'by_group_mape_percent': {group: float(np.mean([
                                  row['eol_ape_percent'] for row in rows
                                  if row['group'] == group])) for group in groups}}
        checkpoints[checkpoint] = by_model
    report = {'status': 'development OOF checkpoint audit, not independent test',
              'note': 'life fractions are chosen retrospectively for evaluation only; '
                      'fixed cycle 50 excludes sparsely sampled groups',
              'checkpoints': checkpoints}
    OUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    for checkpoint, models in checkpoints.items():
        print(checkpoint, {name: (row['scored_cells'],
                                  round(row['cell_macro_mape_percent'], 2))
                           for name, row in models.items()})


if __name__ == '__main__':
    main()
