"""Conditional cell-bootstrap audit of the MIT diagnostic-assisted result.

Only three fixed observed batches exist. Resampling their cells cannot
estimate performance on a new batch, but shows uncertainty around the
development-cell average near the 20% target.
"""

import json
from pathlib import Path

import numpy as np


ROOT = Path('battery_models_v2/mit_qv_challenger')
SEED = 20260927
N_BOOTSTRAP = 10000


def stratified_mean(values_by_batch, rng):
    samples = np.zeros(N_BOOTSTRAP)
    total = sum(len(values) for values in values_by_batch.values())
    for values in values_by_batch.values():
        values = np.asarray(values, dtype=float)
        indices = rng.integers(0, len(values), size=(N_BOOTSTRAP, len(values)))
        samples += values[indices].sum(axis=1)/total
    return samples


def main():
    source = json.loads((ROOT / 'deltaq_stats_cv.json').read_text(encoding='utf-8'))
    baseline = {}
    candidate = {}
    for fold in source['folds']:
        name = fold['held_out']
        baseline[name] = [row['ape_percent']['variance_only']
                          for row in fold['by_cell']]
        candidate[name] = [row['ape_percent']['variance_min']
                           for row in fold['by_cell']]
    difference = {name: np.asarray(baseline[name])-np.asarray(candidate[name])
                  for name in baseline}
    rng = np.random.default_rng(SEED)
    samples = stratified_mean(candidate, rng)
    improvement = stratified_mean(difference, rng)
    report = {'status': 'descriptive MIT development-fold uncertainty only',
              'warning': 'conditional on three already observed batches; not new-batch or independent-test uncertainty',
              'resampling': f'{N_BOOTSTRAP} within-batch cell bootstrap draws, seed {SEED}',
              'candidate': 'ΔQ(V) log variance plus minimum; source 0.88-Ah endpoint',
              'scored_cells': sum(map(len, candidate.values())),
              'mape_percent': float(np.mean([value for group in candidate.values()
                                              for value in group])),
              'conditional_cell_bootstrap_95pct_mape_interval': [float(v) for v in
                  np.quantile(samples, [.025, .975])],
              'variance_only_minus_candidate_improvement_points': float(np.mean([
                  value for group in difference.values() for value in group])),
              'conditional_cell_bootstrap_95pct_improvement_interval': [float(v)
                  for v in np.quantile(improvement, [.025, .975])],
              'by_batch_mape_percent': {name: float(np.mean(values))
                                        for name, values in candidate.items()}}
    (ROOT / 'uncertainty_audit.json').write_text(json.dumps(report, indent=2),
                                                 encoding='utf-8')
    print(report)


if __name__ == '__main__':
    main()
