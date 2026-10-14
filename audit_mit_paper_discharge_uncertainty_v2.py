"""Conditional MIT development-cell uncertainty for the paper-feature route."""

import json

import numpy as np

from audit_mit_deltaq_uncertainty_v2 import (N_BOOTSTRAP, ROOT, SEED,
                                            stratified_mean)


def main():
    source = json.loads((ROOT / 'paper_discharge_cv.json').read_text(encoding='utf-8'))
    old, new = {}, {}
    for fold in source['folds']:
        name = fold['held_out']
        old[name] = np.asarray([row['ape_percent']['variance_min']
                                for row in fold['by_cell']])
        new[name] = np.asarray([row['ape_percent']['plus_capacity_all']
                                for row in fold['by_cell']])
    rng = np.random.default_rng(SEED)
    mape_samples = stratified_mean(new, rng)
    improvement = stratified_mean({name: old[name]-new[name] for name in old}, rng)
    report = {
        'status': 'descriptive MIT development-cell uncertainty, not independent validation',
        'resampling': f'{N_BOOTSTRAP} within-batch cell draws; seed {SEED}',
        'limitation': 'Only three known batches; this does not estimate new-batch transfer uncertainty or model-selection bias.',
        'route': 'ΔQ(V) variance/minimum plus observed capacity early slope, initial capacity, early gain',
        'scored_cells': sum(map(len, new.values())),
        'mape_percent': float(np.mean(np.concatenate(list(new.values())))),
        'conditional_cell_bootstrap_95pct_mape_interval': [float(v) for v in
            np.quantile(mape_samples, [.025, .975])],
        'old_deltaq_variance_min_minus_new_improvement_points': float(np.mean(
            np.concatenate([old[name]-new[name] for name in old]))),
        'conditional_cell_bootstrap_95pct_improvement_interval': [float(v) for v in
            np.quantile(improvement, [.025, .975])],
        'by_batch_mape_percent': {name: float(np.mean(values))
                                  for name, values in new.items()},
    }
    path = ROOT / 'paper_discharge_uncertainty_audit.json'
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('WROTE', path)
    print('SUMMARY', {key: value for key, value in report.items()
                      if key not in ('status', 'limitation', 'route')})


if __name__ == '__main__':
    main()
