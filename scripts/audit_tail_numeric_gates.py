"""Recalculate curve-goal gates from saved new and frozen-baseline arrays.

No trainer metric/aggregation function is reused. This evaluates, never selects.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.prepare_same_type_forecast import save_json, sha


def metric(x, y, p):
    assert len(x) == len(y) == len(p) and len(x) >= 2
    assert np.all(np.isfinite(y)) and np.all(y > 0) and np.all(np.isfinite(p))
    late = x >= (x[0] + x[-1])/2
    return dict(mape_percent=float(100*np.mean(np.abs(p-y)/y)),
        late_mape=float(100*np.mean(np.abs(p[late]-y[late])/y[late])),
        rmse_soh=float(np.sqrt(np.mean((p-y)**2))),
        endpoint_pp=float(100*abs(p[-1]-y[-1])),
        drop_error_pp=float(100*abs((p[0]-p[-1])-(y[0]-y[-1]))),
        truth_drop_pp=float(100*(y[0]-y[-1])),
        prediction_drop_pp=float(100*(p[0]-p[-1])))


def aggregate(rows):
    ids = sorted({r['cell_id'] for r in rows})
    seeds = sorted({r['seed'] for r in rows})
    assert len(seeds) == 5
    fields = tuple(metric(np.arange(2), np.ones(2), np.ones(2)))
    cells = []
    for key in ids:
        selected = [r for r in rows if r['cell_id'] == key]
        assert len(selected) == 5 and {r['seed'] for r in selected} == set(seeds)
        c = dict(cell_id=key, **{f: float(np.mean([r[f] for r in selected])) for f in fields})
        c['joint_pass'] = c['mape_percent'] <= 6 and c['late_mape'] <= 8 and c['endpoint_pp'] <= 8 and c['drop_error_pp'] <= 5
        c['severe_flat_failure'] = c['truth_drop_pp'] >= 5 and c['prediction_drop_pp'] < .3*c['truth_drop_pp']
        cells.append(c)
    summary = {f: dict(mean=float(np.mean([c[f] for c in cells])),
        seed_sd=float(np.std([np.mean([r[f] for r in rows if r['seed'] == s]) for s in seeds], ddof=1))) for f in fields}
    summary.update(cells=len(cells), joint_pass_fraction=float(np.mean([c['joint_pass'] for c in cells])),
        severe_flat_fraction=float(np.mean([c['severe_flat_failure'] for c in cells])),
        endpoint_p90=float(np.quantile([c['endpoint_pp'] for c in cells], .9)),
        drop_error_p90=float(np.quantile([c['drop_error_pp'] for c in cells], .9)))
    return summary, cells


def gates(new, old):
    result = dict(full_mape=new['mape_percent']['mean'] <= 6, late_mape=new['late_mape']['mean'] <= 6,
        rmse=new['rmse_soh']['mean'] <= .04, endpoint_mean=new['endpoint_pp']['mean'] <= 5,
        drop_mean=new['drop_error_pp']['mean'] <= 4, joint_fraction=new['joint_pass_fraction'] >= .9,
        severe_fraction=new['severe_flat_fraction'] <= .1,
        full_no_regression=new['mape_percent']['mean'] <= old['mape_percent']['mean'] + .5)
    for f in ('endpoint_p90', 'drop_error_p90'):
        result[f] = new[f] <= (3 if old[f] <= 3 else .8*old[f])
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--run', type=Path, required=True)
    root = parser.parse_args().run
    cfg = json.loads((root/'config.json').read_text())
    assert json.loads((root/'progress.json').read_text())['training_complete']
    assert json.loads((root/'independent_outer_verification.json').read_text())['passed']
    source = Path(cfg['training_source'])
    assert sha(source/'config.json') == cfg['training_config_sha256']
    training = json.loads((source/'config.json').read_text())
    rows = []; old_rows = []; evidence = []
    baseline = Path(cfg['baseline'])
    for plan in training['plans']:
        for is_old, seeds, destination in ((False, training['seeds'], rows), (True, range(20261004, 20261009), old_rows)):
            for seed in seeds:
                folder = (baseline if is_old else root)/plan['name']/str(seed)
                name = ('test/predictions.npz' if cfg['dataset'] not in ('Zhu2022_NCA', 'NASA') else 'predictions.npz') if is_old else 'test_predictions.npz'
                file = folder/name
                with np.load(file) as arrays:
                    expected = {key+'__100__'+f for key in plan['test'] for f in ('cycle', 'truth', 'prediction')}
                    assert expected <= set(arrays.files)
                    for key in plan['test']:
                        x, y, p = [arrays[key+'__100__'+f] for f in ('cycle', 'truth', 'prediction')]
                        if is_old:
                            with np.load(root/plan['name']/str(training['seeds'][0])/'test_predictions.npz') as reference:
                                np.testing.assert_array_equal(x, reference[key+'__100__cycle'])
                                np.testing.assert_array_equal(y, reference[key+'__100__truth'])
                        destination.append(dict(cell_id=key, seed=seed, **metric(x, y, p)))
                evidence.append(dict(file=str(file.resolve()), sha256=sha(file), frozen_baseline=is_old))
    new, cells = aggregate(rows); old, _ = aggregate(old_rows)
    computed = gates(new, old)
    saved = json.loads((root/'summary.json').read_text())
    for label, actual in (('new', new), ('baseline', old)):
        for field, value in actual.items():
            if isinstance(value, dict):
                for f, v in value.items(): np.testing.assert_allclose(v, saved[label][field][f], atol=1e-10, rtol=0)
            else: np.testing.assert_allclose(value, saved[label][field], atol=1e-10, rtol=0)
    assert computed == saved['gates']
    assert bool(all(computed.values())) == saved['numeric_gates_passed']
    save_json(root/'independent_numeric_gate_audit.json', dict(passed=True, numeric_gates_passed=all(computed.values()),
        new=new, baseline=old, gates=computed, cells=cells, evidence=evidence, source_sha256=sha(__file__),
        interpretation='Correct arithmetic is not proof of achieving goal. No model selection or exclusions.', goal_complete=False))
    print('AUDIT_PASS gates='+str(sum(computed.values()))+'/'+str(len(computed)))


if __name__ == '__main__': main()
