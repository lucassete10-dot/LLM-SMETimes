"""Check refit membership, train-side imputers and full label-free test replay."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from battery_smetimes.curve_fidelity import predict, shape_parameters, targets, condition_key
from battery_smetimes.research_baselines import signal_prefix, build_training_pairs
from scripts.train_curve_fidelity import load_subset
from scripts.prepare_same_type_forecast import sha, save_json


def preprocessing(bundle, cells):
    ids = bundle['train_ids']
    if bundle['arm'] == 'compact':
        # The earlier compact-tree verifier remains available; independently
        # check its full imputer here as for the non-compact arms.
        actual = bundle['compact']; x, _, _, _ = build_training_pairs(cells, ids, [100])
    elif bundle['arm'] == 'local_tree':
        assert set(bundle['global_model']['train_ids']) == set(ids)
        preprocessing(bundle['global_model'], cells)
        for protocol, local in bundle['local'].items():
            expected = {k for k in ids if condition_key(cells[k]) == protocol}
            assert set(local['train_ids']) == expected and len(expected) >= 6
            preprocessing(local, cells)
        return
    elif bundle['arm'] == 'shape':
        actual = bundle; x = np.stack([signal_prefix(cells[k], 100) for k in ids])
        values = np.stack([shape_parameters(*targets(cells[k]), bundle['parameters']['degree']) for k in ids])
        np.testing.assert_allclose(values, bundle['training_shape_parameters'], rtol=0, atol=1e-12)
        np.testing.assert_allclose(np.maximum(values.std(axis=0), .001), bundle['target_scale'], rtol=0, atol=1e-12)
    elif bundle['arm'] == 'ridge':
        actual = bundle['ridge']; x, _, weight, _ = build_training_pairs(cells, ids, [100])
        imputed = SimpleImputer(strategy='median', keep_empty_features=True).fit_transform(x.astype(float))
        offset = imputed[0].copy()
        scaler = StandardScaler().fit(imputed-offset, sample_weight=weight)
        scaler.mean_ += offset
        for field in ('mean_', 'var_', 'scale_'):
            np.testing.assert_allclose(getattr(scaler, field), getattr(actual['scaler'], field), rtol=0, atol=1e-12)
    else:
        actual = bundle; x, _, _, _ = build_training_pairs(cells, ids, [100])
    # Shape fits deliberately preserve the float32 prefix; even-sized medians
    # round in that dtype. Recompute exactly, not with a loosened tolerance.
    original_dtype_input = x if bundle['arm'] == 'shape' else x.astype(float)
    expected = SimpleImputer(strategy='median', keep_empty_features=True).fit(original_dtype_input).statistics_
    np.testing.assert_allclose(actual['imputer'].statistics_, expected, rtol=0, atol=1e-12)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--run', type=Path, required=True)
    root = parser.parse_args().run; cfg = json.loads((root/'config.json').read_text())
    for name, digest in cfg['source_sha256'].items():
        assert sha(root/'source_snapshot'/name) == digest
    out = Path('experiments/curve_fidelity_outer_verification')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True); checked = []; model_count = 0
    for source in cfg['validation_sources']:
        old = Path(source['run']); assert sha(old/'config.json') == source['config_sha256']
        original = json.loads((old/'config.json').read_text()); plan = original['plan']
        cells = load_subset(plan['train']+plan['validation']+plan['test'])
        dev = set(plan['train']+plan['validation'])
        for seed in original['seeds']:
            assert sha(old/str(seed)/'selection.json') == source['selection_sha256'][str(seed)]
            dest = root/plan['name']/str(seed)
            selected = json.loads((dest/'selection.json').read_text())
            assert selected == json.loads((old/str(seed)/'selection.json').read_text())
            assert selected['selected'] == min(selected['candidates'], key=lambda r: r['selection_score'])
            model = joblib.load(dest/'model.joblib')
            assert set(model['train_ids']) == dev and not dev & set(plan['test'])
            assert model['arm'] == selected['selected']['parameters']['arm']
            preprocessing(model, cells)
            model_count += 1
            metrics = {r['cell_id']: r for r in json.loads((dest/'test_metrics.json').read_text())}
            with np.load(dest/'test_predictions.npz') as arrays:
                assert set(metrics) == set(plan['test'])
                for key in plan['test']:
                    cell = cells[key]; x, y = targets(cell)
                    np.testing.assert_array_equal(arrays[key+'__100__cycle'], x)
                    np.testing.assert_array_equal(arrays[key+'__100__truth'], y)
                    saved = arrays[key+'__100__prediction']
                    clean = {k: v for k, v in cell.items() if k not in ('soh','label_cycle','unfiltered_soh')}
                    clean.update(capacity=np.inf, cycle_life=np.inf)
                    poison = dict(clean, cycle=np.r_[clean['cycle'],101,10000],
                        raw=np.vstack([clean['raw'], np.full((2,clean['raw'].shape[1]),np.inf)]))
                    np.testing.assert_allclose(saved, predict(model, clean, x), rtol=0, atol=1e-12)
                    np.testing.assert_allclose(saved, predict(model, poison, x), rtol=0, atol=1e-12)
                    late = x >= (x[0]+x[-1])/2
                    values = dict(mape_percent=100*np.mean(abs(saved-y)/y),
                        mae_soh=np.mean(abs(saved-y)), rmse_soh=np.sqrt(np.mean((saved-y)**2)),
                        late_mape=100*np.mean(abs(saved[late]-y[late])/y[late]),
                        endpoint_pp=100*abs(saved[-1]-y[-1]),
                        drop_error_pp=100*abs((saved[0]-saved[-1])-(y[0]-y[-1])))
                    for field, value in values.items():
                        np.testing.assert_allclose(value, metrics[key][field], rtol=0, atol=1e-10)
                    checked.append(dict(cell_id=key, fold=plan['name'], seed=seed,
                        model_sha256=sha(dest/'model.joblib'), prediction_sha256=sha(dest/'test_predictions.npz'),
                        full_valid_future_targets=len(y), label_removal=True, future_poison=True))
            print('PASS '+plan['name']+' '+str(seed), flush=True)
    assert len({r['cell_id'] for r in checked})*5 == len(checked)
    save_json(out/'verification.json', dict(passed=True, run=str(root.resolve()), rows=checked,
        models=model_count, source_sha256=sha(__file__),
        validation_selection_exact=True, train_only_imputer_recomputed=True,
        full_future_coverage=True, label_free_and_poison_replay=True,
        caution='Checks current experimental implementation, not independent blind-cell generalization.'))
    print('OUTPUT='+str(out.resolve()), flush=True)


if __name__ == '__main__':
    main()
