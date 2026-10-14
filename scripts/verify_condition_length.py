"""Replay protocol-local models; rebuild every local and global preprocessing fit."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from battery_smetimes import continuous_length as point
from battery_smetimes import integrated_length as rate
from battery_smetimes import condition_length as local
from scripts.train_continuous_length_pilot import load_nominal, target
from scripts.prepare_same_type_forecast import sha, save_json


def check_base(b, cells, targets, expected):
    assert b['train_ids'] == expected
    if b['kind'] == 'point':
        xs = []
        for k in expected:
            c, _ = targets[k]
            ii = np.unique(np.linspace(0, len(c)-1, min(96, len(c))).round().astype(int))
            xs.append(point.point_inputs(cells[k], b['origin'], c[ii], b['block']))
        x = np.concatenate(xs)
    else:
        x, y, _ = rate.training_arrays(cells, expected, targets, b['origin'], b['block'], b['window'])
        np.testing.assert_allclose(y, b['training_rate_targets'], rtol=0, atol=1e-12)
        check_base(b['anchor'], cells, targets, expected)
    imp = SimpleImputer(strategy='median', keep_empty_features=True)
    sc = StandardScaler().fit(imp.fit_transform(x))
    for a, v in ((imp.statistics_, b['imputer'].statistics_), (sc.mean_, b['scaler'].mean_), (sc.scale_, b['scaler'].scale_)):
        np.testing.assert_allclose(a, v, rtol=0, atol=1e-12)


def check_bundle(b, cells, targets, expected):
    assert b['train_ids'] == expected
    check_base(b['base'], cells, targets, expected)
    for protocol, child in b['local'].items():
        ids = [k for k in expected if local.key(cells[k]) == protocol]
        assert len(ids) >= b['minimum']
        check_bundle(child, cells, targets, ids)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('run', type=Path); root = ap.parse_args().run
    cfg = json.loads((root/'config.json').read_text())
    status = json.loads((root/'progress.json').read_text())
    assert status['terminal'] and status['success']
    for f, digest in cfg['source_sha256'].items():
        assert sha(Path(f)) == digest, f
    plan = cfg['plan']; cells, _ = load_nominal(plan)
    assert not set(cells) & set(plan['test'])
    targets = {k: target(c, 100) for k, c in cells.items()}
    models = curves = 0
    for path in sorted(root.rglob('model.joblib')):
        folder = path.parent; rec = json.loads((folder/'config.json').read_text())
        assert sha(path) == rec['model_sha256']
        assert set(rec['used_train']) <= set(plan['train'])
        assert not set(rec['used_train']) & set(plan['validation'] + plan['test'])
        b = joblib.load(path); check_bundle(b, cells, targets, rec['used_train'])
        rows = json.loads((folder/'metrics.json').read_text())
        assert [r['cell_id'] for r in rows] == rec['validation']
        with np.load(folder/'validation_predictions.npz', allow_pickle=False) as a:
            assert set(a.files) == {r['cell_id']+'__'+n for r in rows for n in ('cycle','truth','prediction')}
            for row in rows:
                k = row['cell_id']; assert k in plan['validation']
                c, y = targets[k]; p = local.predict(b, cells[k], c)
                child, matched = local.route(b, cells[k])
                assert matched == row['used_local'] and len(child['train_ids']) == row['route_train_count']
                for name, value in (('cycle',c),('truth',y),('prediction',p)):
                    np.testing.assert_allclose(value, a[k+'__'+name], rtol=0, atol=1e-12)
                e = p-y; late = c >= c[0]+.8*(c[-1]-c[0])
                metrics = dict(mape_percent=100*np.mean(abs(e)/y), mae_soh=np.mean(abs(e)),
                               rmse_soh=np.sqrt(np.mean(e**2)), endpoint_pp=100*abs(e[-1]),
                               late20_mae_pp=100*np.mean(abs(e[late])), drop_error_pp=100*abs(e[0]-e[-1]))
                for name, value in metrics.items():
                    np.testing.assert_allclose(value, row[name], rtol=0, atol=1e-10)
                curves += 1
        models += 1
    assert models == status['completed_fits']
    proof = dict(passed=True, verified_models=models, verified_curves=curves,
                 recursive_local_preprocessing_rebuilt=True, exact_weight_replay=True,
                 independent_basic_metrics=True, outer_loaded=False, goal_complete=False,
                 limitation='Development fold only; shape screens are not independently verified final acceptance.')
    save_json(root/'independent_replay_verification.json', proof)
    print(json.dumps(proof))


if __name__ == '__main__':
    main()
