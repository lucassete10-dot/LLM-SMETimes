"""Train-side LOO pace probe with a registered sparse-label window; no outer cells."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from battery_smetimes.continuous_length import features
from battery_smetimes.condition_length import key
from scripts.train_continuous_length_pilot import load_nominal, target
from scripts.prepare_same_type_forecast import save_json, sha


def main():
    base = Path('experiments/condition_length/20261008T020929_340690Z')
    plan = json.loads((base / 'config.json').read_text())['plan']
    cells, _ = load_nominal(dict(plan, validation=[]))
    assert set(cells) == set(plan['train'])
    assert not set(cells) & set(plan['validation'] + plan['test'])
    root = Path('experiments/sparse_pace_probe') / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    root.mkdir(parents=True)
    labels, coverage = {}, []
    for cid, cell in cells.items():
        c, y = target(cell, 100)
        m = (c >= 101) & (c <= 150)
        c, y = c[m], y[m]
        try:
            if len(c) < 3 or c[-1] - c[0] < 20:
                raise ValueError('less than 3 observed labels spanning 20 cycles in 101..150')
            for origin in (75, 100):
                features(cell, origin, 'VIT')
            slope = np.linalg.lstsq(np.c_[np.ones(len(c)), c], y, rcond=None)[0][1]
            centered = -100 * (c-c.mean()) @ (y-y.mean()) / ((c-c.mean()) @ (c-c.mean()))
            np.testing.assert_allclose(-100*slope, centered, atol=1e-10)
            labels[cid] = centered
            coverage.append(dict(cell_id=cid, included=True, observations=len(c), first=float(c[0]), last=float(c[-1])))
        except ValueError as e:
            coverage.append(dict(cell_id=cid, included=False, reason=str(e), not_bad_cell=True))
    files = [Path(__file__), Path('battery_smetimes/continuous_length.py'), Path('battery_smetimes/condition_length.py'), Path('scripts/train_continuous_length_pilot.py')]
    (root / 'source_snapshot').mkdir()
    for path in files:
        shutil.copy2(path, root / 'source_snapshot' / path.name)
    save_json(root / 'config.json', dict(plan=plan, used_train=list(labels), origins=[75,100], blocks=['condition','V','VIT'],
        seed=20261008, window=[101,150], minimum_observations=3, minimum_span=20,
        coverage=coverage, source_sha256={str(p):sha(p) for p in files},
        validation_loaded=False, outer_loaded=False, independent_test=False,
        limitation='Near-term observed slope diagnostic, not full-curve evidence or optimal input length.'))
    print('RUN='+str(root.resolve()), flush=True)
    rows=[]
    try:
        for origin in (75,100):
            for block in ('condition','V','VIT'):
                raw={cid:features(cells[cid],origin,block) for cid in labels}
                for held in labels:
                    train=[cid for cid in labels if cid!=held]
                    enc=OneHotEncoder(sparse_output=False,handle_unknown='ignore')
                    known=enc.fit_transform([[key(cells[cid])] for cid in train])
                    x=np.c_[np.stack([raw[cid] for cid in train]),known]
                    imputer=SimpleImputer(strategy='median',keep_empty_features=True)
                    z=imputer.fit_transform(x); scaler=StandardScaler().fit(z)
                    model=ExtraTreesRegressor(n_estimators=80,min_samples_leaf=2,max_features=.8,random_state=20261008,n_jobs=2)
                    model.fit(scaler.transform(z),[labels[cid] for cid in train])
                    query=np.c_[raw[held][None],enc.transform([[key(cells[held])]])]
                    pred=float(model.predict(scaler.transform(imputer.transform(query)))[0])
                    folder=root/str(origin)/block;folder.mkdir(parents=True,exist_ok=True)
                    path=folder/(held+'.joblib')
                    bundle=dict(train_ids=train,held_out=held,origin=origin,block=block,encoder=enc,imputer=imputer,scaler=scaler,model=model)
                    joblib.dump(bundle,path); replay=joblib.load(path)
                    np.testing.assert_allclose(pred,replay['model'].predict(replay['scaler'].transform(replay['imputer'].transform(query)))[0],atol=1e-12,rtol=0)
                    assert held not in replay['train_ids']
                    peers=[labels[cid] for cid in train if key(cells[cid])==key(cells[held])]
                    rows.append(dict(origin=origin,block=block,cell_id=held,group=key(cells[held]),truth_rate=labels[held],predicted_rate=pred,
                        group_median_baseline=float(np.median(peers or [labels[cid] for cid in train])),model_path=str(path),model_sha256=sha(path)))
                selected=[r for r in rows if r['origin']==origin and r['block']==block]
                print(f'N={origin} {block} rate_MAE_pp_per100={100*np.mean([abs(r["truth_rate"]-r["predicted_rate"]) for r in selected]):.3f}',flush=True)
                save_json(root/'rows.json',rows)
                save_json(root/'progress.json',dict(completed_fits=len(rows),terminal=False))
        save_json(root/'progress.json',dict(completed_fits=len(rows),terminal=True,success=True,goal_complete=False))
    except BaseException:
        import traceback
        save_json(root/'failure.json',dict(traceback=traceback.format_exc()))
        save_json(root/'progress.json',dict(completed_fits=len(rows),terminal=True,success=False))
        raise


if __name__=='__main__':
    main()
