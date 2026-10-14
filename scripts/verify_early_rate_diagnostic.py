"""Rebuild each leave-one-cell-out preprocessing and replay saved diagnostics."""
import argparse,json
from pathlib import Path
import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from battery_smetimes import continuous_length as feature
from battery_smetimes import condition_length as protocol
from scripts.train_continuous_length_pilot import load_nominal,target
from scripts.prepare_same_type_forecast import sha,save_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);root=ap.parse_args().run
    cfg=json.loads((root/'config.json').read_text());status=json.loads((root/'progress.json').read_text())
    assert status['terminal'] and status['success']
    for f,d in cfg['source_sha256'].items():assert sha(Path(f))==d
    ids=cfg['used_train'];plan=dict(cfg['original_plan'],train=ids,validation=[]);cells,_=load_nominal(plan)
    assert set(cells)==set(ids) and not set(cells)&set(plan['test']+cfg['original_plan']['validation'])
    labels={}
    for k in ids:
        c,y=target(cells[k],100);m=(c>=101)&(c<=125)
        # Independent least-squares slope including intercept.
        coef=np.linalg.lstsq(np.c_[np.ones(m.sum()),c[m]],y[m],rcond=None)[0];labels[k]=-100*coef[1]
    rows=json.loads((root/'rows.json').read_text())
    assert len(rows)==status['completed_fits']==len(ids)*len(cfg['origins'])*len(cfg['blocks'])*len(cfg['candidates'])
    for r in rows:
        path=Path(r['model_path']);assert sha(path)==r['model_sha256'];b=joblib.load(path)
        held=r['cell_id'];train=[k for k in ids if k!=held]
        assert b['train_ids']==train and b['held_out']==held
        raw=np.stack([feature.features(cells[k],b['origin'],b['block']) for k in train])
        known=b['encoder'].transform([[protocol.key(cells[k])] for k in train])
        assert set(b['encoder'].categories_[0])=={protocol.key(cells[k]) for k in train}
        x=np.c_[raw,known];imp=SimpleImputer(strategy='median',keep_empty_features=True);sc=StandardScaler().fit(imp.fit_transform(x))
        for a,v in ((imp.statistics_,b['imputer'].statistics_),(sc.mean_,b['scaler'].mean_),(sc.scale_,b['scaler'].scale_)):
            np.testing.assert_allclose(a,v,rtol=0,atol=1e-12)
        query=np.c_[feature.features(cells[held],b['origin'],b['block'])[None],b['encoder'].transform([[protocol.key(cells[held])]])]
        p=b['model'].predict(b['scaler'].transform(b['imputer'].transform(query)))[0]
        np.testing.assert_allclose(p,r['predicted_rate'],rtol=0,atol=1e-12)
        np.testing.assert_allclose(labels[held],r['truth_rate'],rtol=0,atol=1e-10)
        peers=[labels[k] for k in train if protocol.key(cells[k])==protocol.key(cells[held])]
        baseline=np.median(peers if peers else [labels[k] for k in train])
        np.testing.assert_allclose(baseline,r['group_median_baseline'],rtol=0,atol=1e-10)
    proof=dict(passed=True,verified_models=len(rows),training_cells=len(ids),held_cell_excluded=True,
        train_preprocessing_rebuilt=True,independent_OLS_labels=True,exact_weight_replay=True,
        validation_loaded=False,outer_loaded=False,goal_complete=False,
        limitation='Near-term rate diagnostic only; no full-curve or knee accuracy claim.')
    save_json(root/'verification.json',proof);print(json.dumps(proof))


if __name__=='__main__':main()
