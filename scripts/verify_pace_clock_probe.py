"""Rebuild train-derived clock supervision/preprocessing and replay all predictions."""
import argparse,json
from pathlib import Path
import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from battery_smetimes import pace_clock,continuous_length as point
from scripts.verify_condition_length import check_base
from scripts.train_continuous_length_pilot import load_nominal,target
from scripts.prepare_same_type_forecast import sha,save_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);root=ap.parse_args().run
    cfg=json.loads((root/'config.json').read_text());status=json.loads((root/'progress.json').read_text())
    assert status['terminal'] and status['success']
    for f,d in cfg['source_sha256'].items():assert sha(Path(f))==d
    models=curves=0
    for plan in cfg['plans']:
        cells,_=load_nominal(plan);assert not set(cells)&set(plan['test'])
        targets={k:target(v,100) for k,v in cells.items()}
        for path in sorted((root/plan['name']).rglob('model.joblib')):
            folder=path.parent
            rec=json.loads((folder/'config.json').read_text());path=folder/'model.joblib';assert sha(path)==rec['model_sha256']
            b=joblib.load(path);ids=rec['train'];assert b['train_ids']==ids
            assert set(ids)<=set(plan['train']) and not set(ids)&set(plan['validation']+plan['test'])
            clock=folder.name=='pace_clock'
            if clock:
                raw={}
                for cid in ids:
                    c,y=targets[cid];m=(c>=101)&(c<=150);cc=c[m]
                    if len(cc)>=3 and cc[-1]-cc[0]>=20:
                        raw[cid]=-100*np.linalg.lstsq(np.c_[np.ones(m.sum()),cc],y[m],rcond=None)[0][1]
                assert list(raw)==b['pace_train_ids']
                for k,v in raw.items():np.testing.assert_allclose(v,b['raw_rates'][k],atol=1e-10,rtol=0)
                positive={k:max(.0005,v) for k,v in raw.items()}
                refs={g:np.median([positive[k] for k in raw if pace_clock.group(cells[k])==g]) for g in {pace_clock.group(cells[k]) for k in raw}}
                fallback=np.median(list(positive.values()))
                np.testing.assert_allclose(fallback,b['fallback_rate'],atol=1e-10,rtol=0)
                for g,v in refs.items():np.testing.assert_allclose(v,b['reference_rates'][g],atol=1e-10,rtol=0)
                assert set(b['encoder'].categories_[0])==set(refs)
                x=np.c_[np.stack([point.features(cells[k],b['origin'],'VIT') for k in raw]),b['encoder'].transform([[pace_clock.group(cells[k])] for k in raw])]
                imp=SimpleImputer(strategy='median',keep_empty_features=True);sc=StandardScaler().fit(imp.fit_transform(x))
                for a,v in ((imp.statistics_,b['imputer'].statistics_),(sc.mean_,b['scaler'].mean_),(sc.scale_,b['scaler'].scale_)):
                    np.testing.assert_allclose(a,v,atol=1e-12,rtol=0)
                transformed={}
                for k in ids:
                    ref=refs.get(pace_clock.group(cells[k]),fallback);factor=np.clip(positive.get(k,ref)/ref,.25,4)
                    np.testing.assert_allclose(factor,b['training_factors'][k],atol=1e-9,rtol=0)
                    c,y=targets[k];transformed[k]=(b['origin']+(c-b['origin'])*b['training_factors'][k],y)
                check_base(b['shape'],cells,transformed,ids)
            else:check_base(b,cells,targets,ids)
            fn=pace_clock.predict if clock else point.predict
            rows=json.loads((folder/'metrics.json').read_text());assert [r['cell_id'] for r in rows]==rec['validation']
            with np.load(folder/'predictions.npz',allow_pickle=False) as data:
                for row in rows:
                    k=row['cell_id'];assert k in plan['validation'];c,y=targets[k];p=fn(b,cells[k],c)
                    for n,v in [('cycle',c),('truth',y),('prediction',p)]:np.testing.assert_allclose(data[k+'__'+n],v,atol=1e-12,rtol=0)
                    e=p-y;late=c>=c[0]+.8*(c[-1]-c[0])
                    for n,v in dict(mape_percent=100*np.mean(abs(e)/y),mae_soh=np.mean(abs(e)),rmse_soh=np.sqrt(np.mean(e**2)),late20_mae_pp=100*np.mean(abs(e[late]))).items():
                        np.testing.assert_allclose(row[n],v,atol=1e-10,rtol=0)
                    curves+=1
            models+=1
    assert models==status['completed_fits']
    proof=dict(passed=True,verified_models=models,verified_curves=curves,independent_clock_labels=True,
        training_only_clock_and_preprocessing=True,weight_replay=True,independent_basic_metrics=True,outer_loaded=False,
        goal_complete=False,limitation='Development folds only; seed count is recorded in run config; no knee-gate or independent-test acceptance.')
    save_json(root/'verification.json',proof);print(json.dumps(proof))


if __name__=='__main__':main()
