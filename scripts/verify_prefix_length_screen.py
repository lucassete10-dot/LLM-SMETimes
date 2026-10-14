"""Reload each saved weight bundle, rebuild train statistics and replay arrays."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from scripts.train_prefix_length_screen import matrix, target, predict
from scripts.train_tail_boost import load_cells
from scripts.prepare_same_type_forecast import sha, save_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);args=ap.parse_args()
    root=args.run;cfg=json.loads((root/'config.json').read_text())
    assert json.loads((root/'progress.json').read_text())['terminal']
    for file,digest in cfg['source_sha256'].items():
        assert sha(Path(file))==digest, 'source changed: '+file
    count=curves=0;provenance={}
    for plan in cfg['plans']:
        cells=load_cells(plan)
        assert not set(cells)&set(plan['test'])
        for folder in sorted((root/plan['name']).glob('*/*/*')):
            if not (folder/'model.joblib').exists():continue
            cc=json.loads((folder/'config.json').read_text());bundle=joblib.load(folder/'model.joblib')
            assert bundle['train_ids']==cc['used_train']
            assert set(bundle['train_ids'])<=set(plan['train'])
            assert not set(bundle['train_ids'])&set(plan['validation']+plan['test'])
            assert sha(folder/'model.joblib')==cc['model_sha256']
            origin=cc['origin'];xs=[]
            for key in bundle['train_ids']:
                c,y=target(cells[key]);ii=np.unique(np.linspace(0,len(y)-1,min(96,len(y))).round().astype(int))
                xs.append(matrix(cells[key],origin,c[ii]))
            train=np.concatenate(xs)
            imp=SimpleImputer(strategy='median',keep_empty_features=True)
            z=imp.fit_transform(train)
            np.testing.assert_allclose(imp.statistics_,bundle['imputer'].statistics_,atol=1e-12,rtol=0)
            if bundle['scaler'] is not None:
                scale=StandardScaler().fit(z)
                np.testing.assert_allclose(scale.mean_,bundle['scaler'].mean_,atol=1e-12,rtol=0)
                np.testing.assert_allclose(scale.scale_,bundle['scaler'].scale_,atol=1e-12,rtol=0)
            rows=json.loads((folder/'metrics.json').read_text())
            with np.load(folder/'validation_predictions.npz',allow_pickle=False) as a:
                assert set(a.files)=={r['cell_id']+'__'+k for r in rows for k in ('cycle','truth','prediction')}
                for r in rows:
                    key=r['cell_id'];assert key in plan['validation']
                    c,y=target(cells[key]);p=predict(bundle,cells[key],origin,c)
                    for k,v in [('cycle',c),('truth',y),('prediction',p)]:
                        np.testing.assert_allclose(v,a[key+'__'+k],atol=1e-12,rtol=0)
                    errors=p-y
                    independent=dict(mape_percent=float(100*np.mean(abs(errors)/y)),
                        mae_soh=float(np.mean(abs(errors))),rmse_soh=float(np.sqrt(np.mean(errors**2))),
                        endpoint_pp=float(100*abs(errors[-1])),
                        drop_error_pp=float(100*abs(errors[0]-errors[-1])))
                    for k,v in independent.items():
                        np.testing.assert_allclose(v,r[k],atol=1e-10,rtol=0)
                    bad=dict(cells[key],physical_family='wrong family')
                    try:predict(bundle,bad,origin,c)
                    except ValueError:pass
                    else:raise AssertionError('family not rejected')
                    curves+=1
            count+=1
            for name in ('model.joblib','validation_predictions.npz','metrics.json'):
                provenance[str(folder/name)]=sha(folder/name)
    assert count==json.loads((root/'progress.json').read_text())['completed_fits']
    proof=dict(passed=True,verified_saved_models=count,verified_curves=curves,
        train_statistics_rebuilt=True,exact_weight_replay=True,independent_basic_metrics=True,
        physical_gate_checked=True,outer_loaded=False,artifacts_sha256=provenance,
        limitation='No optimizer refit; stage-shape metric and input-length minimum need further verification/full folds.',
        goal_complete=False)
    save_json(root/'independent_replay_verification.json',proof)
    print(json.dumps({k:v for k,v in proof.items() if k!='artifacts_sha256'}))


if __name__=='__main__':main()
