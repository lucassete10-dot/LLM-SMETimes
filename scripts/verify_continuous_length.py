"""Independent saved-weight replay and train-only preprocessing reconstruction."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from battery_smetimes.continuous_length import features, point_inputs, predict
from scripts.train_continuous_length_pilot import load_nominal, target, NCA_AUDIT
from scripts.prepare_same_type_forecast import sha, save_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);root=ap.parse_args().run
    cfg=json.loads((root/'config.json').read_text());progress=json.loads((root/'progress.json').read_text())
    assert progress['terminal'] and progress['success']
    for f,d in cfg['source_sha256'].items():assert sha(Path(f))==d,f
    assert sha(NCA_AUDIT)==cfg['nca_audit_sha256']
    assert sha(root/'registration.md')==cfg['registration_sha256']
    count=curves=0
    for plan in cfg['plans']:
        cells,_=load_nominal(plan);assert not set(cells)&set(plan['test'])
        targets={k:target(c,cfg['common_cutoff']) for k,c in cells.items()}
        for folder in sorted((root/plan['name']).glob('*/*/*')):
            cc=json.loads((folder/'config.json').read_text());b=joblib.load(folder/'model.joblib')
            assert sha(folder/'model.joblib')==cc['model_sha256']
            assert b['train_ids']==cc['used_train']
            assert set(b['train_ids'])<=set(plan['train'])
            assert not set(b['train_ids'])&set(plan['validation']+plan['test'])
            assert cc['validation']==[r['cell_id'] for r in json.loads((folder/'metrics.json').read_text())]
            if b['kind']=='continuous':
                x=np.stack([features(cells[k],b['origin'],b['block']) for k in b['train_ids']])
                assert [r['cell_id'] for r in b['training_shape_fits']]==b['train_ids']
            else:
                xs=[]
                for k in b['train_ids']:
                    c,_=targets[k];ii=np.unique(np.linspace(0,len(c)-1,min(96,len(c))).round().astype(int))
                    xs.append(point_inputs(cells[k],b['origin'],c[ii],b['block']))
                x=np.concatenate(xs)
            imp=SimpleImputer(strategy='median',keep_empty_features=True);z=imp.fit_transform(x)
            sc=StandardScaler().fit(z)
            for a,v in [(imp.statistics_,b['imputer'].statistics_),(sc.mean_,b['scaler'].mean_),(sc.scale_,b['scaler'].scale_)]:
                np.testing.assert_allclose(a,v,rtol=0,atol=1e-12)
            with np.load(folder/'validation_predictions.npz',allow_pickle=False) as data:
                rows=json.loads((folder/'metrics.json').read_text())
                assert set(data.files)=={r['cell_id']+'__'+n for r in rows for n in ('cycle','truth','prediction')}
                for r in rows:
                    key=r['cell_id'];assert key in plan['validation'];c,y=targets[key];p=predict(b,cells[key],c)
                    for n,v in [('cycle',c),('truth',y),('prediction',p)]:
                        np.testing.assert_allclose(v,data[key+'__'+n],rtol=0,atol=1e-12)
                    e=p-y;late=c>=c[0]+.8*(c[-1]-c[0])
                    values=dict(mape_percent=100*np.mean(abs(e)/y),mae_soh=np.mean(abs(e)),rmse_soh=np.sqrt(np.mean(e**2)),
                                endpoint_pp=100*abs(e[-1]),drop_error_pp=100*abs(e[0]-e[-1]),late20_mae_pp=100*np.mean(abs(e[late])))
                    for n,v in values.items():np.testing.assert_allclose(v,r[n],rtol=0,atol=1e-10)
                    if b['kind']=='continuous':
                        dense=predict(b,cells[key],np.linspace(c[0],c[-1],400))
                        assert max(np.diff(dense))<=1e-10 and max(np.diff(dense,n=2))<=1e-9
                    curves+=1
            count+=1
    assert count==progress['completed_fits']
    proof=dict(passed=True,verified_models=count,verified_curves=curves,train_statistics_reconstructed=True,
               exact_saved_weight_replay=True,independent_basic_metrics=True,outer_loaded=False,goal_complete=False,
               limitations=['No optimizer refit','Knee/stage gates not independently verified; this is not final acceptance'])
    save_json(root/'independent_replay_verification.json',proof);print(json.dumps(proof))


if __name__=='__main__':main()
