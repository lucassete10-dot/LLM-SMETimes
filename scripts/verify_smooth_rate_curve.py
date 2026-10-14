"""Check training-only preprocessing, supervision KKT conditions and saved replay."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from battery_smetimes import continuous_length as point
from battery_smetimes import condition_length as local
from battery_smetimes import smooth_rate_curve as smooth
from scripts.train_continuous_length_pilot import load_nominal,target
from scripts.prepare_same_type_forecast import sha,save_json
from scripts.verify_condition_length import check_base


def check(b,cells,targets,ids):
    assert b['train_ids']==ids and b['anchor']['train_ids']==ids
    check_base(b['anchor'],cells,targets,ids)
    x=np.stack([point.features(cells[k],b['origin']) for k in ids])
    imp=SimpleImputer(strategy='median',keep_empty_features=True);sc=StandardScaler().fit(imp.fit_transform(x))
    for a,v in ((imp.statistics_,b['imputer'].statistics_),(sc.mean_,b['scaler'].mean_),(sc.scale_,b['scaler'].scale_)):
        np.testing.assert_allclose(a,v,rtol=0,atol=1e-12)
    for k,r in zip(ids,b['training_shape_fits']):
        assert r['cell_id']==k
        c,y=targets[k];ii=np.unique(np.linspace(0,len(c)-1,min(128,len(c))).round().astype(int))
        def independent_basis(c):
            h=(np.asarray(c,float)-b['origin'])/100
            return np.c_[h,.25*np.logaddexp(0,(h[:,None]-np.array([0.,.5,1.,2.,4.,8.,12.]))/.25)]
        a=independent_basis(c[ii])-independent_basis([c[0]])
        delta=y[0]-y[ii];coef=np.array(r['coefficients'])
        penalty=np.r_[0.,np.ones(7)]
        gradient=a.T@(a@coef-delta)/len(ii)+b['strength']*penalty*coef
        assert np.min(coef)>=0 and np.min(gradient)>=-1e-9
        assert np.max(np.abs(gradient[coef>1e-10]),initial=0)<1e-8
        np.testing.assert_allclose(np.sqrt(np.mean((a@coef-delta)**2)),r['fit_rmse_soh'],rtol=0,atol=1e-12)
    for key,child in b['local'].items():
        check(child,cells,targets,[k for k in ids if local.key(cells[k])==key])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);root=ap.parse_args().run
    cfg=json.loads((root/'config.json').read_text());status=json.loads((root/'progress.json').read_text())
    assert status['terminal'] and status['success']
    for f,digest in cfg['source_sha256'].items():assert sha(Path(f))==digest
    cells,_=load_nominal(cfg['plan']);assert not set(cells)&set(cfg['plan']['test'])
    targets={k:target(c,100) for k,c in cells.items()};count=curves=0
    assert set(cfg['used_train'])<=set(cfg['plan']['train'])
    assert not set(cfg['used_train'])&set(cfg['plan']['validation']+cfg['plan']['test'])
    for path in sorted(root.rglob('model.joblib')):
        rec=json.loads((path.parent/'config.json').read_text());assert sha(path)==rec['model_sha256']
        b=joblib.load(path);check(b,cells,targets,cfg['used_train'])
        rows=json.loads((path.parent/'metrics.json').read_text());assert [r['cell_id'] for r in rows]==cfg['validation']
        with np.load(path.parent/'validation_predictions.npz',allow_pickle=False) as a:
            for r in rows:
                k=r['cell_id'];c,y=targets[k];p=smooth.predict(b,cells[k],c)
                for name,v in [('cycle',c),('truth',y),('prediction',p)]:np.testing.assert_allclose(v,a[k+'__'+name],rtol=0,atol=1e-12)
                e=p-y;late=c>=c[0]+.8*(c[-1]-c[0])
                for name,v in dict(mape_percent=100*np.mean(abs(e)/y),mae_soh=np.mean(abs(e)),rmse_soh=np.sqrt(np.mean(e**2)),late20_mae_pp=100*np.mean(abs(e[late]))).items():
                    np.testing.assert_allclose(v,r[name],rtol=0,atol=1e-10)
                curves+=1
        count+=1
    assert count==status['completed_fits']
    proof=dict(passed=True,verified_models=count,verified_curves=curves,supervision_kkt_checked=True,
        recursive_train_preprocessing_rebuilt=True,weight_replay=True,independent_basic_metrics=True,
        independent_test=False,goal_complete=False,limitation='No independent knee/stage-gate check; single-seed development screen.')
    save_json(root/'independent_replay_verification.json',proof);print(json.dumps(proof))


if __name__=='__main__':main()
