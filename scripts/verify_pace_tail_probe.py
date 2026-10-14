"""Independent preprocessing, supervision and saved prediction audit."""
import argparse,json
from pathlib import Path
import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from battery_smetimes import pace_tail,condition_length as local
from battery_smetimes.continuous_length import features
from scripts.train_continuous_length_pilot import load_nominal,target
from scripts.verify_condition_length import check_bundle
from scripts.prepare_same_type_forecast import sha,save_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);root=ap.parse_args().run
    cfg=json.loads((root/'config.json').read_text());status=json.loads((root/'progress.json').read_text())
    assert status['terminal'] and status['success']
    for f,d in cfg['source_sha256'].items():assert sha(Path(f))==d
    cells,_=load_nominal(cfg['plan']);assert not set(cells)&set(cfg['plan']['test'])
    targets={k:target(c,100) for k,c in cells.items()};count=0
    for origin in cfg['origins']:
        folder=root/str(origin);b=joblib.load(folder/'model.joblib')
        assert b['train_ids']==cfg['train']
        assert not set(b['train_ids'])&set(cfg['validation']+cfg['plan']['test'])
        check_bundle(b['base'],cells,targets,cfg['train'])
        ids=b['pace_train_ids'];assert set(ids)<=set(cfg['train'])
        lab=[]
        for cid in ids:
            c,y=targets[cid];m=(c>=101)&(c<=150)
            lab.append(-100*np.linalg.lstsq(np.c_[np.ones(m.sum()),c[m]],y[m],rcond=None)[0][1])
        np.testing.assert_allclose(lab,b['pace_labels'],atol=1e-10,rtol=0)
        assert set(b['encoder'].categories_[0])=={local.key(cells[k]) for k in ids}
        x=np.c_[np.stack([features(cells[k],origin,'VIT') for k in ids]),b['encoder'].transform([[local.key(cells[k])] for k in ids])]
        imp=SimpleImputer(strategy='median',keep_empty_features=True);sc=StandardScaler().fit(imp.fit_transform(x))
        for a,v in ((imp.statistics_,b['imputer'].statistics_),(sc.mean_,b['scaler'].mean_),(sc.scale_,b['scaler'].scale_)):
            np.testing.assert_allclose(a,v,atol=1e-12,rtol=0)
        for group,end in b['support'].items():
            assert end==max(targets[k][0][-1] for k in cfg['train'] if local.key(cells[k])==group)
        for name,extend in [('paired_local_point',False),('pace_tail',True)]:
            rows=json.loads((folder/(name+'_metrics.json')).read_text())
            with np.load(folder/(name+'_predictions.npz'),allow_pickle=False) as data:
                assert [r['cell_id'] for r in rows]==cfg['validation']
                for r in rows:
                    cid=r['cell_id'];c,y=targets[cid];p=pace_tail.predict(b,cells[cid],c,extend)
                    for key,value in [('cycle',c),('truth',y),('prediction',p)]:
                        np.testing.assert_allclose(data[cid+'__'+key],value,atol=1e-12,rtol=0)
                    error=p-y;late=c>=c[0]+.8*(c[-1]-c[0])
                    for key,value in dict(mape_percent=100*np.mean(abs(error)/y),mae_soh=np.mean(abs(error)),rmse_soh=np.sqrt(np.mean(error**2)),late20_mae_pp=100*np.mean(abs(error[late]))).items():
                        np.testing.assert_allclose(r[key],value,atol=1e-10,rtol=0)
                    count+=1
    proof=dict(passed=True,verified_curves=count,recursive_point_preprocessing=True,pace_preprocessing=True,
        independent_pace_labels=True,training_only_support=True,saved_prediction_replay=True,independent_basic_metrics=True,
        outer_loaded=False,goal_complete=False,limitation='Development probe; not shape-gate acceptance or five-seed stability.')
    save_json(root/'verification.json',proof);print(json.dumps(proof))


if __name__=='__main__':main()
