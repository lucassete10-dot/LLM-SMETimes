"""Reconstruct preprocessing, replay saved weights and recalculate basic metrics."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from battery_smetimes import continuous_length as point
from battery_smetimes import integrated_length as rate
from scripts.train_continuous_length_pilot import load_nominal,target
from scripts.prepare_same_type_forecast import sha,save_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);root=ap.parse_args().run
    cfg=json.loads((root/'config.json').read_text());status=json.loads((root/'progress.json').read_text())
    assert status['terminal'] and status['success']
    for f,d in cfg['source_sha256'].items():assert sha(Path(f))==d,f
    if cfg.get('dataset')=='HUST':
        assert sha(Path(cfg['source_index']))==cfg['index_sha256']
        index=json.loads(Path(cfg['source_index']).read_text());records={r['cell_id']:r for r in index['cells']}
        cells={}
        for k in cfg['train']+cfg['validation']:
            r=records[k];assert sha(Path(r['path']))==r['sha256']
            with np.load(r['path'],allow_pickle=False) as a:cells[k]={n:a[n].copy() for n in a.files}
            cells[k]['physical_family']=r['physical_family']
        groups=[(cfg,cells,root)]
    else:groups=[(plan,load_nominal(plan)[0],root/plan['name']) for plan in cfg['plans']]
    count=curves=0
    for plan,cells,parent in groups:
        assert not set(cells)&set(plan['test'])
        targets={k:target(c,cfg.get('common_cutoff',100)) for k,c in cells.items()}
        for modelpath in sorted(parent.rglob('model.joblib')):
            folder=modelpath.parent;cc=json.loads((folder/'config.json').read_text());b=joblib.load(modelpath)
            assert sha(modelpath)==cc['model_sha256'];assert b['train_ids']==cc['used_train']
            assert set(b['train_ids'])<=set(plan['train']) and not set(b['train_ids'])&set(plan['validation']+plan['test'])
            if b['kind']=='point':
                xs=[]
                for k in b['train_ids']:
                    c,_=targets[k];idx=np.unique(np.linspace(0,len(c)-1,min(96,len(c))).round().astype(int))
                    xs.append(point.point_inputs(cells[k],b['origin'],c[idx],b['block']))
                x=np.concatenate(xs);pred=point.predict
            else:
                x,y,_=rate.training_arrays(cells,b['train_ids'],targets,b['origin'],b['block'],b['window'])
                np.testing.assert_allclose(y,b['training_rate_targets'],rtol=0,atol=1e-12)
                assert b['anchor']['train_ids']==b['train_ids'];pred=rate.predict
            imp=SimpleImputer(strategy='median',keep_empty_features=True);sc=StandardScaler().fit(imp.fit_transform(x))
            for a,v in [(imp.statistics_,b['imputer'].statistics_),(sc.mean_,b['scaler'].mean_),(sc.scale_,b['scaler'].scale_)]:
                np.testing.assert_allclose(a,v,rtol=0,atol=1e-12)
            rows=json.loads((folder/'metrics.json').read_text());assert [r['cell_id'] for r in rows]==cc['validation']
            with np.load(folder/'validation_predictions.npz',allow_pickle=False) as a:
                assert set(a.files)=={r['cell_id']+'__'+n for r in rows for n in ('cycle','truth','prediction')}
                for r in rows:
                    k=r['cell_id'];assert k in plan['validation'];c,y=targets[k];p=pred(b,cells[k],c)
                    for n,v in [('cycle',c),('truth',y),('prediction',p)]:np.testing.assert_allclose(v,a[k+'__'+n],rtol=0,atol=1e-12)
                    e=p-y;late=c>=c[0]+.8*(c[-1]-c[0])
                    values=dict(mape_percent=100*np.mean(abs(e)/y),mae_soh=np.mean(abs(e)),rmse_soh=np.sqrt(np.mean(e**2)),
                                endpoint_pp=100*abs(e[-1]),late20_mae_pp=100*np.mean(abs(e[late])),drop_error_pp=100*abs(e[0]-e[-1]))
                    for n,v in values.items():np.testing.assert_allclose(v,r[n],rtol=0,atol=1e-10)
                    curves+=1
            count+=1
    assert count==status['completed_fits']
    proof=dict(passed=True,verified_models=count,verified_curves=curves,train_preprocessing_rebuilt=True,
               independent_basic_metrics=True,exact_weight_replay=True,outer_loaded=False,goal_complete=False,
               limitation='No refit or independent knee/stage-gate verification; development results only')
    save_json(root/'independent_replay_verification.json',proof);print(json.dumps(proof))


if __name__=='__main__':main()
