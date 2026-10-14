"""Verify sparse-input counts, train-only preprocessing and saved forecasts."""
import argparse,json
from pathlib import Path
import joblib,numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from battery_smetimes import uconn_sparse_prefix as model
from scripts.prepare_same_type_forecast import sha,save_json


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('run',type=Path); root=parser.parse_args().run
    cfg=json.loads((root/'config.json').read_text()); status=json.loads((root/'progress.json').read_text())
    assert status['terminal'] and status['success']
    for path,digest in cfg['source_sha256'].items(): assert sha(Path(path))==digest
    path=Path(cfg['index_path']); assert sha(path)==cfg['index_sha256']
    records={r['cell_id']:r for r in json.loads(path.read_text())['cells']}; cells={}
    for cid in cfg['train']+cfg['validation']:
        rec=records[cid]; assert sha(Path(rec['path']))==rec['sha256']
        with np.load(rec['path'],allow_pickle=False) as a: cells[cid]={k:a[k].copy() for k in a.files}
        cells[cid].update(group_id=rec['group_id'],physical_family=rec['physical_family'])
    assert not set(cells)&set(cfg['test']) and not set(cfg['train'])&set(cfg['validation'])
    assert set(cells[k]['group_id'] for k in cfg['train'])==set(cells[k]['group_id'] for k in cfg['validation'])
    fits=curves=metric_count=0
    for origin in cfg['origins']:
        for block in cfg['candidates']:
            folder=root/str(origin)/block; rec=json.loads((folder/'config.json').read_text())
            path=folder/'model.joblib'; assert sha(path)==rec['model_sha256']; b=joblib.load(path)
            assert b['train_ids']==cfg['train'] and b['origin']==origin and b['block']==block
            xs=[]
            for cid in cfg['train']:
                c=cells[cid]['source_cycle']; xs.append(model.inputs(cells[cid],origin,c[c>origin],block))
            imputer=SimpleImputer(strategy='median',keep_empty_features=True)
            scaler=StandardScaler().fit(imputer.fit_transform(np.concatenate(xs)))
            for actual,expected in [(b['imputer'].statistics_,imputer.statistics_),
                                    (b['scaler'].mean_,scaler.mean_),(b['scaler'].scale_,scaler.scale_)]:
                np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-12)
            rows=json.loads((folder/'metrics.json').read_text())
            assert set(r['cell_id'] for r in rows)==set(cfg['validation'])
            with np.load(folder/'predictions.npz',allow_pickle=False) as arrays:
                for cid in cfg['validation']:
                    cell=cells[cid]; take=cell['source_cycle']>origin
                    c=cell['source_cycle'][take]; y=cell['capacity_ah'][take]/cfg['label_reference_ah']
                    p=model.predict(b,cell,c)
                    for field,value in [('cycle',c),('truth',y),('prediction',p)]:
                        np.testing.assert_allclose(arrays[cid+'__'+field],value,rtol=0,atol=1e-12)
                    for row in [r for r in rows if r['cell_id']==cid]:
                        horizon=row['horizon']; mask=(np.ones(len(c),bool) if horizon=='operational' else
                            c>200 if horizon=='common' else c<=origin+int(horizon.split('_')[1]))
                        cc,yy,pp=c[mask],y[mask],p[mask]; e=pp-yy; late=cc>=cc[0]+.8*(cc[-1]-cc[0])
                        for name,value in dict(mape_percent=100*np.mean(abs(e)/yy),mae_soh=np.mean(abs(e)),
                            rmse_soh=np.sqrt(np.mean(e**2)),late20_mae_pp=100*np.mean(abs(e[late]))).items():
                            np.testing.assert_allclose(row[name],value,rtol=0,atol=1e-10)
                        if horizon in ('operational','common'):
                            assert row['actual_input_rpts']==int(np.sum(cell['source_cycle']<=origin))
                        levels=[[np.median(values[np.argsort(abs(cc-boundary),kind='stable')[:min(5,len(cc))]])
                                 for boundary in np.linspace(cc[0],cc[-1],4)] for values in (yy,pp)]
                        yd,pd=-100*np.diff(levels[0]),-100*np.diff(levels[1])
                        np.testing.assert_allclose(row['stage_error_pp'],max(abs(yd-pd)),rtol=0,atol=1e-10)
                        assert row['acceleration_direction_reversed']==bool(yd[-1]-yd[0]>1 and pd[-1]-pd[0]<-1)
                        metric_count+=1
                    curves+=1
            fits+=1
    assert fits==status['completed_fits']==16
    proof=dict(passed=True,verified_models=fits,verified_curves=curves,verified_metric_rows=metric_count,
        exact_replay=True,train_preprocessing_rebuilt=True,input_observation_counts_verified=True,
        independent_basic_stage_reversal_metrics=True,independent_test=False,goal_complete=False,
        limitation='Provisional capacity reference, group physical protocols pending; single split and seed; not a final knee/curve acceptance.')
    save_json(root/'verification.json',proof); print(json.dumps(proof))


if __name__=='__main__':main()
