"""Reconstruct HUST train statistics and independently replay all length horizons."""
import argparse,json
from pathlib import Path
import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from battery_smetimes import continuous_length as point,voltage_change as rich,voltage_interval_features as physics
from scripts.prepare_same_type_forecast import sha,save_json


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('run',type=Path); root=parser.parse_args().run
    cfg=json.loads((root/'config.json').read_text()); status=json.loads((root/'progress.json').read_text())
    assert status['terminal'] and status['success']
    for name,digest in cfg['source_sha256'].items():
        assert sha(root/'source_snapshot'/Path(name).name)==digest
        if Path(name).name!='train_hust_length_pair.py': assert sha(Path(name))==digest
    path=Path(cfg['source_index']); assert sha(path)==cfg['index_sha256']
    records={r['cell_id']:r for r in json.loads(path.read_text())['cells']}; cells={}
    for cid in cfg['train']+cfg['validation']:
        rec=records[cid]; assert sha(Path(rec['path']))==rec['sha256']
        with np.load(rec['path'],allow_pickle=False) as data: cells[cid]={k:data[k].copy() for k in data.files}
        cells[cid]['physical_family']=rec['physical_family']
    assert not set(cells)&set(cfg['test']) and not set(cfg['train'])&set(cfg['validation'])
    if cfg.get('physics_index'):
        ip=Path(cfg['physics_index']); assert sha(ip)==cfg['physics_index_sha256']
        ir={r['cell_id']:r for r in json.loads(ip.read_text())['cells']}
        for cid in cells:
            rec=ir[cid]; assert sha(Path(rec['path']))==rec['sha256']
            with np.load(rec['path'],allow_pickle=False) as data:
                cells[cid].update({k:data[k].copy() for k in data.files})
    models=curves=metric_rows=0
    for origin in cfg['origins']:
        targets={}
        for cid,cell in cells.items():
            c,y=np.asarray(cell['cycle'],float),np.asarray(cell['truth'],float)
            mask=(c>origin)&np.isfinite(y)&(y>0); targets[cid]=(c[mask],y[mask])
        for name in cfg['candidates']:
            module={'point':point,'voltage_change':rich,'voltage_intervals':physics}[name]; folder=root/str(origin)/name
            record=json.loads((folder/'config.json').read_text()); path=folder/'model.joblib'
            assert sha(path)==record['model_sha256']; bundle=joblib.load(path); ids=record['train']
            assert bundle['train_ids']==ids and set(ids)<=set(cfg['train'])
            assert not set(ids)&set(cfg['validation']+cfg['test'])
            xs=[]
            for cid in ids:
                c,y=targets[cid]; ii=np.unique(np.linspace(0,len(c)-1,min(96,len(c))).round().astype(int))
                xs.append(point.point_inputs(cells[cid],origin,c[ii],'VIT') if name=='point' else module.inputs(cells[cid],origin,c[ii]))
            imputer=SimpleImputer(strategy='median',keep_empty_features=True)
            scaler=StandardScaler().fit(imputer.fit_transform(np.concatenate(xs)))
            for actual,expected in [(bundle['imputer'].statistics_,imputer.statistics_),
                                    (bundle['scaler'].mean_,scaler.mean_),(bundle['scaler'].scale_,scaler.scale_)]:
                np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-12)
            rows=json.loads((folder/'metrics.json').read_text())
            assert set(r['cell_id'] for r in rows)==set(record['validation'])
            with np.load(folder/'predictions.npz',allow_pickle=False) as data:
                for cid in record['validation']:
                    c,y=targets[cid]; p=module.predict(bundle,cells[cid],c)
                    for field,values in [('cycle',c),('truth',y),('prediction',p)]:
                        np.testing.assert_allclose(data[cid+'__'+field],values,rtol=0,atol=1e-12)
                    assert c[0]>origin and c[0]<=origin+2
                    for row in [r for r in rows if r['cell_id']==cid]:
                        horizon=row['horizon']
                        mask=(np.ones(len(c),bool) if horizon=='operational' else
                              c>200 if horizon=='common' else c<=origin+int(horizon.split('_')[1]))
                        cc,yy,pp=c[mask],y[mask],p[mask]; e=pp-yy
                        late=cc>=cc[0]+.8*(cc[-1]-cc[0])
                        for metric,value in dict(mape_percent=100*np.mean(abs(e)/yy),mae_soh=np.mean(abs(e)),
                            rmse_soh=np.sqrt(np.mean(e**2)),late20_mae_pp=100*np.mean(abs(e[late]))).items():
                            np.testing.assert_allclose(row[metric],value,rtol=0,atol=1e-10)
                        levels=[[np.median(values[np.argsort(abs(cc-boundary),kind='stable')[:min(5,len(cc))]])
                                 for boundary in np.linspace(cc[0],cc[-1],4)] for values in (yy,pp)]
                        yd,pd=-100*np.diff(levels[0]),-100*np.diff(levels[1])
                        np.testing.assert_allclose(row['stage_error_pp'],max(abs(yd-pd)),rtol=0,atol=1e-10)
                        assert row['acceleration_direction_reversed']==bool(yd[-1]-yd[0]>1 and pd[-1]-pd[0]<-1)
                        if horizon.startswith('fixed_'): assert row['full_horizon_covered']==bool(c[-1]>=origin+int(horizon.split('_')[1]))
                        metric_rows+=1
                    curves+=1
            models+=1
    assert models==status['completed_fits']==len(cfg['origins'])*len(cfg['candidates'])
    proof=dict(passed=True,verified_models=models,verified_curves=curves,verified_metric_rows=metric_rows,
        training_preprocessing_rebuilt=True,exact_weight_replay=True,operational_coverage_checked=True,
        independent_basic_stage_reversal_metrics=True,independent_test=False,goal_complete=False,
        limitation='One fold/seed exploration. Knee/platform acceptance and multi-fold stability remain unverified.')
    save_json(root/'verification.json',proof); print(json.dumps(proof))


if __name__=='__main__': main()
