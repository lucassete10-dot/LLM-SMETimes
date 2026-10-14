"""Replay four-cohort voltage-feature ablation with train-only statistics."""
import argparse,json
from pathlib import Path
import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from battery_smetimes import voltage_change as rich,continuous_length as point
from scripts.train_continuous_length_pilot import load_nominal,target
from scripts.prepare_same_type_forecast import sha,save_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);root=ap.parse_args().run
    cfg=json.loads((root/'config.json').read_text());status=json.loads((root/'progress.json').read_text())
    assert status['terminal'] and status['success']
    for p,d in cfg['source_sha256'].items():assert sha(Path(p))==d
    models=curves=0
    for plan in cfg['plans']:
        if plan['dataset']=='HUST':
            path=Path(cfg['hust_index']);assert sha(path)==cfg['hust_index_sha256'];index=json.loads(path.read_text())
            records={r['cell_id']:r for r in index['cells']};cells={}
            for k in plan['train']+plan['validation']:
                r=records[k];assert sha(Path(r['path']))==r['sha256']
                with np.load(r['path'],allow_pickle=False) as data:cells[k]={n:data[n].copy() for n in data.files}
                cells[k]['physical_family']=r['physical_family']
        else:cells,_=load_nominal(plan)
        assert not set(cells)&set(plan['test']);targets={k:target(v,100) for k,v in cells.items()}
        for path in sorted((root/plan['name']).rglob('model.joblib')):
            folder=path.parent;name=folder.name;module=point if name=='point' else rich
            rec=json.loads((folder/'config.json').read_text())
            assert sha(path)==rec['model_sha256'];b=joblib.load(path);ids=rec['train']
            assert b['train_ids']==ids and set(ids)<=set(plan['train'])
            assert not set(ids)&set(plan['validation']+plan['test'])
            xs=[]
            for k in ids:
                c,y=targets[k];ii=np.unique(np.linspace(0,len(c)-1,min(96,len(c))).round().astype(int))
                xs.append(point.point_inputs(cells[k],100,c[ii],'VIT') if name=='point' else rich.inputs(cells[k],100,c[ii]))
            x=np.concatenate(xs);imp=SimpleImputer(strategy='median',keep_empty_features=True);sc=StandardScaler().fit(imp.fit_transform(x))
            for a,v in ((imp.statistics_,b['imputer'].statistics_),(sc.mean_,b['scaler'].mean_),(sc.scale_,b['scaler'].scale_)):
                np.testing.assert_allclose(a,v,atol=1e-12,rtol=0)
            rows=json.loads((folder/'metrics.json').read_text());assert [r['cell_id'] for r in rows]==rec['validation']
            with np.load(folder/'predictions.npz',allow_pickle=False) as data:
                for row in rows:
                    k=row['cell_id'];assert k in plan['validation'];c,y=targets[k];p=module.predict(b,cells[k],c)
                    for n,v in [('cycle',c),('truth',y),('prediction',p)]:np.testing.assert_allclose(data[k+'__'+n],v,atol=1e-12,rtol=0)
                    e=p-y;late=c>=c[0]+.8*(c[-1]-c[0])
                    for n,v in dict(mape_percent=100*np.mean(abs(e)/y),mae_soh=np.mean(abs(e)),rmse_soh=np.sqrt(np.mean(e**2)),late20_mae_pp=100*np.mean(abs(e[late]))).items():
                        np.testing.assert_allclose(row[n],v,atol=1e-10,rtol=0)
                    levels=[]
                    for values in (y,p):
                        levels.append([np.median(values[sorted(range(len(c)),key=lambda j:(abs(c[j]-boundary),j))[:min(5,len(c))]]) for boundary in np.linspace(c[0],c[-1],4)])
                    yd=-100*np.diff(levels[0]);pd=-100*np.diff(levels[1])
                    np.testing.assert_allclose(row['stage_error_pp'],max(abs(yd-pd)),atol=1e-10,rtol=0)
                    assert row['acceleration_direction_reversed']==bool(yd[-1]-yd[0]>1 and pd[-1]-pd[0]<-1)
                    curves+=1
            models+=1
    assert models==status['completed_fits']==2*len(cfg['plans'])*len(cfg['seeds'])
    proof=dict(passed=True,verified_models=models,verified_curves=curves,train_preprocessing_rebuilt=True,exact_weight_replay=True,
        independent_basic_metrics=True,independent_stage_and_reversal_metrics=True,outer_loaded=False,goal_complete=False,limitation='One development fold per cohort; seed count in configuration; not knee acceptance or multi-split superiority proof.')
    save_json(root/'verification.json',proof);print(json.dumps(proof))


if __name__=='__main__':main()
