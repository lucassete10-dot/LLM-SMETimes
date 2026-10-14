"""Rebuild both train-only preprocessors and replay phase-forecast evidence."""
import argparse,json
from pathlib import Path
import joblib,numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from battery_smetimes import interval_phase_clock as model,voltage_interval_features as physics
from scripts.train_continuous_length_pilot import target
from scripts.prepare_same_type_forecast import sha,save_json


def check_preprocessing(x,imputer,scaler):
    fresh=SimpleImputer(strategy='median',keep_empty_features=True);s=StandardScaler().fit(fresh.fit_transform(x))
    for a,b in [(fresh.statistics_,imputer.statistics_),(s.mean_,scaler.mean_),(s.scale_,scaler.scale_)]:
        np.testing.assert_allclose(a,b,rtol=0,atol=1e-12)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path);root=parser.parse_args().run
    cfg=json.loads((root/'config.json').read_text());progress=json.loads((root/'progress.json').read_text())
    assert progress['terminal'] and progress['success'] and sha(root/'model.joblib')==progress['model_sha256']
    for name,digest in cfg['source_sha256'].items():
        assert sha(root/'source_snapshot'/Path(name).name)==digest
        if not name.endswith('train_hust_interval_phase_probe.py'):assert sha(Path(name))==digest
    cells={}
    for key,digestkey in [('source_index','index_sha256'),('physics_index','physics_index_sha256')]:
        assert sha(Path(cfg[key]))==cfg[digestkey]
        for rec in json.loads(Path(cfg[key]).read_text())['cells']:
            cid=rec['cell_id']
            if cid not in cfg['train']+cfg['validation']:continue
            assert sha(Path(rec['path']))==rec['sha256']
            with np.load(rec['path'],allow_pickle=False) as d:cells.setdefault(cid,{}).update({k:d[k].copy() for k in d.files})
            if 'physical_family' in rec:cells[cid]['physical_family']=rec['physical_family']
    b=joblib.load(root/'model.joblib');assert b['train_ids']==cfg['train']
    assert not set(cfg['train'])&set(cfg['validation']+cfg['test']) and not set(cells)&set(cfg['test'])
    check_preprocessing(np.stack([physics.features(cells[cid],100) for cid in b['train_ids']]),b['clock_imputer'],b['clock_scaler'])
    xs=[]
    for cid in b['train_ids']:
        c,y=target(cells[cid],100)
        candidates=[j for j in range(len(y)-4) if all(y[j:j+5]<=.9)]
        assert candidates;duration=float(c[candidates[0]]-100);assert duration==b['training_durations'][cid]
        ii=np.unique(np.linspace(0,len(c)-1,min(96,len(c))).round().astype(int))
        xs.append(physics.inputs(cells[cid],100,100+(c[ii]-100)*100/duration))
    check_preprocessing(np.concatenate(xs),b['imputer'],b['scaler'])
    rows=json.loads((root/'metrics.json').read_text());count=0
    with np.load(root/'predictions.npz',allow_pickle=False) as data:
        for cid in cfg['validation']:
            c,y=target(cells[cid],100);p=model.predict(b,cells[cid],c)
            for field,value in [('cycle',c),('truth',y),('prediction',p)]:np.testing.assert_allclose(data[cid+'__'+field],value,rtol=0,atol=1e-12)
            for horizon,mask in [('operational',np.ones(len(c),bool)),('common',c>200)]:
                r=next(r for r in rows if r['cell_id']==cid and r['horizon']==horizon)
                cc,yy,pp=c[mask],y[mask],p[mask];err=pp-yy;late=cc>=cc[0]+.8*(cc[-1]-cc[0])
                checks=dict(mape_percent=100*np.mean(abs(err)/yy),mae_soh=np.mean(abs(err)),rmse_soh=np.sqrt(np.mean(err**2)),
                    start_pp=100*abs(err[0]),endpoint_pp=100*abs(err[-1]),drop_error_pp=100*abs(err[-1]-err[0]),
                    late20_mae_pp=100*np.mean(abs(err[late])))
                total=np.sum((yy-yy.mean())**2)
                if total>1e-12:checks['r2']=1-np.sum(err**2)/total
                else:assert r['r2'] is None
                for key,value in checks.items():np.testing.assert_allclose(r[key],value,rtol=0,atol=1e-10)
                stages=[-100*np.diff([np.median(v[np.argsort(abs(cc-x),kind='stable')[:min(5,len(cc))]]) for x in np.linspace(cc[0],cc[-1],4)]) for v in [yy,pp]]
                np.testing.assert_allclose(r['stage_error_pp'],np.max(abs(stages[0]-stages[1])),rtol=0,atol=1e-10)
                assert r['acceleration_direction_reversed']==bool(stages[0][-1]-stages[0][0]>1 and stages[1][-1]-stages[1][0]<-1)
                count+=1
    assert count==len(rows)==2*len(cfg['validation'])
    proof=dict(passed=True,verified_curves=len(cfg['validation']),metric_rows=count,train_preprocessing_rebuilt=True,
        crossing_supervision_recomputed_from_train_only=True,exact_prediction_replay=True,independent_basic_stage_metrics=True,
        independent_test=False,goal_complete=False)
    save_json(root/'verification.json',proof);print(json.dumps(proof))


if __name__=='__main__':main()
