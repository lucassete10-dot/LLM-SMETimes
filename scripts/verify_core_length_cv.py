"""Independent train-statistics reconstruction and full core-length replay."""
import argparse,json
from pathlib import Path
import joblib,numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from battery_smetimes import continuous_length as model,sparse_history_point as sparse
from scripts.train_core_length_cv import load
from scripts.train_continuous_length_pilot import target
from scripts.prepare_same_type_forecast import sha,save_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path);root=parser.parse_args().run
    cfg=json.loads((root/'config.json').read_text());state=json.loads((root/'progress.json').read_text())
    backend=sparse if cfg.get('feature_backend')=='sparse_history' else model
    assert state['terminal'] and state['success']
    for key,digest in cfg['source_sha256'].items():
        assert sha(root/'source_snapshot'/Path(key).name)==digest
        if not key.endswith('train_core_length_cv.py'):assert sha(Path(key))==digest
    for key in ['registry','splits','integrity']:assert sha(Path(cfg[key]))==cfg[key+'_sha256']
    if cfg.get('signal_index'):assert sha(Path(cfg['signal_index']))==cfg['signal_index_sha256']
    vals=[cid for plan in cfg['plans'] for cid in plan['validation']]
    assert len(vals)==len(set(vals)) and set(vals)<=set(cfg['development'])
    if len(cfg['plans'])==5:assert set(vals)==set(cfg['development'])
    assert not set(vals)&set(cfg['reserve'])
    coverage=json.loads((root/'coverage.json').read_text());models=curves=rowcount=0
    for plan in cfg['plans']:
        assert not set(plan['train'])&set(plan['validation']+plan['test'])
        cells=load(plan,cfg.get('signal_index'));assert not set(cells)&set(cfg['reserve'])
        for origin in cfg['origins']:
            targets={cid:target(c,origin) for cid,c in cells.items()}
            for arm in cfg['arms']:
                folder=root/plan['name']/str(origin)/arm
                if not folder.exists():
                    cov=next(c for c in coverage if c['fold']==plan['name'] and c['origin']==origin and c['arm']==arm)
                    assert len(cov['train'])<3 or not cov['validation'];continue
                rec=json.loads((folder/'config.json').read_text());assert sha(folder/'model.joblib')==rec['model_sha256']
                b=joblib.load(folder/'model.joblib');assert b['train_ids']==rec['train']
                assert set(rec['train'])<=set(plan['train']) and set(rec['validation'])<=set(plan['validation'])
                xs=[]
                for cid in rec['train']:
                    c,y=targets[cid];ii=np.unique(np.linspace(0,len(c)-1,min(96,len(c))).round().astype(int))
                    xs.append(backend.point_inputs(cells[cid],origin,c[ii],'VIT'))
                fresh=SimpleImputer(strategy='median',keep_empty_features=True);scaler=StandardScaler().fit(fresh.fit_transform(np.concatenate(xs)))
                for a,e in [(b['imputer'].statistics_,fresh.statistics_),(b['scaler'].mean_,scaler.mean_),(b['scaler'].scale_,scaler.scale_)]:
                    np.testing.assert_allclose(a,e,rtol=0,atol=1e-12)
                rows=json.loads((folder/'metrics.json').read_text())
                assert {r['cell_id'] for r in rows}==set(rec['validation'])
                with np.load(folder/'predictions.npz',allow_pickle=False) as data:
                    for cid in rec['validation']:
                        c,y=targets[cid];p=backend.predict(b,cells[cid],c)
                        for field,value in [('cycle',c),('truth',y),('prediction',p)]:np.testing.assert_allclose(data[cid+'__'+field],value,rtol=0,atol=1e-12)
                        assert c[0]>origin
                        for row in [r for r in rows if r['cell_id']==cid]:
                            horizon=row['horizon'];mask=np.ones(len(c),bool) if horizon=='operational' else c>cfg['common_cutoff'] if horizon=='common' else c<=origin+int(horizon.split('_')[1])
                            cc,yy,pp=c[mask],y[mask],p[mask];err=pp-yy;late=cc>=cc[0]+.8*(cc[-1]-cc[0])
                            checks=dict(mape_percent=100*np.mean(abs(err)/yy),mae_soh=np.mean(abs(err)),rmse_soh=np.sqrt(np.mean(err**2)),
                                start_pp=100*abs(err[0]),endpoint_pp=100*abs(err[-1]),drop_error_pp=100*abs(err[-1]-err[0]),
                                late20_mae_pp=100*np.mean(abs(err[late])),late20_mape_percent=100*np.mean(abs(err[late])/yy[late]))
                            total=np.sum((yy-yy.mean())**2)
                            if total>1e-12:checks['r2']=1-np.sum(err**2)/total
                            else:assert row['r2'] is None
                            stages=[-100*np.diff([np.median(v[np.argsort(abs(cc-x),kind='stable')[:min(5,len(cc))]]) for x in np.linspace(cc[0],cc[-1],4)]) for v in [yy,pp]]
                            checks['stage_error_pp']=np.max(abs(stages[0]-stages[1]))
                            checks['fade_normalized_mae']=100*np.mean(abs(err))/max(100*abs(yy[0]-yy[-1]),2)
                            for key,value in checks.items():
                                # R2 can have enormous magnitude on a nearly constant short horizon.
                                # Permit machine-roundoff relative error, not a changed performance gate.
                                np.testing.assert_allclose(row[key],value,rtol=1e-12 if key=='r2' else 0,atol=1e-10)
                            assert row['acceleration_direction_reversed']==bool(stages[0][-1]-stages[0][0]>1 and stages[1][-1]-stages[1][0]<-1)
                            assert row['short_coverage']==bool(len(cc)<10 or cc[-1]-cc[0]<100)
                            if horizon.startswith('fixed_'):assert row['full_horizon_covered']==bool(c[-1]>=origin+int(horizon.split('_')[1]))
                            rowcount+=1
                        curves+=1
                models+=1
    assert models==state['completed_fits']
    proof=dict(passed=True,verified_models=models,verified_curves=curves,verified_metric_rows=rowcount,
        fixed_development_cells=len(cfg['development']),validated_development_cells=len(vals),
        each_development_cell_validates_in_exactly_one_fold=len(cfg['plans'])==5,
        preprocessing_rebuilt=True,exact_prediction_replay=True,independent_basic_stage_reversal_metrics=True,
        outer_loaded=False,independent_test=False,goal_complete=False)
    save_json(root/'verification.json',proof);print(json.dumps(proof))


if __name__=='__main__':main()
