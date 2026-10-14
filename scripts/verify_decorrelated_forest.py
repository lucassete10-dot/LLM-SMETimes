"""Independent preprocessing and sklearn reconstruction of all new forests."""
import argparse,json
from pathlib import Path
import joblib,numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from threadpoolctl import threadpool_limits
from battery_smetimes.decorrelated_forest import predict
from battery_smetimes.temporal_features import matrix
from battery_smetimes.tail_boost import labels,is_nca
from scripts.train_crossfit_drift_smoke import clean
from scripts.train_tail_boost import load_cells,NCA
from scripts.train_curve_fidelity import AUDIT
from scripts.prepare_same_type_forecast import save_json,sha
from scripts.audit_tail_numeric_gates import metric


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    root=parser.parse_args().run;cfg=json.loads((root/'config.json').read_text())
    assert json.loads((root/'progress.json').read_text())['training_complete']
    if cfg.get('fivefold'):
        for entry in cfg['provenance']:assert sha(entry['file'])==entry['sha256']
    else:
        source=Path(cfg['source_run']);assert sha(source/'config.json')==cfg['source_config_sha256']
    assert sha(cfg.get('preregistration_file','docs/soh_decorrelated_forest_stage_20261005.md'))==cfg['preregistration_sha256']
    for name,digest in cfg['source_sha256'].items():assert sha(root/'source_snapshot'/name)==digest
    for f in ('battery_smetimes/decorrelated_forest.py','battery_smetimes/temporal_features.py','battery_smetimes/tail_boost.py','battery_smetimes/research_baselines.py','battery_smetimes/forecast.py'):
        assert sha(f)==cfg['source_sha256'][Path(f).name]
    if cfg.get('fivefold'):
        assert cfg['candidates_by_dataset']=={'MIT':[None,dict(max_features=.5,augmented=False)],'XJTU':[None,dict(max_features=.25,augmented=True)],'Zhu2022_NCA':[None,dict(max_features=.25,augmented=False)]}
    else:assert cfg['candidates']==[None]+[dict(max_features=f,augmented=a) for f in (.25,.5) for a in (False,True)]
    registry=json.loads((AUDIT/'cell_registry.json').read_text())['cells'];nca=json.loads((NCA/'index.json').read_text())['cells']
    for key,digest in cfg['prepared_sha256'].items():assert digest==(registry[key]['prepared_sha256'] if key in registry else nca[key]['sha256'])
    reports=json.loads((root/'summary.json').read_text());checked=[]
    for plan in cfg['plans']:
        cells=load_cells(plan);assert set(cells)==set(plan['train']+plan['validation']) and not set(cells)&set(plan['test'])
        control_path=Path(cfg['control_paths'][plan['name']]) if cfg.get('fivefold') else source/plan['name']/'base.joblib'
        assert sha(control_path)==cfg['base_sha256'][plan['name']]
        candidates=cfg['candidates_by_dataset'][plan['dataset']] if cfg.get('fivefold') else cfg['candidates']
        for index,params in enumerate(candidates):
            folder=root/plan['name']/f'candidate_{index}';b=joblib.load(folder/'model.joblib')
            assert b['train_ids']==plan['train'];independent=None
            if params is None:
                assert b['base']['train_ids']==plan['train'] and b['base']['seed']==cfg['seed']
                assert b['base']['parameters']==dict(model='extra_trees',leaf=1,tail_weight=3,local=False)
            if params is not None:
                assert b['seed']==cfg['seed'] and b['max_features']==params['max_features'] and b['augmented']==params['augmented']
                xs=[];ys=[];ws=[]
                for key in plan['train']:
                    x,y=labels(cells[key]);idx=np.unique(np.linspace(0,len(y)-1,min(96,len(y))).round().astype(int))
                    xs.append(matrix(clean(cells[key]),x[idx],params['augmented']));ys.append(y[idx])
                    w=np.linspace(1,3,len(idx));ws.append(w/w.sum())
                z,y,w=np.concatenate(xs),np.concatenate(ys),np.concatenate(ws);w*=len(w)/w.sum()
                median=[]
                for j in range(z.shape[1]):
                    v=z[:,j][np.isfinite(z[:,j])];median.append(np.median(v) if len(v) else 0)
                median=np.asarray(median)
                np.testing.assert_allclose(median,b['imputer'].statistics_,atol=1e-12,rtol=0)
                z=np.where(np.isfinite(z),z,median)
                p=b['model'].get_params();fixed=dict(n_estimators=128,min_samples_leaf=1,max_depth=None,max_features=params['max_features'],random_state=cfg['seed'],n_jobs=2,bootstrap=False)
                for k,v in fixed.items():assert p[k]==v
                independent=ExtraTreesRegressor(**fixed).fit(z,y,sample_weight=w)
            rows={r['cell_id']:r for r in json.loads((folder/'validation_metrics.json').read_text())}
            assert set(rows)==set(plan['validation']);scores=[];values_all=[]
            with np.load(folder/'validation_predictions.npz') as arrays:
                assert len(arrays.files)==3*len(rows)
                for key in plan['validation']:
                    c=cells[key];x,y=labels(c);cc=clean(c);pred=predict(b,cc,x)
                    np.testing.assert_array_equal(x,arrays[key+'__100__cycle']);np.testing.assert_array_equal(y,arrays[key+'__100__truth'])
                    np.testing.assert_allclose(pred,arrays[key+'__100__prediction'],atol=1e-12,rtol=0)
                    if independent is not None:
                        q=matrix(cc,x,params['augmented']);q=np.where(np.isfinite(q),q,median)
                        np.testing.assert_allclose(pred,independent.predict(q),atol=1e-12,rtol=0)
                    if is_nca(cc):
                        poison=dict(cc,early_cycle=np.r_[cc['early_cycle'],101],early_raw=np.vstack([cc['early_raw'],np.full((1,cc['early_raw'].shape[1]),np.inf)]),truth=np.inf,cycle_life=np.inf)
                    else:
                        poison=dict(cc,cycle=np.r_[cc['cycle'],101],raw=np.vstack([cc['raw'],np.full((1,cc['raw'].shape[1]),np.inf)]),soh=np.inf,cycle_life=np.inf)
                    np.testing.assert_allclose(pred,predict(b,poison,x),atol=1e-12,rtol=0)
                    try:predict(b,dict(cc,physical_family='different'),x)
                    except ValueError:pass
                    else:raise AssertionError('family isolation failed')
                    values=metric(x,y,pred)
                    for f,v in values.items():np.testing.assert_allclose(v,rows[key][f],atol=1e-10,rtol=0)
                    scores.append(sum(values[f] for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp'))/4)
                    values_all.append(values);checked.append(dict(cell_id=key,candidate=index,points=len(x),model_sha256=sha(folder/'model.joblib'),predictions_sha256=sha(folder/'validation_predictions.npz')))
            report=next(r for r in reports if r['fold']==plan['name'] and r['candidate']==index)
            np.testing.assert_allclose(np.mean(scores)+.25*np.quantile(scores,.9),report['score'],atol=1e-10,rtol=0)
            for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp'):
                np.testing.assert_allclose(np.mean([v[f] for v in values_all]),report[f],atol=1e-10,rtol=0)
            print('PASS '+plan['name']+' candidate='+str(index),flush=True)
    expected=sum(len(p['validation'])*(2 if cfg.get('fivefold') else len(cfg['candidates'])) for p in cfg['plans'])
    assert len(checked)==expected
    save_json(root/'independent_verification.json',dict(passed=True,checked=checked,retrained_models=15 if cfg.get('fivefold') else 16,source_sha256=sha(__file__),outer_evaluated=False,goal_complete=False))
    print('CURVES='+str(len(checked)))


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
