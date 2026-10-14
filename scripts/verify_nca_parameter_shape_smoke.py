"""Separate supervised-head fitting, trajectory formula, and saved-curve audit."""
import json,sys
from pathlib import Path
import joblib,numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from threadpoolctl import threadpool_limits
from battery_smetimes.tail_boost import inputs
from battery_smetimes.curve_fidelity import shape_parameters
from battery_smetimes.knee_forecast import fit_knee
from battery_smetimes.nca_parameter_shape import predict
from battery_smetimes.tail_selection import predict as control_predict
from scripts.train_tail_boost import load_cells,NCA
from scripts.train_crossfit_drift_smoke import clean
from scripts.prepare_same_type_forecast import sha,save_json
from scripts.train_nasa_robust_inner import detail
from scripts.verify_nasa_robust_inner import close


def main(root):
    cfg=json.loads((root/'config.json').read_text());plan=cfg['plan'];cells=load_cells(plan)
    assert len(plan['train'])==39 and len(plan['validation'])==5 and len(plan['test'])==11
    assert set(cells)==set(plan['train']+plan['validation']) and not set(cells)&set(plan['test'])
    assert not set(plan['train'])&set(plan['validation'])
    assert sha(Path(cfg['parent'])/'config.json')==cfg['parent_config_sha256']
    assert sha(Path(cfg['control_model_file']))==cfg['control_model_sha256']
    assert sha(NCA/'index.json')==cfg['nca_index_sha256']
    index=json.loads((NCA/'index.json').read_text())
    for k,d in cfg['prepared_sha256'].items():assert index['cells'][k]['sha256']==d
    for f,d in cfg['source_sha256'].items():assert sha(root/'source_snapshot'/Path(f).name)==d
    scores=[];curves=0
    for i,parameters in enumerate([None]+cfg['candidates']):
        out=root/f'candidate_{i}';b=joblib.load(out/'model.joblib');fn=control_predict if i==0 else predict
        fresh=None
        if i:
            assert b['train_ids']==plan['train'] and b['parameters']==parameters
            x=np.vstack([inputs(clean(cells[k]),[101.]) for k in plan['train']])
            coef=np.array([(fit_knee if parameters['shape']=='knee' else shape_parameters)(cells[k]['cycle'],cells[k]['truth']) for k in plan['train']])
            np.testing.assert_array_equal(coef,b['training_coefficients'])
            scale=np.maximum(np.std(coef,axis=0),.001);np.testing.assert_array_equal(scale,b['scale'])
            imp=SimpleImputer(strategy='median',keep_empty_features=True);z=imp.fit_transform(x)
            np.testing.assert_array_equal(imp.statistics_,b['imputer'].statistics_)
            estimator=ExtraTreesRegressor(n_estimators=128,min_samples_leaf=parameters['leaf'],random_state=cfg['seed'],n_jobs=2).fit(z,coef/scale)
            fresh=dict(b,imputer=imp,model=estimator)
        for role,ids in [('training',plan['train']),('validation',plan['validation'])]:
            stored=json.loads((out/(role+'_metrics.json')).read_text());rows=[]
            with np.load(out/(role+'_predictions.npz'),allow_pickle=False) as a:
                assert set(a.files)=={k+'__100__'+f for k in ids for f in ('cycle','truth','prediction')}
                for pos,k in enumerate(ids):
                    c,y,p=[a[k+'__100__'+f] for f in ('cycle','truth','prediction')]
                    np.testing.assert_array_equal(c,cells[k]['cycle']);np.testing.assert_array_equal(y,cells[k]['truth'])
                    query=clean(cells[k]);np.testing.assert_allclose(p,fn(b,query,c),rtol=0,atol=1e-12)
                    if fresh:
                        np.testing.assert_allclose(p,predict(fresh,query,c),rtol=0,atol=1e-12)
                        coef=b['model'].predict(b['imputer'].transform(inputs(query,[101.])))[0]*b['scale'];h=(c-100)/100
                        oracle=coef[0]-coef[1]*h-coef[2]*(np.maximum(h-coef[3],0)**2 if parameters['shape']=='knee' else h*h)
                        np.testing.assert_allclose(p,oracle,rtol=0,atol=1e-12)
                        assert np.all(np.diff(p)<=1e-12)
                    poison=dict(query,truth=np.inf,cycle=np.inf,soh=np.inf,capacity=np.inf,cycle_life=np.inf,
                        early_cycle=np.r_[query['early_cycle'],101.],early_raw=np.vstack([query['early_raw'],np.full((1,32),np.inf)]))
                    np.testing.assert_allclose(p,fn(b,poison,c),rtol=0,atol=1e-12)
                    try:fn(b,dict(query,physical_family='INVALID'),c)
                    except ValueError:pass
                    else:raise AssertionError('family gate')
                    r=dict(cell_id=k,**detail(c,y,p),start_error_pp=float(100*abs(p[0]-y[0])));close(stored[pos],r)
                    for f,v in dict(mape_percent=100*np.mean(abs(p-y)/y),endpoint_pp=100*abs(p[-1]-y[-1]),
                        drop_error_pp=100*abs((p[0]-p[-1])-(y[0]-y[-1]))).items():np.testing.assert_allclose(r[f],v,rtol=0,atol=1e-10)
                    rows.append(r);curves+=1
            if role=='validation':vr=rows
        s=np.array([np.mean([r[f] for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp')]) for r in vr])
        scores.append(dict(candidate=i,parameters=parameters,score=float(s.mean()+.25*np.percentile(s,90)),
            **{f:float(np.mean([r[f] for r in vr])) for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp','start_error_pp')}))
    sel=json.loads((root/'selection.json').read_text())
    for a,b in zip(sel['candidates'],scores):close(a,b)
    best=min(scores,key=lambda r:(r['score'],r['candidate']));old=scores[0]
    expand=best['candidate']>0 and best['score']<old['score'] and best['mape_percent']<=old['mape_percent']+.1 and best['endpoint_pp']<=old['endpoint_pp'] and best['drop_error_pp']<=old['drop_error_pp']
    close(sel['selected'],best);assert sel['expand']==expand and curves==220
    save_json(root/'independent_verification.json',dict(passed=True,fresh_heads=4,curves=curves,expand=expand,outer_loaded=False,
        scope='Separate head fitting and trajectory arithmetic; shared input and training coefficient solvers. Prediction equality, not whole-state comparison.',
        verifier_sha256=sha(Path(__file__))))
    print('PASS:4 refitted heads/220 curves/shape formula/monotonicity/input isolation/selection; expand='+str(expand))


if __name__=='__main__':
    with threadpool_limits(limits=2):main(Path(sys.argv[1]))
