"""Separate sampling/refits, saved-curve replay, metrics and isolation checks."""
import json,sys
from pathlib import Path
import joblib,numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from threadpoolctl import threadpool_limits
from battery_smetimes.tail_boost import inputs,predict
from battery_smetimes.tail_selection import predict as control_predict
from scripts.train_tail_boost import load_cells,NCA
from scripts.train_crossfit_drift_smoke import clean
from scripts.prepare_same_type_forecast import sha,save_json
from scripts.train_nasa_robust_inner import detail
from scripts.verify_nasa_robust_inner import close


def main(root):
    cfg=json.loads((root/'config.json').read_text());plan=cfg['plan'];cells=load_cells(plan)
    assert len(plan['train'])==39 and len(plan['validation'])==5 and len(plan['test'])==11
    assert not set(cells)&set(plan['test']) and not set(plan['train'])&set(plan['validation'])
    assert sha(Path(cfg['parent'])/'config.json')==cfg['parent_config_sha256']
    assert sha(Path(cfg['control_model_file']))==cfg['control_model_sha256']
    assert sha(NCA/'index.json')==cfg['nca_index_sha256']
    index=json.loads((NCA/'index.json').read_text())
    for k,d in cfg['prepared_sha256'].items():assert d==index['cells'][k]['sha256']
    for f,d in cfg['source_sha256'].items():assert sha(root/'source_snapshot'/Path(f).name)==d
    xs=[];ys=[];ws=[]
    for k in plan['train']:
        c,t=cells[k]['cycle'],cells[k]['truth'];idx=np.unique(np.rint(np.linspace(0,len(t)-1,min(96,len(t)))).astype(int))
        xs.append(inputs(clean(cells[k]),c[idx]));ys.append(t[idx]);ws.append(np.full(len(idx),1/len(idx)))
    x,y,w=np.concatenate(xs),np.concatenate(ys),np.concatenate(ws);w*=len(w)/w.sum()
    scores=[];curves=0
    for i,q in enumerate([None]+cfg['quantiles']):
        folder=root/f'candidate_{i}';bundle=joblib.load(folder/'model.joblib')
        fn=control_predict if i==0 else predict
        fresh=None
        if i:
            assert bundle['train_ids']==plan['train'] and not bundle['local'] and bundle['quantile']==q
            imp=SimpleImputer(strategy='median',keep_empty_features=True);z=imp.fit_transform(x.astype(float))
            np.testing.assert_array_equal(imp.statistics_,bundle['imputer'].statistics_)
            model=HistGradientBoostingRegressor(**cfg['estimator_parameters'],quantile=q).fit(z,y,sample_weight=w)
            fresh=dict(bundle,imputer=imp,model=model)
        for role,ids in [('training',plan['train']),('validation',plan['validation'])]:
            stored=json.loads((folder/(role+'_metrics.json')).read_text());rows=[]
            with np.load(folder/(role+'_predictions.npz'),allow_pickle=False) as a:
                assert set(a.files)=={k+'__100__'+f for k in ids for f in ('cycle','truth','prediction')}
                for pos,k in enumerate(ids):
                    c,t,p=[a[k+'__100__'+f] for f in ('cycle','truth','prediction')]
                    np.testing.assert_array_equal(c,cells[k]['cycle']);np.testing.assert_array_equal(t,cells[k]['truth'])
                    query=clean(cells[k]);np.testing.assert_allclose(p,fn(bundle,query,c),atol=1e-12,rtol=0)
                    if fresh:np.testing.assert_allclose(p,predict(fresh,query,c),atol=1e-12,rtol=0)
                    poison=dict(query,truth=np.inf,cycle=np.inf,soh=np.inf,capacity=np.inf,cycle_life=np.inf,
                        early_cycle=np.r_[query['early_cycle'],101.],early_raw=np.vstack([query['early_raw'],np.full((1,query['early_raw'].shape[1]),np.inf)]))
                    np.testing.assert_allclose(p,fn(bundle,poison,c),atol=1e-12,rtol=0)
                    try:fn(bundle,dict(query,physical_family='INVALID_FAMILY'),c)
                    except ValueError:pass
                    else:raise AssertionError('family gate missing')
                    rr=dict(cell_id=k,**detail(c,t,p),start_error_pp=float(100*abs(p[0]-t[0])))
                    close(stored[pos],rr)
                    # Separate raw arithmetic for primary shape and MAPE measures.
                    for field,value in dict(mape_percent=100*np.mean(np.abs(p-t)/t),endpoint_pp=100*abs(p[-1]-t[-1]),
                        drop_error_pp=100*abs((p[0]-p[-1])-(t[0]-t[-1]))).items():
                        np.testing.assert_allclose(rr[field],value,atol=1e-10,rtol=0)
                    rows.append(rr);curves+=1
            if role=='validation':vr=rows
        per=np.array([np.mean([r[f] for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp')]) for r in vr])
        scores.append(dict(candidate=i,quantile=q,score=float(per.mean()+.25*np.percentile(per,90)),
            **{f:float(np.mean([r[f] for r in vr])) for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp','start_error_pp')}))
    sel=json.loads((root/'selection.json').read_text())
    for a,b in zip(sel['candidates'],scores):close(a,b)
    best=min(scores,key=lambda r:(r['score'],r['candidate']));old=scores[0]
    expand=best['candidate']>0 and best['score']<old['score'] and best['mape_percent']<=old['mape_percent']+.1 and best['endpoint_pp']<=old['endpoint_pp'] and best['drop_error_pp']<=old['drop_error_pp']
    close(sel['selected'],best);assert sel['expand']==expand and curves==176
    save_json(root/'independent_verification.json',dict(passed=True,curves=curves,fresh_refits=3,expand=expand,
        scope='Separate training sampling and estimator refits; shared feature helper; prediction equality, not full-state comparison.',
        outer_loaded=False,verifier_sha256=sha(Path(__file__))))
    print('PASS: 3 fresh refits, 176 curves, input poisoning, family gate, metrics and selection; expand='+str(expand))


if __name__=='__main__':
    with threadpool_limits(limits=2):main(Path(sys.argv[1]))
