"""Independent input construction, four boosting refits and all saved score checks."""
import json,sys
from pathlib import Path
import joblib,numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from scripts.verify_xjtu_segment_features import independent_inputs
from battery_smetimes.xjtu_regularized_boosting import predict
from battery_smetimes.xjtu_segment_features import predict as base_predict
from battery_smetimes.curve_fidelity import targets,condition_key
from scripts.train_curve_fidelity import load_subset
from scripts.train_crossfit_drift_smoke import clean
from scripts.train_nasa_robust_inner import detail
from scripts.verify_mit_fixed_origin_dynamics import serial_trees
from scripts.verify_nasa_robust_inner import close
from scripts.prepare_same_type_forecast import sha,save_json


def refit(cells,ids,b,loss,seed):
    assert b['train_ids']==ids and b['loss']==loss and b['seed']==seed
    xs=[];ys=[];weights=[]
    for k in ids:
        c,y=targets(cells[k]);ix=np.unique(np.rint(np.linspace(0,len(c)-1,min(96,len(c)))).astype(int))
        xs.append(independent_inputs(cells[k],c[ix],dict(block='VIT',context='segments')));ys.append(y[ix]);w=np.linspace(1,3,len(ix));weights.append(w/w.sum())
    x,y,w=np.concatenate(xs),np.concatenate(ys),np.concatenate(weights);w*=len(w)/w.sum()
    imp=SimpleImputer(strategy='median',keep_empty_features=True);z=imp.fit_transform(x)
    np.testing.assert_allclose(imp.statistics_,b['imputer'].statistics_,rtol=0,atol=1e-12)
    parameters=dict(loss=loss,max_iter=200,learning_rate=.05,max_leaf_nodes=15,max_depth=None,min_samples_leaf=20,l2_regularization=10,max_bins=64,early_stopping=False,random_state=seed)
    for k,v in parameters.items():assert b['model'].get_params()[k]==v
    model=HistGradientBoostingRegressor(**parameters).fit(z,y,sample_weight=w)
    assert model.n_iter_==b['model'].n_iter_==200
    np.testing.assert_array_equal(model._baseline_prediction,b['model']._baseline_prediction)
    assert len(model._bin_mapper.bin_thresholds_)==len(b['model']._bin_mapper.bin_thresholds_)
    for x,y in zip(model._bin_mapper.bin_thresholds_,b['model']._bin_mapper.bin_thresholds_):np.testing.assert_array_equal(x,y)
    assert len(model._predictors)==len(b['model']._predictors)==200
    for x,y in zip(model._predictors,b['model']._predictors):
        assert len(x)==len(y)
        for t,u in zip(x,y):np.testing.assert_array_equal(t.nodes,u.nodes)
    return dict(b,model=model,imputer=imp)


def replay(b,q,c):
    key=condition_key(q)
    if key in b['local']:return replay(b['local'][key],q,c)
    return b['model'].predict(b['imputer'].transform(independent_inputs(q,c,dict(block='VIT',context='segments'))))


def main(root):
    cfg=json.loads((root/'config.json').read_text());plan=cfg['plan'];cells=load_subset(plan['train']+plan['validation'])
    assert (len(plan['train']),len(plan['validation']),len(plan['test']))==(31,13,11) and not set(cells)&set(plan['test'])
    for f,d in cfg['input_sha256'].items():assert sha(Path(f))==d
    for f,d in cfg['source_sha256'].items():assert sha(root/'source_snapshot'/Path(f).name)==d
    for k,d in cfg['prepared_sha256'].items():assert cells[k]['prepared_sha256']==d
    base=joblib.load(cfg['model_file']);serial_trees(base);scores=[];curves=0;components=0
    for i,loss in enumerate([None]+cfg['losses']):
        out=root/f'candidate_{i}';b=joblib.load(out/'model.joblib');assert b['train_ids']==plan['train'] and b['family']==base['family']
        if loss is None:serial_trees(b)
        else:
            b=refit(cells,plan['train'],b,loss,cfg['seed']);components+=1;groups={}
            for k in plan['train']:groups.setdefault(condition_key(cells[k]),[]).append(k)
            required={k:v for k,v in groups.items() if len(v)>=6};assert set(b['local'])==set(required)
            b['local']={k:refit(cells,ids,b['local'][k],loss,cfg['seed']) for k,ids in required.items()};components+=len(required)
        for role,ids in [('training',plan['train']),('validation',plan['validation'])]:
            saved=json.loads((out/(role+'_metrics.json')).read_text());rows=[];assert len(saved)==len(ids)
            with np.load(out/(role+'_predictions.npz'),allow_pickle=False) as a:
                assert set(a.files)=={k+'__100__'+f for k in ids for f in ('cycle','truth','prediction')}
                for pos,k in enumerate(ids):
                    c,y=targets(cells[k]);q=clean(cells[k]);p=replay(b,q,c) if loss is not None else base_predict(base,q,c)
                    np.testing.assert_array_equal(a[k+'__100__cycle'],c);np.testing.assert_array_equal(a[k+'__100__truth'],y);np.testing.assert_allclose(a[k+'__100__prediction'],p,rtol=0,atol=1e-12)
                    fn=predict if loss is not None else base_predict;np.testing.assert_allclose(fn(b,q,c),p,rtol=0,atol=1e-12)
                    raw=q['raw'].copy();raw[:,[22,45]]=np.inf
                    poison=dict(q,soh=np.inf,label_cycle=np.inf,reference_capacity_ah=np.inf,cycle_life=np.inf,cycle=np.r_[q['cycle'],101.],raw=np.vstack([raw,np.full((1,raw.shape[1]),np.inf)]))
                    np.testing.assert_allclose(fn(b,poison,c),p,rtol=0,atol=1e-12)
                    try:fn(b,dict(q,physical_family='INVALID'),c)
                    except ValueError:pass
                    else:raise AssertionError('family')
                    for bad in ([],[100.],[np.inf],[[101.]]):
                        try:fn(b,q,bad)
                        except ValueError:pass
                        else:raise AssertionError('coordinates')
                    r=dict(cell_id=k,**detail(c,y,p),start_error_pp=float(100*abs(p[0]-y[0])));close(saved[pos],r);rows.append(r);curves+=1
            if role=='validation':vr=rows
        v=np.array([np.mean([r[f] for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp')]) for r in vr])
        scores.append(dict(candidate=i,loss=loss,score=float(v.mean()+.25*np.percentile(v,90)),**{f:float(np.mean([r[f] for r in vr])) for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp','start_error_pp')}))
    sel=json.loads((root/'selection.json').read_text());assert len(sel['candidates'])==3
    for x,y in zip(sel['candidates'],scores):close(x,y)
    best=min(scores,key=lambda r:(r['score'],r['candidate']));old=scores[0];expand=best['candidate']>0 and best['score']<old['score'] and best['mape_percent']<=old['mape_percent']+.1 and best['endpoint_pp']<=old['endpoint_pp'] and best['drop_error_pp']<=old['drop_error_pp']
    close(sel['selected'],best);assert sel['expand']==expand and curves==132 and components==4
    save_json(root/'independent_verification.json',dict(passed=True,independent_component_refits=components,exact_boosting_nodes=True,curves=132,expand=expand,outer_loaded=False,
        scope='Prior independent feature construction, fresh HGB/median statistics/exact bins and nodes, separate score; shared metrics helpers.',verifier_sha256=sha(Path(__file__))))
    print('PASS:4freshcomponents/exactbinsnodes,132curves/metrics/selection/poison/family/SHA;expand='+str(expand))


if __name__=='__main__':
    with threadpool_limits(limits=1):main(Path(sys.argv[1]))
