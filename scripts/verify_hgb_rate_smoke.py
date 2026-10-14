"""Independent supervision reconstruction, estimator refit and full curve replay."""
import argparse,json
from pathlib import Path
import joblib,numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from threadpoolctl import threadpool_limits
from scripts.train_curve_fidelity import load_subset
from scripts.prepare_same_type_forecast import sha,save_json
from scripts.train_crossfit_drift_smoke import clean
from battery_smetimes.tail_boost import labels,inputs
from battery_smetimes.fade_rate import predict
from battery_smetimes.curve_fidelity import curve_metrics,selection_score

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True,type=Path);root=parser.parse_args().run
    cfg=json.loads((root/'config.json').read_text());plan=cfg['plan'];seed=cfg['seed']
    assert not set(plan['train'])&set(plan['validation']) and not set(plan['train']+plan['validation'])&set(plan['test'])
    for f,h in cfg['source_sha256'].items():assert sha(f)==h
    source=Path(cfg['source_run']);inner=source/plan['name']/str(seed)/'inner'
    assert sha(inner/'components.joblib')==cfg['source_components_sha256'] and sha(inner/'selection.json')==cfg['source_selection_sha256']
    proof=json.loads((source/'independent_verification.json').read_text());assert proof['passed']
    base=joblib.load(inner/'components.joblib')['rate'];assert base['train_ids']==plan['train']
    cells=load_subset(plan['train']+plan['validation']);assert not set(cells)&set(plan['test'])
    xs=[];ys=[];ws=[]
    for key,a in zip(plan['train'],base['training_anchors']):
        c,y=labels(cells[key]);indices=np.unique(np.rint(np.linspace(0,len(c)-1,min(len(c),96))).astype(int));coords=c[indices]
        xs.append(inputs(cells[key],coords));ys.append((a-y[indices])/(1+(coords-100)/100))
        weights=np.linspace(1,3,len(indices));ws.append(weights/weights.sum())
    x=np.concatenate(xs);y=np.concatenate(ys);w=np.concatenate(ws);w*=len(w)/w.sum()
    imputer=SimpleImputer(strategy='median',keep_empty_features=True).fit(x)
    np.testing.assert_allclose(imputer.statistics_,base['imputer'].statistics_,rtol=0,atol=0)
    rows_all=[];curves=0
    for i,params in enumerate(cfg['candidates']):
        folder=root/f'candidate_{i}';b=joblib.load(folder/'model.joblib');assert b['train_ids']==plan['train'] and b['family']==base['family']
        fresh=HistGradientBoostingRegressor(**params,learning_rate=.05,max_iter=120,min_samples_leaf=20,max_bins=64,early_stopping=False,random_state=seed).fit(imputer.transform(x),y,sample_weight=w)
        rows=[]
        for role in ('training','validation'):
            ids=plan['train'] if role=='training' else plan['validation'];saved=json.loads((folder/(role+'_metrics.json')).read_text())
            with np.load(folder/(role+'_predictions.npz')) as arrays:
                assert set(arrays.files)=={k+'__100__'+f for k in ids for f in ('cycle','truth','prediction')}
                for j,key in enumerate(ids):
                    c,t=labels(cells[key]);cell=clean(cells[key]);prefix=key+'__100__';np.testing.assert_array_equal(arrays[prefix+'cycle'],c);np.testing.assert_array_equal(arrays[prefix+'truth'],t)
                    rate=fresh.predict(imputer.transform(inputs(cell,c)))
                    # Anchor is inherited verbatim from audited, train-only source.
                    from battery_smetimes.conditional_drift import prefix as early
                    a=float(base['anchor_model'].predict(base['anchor_imputer'].transform(early(cell)[None]))[0]);p=a-(1+(c-100)/100)*rate
                    np.testing.assert_allclose(p,arrays[prefix+'prediction'],rtol=0,atol=1e-12);np.testing.assert_allclose(p,predict(b,cell,c),rtol=0,atol=1e-12)
                    metrics=curve_metrics(c,t,p)
                    for field,value in metrics.items():np.testing.assert_allclose(value,saved[j][field],rtol=0,atol=1e-10)
                    assert saved[j]['cell_id']==key
                    if role=='validation':rows.append(metrics)
                    curves+=1
        rows_all.append(dict(candidate=i,score=selection_score(rows),endpoint_pp=float(np.mean([r['endpoint_pp'] for r in rows])),drop_error_pp=float(np.mean([r['drop_error_pp'] for r in rows]))))
    selection=json.loads((root/'selection.json').read_text());chosen=min(rows_all,key=lambda r:(r['score'],r['candidate']))
    for f,v in chosen.items():np.testing.assert_allclose(v,selection['selected'][f],atol=1e-10,rtol=0)
    for control in selection['controls'].values():
        assert sha(control['file'])==control['sha256'];rows=json.loads(Path(control['file']).read_text());np.testing.assert_allclose(control['score'],selection_score(rows),rtol=0,atol=1e-12)
    screen=all(chosen['score']<v['score'] and chosen['endpoint_pp']<=v['endpoint_pp'] and chosen['drop_error_pp']<=v['drop_error_pp'] for v in selection['controls'].values());assert screen==selection['expand_fivefold_smoke']
    save_json(root/'independent_verification.json',dict(passed=True,curves=curves,independent_rate_refits=4,anchor_scope='unchanged audited Stage30 train-only anchor; not re-fitted here',outer_loaded=False,screen_passed=screen,source_sha256=sha(__file__),goal_complete=False));print(json.dumps(dict(passed=True,curves=curves,screen=screen)),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
