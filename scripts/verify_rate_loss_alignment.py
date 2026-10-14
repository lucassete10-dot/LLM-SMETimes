"""Independent weight reconstruction and ET refit; no outer cells."""
import argparse,json
from pathlib import Path
import joblib,numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from threadpoolctl import threadpool_limits
from scripts.train_curve_fidelity import load_subset
from scripts.train_crossfit_drift_smoke import clean
from scripts.prepare_same_type_forecast import sha,save_json
from battery_smetimes.tail_boost import labels,inputs
from battery_smetimes.conditional_drift import prefix
from battery_smetimes.curve_fidelity import curve_metrics,selection_score

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);root=parser.parse_args().run;cfg=json.loads((root/'config.json').read_text());plan=cfg['plan'];seed=cfg['seed']
    assert not set(plan['train'])&set(plan['validation']) and not set(plan['train']+plan['validation'])&set(plan['test'])
    for f,h in cfg['source_sha256'].items():assert sha(f)==h
    source=Path(cfg['source_run']);inner=source/plan['name']/str(seed)/'inner'
    assert sha(inner/'components.joblib')==cfg['source_components_sha256'] and sha(inner/'selection.json')==cfg['source_selection_sha256']
    assert json.loads((source/'independent_verification.json').read_text())['passed']
    base=joblib.load(inner/'components.joblib')['rate'];assert base['train_ids']==plan['train'];cells=load_subset(plan['train']+plan['validation']);assert not set(cells)&set(plan['test'])
    selections=[];curves=0
    for i,params in enumerate(cfg['candidates']):
        xs=[];targets=[];weights=[]
        for key,a in zip(plan['train'],base['training_anchors']):
            c,y=labels(cells[key]);idx=np.unique(np.rint(np.linspace(0,len(c)-1,min(96,len(c)))).astype(int));d=1+(c[idx]-100)/100
            xs.append(inputs(cells[key],c[idx]));targets.append((a-y[idx])/d)
            v=np.linspace(1,3,len(idx))*np.power(d,params['power']);weights.append(v/np.sum(v))
        x=np.concatenate(xs);target=np.concatenate(targets);w=np.concatenate(weights);w*=len(w)/w.sum();imputer=SimpleImputer(strategy='median',keep_empty_features=True).fit(x)
        np.testing.assert_array_equal(imputer.statistics_,base['imputer'].statistics_)
        folder=root/f'candidate_{i}';b=joblib.load(folder/'model.joblib');assert b['train_ids']==plan['train'] and b['family']==base['family'] and b['rate_loss_parameters']==params
        with np.load(folder/'supervision.npz') as saved:
            np.testing.assert_array_equal(saved['target'],target);np.testing.assert_allclose(saved['weight'],w,atol=1e-14,rtol=0)
        fresh=ExtraTreesRegressor(n_estimators=128,min_samples_leaf=params['leaf'],max_depth=None,random_state=seed,n_jobs=2).fit(imputer.transform(x),target,sample_weight=w)
        validation=[]
        for role in ('training','validation'):
            ids=plan['train'] if role=='training' else plan['validation'];metrics=json.loads((folder/(role+'_metrics.json')).read_text())
            with np.load(folder/(role+'_predictions.npz')) as saved:
                assert set(saved.files)=={k+'__100__'+f for k in ids for f in ('cycle','truth','prediction')}
                for j,key in enumerate(ids):
                    c,y=labels(cells[key]);query=clean(cells[key]);a=float(base['anchor_model'].predict(base['anchor_imputer'].transform(prefix(query)[None]))[0]);xx=imputer.transform(inputs(query,c));p=a-(1+(c-100)/100)*fresh.predict(xx)
                    np.testing.assert_array_equal(saved[key+'__100__cycle'],c);np.testing.assert_array_equal(saved[key+'__100__truth'],y);np.testing.assert_allclose(saved[key+'__100__prediction'],p,rtol=0,atol=1e-12)
                    np.testing.assert_allclose(b['model'].predict(b['imputer'].transform(inputs(query,c))),fresh.predict(xx),rtol=0,atol=1e-12)
                    r=curve_metrics(c,y,p);assert metrics[j]['cell_id']==key
                    for f,v in r.items():np.testing.assert_allclose(v,metrics[j][f],rtol=0,atol=1e-10)
                    if role=='validation':validation.append(r)
                    curves+=1
        selections.append(dict(candidate=i,score=selection_score(validation),endpoint_pp=float(np.mean([r['endpoint_pp'] for r in validation])),drop_error_pp=float(np.mean([r['drop_error_pp'] for r in validation]))))
    selection=json.loads((root/'selection.json').read_text());chosen=min(selections,key=lambda r:(r['score'],r['candidate']))
    for f,v in chosen.items():np.testing.assert_allclose(selection['selected'][f],v,rtol=0,atol=1e-10)
    for ctrl in selection['controls'].values():
        assert sha(ctrl['file'])==ctrl['sha256'];rows=json.loads(Path(ctrl['file']).read_text());np.testing.assert_allclose(ctrl['score'],selection_score(rows),rtol=0,atol=1e-12)
        for field in ('endpoint_pp','drop_error_pp'):np.testing.assert_allclose(ctrl[field],np.mean([r[field] for r in rows]),rtol=0,atol=1e-12)
    screen=all(chosen['score']<v['score'] and chosen['endpoint_pp']<=v['endpoint_pp'] and chosen['drop_error_pp']<=v['drop_error_pp'] for v in selection['controls'].values());assert screen==selection['expand_fivefold_smoke']
    save_json(root/'independent_verification.json',dict(passed=True,curves=curves,independent_rate_refits=4,screen_passed=screen,outer_loaded=False,anchor_scope='inherited audited train-only Stage30 anchor',source_sha256=sha(__file__),goal_complete=False));print(json.dumps(dict(passed=True,curves=curves,screen=screen)),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
