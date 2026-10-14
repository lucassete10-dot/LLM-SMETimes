"""Independent HGB supervision/refit and reference algorithm replay on inner CV."""
import argparse,json
from pathlib import Path
import joblib,numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits
from battery_smetimes.tail_boost import labels,inputs
from scripts.train_tail_boost import load_cells
from scripts.train_crossfit_drift_smoke import clean
from scripts.verify_neighbor_curve_nca_formal import reference_replay,score
from scripts.audit_rate_blend_metrics import metrics
from scripts.prepare_same_type_forecast import save_json,sha

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);root=parser.parse_args().run;cfg=json.loads((root/'config.json').read_text());plan=cfg['parent_plan'];ids=plan['train'];seed=cfg['seed']
    for f,h in cfg['source_sha256'].items():assert sha(f)==h and sha(root/'source_snapshot'/Path(f).name)==h
    assert sha(Path(cfg['source_route'])/'config.json')==cfg['source_config_sha256']
    cells=load_cells(dict(plan,validation=[]));assert set(cells)==set(ids) and not set(cells)&set(plan['validation']+plan['test'])
    expected=[]
    for i,(tr,va) in enumerate(KFold(n_splits=3,shuffle=True,random_state=20261005).split(ids)):expected.append(dict(name='inner'+str(i+1),train=[ids[j] for j in tr],validation=[ids[j] for j in va]))
    assert cfg['splits']==expected;all_validation=[k for p in expected for k in p['validation']];assert len(all_validation)==len(set(all_validation))==39
    pooled=[[] for _ in cfg['candidates']];fold_scores=[];count=0
    for split in expected:
        tr,va=split['train'],split['validation'];folder=root/split['name'];parts=joblib.load(folder/'components.joblib');assert parts['primary']['train_ids']==tr;reference=reference_replay(parts['analog'],cells,tr);xs=[];ys=[];ws=[]
        for key in tr:
            c,y=labels(cells[key]);idx=np.unique(np.rint(np.linspace(0,len(c)-1,min(96,len(c)))).astype(int));xs.append(inputs(clean(cells[key]),c[idx]));ys.append(y[idx]);ws.append(np.full(len(idx),1/len(idx)))
        x=np.concatenate(xs);y=np.concatenate(ys);w=np.concatenate(ws);w*=len(w)/w.sum();imputer=SimpleImputer(strategy='median',keep_empty_features=True).fit(x);inner=parts['primary']['inner'];np.testing.assert_array_equal(imputer.statistics_,inner['imputer'].statistics_)
        model=HistGradientBoostingRegressor(max_iter=120,learning_rate=.05,max_leaf_nodes=15,min_samples_leaf=20,l2_regularization=1,max_bins=64,early_stopping=False,random_state=seed).fit(imputer.transform(x),y,sample_weight=w)
        validation_scores=[]
        for role in ('training','validation'):
            members=tr if role=='training' else va;cached={}
            with np.load(folder/(role+'_components.npz')) as components:
                assert set(components.files)=={k+'__100__'+f for k in members for f in ('cycle','truth','primary','neighbor1','neighbor3')}
                for key in members:
                    c,y=labels(cells[key]);query=clean(cells[key]);base=model.predict(imputer.transform(inputs(query,c)));np.testing.assert_allclose(base,inner['model'].predict(inner['imputer'].transform(inputs(query,c))),rtol=0,atol=1e-12)
                    near={k:reference(query,c,k,key if role=='training' else None) for k in (1,3)}
                    for f,v in [('cycle',c),('truth',y),('primary',base),('neighbor1',near[1]),('neighbor3',near[3])]:np.testing.assert_allclose(v,components[key+'__100__'+f],rtol=0,atol=1e-12)
                    cached[key]=(c,y,base,near)
            for i,params in enumerate(cfg['candidates']):
                dest=folder/f'candidate_{i}';stored=json.loads((dest/(role+'_metrics.json')).read_text());assert [r['cell_id'] for r in stored]==members;rows=[]
                with np.load(dest/(role+'_predictions.npz')) as a:
                    assert set(a.files)=={k+'__100__'+f for k in members for f in ('cycle','truth','prediction')}
                    for j,key in enumerate(members):
                        c,y,p,q=cached[key];prediction=(1-params['weight'])*p+params['weight']*q[params['k']]
                        for f,v in [('cycle',c),('truth',y),('prediction',prediction)]:np.testing.assert_allclose(v,a[key+'__100__'+f],rtol=0,atol=1e-12)
                        r=dict(cell_id=key,**metrics(c,y,prediction));rows.append(r);count+=1
                        for f,v in r.items():
                            if f!='cell_id':np.testing.assert_allclose(v,stored[j][f],rtol=0,atol=1e-10)
                if role=='validation':pooled[i].extend(rows);validation_scores.append(score(rows))
        fold_scores.append(validation_scores);print('VERIFIED '+split['name'],flush=True)
    selected=int(np.argmin([score(r) for r in pooled]));record=json.loads((root/'selection.json').read_text());assert record['selected']['candidate']==selected
    for i,rows in enumerate(pooled):
        saved=json.loads((root/f'pooled_candidate_{i}_metrics.json').read_text());assert [r['cell_id'] for r in saved]==[r['cell_id'] for r in rows]
        for r,t in zip(rows,saved):
            for f,v in r.items():
                if f!='cell_id':np.testing.assert_allclose(v,t[f],rtol=0,atol=1e-10)
        stored=record['candidates'][i];np.testing.assert_allclose(stored['score'],score(rows),rtol=0,atol=1e-10)
        for field in ('mape_percent','endpoint_pp','drop_error_pp'):np.testing.assert_allclose(stored[field],np.mean([r[field] for r in rows]),rtol=0,atol=1e-10)
    wins=sum(r[selected]<r[0] for r in fold_scores);assert wins==record['subfold_score_wins'];a,b=record['selected'],record['control'];screen=a['score']<b['score'] and a['endpoint_pp']<=b['endpoint_pp'] and a['drop_error_pp']<=b['drop_error_pp'] and wins>=2;assert screen==record['expand_all_outer_training_cv']
    save_json(root/'independent_verification.json',dict(passed=True,curves=count,independent_hgb_refits=3,heldout_cells=39,screen_passed=screen,original_validation_loaded=False,outer_loaded=False,source_sha256=sha(__file__),reference_verifier_sha256=sha('scripts/verify_neighbor_curve_nca_formal.py'),goal_complete=False));print(json.dumps(dict(passed=True,curves=count,screen=screen)),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
