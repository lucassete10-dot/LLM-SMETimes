"""Refit diagnostic heads and audit all curves, splits and train-only shuffling."""
import json,sys
from pathlib import Path
import joblib,numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from threadpoolctl import threadpool_limits
from battery_smetimes.curve_fidelity import shape_parameters
from scripts.train_nca_feature_discrimination import features,predict,FIELDS
from scripts.train_tail_boost import load_cells,NCA
from scripts.train_crossfit_drift_smoke import clean
from scripts.train_nasa_robust_inner import detail
from scripts.prepare_same_type_forecast import sha,save_json
from scripts.verify_nasa_robust_inner import close


def main(root):
    cfg=json.loads((root/'config.json').read_text());plan=cfg['plan'];cells=load_cells(dict(plan,validation=[],test=[]))
    assert set(cells)==set(plan['train']) and not set(cells)&set(plan['validation']+plan['test'])
    assert sha(Path(cfg['parent'])/'config.json')==cfg['parent_config_sha256'] and sha(NCA/'index.json')==cfg['nca_index_sha256']
    for f,d in cfg['source_sha256'].items():assert sha(root/'source_snapshot'/Path(f).name)==d
    targetfile=root/'training_coefficient_targets.json';assert sha(targetfile)==cfg['targets_sha256']
    coefficients=json.loads(targetfile.read_text())
    for k in plan['train']:np.testing.assert_array_equal(coefficients[k],shape_parameters(cells[k]['cycle'],cells[k]['truth'],2))
    assert sorted(k for s in cfg['splits'] for k in s['heldout'])==sorted(plan['train'])
    records=[];curves=0;heads=0
    for seed in cfg['seeds']:
        for split in cfg['splits']:
            tr,va=split['train'],split['heldout'];assert not set(tr)&set(va) and set(tr+va)==set(plan['train'])
            groups={}
            for k in tr:groups.setdefault(tuple(cells[k]['conditions']),[]).append(k)
            rng=np.random.default_rng(seed+1000*split['fold']);shuffled={}
            for members in groups.values():
                dest=rng.permutation(np.array(members))
                for k,v in zip(members,dest):shuffled[k]=str(v)
            for arm in cfg['arms']:
                out=root/str(seed)/('fold'+str(split['fold']))/arm;b=joblib.load(out/'model.joblib');savedsplit=json.loads((out/'split.json').read_text())
                expected=shuffled if arm=='prefix_shuffled' else {k:k for k in tr}
                assert b['train_ids']==tr and b['supervision_mapping']==expected==savedsplit['supervision_mapping'] and b['arm']==arm and b['seed']==seed
                x=np.array([features(clean(cells[k]),arm) for k in tr]);y=np.array([coefficients[expected[k]] for k in tr]);scale=np.maximum(np.std(y,axis=0),.001)
                imputer=SimpleImputer(strategy='median',keep_empty_features=True);z=imputer.fit_transform(x)
                np.testing.assert_array_equal(scale,b['scale']);np.testing.assert_array_equal(imputer.statistics_,b['imputer'].statistics_)
                model=ExtraTreesRegressor(n_estimators=128,min_samples_leaf=1,random_state=seed,n_jobs=1).fit(z,y/scale);heads+=1
                # Serial inference avoids noisy worker-config warnings; prediction function is unchanged.
                b['model'].n_jobs=1;fresh=dict(b,model=model,imputer=imputer)
                for role,ids in [('training',tr),('validation',va)]:
                    stored=json.loads((out/(role+'_metrics.json')).read_text());assert len(stored)==len(ids)
                    with np.load(out/(role+'_predictions.npz'),allow_pickle=False) as a:
                        assert set(a.files)=={k+'__100__'+f for k in ids for f in ('cycle','truth','prediction')}
                        for pos,k in enumerate(ids):
                            c,t,p=[a[k+'__100__'+f] for f in ('cycle','truth','prediction')];q=clean(cells[k])
                            np.testing.assert_array_equal(c,cells[k]['cycle']);np.testing.assert_array_equal(t,cells[k]['truth'])
                            np.testing.assert_allclose(p,predict(b,q,c),rtol=0,atol=1e-12);np.testing.assert_allclose(p,predict(fresh,q,c),rtol=0,atol=1e-12)
                            poison=dict(q,truth=np.inf,cycle=np.inf,capacity=np.inf,cycle_life=np.inf,
                                early_cycle=np.r_[q['early_cycle'],101],early_raw=np.vstack([q['early_raw'],np.full((1,32),np.inf)]))
                            np.testing.assert_allclose(p,predict(b,poison,c),rtol=0,atol=1e-12)
                            try:predict(b,dict(q,physical_family='INVALID'),c)
                            except ValueError:pass
                            else:raise AssertionError('family gate')
                            h=(c-100)/100;coef=b['model'].predict(imputer.transform(features(q,arm)[None]))[0]*scale
                            np.testing.assert_allclose(p,coef[0]-coef[1]*h-coef[2]*h*h,rtol=0,atol=1e-12)
                            r=dict(cell_id=k,arm=arm,seed=seed,diagnostic_fold=split['fold'],role=role,**detail(c,t,p));close(stored[pos],r)
                            np.testing.assert_allclose(r['mape_percent'],100*np.mean(abs(p-t)/t),rtol=0,atol=1e-10)
                            curves+=1
                            if role=='validation':records.append(r)
            print(f'VERIFIED seed={seed} fold={split["fold"]}',flush=True)
    stored=json.loads((root/'oof_rows.json').read_text());assert len(stored)==len(records)==780
    for a,b in zip(stored,records):close(a,b)
    result=json.loads((root/'summary.json').read_text());seedmeans=json.loads((root/'seed_means.json').read_text())
    for arm in cfg['arms']:
        means=[]
        for seed in cfg['seeds']:
            rr=[r for r in records if r['seed']==seed and r['arm']==arm];assert len(rr)==39 and len({r['cell_id'] for r in rr})==39
            means.append({f:float(np.mean([r[f] for r in rr])) for f in FIELDS})
        for a,b in zip(seedmeans[arm],means):close(a,b)
        for f in FIELDS:close(result['arms'][arm][f],dict(mean=float(np.mean([r[f] for r in means])),seed_sd=float(np.std([r[f] for r in means],ddof=1))))
    assert heads==100 and curves==3900 and result['fits']==heads and result['curves']==curves and not result['adopted']
    save_json(root/'independent_verification.json',dict(passed=True,fresh_heads=heads,curves=curves,oof_records=780,unique_train_cells=39,
        original_validation_loaded=False,outer_loaded=False,scope='Separate head refits and permutation/split audit; shared feature and coefficient solvers. Prediction equality, not full-state comparison.',
        goal_complete=False,verifier_sha256=sha(Path(__file__))))
    print('PASS:100 fresh heads/3900curves/780OOF rows; no original validation or outer reads.')


if __name__=='__main__':
    with threadpool_limits(limits=2):main(Path(sys.argv[1]))
