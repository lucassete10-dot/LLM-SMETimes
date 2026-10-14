"""TRAIN-only OOF feature/negative-control comparison, frozen candidate budget."""
import json,shutil,logging,time
from pathlib import Path
from datetime import datetime,timezone
import joblib,numpy as np
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesRegressor
from threadpoolctl import threadpool_limits
from battery_smetimes.tail_boost import inputs
from battery_smetimes.nca_segment_features import prefix as segments
from battery_smetimes.curve_fidelity import shape_parameters
from scripts.train_tail_boost import load_cells,NCA
from scripts.train_crossfit_drift_smoke import clean
from scripts.train_nasa_robust_inner import detail
from scripts.prepare_same_type_forecast import sha,save_json

PARENT=Path('experiments/tail_boost/20261005T034431_681399Z')
ARMS=['conditions','prefix','segments','prefix_shuffled']
FIELDS=['mape_percent','mae_soh','rmse_soh','r2','late_mape','endpoint_pp','drop_error_pp']


def features(cell,arm):
    if arm=='conditions':return np.asarray(cell['conditions'],dtype=float)
    if arm=='segments':return segments(cell,dict(block='VI',context='segments'))
    if arm in ('prefix','prefix_shuffled'):return inputs(cell,[101.])[0]
    raise ValueError('unknown arm')


def predict(b,cell,c):
    if cell['physical_family']!=b['family']:raise ValueError('different family')
    c=np.asarray(c,dtype=float)
    if c.ndim!=1 or not len(c) or not np.isfinite(c).all() or np.any(c<=100):raise ValueError('bad future')
    a=b['model'].predict(b['imputer'].transform(features(cell,b['arm'])[None]))[0]*b['scale']
    h=(c-100)/100;return a[0]-a[1]*h-a[2]*h*h


def permutation(cells,ids,seed,fold):
    rng=np.random.default_rng(seed+1000*fold);mapping={k:k for k in ids};groups={}
    for k in ids:groups.setdefault(tuple(cells[k]['conditions']),[]).append(k)
    for members in groups.values():
        for k,v in zip(members,rng.permutation(members)):mapping[k]=str(v)
    return mapping


def main():
    pc=json.loads((PARENT/'config.json').read_text());plan=next(p for p in pc['plans'] if p['name']=='NCA_fold1')
    cells=load_cells(dict(plan,validation=[],test=[]));ids=list(plan['train'])
    assert set(cells)==set(ids) and not set(cells)&set(plan['validation']+plan['test'])
    root=Path('experiments/nca_feature_discrimination')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');root.mkdir(parents=True);print('RUN='+str(root.resolve()),flush=True)
    snap=root/'source_snapshot';snap.mkdir();files=[Path(__file__),Path('battery_smetimes/nca_segment_features.py'),Path('battery_smetimes/tail_boost.py'),Path('battery_smetimes/curve_fidelity.py'),Path('docs/soh_nca_feature_discrimination_20261007.md')]
    for f in files:shutil.copy2(f,snap/f.name)
    coeff={k:shape_parameters(cells[k]['cycle'],cells[k]['truth'],2) for k in ids}
    save_json(root/'training_coefficient_targets.json',{k:v.tolist() for k,v in coeff.items()})
    splits=[dict(fold=i+1,train=[ids[j] for j in tr],heldout=[ids[j] for j in va]) for i,(tr,va) in enumerate(KFold(5,shuffle=True,random_state=20261005).split(ids))]
    seeds=list(range(20261005,20261010));index=json.loads((NCA/'index.json').read_text())
    save_json(root/'config.json',dict(parent=str(PARENT.resolve()),parent_config_sha256=sha(PARENT/'config.json'),plan=plan,splits=splits,seeds=seeds,arms=ARMS,
        source_sha256={str(f):sha(f) for f in files},nca_index_sha256=sha(NCA/'index.json'),prepared_sha256={k:index['cells'][k]['sha256'] for k in ids},
        targets_sha256=sha(root/'training_coefficient_targets.json'),original_validation_loaded=False,outer_loaded=False,
        estimator=dict(n_estimators=128,min_samples_leaf=1,n_jobs=2),scope='Training-side OOF feature diagnosis, no model adoption'))
    logging.basicConfig(level=logging.INFO,handlers=[logging.FileHandler(root/'training.log',encoding='utf-8'),logging.StreamHandler()]);start=time.perf_counter();records=[];curves=0;fits=0
    for seed in seeds:
        for split in splits:
            tr=split['train'];va=split['heldout'];mapping=permutation(cells,tr,seed,split['fold'])
            for arm in ARMS:
                out=root/str(seed)/('fold'+str(split['fold']))/arm;out.mkdir(parents=True)
                used=mapping if arm=='prefix_shuffled' else {k:k for k in tr}
                x=np.stack([features(clean(cells[k]),arm) for k in tr]);y=np.stack([coeff[used[k]] for k in tr]);scale=np.maximum(y.std(axis=0),.001)
                imp=SimpleImputer(strategy='median',keep_empty_features=True);model=ExtraTreesRegressor(n_estimators=128,min_samples_leaf=1,random_state=seed,n_jobs=2).fit(imp.fit_transform(x),y/scale);fits+=1
                b=dict(model=model,imputer=imp,scale=scale,train_ids=tr,supervision_mapping=used,family=cells[tr[0]]['physical_family'],arm=arm,seed=seed);joblib.dump(b,out/'model.joblib')
                save_json(out/'split.json',dict(**split,arm=arm,seed=seed,supervision_mapping=used))
                for role,members in [('training',tr),('validation',va)]:
                    rows=[];arrays={}
                    for k in members:
                        c,y=cells[k]['cycle'],cells[k]['truth'];p=predict(b,clean(cells[k]),c)
                        r=dict(cell_id=k,arm=arm,seed=seed,diagnostic_fold=split['fold'],role=role,**detail(c,y,p));rows.append(r);curves+=1
                        arrays.update({k+'__100__'+f:v for f,v in [('cycle',c),('truth',y),('prediction',p)]})
                    save_json(out/(role+'_metrics.json'),rows);np.savez_compressed(out/(role+'_predictions.npz'),**arrays)
                    if role=='validation':records.extend(rows)
            logging.info('SAVED seed=%s fold=%s',seed,split['fold'])
    summary={};seedmeans={}
    for arm in ARMS:
        seedmeans[arm]=[{f:float(np.mean([r[f] for r in records if r['arm']==arm and r['seed']==s])) for f in FIELDS} for s in seeds]
        summary[arm]={f:dict(mean=float(np.mean([r[f] for r in seedmeans[arm]])),seed_sd=float(np.std([r[f] for r in seedmeans[arm]],ddof=1))) for f in FIELDS}
    save_json(root/'oof_rows.json',records);save_json(root/'seed_means.json',seedmeans)
    save_json(root/'summary.json',dict(arms=summary,unique_train_cells=39,oof_rows=len(records),fits=fits,curves=curves,outer_loaded=False,original_validation_loaded=False,
        adopted=False,goal_complete=False,interpretation='Training-side OOF; not original outer forecast accuracy or causal proof.'))
    save_json(root/'progress.json',dict(complete=True,fits=fits,curves=curves,seconds=time.perf_counter()-start,outer_loaded=False))
    print(json.dumps(dict(arms=summary,fits=fits,curves=curves)),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
