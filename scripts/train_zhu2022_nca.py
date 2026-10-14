"""Fixed whole-cell nested future prediction on new raw-V/I NCA cohort."""
import argparse
from datetime import datetime,timezone
import json
import logging
from pathlib import Path
import shutil
import traceback
import time
import joblib
import numpy as np
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from battery_smetimes.research_baselines import regression_metrics
from scripts.prepare_same_type_forecast import save_json,sha

PREP=Path('data/interim/zhu2022_nca/20261004T161215_905258Z')
CANDIDATES={'ridge':[dict(alpha=a) for a in (1.,10.,100.)],
    'extra_trees':[dict(leaf=n,trees=64,depth=8) for n in (3,6,12)]}

def matrix(cell,future):
    future=np.asarray(future,dtype=float)
    if not np.isfinite(future).all() or np.any(future<=100):raise ValueError('invalid requested future cycles')
    h=(future-100)/100.
    p=np.broadcast_to(cell['prefix'],(len(h),len(cell['prefix'])))
    return np.c_[p,h,h*h,np.log1p(h),p*h[:,None]]

def fit(cells,ids,model_name,parameters,seed):
    family=cells[ids[0]]['physical_family'];assert all(cells[k]['physical_family']==family for k in ids)
    started=time.perf_counter();xs=[];ys=[];ws=[]
    for key in ids:
        c=cells[key];take=np.unique(np.linspace(0,len(c['truth'])-1,min(96,len(c['truth']))).round().astype(int))
        xs.append(matrix(c,c['cycle'][take]));ys.append(c['truth'][take]);ws.append(np.full(len(take),1/len(take)))
    x,y,w=np.concatenate(xs),np.concatenate(ys),np.concatenate(ws);w*=len(w)/w.sum()
    imputer=SimpleImputer(strategy='median',keep_empty_features=True);x=imputer.fit_transform(x)
    scaler=None
    if model_name=='ridge':
        offset=x[0].copy();scaler=StandardScaler().fit(x-offset,sample_weight=w);scaler.mean_+=offset;x=scaler.transform(x)
        estimator=Ridge(alpha=parameters['alpha'],solver='svd')
    else:estimator=ExtraTreesRegressor(n_estimators=parameters['trees'],max_depth=parameters['depth'],
        min_samples_leaf=parameters['leaf'],random_state=seed,n_jobs=4)
    estimator.fit(x,y,sample_weight=w)
    return dict(model=estimator,model_name=model_name,imputer=imputer,scaler=scaler,parameters=parameters,
        train_ids=list(ids),family=family,seed=seed,training_pairs=len(y),fit_seconds=time.perf_counter()-started)

def predict(bundle,cell,future):
    if cell['physical_family']!=bundle['family']:raise ValueError('different battery physical family')
    x=bundle['imputer'].transform(matrix(cell,future))
    if bundle['scaler'] is not None:x=bundle['scaler'].transform(x)
    p=bundle['model'].predict(x)
    if not np.isfinite(p).all():raise ValueError('nonfinite forecast')
    return p

def evaluate(bundle,cells,ids):
    rows=[];arrays={}
    for key in ids:
        cell=cells[key];x,y=cell['cycle'],cell['truth'];started=time.perf_counter();p=predict(bundle,cell,x)
        milliseconds=(time.perf_counter()-started)*1000
        clean={k:v for k,v in cell.items() if k not in ('truth','cycle')}
        np.testing.assert_array_equal(matrix(cell,x),matrix(clean,x))
        # Parallel tree aggregation may differ by1ulp; inputs must remain exact.
        np.testing.assert_allclose(p,predict(bundle,clean,x),rtol=0,atol=1e-12)
        last=x>x[-1]-100;late=x>=(x[0]+x[-1])/2
        rows.append(dict(cell_id=key,dataset='Zhu2022_NCA',origin=100,future_targets=len(y),
            metrics=regression_metrics(y,p),late_half=regression_metrics(y[late],p[late]),
            last100=regression_metrics(y[last],p[last]),last100_observed_targets=int(last.sum()),
            last100_observed_span=float(np.ptp(x[last])),endpoint_abs_soh_pp=float(abs(y[-1]-p[-1])*100),
            whole_curve_inference_ms=milliseconds))
        arrays.update({key+'__100__'+f:a for f,a in [('cycle',x),('truth',y),('prediction',p)]})
    return rows,arrays

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    out=Path('experiments/zhu2022_nca_forecast')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');out.mkdir(parents=True)
    shutil.copy2(__file__,out/Path(__file__).name);shutil.copy2('docs/zhu2022_nca_training_plan.md',out/'plan.md')
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'training.log',encoding='utf-8'),logging.StreamHandler()])
    try:
        audit=json.loads((PREP/'audit.json').read_text());assert audit['complete']
        prepared_config=json.loads((PREP/'config.json').read_text())
        if not prepared_config['label_rules'].startswith('full-charge4.18V'):
            raise ValueError('obsolete peak-Q-only labels; complete-protocol audit required')
        cells=joblib.load(PREP/'prepared_cells.joblib');ids=sorted(cells);assert len(ids)==audit['eligible_cells'];plans=[]
        for fold,(dev,test) in enumerate(KFold(5,shuffle=True,random_state=20261004).split(ids),1):
            dev=np.random.RandomState(20261004+fold).permutation(dev);nval=max(1,int(np.ceil(len(dev)*.1)))
            plans.append(dict(name=f'NCA_fold{fold}',train=[ids[i] for i in dev[nval:]],validation=[ids[i] for i in dev[:nval]],test=[ids[i] for i in test]))
        seeds=list(range(20261004,20261009))
        save_json(out/'config.json',dict(plans=plans,seeds=seeds,candidates=CANDIDATES,smoke=args.smoke,
            source_sha256=sha(__file__),prepared_sha256=sha(PREP/'prepared_cells.joblib'),audit_sha256=sha(PREP/'audit.json'),
            prepared_directory=str(PREP.resolve()),selection='inner validation cell-macro MAPE only',origin=100,
            deterministic_models=['ridge'],curves_status='pending export'))
        for model_name,candidates in CANDIDATES.items():
            all_rows=[]
            for plan in plans[:1] if args.smoke else plans:
                assert not set(plan['train']+plan['validation'])&set(plan['test'])
                for seed in seeds[:1] if args.smoke else seeds:
                    folder=out/model_name/plan['name']/str(seed);folder.mkdir(parents=True);scores=[]
                    for i,parameters in enumerate(candidates):
                        dest=folder/f'candidate_{i}';dest.mkdir();bundle=fit(cells,plan['train'],model_name,parameters,seed)
                        joblib.dump(bundle,dest/'model.joblib');report,arrays=evaluate(bundle,cells,plan['validation'])
                        save_json(dest/'validation_metrics.json',report);np.savez_compressed(dest/'validation_predictions.npz',**arrays)
                        scores.append(dict(parameters=parameters,validation_mape=float(np.mean([r['metrics']['mape_percent'] for r in report]))))
                    selected=min(scores,key=lambda r:r['validation_mape']);save_json(folder/'selection.json',dict(candidates=scores,selected=selected))
                    bundle=fit(cells,plan['train']+plan['validation'],model_name,selected['parameters'],seed);joblib.dump(bundle,folder/'model.joblib')
                    report,arrays=evaluate(bundle,cells,plan['test']);save_json(folder/'metrics.json',report)
                    np.savez_compressed(folder/'predictions.npz',**arrays);all_rows.extend([dict(r,fold=plan['name'],seed=seed) for r in report])
                logging.info('%s %s saved',model_name,plan['name'])
            save_json(out/model_name/'by_cell.json',all_rows)
            means=[np.mean([r['metrics']['mape_percent'] for r in all_rows if r['seed']==s]) for s in sorted({r['seed'] for r in all_rows})]
            save_json(out/model_name/'summary.json',dict(smoke=args.smoke,mape_mean=float(np.mean(means)),
                mape_sd=float(np.std(means,ddof=1)) if len(means)>1 else None,cells=len({r['cell_id'] for r in all_rows}),
                deterministic_repetition=model_name=='ridge',curves_status='pending',role='public-development whole-cell future forecast'))
            logging.info('%s MAPE %.4f',model_name,np.mean(means))
        save_json(out/'progress.json',dict(training_complete=True,smoke=args.smoke,curves_status='pending verification/export'))
        print('OUTPUT='+str(out.resolve()),flush=True)
    except BaseException:
        save_json(out/'failure.json',dict(traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
