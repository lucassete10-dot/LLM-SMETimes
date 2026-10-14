"""Preregistered horizon interactions, frozen nested NASA splits."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import traceback
import logging
import time
import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from battery_smetimes.external_soh import load_context
from battery_smetimes.research_baselines import build_training_pairs,signal_prefix,evaluate_baseline
from battery_smetimes.forecast import query_matrix
from battery_smetimes.physical_types import require_same_registered_family
from scripts.prepare_same_type_forecast import save_json,sha

RELEASE=Path('deployment/stm32_release_trees/20261004T083706_476897Z')
ALPHAS=[.1,10.,100.]

def expand(x):
    x=np.asarray(x,dtype=np.float64)
    return np.c_[x,x[:,:-3]*x[:,-2,None],x[:,:-3]*x[:,-1,None]]

def fit(cells,ids,registry,alpha,seed):
    family=cells[ids[0]]['physical_family']
    require_same_registered_family(family,[cells[k]['physical_family'] for k in ids],registry)
    x,y,w,used=build_training_pairs(cells,ids,[100]);x=expand(x)
    imputer=SimpleImputer(strategy='median',keep_empty_features=True);x=imputer.fit_transform(x)
    offset=x[0].copy();scaler=StandardScaler().fit(x-offset,sample_weight=w);scaler.mean_+=offset
    model=Ridge(alpha=alpha,solver='svd').fit(scaler.transform(x),y,sample_weight=w)
    return dict(model=model,imputer=imputer,scaler=scaler,train_ids=list(ids),used_train_ids=used,
        registry=registry,physical_family=family,alpha=alpha,seed=seed,origins=[100],
        input_contract='existing label-free early statistics plus requested-horizon interactions')

def predict(bundle,cell,origin,future):
    require_same_registered_family(cell['physical_family'],[bundle['physical_family']],bundle['registry'])
    x=expand(query_matrix(signal_prefix(cell,origin),origin,future))
    return bundle['model'].predict(bundle['scaler'].transform(bundle['imputer'].transform(x)))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');parser.add_argument('--validation-only',action='store_true');args=parser.parse_args()
    out=Path('experiments/nasa_interaction_ridge')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');out.mkdir(parents=True)
    shutil.copy2(__file__,out/Path(__file__).name);shutil.copy2('docs/nasa_interaction_experiment_plan.md',out/'plan.md')
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',
        handlers=[logging.FileHandler(out/'training.log',encoding='utf-8'),logging.StreamHandler()])
    sources=[Path('battery_smetimes/research_baselines.py'),Path('battery_smetimes/forecast.py'),Path('battery_smetimes/physical_types.py')]
    for source in sources:
        dest=out/'source_snapshot'/source;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
    ctx=load_context(RELEASE);cells=ctx['cells']
    plans=[p for p in json.loads((RELEASE/'frozen_audit/splits.json').read_text())['plans'] if p['dataset']=='NASA']
    seeds=list(range(20261004,20261009));rows=[]
    validation_only=args.smoke or args.validation_only
    save_json(out/'config.json',dict(smoke=args.smoke,validation_only=validation_only,alphas=ALPHAS,plans=plans,seeds=seeds,
        source_sha256=sha(__file__),plan_sha256=sha('docs/nasa_interaction_experiment_plan.md'),
        dependency_sha256={str(p):sha(p) for p in sources},frozen_manifest_sha256=sha(RELEASE/'manifest.json'),
        origin=100,deterministic_repetition=True,selection='inner validation only',reference_release=str(RELEASE.resolve())))
    try:
        for plan in plans[:1] if args.smoke else plans:
            for seed in seeds[:1] if args.smoke else seeds:
                started=time.perf_counter()
                folder=out/plan['name']/str(seed);folder.mkdir(parents=True);scores=[]
                for i,alpha in enumerate(ALPHAS):
                    dest=folder/f'candidate_{i}';dest.mkdir()
                    bundle=fit(cells,plan['train'],ctx['registry'],alpha,seed);joblib.dump(bundle,dest/'model.joblib')
                    report,arrays=evaluate_baseline(bundle,cells,plan['validation'],predictor=predict)
                    save_json(dest/'validation_metrics.json',report);np.savez_compressed(dest/'validation_predictions.npz',**arrays)
                    scores.append(dict(alpha=alpha,validation_mape=report['summary']['100']['cell_macro_mape_percent']))
                    logging.info('candidate fold=%s seed=%s index=%s alpha=%s validation_mape=%.8f',plan['name'],seed,i,alpha,scores[-1]['validation_mape'])
                selected=min(scores,key=lambda r:r['validation_mape'])
                save_json(folder/'selection.json',dict(candidates=scores,selected=selected))
                logging.info('selected fold=%s seed=%s parameters=%s validation_only=%s elapsed_seconds=%.4f',plan['name'],seed,selected,validation_only,time.perf_counter()-started)
                if validation_only:continue
                bundle=fit(cells,plan['train']+plan['validation'],ctx['registry'],selected['alpha'],seed)
                joblib.dump(bundle,folder/'model.joblib');report,arrays=evaluate_baseline(bundle,cells,plan['test'],predictor=predict)
                save_json(folder/'metrics.json',report);np.savez_compressed(folder/'predictions.npz',**arrays)
                rows.extend([dict(r,fold=plan['name'],seed=seed) for r in report['by_cell']])
            print(plan['name'],'saved',flush=True)
        save_json(out/'by_cell.json',rows)
        save_json(out/'progress.json',dict(training_complete=True,smoke=args.smoke,
            outer_predictions_created=not validation_only,curves_status='inner validation predictions saved; no outer curves' if validation_only else 'pending export'))
        print('OUTPUT='+str(out.resolve()),flush=True)
        logging.info('complete validation_only=%s outer_rows=%s',validation_only,len(rows))
    except BaseException:
        save_json(out/'failure.json',dict(traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
