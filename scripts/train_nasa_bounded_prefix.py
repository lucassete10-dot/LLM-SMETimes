"""Nested selection of bounded early-input Ridge, never bound labels/horizon."""
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
from battery_smetimes.external_soh import load_context
from battery_smetimes.research_baselines import fit_baseline,evaluate_baseline,signal_prefix,build_training_pairs
from battery_smetimes.forecast import query_matrix
from battery_smetimes.physical_types import require_same_registered_family
from scripts.prepare_same_type_forecast import save_json,sha

RELEASE=Path('deployment/stm32_release_trees/20261004T083706_476897Z')
CANDIDATES=[dict(alpha=a,prefix_z_limit=limit) for limit in (None,3.,6.) for a in (.1,10.,100.)]

def predict(bundle,cell,origin,future):
    require_same_registered_family(cell['physical_family'],[bundle['physical_family']],bundle['registry'])
    x=query_matrix(signal_prefix(cell,origin),origin,future).astype(float)
    z=bundle['scaler'].transform(bundle['imputer'].transform(x))
    limit=bundle['parameters']['prefix_z_limit']
    if limit is not None:z[:,:-3]=np.clip(z[:,:-3],-limit,limit)
    return bundle['model'].predict(z)

def fit(cells,ids,registry,parameters,seed):
    bundle=fit_baseline(cells,ids,cells[ids[0]]['physical_family'],registry,
        dict(origins_source_cycles=[100]),'ridge',dict(alpha=parameters['alpha']),seed)
    bundle['parameters']=dict(parameters)
    x,_,_,_=build_training_pairs(cells,ids,[100])
    z=bundle['scaler'].transform(bundle['imputer'].transform(x.astype(float)))
    limit=parameters['prefix_z_limit']
    if limit is not None:assert np.max(np.abs(z[:,:-3]))<=limit+1e-8
    return bundle

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--validation-only',action='store_true');args=parser.parse_args()
    out=Path('experiments/nasa_bounded_prefix')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');out.mkdir(parents=True)
    shutil.copy2(__file__,out/Path(__file__).name);shutil.copy2('docs/nasa_bounded_prefix_plan.md',out/'plan.md')
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',
        handlers=[logging.FileHandler(out/'training.log',encoding='utf-8'),logging.StreamHandler()])
    sources=[Path('battery_smetimes/research_baselines.py'),Path('battery_smetimes/forecast.py'),Path('battery_smetimes/physical_types.py')]
    for source in sources:
        dest=out/'source_snapshot'/source;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
    ctx=load_context(RELEASE);cells=ctx['cells'];plans=[p for p in json.loads((RELEASE/'frozen_audit/splits.json').read_text())['plans'] if p['dataset']=='NASA']
    seeds=list(range(20261004,20261009));rows=[]
    save_json(out/'config.json',dict(plans=plans,seeds=seeds,candidates=CANDIDATES,
        validation_only=args.validation_only,source_sha256=sha(__file__),plan_sha256=sha('docs/nasa_bounded_prefix_plan.md'),
        dependency_sha256={str(p):sha(p) for p in sources},frozen_manifest_sha256=sha(RELEASE/'manifest.json'),
        selection='inner validation only',origin=100,reference_release=str(RELEASE.resolve()),deterministic_repetition=True))
    try:
        for plan in plans:
            for seed in seeds:
                started=time.perf_counter()
                folder=out/plan['name']/str(seed);folder.mkdir(parents=True);scores=[]
                for i,parameters in enumerate(CANDIDATES):
                    dest=folder/f'candidate_{i}';dest.mkdir();bundle=fit(cells,plan['train'],ctx['registry'],parameters,seed)
                    joblib.dump(bundle,dest/'model.joblib');report,arrays=evaluate_baseline(bundle,cells,plan['validation'],predictor=predict)
                    save_json(dest/'validation_metrics.json',report);np.savez_compressed(dest/'validation_predictions.npz',**arrays)
                    scores.append(dict(parameters=parameters,validation_mape=report['summary']['100']['cell_macro_mape_percent']))
                    logging.info('candidate fold=%s seed=%s index=%s parameters=%s validation_mape=%.8f',plan['name'],seed,i,parameters,scores[-1]['validation_mape'])
                selected=min(scores,key=lambda r:r['validation_mape']);save_json(folder/'selection.json',dict(candidates=scores,selected=selected))
                if args.validation_only:continue
                bundle=fit(cells,plan['train']+plan['validation'],ctx['registry'],selected['parameters'],seed);joblib.dump(bundle,folder/'model.joblib')
                report,arrays=evaluate_baseline(bundle,cells,plan['test'],predictor=predict)
                save_json(folder/'metrics.json',report);np.savez_compressed(folder/'predictions.npz',**arrays)
                rows.extend([dict(r,fold=plan['name'],seed=seed) for r in report['by_cell']])
                logging.info('saved fold=%s seed=%s selected=%s elapsed_seconds=%.4f test_cells=%s',plan['name'],seed,selected,time.perf_counter()-started,plan['test'])
            print(plan['name'],'saved',flush=True)
        save_json(out/'by_cell.json',rows)
        if not args.validation_only:
            means=[np.mean([r['metrics']['mape_percent'] for r in rows if r['seed']==s]) for s in seeds]
            save_json(out/'summary.json',dict(mape_mean=float(np.mean(means)),mape_sd=float(np.std(means,ddof=1)),
                cells=4,seeds=5,role='public development LOBO',curves_status='pending'))
        save_json(out/'progress.json',dict(training_complete=True,validation_only=args.validation_only,
            outer_predictions_created=not args.validation_only,curves_status='no outer evaluation' if args.validation_only else 'pending'))
        print('OUTPUT='+str(out.resolve()),flush=True)
        logging.info('complete units=%s validation_only=%s',len(rows),args.validation_only)
    except BaseException:
        save_json(out/'failure.json',dict(traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
