"""Nested NASA early-signal calibration -> future trajectory, no query Q/SOH."""
from datetime import datetime,timezone
import argparse
import json
from pathlib import Path
import shutil
import traceback
import logging
import time
import joblib
import numpy as np
from sklearn.linear_model import Ridge,HuberRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from battery_smetimes.external_soh import load_context
from battery_smetimes.research_baselines import evaluate_baseline,fit_baseline,predict_baseline
from scripts.prepare_same_type_forecast import save_json,sha

COLS=[5,10,34,35,38,40]
CANDIDATES=[dict(alpha=a,window=w) for a in (.1,10.,100.) for w in (30,60)]
RELEASE=Path('deployment/stm32_release_trees/20261004T083706_476897Z')


def inputs(cell):
    take=cell['cycle']<=100;cycle=cell['cycle'][take];raw=cell['raw'][take][:,COLS].astype(np.float64)
    if len(cycle)<60:raise ValueError('insufficient early source signals')
    reference=np.nanmedian(raw[:5],axis=0)
    return cycle,raw-reference


def fit(cells,ids,parameters,seed):
    xs=[];ys=[];ws=[]
    for key in ids:
        c=cells[key];cycles,x=inputs(c)
        lookup={int(n):float(y) for n,y in zip(c['label_cycle'],c['soh']) if np.isfinite(y) and y>0 and n<=100}
        take=np.array([int(n) in lookup for n in cycles]);y=np.array([lookup[int(n)]-1 for n in cycles[take]])
        xs.append(x[take]);ys.append(y);ws.append(np.full(len(y),1/len(ids)/len(y)))
    x=np.concatenate(xs);y=np.concatenate(ys);w=np.concatenate(ws)
    imputer=SimpleImputer(strategy='median',keep_empty_features=True);x=imputer.fit_transform(x)
    scaler=StandardScaler(with_mean=False).fit(x,sample_weight=w)
    model=Ridge(alpha=parameters['alpha'],solver='svd').fit(scaler.transform(x),y,sample_weight=w)
    return dict(model=model,imputer=imputer,scaler=scaler,parameters=parameters,seed=seed,
        train_ids=list(ids),origins=[100],physical_family=cells[ids[0]]['physical_family'],
        input_contract='six early partial-waveform delta statistics; no query Q/SOH or future signals',
        columns=COLS)


def predict(bundle,cell,origin,future):
    if origin!=100 or cell['physical_family']!=bundle['physical_family']:raise ValueError('unsupported origin/family')
    cycles,x=inputs(cell)
    y=1+bundle['model'].predict(bundle['scaler'].transform(bundle['imputer'].transform(x)))
    take=cycles>100-bundle['parameters']['window'];t=(cycles[take]-100)/100
    trend=HuberRegressor(alpha=.01,max_iter=500).fit(t[:,None],y[take])
    return trend.predict(((np.asarray(future)-100)/100)[:,None])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=['signal_trajectory','ridge'],default='signal_trajectory');args=parser.parse_args()
    candidates=CANDIDATES if args.arm=='signal_trajectory' else [dict(alpha=a) for a in (.1,10.,100.)]
    predictor=predict if args.arm=='signal_trajectory' else predict_baseline
    out=Path('experiments/nasa_signal_trajectory')/args.arm/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True);shutil.copy2(__file__,out/Path(__file__).name)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',
        handlers=[logging.FileHandler(out/'training.log',encoding='utf-8'),logging.StreamHandler()])
    sources=[Path('battery_smetimes/research_baselines.py'),Path('battery_smetimes/forecast.py'),Path('battery_smetimes/physical_types.py')]
    for source in sources:
        dest=out/'source_snapshot'/source;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
    context=load_context(RELEASE);cells=context['cells'];frozen=RELEASE/'frozen_audit'
    def fitter(ids,parameters,seed):
        if args.arm=='signal_trajectory':return fit(cells,ids,parameters,seed)
        return fit_baseline(cells,ids,cells[ids[0]]['physical_family'],context['registry'],
            dict(origins_source_cycles=[100]),'ridge',parameters,seed)
    plans=[p for p in json.loads((frozen/'splits.json').read_text())['plans'] if p['dataset']=='NASA']
    seeds=list(range(20261004,20261009));rows=[]
    save_json(out/'config.json',dict(arm=args.arm,candidates=candidates,seeds=seeds,plans=plans,origin=100,
        seed_note='deterministic Ridge/Huber repetitions are not independent experimental replication',
        selection='inner validation only, no repeated outer-score candidate changes',source_sha256=sha(__file__),
        dependency_sha256={str(p):sha(p) for p in sources},frozen_manifest_sha256=sha(RELEASE/'manifest.json'),
        reference_release=str(RELEASE.resolve()),frozen_config_sha256=sha(frozen/'config.json')))
    try:
        for plan in plans:
            if set(plan['train']+plan['validation'])&set(plan['test']):raise ValueError('cell leakage')
            for seed in seeds:
                started=time.perf_counter()
                dest=out/plan['name']/str(seed);dest.mkdir(parents=True);scores=[]
                for i,parameters in enumerate(candidates):
                    candidate=dest/f'candidate_{i}';candidate.mkdir()
                    bundle=fitter(plan['train'],parameters,seed);joblib.dump(bundle,candidate/'model.joblib')
                    report,arrays=evaluate_baseline(bundle,cells,plan['validation'],predictor=predictor)
                    save_json(candidate/'validation_metrics.json',report)
                    np.savez_compressed(candidate/'validation_predictions.npz',**arrays)
                    scores.append(dict(parameters=parameters,validation_mape=report['summary']['100']['cell_macro_mape_percent']))
                    logging.info('candidate fold=%s seed=%s index=%s parameters=%s validation_mape=%.8f',plan['name'],seed,i,parameters,scores[-1]['validation_mape'])
                selected=min(scores,key=lambda r:r['validation_mape'])
                save_json(dest/'selection.json',dict(candidates=scores,selected=selected))
                bundle=fitter(plan['train']+plan['validation'],selected['parameters'],seed)
                joblib.dump(bundle,dest/'model.joblib')
                report,arrays=evaluate_baseline(bundle,cells,plan['test'],predictor=predictor)
                save_json(dest/'metrics.json',report);np.savez_compressed(dest/'predictions.npz',**arrays)
                for key in plan['test']:
                    c=cells[key];future=arrays[key+'__100__cycle'];clean={k:v for k,v in c.items() if k not in ('soh','label_cycle','unfiltered_soh')}
                    np.testing.assert_allclose(predictor(bundle,clean,100,future),arrays[key+'__100__prediction'],rtol=0,atol=0)
                rows.extend([dict(row,fold=plan['name'],seed=seed) for row in report['by_cell']])
                logging.info('saved fold=%s seed=%s selected=%s elapsed_seconds=%.4f test_cells=%s',plan['name'],seed,selected,time.perf_counter()-started,plan['test'])
            save_json(out/'progress.json',dict(complete=False,completed_fold=plan['name']));print(plan['name'],'saved',flush=True)
        save_json(out/'by_cell.json',rows)
        values=[np.mean([r['metrics']['mape_percent'] for r in rows if r['seed']==s]) for s in seeds]
        save_json(out/'summary.json',dict(dataset='NASA',cells=4,seeds=5,mape_mean=float(np.mean(values)),
            mape_sd=float(np.std(values,ddof=1)),passed_6_percent=bool(np.mean(values)<=6),
            role='development LOBO, not a new independent evaluation',curves_status='pending export'))
        save_json(out/'progress.json',dict(complete=True,units=20,curves_status='pending export'))
        print('OUTPUT='+str(out.resolve()),'MAPE='+str(np.mean(values)),flush=True)
        logging.info('complete units=%s cell_macro_mape=%.8f',len(rows),np.mean(values))
    except BaseException:
        save_json(out/'failure.json',dict(traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
