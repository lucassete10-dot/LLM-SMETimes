"""Matched-horizon, held-out-cell inner screening of early input lengths."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import shutil
import time
import traceback
import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from battery_smetimes.research_baselines import signal_prefix, regression_metrics
from scripts.prepare_zhu2022_nca import prefix as nca_prefix
from scripts.train_tail_boost import load_cells
from scripts.train_curve_fidelity import AUDIT
from scripts.train_nca_curve_fidelity import OLD
from scripts.prepare_same_type_forecast import sha, save_json

ORIGINS = [10, 20, 30, 50, 75, 100]
SEEDS = list(range(20261008, 20261013))
CANDIDATES = [('ridge', 10), ('extra_trees', 1), ('extra_trees', 4)]


def prefix(cell, origin):
    if 'early_cycle' not in cell:
        return signal_prefix(cell, origin)
    take = np.asarray(cell['early_cycle']) <= origin
    return nca_prefix(cell['early_cycle'][take], cell['early_raw'][take], cell['conditions'])


def target(cell):
    if 'early_cycle' in cell:
        c, y = cell['cycle'], cell['truth']
    else:
        c, y = cell['label_cycle'], cell['soh']
    take = (c > 100) & np.isfinite(y) & (y > 0)
    return c[take].astype(float), y[take].astype(float)


def matrix(cell, origin, cycle):
    p = prefix(cell, origin)
    h = (np.asarray(cycle) - origin) / 100
    return np.c_[np.broadcast_to(p, (len(h), len(p))), np.asarray(cycle)/100, h, np.log1p(h)]


def metrics(c, y, p):
    base = regression_metrics(y, p)
    late = c >= c[0] + .8*(c[-1]-c[0])
    base['late20_mape_percent'] = regression_metrics(y[late], p[late])['mape_percent']
    base['endpoint_pp'] = float(100*abs(y[-1]-p[-1]))
    base['start_pp'] = float(100*abs(y[0]-p[0]))
    base['drop_error_pp'] = float(100*abs((y[0]-y[-1])-(p[0]-p[-1])))
    levels = []
    for values in (y, p):
        levels.append(np.array([np.median(values[np.argsort(abs(c-b), kind='stable')[:min(5,len(c))]])
            for b in np.linspace(c[0], c[-1], 4)]))
    yd, pd = -100*np.diff(levels[0]), -100*np.diff(levels[1])
    base['stage_error_pp'] = float(np.max(abs(yd-pd)))
    base['acceleration_direction_reversed'] = bool(yd[-1]-yd[0] > 1 and pd[-1]-pd[0] < -1)
    base['fade_normalized_mae'] = float(base['mae_soh']*100/max(abs((y[0]-y[-1])*100),2))
    base['short_coverage'] = bool(len(c)<10 or c[-1]-c[0]<100)
    base['curve_screen_pass'] = bool(base['mape_percent']<=6 and base['mae_soh']<=.01
        and base['late20_mape_percent']<=2 and base['endpoint_pp']<=2
        and base['drop_error_pp']<=2 and base['stage_error_pp']<=2
        and base['fade_normalized_mae']<=.25 and not base['acceleration_direction_reversed']
        and not base['short_coverage'])
    return base


def predict(bundle, cell, origin, cycle):
    if cell['physical_family'] != bundle['family']:
        raise ValueError('physical family mismatch')
    x = bundle['imputer'].transform(matrix(cell, origin, cycle))
    if bundle['scaler'] is not None:
        x = bundle['scaler'].transform(x)
    return bundle['model'].predict(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--full', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    plans = json.loads((AUDIT/'splits.json').read_text())['plans']
    plans += [dict(p, dataset='Zhu2022_NCA') for p in json.loads((OLD/'config.json').read_text())['plans']]
    if not args.full:
        plans = [p for p in plans if p['name'].endswith('fold1')]
    seeds = SEEDS[:1] if args.smoke else SEEDS
    root = Path('experiments/prefix_length_screen')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    root.mkdir(parents=True)
    print('RUN='+str(root.resolve()), flush=True)
    sources = [Path(__file__), Path('battery_smetimes/research_baselines.py'),
        Path('battery_smetimes/forecast.py'), Path('scripts/train_tail_boost.py'),
        Path('scripts/train_curve_fidelity.py'), Path('scripts/prepare_zhu2022_nca.py')]
    snap = root/'source_snapshot'; snap.mkdir()
    for file in sources:
        shutil.copy2(file, snap/file.name)
    shutil.copy2('docs/soh_prefix_length_registration_20261008.md', root/'registration.md')
    save_json(root/'config.json', dict(plans=plans, origins=ORIGINS, seeds=seeds,
        candidates=CANDIDATES, source_sha256={str(p):sha(p) for p in sources},
        outer_loaded=False, common_future='source cycle >100; same targets across origins',
        full=args.full, smoke=args.smoke, final_goal_complete=False))
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(root/'training.log', encoding='utf-8')])
    allrows = []; coverage = []; started = time.perf_counter(); fits = 0
    try:
        for plan in plans:
            assert not set(plan['train']) & set(plan['validation'])
            assert not set(plan['train']+plan['validation']) & set(plan['test'])
            cells = load_cells(plan)
            families = {c['physical_family'] for c in cells.values()}
            assert len(families)==1 and not set(cells)&set(plan['test'])
            usable = {}
            for role in ('train','validation'):
                keep = []; skipped = []
                for key in plan[role]:
                    try:
                        if len(target(cells[key])[0]) < 2:
                            raise ValueError('insufficient common future')
                        for origin in ORIGINS:
                            prefix(cells[key], origin)
                    except ValueError as e:
                        skipped.append(dict(cell_id=key, reason=str(e)))
                    else:
                        keep.append(key)
                usable[role]=keep
                coverage.append(dict(fold=plan['name'], role=role, original=len(plan[role]),
                    matched_cells=keep, skipped=skipped))
            save_json(root/'coverage.json', coverage)
            if not usable['train'] or not usable['validation']:
                logging.warning('No matched cohort for %s', plan['name']); continue
            destplan = root/plan['name']; destplan.mkdir()
            save_json(destplan/'data_provenance.json', {key:cells[key].get('prepared_sha256',
                'NCA per-cell index SHA checked by load_cells') for key in cells})
            for origin in ORIGINS:
                xs=[]; ys=[]; ws=[]
                for key in usable['train']:
                    c,y=target(cells[key]); ii=np.unique(np.linspace(0,len(y)-1,min(96,len(y))).round().astype(int))
                    xs.append(matrix(cells[key],origin,c[ii]));ys.append(y[ii]);ws.append(np.ones(len(ii))/len(ii))
                x,y,w=np.concatenate(xs),np.concatenate(ys),np.concatenate(ws); w*=len(w)/w.sum()
                for seed in seeds:
                    for model_name,parameter in CANDIDATES:
                        folder=destplan/str(origin)/str(seed)/(model_name+'_'+str(parameter));folder.mkdir(parents=True)
                        imputer=SimpleImputer(strategy='median',keep_empty_features=True);z=imputer.fit_transform(x)
                        scaler=None; fitstart=time.perf_counter()
                        if model_name=='ridge':
                            scaler=StandardScaler();z=scaler.fit_transform(z)
                            model=Ridge(alpha=parameter,solver='svd')
                        else:
                            model=ExtraTreesRegressor(n_estimators=80,min_samples_leaf=parameter,
                                max_features=.8,random_state=seed,n_jobs=2)
                        model.fit(z,y,sample_weight=w)
                        bundle=dict(model=model,imputer=imputer,scaler=scaler,family=next(iter(families)),
                            train_ids=usable['train'],origin=origin,seed=seed,fit_seconds=time.perf_counter()-fitstart)
                        joblib.dump(bundle,folder/'model.joblib')
                        save_json(folder/'config.json',dict(plan=plan,used_train=usable['train'],
                            origin=origin,seed=seed,model=model_name,parameter=parameter,
                            model_sha256=sha(folder/'model.joblib'),fit_seconds=bundle['fit_seconds'],
                            input_width=x.shape[1],training_pairs=len(y),outer_loaded=False))
                        rows=[];arrays={}
                        for key in usable['validation']:
                            cell=cells[key];c,truth=target(cell);p=predict(bundle,cell,origin,c)
                            poison=deepcopy(cell)
                            for name in ('soh','truth'):
                                if name in poison:poison[name]=np.full_like(poison[name],np.inf)
                            rawkey,cykey=('early_raw','early_cycle') if 'early_cycle' in cell else ('raw','cycle')
                            poison[rawkey][poison[cykey]>origin]=np.inf
                            np.testing.assert_allclose(p,predict(bundle,poison,origin,c),atol=1e-12,rtol=0)
                            rows.append(dict(dataset=plan['dataset'],fold=plan['name'],cell_id=key,
                                origin=origin,seed=seed,model=model_name,parameter=parameter,
                                actual_input_records=int(np.sum(cell[cykey]<=origin)),**metrics(c,truth,p)))
                            arrays.update({key+'__'+f:v for f,v in [('cycle',c),('truth',truth),('prediction',p)]})
                        np.savez_compressed(folder/'validation_predictions.npz',**arrays)
                        save_json(folder/'metrics.json',rows);allrows.extend(rows);fits+=1
                        logging.info('saved %s origin=%s seed=%s model=%s parameter=%s fits=%s',plan['name'],origin,seed,model_name,parameter,fits)
                        save_json(root/'progress.json',dict(terminal=False,completed_fits=fits,
                            seconds=time.perf_counter()-started,outer_loaded=False))
                    print(f'FINISHED {plan["name"]} origin={origin} seed={seed} fits={fits}',flush=True)
        save_json(root/'all_validation_metrics.json',allrows)
        report=[]
        for dataset in sorted({r['dataset'] for r in allrows}):
            for origin in ORIGINS:
                for model_name,parameter in CANDIDATES:
                    rows=[r for r in allrows if (r['dataset'],r['origin'],r['model'],r['parameter'])==(dataset,origin,model_name,parameter)]
                    perseed=[]
                    for seed in seeds:
                        rr=[r for r in rows if r['seed']==seed]
                        if rr:perseed.append({k:float(np.mean([r[k] for r in rr])) for k in
                            ('mape_percent','mae_soh','late20_mape_percent','endpoint_pp','drop_error_pp','stage_error_pp','curve_screen_pass')})
                    if perseed:
                        report.append(dict(dataset=dataset,origin=origin,model=model_name,parameter=parameter,
                            unique_validation_cells=len({r['cell_id'] for r in rows}),seeds=len(perseed),
                            **{k:dict(mean=float(np.mean([s[k] for s in perseed])),
                                sd=float(np.std([s[k] for s in perseed],ddof=1)) if len(perseed)>1 else None) for k in perseed[0]}))
        save_json(root/'summary.json',dict(rows=report,stage='inner input-length baseline screening',
            minimal_origin_conclusion='pending all folds and improved main model confirmation',
            short_coverage_count=sum(r['short_coverage'] for r in allrows),outer_loaded=False,goal_complete=False))
        save_json(root/'progress.json',dict(terminal=True,completed_fits=fits,seconds=time.perf_counter()-started,
            input_poison_checks_passed=True,outer_loaded=False,goal_complete=False))
        print('COMPLETE '+str(root.resolve()),flush=True)
    except BaseException:
        save_json(root/'failure.json',dict(traceback=traceback.format_exc(),completed_fits=fits))
        raise


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
