"""Paired first-fold feature ablation; validation only, no outer data."""
import json, logging, shutil
from datetime import datetime, timezone
from pathlib import Path
import joblib, numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from threadpoolctl import threadpool_limits
from battery_smetimes.temporal_features import matrix
from battery_smetimes.tail_boost import labels, is_nca
from battery_smetimes.curve_fidelity import curve_metrics
from scripts.train_tail_boost import load_cells
from scripts.prepare_same_type_forecast import save_json, sha


def main():
    routes = [('MIT','20261005T034441_870182Z'), ('XJTU','20261005T034452_066902Z'),
        ('Zhu2022_NCA','20261005T034431_681399Z'), ('NASA','20261005T034001_140404Z')]
    root = Path('experiments/temporal_feature_smoke')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    root.mkdir(parents=True); snap = root/'source_snapshot'; snap.mkdir()
    sources = [Path(__file__), Path('battery_smetimes/temporal_features.py'), Path('battery_smetimes/tail_boost.py'),
        Path('battery_smetimes/forecast.py'), Path('battery_smetimes/research_baselines.py'),
        Path('scripts/train_tail_boost.py'), Path('scripts/prepare_zhu2022_nca.py'), Path('scripts/train_zhu2022_nca.py')]
    for file in sources: shutil.copy2(file, snap/file.name)
    plans = []; provenance = []
    for dataset, run in routes:
        file = Path('experiments/tail_boost')/run/'config.json'; cfg=json.loads(file.read_text())
        plans.append(next(p for p in cfg['plans'] if p['dataset']==dataset))
        provenance.append(dict(file=str(file.resolve()),sha256=sha(file)))
    save_json(root/'config.json',dict(plans=plans,seed=20261005,candidates=[False,True],smoke=True,
        fixed_model=dict(trees=128,leaf=1,tail_weight=3),provenance=provenance,
        source_sha256={f.name:sha(f) for f in sources},preregistration_sha256=sha('docs/soh_temporal_features_stage_20261005.md')))
    logging.basicConfig(level=logging.INFO,handlers=[logging.FileHandler(root/'training.log',encoding='utf-8'),logging.StreamHandler()])
    reports = []
    for plan in plans:
        cells=load_cells(plan);assert set(cells)==set(plan['train']+plan['validation'])
        families={c['physical_family'] for c in cells.values()};assert len(families)==1
        for augmented in (False,True):
            folder=root/plan['name']/str(int(augmented));folder.mkdir(parents=True)
            logging.info('TRAIN %s augmented=%s',plan['name'],augmented)
            xs=[];ys=[];weights=[]
            for key in plan['train']:
                x,y=labels(cells[key]);idx=np.unique(np.linspace(0,len(y)-1,min(96,len(y))).round().astype(int))
                xs.append(matrix(cells[key],x[idx],augmented));ys.append(y[idx]);w=np.linspace(1,3,len(idx));weights.append(w/w.sum())
            xx=np.concatenate(xs);yy=np.concatenate(ys);ww=np.concatenate(weights);ww*=len(ww)/ww.sum()
            imputer=SimpleImputer(strategy='median',keep_empty_features=True);z=imputer.fit_transform(xx)
            model=ExtraTreesRegressor(n_estimators=128,min_samples_leaf=1,random_state=20261005,n_jobs=2)
            model.fit(z,yy,sample_weight=ww)
            joblib.dump(dict(model=model,imputer=imputer,train_ids=plan['train'],family=next(iter(families)),augmented=augmented),folder/'model.joblib')
            rows=[];arrays={}
            for key in plan['validation']:
                c=cells[key];x,y=labels(c)
                remove=('cycle','truth') if is_nca(c) else ('soh','label_cycle','unfiltered_soh','reference_capacity_ah')
                clean={k:v for k,v in c.items() if k not in remove}; features=matrix(clean,x,augmented)
                p=model.predict(imputer.transform(features))
                if is_nca(c):
                    poison=dict(clean,early_cycle=np.r_[clean['early_cycle'],101],early_raw=np.vstack([clean['early_raw'],np.full((1,clean['early_raw'].shape[1]),np.inf)]),truth=np.inf,cycle_life=np.inf)
                else:
                    poison=dict(clean,cycle=np.r_[clean['cycle'],101],raw=np.vstack([clean['raw'],np.full((1,clean['raw'].shape[1]),np.inf)]),soh=np.inf,cycle_life=np.inf)
                np.testing.assert_array_equal(features,matrix(poison,x,augmented))
                rows.append(dict(cell_id=key,**curve_metrics(x,y,p)))
                arrays.update({key+'__100__'+f:a for f,a in [('cycle',x),('truth',y),('prediction',p)]})
            save_json(folder/'validation_metrics.json',rows);np.savez_compressed(folder/'validation_predictions.npz',**arrays)
            scores=[(r['mape_percent']+r['late_mape']+r['endpoint_pp']+r['drop_error_pp'])/4 for r in rows]
            report=dict(dataset=plan['dataset'],augmented=augmented,validation_score=float(np.mean(scores)+.25*np.quantile(scores,.9)),
                **{f:float(np.mean([r[f] for r in rows])) for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp')})
            reports.append(report);save_json(root/'summary.json',reports);logging.info('SAVED %s',report)
    save_json(root/'progress.json',dict(training_complete=True,goal_complete=False,outer_evaluated=False))
    print('OUTPUT='+str(root.resolve()))


if __name__=='__main__':
    with threadpool_limits(limits=2): main()
