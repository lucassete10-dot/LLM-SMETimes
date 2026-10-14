"""Paired drift-correction smoke; outer cells never loaded."""
import argparse,json,logging,shutil,time
from datetime import datetime,timezone
from pathlib import Path
import joblib,numpy as np
from threadpoolctl import threadpool_limits
from battery_smetimes.tail_boost import fit,labels,is_nca,predict as base_predict
from battery_smetimes.crossfit_drift import split_ids,fit_head,predict
from battery_smetimes.curve_fidelity import curve_metrics,selection_score
from scripts.train_tail_boost import load_cells
from scripts.prepare_same_type_forecast import save_json,sha


def clean(cell):
    remove=('cycle','truth') if is_nca(cell) else ('soh','label_cycle','unfiltered_soh','reference_capacity_ah')
    return {k:v for k,v in cell.items() if k not in remove}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dataset');parser.add_argument('--all-folds',action='store_true')
    args=parser.parse_args()
    seed=20261005;start=time.perf_counter()
    routes=[('MIT','20261005T034441_870182Z'),('XJTU','20261005T034452_066902Z'),
        ('Zhu2022_NCA','20261005T034431_681399Z'),('NASA','20261005T034001_140404Z')]
    root=Path('experiments/crossfit_drift')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    root.mkdir(parents=True);snap=root/'source_snapshot';snap.mkdir()
    sources=[Path(__file__),Path('battery_smetimes/crossfit_drift.py'),Path('battery_smetimes/tail_boost.py'),
        Path('battery_smetimes/research_baselines.py'),Path('battery_smetimes/forecast.py'),
        Path('scripts/train_tail_boost.py'),Path('scripts/train_curve_fidelity.py'),
        Path('scripts/prepare_zhu2022_nca.py'),Path('scripts/train_zhu2022_nca.py')]
    for f in sources:shutil.copy2(f,snap/f.name)
    plans=[];provenance=[];data_hashes={}
    for dataset,stamp in routes:
        if args.dataset and dataset!=args.dataset:continue
        file=Path('experiments/tail_boost')/stamp/'config.json';cfg=json.loads(file.read_text())
        selected=[p for p in cfg['plans'] if p['dataset']==dataset]
        selected=selected if args.all_folds else selected[:1];plans.extend(selected)
        provenance.append(dict(file=str(file.resolve()),sha256=sha(file)))
        for plan in selected:
            for key in plan['train']+plan['validation']:data_hashes[key]=cfg['prepared_sha256'][key]
    assert plans, 'no matching dataset'
    candidates=[None,dict(degree=1,alpha=10),dict(degree=2,alpha=10),dict(degree=2,alpha=100)]
    params=dict(model='extra_trees',leaf=1,tail_weight=3,local=False)
    save_json(root/'config.json',dict(plans=plans,seed=seed,parameters=params,candidates=candidates,
        smoke=True,provenance=provenance,prepared_sha256=data_hashes,source_sha256={f.name:sha(f) for f in sources},
        preregistration_sha256=sha('docs/soh_crossfit_drift_stage_20261005.md')))
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',
        handlers=[logging.FileHandler(root/'training.log',encoding='utf-8'),logging.StreamHandler()])
    reports=[]
    for plan in plans:
        folder=root/plan['name'];folder.mkdir();cells=load_cells(plan)
        assert set(cells)==set(plan['train']+plan['validation']) and not set(cells)&set(plan['test'])
        splits=split_ids(plan['train'],seed);save_json(folder/'oof_splits.json',splits)
        parts=[];truth=[];predictions=[];weights=[];oof_arrays={}
        for index,split in enumerate(splits):
            logging.info('OOF_FIT %s part=%s train=%s held=%s',plan['name'],index,len(split['train']),len(split['held']))
            base=fit(cells,split['train'],params,seed);joblib.dump(base,folder/f'oof_{index}.joblib')
            assert not set(base['train_ids'])&set(split['held'])
            for key in split['held']:
                x,y=labels(cells[key]);idx=np.unique(np.linspace(0,len(y)-1,min(96,len(y))).round().astype(int))
                x,y=x[idx],y[idx];p=base_predict(base,clean(cells[key]),x)
                parts.append(x);truth.append(y);predictions.append(p);w=np.linspace(1,3,len(x));weights.append(w/w.sum())
                oof_arrays.update({key+'__'+f:a for f,a in [('cycle',x),('truth',y),('prediction',p)]})
        x,y,p,w=[np.concatenate(v) for v in (parts,truth,predictions,weights)];w*=len(w)/w.sum()
        np.savez_compressed(folder/'oof_residuals.npz',**oof_arrays,aggregate_cycle=x,aggregate_residual=y-p,aggregate_weight=w)
        logging.info('FULL_TRAIN_FIT %s',plan['name']);base=fit(cells,plan['train'],params,seed)
        joblib.dump(base,folder/'base.joblib')
        for index,candidate in enumerate(candidates):
            dest=folder/f'candidate_{index}';dest.mkdir();head=None if candidate is None else fit_head(x,y-p,w,**candidate)
            bundle=dict(base=base,head=head,train_ids=plan['train'],seed=seed);joblib.dump(bundle,dest/'model.joblib')
            rows=[];arrays={}
            for key in plan['validation']:
                cell=cells[key];xx,yy=labels(cell);cc=clean(cell);pp=predict(bundle,cc,xx)
                if is_nca(cc):
                    poison=dict(cc,early_cycle=np.r_[cc['early_cycle'],101],early_raw=np.vstack([cc['early_raw'],np.full((1,cc['early_raw'].shape[1]),np.inf)]),truth=np.inf,cycle_life=np.inf)
                else:
                    poison=dict(cc,cycle=np.r_[cc['cycle'],101],raw=np.vstack([cc['raw'],np.full((1,cc['raw'].shape[1]),np.inf)]),soh=np.inf,cycle_life=np.inf)
                np.testing.assert_allclose(pp,predict(bundle,poison,xx),atol=1e-12,rtol=0)
                try:predict(bundle,dict(cc,physical_family='different'),xx)
                except ValueError:pass
                else:raise AssertionError('family gate failed')
                rows.append(dict(cell_id=key,**curve_metrics(xx,yy,pp)))
                arrays.update({key+'__100__'+f:a for f,a in [('cycle',xx),('truth',yy),('prediction',pp)]})
            save_json(dest/'validation_metrics.json',rows);np.savez_compressed(dest/'validation_predictions.npz',**arrays)
            report=dict(dataset=plan['dataset'],fold=plan['name'],candidate=index,parameters=candidate,score=selection_score(rows),
                **{f:float(np.mean([r[f] for r in rows])) for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp')})
            reports.append(report);save_json(root/'summary.json',reports);logging.info('SAVED %s',report)
    save_json(root/'progress.json',dict(training_complete=True,seconds=time.perf_counter()-start,outer_evaluated=False,goal_complete=False))
    print('OUTPUT='+str(root.resolve()))


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
