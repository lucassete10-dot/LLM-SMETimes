"""Separate Oxford sparse-diagnostic experiment; never claim dense measured I."""
import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import shutil
import traceback
import joblib
import numpy as np
from scipy.io import loadmat
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from battery_smetimes.research_baselines import regression_metrics
from scripts.prepare_same_type_forecast import save_json, sha

RAW=Path('data/raw/oxford_degradation/20261004T143528_639736Z')
FAMILY='Oxford_Kokam_SLPB533459H4_740mAh_40C'
ALPHAS=[1.,10.,100.]

def elapsed(t):
    t=np.asarray(t,dtype=np.float64).ravel()
    if len(t)<3 or not np.isfinite(t).all() or np.any(np.diff(t)<=0):
        raise ValueError('nonfinite or non-increasing diagnostic time')
    dt=float(np.median(np.diff(t)))
    if .5<dt<2:factor=1.;unit='seconds'
    elif .5<dt*86400<2 and np.median(t)>700000:
        factor=86400.;unit='MATLAB serial days inferred from absolute magnitude and 1-second cadence'
    else:raise ValueError('unresolved time unit')
    return (t-t[0])*factor,unit

def diagnostic_features(record):
    # Explicit allowlist: capacity q and diagnostic duration are never inputs.
    t,_=elapsed(record['t'])
    v=np.asarray(record['v'],dtype=float).ravel();temp=np.asarray(record['T'],dtype=float).ravel()
    if len(v)!=len(t) or len(temp)!=len(t):raise ValueError('signal lengths differ')
    take=t<=900
    if t[take][-1]<899 or not np.isfinite(v[take]).all() or not np.isfinite(temp[take]).all():
        raise ValueError('insufficient finite early 900-second signals')
    grid=np.arange(0.,901.,90.)
    # Last grid point uses last available <=900s sample; no future interpolation.
    def features(a):
        a=a[take]
        return np.r_[np.interp(grid,t[take],a),a.mean(),a.std(),a.min(),a.max(),a[-1]-a[0]]
    return np.r_[features(v),features(temp),.740,40.]

def prepare(out):
    raw=loadmat(RAW/'Oxford_Battery_Degradation_Dataset_1.mat',simplify_cells=True)
    cells={};audit=[];exclusions=[];units=set()
    for key in sorted(k for k in raw if k.startswith('Cell')):
        records=raw[key]
        assert 'cyc0000' in records and 'cyc0100' in records
        first=diagnostic_features(records['cyc0000']['C1dc'])
        second=diagnostic_features(records['cyc0100']['C1dc'])
        for name in ('cyc0000','cyc0100'):
            poisoned=dict(records[name]['C1dc'],q=np.array([np.inf]))
            np.testing.assert_array_equal(diagnostic_features(poisoned),diagnostic_features(records[name]['C1dc']))
        cycles=[];capacities=[];rates=[]
        for name,record in sorted(records.items()):
            n=int(name[3:])
            try:
                dc=record['C1dc'];t,unit=elapsed(dc['t']);units.add(unit)
                q=np.asarray(dc['q'],dtype=float).ravel()
                if len(q)!=len(t) or not np.isfinite(q).all():raise ValueError('invalid capacity trace')
                if np.any(np.diff(q)>1e-5):raise ValueError('nonmonotone discharge charge trace')
                capacity=float(abs(q[-1]-q[0]))
                if not 0<capacity<1000:raise ValueError('capacity outside physical 0-1000mAh range')
                rate=capacity/(t[-1]/3600)/1000
                if not .70<rate<.78:raise ValueError('time/capacity inconsistent with documented .740A protocol')
                cycles.append(n);capacities.append(capacity);rates.append(rate)
            except (ValueError,KeyError) as e:
                exclusions.append(dict(cell_id=key,cycle=n,reason=str(e)))
        assert cycles and cycles[0]==0
        cycles=np.asarray(cycles);soh=np.asarray(capacities)/capacities[0]
        future=cycles>100
        if future.sum()<2:raise ValueError('insufficient future labels: '+key)
        cells[key]=dict(prefix=np.r_[first,second-first],cycle=cycles[future],truth=soh[future],physical_family=FAMILY)
        audit.append(dict(cell_id=key,diagnostic_records=len(records),valid_labels=len(cycles),
            source_cycle_min=int(cycles.min()),source_cycle_max=int(cycles.max()),
            early_diagnostic_source_cycles=[0,100],future_targets=int(future.sum()),
            initial_reference_mAh=capacities[0],protocol_consistency_A_min=float(min(rates)),
            protocol_consistency_A_max=float(max(rates))))
    save_json(out/'audit_report.json',dict(cells=audit,exclusions=exclusions,physical_family=FAMILY,
        manufacturer='Kokam',model='SLPB533459H4',nominal_mAh=740,temperature_C=40,
        chemistry='not specified by supplied official README; no inferred chemistry pooling',
        license='ODbL 1.0 and DbCL 1.0; cite dataset DOI and Birkl 2017 thesis',
        time_units=sorted(units),time_check='absolute timestamp/cadence plus .740A q/duration consistency; q is audit/label only',
        measured_current_available=False,current_input='known .740A C1dc protocol constant, NOT measured I',
        input='V/T first 900 seconds at diagnostics source cycles 0 and 100; known current and chamber temperature',
        label='abs(final q - initial q) / same-cell source-cycle-0 comparable discharge capacity',
        capacity_input_poison_check=True,dense_100_cycle_VI_task_eligible=False,
        sparse_diagnostic_supplement_eligible=True,raw_sha256=sha(RAW/'Oxford_Battery_Degradation_Dataset_1.mat')))
    joblib.dump(cells,out/'prepared_cells.joblib')
    return cells

def matrix(cell,future=None):
    future=cell['cycle'] if future is None else np.asarray(future)
    h=(future-100)/1000.
    p=np.broadcast_to(cell['prefix'],(len(h),len(cell['prefix'])))
    return np.c_[p,h,h*h,np.log1p(h),p*h[:,None]]

def fit(cells,ids,alpha):
    x=np.concatenate([matrix(cells[k]) for k in ids]);y=np.concatenate([cells[k]['truth'] for k in ids])
    w=np.concatenate([np.full(len(cells[k]['truth']),1/len(cells[k]['truth'])) for k in ids]);w*=len(w)/w.sum()
    assert np.isfinite(x).all()
    offset=x[0].copy();scaler=StandardScaler().fit(x-offset,sample_weight=w);scaler.mean_+=offset
    model=Ridge(alpha=alpha,solver='svd').fit(scaler.transform(x),y,sample_weight=w)
    # Independently check training-only weighted preprocessing.
    np.testing.assert_allclose(scaler.mean_,np.average(x,axis=0,weights=w),rtol=1e-10,atol=1e-10)
    return dict(model=model,scaler=scaler,train_ids=list(ids),physical_family=FAMILY,alpha=alpha)

def predict(bundle,cell,future=None):
    assert cell['physical_family']==bundle['physical_family']
    return bundle['model'].predict(bundle['scaler'].transform(matrix(cell,future)))

def evaluate(bundle,cells,ids):
    rows=[];arrays={}
    for key in ids:
        c=cells[key];x,y=c['cycle'],c['truth'];p=predict(bundle,c)
        # Predicting any requested cycles must not require or depend on labels.
        clean={k:v for k,v in c.items() if k not in ('truth','cycle')}
        np.testing.assert_array_equal(p,predict(bundle,clean,x))
        late=x>=(x[0]+x[-1])/2;last=x>x[-1]-100
        rows.append(dict(cell_id=key,dataset='Oxford',origin=100,future_targets=len(y),
            metrics=regression_metrics(y,p),late_half=regression_metrics(y[late],p[late]),
            last100=regression_metrics(y[last],p[last]),last100_observed_targets=int(last.sum()),
            endpoint_abs_soh_pp=float(abs(y[-1]-p[-1])*100)))
        arrays.update({key+'__100__'+f:a for f,a in [('cycle',x),('truth',y),('prediction',p)]})
    return rows,arrays

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    out=Path('experiments/oxford_sparse_diagnostic')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');out.mkdir(parents=True)
    shutil.copy2(__file__,out/Path(__file__).name)
    logging.basicConfig(level=logging.INFO,handlers=[logging.FileHandler(out/'training.log',encoding='utf-8'),logging.StreamHandler()])
    try:
        cells=prepare(out);ids=sorted(cells);plans=[]
        for fold,(dev,test) in enumerate(KFold(5,shuffle=True,random_state=20261004).split(ids),1):
            development=[ids[i] for i in dev]
            plans.append(dict(name=f'Oxford_fold{fold}',train=development[:-1],validation=development[-1:],test=[ids[i] for i in test]))
        seeds=list(range(20261004,20261009))
        save_json(out/'config.json',dict(plans=plans,seeds=seeds,alphas=ALPHAS,smoke=args.smoke,
            source_sha256=sha(__file__),origin=100,input_diagnostics=[0,100],
            selection='single inner validation cell only; fixed outer 5fold',
            interpretation='separate sparse diagnostic supplement; known constant I, not dense 100-cycle measured V/I',
            deterministic_repetition=True))
        rows=[]
        for plan in plans[:1] if args.smoke else plans:
            assert not set(plan['train']+plan['validation'])&set(plan['test'])
            for seed in seeds[:1] if args.smoke else seeds:
                dest=out/plan['name']/str(seed);dest.mkdir(parents=True);scores=[]
                for i,alpha in enumerate(ALPHAS):
                    folder=dest/f'candidate_{i}';folder.mkdir();bundle=fit(cells,plan['train'],alpha)
                    report,arrays=evaluate(bundle,cells,plan['validation'])
                    joblib.dump(bundle,folder/'model.joblib');save_json(folder/'validation_metrics.json',report)
                    np.savez_compressed(folder/'validation_predictions.npz',**arrays)
                    scores.append(dict(alpha=alpha,validation_mape=float(np.mean([r['metrics']['mape_percent'] for r in report]))))
                selected=min(scores,key=lambda r:r['validation_mape'])
                save_json(dest/'selection.json',dict(candidates=scores,selected=selected))
                bundle=fit(cells,plan['train']+plan['validation'],selected['alpha'])
                joblib.dump(bundle,dest/'model.joblib');report,arrays=evaluate(bundle,cells,plan['test'])
                save_json(dest/'metrics.json',report);np.savez_compressed(dest/'predictions.npz',**arrays)
                rows.extend([dict(r,seed=seed,fold=plan['name']) for r in report])
            logging.info('%s saved',plan['name'])
        means=[float(np.mean([r['metrics']['mape_percent'] for r in rows if r['seed']==s])) for s in sorted({r['seed'] for r in rows})]
        if not args.smoke:
            for seed in seeds:assert sorted(r['cell_id'] for r in rows if r['seed']==seed)==ids
        save_json(out/'by_cell.json',rows)
        save_json(out/'summary.json',dict(complete=True,smoke=args.smoke,cells=len({r['cell_id'] for r in rows}),
            mape_mean=float(np.mean(means)),mape_sd=float(np.std(means,ddof=1)) if len(means)>1 else None,
            cells_le6_fraction=float(np.mean([r['metrics']['mape_percent']<=6 for r in rows])),
            deterministic_repetition=True,role='public-development sparse diagnostic forecast',curves_status='pending'))
        print('OUTPUT='+str(out.resolve()),'MAPE='+str(np.mean(means)),flush=True)
    except BaseException:
        save_json(out/'failure.json',dict(traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
