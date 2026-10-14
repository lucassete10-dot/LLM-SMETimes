"""Verify frozen NASA Ridge test coverage, metrics and future-input isolation."""
from datetime import datetime,timezone
import argparse
import json
from pathlib import Path
import shutil
import traceback
import joblib
import numpy as np
from battery_smetimes.external_soh import load_context
from battery_smetimes.research_baselines import predict_baseline,regression_metrics,build_training_pairs
from scripts.prepare_same_type_forecast import save_json,sha

RUN=Path('experiments/nasa_signal_trajectory/ridge/20261004T143636_110096Z')
RELEASE=Path('deployment/stm32_release_trees/20261004T083706_476897Z')
OUTPUT=None


def main():
    global OUTPUT,RUN
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,default=RUN);parser.add_argument('--bounded-prefix',action='store_true');parser.add_argument('--signal-trajectory',action='store_true');args=parser.parse_args()
    if args.bounded_prefix and args.signal_trajectory:raise ValueError('choose one predictor arm')
    RUN=args.run
    if args.signal_trajectory:
        from scripts.train_nasa_signal_trajectory import predict as predictor, inputs
    elif args.bounded_prefix:
        from scripts.train_nasa_bounded_prefix import predict as predictor
    else:predictor=predict_baseline
    out=Path('experiments/nasa_ridge_verification')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    OUTPUT=out
    out.mkdir(parents=True);shutil.copy2(__file__,out/Path(__file__).name)
    cfg=json.loads((RUN/'config.json').read_text());cells=load_context(RELEASE)['cells'];rows=[]
    for plan in cfg['plans']:
        for seed in cfg['seeds']:
            folder=RUN/plan['name']/str(seed);bundle=joblib.load(folder/'model.joblib')
            assert set(bundle['train_ids'])==set(plan['train']+plan['validation'])
            assert not set(bundle['train_ids'])&set(plan['test'])
            selection=json.loads((folder/'selection.json').read_text())
            assert selection['selected']==min(selection['candidates'],key=lambda r:r['validation_mape'])
            if args.signal_trajectory:
                xs,ws=[],[]
                for key in bundle['train_ids']:
                    c=cells[key];cyc,features=inputs(c)
                    label_cycles={int(n) for n,y in zip(c['label_cycle'],c['soh']) if np.isfinite(y) and y>0 and n<=100}
                    take=np.array([int(n) in label_cycles for n in cyc])
                    xs.append(features[take]);ws.append(np.full(int(take.sum()),1/len(bundle['train_ids'])/take.sum()))
                x,w=np.concatenate(xs),np.concatenate(ws)
            else:x,_,w,_=build_training_pairs(cells,bundle['train_ids'],[100])
            # Original Ridge imputer computes medians after conversion to float64.
            fill=np.array([np.nanmedian(c) if np.isfinite(c).any() else 0. for c in x.astype(np.float64).T])
            np.testing.assert_allclose(fill,bundle['imputer'].statistics_,rtol=0,atol=0)
            transformed=bundle['imputer'].transform(x.astype(float))
            mean=np.average(transformed,axis=0,weights=w)
            variance=np.average((transformed-mean)**2,axis=0,weights=w)
            np.testing.assert_allclose(mean,bundle['scaler'].mean_,rtol=1e-9,atol=1e-9)
            np.testing.assert_allclose(variance,bundle['scaler'].var_,rtol=1e-9,atol=1e-9)
            report=json.loads((folder/'metrics.json').read_text());assert not report['skipped']
            assert {r['cell_id'] for r in report['by_cell']}==set(plan['test'])
            with np.load(folder/'predictions.npz',allow_pickle=False) as arrays:
                for row in report['by_cell']:
                    key=row['cell_id'];c=cells[key];prefix=key+'__100__'
                    cycles,y,p=[arrays[prefix+f] for f in ('cycle','truth','prediction')]
                    valid=np.isfinite(c['soh'])&(c['soh']>0)&(c['label_cycle']>100)
                    np.testing.assert_array_equal(cycles,c['label_cycle'][valid])
                    np.testing.assert_array_equal(y,c['soh'][valid])
                    for k,v in regression_metrics(y,p).items():np.testing.assert_allclose(v,row['metrics'][k],rtol=0,atol=1e-12)
                    clean={k:v for k,v in c.items() if k not in ('soh','label_cycle','unfiltered_soh')}
                    np.testing.assert_allclose(p,predictor(bundle,clean,100,cycles),rtol=0,atol=0)
                    poisoned=dict(clean,cycle=np.r_[clean['cycle'],101,102],raw=np.vstack([clean['raw'],np.full((2,44),np.inf)]))
                    np.testing.assert_allclose(p,predictor(bundle,poisoned,100,cycles),rtol=0,atol=0)
                    row.update(seed=seed,fold=plan['name'],last100=regression_metrics(y[cycles>cycles[-1]-100],p[cycles>cycles[-1]-100]),
                        endpoint_abs_soh_pp=float(abs(y[-1]-p[-1])*100),model_sha256=sha(folder/'model.joblib'))
                    rows.append(row)
    means=[np.mean([r['metrics']['mape_percent'] for r in rows if r['seed']==s]) for s in cfg['seeds']]
    expected=json.loads((RUN/'summary.json').read_text());np.testing.assert_allclose(np.mean(means),expected['mape_mean'],atol=1e-12,rtol=0)
    save_json(out/'verification.json',dict(passed=True,units=len(rows),cells=len({r['cell_id'] for r in rows}),
        origin=100,mape_mean=float(np.mean(means)),mape_seed_sd=float(np.std(means,ddof=1)),
        deterministic_repetition=True,full_valid_future_coverage=True,label_removal=True,future_poison=True,
        training_medians_recomputed=True,training_weighted_scaler_recomputed=True,
        bounded_prefix=args.bounded_prefix,signal_trajectory=args.signal_trajectory,curve_export='pending',rows=rows,
        run=str(RUN.resolve()),run_config_sha256=sha(RUN/'config.json'),
        role='public development LOBO, not new independent-cell or final visual acceptance'))
    print('OUTPUT='+str(out.resolve()),'PASS units='+str(len(rows)))


if __name__=='__main__':
    try:main()
    except BaseException:
        if OUTPUT is not None:save_json(OUTPUT/'failure.json',dict(traceback=traceback.format_exc()))
        raise
