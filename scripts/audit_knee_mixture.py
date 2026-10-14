"""Reproduce direct parameter regression, then audit nonlinear curve mixture."""
import argparse,json
from datetime import datetime,timezone
from pathlib import Path
import joblib,numpy as np
from sklearn.impute import SimpleImputer
from battery_smetimes.knee_forecast import early_features,fit_knee,predict,trajectory
from battery_smetimes.curve_fidelity import targets,curve_metrics,selection_score
from scripts.train_curve_fidelity import load_subset
from scripts.prepare_same_type_forecast import save_json,sha


def weights(bundle,cells,cell):
    ids=bundle['train_ids']; windows=bundle['parameters']['windows']
    x=np.stack([early_features(cells[k],windows) for k in ids])
    leaves=bundle['model'].apply(bundle['imputer'].transform(x))
    query=bundle['model'].apply(bundle['imputer'].transform(early_features(cell,windows)[None]))[0]
    matches=leaves==query[None]
    w=(matches/np.maximum(matches.sum(axis=0),1)[None]).mean(axis=1)
    np.testing.assert_allclose(w.sum(),1,atol=1e-12,rtol=0)
    return w


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    run=parser.parse_args().run;cfg=json.loads((run/'config.json').read_text())
    for name,digest in cfg['source_sha256'].items():
        assert sha(run/'source_snapshot'/name)==digest
    out=Path('experiments/knee_mixture_audit')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True); reports=[];checked=0
    for plan in cfg['plans']:
        cells=load_subset(plan['train']+plan['validation'])
        assert not set(cells)&set(plan['test'])
        for i,parameters in enumerate(cfg['candidates']):
            folder=run/plan['name']/f'candidate_{i}';b=joblib.load(folder/'model.joblib')
            assert b['train_ids']==plan['train'] and b['parameters']==parameters
            expected=np.stack([fit_knee(*targets(cells[k])) for k in plan['train']])
            np.testing.assert_allclose(expected,b['training_knee_parameters'],atol=1e-12,rtol=0)
            np.testing.assert_allclose(np.maximum(expected.std(axis=0),.001),b['target_scale'],atol=1e-12,rtol=0)
            train_x=np.stack([early_features(cells[k],parameters['windows']) for k in plan['train']])
            medians=SimpleImputer(strategy='median',keep_empty_features=True).fit(train_x).statistics_
            np.testing.assert_allclose(medians,b['imputer'].statistics_,atol=1e-12,rtol=0)
            oracle=[]
            for j,k in enumerate(plan['train']):
                x,y=targets(cells[k]);p=trajectory(expected[j],x)
                oracle.append(dict(cell_id=k,**curve_metrics(x,y,p)))
            rows=[];arrays={}; old_metrics=json.loads((folder/'validation_metrics.json').read_text())
            with np.load(folder/'validation_predictions.npz') as old:
                for key in plan['validation']:
                    c=cells[key];x,y=targets(c)
                    clean={k:v for k,v in c.items() if k not in ('soh','label_cycle','reference_capacity_ah')}
                    direct=predict(b,clean,x)
                    np.testing.assert_allclose(direct,old[key+'__100__prediction'],atol=1e-12,rtol=0)
                    np.testing.assert_array_equal(x,old[key+'__100__cycle']);np.testing.assert_array_equal(y,old[key+'__100__truth'])
                    w=weights(b,cells,clean)
                    # Exact forest parameter mean reconstructed from training membership.
                    forest=b['model'].predict(b['imputer'].transform(early_features(clean,parameters['windows'])[None]))[0]*b['target_scale']
                    np.testing.assert_allclose(w@expected,forest,atol=1e-10,rtol=0)
                    curves=np.stack([trajectory(coef,x) for coef in expected])
                    p=w@curves
                    poison=dict(clean,cycle=np.r_[clean['cycle'],101,10000],
                        raw=np.vstack([clean['raw'],np.full((2,clean['raw'].shape[1]),np.inf)]))
                    np.testing.assert_array_equal(w,weights(b,cells,poison))
                    final=np.array([targets(cells[k])[0][-1] for k in plan['train']])
                    unsupported=float(w[final<x[-1]].sum())
                    row=dict(cell_id=key,**curve_metrics(x,y,p),training_tail_extrapolation_weight=unsupported)
                    rows.append(row);checked+=1
                    arrays.update({key+'__100__'+f:a for f,a in [('cycle',x),('truth',y),('prediction',p)]})
            dest=out/plan['name']/f'candidate_{i}';dest.mkdir(parents=True)
            save_json(dest/'validation_metrics.json',rows);np.savez_compressed(dest/'validation_predictions.npz',**arrays)
            save_json(dest/'training_fit_metrics.json',oracle)
            report=dict(fold=plan['name'],candidate=i,parameters=parameters,
                direct_score=selection_score(old_metrics),mixture_score=selection_score(rows),
                training_fit_mean_mape=float(np.mean([r['mape_percent'] for r in oracle])),
                **{k:float(np.mean([r[k] for r in rows])) for k in ('mape_percent','late_mape','endpoint_pp','drop_error_pp','joint_pass','training_tail_extrapolation_weight')})
            reports.append(report);print(json.dumps(report),flush=True)
    save_json(out/'audit.json',dict(passed=True,source_run=str(run.resolve()),
        source_config_sha256=sha(run/'config.json'),source_sha256=sha(__file__),reports=reports,
        checked_validation_curves=checked,outer_test_loaded=False,goal_complete=False,
        note='Fixed inference diagnosis on eight archived models; not full CV or deployment adoption.'))
    print('OUTPUT='+str(out.resolve()))


if __name__=='__main__':main()
