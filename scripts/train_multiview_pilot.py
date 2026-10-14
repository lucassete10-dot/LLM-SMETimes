"""Two registered configurations, fixed MIT inner folds, no outer-cell loading."""
import argparse
import json
import shutil
import traceback
from pathlib import Path
from datetime import datetime,timezone
import joblib
import numpy as np
from battery_smetimes.multiview_pilot import fit,evaluate,predict,manifest
from battery_smetimes.curve_fidelity import selection_score,curve_metrics
from scripts.train_curve_fidelity import load_subset
from scripts.prepare_same_type_forecast import save_json,sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    source=Path('experiments/mit_feature_blend_outer/20261006T144031_590137Z')
    cfg=json.loads((source/'config.json').read_text());inner=Path(cfg['inner'])
    assert sha(inner/'config.json')==cfg['inner_config_sha256']
    for directory in ('experiments/goal_v2_raw_features/20261007T104646_881583Z',
                      'experiments/goal_v2_nca_raw_features/20261007T105630_854577Z'):
        assert json.loads((Path(directory)/'report.json').read_text())['passed']
    plans=cfg['plans'][:1] if args.smoke else cfg['plans'];seeds=cfg['seeds'][:1] if args.smoke else cfg['seeds']
    epochs=20 if args.smoke else 200
    root=Path('experiments/multiview_pilot')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');root.mkdir(parents=True)
    print('RUN='+str(root.resolve()),flush=True)
    snap=root/'source_snapshot';snap.mkdir()
    files=[Path(__file__),Path('battery_smetimes/multiview_pilot.py'),Path('battery_smetimes/mit_segment_features.py'),
           Path('battery_smetimes/curve_fidelity.py'),Path('docs/soh_multiview_pilot_registration_20261007.md')]
    for f in files:shutil.copy2(f,snap/f.name)
    save_json(root/'config.json',dict(plans=plans,seeds=seeds,epochs=epochs,smoke=args.smoke,modes=['concat','gate'],
              source_sha256={str(f):sha(f) for f in files},baseline_inner=str(inner.resolve()),
              baseline_config_sha256=sha(inner/'config.json'),input_manifest=manifest(),outer_loaded=False,
              registration_sha256=sha(files[-1]),goal_complete=False))
    reports=[]
    try:
        for plan in plans:
            cells=load_subset(plan['train']+plan['validation']);assert not set(cells)&set(plan['test'])
            for seed in seeds:
                previous=inner/plan['name']/str(seed);selection=json.loads((previous/'selection.json').read_text())['selected']
                baseline_file=previous/f"candidate_{selection['candidate']}"/'validation_predictions.npz'
                baseline_rows=[]
                with np.load(baseline_file) as a:
                    for k in plan['validation']:
                        c,y,p=[a[k+'__100__'+f] for f in ('cycle','truth','prediction')]
                        baseline_rows.append(dict(cell_id=k,**curve_metrics(c,y,p)))
                baseline_score=selection_score(baseline_rows)
                np.testing.assert_allclose(baseline_score,selection['score'],rtol=0,atol=1e-12)
                for mode in ('concat','gate'):
                    dest=root/plan['name']/str(seed)/mode;dest.mkdir(parents=True)
                    b,history=fit(cells,plan['train'],plan['validation'],seed,mode,epochs)
                    joblib.dump(b,dest/'model.joblib');save_json(dest/'history.json',history)
                    for role in ('train','validation'):
                        rows,arrays=evaluate(b,cells,plan[role]);save_json(dest/(role+'_metrics.json'),rows)
                        np.savez_compressed(dest/(role+'_predictions.npz'),**arrays)
                        for k in plan[role]:
                            c=arrays[k+'__100__cycle'];p=arrays[k+'__100__prediction'];q=dict(cells[k],raw=cells[k]['raw'].copy())
                            q.update(soh=np.inf,label_cycle=np.inf,reference_capacity_ah=np.inf,cycle_life=np.inf)
                            q['cycle']=np.r_[q['cycle'],100000];q['raw']=np.vstack([q['raw'],np.full((1,51),np.inf)])
                            q['raw'][:,[38,39]]=np.inf
                            np.testing.assert_array_equal(p,predict(b,q,c))
                            try:predict(b,dict(cells[k],physical_family='WRONG'),c)
                            except ValueError:pass
                            else:raise AssertionError('family gate failed')
                    validation=json.loads((dest/'validation_metrics.json').read_text())
                    with np.load(baseline_file) as a,np.load(dest/'validation_predictions.npz') as new:
                        for k in plan['validation']:
                            for f in ('cycle','truth'):np.testing.assert_array_equal(a[k+'__100__'+f],new[k+'__100__'+f])
                    row=dict(fold=plan['name'],seed=seed,mode=mode,score=selection_score(validation),
                             baseline_score=baseline_score,baseline_array_sha256=sha(baseline_file),
                             best_epoch=b['best_epoch'],parameter_count=b['parameter_count'],fit_seconds=b['fit_seconds'])
                    reports.append(row);save_json(dest/'summary.json',row);save_json(root/'task_results.json',reports)
                    save_json(root/'progress.json',dict(terminal=False,models=len(reports),outer_loaded=False))
                    print(json.dumps(row),flush=True)
        summaries=[]
        for mode in ('concat','gate'):
            selected=[r for r in reports if r['mode']==mode]
            means=[np.mean([r['score'] for r in selected if r['seed']==seed]) for seed in seeds]
            base=[np.mean([r['baseline_score'] for r in selected if r['seed']==seed]) for seed in seeds]
            summaries.append(dict(mode=mode,score_mean=float(np.mean(means)),score_seed_sd=float(np.std(means,ddof=1)) if len(seeds)>1 else None,
                                  baseline_score_mean=float(np.mean(base)),improved=bool(np.mean(means)<np.mean(base))))
        save_json(root/'summary.json',dict(candidates=summaries,smoke=args.smoke,outer_expansion_authorized=False,
                                         independent_verification_pending=True,goal_complete=False))
        save_json(root/'progress.json',dict(terminal=True,passed=True,models=len(reports),outer_loaded=False))
    except BaseException:
        save_json(root/'failure.json',dict(traceback=traceback.format_exc(),models=len(reports)))
        save_json(root/'progress.json',dict(terminal=True,passed=False,models=len(reports),outer_loaded=False))
        raise


if __name__=='__main__':main()
