"""Four-candidate inner-only knee smoke on frozen MIT/XJTU first folds."""
from datetime import datetime,timezone
import json,logging,shutil,traceback
from pathlib import Path
import joblib
import numpy as np
from battery_smetimes.knee_forecast import fit,predict
from battery_smetimes.curve_fidelity import targets,curve_metrics,selection_score
from scripts.train_curve_fidelity import AUDIT,load_subset
from scripts.prepare_same_type_forecast import save_json,sha


def main():
    out=Path('experiments/knee_forecast')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True); snap=out/'source_snapshot';snap.mkdir()
    sources=[Path(__file__),Path('battery_smetimes/knee_forecast.py'),
        Path('battery_smetimes/curve_fidelity.py'),Path('battery_smetimes/research_baselines.py'),
        Path('battery_smetimes/forecast.py'),Path('scripts/train_curve_fidelity.py')]
    for p in sources:shutil.copy2(p,snap/p.name)
    shutil.copy2('docs/soh_knee_stage_20261005.md',out/'plan.md')
    plans=[p for p in json.loads((AUDIT/'splits.json').read_text())['plans'] if p['name'] in ('MIT_fold1','XJTU_fold1')]
    candidates=[dict(windows=w,leaf=l) for w in (False,True) for l in (1,3)]
    registry=json.loads((AUDIT/'cell_registry.json').read_text())['cells']
    loaded={k for p in plans for k in p['train']+p['validation']}
    save_json(out/'config.json',dict(plans=plans,candidates=candidates,seeds=[20261005],smoke=True,
        source_sha256={p.name:sha(p) for p in sources},
        prepared_sha256={k:registry[k]['prepared_sha256'] for k in loaded},
        splits_sha256=sha(AUDIT/'splits.json'),outer_test_loaded=False,
        note='training parameter supervision only; fixed selection_score; no lifespan input'))
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',
        handlers=[logging.FileHandler(out/'training.log',encoding='utf-8'),logging.StreamHandler()])
    summaries=[]
    try:
        for plan in plans:
            cells=load_subset(plan['train']+plan['validation'])
            assert not set(cells)&set(plan['test'])
            for i,parameters in enumerate(candidates):
                logging.info('TRAIN %s candidate=%s %s',plan['name'],i,parameters)
                folder=out/plan['name']/f'candidate_{i}';folder.mkdir(parents=True)
                b=fit(cells,plan['train'],20261005,parameters);joblib.dump(b,folder/'model.joblib')
                rows=[];arrays={}
                for key in plan['validation']:
                    c=cells[key];cycle,y=targets(c)
                    clean={k:v for k,v in c.items() if k not in ('soh','label_cycle','reference_capacity_ah')}
                    p=predict(b,clean,cycle)
                    poison=dict(clean,cycle=np.r_[clean['cycle'],101,10000],
                        raw=np.vstack([clean['raw'],np.full((2,clean['raw'].shape[1]),np.inf)]),cycle_life=np.inf)
                    np.testing.assert_allclose(p,predict(b,poison,cycle),atol=1e-12,rtol=0)
                    try:predict(b,dict(clean,physical_family='different'),cycle)
                    except ValueError:pass
                    else:raise AssertionError('family gate failed')
                    rows.append(dict(cell_id=key,**curve_metrics(cycle,y,p)))
                    arrays.update({key+'__100__'+f:a for f,a in [('cycle',cycle),('truth',y),('prediction',p)]})
                save_json(folder/'validation_metrics.json',rows)
                np.savez_compressed(folder/'validation_predictions.npz',**arrays)
                summary=dict(fold=plan['name'],candidate=i,parameters=parameters,selection_score=selection_score(rows),
                    **{k:float(np.mean([r[k] for r in rows])) for k in ('mape_percent','late_mape','endpoint_pp','drop_error_pp','joint_pass')})
                summaries.append(summary);save_json(out/'summary.json',summaries)
                logging.info('SAVED %s candidate=%s score=%.3f MAPE=%.3f endpoint=%.3f drop=%.3f',
                    plan['name'],i,summary['selection_score'],summary['mape_percent'],summary['endpoint_pp'],summary['drop_error_pp'])
            choices=[r for r in summaries if r['fold']==plan['name']]
            save_json(out/plan['name']/'selection.json',dict(selected=min(choices,key=lambda r:r['selection_score']),
                candidates=choices,used_only_validation=True))
        save_json(out/'progress.json',dict(training_complete=True,smoke=True,goal_complete=False,
            outer_evaluation_complete=False,label_removal_and_future_poison_passed=True))
        print('OUTPUT='+str(out.resolve()))
    except BaseException:
        save_json(out/'failure.json',dict(traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
