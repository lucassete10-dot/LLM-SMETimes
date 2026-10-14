"""Preregistered NASA inner-validation smoke; never loads outer cells."""
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import shutil
import traceback
import joblib
import numpy as np
from battery_smetimes.curve_state_transfer import fit, predict
from battery_smetimes.curve_fidelity import targets, curve_metrics, selection_score
from scripts.train_curve_fidelity import load_subset, AUDIT
from scripts.prepare_same_type_forecast import save_json, sha


def main():
    out = Path('experiments/curve_state_transfer')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True)
    files = [Path(__file__),Path('battery_smetimes/curve_state_transfer.py'),
        Path('battery_smetimes/curve_fidelity.py'),Path('battery_smetimes/research_baselines.py'),
        Path('battery_smetimes/forecast.py'),Path('scripts/train_curve_fidelity.py')]
    snapshot=out/'source_snapshot'; snapshot.mkdir()
    for p in files:
        shutil.copy2(p,snapshot/p.name)
    shutil.copy2('docs/soh_curve_state_transfer_stage_20261005.md',out/'plan.md')
    plans=[p for p in json.loads((AUDIT/'splits.json').read_text())['plans'] if p['dataset']=='NASA']
    candidates=[dict(state_alpha=a,fade_alpha=b) for a in (.1,10.,100.) for b in (10.,100.)]
    save_json(out/'config.json',dict(plans=plans,candidates=candidates,seed=20261005,smoke=True,
        source_sha256={p.name:sha(p) for p in files},splits_sha256=sha(AUDIT/'splits.json'),
        scope='inner validation only; no outer cells loaded; deterministic models',
        selection='same fixed curve-fidelity selection_score'))
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',
        handlers=[logging.FileHandler(out/'training.log',encoding='utf-8'),logging.StreamHandler()])
    summaries=[]
    try:
        for plan in plans:
            cells=load_subset(plan['train']+plan['validation'])
            assert not set(cells)&set(plan['test'])
            for index,parameters in enumerate(candidates):
                logging.info('TRAIN %s candidate=%s %s',plan['name'],index,parameters)
                folder=out/plan['name']/f'candidate_{index}'; folder.mkdir(parents=True)
                bundle=fit(cells,plan['train'],parameters)
                joblib.dump(bundle,folder/'model.joblib')
                rows=[]; arrays={}
                for key in plan['validation']:
                    c=cells[key]; cycle,y=targets(c)
                    clean={k:v for k,v in c.items() if k not in ('soh','label_cycle','reference_capacity_ah')}
                    p=predict(bundle,clean,cycle)
                    poisoned=dict(clean,cycle=np.r_[clean['cycle'],101,10000],
                        raw=np.vstack([clean['raw'],np.full((2,clean['raw'].shape[1]),np.inf)]),
                        future_soh=np.inf,cycle_life=np.inf)
                    np.testing.assert_allclose(p,predict(bundle,poisoned,cycle),rtol=0,atol=1e-12)
                    try:
                        predict(bundle,dict(clean,physical_family='different'),cycle)
                    except ValueError:
                        pass
                    else:
                        raise AssertionError('family gate failed')
                    rows.append(dict(cell_id=key,**curve_metrics(cycle,y,p)))
                    arrays.update({key+'__100__'+f:a for f,a in [('cycle',cycle),('truth',y),('prediction',p)]})
                save_json(folder/'validation_metrics.json',rows)
                np.savez_compressed(folder/'validation_predictions.npz',**arrays)
                summary=dict(fold=plan['name'],candidate=index,parameters=parameters,
                    selection_score=selection_score(rows),metrics=rows)
                summaries.append(summary); save_json(out/'summary.json',summaries)
                logging.info('SAVED %s candidate=%s score=%.3f MAPE=%.3f endpoint=%.3f',
                    plan['name'],index,summary['selection_score'],rows[0]['mape_percent'],rows[0]['endpoint_pp'])
            scores=[r for r in summaries if r['fold']==plan['name']]
            save_json(out/plan['name']/'selection.json',dict(selected=min(scores,key=lambda r:r['selection_score']),
                candidates=scores,original_test_loaded=False))
        save_json(out/'progress.json',dict(training_complete=True,outer_evaluation_complete=False,
            goal_complete=False,input_isolation_checks_passed=True))
        print('OUTPUT='+str(out.resolve()))
    except BaseException:
        save_json(out/'failure.json',dict(traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    main()
