"""Recompute all held-out metrics and inspect every saved CV model membership."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import traceback
import joblib
import numpy as np
from battery_smetimes.research_baselines import regression_metrics
from scripts.prepare_same_type_forecast import save_json, sha

RUNS = {
    'compact_tree': Path('experiments/stm32_compact_trees/runs/20261004T074429_087457Z'),
    'compact_mlp': Path('experiments/stm32_soh_multiseed/runs/20261004T075651_692988Z'),
}


def check(condition, message):
    if not condition:
        raise ValueError(message)


def audit(run, kind):
    cfg = json.loads((run/'config.json').read_text())
    plans = json.loads((run/'splits.json').read_text())['plans']
    frozen_plans = json.loads((run/'frozen_audit/splits.json').read_text())['plans']
    check(plans==frozen_plans, 'CV splits differ from frozen audit')
    records = json.loads((run/'frozen_audit/cell_registry.json').read_text())['cells']
    progress = json.loads((run/'progress.json').read_text())
    seeds = cfg['seeds']
    check(len(seeds)>=5 and len(set(seeds))==len(seeds), 'insufficient distinct seeds')
    expected_jobs = {(p['name'],s) for p in plans for s in seeds}
    jobs = [(j['fold'],j['seed']) for j in progress['completed']]
    check(progress['complete'] and set(jobs)==expected_jobs and len(jobs)==len(expected_jobs),
          'completed units do not match required folds and seeds')
    fingerprints, rows, queries = {}, [], 0
    source_checks=[]
    for name,digest in cfg['source_sha256'].items():
        normalized=name.replace('\\','/')
        relative=Path('scripts')/Path(normalized).name if ':' in normalized else Path(normalized)
        path=run/'source_snapshot'/relative
        check(path.exists() and sha(path)==digest, f'source snapshot mismatch {name}')
        source_checks.append(str(relative))
    modes = ['test'] if kind=='compact_tree' else ['float','int8']
    for plan in plans:
        groups=[set(plan[k]) for k in ('train','validation','test')]
        a,b,t=groups
        check(not (a&b or a&t or b&t), 'cell-role overlap')
        check(a|b|t=={k for k,r in records.items() if r['metadata']['dataset']==plan['dataset']},
              'fold does not cover complete dataset')
        families={records[k]['physical_family'] for k in a|b|t}
        check(len(families)==1, 'physical families mixed')
        for seed in seeds:
            dest=run/plan['name']/str(seed)
            bundle=joblib.load(dest/'model.joblib')
            check(set(bundle['train_ids'])==a|b, 'refit model membership differs from development cells')
            check(bundle['seed']==seed and bundle['physical_family'] in families,
                  'model seed/family mismatch')
            selection=json.loads((dest/'selection.json').read_text())
            check(selection['selected']==min(selection['candidates'],key=lambda r:r['validation_mape']),
                  'selection differs from lowest inner-validation score')
            if kind=='compact_tree':
                for i,candidate in enumerate(selection['candidates']):
                    cd=dest/f'candidate_{i}'
                    inner=joblib.load(cd/'model.joblib')
                    check(set(inner['train_ids'])==a, 'inner candidate used validation/test cells')
                    vr=json.loads((cd/'validation_metrics.json').read_text())
                    check({r['cell_id'] for r in vr['by_cell']}==b, 'validation membership mismatch')
                    np.testing.assert_allclose(candidate['validation_mape'],
                        vr['summary']['100']['cell_macro_mape_percent'],rtol=0,atol=1e-12)
                    fingerprints[str(cd.relative_to(run))+'/model.joblib']=sha(cd/'model.joblib')
            fingerprints[str(dest.relative_to(run))+'/model.joblib']=sha(dest/'model.joblib')
            for mode in modes:
                folder=dest/mode
                report=json.loads((folder/'metrics.json').read_text())
                check(not report['skipped'], 'held-out cells skipped: requires explicit review')
                check(len(report['by_cell'])==len(t) and {r['cell_id'] for r in report['by_cell']}==t,
                      'held-out coverage mismatch')
                check((folder/'curves_origin100.png').is_file(), 'held-out curve artifact missing')
                with np.load(folder/'predictions.npz',allow_pickle=False) as arrays:
                    check(set(arrays.files)=={f'{k}__100__{f}' for k in t for f in ('cycle','truth','prediction')},
                          'held-out prediction arrays membership mismatch')
                    for row in report['by_cell']:
                        check(row['origin']==100 and row['physical_family'] in families,'target contract mismatch')
                        prefix=row['cell_id']+'__100__'
                        x,y,p=[arrays[prefix+f] for f in ('cycle','truth','prediction')]
                        check(len(x)==row['future_targets'] and np.all(x>100) and np.all(np.diff(x)>0),
                              'invalid target source cycles')
                        check(np.isfinite(y).all() and np.isfinite(p).all() and (y>0).all(),
                              'invalid predictions/truth')
                        metrics=regression_metrics(y,p)
                        for field,value in metrics.items():
                            np.testing.assert_allclose(value,row['metrics'][field],rtol=0,atol=1e-12)
                        rows.append(dict(dataset=plan['dataset'],seed=seed,fold=plan['name'],
                                         cell_id=row['cell_id'],mode=mode,**metrics))
                        queries+=len(x)
                    macro=float(np.mean([r['metrics']['mape_percent'] for r in report['by_cell']]))
                    np.testing.assert_allclose(macro,report['summary']['100']['cell_macro_mape_percent'],rtol=0,atol=1e-12)
                for name in ('metrics.json','predictions.npz','curves_origin100.png'):
                    fingerprints[str((folder/name).relative_to(run))]=sha(folder/name)
    summaries={}
    saved_summary=json.loads((run/'summary.json').read_text())
    for domain in ('MIT','XJTU','NASA'):
        for mode in modes:
            subset=[r for r in rows if r['dataset']==domain and r['mode']==mode]
            wanted={k for k,r in records.items() if r['metadata']['dataset']==domain}
            for s in seeds:
                ids=[r['cell_id'] for r in subset if r['seed']==s]
                check(set(ids)==wanted and len(ids)==len(wanted), 'cell not tested exactly once per seed')
            summaries[domain+'/'+mode]={}
            for metric in ('mape_percent','rmse_soh','mae_soh','r2'):
                means=[np.mean([r[metric] for r in subset if r['seed']==s]) for s in seeds]
                summaries[domain+'/'+mode][metric]=dict(mean=float(np.mean(means)),sd=float(np.std(means,ddof=1)))
                expected=saved_summary[domain][metric] if kind=='compact_tree' else saved_summary[domain][mode][metric]
                for statistic in ('mean','sd'):
                    np.testing.assert_allclose(summaries[domain+'/'+mode][metric][statistic],
                                               expected[statistic],rtol=0,atol=1e-12)
    return dict(passed=True,run=str(run.resolve()),jobs=len(jobs),cell_mode_seed_records=len(rows),
        future_query_records=queries,summary_recomputed=summaries,source_snapshots_verified=source_checks,
        files_sha256=fingerprints,
        scope='Saved memberships, frozen splits, inner tree candidates, validation-score selection, all held-out arrays and four metrics. Not a causal proof of every historical fitting operation, human figure QA, hardware runtime, or independent new-cell accuracy.')


if __name__=='__main__':
    out=Path('experiments/stm32_cv_artifact_audit')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True);shutil.copy2(__file__,out/Path(__file__).name)
    try:
        results={}
        for kind,run in RUNS.items():
            result=audit(run,kind);save_json(out/(kind+'.json'),result);results[kind]=result
            print(kind,'PASS',result['jobs'],result['cell_mode_seed_records'],flush=True)
        save_json(out/'progress.json',dict(complete=True,passed=True,runs=list(results)))
        print('OUTPUT='+str(out.resolve()))
    except BaseException:
        save_json(out/'failure.json',dict(traceback=traceback.format_exc()))
        raise
