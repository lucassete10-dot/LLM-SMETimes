"""Recompute all four cohort held-out metrics, preserving scope and gate failures."""
import json,shutil
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from scripts.prepare_same_type_forecast import sha,save_json
from scripts.audit_rate_blend_metrics import metrics,summarize,checks

SOURCES=[
    ('MIT','experiments/mit_feature_blend_outer/20261006T144031_590137Z',134,'independent_verification.json',False),
    ('XJTU','experiments/xjtu_segment_outer/20261006T134046_882392Z',55,'independent_verification.json',False),
    ('NASA','experiments/nasa_slope_blend_formal/20261005T163202_144477Z',4,'independent_verification.json',False),
    ('Zhu2022 NCA','experiments/tail_boost_outer/20261005T040409_096593Z',55,'independent_outer_verification.json',True),
]


def main():
    root=Path('experiments/four_cohort_progress_audit')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    root.mkdir(parents=True);print('RUN='+str(root.resolve()),flush=True)
    snap=root/'source_snapshot';snap.mkdir()
    for f in [Path(__file__),Path('scripts/audit_rate_blend_metrics.py')]:shutil.copy2(f,snap/f.name)
    results=[];evidence=[];cell_reports=[]
    for name,path,count,proof_name,retained in SOURCES:
        source=Path(path);cfg=json.loads((source/'config.json').read_text());stored=json.loads((source/'summary.json').read_text())
        proof=json.loads((source/proof_name).read_text());assert proof['passed']
        if 'plans' not in cfg:
            parent=Path(cfg['training_source']);assert sha(parent/'config.json')==cfg['training_config_sha256']
            pc=json.loads((parent/'config.json').read_text());plans=[p for p in pc['plans'] if p['dataset']==cfg['dataset']];seeds=pc['seeds']
            evidence.append(dict(file=str((parent/'config.json').resolve()),sha256=sha(parent/'config.json')))
        else:plans=cfg['plans'];seeds=cfg['seeds']
        assert len(seeds)==5
        allids=[k for p in plans for k in p['test']];assert len(allids)==len(set(allids))==count
        rows=[]
        for plan in plans:
            assert not set(plan['train'])&set(plan['validation']) and not set(plan['train']+plan['validation'])&set(plan['test'])
            for seed in seeds:
                file=source/plan['name']/str(seed)/'test_predictions.npz'
                with np.load(file) as arrays:
                    assert set(arrays.files)=={k+'__100__'+f for k in plan['test'] for f in ('cycle','truth','prediction')}
                    for key in plan['test']:
                        c,y,p=[arrays[key+'__100__'+f] for f in ('cycle','truth','prediction')];e=p-y
                        ss=float(np.sum((y-y.mean())**2))
                        rows.append(dict(cell_id=key,seed=seed,fold=plan['name'],**metrics(c,y,p),
                            mae_soh=float(np.mean(abs(e))),r2=1-float(e@e)/ss if ss else None,
                            r2_undefined=not bool(ss),short_coverage=bool(len(c)<10 or c[-1]-c[0]<100),
                            future_points=len(c),future_span=float(c[-1]-c[0])))
                evidence.append(dict(file=str(file.resolve()),sha256=sha(file)))
        recomputed,bycell=summarize(rows)
        for f,value in recomputed.items():
            if isinstance(value,dict):
                for k,v in value.items():np.testing.assert_allclose(v,stored['new'][f][k],rtol=0,atol=1e-10)
            else:np.testing.assert_allclose(value,stored['new'][f],rtol=0,atol=1e-10)
        gates=checks(recomputed,stored['baseline']);assert gates==stored['gates'] and all(gates.values())==stored['numeric_gates_passed']
        additional={}
        for field in ('mae_soh','r2'):
            seed_means=[np.mean([r[field] for r in rows if r['seed']==s and r[field] is not None]) for s in seeds]
            additional[field]=dict(mean=float(np.mean(seed_means)),seed_sd=float(np.std(seed_means,ddof=1)))
        result=dict(dataset=name,run=str(source.resolve()),currently_retained=retained,
            selection_status='retained but incomplete goal' if retained else 'audited full development-CV candidate, not final adopted',
            summary=recomputed,additional_metrics=additional,gates=gates,passed_gates=sum(gates.values()),
            failing_gates=[k for k,v in gates.items() if not v],joint_cells=sum(r['joint_pass'] for r in bycell),
            short_coverage_cells=len({r['cell_id'] for r in rows if r['short_coverage']}),
            r2_undefined_cell_seed_records=sum(r['r2_undefined'] for r in rows),
            repeat_interpretation='deterministic repetitions, not independent seeds or CI' if name=='NASA' else '5 training seeds; cohort macro mean ± sample seed SD; not CI')
        results.append(result);cell_reports.append(dict(dataset=name,rows=rows,by_cell=bycell))
        for f in ('config.json','summary.json',proof_name):evidence.append(dict(file=str((source/f).resolve()),sha256=sha(source/f)))
        print(name+' '+str(sum(gates.values()))+'/10 gates; joint '+str(result['joint_cells'])+'/'+str(count),flush=True)
    save_json(root/'config.json',dict(sources=SOURCES,source_sha256=sha(__file__),metrics_source_sha256=sha('scripts/audit_rate_blend_metrics.py'),
        operation='reporting only, no model selection or fitting',goal_complete=False))
    save_json(root/'source_evidence.json',evidence);save_json(root/'cohort_results.json',results);save_json(root/'cell_results.json',cell_reports)
    table=['# Four-cohort current evidence audit','',
        'Full held-out-cell development CV, not new blind validation. No models changed or adopted by this report.',
        'All numerical gates recomputed from saved predictions. Original four-cohort scope remains incomplete.','',
        '|Cohort|MAPE % ± seed SD|Late MAPE %|RMSE 0–1|End pp|Drop pp|Joint|Gates|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in results:
        s=r['summary'];table.append(f"|{r['dataset']}|{s['mape_percent']['mean']:.3f} ± {s['mape_percent']['seed_sd']:.3f}|{s['late_mape']['mean']:.3f}|{s['rmse_soh']['mean']:.4f}|{s['endpoint_pp']['mean']:.3f}|{s['drop_error_pp']['mean']:.3f}|{r['joint_cells']}/{s['cells']}|{r['passed_gates']}/10|")
    table += ['', 'NASA is deterministic; its zero repeat SD is not evidence of zero uncertainty.',
        'MAE and per-cell R², coverage flags and complete cell/seed records are in the adjacent JSON report.',
        'Missing final requirements: all four cohorts passing, final curve review, final Excel update and independent unseen-data assessment.',
        'Do not delete normal failed cells or choose per-test-cell models from this report.']
    # Reports are generated artifacts, not local source edits.
    (root/'audit_report.md').write_text('\n'.join(table)+'\n',encoding='utf-8')
    save_json(root/'progress.json',dict(complete=True,cohorts=4,outer_cells=248,outer_cell_seed_records=1240,all_numeric_gates_passed=False,goal_complete=False))


if __name__=='__main__':main()
