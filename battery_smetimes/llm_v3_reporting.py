"""Descriptive reference-quality reporting; never used to rank TEST candidates."""
from collections import defaultdict
from pathlib import Path
from battery_smetimes.llm_curve_quality_v3 import curve_metrics_v3,reference_pass,QUALITY
from battery_smetimes.llm_reporting import finite_metric
from battery_smetimes.llm_v3_workflow import SEEDS


def enrich_quality_rows(rows,archive,role):
    for row in rows:
        entries=[entry for entry in archive[row['cell_id']][role]
                 if entry['fold']==row['fold'] and entry['seed']==row['seed']]
        if len(entries)!=1:raise ValueError('duplicate or missing raw curve for reporting')
        entry=entries[0]
        actual=curve_metrics_v3(entry['x'],entry['y'],entry['p'])
        row.update(actual,reference_numeric_pass=reference_pass(actual))
        row['reference_quality_failures']=[k for k,v in QUALITY.items()
            if not finite_metric(actual,k) or actual[k]>v]


def quality_summary(rows,requested):
    if len(rows)>requested:raise ValueError('quality report exceeds requested coverage')
    groups=defaultdict(list)
    for row in rows:groups[(row['cell_id'],row['fold'])].append(row)
    complete=stable=primary_stable=0
    for entries in groups.values():
        if len({r['seed'] for r in entries})!=len(entries):raise ValueError('duplicate reporting seed')
        if {r['seed'] for r in entries}!=set(SEEDS):continue
        complete+=1
        primary=all(finite_metric(r,'mape_percent') and r['mape_percent']<=3 for r in entries)
        primary_stable+=primary
        stable+=primary and sum(reference_pass(r) for r in entries)>=2
    return {'thresholds':QUALITY,'requested_cell_seed_instances':requested,
        'evaluated_cell_seed_instances':len(rows),'missing_cell_seed_instances':requested-len(rows),
        'reference_numeric_passes':sum(reference_pass(r) for r in rows),
        'reference_pass_fraction':sum(reference_pass(r) for r in rows)/requested if requested else None,
        'stage_evidence_unavailable_instances':sum(not finite_metric(r,'stage_drop_max_pp') for r in rows),
        'observed_cell_fold_groups':len(groups),'three_seed_complete_cell_fold_groups':complete,
        'stable_primary_cell_fold_groups':primary_stable,'stable_reference_cell_fold_groups':stable,
        'stability_denominator_note':'Only observed cell-fold groups; full requested/missing seed denominator shown separately.',
        'shape_visual_review_required':True,'goal_complete':False}


def write_quality_readme(out,summary):
    lines=['','## v3 reference-curve quality','',
        'Actual model checkpoint selection versions: '+', '.join(summary['model_selection_versions'])+'. Applying this reporting standard does not retrain or change historical models.',
        'Reference: MAPE<=1.5%, late MAPE<=3%, start<=1pp, endpoint<=3pp, stage drop error<=3pp. Primary MAPE target remains3%. Numeric reference passing still requires curve review.',
        '', '| Role | Reference passes / requested seed instances | Complete three-seed cell-fold groups | Stable reference groups | Stage unavailable |',
        '|---|---:|---:|---:|---:|']
    for role,r in summary['v3_reference_quality'].items():
        lines.append(f'| {role} | {r["reference_numeric_passes"]}/{r["requested_cell_seed_instances"]} | {r["three_seed_complete_cell_fold_groups"]} | {r["stable_reference_cell_fold_groups"]} | {r["stage_evidence_unavailable_instances"]} |')
    lines+=['','Stability requires two of three seeds to meet all reference metrics and all three to meet primary MAPE<=3%. Sparse stage evidence is unavailable, never zero. TRAIN groups repeated across folds are not independent physical cells. All failed curves remain in the report.','']
    path=Path(out)/'README.md'
    with path.open('a',encoding='utf-8') as file:file.write('\n'.join(lines))
