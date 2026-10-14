"""Dataset contracts and readable result summaries from pinned metadata."""
import json
from pathlib import Path
from scripts.audit_llm_curve_repair import fingerprint

SIGNALS={
    'MIT':'Measured V/I/T and time-derived features from the fixed first 300 seconds of visible-cycle main discharge. Author-stored current is converted using fixed nominal 1.1 A/C.',
    'XJTU':'Measured early charge/diagnostic-discharge V/I/T features in the prepared fixed-window schema; actual visible source cycles are retained.',
    'HUST':'Measured V/I in each visible cycle, fixed first 300 seconds of discharge. Measured temperature is unavailable.',
    'UConn_ISU_ILCC':'Six measured slow-pulse voltage profiles (three controlled SOC levels, two pulse directions), 101 samples each. Sparse RPT observations use cumulative aging-source cycles; measured I/T are unavailable.',
    'Zhu2022_NCA':'Measured V/I and time, fixed first 900 seconds of comparable diagnostic discharge. Measured temperature is unavailable.'}


def dataset_contract(cfg,dataset):
    if 'unified_external_index' in cfg:
        index=Path(cfg['unified_external_index'])
        if fingerprint(index)!=cfg['unified_external_index_sha256']:raise ValueError('external contract index changed')
        manifest=json.loads(index.read_text(encoding='utf-8'));registry_path=index.parent/'registry.json'
        if fingerprint(registry_path)!=manifest['registry_sha256']:raise ValueError('external contract registry changed')
        members=[r for r in manifest['cells'] if r['metadata']['dataset']==dataset]
        families={r['physical_family'] for r in members};ids={r['cell_id'] for r in members};limitations=manifest['limitations'][dataset]
        plans=[p for p in manifest['plans'] if p['dataset']==dataset]
    else:
        audit=Path(cfg['frozen_plan']);registry_path=audit/'source_snapshot/battery_physical_registry.json';index=audit/'splits.json'
        plans=[p for p in json.loads(index.read_text(encoding='utf-8'))['plans'] if p['dataset']==dataset]
        ids={cid for p in plans for role in ('train','validation','test') for cid in p[role]}
        registered=json.loads((audit/'cell_registry.json').read_text(encoding='utf-8'))['cells']
        families={registered[cid]['physical_family'] for cid in ids};limitations=[]
    registry=json.loads(registry_path.read_text(encoding='utf-8'))
    family_records={family:registry['families'][family] for family in families}
    for record in family_records.values():limitations.extend(record.get('limitations',[]))
    return {'dataset':dataset,'source_cells':len(ids),'closed_source_cohort_only':True,'families':family_records,
        'signals':SIGNALS[dataset],'origin_source_cycle_limit':cfg['origin'],
        'label_reference':'Initial accepted comparable discharge capacity. Capacity and SOH are supervision/reference only, never inference features.',
        'raw_capacity_labels_not_smoothed':True,'split_plans':[{k:p[k] for k in ('name','train','validation','test')} for p in plans],
        'limitations':list(dict.fromkeys(limitations)),'metadata_source_sha256':{str(index):fingerprint(index),str(registry_path):fingerprint(registry_path)},
        'independent_confirmation_certified':False}


def write_readable_summary(out,summary):
    out=Path(out);contract=summary['dataset_contract'];lines=[f"# {summary['dataset']} LLM+SMETimes results",'',
        f"Completed runs: {summary['runs']}/{summary['planned_runs']}. Dataset complete: {summary['dataset_complete']}. The overall five-dataset goal is not certified by this report.",
        '', f"| Role | Requested cell-seed instances | Evaluated | MAPE<={summary['primary_reporting_standard']['numeric_thresholds']['mape_percent']:g}% | Practical joint | Mean MAPE | Median MAPE | Worst MAPE |",
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(value):return 'unavailable' if value is None else f'{value:.3f}%'
    for role,r in summary['roles'].items():
        d=r['metric_distributions']['mape_percent'];requested=r['requested_cell_seed_instances']
        lines.append(f"| {role} | {requested} | {r['evaluated_cell_seed_instances']} | {r['mape_target_passes']}/{requested} | {r['practical_numeric_passes']}/{requested} | {fmt(d['mean'])} | {fmt(d['median'])} | {fmt(d['maximum'])} |")
    lines+=['','Pass counts use all requested instances of the completed runs; unavailable observations remain in the denominator. Repeated seed instances are not independent physical cells. MAPE success does not certify endpoint or trajectory-shape fidelity.',
        '',f"Input: {contract['signals']}",'',f"Label reference: {contract['label_reference']}",'',
        f"Closed cohort size: {contract['source_cells']} cells. Origin budget: first {contract['origin_source_cycle_limit']} actual source cycles. Families are trained separately; no automatic matching to anonymous external cells.",
        '', 'Known limitations:']
    lines += ['- '+item for item in contract['limitations']] or ['- No additional source-specific limitation recorded.']
    lines+=['','The first-fold plotting choice is fixed before examining errors. TEST predictions take priority; a cell\'s TRAIN predictions never enter its TEST seed mean. Raw truth and full observed source ranges are retained. Bands are +/- one seed SD, not confidence intervals.',
        '', 'Artifacts: summary.json, per_cell_seed_metrics.json, all_cells.pdf, figures/*.png and *.svg, figure_manifest.json, plot_source.py. The manifest records automatic and visual QA separately. Actual run folders preserve checkpoint weights, preprocessors, raw NPZ predictions, replay/isolation checks and source snapshots.']
    (out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
