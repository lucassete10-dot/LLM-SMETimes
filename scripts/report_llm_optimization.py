"""Readable algorithm comparisons, based on saved validation-only evidence."""
import json
from pathlib import Path
from scripts.prepare_same_type_forecast import save_json
from scripts.audit_llm_curve_repair import fingerprint


def write_optimization_report(folder,state,plan):
    folder=Path(folder);baseline=state['baseline_validation']
    score_text=lambda v:f'{v:.5f}' if v is not None else 'unavailable (stage coverage)'
    lines=[f"# {state['dataset']} LLM+SMETimes algorithm comparison",'',
        'Selection uses validation only. TEST forecasts do not rank candidates. Scores are cell-macro MAE + tail MAE + 0.5 x stage error, in SOH percentage points.',
        '',f"Baseline: score {score_text(baseline['score'])}; worst validation MAE {baseline['worst_mae_pp']:.5f} pp; MAPE<={plan['numeric_thresholds']['mape_percent']:g}% {baseline['mape_passes']}/{baseline['cells']}.",
        '', f"| Candidate | Validation score | Improvement | Worst VAL MAE / pp | MAPE<={plan['numeric_thresholds']['mape_percent']:g}% | Screening |",
        '|---|---:|---:|---:|---:|---|']
    coverage_notes=[]
    for item in plan['screen_candidates']:
        slot=state['screening'].get(item['name'],{});value=slot.get('validation')
        if value:
            improvement=f"{100*(1-value['score']/baseline['score']):.2f}%" if value['score'] is not None and baseline['score'] else 'unavailable'
            lines.append(f"| {item['name']} | {score_text(value['score'])} | {improvement} | {value['worst_mae_pp']:.5f} | {value['mape_passes']}/{value['cells']} | {'passed' if slot.get('screen_passed') else 'not passed'} |")
            if value.get('stage_unavailable_cells'):coverage_notes.append(f"Stage metric unavailable for {item['name']}: {', '.join(value['stage_unavailable_cells'])}.")
        else:lines.append(f"| {item['name']} | pending | pending | pending | pending | {slot.get('state','pending')} |")
    lines+=['',*coverage_notes,'',f"State: {state['state']}. Winner: {state.get('winner') or 'none confirmed'}.",
        '',state.get('conclusion','Screening or replication is still incomplete; no benefit has been established.')]
    if state.get('locked_validation_evidence'):
        lines+=['','| Confirmation instance | Baseline score | Candidate score |', '|---|---:|---:|']
        for index,(base,candidate) in enumerate(zip(state['locked_validation_evidence']['baseline'],state['locked_validation_evidence']['candidate'])):
            seed=[plan['screen_seed'],*plan['confirmation_seeds']][index]
            lines.append(f"| seed {seed} | {base['score']:.5f} | {candidate['score']:.5f} |")
    lines+=['','All failed candidates are retained. A replicated validation benefit is distinct from TEST accuracy success.',
        'Raw metrics, source snapshots, model checksums and replay/isolation audits stay in each linked run; physical-cohort and sparse-input limitations remain applicable.']
    (folder/'algorithm_comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save_json(folder/'comparison_manifest.json',{'dataset':state['dataset'],'state':state['state'],
        'report_sha256':fingerprint(folder/'algorithm_comparison.md'),'selection_scope':'validation only',
        'test_used_for_selection':False,'optimization_accuracy_improved':state.get('optimization_accuracy_improved',False),'goal_complete':False})
