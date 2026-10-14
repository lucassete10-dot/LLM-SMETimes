"""Record the performed individual visual review; never infer algorithm success."""
import json, shutil
from pathlib import Path
from scripts.prepare_same_type_forecast import save_json


def main():
    run=Path('experiments/llm_soh_mvp/20261008T125806_891108Z')
    root=Path('deliveries/MIT_LLM_SMETimes_MVP_'+run.name)
    manifest=json.loads((root/'evaluation_manifest.json').read_text())
    for row in manifest['figures']:
        if not (root/row['file']).is_file(): raise FileNotFoundError(row['file'])
        row['visual_review']=True
        row['review_note']='Both raw truth and unmodified prediction visible; title, legend and reported failure readable. Curve accuracy is not presentation QA.'
    manifest['all_individual_visual_review']=True
    manifest['visual_notes']=['Some long-span plots have closely spaced horizontal ticks; source PDF/SVG and original-size PNG retained for inspection.',
        'All 27 show substantial curve-shape failure; no performance-success claim.',
        'Single seed: no uncertainty band; 300 dpi PNG and editable PDF/SVG are internal experiment outputs, not journal submission certification.']
    save_json(root/'evaluation_manifest.json',manifest)
    shutil.copy2(run/'gate_diagnostic.json',root/'saved_results/gate_diagnostic.json')
    save_json(root/'presentation_review.json',{'individual_figures_reviewed':27,
        'raw_labels_modified':False,'plot_smoothing':False,'per_cell_model_selection':False,
        'notes':manifest['visual_notes']})
    # Human-readable per-cell index; avoid requiring spreadsheet software.
    metrics=json.loads((root/'metrics.json').read_text())['per_cell']
    lines=['# 全部27颗 MIT MVP 留出电芯','',
        '输入前100次源循环；单种子开发验证；联合达标0/27。点击电芯编号查看原图。','',
        '|电芯|MAPE (%)|MAE (pp)|尾段 MAE (pp)|阶段误差 (pp)|',
        '|---|---:|---:|---:|---:|']
    for row in manifest['figures']:
        cid=row['cell_id'];r=metrics[cid]
        lines.append(f'|[{cid}]({row["file"].replace(chr(92),"/")})|{r["mape_percent"]:.2f}|{r["mae_pp"]:.2f}|{r["tail_mae_pp"]:.2f}|{r["stage_drop_max_pp"]:.2f}|')
    (root/'all_curves.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    status=json.loads((run/'status.json').read_text())
    status['delivery_artifacts_complete']=True
    status['delivery_folder']=str(root.resolve())
    status['delivery_complete']=True
    status['handoff_evidence']='User-facing links to all_curves.md and stage_report.md provided in completed goal turn.'
    save_json(run/'status.json',status)
    print(root.resolve())


if __name__=='__main__': main()
