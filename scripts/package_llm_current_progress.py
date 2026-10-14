"""Package verified current MIT LLM experiments, with every development cell."""
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from scripts.prepare_same_type_forecast import save_json, sha

SKILL = Path('C:/Users/Lenovo/.codex/skills/nature-figure/scripts')
sys.path.insert(0, str(SKILL))
from audit_panel_alignment import require_matplotlib_panel_alignment

RUNS = [
    ('01_Control', '20261008T133739_668360Z', 'llm_integrated_start_aux'),
    ('02_Numeric_LoRA', '20261009T121244_394282Z', 'llm_numeric_lora4_start_aux'),
    ('03_Slower_LoRA', '20261009T121858_535131Z', 'llm_numeric_lora4_slow_start_aux'),
    ('04_Knee_supervision', '20261009T122859_397226Z', 'llm_observed_knee_start_aux'),
]


def main():
    root = Path('deliveries/LLM_SMETimes_MIT_progress_20261009_v1')
    root.mkdir(parents=True, exist_ok=False)
    qa = root / 'figure_sources'
    qa.mkdir()
    shutil.copy2(__file__, qa / Path(__file__).name)
    figures, summaries, all_metrics = [], [], {}
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': 9, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
                         'axes.spines.top': False, 'axes.spines.right': False})
    with PdfPages(qa / 'all_curves.pdf') as pdf:
        for label, stamp, name in RUNS:
            run = Path('experiments/llm_soh_mvp') / stamp
            cfg = json.loads((run / 'config.json').read_text())
            replay = json.loads((run / 'verification.json').read_text())
            arithmetic = json.loads((run / 'metric_verification.json').read_text())
            if not all(v['passed'] for v in replay) or not arithmetic['passed']:
                raise ValueError('unverified experiment')
            if not json.loads((run / 'status.json').read_text())['training_complete']:
                raise ValueError('unfinished experiment')
            # User-requested snapshot copies completed results, never alters runs.
            shutil.copytree(run, root / 'saved_results' / label)
            folder = root / 'MIT' / f'input_{cfg["origin"]:03d}' / cfg['fold'] / label
            folder.mkdir(parents=True)
            source = run / name
            result = json.loads((source / 'result.json').read_text())
            metrics = json.loads((source / 'heldout_metrics.json').read_text())
            validation = json.loads((source / 'validation_metrics.json').read_text())
            all_metrics[label] = {'candidate': name, 'validation': validation, 'development_holdout': metrics}
            for role, rows in [('validation', validation), ('development_holdout', metrics)]:
                values = list(rows.values())
                summaries.append({'version': label, 'candidate': name, 'role': role, 'cells': len(values),
                    'joint_passes': sum(v['joint_pass'] for v in values),
                    **{k: float(np.mean([v[k] for v in values])) for k in
                       ('mape_percent', 'mae_pp', 'tail_mae_pp', 'stage_drop_max_pp')},
                    'selected_epoch': result['selected_epoch']})
            with np.load(source / 'heldout_predictions.npz') as data:
                for cid, row in sorted(metrics.items()):
                    prefix = f'{cid}__{cfg["origin"]}__'
                    x, y, p = [data[prefix + k] for k in ('cycle', 'truth', 'prediction')]
                    assert np.all(np.diff(x) > 0) and np.isfinite(np.c_[x, y, p]).all()
                    fig = plt.figure(figsize=(8, 5))
                    ax = fig.add_axes([.12, .19, .85, .56])
                    ax.plot(x, y, color='#222222', lw=1.6, label='Observed SOH')
                    ax.plot(x, p, color='#2673A6', lw=1.6, label='LLM + SMETimes forecast')
                    ax.set(xlabel=f'Future source cycle | input: first {cfg["origin"]} source cycles',
                           ylabel='SOH (initial capacity reference)')
                    ax.grid(alpha=.15); ax.margins(x=.015, y=.07)
                    fig.text(.12, .96, 'MIT | ' + cid, fontsize=12, va='top')
                    fig.text(.12, .90, f'{label} | {cfg["fold"]} | Single seed | Development holdout', fontsize=8, va='top')
                    fig.text(.12, .85, f'MAPE {row["mape_percent"]:.2f}% | MAE {row["mae_pp"]:.2f} pp | Tail MAE {row["tail_mae_pp"]:.2f} pp', fontsize=8, va='top')
                    fig.text(.12, .80, f'Stage drop error {row["stage_drop_max_pp"]:.2f} pp | Joint criteria: ' +
                             ('Passed' if row['joint_pass'] else 'Not passed'), fontsize=8, va='top')
                    fig.legend(*ax.get_legend_handles_labels(), loc='lower left', bbox_to_anchor=(.105, .025),
                               ncol=2, fontsize=8, frameon=False)
                    stem = qa / f'{label}_{cid}'
                    require_matplotlib_panel_alignment(fig, json_out=str(stem) + '.alignment.json', strict=True)
                    png = folder / (cid + '.png')
                    fig.savefig(png, dpi=300)
                    fig.savefig(str(stem) + '.svg')
                    pdf.savefig(fig)
                    plt.close(fig)
                    figures.append({'version': label, 'cell_id': cid, 'file': png.relative_to(root).as_posix(),
                                    'sha256': sha(png), 'source_predictions_sha256': sha(source / 'heldout_predictions.npz')})
            print('EXPORTED', label, len(metrics), flush=True)
    for script, extras in [('audit_pdf_text.py', ['--min-pt', '5']),
                           ('audit_figure_collisions.py', ['--json-out', str(qa / 'collision_audit.json')])]:
        subprocess.run([sys.executable, str(SKILL / script), str(qa / 'all_curves.pdf'), *extras], check=True)
    save_json(root / 'metrics.json', {'summary': summaries, 'per_cell': all_metrics})
    save_json(root / 'evaluation_manifest.json', {
        'dataset': 'MIT', 'input_cycles': 100, 'fold': 'MIT_fold1', 'seeds': 1,
        'different_cells': len({r['cell_id'] for r in figures}), 'images': len(figures), 'figures': figures,
        'selection': 'fixed validation rule, no per-cell or heldout candidate selection',
        'independent_confirmation': False, 'full_goal_complete': False,
        'missing_new_dataset_results': ['XJTU', 'HUST', 'UConn_ISU_ILCC', 'Zhu2022_NCA'],
        'automated_pdf_checks_passed': True, 'individual_visual_review_complete': False})
    lines = ['# MIT LLM＋SMETimes 最新训练结果', '',
             '包含4项配置×同一批27颗留出开发电芯，共108张主图。输入前100次源循环；1折、1种子。', '',
             '01_Control 是保留的主配置；02–04是近期候选，不能视为已采用的改进。所有模型实际经过预训练LLM。', '',
             '## 指标（先看验证侧选择，再看留出开发结果）', '',
             '|版本|评价身份|电芯数|MAPE %|MAE pp|尾段 MAE pp|阶段误差 pp|联合达标|',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        lines.append(f'|{s["version"]}|{s["role"]}|{s["cells"]}|{s["mape_percent"]:.2f}|{s["mae_pp"]:.2f}|{s["tail_mae_pp"]:.2f}|{s["stage_drop_max_pp"]:.2f}|{s["joint_passes"]}/{s["cells"]}|')
    lines += ['', '## 范围和限制', '',
              '- 这些是开发结果，不能宣称独立确认，也没有达到大部分电芯曲线联合贴合的目标。',
              '- 图片未平滑、裁剪或强行对齐；成功与失败全部保留。单种子无置信区间或虚构波动带。',
              '- 本次是MIT最新优化快照，没有用历史树模型或其他数据集旧图填充。其余四数据集尚无本轮完整结果。',
              '- 100次是本轮统一输入预算，不是已证明的最优循环数。',
              '- saved_results包含配置、模型外围权重、预处理器、日志、逐点预测、划分和代码快照。',
              '- 冻结LLM骨干按记录的检查点和哈希恢复，未重复打包约2.47GB骨干权重。',
              '- figure_sources/all_curves.pdf包含全部图片的可选中文字版本；指标详见metrics.json。',
              '', '## 图片索引', '']
    for r in figures:
        lines.append(f'- [{r["version"]} / {r["cell_id"]}]({r["file"]})')
    (root / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (qa / 'figure_contract.md').write_text(
        'Question: How closely does each saved LLM prediction follow its unmodified development-cell SOH?\n'
        'Single-panel quantitative trajectory; no aggregation, uncertainty, smoothing, label exclusion or individual model selection.\n'
        'Mapping: saved cycle -> horizontal source cycle; truth/prediction -> SOH on common initial-capacity reference.\n'
        'Internal 8x5 inch experiment preview, 300 dpi PNG and editable multipage PDF; not journal submission certification.\n'
        'Reuse: structural adaptation of the existing project MVP exporter, unchanged metric/data semantics.\n', encoding='utf-8')
    archive = root.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for file in sorted(root.rglob('*')):
            if file.is_file():
                z.write(file, file.relative_to(root.parent))
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError('ZIP integrity failure')
        assert sum(n.endswith('.png') for n in z.namelist()) == len(figures)
    print(json.dumps({'zip': str(archive.resolve()), 'bytes': archive.stat().st_size,
                      'images': len(figures), 'different_cells': 27, 'summary': summaries}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
