"""Reproduce all verified MIT checkpoint curves without smoothing or selection."""
import csv
import html
import json
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SKILL = Path('C:/Users/Lenovo/.codex/skills/nature-figure/scripts')
sys.path.insert(0, str(SKILL))
from audit_panel_alignment import require_matplotlib_panel_alignment


def main():
    summary_path = Path('experiments/core_length_cv/MIT_five_seed_summary_20261008.json')
    summary = json.loads(summary_path.read_text())
    roots = [Path(s['run']) for s in summary['sources']]
    configs = [json.loads((r/'config.json').read_text()) for r in roots]
    assert all(c['plans'] == configs[0]['plans'] for c in configs)
    assert all(json.loads((r/'verification.json').read_text())['passed'] for r in roots)
    seeds = [c['seeds'][0] for c in configs]
    cell_metrics = {r['cell_id']: r for r in summary['cell_rows']}
    out = Path('deliveries/MIT_main_20261008_five_seed')
    folder = out/'MIT/input_100'
    folder.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 9, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
        'axes.spines.top': False, 'axes.spines.right': False})
    manifest, metric_rows = [], []
    for plan in configs[0]['plans']:
        arrays = [dict(np.load(r/plan['name']/'100/available/predictions.npz', allow_pickle=False)) for r in roots]
        for cid in plan['validation']:
            cycle, truth = [arrays[0][cid+'__'+k] for k in ('cycle', 'truth')]
            assert np.all(np.diff(cycle)>0) and np.all(cycle>100)
            assert np.all(np.isfinite(truth)) and np.all(truth>0)
            for a in arrays:
                np.testing.assert_array_equal(cycle, a[cid+'__cycle'])
                np.testing.assert_array_equal(truth, a[cid+'__truth'])
            stack = np.stack([a[cid+'__prediction'] for a in arrays])
            assert np.all(np.isfinite(stack))
            mean, sd = stack.mean(0), stack.std(0, ddof=1)
            row = cell_metrics[cid]
            mm = row['mean_prediction_metrics']
            fig = plt.figure(figsize=(8, 5))
            ax = fig.add_axes([.12, .19, .85, .56])
            ax.plot(cycle, truth, color='#222222', lw=1.6, label='Observed SOH')
            ax.plot(cycle, mean, color='#2673A6', lw=1.6, label='Forecast mean')
            ax.fill_between(cycle, mean-sd, mean+sd, color='#2673A6', alpha=.15,
                linewidth=0, label='One seed SD (not a CI)')
            ax.set(xlabel='Future source cycle | input: first 100 source cycles', ylabel='SOH (capacity / rated capacity)')
            ax.grid(alpha=.15)
            ax.margins(x=.015, y=.07)
            fig.text(.12, .96, 'MIT | '+cid, fontsize=12, va='top')
            fig.text(.12, .90, 'Held-out development cell | Five seeds | Core point model', fontsize=8, va='top')
            fig.text(.12, .85, f"Seed-mean MAPE {row['seed_mean_mape']:.2f}% +/- {row['seed_sd_mape']:.2f}%"
                f" | Mean-curve late MAE {mm['late20_mae_pp']:.2f} pp", fontsize=8, va='top')
            fig.text(.12, .80, 'Curve-shape success not established | No independent-test claim', fontsize=8, va='top')
            fig.legend(*ax.get_legend_handles_labels(), loc='lower left', bbox_to_anchor=(.105,.02), ncol=3, fontsize=8, frameon=False)
            stem = folder/cid
            require_matplotlib_panel_alignment(fig, json_out=str(stem)+'.alignment.json', tolerance_pt=1.5, strict=True)
            fig.savefig(str(stem)+'.png', dpi=300)
            fig.savefig(str(stem)+'.pdf')
            fig.savefig(str(stem)+'.svg')
            plt.close(fig)
            for script, extra in [('audit_pdf_text.py', ['--min-pt','5']),
                ('audit_figure_collisions.py', ['--json-out',str(stem)+'.collision.json'])]:
                subprocess.run([sys.executable,str(SKILL/script),str(stem)+'.pdf',*extra], check=True, stdout=subprocess.DEVNULL)
            with Path(str(stem)+'.csv').open('w',newline='',encoding='utf-8-sig') as f:
                writer=csv.writer(f)
                writer.writerow(['source_cycle','true_soh','forecast_mean','seed_sd',*[f'prediction_seed_{s}' for s in seeds]])
                writer.writerows(zip(cycle,truth,mean,sd,*stack))
            rec = dict(cell_id=cid,fold=plan['name'],input_cycles=100,
                seed_mean_mape_percent=row['seed_mean_mape'],seed_sd_mape_percent=row['seed_sd_mape'],
                **{'mean_curve_'+k:v for k,v in mm.items()})
            metric_rows.append(rec)
            manifest.append(dict(cell_id=cid,fold=plan['name'],png=str(stem.resolve())+'.png',
                automated_pdf_qa=True,individual_visual_review=False))
            print(f'Exported {len(manifest)}/107: {cid}',flush=True)
    assert len(manifest)==107 and len({r['cell_id'] for r in manifest})==107
    with (out/'cell_metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(metric_rows[0]));writer.writeheader();writer.writerows(metric_rows)
    length = json.loads((roots[0]/'length_summary.json').read_text())
    length_rows=[r for r in length['summary'] if r['arm']=='available' and r['horizon']=='common']
    with (out/'input_length_comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(length_rows[0]));writer.writeheader();writer.writerows(length_rows)
    document=dict(planned_cells=107,completed_cells=107,images=107,seeds=seeds,
        source_summary=str(summary_path.resolve()),sources=summary['sources'],figures=manifest,
        independent_test=False,goal_complete=False,shape_success=False,
        input_cycles=100,input_selection_status='provisional; no reliably shape-qualified N',
        workbook_pending=True,visual_review_complete=False)
    (out/'manifest.json').write_text(json.dumps(document,indent=2),encoding='utf-8')
    cards='\n'.join(f'<section><h2>{html.escape(r["cell_id"])}</h2><img loading="lazy" src="MIT/input_100/{r["cell_id"]}.png" alt="{r["cell_id"]}"></section>' for r in manifest)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>MIT SOH 开发验证曲线</title>'
        '<style>body{max-width:1050px;margin:24px auto;font:16px sans-serif}img{width:100%}section{margin:35px 0}</style>'
        '<h1>MIT：107颗留出开发电芯，输入100次源循环</h1>'
        '<p>五折×五种子。蓝线为均值，阴影为种子SD，不是预测置信区间。未做平滑。曲线形状尚未达标，非独立测试。逐张人工审查仍在进行。</p>'+cards,encoding='utf-8')
    (out/'stage_report.md').write_text('# MIT阶段结果\n\n107颗留出开发电芯，5折×5种子，全部主曲线及逐点预测已保存。'
        '\n\n种子与电芯等权平均MAPE 1.82988%，种子SD 0.01728%。尾段MAE 3.21691个百分点，阶段下降量误差6.04393个百分点。106/107颗种子平均MAPE≤6%，但不能据此宣称曲线形状达标。'
        '\n\n输入10至100循环已比较，共同未来区间的结果见长度表。N100是阶段配置，可靠较短输入长度仍未确定。'
        '\n\n全部PDF已执行字体与碰撞自动检查；人工逐张审查尚未完成。Excel尚待完成，当前CSV可直接用Excel打开。模型与配置保留在manifest列出的原实验目录，不重复复制。未生成ZIP。',encoding='utf-8')
    print(out.resolve(),flush=True)


if __name__=='__main__':
    main()
