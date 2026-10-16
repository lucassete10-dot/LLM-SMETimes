"""Raw-data curves for frozen held-out SOH estimation / extrapolation MVP."""
import argparse
from pathlib import Path
import subprocess
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from battery_smetimes.llm_full_life import read
from battery_smetimes.llm_curve_quality_v3 import curve_metrics_v3
from scripts.audit_llm_curve_repair import fingerprint, load_frozen
from scripts.train_llm_full_life import save


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,default=Path('experiments/llm_reconstruction/20261016_heldout_MVP'))
    root=ap.parse_args().root; result=read(root/'result.json'); assert result['complete']
    skill=Path('C:/Users/Lenovo/.codex/skills/nature-figure/scripts');sys.path.insert(0,str(skill))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':9,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    # Five separate single-axis validation diagnostics; no uncertainty interval
    # is justified by one fixed seed and one selected physical cell per dataset.
    figures=[]
    with PdfPages(root/'all_MVP_curves.pdf') as book:
        for row in result['cells']:
            ds=row['dataset']; folder=root/ds
            assert fingerprint(folder/'predictions.npz')==row['predictions_sha256']
            registration=read(root/'registration.json')
            chosen=next(c for c in registration['cells'] if c['dataset']==ds)
            run=Path(chosen['model']);cfg=read(run/'config.json');index=read(cfg['index'])
            assert fingerprint(run/'model.pt')==row['model_sha256']
            _,cells,plan,_,_=load_frozen(index['base_settings']); cid=row['cell_id']
            assert cid==plan['validation'][0] and cid not in cfg['train_ids']
            cell=cells[cid]; take=np.isfinite(cell['soh'])&(cell['soh']>0)&(cell['label_cycle']>100)
            with np.load(folder/'predictions.npz',allow_pickle=False) as a:
                x=a['cycle'];y=a['truth']
                np.testing.assert_array_equal(x,cell['label_cycle'][take]);np.testing.assert_array_equal(y,cell['soh'][take])
                for mode,key,field in [('prefix_only','prefix_only_extrapolation','prefix_only_metrics'),
                                       ('full_signal','full_signal_prediction','full_signal_metrics')]:
                    p=a[key];m=curve_metrics_v3(x,y,p)
                    for name in ('mape_percent','mae_pp','tail_mae_pp','endpoint_error_pp'):
                        np.testing.assert_allclose(m[name],row[field][name],atol=1e-9,rtol=0)
                    fig=plt.figure(figsize=(8,5));ax=fig.add_axes([.12,.20,.85,.55])
                    prefix=mode=='prefix_only'
                    label='LLM + SMETimes direct extrapolation' if prefix else 'LLM + SMETimes SOH estimate'
                    ax.plot(x,y,color='#222222',lw=1.5,marker='o' if ds=='UConn_ISU_ILCC' else None,
                            markersize=2.5,label='Observed SOH')
                    ax.plot(x,p,color='#2673A6',lw=1.3,label=label)
                    ax.set(xlabel='Future source cycle (signals only through cycle 100)' if prefix else
                           'Source cycle (complete-life signals available)',ylabel='SOH (initial capacity reference)')
                    ax.grid(alpha=.15)
                    fig.text(.12,.95,ds+' | '+cid,fontsize=12)
                    fig.text(.12,.89,'Held-out validation cell | Frozen TRAIN model | Single seed',fontsize=8)
                    fig.text(.12,.84,f"MAPE {m['mape_percent']:.3f}% | MAE {m['mae_pp']:.3f} pp | Tail {m['tail_mae_pp']:.3f} pp",fontsize=8)
                    subtitle='Prefix-only head extrapolation | No forecast-specific training' if prefix else 'Full-signal SOH estimation | Not a future forecast'
                    fig.text(.12,.79,subtitle+f" | Endpoint {m['endpoint_error_pp']:.3f} pp",fontsize=8)
                    fig.legend(*ax.get_legend_handles_labels(),loc='lower left',bbox_to_anchor=(.12,.025),
                               frameon=False,ncol=2,borderaxespad=0,fontsize=8)
                    stem=folder/mode
                    require_matplotlib_panel_alignment(fig,json_out=str(stem)+'.alignment.json',strict=True)
                    fig.savefig(str(stem)+'.png',dpi=300);fig.savefig(str(stem)+'.svg')
                    fig.savefig(str(stem)+'.pdf');book.savefig(fig);plt.close(fig)
                    figures.append(dict(dataset=ds,cell_id=cid,mode=mode,png=str(stem.resolve())+'.png',
                        png_sha256=fingerprint(str(stem)+'.png'),pdf=str(stem.resolve())+'.pdf',metrics=m))
    qa=[]
    for tool,extra in [('audit_pdf_text.py',['--min-pt','5']),
                       ('audit_figure_collisions.py',['--json-out',str(root/'collision_audit.json')])]:
        r=subprocess.run([sys.executable,str(skill/tool),str(root/'all_MVP_curves.pdf'),*extra],capture_output=True,
                         text=True,encoding='utf-8')
        qa.append(dict(tool=tool,exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr))
    if any(r['exit_code'] for r in qa):raise ValueError('MVP PDF layout audit failed')
    save(root/'figure_manifest.json',dict(figures=figures,qa=qa,actual_visual_review='pending',
        pdf_sha256=fingerprint(root/'all_MVP_curves.pdf'),raw_truth_exact=True,
        frozen_models_unchanged=True,role='validation',no_training=True))
    lines=['# 已训练模型：每个数据集一颗未训练电芯 MVP','',
        '每个数据集固定取原始划分 validation 列表的第一颗电芯；运行前固定选择，没有按效果挑选，没有重训、调参或更换模型。仅检查5颗电芯，不能代表整个数据集泛化。单种子。',
        '', '## 先看预测尝试','',
        '当前模型训练任务是完整信号到同一时段SOH的重建，没有训练前100周期预测未来。本次直接外推是能力诊断：只输入source cycle≤100的信号，所有未来查询使用最后观测对应的patch特征，加实际未来cycle交给原重建头。不使用未来信号、未来SOH或电芯ID；不使用真实总寿命归一化。查询末尾变短的重放已核对，不改变前面预测。该外推规则未经预测任务训练，失败不等于LLM+SMETimes预测方法无效。',
        '', '| 数据集 | 固定电芯 | 前100外推 MAPE | 同时段完整信号估计 MAPE | 预测尝试 | 完整信号估计 |',
        '|---|---|---:|---:|---|---|']
    for row in result['cells']:
        ds=row['dataset'];folder=(root/ds).resolve()
        lines.append(f"| {ds} | {row['cell_id']} | {row['prefix_only_metrics']['mape_percent']:.3f}% | {row['full_signal_metrics']['mape_percent']:.3f}% | [图]({(folder/'prefix_only.pdf').as_posix()}) | [图]({(folder/'full_signal.pdf').as_posix()}) |")
    lines+=['',f"[全部10张对照图 PDF]({(root/'all_MVP_curves.pdf').resolve().as_posix()})",'',
        '完整信号估计读取该未训练电芯整个寿命阶段的信号，并可使用同一时段附近信号。两种任务均只在source cycle>100的原始SOH标签上计分，黑线完全保留原值。完整信号估计不属于前100预测未来。',
        '', '## 目前结论与下一步','',
        '直接外推在HUST、XJTU、UConn上仍明显不足。说明重建能力尚未转化为未来预测能力，不应靠增加重建轮数解决。下一步应在TRAIN电芯上以多个历史截点构造“仅过去信号→未来SOH”监督任务，保留已学编码器初始化，训练面向未来的非线性预测头；预测阶段严格只读前100信号。采用独立验证集合比较趋势、拐点、尾段和终点，再冻结模型测试；保留本次5颗为已查看的开发样本。',
        '', '## 对照图','']
    for row in result['cells']:
        folder=(root/row['dataset']).resolve()
        lines.extend([f"### {row['dataset']} / {row['cell_id']}",'',
            f"![前100周期直接外推]({(folder/'prefix_only.png').as_posix()})",'',
            f"![完整信号SOH估计]({(folder/'full_signal.png').as_posix()})",''])
    (root/'results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(dict(figures=len(figures),pdf=str(root/'all_MVP_curves.pdf')),flush=True)


if __name__=='__main__':main()
