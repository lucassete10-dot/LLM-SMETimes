"""Summarize measured TRAIN/VAL symptoms without asserting a causal repair."""
import argparse
import json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args()
    q=json.loads((args.root/'queue.json').read_text(encoding='utf-8'))
    lines=['# v4 首折基线诊断','',
           '以下来自五数据集首折、种子20260930、120轮上限的已完成R0。与全折三种子结果的统计范围不同。TRAIN/VAL所有电芯参与，未读取TEST预测。',
           '', '| 数据集 | 角色 | 电芯 | 起点误差>1pp | 尾段MAPE>3% | 速率节点平均CV | 局部恢复接近幅度上限 |',
           '|---|---|---:|---:|---:|---:|---:|']
    complete=0;links=[]
    for name in q['sequence']:
        path=args.root/'diagnostics'/name/'diagnosis.json'
        if not path.exists():continue
        record=json.loads(path.read_text(encoding='utf-8'));complete+=1
        for role,s in record['summary'].items():
            lines.append(f'| {name} | {role} | {s["cells"]} | {s["start_error_gt1pp"]} | {s["late_mape_gt3percent"]} | {s["mean_rate_cv"]:.3f} | {100*s["mean_local_saturation_fraction"]:.1f}% |')
        links.append(f'- [{name} 逐电芯原始诊断]({path.resolve().as_posix()})')
    lines += ['',f'诊断完成范围：{complete}/5。', '',*links,'',
              'CV是实际预测跨度内速率节点的变异系数；值小提示接近恒定衰减速率，不等同于整条预测必然线性。恢复分支饱和比例以 |residual|≥0.95×原有1pp幅度上限计算，且只统计实际跨度内节点。',
              '', '每个诊断均独立复拟合TRAIN预处理器，并将模型输入的标签删除、未来原始信号污染为Inf，验证输入完全一致。标准化异常比例只统计真实可用通道。未增加真实SOH起点锚点，也未改变标签。',
              '', '这些量化结果不能单独证明因果。K候选验证速率节点分辨率；L候选验证尾段/阶段监督权重。收益须经同折、同预算、同种子VAL比较及三种子确认。']
    (args.root/'diagnostics.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'diagnosed_datasets':complete,'report':str((args.root/'diagnostics.md').resolve())}))


if __name__=='__main__':main()
