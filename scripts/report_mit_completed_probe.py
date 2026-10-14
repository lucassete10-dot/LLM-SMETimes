"""Write a milestone report without repackaging or declaring curve success."""
import argparse,json
from pathlib import Path
import numpy as np
from scripts.prepare_same_type_forecast import save_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);ap.add_argument('head',type=Path);a=ap.parse_args()
    assert json.loads((a.run/'verification.json').read_text())['passed']
    rows=json.loads((a.run/'all_rows.json').read_text());summary=json.loads((a.run/'summary.json').read_text())
    lines=['# MIT 曲线优化：第一轮开发验证','',
        '范围：107 颗开发电芯，固定五折、一个种子（20261008）。不是独立测试，也不是最终多种子确认。',
        '50/100 次输入均在共同的第 100 次之后评价；所有电芯保留，未按预测误差排除。',
        '新输入读取已完成的早期主 4C 放电片段，不读取容量标签。电流积分仅作测量特征，不直接当作标准容量或 SOH。','',
        '|输入循环数|方案|MAPE %|尾段 MAE pp|阶段下降量误差 pp|严格筛查通过数|',
        '|---|---|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"|{r['origin']}|{r['arm']}|{r['mape_percent']:.3f}|{r['late20_mae_pp']:.3f}|{r['stage_error_pp']:.3f}|{r['strict_screen_pass']}/{r['cells']}|")
    lines+=['','严格筛查是已有开发指标，不包括已经确认的拐点/平台检验，不能视为最终曲线达标。','',
        '## 四个诊断案例（相同种子对照）','',
        '|电芯|方案|MAPE %|尾段 MAE pp|终点误差 pp|','|---|---|---:|---:|---:|']
    for cid in ['b1c1','b1c4','b1c21','b1c29']:
        for r in rows:
            if r['cell_id']==cid and r['origin']==100:
                lines.append(f"|{cid}|{r['arm']}|{r['mape_percent']:.3f}|{r['late20_mae_pp']:.3f}|{r['endpoint_pp']:.3f}|")
    paired=[]
    for n in [50,100]:
        old={r['cell_id']:r for r in rows if r['origin']==n and r['arm']=='prefix300'}
        new={r['cell_id']:r for r in rows if r['origin']==n and r['arm']=='completed_past'}
        paired.append(dict(origin=n,cells=len(old),mape_improved=sum(new[c]['mape_percent']<old[c]['mape_percent'] for c in old),
            tail_improved=sum(new[c]['late20_mae_pp']<old[c]['late20_mae_pp'] for c in old),
            stage_improved=sum(new[c]['stage_error_pp']<old[c]['stage_error_pp'] for c in old),
            all_three_improved=sum(all(new[c][k]<old[c][k] for k in ['mape_percent','late20_mae_pp','stage_error_pp']) for c in old)))
    save_json(a.run/'paired_improvement.json',paired)
    lines+=['','## 结论与后续','',
        '- 新特征有开发验证收益，但四个案例并非全项改善，尚未达到“大部分曲线贴合良好”的完成标准。',
        '- 代码无末值延长/输出裁剪；叶节点贡献可精确重建预测，已保存 diagnostic_support.json。',
        '- b1c29 第 916 次预测中，原模型约 68.75% 权重来自更早至少 20 次的训练记录，说明尾部存在时间支持错位。新增特征并未消除此问题。',
        '- 连续 level/rate/knee/acceleration 候选五折试验尾段误差更差，拒绝采用，不因曲线更平滑而替换。',
        '- 50/100 次只是本轮对照，不能宣布最优输入循环数；还需扩展候选长度与多种子确认。',
        '- 下一轮优先：让个体早期信号对衰减速度/阶段位置真正起作用，处理训练记录终止后支持错位；不能简单补尾、强制弯曲或对诊断电芯逐颗调参。','',
        '## 保存结果','',f'- 逐电芯数值：{a.run}/all_rows.json',f'- 各折权重和逐点预测：{a.run}',f'- 连续候选：{a.head}',
        '- 不生成新的 ZIP；本轮报告不代表最终完成。']
    (a.run/'stage_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(paired,indent=2));print('REPORT='+str((a.run/'stage_report.md').resolve()))


if __name__=='__main__':main()
