"""Report completed v3 TRAIN/VAL comparisons; never consume TEST metrics."""
import argparse
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from scripts.run_llm_curve_v3_r0 import read
from scripts.prepare_same_type_forecast import save_json
from battery_smetimes.llm_v3_workflow import (
    SEEDS,trial_key,summarize_validation,screen_pass,confirmation_pass)
from battery_smetimes.llm_reporting import finite_metric


def mean_errors(evidence):
    rows=list(evidence['by_cell'].values());result={}
    for key in ('mae_pp','start_error_pp','tail_mae_pp','endpoint_error_pp','stage_drop_max_pp'):
        values=[r[key] for r in rows if finite_metric(r,key)]
        result[key]={'mean':float(np.mean(values)) if values else None,
                     'available':len(values),'requested':len(rows)}
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args()
    root=args.root;workflow=read(root/'workflow.json');records=[];confirmations=[]
    lines=['# v3 实际验证对照','',
        '只统计已完成的同划分、同种子、同预算验证对照。分数越低越好；没有胜出的候选仍保留结果。TEST没有用于本报告的算法排名。','',
        '| 数据集 | 候选 | 预算 | R0分数 | 候选分数 | 相对改善 | MAPE≤3% R0→候选 | 最差MAE/pp R0→候选 | 筛选通过 |',
        '|---|---|---:|---:|---:|---:|---|---|---|']
    for dataset in workflow['sequence']:
        state=workflow['datasets'][dataset];fold=state['folds'][0]
        reference=state['trials'][trial_key('R0',fold,SEEDS[0],120)]
        base=summarize_validation(reference['evidence'])
        for config in state['screen_order']:
            candidate=state['trials'].get(trial_key(config,fold,SEEDS[0],120))
            if not candidate or candidate['state']!='complete':continue
            summary=summarize_validation(candidate['evidence'])
            relative=(base['score']-summary['score'])/base['score'] if base['score'] else None
            record={'dataset':dataset,'candidate':config,'fold':fold,'seed':SEEDS[0],'epochs':120,
                'reference':base,'candidate_validation':summary,'relative_score_improvement':relative,
                'screen_passed':bool(screen_pass(summary,base)),
                'reference_run':reference['run'],'candidate_run':candidate['run']}
            record['reference_mean_errors']=mean_errors(reference['evidence'])
            record['candidate_mean_errors']=mean_errors(candidate['evidence'])
            records.append(record);text='unavailable' if relative is None else f'{relative*100:.2f}%'
            lines.append(f'| {dataset} | {config} | 120 | {base["score"]:.4f} | {summary["score"]:.4f} | {text} | {base["mape_passes"]}/{base["cells"]} → {summary["mape_passes"]}/{summary["cells"]} | {base["worst_mae_pp"]:.3f} → {summary["worst_mae_pp"]:.3f} | {record["screen_passed"]} |')
        winner=state.get('screen_winner')
        if winner:
            keys=[trial_key(c,fold,s,240) for c in ('R0',winner) for s in SEEDS]
            if all(state['trials'].get(k,{}).get('state')=='complete' for k in keys):
                refs=[summarize_validation(state['trials'][trial_key('R0',fold,s,240)]['evidence']) for s in SEEDS]
                candidates=[summarize_validation(state['trials'][trial_key(winner,fold,s,240)]['evidence']) for s in SEEDS]
                confirmations.append({'dataset':dataset,'candidate':winner,'budget':240,
                    'seeds':list(SEEDS),'references':refs,'candidates':candidates,
                    'confirmation_passed':bool(confirmation_pass(candidates,refs))})
    if not records:lines+=['','实际候选对照尚未完成；不能把注册配置或流程测试当作算法收益。']
    lines+=['','## 曲线各阶段误差（pp，R0 → 候选）','',
        '| 数据集/候选 | 起点 | 全程MAE | 后期MAE | 终点 | 阶段下降量 |',
        '|---|---:|---:|---:|---:|---:|']
    def pair(record,key):
        parts=[]
        for name in ('reference_mean_errors','candidate_mean_errors'):
            value=record[name][key]
            text='unavailable' if value['mean'] is None else f'{value["mean"]:.3f}'
            if value['available']!=value['requested']:text+=f' ({value["available"]}/{value["requested"]})'
            parts.append(text)
        return ' → '.join(parts)
    for record in records:
        lines.append('| '+record['dataset']+'/'+record['candidate']+' | '+' | '.join(pair(record,k) for k in
            ('start_error_pp','mae_pp','tail_mae_pp','endpoint_error_pp','stage_drop_max_pp'))+' |')
    lines+=['','阶段证据不足时明确列可评价分母，不填零；图集需另外审阅局部跳变和加速方向。','', '## 三种子同预算确认','']
    for record in confirmations:lines.append(f'- {record["dataset"]} 候选 {record["candidate"]}：通过={record["confirmation_passed"]}，种子20260930–20260932，预算240轮。')
    if not confirmations:lines.append('尚无完整的候选与 R0 配对三种子对照，不能宣称稳定算法收益；R0 本身的三种子确认单独列在下方。')
    lines+=['','## R0 三种子确认进度','']
    for dataset in workflow['sequence']:
        state=workflow['datasets'][dataset];fold=state['folds'][0]
        completed=sum(state['trials'].get(trial_key('R0',fold,s,240),{}).get('state')=='complete' for s in SEEDS)
        lines.append(f'- {dataset}：{completed}/3，首折 {fold}，预算240轮；该进度不代表参考曲线精度目标已达到。')
    lines+=['','## 真实验证预测图','',
        '以下图来自实际模型和原始观测点。数字筛选资格与图形视觉审阅分别记录；图集存在不等于精度通过。','']
    labels={'UConn_ISU_ILCC':'UConn','Zhu2022_NCA':'NCA'}
    for record in records:
        label=labels.get(record['dataset'],record['dataset'])
        links=[]
        for config in ('R0',record['candidate']):
            report=root/'trial_reports'/f'{label}_{config}'/'validation'
            pdf=report/'all_cells.pdf'
            if pdf.exists():
                links.append(f'[{config} 全部验证曲线]({pdf.resolve().as_posix()})')
            else:links.append(f'{config} 图集待导出')
        lines.append(f'- {record["dataset"]}/{record["candidate"]}：'+ '；'.join(links))
    lines+=['','分数、MAPE通过数和最差电芯保护条件一起决定候选资格。筛选通过不等于最终TEST达标，也不代表全部曲线达到参考水准。完整51槽位覆盖与最终图集核验仍须完成。','']
    (root/'validation_comparisons.md').write_text('\n'.join(lines),encoding='utf-8')
    save_json(root/'validation_comparisons.json',{'utc':datetime.now(timezone.utc).isoformat(),
        'screen_comparisons':records,'confirmation_comparisons':confirmations,
        'test_metrics_consumed':False,'goal_complete':False})
    print({'completed_screen_comparisons':len(records),'completed_confirmation_comparisons':len(confirmations)})


if __name__=='__main__':main()
