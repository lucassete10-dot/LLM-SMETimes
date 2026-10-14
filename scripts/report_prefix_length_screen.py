"""Human-readable baseline screen with explicit common-cohort coverage."""
import argparse
import json
from pathlib import Path
from scripts.prepare_same_type_forecast import save_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);args=ap.parse_args()
    root=args.run
    proof=json.loads((root/'independent_replay_verification.json').read_text())
    assert proof['passed']
    cfg=json.loads((root/'config.json').read_text())
    summary=json.loads((root/'summary.json').read_text())
    coverage=json.loads((root/'coverage.json').read_text())
    lines=['# 前期输入长度对照：训练侧初筛','',
        '以下是固定一折内层验证的基线结果；不是全部折、不是已优化主模型的结果。',
        '输入10/20/30/50/75/100个绝对源循环，统一预测第101循环起同一段真实未来。',
        'XJTU最短输入交集仅部分工况：不能把这张表推广至全部55颗电池。',
        '各候选等额预算，外层测试电芯未读取；容量标签和后续电信号污染测试通过。',
        '',f'核验模型：{proof["verified_saved_models"]}；核验预测曲线：{proof["verified_curves"]}。','',
        '|折|角色|原数量|共同可用数量|','|---|---|---:|---:|']
    for row in coverage:
        lines.append(f'|{row["fold"]}|{row["role"]}|{row["original"]}|{len(row["matched_cells"])}|')
    lines+=['','## 全部候选结果','',
        '曲线通过率还包括MAE、后期/终点/下降/阶段误差、方向及覆盖条件。',
        '候选逐种子评测后汇总；种子SD不是置信区间；确定性Ridge重复SD=0。','',
        '|数据集|输入循环|模型/参数|MAPE均值±SD(%)|MAE(pp)|末20%MAPE(%)|终点(pp)|阶段误差(pp)|曲线通过率|',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for row in summary['rows']:
        m=row['mape_percent'];sd='NA' if m['sd'] is None else f'{m["sd"]:.3f}'
        lines.append(f'|{row["dataset"]}|{row["origin"]}|{row["model"]}/{row["parameter"]}|'
            f'{m["mean"]:.3f} ± {sd}|{100*row["mae_soh"]["mean"]:.3f}|'
            f'{row["late20_mape_percent"]["mean"]:.3f}|{row["endpoint_pp"]["mean"]:.3f}|'
            f'{row["stage_error_pp"]["mean"]:.3f}|{100*row["curve_screen_pass"]["mean"]:.1f}%|')
    lines+=['','最短输入长度结论尚未成立：必须继续全部折、覆盖充足队列和主模型确认。',
        '寿命比例属于回顾性分析；未来真实寿命不作为模型输入。']
    (root/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    records=json.loads(Path('experiments/soh_research_v5/audits/20260930T074750_792569Z/cell_registry.json').read_text())['cells']
    references={k:v['reference_established_source_cycle'] for k,v in records.items()
        if v['metadata']['dataset']=='XJTU'}
    save_json(root/'xjtu_reference_coverage.json',dict(
        source='frozen physical-cell registry; metadata only, no outer scores',
        eligible_reference_counts={str(origin):sum(v<=origin for v in references.values()) for origin in cfg['origins']},
        reference_cycles=references,
        note='Reference readiness alone is not sufficient input availability; common-cohort screening uses all-origin intersection.'))
    print(str((root/'report.md').resolve()))


if __name__=='__main__':main()
