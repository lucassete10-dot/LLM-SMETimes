"""Evidence-backed cumulative baseline overview; no training or selection."""
import argparse,json
from datetime import datetime,timezone
from pathlib import Path
from battery_smetimes.llm_reporting import summarize_curve_rows
from scripts.audit_llm_curve_repair import fingerprint
from scripts.prepare_same_type_forecast import save_json

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--queue',type=Path,required=True);args=ap.parse_args()
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    root=args.queue;q=read(root/'queue.json');thresholds=read(Path('configs/llm_curve_reporting_practical_v1.json'))['numeric_thresholds']
    out=root/'reports';out.mkdir(exist_ok=True);records={};sources={}
    lines=['# LLM + SMETimes 各数据集阶段总览','',
        '累计已完成并核验的基线运行；各数据集折次、种子和采样密度不同，不据此进行算法排名。MAPE 为等电芯×种子权重均值，全部实用指标通过数另列。未完成运行不计入成绩，也不记成零误差。','',
        f"| 数据集 | 完成/计划训练 | 测试电芯×种子实例 | MAPE≤{thresholds['mape_percent']:g}% | 平均 MAPE | 后期 MAE/pp | 实用联合通过 |",
        '|---|---:|---:|---:|---:|---:|---:|']
    for dataset in q['sequence']:
        planned=[t for t in q['tasks'] if t['dataset']==dataset];done=[t for t in planned if t['state']=='complete']
        rows=[];requested=0
        for task in done:
            run=Path(task['run']);verification=read(run/'repair_verification.json')
            if not verification['passed'] or fingerprint(run/'model.pt')!=task['model_sha256'] or verification['model_sha256']!=task['model_sha256']:raise ValueError('completed run evidence mismatch')
            path=run/'strict_acceptance.json';record=read(path)['test'];sources[str(path.resolve())]=fingerprint(path)
            if record['cells_requested']!=task['role_counts']['test']:raise ValueError('TEST denominator mismatch')
            requested+=record['cells_requested']
            rows.extend({'cell_id':cid,'fold':task['fold'],'seed':task['seed'],**metric} for cid,metric in record['by_cell'].items())
        stats=summarize_curve_rows(rows,requested,thresholds)
        records[dataset]={'completed_runs':len(done),'planned_runs':len(planned),'test':stats}
        if not done:
            lines.append(f'| {dataset} | 0/{len(planned)} | 0 | 待完成 | 待完成 | 待完成 | 待完成 |')
            continue
        fmt=lambda v:f'{v:.3f}' if v is not None else '待完成'
        lines.append(f"| {dataset} | {len(done)}/{len(planned)} | {requested} | {stats['mape_target_passes']}/{requested} | {fmt(stats['mape_percent'])}% | {fmt(stats['tail_mae_pp'])} | {stats['practical_numeric_passes']}/{requested} |")
    lines+=['','## 图集与范围','']
    for dataset,record in records.items():
        candidates=sorted(p for p in (out/dataset/'progress').glob('completed_*/all_cells.pdf') if read(p.parent/'summary.json')['seeds']==q['seeds'])
        if candidates:
            pdf=candidates[-1];summary=read(pdf.parent/'summary.json');manifest=read(pdf.parent/'figure_manifest.json')
            records[dataset]['figure_report']={'pdf':str(pdf.resolve()),'runs':summary['runs'],'figures':len(manifest['figures']),'visual_review':manifest['visual_review'],'primary_target_percent':summary['primary_reporting_standard']['numeric_thresholds']['mape_percent']}
            lines.append(f"- [{dataset} 图集]({pdf.resolve().as_posix()})：该版本覆盖 {len(manifest['figures'])} 颗绘图电芯，报告轮数 {records[dataset]['figure_report']['runs']}；以图内 fold、role、种子数为准。")
        else:lines.append(f'- {dataset}：首轮未完成，暂无核验后的图集。')
    lines+=['','表中统计所有已完成运行，图集可能来自较早阶段快照。TEST 未参与候选算法选择；未达标、缺失和覆盖分母保留在各数据集原始报告。单种子不是稳定性结论，稀疏 RPT 不能证明逐循环曲线准确。','',
        f"完整目标仍包括{len(q['tasks'])}次基线训练、每个数据集有限算法对照及确认、最终全范围核验与交付；本总览不是 Goal 完成声明。"]
    (out/'MVP_overview.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save_json(out/'MVP_overview.json',{'utc':datetime.now(timezone.utc).isoformat(),'datasets':records,'source_sha256':sources,'report_source_sha256':fingerprint(Path(__file__)),'goal_complete':False})
    print(json.dumps({'primary_mape_target_percent':thresholds['mape_percent'],'datasets':{k:{'completed':v['completed_runs'],'requested_test_instances':v['test']['requested_cell_seed_instances'],'MAPE_target_passes':v['test']['mape_target_passes']} for k,v in records.items()}},indent=2))

if __name__=='__main__':main()
