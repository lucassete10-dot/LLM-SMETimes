"""Summarize independently replayed inner experiments; never select on outer data."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.prepare_same_type_forecast import save_json, sha
from scripts.verify_multiview_pilot import curve_metrics, selection_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    root = parser.parse_args().run
    read = lambda path: json.loads(path.read_text(encoding='utf-8'))
    cfg = read(root / 'config.json')
    proof = read(root / 'independent_input_verification.json')
    assert proof['passed'] and not cfg['smoke'] and len(cfg['seeds']) == 5
    rows = [r for r in read(root / 'verified_cell_metrics.json') if r['role'] == 'validation']
    baseline_root = Path(cfg['baseline_inner'])
    assert sha(baseline_root / 'config.json') == cfg['baseline_config_sha256']
    for plan in cfg['plans']:
        for seed in cfg['seeds']:
            folder = baseline_root / plan['name'] / str(seed)
            chosen = read(folder / 'selection.json')['selected']
            source = folder / f"candidate_{chosen['candidate']}" / 'validation_predictions.npz'
            recorded = read(root / plan['name'] / str(seed) / 'concat' / 'summary.json')
            assert sha(source) == recorded['baseline_array_sha256']
            group = []
            with np.load(source) as arrays:
                with np.load(root / plan['name'] / str(seed) / 'concat' / 'validation_predictions.npz') as new:
                    for key in plan['validation']:
                        c, y, p = [arrays[key + '__100__' + field] for field in ('cycle', 'truth', 'prediction')]
                        for field, value in [('cycle', c), ('truth', y)]:
                            np.testing.assert_array_equal(value, new[key + '__100__' + field])
                        row = dict(fold=plan['name'], seed=seed, mode='frozen_baseline', role='validation',
                                   cell_id=key, **curve_metrics(c, y, p))
                        rows.append(row)
                        group.append(row)
            np.testing.assert_allclose(selection_score(group), chosen['score'], rtol=0, atol=1e-12)
    fields = ['mape_percent', 'late_mape', 'rmse_soh', 'mae_soh', 'endpoint_pp', 'drop_error_pp']
    summaries = []
    tasks = read(root / 'task_results.json')
    producer = read(root / 'summary.json')
    for mode in ['frozen_baseline', 'concat', 'gate']:
        seed_rows = []
        for seed in cfg['seeds']:
            folds = []
            for plan in cfg['plans']:
                group = [r for r in rows if r['mode'] == mode and r['seed'] == seed and r['fold'] == plan['name']]
                assert {r['cell_id'] for r in group} == set(plan['validation']) and len(group) == len(plan['validation'])
                folds.append(dict(score=selection_score(group), **{f: float(np.mean([r[f] for r in group])) for f in fields}))
            seed_rows.append(dict(seed=seed, **{f: float(np.mean([r[f] for r in folds])) for f in ['score'] + fields}))
        summary = dict(mode=mode, by_seed=seed_rows, validation_records=sum(r['mode'] == mode for r in rows),
                       **{f: dict(mean=float(np.mean([r[f] for r in seed_rows])),
                                  seed_sd=float(np.std([r[f] for r in seed_rows], ddof=1))) for f in ['score'] + fields})
        if mode != 'frozen_baseline':
            old = next(r for r in producer['candidates'] if r['mode'] == mode)
            np.testing.assert_allclose(summary['score']['mean'], old['score_mean'], rtol=0, atol=1e-12)
            np.testing.assert_allclose(summary['score']['seed_sd'], old['score_seed_sd'], rtol=0, atol=1e-12)
            selected = [r for r in tasks if r['mode'] == mode]
            counts = {r['parameter_count'] for r in selected}
            assert len(counts) == 1
            summary.update(parameter_count=counts.pop(), training_seconds_total=sum(r['fit_seconds'] for r in selected),
                           inference_curve_ms_median=float(np.median([r['inference_curve_ms'] for r in rows if r['mode'] == mode])))
        summaries.append(summary)
    base_score = summaries[0]['score']['mean']
    for row in summaries[1:]:
        row['improved'] = row['score']['mean'] < base_score
    assert not any(row['improved'] for row in summaries[1:]), 'Unexpected improvement requires registered adoption review'
    save_json(root / 'verified_comparison_cell_metrics.json', rows)
    save_json(root / 'verified_comparison_summary.json', dict(summaries=summaries, passed=True,
              aggregation='cell equal-weight within each inner fold; fold equal-weight within seed; five-seed mean/sample SD',
              note='Validation checkpoint selection results, not outer performance or blind-test accuracy. Score is not MAPE.',
              adopted=False, outer_expansion_authorized=False, decision='retain frozen prior schemes',
              source_sha256=sha(__file__), proof_sha256=sha(root / 'independent_input_verification.json'), goal_complete=False))
    lines = ['# 多视角有限训练侧比较：核验结果', '',
             'MIT固定5折×5种子×2配置，共50个模型。全部5360条训练/验证曲线已逐点重放一致；训练侧填补与标准化重新核算通过。', '',
             '两种新结构均未胜出，保留已有方案，不开展新候选的外层评测，也不追加配置搜索。', '',
             '| 设置 | 综合score（不是MAPE） | MAPE % | 后半MAPE % | 末点误差 pp | drop误差 pp |',
             '|---|---:|---:|---:|---:|---:|']
    for row in summaries:
        values = [f"{row[f]['mean']:.4f} ± {row[f]['seed_sd']:.4f}" for f in ['score', 'mape_percent', 'late_mape', 'endpoint_pp', 'drop_error_pp']]
        lines.append('| ' + row['mode'] + ' | ' + ' | '.join(values) + ' |')
    lines += ['', '这些数字来自内层验证：验证电芯参与checkpoint选择，不能当作最终未见电芯准确率。每折电芯等权，折内汇总后折等权；标准差是5种子样本标准差，不是置信区间。', '',
              '拼接模型10577参数，门控模型10724参数。完整耗时、逐芯指标、推理耗时与来源SHA见同目录verified_comparison_summary.json和verified_comparison_cell_metrics.json。CPU批量整条曲线推理耗时不是STM32耗时。', '',
              '输入是早期电学状态、历史演化、已知工况三个视角；直接按请求的绝对循环坐标输出SOH，没有未来容量/真实寿命输入。来源及原文核验边界见预登记和论文阅读卡；这是受启发的候选，不是精确论文复现。', '',
              '核验范围：保存权重重放、训练统计、指标算术、checkpoint排序与输入隔离；未独立重做50次优化器训练。实验本身包含固定5种子，完整日志、配置、权重保留。', '',
              '四队列最终交付尚未完成，Goal继续保持active。']
    # Generated experiment report, not an edit to the frozen registration/source snapshot.
    (root / 'verified_comparison_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
