"""Describe first-fold R0 TRAIN/VAL errors from audited raw predictions only."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from battery_smetimes.llm_curve_quality_v3 import curve_metrics_v3, reference_pass


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def role_summary(run, origin, role):
    archive = run / f'{role}_audited_predictions.npz'
    rows = {}
    suffix = f'__{origin}__cycle'
    with np.load(archive, allow_pickle=False) as data:
        for key in data.files:
            if not key.endswith(suffix):
                continue
            cid = key[:-len(suffix)]
            prefix = f'{cid}__{origin}__'
            x, y, p = [data[prefix + k] for k in ('cycle', 'truth', 'prediction')]
            row = curve_metrics_v3(x, y, p)
            row['endpoint_signed_error_pp'] = float(100 * (p[-1] - y[-1]))
            rows[cid] = row
    if not rows:
        raise ValueError('No raw curves: ' + str(archive))
    mean = {k: float(np.mean([row[k] for row in rows.values()]))
            for k in ('mape_percent', 'mae_pp', 'tail_mae_pp')}
    failures = {name: [cid for cid, row in rows.items()
                      if row[key] is not None and row[key] > limit]
                for name, key, limit in (
                    ('mape_above_3_percent', 'mape_percent', 3),
                    ('late_mape_above_3_percent', 'late_mape_percent', 3),
                    ('start_above_1_pp', 'start_error_pp', 1),
                    ('endpoint_above_3_pp', 'endpoint_error_pp', 3),
                    ('stage_above_3_pp', 'stage_drop_max_pp', 3))}
    failures['stage_unavailable'] = [cid for cid, row in rows.items()
                                     if row['stage_drop_max_pp'] is None]
    return {'cells': len(rows), 'mean': mean,
            'primary_passes': sum(row['mape_percent'] <= 3 for row in rows.values()),
            'reference_numeric_passes': sum(reference_pass(row) for row in rows.values()),
            'failure_cells': failures, 'by_cell': rows,
            'raw_predictions': str(archive.resolve()), 'raw_sha256': digest(archive)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    root = ap.parse_args().root
    records = []
    lines = ['# 五数据集 R0 训练与验证差距', '',
             '首个冻结划分、seed20260930、120轮预算。直接由原始已审计预测重新计算；不读取 TEST，不用训练成绩挑候选。各数据集工况和划分不同，不据此排名。', '',
             '| 数据集 | TRAIN/VAL电芯 | 平均MAPE TRAIN→VAL | MAE/pp TRAIN→VAL | 后期MAE/pp TRAIN→VAL | VAL−TRAIN MAE/pp |',
             '|---|---|---|---|---|---:|']
    for task in read(root / 'R0_queue.json')['tasks']:
        if task['state'] != 'complete':
            raise ValueError('R0 quick round is incomplete')
        run = Path(task['run'])
        cfg = read(run / 'config.json')
        proof = read(run / 'repair_verification.json')
        if any(cfg[key] != task[key] for key in ('dataset', 'fold', 'seed')):
            raise ValueError('R0 queue/config identity mismatch')
        if (cfg.get('selection_version') != 'curve_quality_v3' or not proof['passed']
                or proof['model_sha256'] != digest(run / 'model.pt')
                or not {'training', 'validation'} <= set(proof['role_scope'])):
            raise ValueError('R0 evidence invalid: ' + str(run))
        train = role_summary(run, cfg['origin'], 'training')
        val = role_summary(run, cfg['origin'], 'validation')
        if (train['cells'] != task['role_counts']['train']
                or val['cells'] != task['role_counts']['validation']
                or set(train['by_cell']) != set(read(run / 'TRAIN_metrics.json'))
                or set(val['by_cell']) != set(read(run / 'selected_metrics.json'))):
            raise ValueError('R0 raw prediction coverage mismatch')
        gap = val['mean']['mae_pp'] - train['mean']['mae_pp']
        records.append({'dataset': cfg['dataset'], 'fold': cfg['fold'], 'seed': cfg['seed'],
                        'run': str(run.resolve()), 'model_sha256': proof['model_sha256'],
                        'training': train, 'validation': val, 'mae_gap_pp': gap})
        pairs = [f'{train["mean"][k]:.3f} → {val["mean"][k]:.3f}'
                 for k in ('mape_percent', 'mae_pp', 'tail_mae_pp')]
        lines.append(f'| {cfg["dataset"]} | {train["cells"]}/{val["cells"]} | '
                     + ' | '.join(pairs) + f' | {gap:.3f} |')
    lines += ['', '## 验证失败类型', '',
              '类型可重叠；不足采样记 unavailable，不能填零。尾段偏差需与已审阅原始曲线一起判断，差距本身不能证明过拟合原因。', '']
    for record in records:
        val = record['validation']
        lines.append(f'### {record["dataset"]}')
        lines += ['', f'MAPE≤3%：{val["primary_passes"]}/{val["cells"]}；参考数值通过：{val["reference_numeric_passes"]}/{val["cells"]}。', '']
        for name, ids in val['failure_cells'].items():
            lines.append(f'- {name}：{len(ids)}/{val["cells"]}；' + (', '.join(ids) if ids else '无'))
        lines.append('')
    lines += ['训练均值小于验证均值不等于 TRAIN 完美拟合，也不等于所有验证电芯失败。用逐电芯指标和真实图定位起点、后期及阶段误差；不通过加轮数或平滑真值掩盖这些问题。', '',
              f'[完整数值、逐电芯名单和原始预测哈希]({(root / "R0_train_validation_gap.json").resolve().as_posix()})', '']
    (root / 'R0_train_validation_gap.md').write_text('\n'.join(lines), encoding='utf-8')
    (root / 'R0_train_validation_gap.json').write_text(json.dumps(
        {'records': records, 'test_consumed': False, 'source_sha256': digest(Path(__file__)),
         'goal_complete': False}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'datasets': len(records), 'raw_curve_instances': sum(
        row[role]['cells'] for row in records for role in ('training', 'validation'))}))


if __name__ == '__main__':
    main()
