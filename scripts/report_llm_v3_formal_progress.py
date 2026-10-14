"""Describe locked TEST predictions with full planned denominators, without ranking configs."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from battery_smetimes.llm_curve_quality_v3 import curve_metrics_v3, reference_pass
from battery_smetimes.llm_v3_reporting import quality_summary
from battery_smetimes.llm_v3_workflow import SEEDS


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    root = ap.parse_args().root
    workflow = read(root / 'workflow.json')
    records = []
    lines = ['# v3 正式留出预测进度', '',
             '只描述已锁定配置的 TEST 预测；不据此排名候选或修改训练。未完成运行保留在计划分母中，均值只覆盖已评价数据，不能解释为全数据集成绩。', '',
             '| 数据集 | 正式槽位 | TEST电芯×种子 已评价/计划 | MAPE≤3% 已通过/计划 | 已评价平均MAPE | 三种子完整电芯/计划电芯 | 稳定参考数值通过/计划电芯 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for dataset in workflow['sequence']:
        state = workflow['datasets'][dataset]
        requested_slots = len(state['folds']) * len(SEEDS)
        requested = sum(state['templates'][fold]['role_counts']['test']
                        for fold in state['folds']) * len(SEEDS)
        lock = root / 'selection_locks' / f'{dataset}.json'
        if not lock.exists() or read(lock) != state['selection_lock']:
            raise ValueError('No valid selection lock: ' + dataset)
        rows, sources = [], []
        for key, slot in state['formal_slots'].items():
            if not slot.get('test_verified'):
                continue
            run = Path(slot['run'])
            cfg = read(run / 'config.json')
            proof = read(run / 'repair_verification.json')
            if (not proof['passed'] or set(proof['role_scope']) != {'training', 'validation', 'test'}
                    or slot['selection_lock_sha256'] != digest(lock)
                    or proof['model_sha256'] != digest(run / 'model.pt')):
                raise ValueError('TEST/model/lock evidence changed: ' + key)
            archive = run / 'test_audited_predictions.npz'
            expected_ids = set(read(run / 'strict_acceptance.json')['test']['by_cell'])
            observed_ids = set()
            suffix = f'__{cfg["origin"]}__cycle'
            with np.load(archive, allow_pickle=False) as data:
                for array_key in data.files:
                    if not array_key.endswith(suffix):
                        continue
                    cid = array_key[:-len(suffix)]
                    prefix = f'{cid}__{cfg["origin"]}__'
                    observed_ids.add(cid)
                    row = curve_metrics_v3(*[data[prefix + k] for k in ('cycle', 'truth', 'prediction')])
                    row.update(cell_id=cid, fold=cfg['fold'], seed=cfg['seed'])
                    rows.append(row)
            if observed_ids != expected_ids:
                raise ValueError('TEST archive coverage differs from replay proof')
            sources.append({'slot': key, 'run': str(run.resolve()), 'model_sha256': proof['model_sha256'],
                            'predictions': str(archive.resolve()), 'predictions_sha256': digest(archive)})
        if len({(r['cell_id'], r['fold'], r['seed']) for r in rows}) != len(rows):
            raise ValueError('Duplicate TEST prediction instances')
        quality = quality_summary(rows, requested)
        primary = sum(r['mape_percent'] <= 3 for r in rows)
        mean = float(np.mean([r['mape_percent'] for r in rows])) if rows else None
        record = {'dataset': dataset, 'winner': state['winner'], 'verified_formal_slots': len(sources),
                  'planned_formal_slots': requested_slots, 'planned_test_seed_instances': requested,
                  'evaluated_test_seed_instances': len(rows), 'primary_passes': primary,
                  'evaluated_mean_mape_percent': mean, 'quality': quality,
                  'source_predictions': sources, 'per_cell_seed': rows,
                  'dataset_formal_complete': len(sources) == requested_slots}
        records.append(record)
        mean_text = 'unavailable' if mean is None else f'{mean:.3f}%'
        cells = requested // len(SEEDS)
        lines.append(f'| {dataset} | {len(sources)}/{requested_slots} | {len(rows)}/{requested} | '
                     f'{primary}/{requested} | {mean_text} | '
                     f'{quality["three_seed_complete_cell_fold_groups"]}/{cells} | '
                     f'{quality["stable_reference_cell_fold_groups"]}/{cells} |')
    lines += ['', '稳定参考数值通过要求至少2/3种子满足参考指标，且全部3种子 MAPE≤3%；数字通过还需视觉核对。UConn 的观测点达标不证明逐循环连续预测精度。HUST/UConn 的固定 TEST 子集不是全部源电芯。', '',
              '未完成槽位与缺失种子尚未产生结果，不能填零误差或视为已通过。此进度表不替代最终全覆盖图集、失败名单及完成审计。', '',
              f'[逐电芯原始重算指标及来源哈希]({(root / "formal_progress.json").resolve().as_posix()})', '']
    (root / 'formal_progress.md').write_text('\n'.join(lines), encoding='utf-8')
    (root / 'formal_progress.json').write_text(json.dumps(
        {'utc': datetime.now(timezone.utc).isoformat(), 'datasets': records,
         'test_used_for_selection': False, 'visual_review_not_certified_by_this_script': True,
         'source_sha256': digest(Path(__file__)), 'goal_complete': False},
        ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'formal_slots': sum(r['verified_formal_slots'] for r in records),
                      'planned_slots': sum(r['planned_formal_slots'] for r in records)}))


if __name__ == '__main__':
    main()
