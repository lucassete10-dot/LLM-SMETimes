"""Recompute complete A/B and per-cell tables without altering adoption history."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np

from scripts.audit_four_cohort_progress import SOURCES
from scripts.audit_rate_blend_metrics import metrics, summarize, checks
from scripts.prepare_same_type_forecast import save_json, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    root = args.out or Path('results/goal_v2_tables') / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    root.mkdir(parents=True, exist_ok=False)
    print('OUTPUT=' + str(root.resolve()), flush=True)
    shutil.copy2(__file__, root / Path(__file__).name)
    cohorts, cell_rows, seed_rows, point_sources, failures = [], [], [], [], []
    for dataset, path, count, proof_name, retained in SOURCES:
        source = Path(path)
        read = lambda p: json.loads(p.read_text(encoding='utf-8'))
        cfg, historical = read(source / 'config.json'), read(source / 'summary.json')
        assert read(source / proof_name)['passed']
        if 'plans' in cfg:
            plans, seeds = cfg['plans'], cfg['seeds']
        else:
            parent = Path(cfg['training_source'])
            assert sha(parent / 'config.json') == cfg['training_config_sha256']
            pcfg = read(parent / 'config.json')
            plans, seeds = [p for p in pcfg['plans'] if p['dataset'] == cfg['dataset']], pcfg['seeds']
        expected = [k for plan in plans for k in plan['test']]
        assert len(expected) == len(set(expected)) == count and len(seeds) == 5
        rows = []
        references = {}
        for plan in plans:
            assert not set(plan['train']) & set(plan['validation'])
            assert not set(plan['train'] + plan['validation']) & set(plan['test'])
            for seed in seeds:
                file = source / plan['name'] / str(seed) / 'test_predictions.npz'
                with np.load(file) as arrays:
                    assert set(arrays.files) == {k + '__100__' + f for k in plan['test'] for f in ('cycle', 'truth', 'prediction')}
                    for key in plan['test']:
                        c, y, p = [arrays[key + '__100__' + f] for f in ('cycle', 'truth', 'prediction')]
                        if key in references:
                            np.testing.assert_array_equal(c, references[key][0])
                            np.testing.assert_array_equal(y, references[key][1])
                        else:
                            references[key] = (c.copy(), y.copy())
                        e = p - y
                        ss = float(np.sum((y - y.mean()) ** 2))
                        rows.append(dict(dataset=dataset, cell_id=key, seed=seed, fold=plan['name'],
                            **metrics(c, y, p), mae_soh=float(np.mean(abs(e))),
                            r2=1 - float(e @ e) / ss if ss else None, r2_undefined=not bool(ss),
                            start_error_pp=float(100 * abs(e[0])), future_points=len(c),
                            future_first_cycle=float(c[0]), future_last_cycle=float(c[-1]),
                            future_span=float(c[-1] - c[0]), short_coverage=bool(len(c) < 10 or c[-1] - c[0] < 100),
                            prediction_file=str(file.resolve()), prediction_sha256=sha(file)))
                point_sources.append(dict(dataset=dataset, fold=plan['name'], seed=seed,
                                          file=str(file.resolve()), sha256=sha(file), cells=plan['test']))
        summary, cells = summarize(rows)
        for field, value in summary.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    np.testing.assert_allclose(v, historical['new'][field][k], rtol=0, atol=1e-10)
            else:
                np.testing.assert_allclose(value, historical['new'][field], rtol=0, atol=1e-10)
        old = historical['baseline']
        gates = checks(summary, old)
        assert gates == historical['gates']
        a = {k: gates[k] for k in ('full_mape', 'late_mape', 'rmse')}
        b = {k: gates[k] for k in ('endpoint_mean', 'drop_mean', 'joint_fraction', 'severe_fraction',
                                   'full_no_regression', 'endpoint_p90', 'drop_error_p90')}
        if dataset == 'NASA':
            b['joint_fraction'] = sum(c['joint_pass'] for c in cells) == 4
        additional = {}
        for field in ('mae_soh', 'r2', 'start_error_pp'):
            means = []
            for seed in seeds:
                values = [r[field] for r in rows if r['seed'] == seed and r[field] is not None]
                means.append(float(np.mean(values)) if values else None)
            finite = [v for v in means if v is not None]
            additional[field] = dict(mean=float(np.mean(finite)) if finite else None,
                                    seed_sd=float(np.std(finite, ddof=1)) if len(finite) > 1 else None)
        for cell in cells:
            group = [r for r in rows if r['cell_id'] == cell['cell_id']]
            for field in ('mae_soh', 'r2', 'start_error_pp'):
                values = [r[field] for r in group if r[field] is not None]
                cell[field] = float(np.mean(values)) if values else None
                cell[field + '_seed_sd'] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            for field in ('mape_percent', 'late_mape', 'rmse_soh', 'endpoint_pp', 'drop_error_pp'):
                cell[field + '_seed_sd'] = float(np.std([r[field] for r in group], ddof=1))
            cell.update(dataset=dataset, fold=group[0]['fold'], future_points=group[0]['future_points'],
                        future_first_cycle=group[0]['future_first_cycle'], future_last_cycle=group[0]['future_last_cycle'],
                        future_span=group[0]['future_span'], short_coverage=group[0]['short_coverage'],
                        r2_undefined=group[0]['r2_undefined'], mape_6_pass=cell['mape_percent'] <= 6)
            violations = [f for f, limit in [('mape_percent', 6), ('late_mape', 8), ('endpoint_pp', 8), ('drop_error_pp', 5)] if cell[f] > limit]
            if cell['severe_flat_failure']:
                violations.append('severe_flat_failure')
            cell['failed_curve_criteria'] = violations
            cell['failure_interpretation'] = 'prediction mismatch; not evidence of invalid data' if violations else ''
            if violations:
                failures.append(cell.copy())
            cell_rows.append(cell)
        seed_rows.extend(rows)
        worst = max(cells, key=lambda r: r['mape_percent'])
        cohort = dict(dataset=dataset, cells=count, summary=summary, additional_metrics=additional,
            a_numeric_gates=a, a_numeric_pass=all(a.values()), b_gates=b, b_pass=all(b.values()),
            mape_6_pass_count=sum(c['mape_6_pass'] for c in cells),
            mape_6_pass_fraction=sum(c['mape_6_pass'] for c in cells) / count,
            joint_pass_count=sum(c['joint_pass'] for c in cells), worst_cell=worst,
            short_coverage_cells=sum(c['short_coverage'] for c in cells),
            baseline_summary=old, source_run=str(source.resolve()), config_sha256=sha(source / 'config.json'),
            proof_file=str((source / proof_name).resolve()), proof_sha256=sha(source / proof_name),
            selection_status='retained reference' if retained else 'historical inner-locked candidate; final adoption review pending',
            repeat_interpretation='deterministic repeats, not uncertainty or independent seeds' if dataset == 'NASA' else '5 training seeds; sample SD, not CI')
        cohorts.append(cohort)
        print(dataset + ': numerical A=' + str(cohort['a_numeric_pass']) + '; joint=' + str(cohort['joint_pass_count']) + '/' + str(count), flush=True)
    assert len(cell_rows) == 248 and len(seed_rows) == 1240
    save_json(root / 'cohorts.json', cohorts)
    save_json(root / 'per_cell.json', cell_rows)
    save_json(root / 'per_cell_seed.json', seed_rows)
    save_json(root / 'curve_failures.json', failures)
    save_json(root / 'point_prediction_sources.json', point_sources)
    save_json(root / 'manifest.json', dict(source_sha256=sha(__file__), cells=248, cell_seed_records=1240,
              metric_recalculation_passed=True, numeric_A_all_pass=all(c['a_numeric_pass'] for c in cohorts),
              metric_definition='SOH 0–1; late cycle >= (first actual future cycle + last actual future cycle)/2; >100 only; cell equal-weight per seed then mean/sample SD',
              final_model_adoption='pending; this report never upgrades rejected candidates',
              delivery_visual_excel_review='pending', goal_complete=False))
    lines = ['# 四队列开发交叉验证：A级数值与B级增强表', '',
             '本报告仅汇总已锁定历史方案，尚未完成最终采用、全图及Excel核验，不据此宣布Goal完成。', '',
             '| 队列 | 全MAPE % | 后半MAPE % | RMSE SOH | 数值A | 逐芯MAPE≤6% | 联合通过 | 最差芯MAPE % |',
             '|---|---:|---:|---:|---|---:|---:|---:|']
    for r in cohorts:
        s = r['summary']
        lines.append(f"| {r['dataset']} | {s['mape_percent']['mean']:.4f} ± {s['mape_percent']['seed_sd']:.4f} | {s['late_mape']['mean']:.4f} | {s['rmse_soh']['mean']:.5f} | {r['a_numeric_pass']} | {r['mape_6_pass_count']}/{r['cells']} | {r['joint_pass_count']}/{r['cells']} | {r['worst_cell']['mape_percent']:.4f} |")
    lines += ['', 'NASA零重复SD不是零泛化不确定性。短覆盖、电芯ID、全部指标/种子SD、未过联合条件的正常芯及逐点NPZ来源见旁边JSON。', '',
              '后半段按实际未来首末循环坐标中点划分，不按数组点数一半；drop比较未来首末点下降差。SOH误差pp=0–1误差×100。', '',
              '失败原因列是预测形态/误差判据，不是断言该电芯损坏，更不能据此排除。最终数据完整性排除规则另交付。']
    (root / 'numeric_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
