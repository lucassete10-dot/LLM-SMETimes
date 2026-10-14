"""Inventory existing outer bundles; no fitting or outer-driven selection.

Contract classifications describe active prediction branches, not label leakage.
This audit is necessary evidence, not a replacement for independent replay.
"""
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import shutil

import joblib

from scripts.audit_four_cohort_progress import SOURCES
from scripts.prepare_same_type_forecast import sha, save_json


def active_paths(bundle, cohort):
    if cohort == 'MIT':
        weight = float(bundle['new_weight'])
        paths = []
        if weight < 1:
            paths.append(dict(branch='legacy', weight=1-weight,
                              compliant=False, reason='all raw prefix columns, including early Ah/Wh'))
        if weight > 0:
            paths.append(dict(branch='voltage_segments', weight=weight,
                              compliant=True, reason='explicit voltage-column allowlist'))
        return paths
    if cohort == 'XJTU':
        segment = bundle['mode'] == 'segment'
        return [dict(branch=bundle['mode'], compliant=segment,
                     reason='explicit V/VI/VIT allowlist' if segment else
                     'legacy raw prefix includes early partial-Ah channels')]
    if cohort == 'NASA':
        calibration = bundle['calibration']
        weight = float(bundle['weight']) if calibration is not None else 0.
        paths = []
        if weight < 1:
            paths.append(dict(branch='primary_legacy', weight=1-weight,
                              compliant=False, reason='primary uses all NASA raw columns including early Ah/Wh'))
        if calibration is not None:
            paths.append(dict(branch='calibration', weight=weight,
                              compliant=True, reason='fixed voltage/current/temperature columns'))
        # Even weight=1 evaluates the legacy primary in the current implementation.
        if weight == 1:
            paths.append(dict(branch='evaluated_zero_weight_primary', weight=0.,
                              compliant=False, reason='legacy computation still required by current predict implementation'))
        return paths
    if cohort == 'Zhu2022 NCA':
        known = bundle['mode'] in ('tail', 'reference_nca')
        return [dict(branch=bundle['mode'], compliant=known,
                     reason='early V/I signal prefix; capacity used only as training label' if known else
                     'unreviewed branch; requires manual audit')]
    raise ValueError(cohort)


def main():
    root = Path('experiments/goal_v2_input_audit') / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    root.mkdir(parents=True)
    print('RUN=' + str(root.resolve()), flush=True)
    shutil.copy2(__file__, root / 'audit_source.py')
    evidence, rows, reports = [], [], []
    prior = Path('experiments/four_cohort_progress_audit/20261006T152412_391404Z/cohort_results.json')
    metrics = {r['dataset']: r for r in json.loads(prior.read_text())}
    for cohort, path, count, proof_name, _ in SOURCES:
        source = Path(path)
        cfg = json.loads((source / 'config.json').read_text())
        proof = json.loads((source / proof_name).read_text())
        assert proof['passed'], 'historical independent replay failed'
        if 'plans' in cfg:
            plans, seeds = cfg['plans'], cfg['seeds']
        else:
            parent = Path(cfg['training_source']) / 'config.json'
            assert sha(parent) == cfg['training_config_sha256']
            parent_cfg = json.loads(parent.read_text())
            plans = [p for p in parent_cfg['plans'] if p['dataset'] == cfg['dataset']]
            seeds = parent_cfg['seeds']
            evidence.append(dict(file=str(parent.resolve()), sha256=sha(parent)))
        tests = [k for p in plans for k in p['test']]
        assert len(tests) == len(set(tests)) == count
        assert len(seeds) == len(set(seeds)) == 5
        cohort_rows = []
        for plan in plans:
            train, val, test = map(set, (plan['train'], plan['validation'], plan['test']))
            assert not (train & val or train & test or val & test)
            for seed in seeds:
                file = source / plan['name'] / str(seed) / 'model.joblib'
                bundle = joblib.load(file)
                assert set(bundle['train_ids']) == train | val
                assert len(bundle['train_ids']) == len(train | val)
                # Check nested local/global fitted models cannot include held-out cells.
                def walk(value):
                    if isinstance(value, dict):
                        if 'train_ids' in value:
                            assert set(value['train_ids']) <= train | val
                        for nested in value.values():
                            walk(nested)
                    elif isinstance(value, (list, tuple)):
                        for nested in value:
                            walk(nested)
                walk(bundle)
                paths = active_paths(bundle, cohort)
                row = dict(dataset=cohort, fold=plan['name'], seed=seed,
                           model_file=str(file.resolve()), model_sha256=sha(file),
                           train_cells=len(train | val), test_cells=len(test),
                           family=bundle.get('family'), branches=paths,
                           input_contract_pass=all(p['compliant'] for p in paths),
                           input_review_status='explicit_allowlist_verified' if all(p['compliant'] for p in paths) else 'derived_integral_provenance_pending_not_proven_leakage',
                           split_and_nested_train_ids_pass=True)
                cohort_rows.append(row)
        summary = metrics[cohort]['summary']
        gates = {name: summary[name]['mean'] <= limit for name, limit in
                 [('mape_percent', 6.), ('late_mape', 6.), ('rmse_soh', .04)]}
        report = dict(dataset=cohort, cells=count, fitted_tasks=len(cohort_rows),
                      compliant_tasks=sum(r['input_contract_pass'] for r in cohort_rows),
                      active_branch_counts=dict(Counter(p['branch'] for r in cohort_rows for p in r['branches'])),
                      numeric_A_gates=gates, numeric_A_pass=all(gates.values()),
                      input_contract_pass=all(r['input_contract_pass'] for r in cohort_rows),
                      metrics=summary, final_adopted=False)
        reports.append(report)
        rows.extend(cohort_rows)
        for name in ('config.json', proof_name, 'choice_lock.json', 'inner_lock.json', 'reference_lock.json'):
            file = source / name
            if file.exists():
                evidence.append(dict(file=str(file.resolve()), sha256=sha(file)))
        print(f"{cohort}: {report['compliant_tasks']}/{len(cohort_rows)} input-compliant tasks", flush=True)
    evidence.append(dict(file=str(prior.resolve()), sha256=sha(prior)))
    for path in ('battery_smetimes/mit_feature_blend.py', 'battery_smetimes/mit_segment_features.py',
                 'battery_smetimes/xjtu_segment_features.py', 'battery_smetimes/tail_selection.py',
                 'battery_smetimes/tail_boost.py', 'battery_smetimes/nasa_slope_blend.py',
                 'battery_smetimes/nasa_cycle_calibration.py', 'battery_smetimes/research_baselines.py',
                 'scripts/prepare_zhu2022_nca.py'):
        file = Path(path)
        evidence.append(dict(file=str(file.resolve()), sha256=sha(file)))
    save_json(root / 'tasks.json', rows)
    save_json(root / 'cohorts.json', reports)
    save_json(root / 'source_evidence.json', evidence)
    save_json(root / 'config.json', dict(operation='read-only bundle inventory, no selection or fitting',
              source_sha256=sha(__file__), goal_complete=False,
              limits='branch/source audit; numeric gates refer to prior independently recomputed predictions; not full completion audit'))
    lines = ['# Goal v2 input inventory', '',
             'All existing outer model bundles inspected. No model changed or selected.',
             'Early current integrals are not proof of future-label leakage. They require an explicit contract decision; legacy raw-column paths are not yet eligible as v2 main models.', '',
             '|Cohort|Cells|Tasks|Explicit-allowlist verified tasks|Numeric A passes|',
             '|---|---:|---:|---:|---|']
    for r in reports:
        lines.append(f"|{r['dataset']}|{r['cells']}|{r['fitted_tasks']}|{r['compliant_tasks']}|{r['numeric_A_pass']}|")
    lines += ['', 'Exact paths, weights and model hashes: tasks.json. Source and historical replay hashes: source_evidence.json.',
              'Current source hashes are recorded, not asserted equal to every historical snapshot. Further provenance/replay and final deliverables remain required.']
    (root / 'audit_report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
