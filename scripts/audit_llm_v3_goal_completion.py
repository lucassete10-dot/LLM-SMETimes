"""Bind the bounded v3 execution requirements to actual final evidence."""
import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

from battery_smetimes.llm_v3_workflow import SEEDS, OVERRIDES, screen_pass, summarize_validation, work_goal


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def goal_text(p):
    lines = Path(p).read_text(encoding='utf-8-sig').splitlines()
    # The desktop objective omits the Markdown heading marker and uses LF.
    # Preserve every substantive line, including all numbers and requirements.
    lines[0] = lines[0].removeprefix('# ')
    return lines


def require(value, message):
    if not value:
        raise ValueError(message)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--goal', type=Path, required=True)
    args = ap.parse_args()
    root = args.root.resolve()
    goal_hash = sha(args.goal)
    require(goal_text(args.goal) == goal_text('docs/LLM_SMETimes_round_robin_curve_goal_v3_20261013.md'), 'Goal content differs')
    workflow = read(root / 'workflow.json')
    sequence = ['MIT', 'XJTU', 'HUST', 'UConn_ISU_ILCC', 'Zhu2022_NCA']
    require(workflow['sequence'] == sequence, 'Dataset sequence differs')
    registration = read(root / 'workflow_registration.json')
    require(registration['maximum_fits'] == 91 and registration['formal_slots'] == 51, 'Budget differs')
    for file, digest in registration['sources_sha256'].items():
        require(sha(file) == digest, 'Workflow source drift')
    tests = read(root / 'workflow_behavior_tests.json')
    require(tests['exit_code'] == 0 and 'Ran 27 tests' in tests['stderr'] and 'OK' in tests['stderr'], 'Tests incomplete')
    for name, digest in tests['test_source_sha256'].items():
        require(sha(Path('tests') / (name + '.py')) == digest, 'Test source drift')
    for file, digest in tests['implementation_sha256'].items():
        require(sha(file) == digest, 'Tested implementation drift')
    audit = read(root / 'completed_v3_evidence_audit.json')
    require(audit['passed'] and not audit['failures'] and audit['audited_runs'] == 66, 'Model audit incomplete')
    require(sha('scripts/audit_llm_v3_evidence.py') == audit['source_sha256'], 'Model auditor changed')
    audit_by_run = {str(Path(r['run']).resolve()): r for r in audit['records']}
    queue = read(root / 'R0_queue.json')
    require(len(queue['tasks']) == 5 and all(t['state'] == 'complete' for t in queue['tasks']), 'Quick R0 missing')
    input_dir = root / 'input_contract_audit'
    binding = read(input_dir / 'audit_binding.json')
    require(binding['goal_sha256'] == goal_hash, 'Input audit bound to another goal')
    for filename, key in [('queue.json', 'queue_sha256'), ('all_dataset_input_isolation_audit.json', 'audit_sha256'), ('dataset_contracts.json', 'dataset_contracts_sha256')]:
        require(sha(input_dir / filename) == binding[key], 'Input audit evidence changed')
    input_audit = read(input_dir / 'all_dataset_input_isolation_audit.json')
    require(input_audit['passed'] and not input_audit['failures'] and len(input_audit['folds']) == 17
            and input_audit['distinct_source_cells'] == 385
            and input_audit['checked_fold_cell_instances'] == input_audit['requested_fold_cell_instances'] == 1361,
            'Input audit coverage incomplete')
    migration = read('experiments/llm_curve_v3/migration_latest.json')
    snapshot = Path(migration['folder'])
    require(all((snapshot / n).exists() for n in ['queue.json', 'active_process.json', 'status.json',
            'llm_curve_full_v1.json', 'llm_curve_optimization_plan_20261012.json',
            'LLM_SMETimes_round_robin_curve_goal_v3_20261013.md']), 'Migration snapshot missing')
    require(goal_text(snapshot / 'LLM_SMETimes_round_robin_curve_goal_v3_20261013.md') == goal_text(args.goal), 'Migration goal differs')
    mapping = read(root / 'legacy_to_v3_formal_mapping.json')
    require(len(mapping['slots']) == 51 and mapping['all_legacy_models_preserved'], 'Legacy mapping missing')
    legacy = read('experiments/llm_curve_full/20261011_MIT_then_other_datasets/queue.json')
    require(sum(t['state'] == 'superseded_v3' for t in legacy['tasks']) == 34, 'Legacy pending tasks not superseded')
    finished = read(root / 'legacy_finished.json')
    require(finished['passed'] and sha(Path(finished['run']) / 'model.pt') == finished['model_sha256'], 'Legacy active run not preserved')
    expected_source = dict(zip(sequence, [134, 55, 77, 64, 55]))
    expected_test = dict(zip(sequence, [134, 55, 16, 11, 55]))
    datasets, all_runs, formal_runs, files = {}, set(), set(), {}
    for dataset in sequence:
        ds = workflow['datasets'][dataset]
        require(ds['phase'] == 'complete' and ds['training_and_evaluation_complete'], 'Formal dataset incomplete')
        folds = ds['folds']
        require(len(folds) == (1 if dataset in ('HUST', 'UConn_ISU_ILCC') else 5), 'Frozen fold count differs')
        require(ds['screen_order'] == ['A', 'B'] and not ds['conditional_candidates_enabled'], 'Unexpected conditional candidates')
        noise = ds['conditional_noise_evidence']
        require(not noise['test_used'] and not (noise['training_validation_mae_ratio'] >= 2
                and noise['noisy_training_cell_fraction'] >= .2), 'Conditional noise decision invalid')
        base_key = f'R0:{folds[0]}:{SEEDS[0]}:120'
        base = summarize_validation(ds['trials'][base_key]['evidence'])
        for candidate in ('A', 'B'):
            trial = ds['trials'][f'{candidate}:{folds[0]}:{SEEDS[0]}:120']
            require(not screen_pass(summarize_validation(trial['evidence']), base), 'Rejected candidate actually eligible')
        require(ds['screen_winner'] is None and ds['winner'] == 'R0', 'Winner decision differs')
        lockpath = root / 'selection_locks' / f'{dataset}.json'
        lock = read(lockpath)
        chosen = [ds['trials'][f'R0:{folds[0]}:{seed}:240']['evidence'] for seed in SEEDS]
        require(lock == ds['selection_lock'] and lock['validation_evidence'] == chosen
                and lock['seeds'] == list(SEEDS) and not lock['test_used_for_selection'], 'Selection lock inconsistent')
        require(lock['curve_work_goal_reached'] == work_goal(chosen), 'Work goal flag differs')
        for key, trial in ds['trials'].items():
            run = Path(trial['run']).resolve()
            require(trial['state'] == 'complete' and str(run) in audit_by_run, 'Unverified actual trial')
            all_runs.add(str(run))
            record = audit_by_run[str(run)]
            require(sha(run / 'model.pt') == record['model_sha256'] == trial['evidence']['model_sha256'], 'Model hash changed')
            require(sha(run / 'validation_audited_predictions.npz') == trial['evidence']['predictions_sha256'], 'Validation evidence changed')
            cfg = read(run / 'config.json')
            budget = 120 if key.endswith(':120') else 240
            candidate, fold, seed, _ = key.split(':')
            expected_cfg = {**ds['R0_settings'], **OVERRIDES[candidate], 'fold': fold, 'seed': int(seed),
                            'max_tiny_epochs': budget, 'validation_min_epochs': 40 if budget == 120 else 60,
                            'validation_patience': 4 if budget == 120 else 6}
            # The trainer resolves full TRAIN label coverage from each fold;
            # the settings' sampling placeholder is not the resolved count.
            if expected_cfg.get('full_train_targets'):
                with np.load(run / 'training_audited_predictions.npz', allow_pickle=False) as arrays:
                    expected_cfg['future_targets'] = max(len(arrays[f'{cid}__100__cycle']) for cid in cfg['train_ids'])
            require(all(cfg.get(k) == v for k, v in expected_cfg.items()), 'Single-factor settings differ')
            require(cfg['max_tiny_epochs'] == budget and cfg['validation_every'] == 10
                    and cfg['validation_min_epochs'] == (40 if budget == 120 else 60)
                    and cfg['validation_patience'] == (4 if budget == 120 else 6)
                    and cfg['validation_min_relative_improvement'] == .005
                    and cfg['origin'] == 100 and cfg['selection_version'] == 'curve_quality_v3', 'Actual training contract differs')
            require(cfg['architecture'] == 'query_rate_local' and cfg['mode'] == 'development', 'Model family changed')
            history = read(run / 'training_history.json')
            require(1 <= history[-1]['epoch'] <= budget and all(r['epoch'] % 10 == 0
                    for r in history if 'validation_quality_score' in r), 'Epoch/evaluation schedule differs')
            anchor, stale, first_stop = None, 0, None
            for row in history:
                if 'validation_quality_score' in row:
                    score = row['validation_quality_score']
                    if anchor is None or score < anchor * .995:
                        anchor, stale = score, 0
                    else:
                        stale += 1
                if row['epoch'] >= cfg['validation_min_epochs'] and stale >= cfg['validation_patience']:
                    first_stop = row['epoch']
                    break
            require(history[-1]['epoch'] == (first_stop if first_stop is not None else budget), 'Actual early-stop schedule differs')
            require(record['v3_selection']['earliest_minimum_score_verified']
                    and record['v3_selection']['frozen_validation_policy_verified'], 'Selection evidence incomplete')
            proof = read(run / 'repair_verification.json')
            require(proof['passed'] and all(r['passed'] for r in proof['per_cell']), 'Replay isolation incomplete')
            for probe in proof['per_cell']:
                require(all(probe[k] <= proof['tolerance_soh'] for k in
                    ['singleton_query_error_soh', 'reordered_query_error_soh', 'future_and_label_poison_error_soh']), 'Replay tolerance exceeded')
                # First TEST export has no pre-reload prediction cache.
                # TRAIN/VAL compare the freshly reloaded model against the
                # original selected model predictions in every actual run.
                if probe['role'] in ('training', 'validation'):
                    require(probe['reload_max_error_soh'] <= proof['tolerance_soh'], 'Reload tolerance exceeded')
            for file, digest in read(run / 'source_manifest.json')['sha256'].items():
                require(sha(run / 'source_snapshot' / Path(file).resolve().relative_to(Path.cwd().resolve())) == digest, 'Training source snapshot changed')
        expected_keys = {f'R0:{fold}:{seed}:240' for fold in folds for seed in SEEDS}
        require(set(ds['formal_slots']) == expected_keys, 'Missing formal slots')
        for key, slot in ds['formal_slots'].items():
            require(slot['test_verified'] and slot['run'] == ds['trials'][key]['run']
                    and slot['selection_lock_sha256'] == sha(lockpath), 'Formal reuse/lock differs')
            formal_runs.add(str(Path(slot['run']).resolve()))
        # Recompute every final report row from original NPZ, validate review
        # bytes, PDF QA, source membership, role coverage and failure lists.
        subprocess.run([sys.executable, '-m', 'scripts.finalize_llm_v3_dataset_delivery',
                        '--root', str(root), '--dataset', dataset], check=True, capture_output=True, text=True)
        cohort = 'completed_003' if len(folds) == 1 else 'completed_015'
        dest = root / 'reports' / dataset / cohort
        manifest = read(dest / 'delivery_manifest.json')
        roles = manifest['delivery_checks']['roles']
        source_ids, test_ids = set(), set()
        for role in ('training', 'validation', 'test'):
            folder = dest / role
            summary = read(folder / 'summary.json')
            rows = read(folder / 'per_cell_seed_metrics.json')[role]
            source_ids.update(row['cell_id'] for row in rows)
            if role == 'test':
                test_ids.update(row['cell_id'] for row in rows)
                groups = {}
                for row in rows:
                    groups.setdefault(row['cell_id'], []).append(row['seed'])
                require(all(sorted(s) == list(SEEDS) for s in groups.values()), 'Physical TEST seed coverage differs')
            for file, digest in summary['source_predictions_sha256'].items():
                require(sha(file) == digest, 'Original NPZ changed')
                files[str(Path(file).resolve())] = digest
            review = read(folder / 'visual_review.json')
            require(not review.get('pending_figures') and len(review['figures']) == roles[role]['reviewed_figures'], 'Visual review incomplete')
            for row in review['figures']:
                require(sha(row['png']) == row['sha256'], 'Reviewed image changed')
                if row.get('inherited_from_review'):
                    src = read(row['inherited_from_review'])
                    require(sha(row['inherited_from_review']) == row['source_review_sha256'], 'Inherited review changed')
                    matches = [f for f in src['figures'] if all(f[k] == row[k] for k in ('cell_id', 'fold', 'role', 'seeds', 'sha256'))]
                    require(len(matches) == 1, 'Review inheritance identity differs')
            for filename in ['all_cells.pdf', 'per_cell_seed_metrics.json', 'summary.json',
                             'visual_review.json', 'reference_failures.csv', 'figure_manifest.json', 'plot_source.py']:
                files[str((folder / filename).resolve())] = sha(folder / filename)
        require(len(source_ids) == expected_source[dataset] and len(test_ids) == expected_test[dataset], 'Physical cell coverage incomplete')
        datasets[dataset] = {'formal_slots': len(expected_keys), 'source_cells': len(source_ids), 'test_cells': len(test_ids),
                            'roles': roles, 'accuracy_work_goal_reached': lock['curve_work_goal_reached'],
                            'conditional_candidates': 'not_triggered', 'winner': 'R0'}
        files[str(lockpath)] = sha(lockpath)
        files[str(dest / 'delivery_manifest.json')] = sha(dest / 'delivery_manifest.json')
    require(len(all_runs) == 66 and all_runs == set(audit_by_run) and len(formal_runs) == 51, 'Fit budget/full audit membership differs')
    gap = read(root / 'R0_train_validation_gap.json')
    comparisons = read(root / 'validation_comparisons.json')
    require(len(gap['records']) == 5 and not gap['test_consumed'] and len(comparisons['screen_comparisons']) == 10
            and not comparisons['test_metrics_consumed'], 'Required comparison delivery missing')
    for file in ['workflow.json', 'workflow_registration.json', 'workflow_behavior_tests.json',
                 'completed_v3_evidence_audit.json', 'legacy_to_v3_formal_mapping.json',
                 'R0_train_validation_gap.json', 'validation_comparisons.json', 'formal_progress.json']:
        files[str(root / file)] = sha(root / file)
    report = {'utc': datetime.now(timezone.utc).isoformat(), 'goal_sha256': goal_hash,
              'markdown_goal_sha256': sha('docs/LLM_SMETimes_round_robin_curve_goal_v3_20261013.md'),
              'goal_content_equivalence': 'Only first-line Markdown heading marker and newline encoding differ',
              'execution_requirements_verified': True, 'accuracy_work_goal_reached': all(d['accuracy_work_goal_reached'] for d in datasets.values()),
              'actual_unique_fits': len(all_runs), 'formal_slots': len(formal_runs), 'fit_budget_cap': 91,
              'source_cells': sum(d['source_cells'] for d in datasets.values()),
              'reviewed_final_figures': sum(v['reviewed_figures'] for d in datasets.values() for v in d['roles'].values()),
              'datasets': datasets, 'evidence_sha256': files,
              'scope': 'Bounded v3 execution and delivery completed; universal reference accuracy not achieved',
              'optional_300_epoch_extension': 'Not used; optional diagnostic, no unlocked comparison substituted',
              'source_sha256': sha(__file__)}
    (root / 'goal_completion_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ['execution_requirements_verified', 'accuracy_work_goal_reached', 'actual_unique_fits', 'formal_slots', 'source_cells', 'reviewed_final_figures']}))


if __name__ == '__main__':
    main()
