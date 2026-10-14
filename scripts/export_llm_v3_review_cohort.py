"""Pre-render fully verified three-seed folds for later hash-bound visual QA."""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from scripts.audit_llm_curve_repair import fingerprint
from battery_smetimes.llm_v3_workflow import SEEDS, trial_key


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--dataset', required=True)
    args = ap.parse_args()
    root = args.root
    state = read(root / 'workflow.json')['datasets'][args.dataset]
    lock = root / 'selection_locks' / f'{args.dataset}.json'
    if read(lock) != state['selection_lock']:
        raise ValueError('Validation selection lock differs')
    ready = [fold for fold in state['folds'] if all(
        state['formal_slots'].get(trial_key(state['winner'], fold, seed, 240), {}).get('test_verified')
        for seed in SEEDS)]
    if not ready:
        raise ValueError('No complete three-seed formal fold to review')
    out = root / 'qa_cohorts' / args.dataset / f'complete_folds_{len(ready):02d}'
    if (out / 'review_cohort_manifest.json').exists() or any(out.glob('*/figure_manifest.json')):
        raise ValueError('Existing review cohort: preserve its artifacts and review evidence')
    tasks = []
    for fold in state['folds']:
        for seed in SEEDS:
            slot = state['formal_slots'].get(trial_key(state['winner'], fold, seed, 240), {})
            task = {'dataset': args.dataset, 'fold': fold, 'seed': seed,
                    'role_counts': state['templates'][fold]['role_counts'],
                    'state': 'outside_three_seed_review_cohort',
                    'actual_slot_test_verified': bool(slot.get('test_verified'))}
            if fold in ready:
                run = Path(slot['run'])
                proof = read(run / 'repair_verification.json')
                if (not proof['passed'] or set(proof['role_scope']) != {'training', 'validation', 'test'}
                        or slot['selection_lock_sha256'] != fingerprint(lock)
                        or proof['model_sha256'] != fingerprint(run / 'model.pt')):
                    raise ValueError('Formal prediction evidence invalid')
                task.update(state='complete', run=str(run.resolve()), model_sha256=proof['model_sha256'])
            tasks.append(task)
    bridge = out / 'source_bridge'
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(__file__, out / 'export_source.py')
    save(bridge / 'queue.json', {'tasks': tasks, 'origin': 100, 'seeds': list(SEEDS),
        'sequence': [args.dataset], 'scope': 'Visual pre-review cohort, not current full training progress',
        'full_formal_fold_list_preserved': True, 'goal_complete': False})
    manifest = {'dataset': args.dataset, 'review_folds': ready, 'planned_folds': state['folds'],
                'planned_formal_slots': len(tasks), 'review_formal_slots': len(ready) * len(SEEDS),
                'selection_lock_sha256': fingerprint(lock), 'visual_review': 'pending',
                'full_coverage_not_claimed': True, 'source_sha256': fingerprint(Path(__file__)),
                'source_snapshot': str((out / 'export_source.py').resolve()),
                'final_review_reuse_requires_identical_png_hashes': True, 'goal_complete': False}
    save(out / 'review_cohort_manifest.json', manifest)
    for role in ('training', 'validation', 'test'):
        p = out / role
        if (p / 'figure_manifest.json').exists():
            raise ValueError('Existing cohort output: do not overwrite reviewed artifacts')
        subprocess.run([sys.executable, '-m', 'scripts.report_llm_full_dataset', '--queue', str(bridge),
                        '--dataset', args.dataset, '--progress-only', '--quality-v3',
                        '--figure-roles', role, '--output-dir', str(p)], check=True)
    save(out / 'review_cohort_manifest.json', {**manifest, 'rendering_complete': True})
    print(json.dumps({'dataset': args.dataset, 'out': str(out.resolve()), 'folds': ready}))


if __name__ == '__main__':
    main()
