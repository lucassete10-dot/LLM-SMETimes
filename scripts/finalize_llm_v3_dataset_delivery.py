"""Finalize a fully covered, visually reviewed dataset; never certify goal accuracy."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from battery_smetimes.llm_curve_quality_v3 import QUALITY, curve_metrics_v3, reference_pass
from battery_smetimes.llm_reporting import finite_metric
from battery_smetimes.llm_v3_workflow import SEEDS


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--dataset', required=True)
    args = ap.parse_args()
    root = args.root
    state = read(root / 'workflow.json')['datasets'][args.dataset]
    if state['phase'] != 'complete':
        raise ValueError('Formal dataset coverage is incomplete')
    lock = root / 'selection_locks' / f'{args.dataset}.json'
    if read(lock) != state['selection_lock']:
        raise ValueError('Selection lock differs')
    expected = len(state['folds']) * len(SEEDS)
    slots = state['formal_slots']
    if len(slots) != expected or not all(s.get('test_verified') for s in slots.values()):
        raise ValueError('Not all formal slots have verified TEST evidence')
    runs = {}
    for slot in slots.values():
        cfg = read(Path(slot['run']) / 'config.json')
        identity = (cfg['fold'], cfg['seed'])
        if identity in runs or cfg['fold'] not in state['folds'] or cfg['seed'] not in SEEDS:
            raise ValueError('Duplicate or unexpected formal run identity')
        runs[identity] = (Path(slot['run']), cfg['origin'])
        proof = read(Path(slot['run']) / 'repair_verification.json')
        if (not proof['passed'] or slot['selection_lock_sha256'] != sha(lock)
                or set(proof['role_scope']) != {'training', 'validation', 'test'}
                or proof['model_sha256'] != sha(Path(slot['run']) / 'model.pt')):
            raise ValueError('Formal model/selection evidence changed')
    out = root / 'reports' / args.dataset / f'completed_{expected:03d}'
    manifest = read(out / 'delivery_manifest.json')
    roles = {}
    # Validate all artifacts before writing any status or failure list.
    checked = []
    for role in ('training', 'validation', 'test'):
        p = out / role
        summary = read(p / 'summary.json')
        figures = read(p / 'figure_manifest.json')
        review = read(p / 'visual_review.json')
        collision = read(p / 'collision_audit.json')
        rows = read(p / 'per_cell_seed_metrics.json')[role]
        quality = summary['v3_reference_quality'][role]
        if (not summary['dataset_complete'] or summary['runs'] != expected
                or summary['planned_runs'] != expected
                or quality['evaluated_cell_seed_instances'] != len(rows)
                or quality['missing_cell_seed_instances'] != 0):
            raise ValueError('Full numeric role coverage is unproven: ' + role)
        if len({(r['cell_id'], r['fold'], r['seed']) for r in rows}) != len(rows):
            raise ValueError('Duplicate cell/fold/seed rows')
        actual = {}
        for (fold, seed), (run, origin) in runs.items():
            suffix = f'__{origin}__cycle'
            with np.load(run / f'{role}_audited_predictions.npz', allow_pickle=False) as archive:
                for key in archive.files:
                    if key.endswith(suffix):
                        cid = key[:-len(suffix)]
                        prefix = f'{cid}__{origin}__'
                        actual[(cid, fold, seed)] = curve_metrics_v3(
                            *[archive[prefix + k] for k in ('cycle', 'truth', 'prediction')])
        if set(actual) != {(r['cell_id'], r['fold'], r['seed']) for r in rows}:
            raise ValueError('Report membership differs from original role archives')
        for row in rows:
            metrics = actual[(row['cell_id'], row['fold'], row['seed'])]
            for key in (*QUALITY, 'mae_pp', 'tail_mae_pp'):
                value = metrics[key]
                if ((value is None and row.get(key) is not None)
                        or (value is not None and (not finite_metric(row, key)
                            or not np.isclose(row[key], value, rtol=0, atol=1e-9)))):
                    raise ValueError('Report metric differs from raw prediction: ' + key)
            if row['reference_numeric_pass'] != reference_pass(metrics):
                raise ValueError('Reference pass flag differs from raw prediction')
        for filename, digest in summary['source_predictions_sha256'].items():
            if sha(Path(filename)) != digest:
                raise ValueError('Source predictions changed: ' + filename)
        reviewed = {r['cell_id']: r for r in review['figures']}
        if (len(reviewed) != len(figures['figures']) or collision['verdict'] != 'PASS'
                or collision['page_count'] != len(figures['figures'])
                or not figures['qa'] or any(q['exit_code'] for q in figures['qa'])):
            raise ValueError('Figure review or automatic PDF QA incomplete')
        for figure in figures['figures']:
            png = Path(figure['png'])
            reviewed_row = reviewed[figure['cell_id']]
            if (set(figure['seeds']) != set(SEEDS) or figure['role'] != role
                    or Path(reviewed_row['png']).resolve() != png.resolve()
                    or reviewed_row['sha256'] != sha(png)
                    or not png.with_suffix('.svg').exists()
                    or not png.with_suffix('.alignment.json').exists()):
                raise ValueError('Reviewed figure changed or missing exports')
        failures = []
        for row in rows:
            failed = [k for k, limit in QUALITY.items()
                      if not finite_metric(row, k) or row[k] > limit]
            if not reference_pass(row):
                failures.append({
                    'dataset': args.dataset, 'role': role, 'cell_id': row['cell_id'],
                    'fold': row['fold'], 'seed': row['seed'],
                    'primary_mape_pass': finite_metric(row, 'mape_percent') and row['mape_percent'] <= 3,
                    **{k: row.get(k) if finite_metric(row, k) else 'unavailable' for k in QUALITY},
                    'reference_failed_metrics': ';'.join(failed)})
        roles[role] = {
            'evaluated_seed_instances': len(rows), 'reference_failed_instances': len(failures),
            'stable_primary_groups': quality['stable_primary_cell_fold_groups'],
            'stable_reference_groups': quality['stable_reference_cell_fold_groups'],
            'reviewed_figures': len(reviewed), 'pdf_sha256': sha(p / 'all_cells.pdf'),
            'visual_review_sha256': sha(p / 'visual_review.json')}
        checked.append((p, figures, review, failures))
    for p, figures, review, failures in checked:
        figures['visual_review'] = {
            'state': 'completed', 'scope': review['scope'],
            'review_file': str((p / 'visual_review.json').resolve()),
            'review_sha256': sha(p / 'visual_review.json'),
            'numeric_accuracy_pass_not_implied': True}
        save(p / 'figure_manifest.json', figures)
        fields = ['dataset', 'role', 'cell_id', 'fold', 'seed', 'primary_mape_pass',
                  *QUALITY, 'reference_failed_metrics']
        with (p / 'reference_failures.csv').open('w', newline='', encoding='utf-8-sig') as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(failures)
        roles[p.name]['reference_failure_csv_sha256'] = sha(p / 'reference_failures.csv')
    manifest.update(visual_review='completed_all_roles',
                    delivery_checks={'roles': roles, 'source_sha256': sha(Path(__file__)),
                                     'selection_lock_sha256': sha(lock),
                                     'execution_scope': 'This dataset only; no whole-goal completion claim'},
                    goal_complete=False)
    save(out / 'delivery_manifest.json', manifest)
    print(json.dumps({'dataset': args.dataset, 'formal_slots': expected, 'roles': roles}))


if __name__ == '__main__':
    main()
