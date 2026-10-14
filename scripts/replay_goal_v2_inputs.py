"""Replay every frozen outer prediction with label/future-input poisoning.

No training, selection, or modification of historical artifacts.
"""
import copy
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from scripts.audit_four_cohort_progress import SOURCES
from scripts.prepare_same_type_forecast import sha, save_json
from scripts.train_curve_fidelity import AUDIT, load_subset
from scripts.train_tail_boost import NCA
from scripts.train_crossfit_drift_smoke import clean
from battery_smetimes.mit_feature_blend import predict as mit_predict
from scripts.train_xjtu_segment_outer import predict as xjtu_predict
from scripts.train_nasa_slope_blend_formal import predict as nasa_predict
from battery_smetimes.tail_selection import predict as nca_predict


def resolve_source(key, expected):
    candidates = [Path(key)]
    if len(Path(key).parts) == 1:
        candidates += [Path(folder) / key for folder in ('battery_smetimes', 'scripts', 'docs')]
    matches = [p for p in candidates if p.is_file() and sha(p) == expected]
    if not matches:
        raise ValueError('current source no longer matches frozen digest: ' + key)
    return matches[0]


def poisoned(query, nca):
    value = copy.deepcopy(query)
    for key in ('soh', 'label_cycle', 'unfiltered_soh', 'reference_capacity_ah',
                'capacity_ah', 'cycle_life', 'actual_eol'):
        value[key] = np.inf
    cycle_key, raw_key = ('early_cycle', 'early_raw') if nca else ('cycle', 'raw')
    c, x = np.asarray(value[cycle_key]), np.asarray(value[raw_key]).copy()
    x[c > 100] = np.inf
    value[cycle_key] = np.r_[c, 100000]
    value[raw_key] = np.vstack([x, np.full((1, x.shape[1]), np.inf)])
    if nca:
        value['truth'] = np.inf
        value['cycle'] = np.inf
        value['prefix'] = np.full_like(value['prefix'], np.inf)
    return value


def main():
    root = Path('experiments/goal_v2_input_replay') / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    root.mkdir(parents=True)
    print('RUN=' + str(root.resolve()), flush=True)
    shutil.copy2(__file__, root / 'replay_source.py')
    rows, evidence, summaries = [], [], []
    start = datetime.now(timezone.utc)
    try:
        for name, path, expected_count, proof_name, _ in SOURCES:
            source = Path(path)
            cfg = json.loads((source / 'config.json').read_text())
            assert json.loads((source / proof_name).read_text())['passed']
            for key, digest in cfg['source_sha256'].items():
                file = resolve_source(key, digest)
                snapshot = source / 'source_snapshot' / file.name
                assert sha(snapshot) == digest
                evidence.append(dict(cohort=name, file=str(file.resolve()), sha256=digest,
                                     snapshot=str(snapshot.resolve())))
            if 'plans' in cfg:
                plans, seeds = cfg['plans'], cfg['seeds']
            else:
                parent = Path(cfg['training_source']) / 'config.json'
                assert sha(parent) == cfg['training_config_sha256']
                pc = json.loads(parent.read_text())
                plans = [p for p in pc['plans'] if p['dataset'] == cfg['dataset']]
                seeds = pc['seeds']
            if 'choice_lock_sha256' in cfg:
                lock = source / 'choice_lock.json'
                assert sha(lock) == cfg['choice_lock_sha256']
                records = json.loads(lock.read_text())
                assert records['outer_loaded'] is False
                inner = Path(cfg['inner'])
                assert sha(inner / 'config.json') == cfg['inner_config_sha256']
                assert sha(inner / 'independent_verification.json') == cfg['inner_proof_sha256']
                for record in records['records']:
                    child = inner / record['fold'] / str(record['seed'])
                    for file, field in [('selection.json', 'selection_sha256'),
                                        ('config.json', 'config_sha256'),
                                        ('independent_verification.json', 'proof_sha256')]:
                        assert sha(child / file) == record[field]
            predictions = {'MIT': mit_predict, 'XJTU': xjtu_predict,
                           'NASA': nasa_predict, 'Zhu2022 NCA': nca_predict}
            dataset_rows = []
            for plan in plans:
                if name == 'Zhu2022 NCA':
                    index_file = NCA / 'index.json'
                    index = json.loads(index_file.read_text())
                    cells = {}
                    for key in plan['test']:
                        entry = index['cells'][key]
                        file = NCA / entry['file']
                        assert sha(file) == entry['sha256']
                        cells[key] = joblib.load(file)
                        evidence.append(dict(file=str(file.resolve()), sha256=entry['sha256']))
                else:
                    cells = load_subset(plan['test'])
                    for key, cell in cells.items():
                        evidence.append(dict(file=cell['prepared_file'], sha256=cell['prepared_sha256']))
                for seed in seeds:
                    model_file = source / plan['name'] / str(seed) / 'model.joblib'
                    array_file = model_file.with_name('test_predictions.npz')
                    bundle = joblib.load(model_file)
                    assert set(bundle['train_ids']) == set(plan['train'] + plan['validation'])
                    with np.load(array_file) as arrays:
                        for key, cell in cells.items():
                            cycle = arrays[key + '__100__cycle']
                            stored = arrays[key + '__100__prediction']
                            query = clean(cell)
                            early = 'early_cycle' if name == 'Zhu2022 NCA' else 'cycle'
                            assert np.any(np.asarray(query[early]) <= 100)
                            prediction = predictions[name](bundle, query, cycle)
                            np.testing.assert_allclose(prediction, stored, rtol=0, atol=1e-12)
                            altered = predictions[name](bundle, poisoned(query, name == 'Zhu2022 NCA'), cycle)
                            np.testing.assert_allclose(altered, prediction, rtol=0, atol=1e-12)
                            bad = dict(query, physical_family='CONTRACT_WRONG_FAMILY')
                            try:
                                predictions[name](bundle, bad, cycle)
                            except ValueError:
                                pass
                            else:
                                raise AssertionError('different physical-family input accepted')
                            row = dict(dataset=name, fold=plan['name'], seed=seed, cell_id=key,
                                       points=len(cycle), replay_max_abs=float(np.max(np.abs(prediction-stored))),
                                       poison_max_abs=float(np.max(np.abs(altered-prediction))),
                                       wrong_family_rejected=True)
                            rows.append(row)
                            dataset_rows.append(row)
                    evidence.extend(dict(file=str(f.resolve()), sha256=sha(f)) for f in (model_file, array_file))
                    save_json(root / 'progress.json', dict(last_fold=plan['name'], last_seed=seed,
                              replayed_cell_seed_rows=len(rows), terminal=False))
                print(name + ' ' + plan['name'] + ' replayed', flush=True)
            assert len(dataset_rows) == expected_count * 5
            assert len({r['cell_id'] for r in dataset_rows}) == expected_count
            summaries.append(dict(dataset=name, cells=expected_count, rows=len(dataset_rows), passed=True))
        assert len(rows) == 1240
        save_json(root / 'rows.json', rows)
        save_json(root / 'source_evidence.json', evidence)
        save_json(root / 'report.json', dict(passed=True, cohorts=summaries, rows=len(rows),
                  current_sources_equal_historical_snapshots=True,
                  tests='exact saved forecast replay; capacity/labels/EOL poisoning; future-signal poisoning; family rejection',
                  limit='does not reconstruct every raw feature from original files or re-fit every estimator',
                  started_utc=start.isoformat(), finished_utc=datetime.now(timezone.utc).isoformat(),
                  source_sha256=sha(__file__), goal_complete=False))
        save_json(root / 'progress.json', dict(terminal=True, passed=True, replayed_cell_seed_rows=len(rows)))
    except BaseException:
        import traceback
        save_json(root / 'failure.json', dict(traceback=traceback.format_exc(), rows_completed=len(rows)))
        save_json(root / 'progress.json', dict(terminal=True, passed=False, replayed_cell_seed_rows=len(rows)))
        raise


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
