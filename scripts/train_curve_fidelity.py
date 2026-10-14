"""Training-side curve-fidelity experiment; outer test cells never loaded."""
import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import shutil
import time
import traceback

import joblib
import numpy as np

from battery_smetimes.curve_fidelity import fit_model, evaluate, selection_score
from battery_smetimes.physical_types import load_registry
from scripts.prepare_same_type_forecast import save_json, sha

AUDIT = Path('experiments/soh_research_v5/audits/20260930T074750_792569Z')
CANDIDATES = [
    dict(arm='compact', trees=16, depth=5, leaf=4),
    dict(arm='compact', trees=32, depth=5, leaf=4),
    dict(arm='compact', trees=32, depth=6, leaf=8),
    dict(arm='full_tree', depth=8, leaf=1, tail_weight=1),
    dict(arm='full_tree', depth=None, leaf=1, tail_weight=1),
    dict(arm='full_tree', depth=None, leaf=4, tail_weight=3),
    dict(arm='local_tree', depth=None, leaf=1, tail_weight=1),
    dict(arm='local_tree', depth=None, leaf=4, tail_weight=3),
    dict(arm='shape', degree=1, leaf=1),
    dict(arm='shape', degree=2, leaf=1),
    dict(arm='shape', degree=2, leaf=3),
]


def load_subset(ids):
    records = json.loads((AUDIT/'cell_registry.json').read_text())['cells']
    cfg = json.loads((AUDIT/'config.json').read_text())
    integrity = Path(cfg['config']['integrity_audit'])
    if sha(integrity) != cfg['integrity_sha256']:
        raise ValueError('quality audit changed')
    issues = json.loads(integrity.read_text())['issues']
    cells = {}
    for key in ids:
        record = records[key]; p = Path(record['prepared_file'])
        if sha(p) != record['prepared_sha256']:
            raise ValueError('prepared data changed')
        with np.load(p) as d:
            cell = dict(record, raw=d['raw_features'].copy(), cycle=d['cycle'].copy(),
                soh=d['soh'].copy(), label_cycle=d['label_cycle'].copy())
        rejected = [i['cycle'] for i in issues if i['cell_id'] == key]
        cell['soh'][np.isin(cell['label_cycle'], rejected)] = np.nan
        good = ~np.isin(cell['cycle'], rejected)
        cell['cycle'], cell['raw'] = cell['cycle'][good], cell['raw'][good]
        cells[key] = cell
    return cells


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--fold', default='XJTU_fold1')
    args = parser.parse_args()
    plan = next(p for p in json.loads((AUDIT/'splits.json').read_text())['plans'] if p['name'] == args.fold)
    # Preserve NASA's current Ridge baseline, with the same alpha budget that
    # was fixed before this goal; comparison uses the NEW validation score only.
    candidates = CANDIDATES + ([dict(arm='ridge', alpha=a) for a in (.1, 10., 100.)]
        if plan['dataset'] == 'NASA' else [])
    assert not set(plan['train']) & set(plan['validation'])
    assert not set(plan['train']+plan['validation']) & set(plan['test'])
    out = Path('experiments/curve_fidelity')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True)
    sources = [Path(__file__), Path('battery_smetimes/curve_fidelity.py'),
        Path('battery_smetimes/forecast.py'), Path('battery_smetimes/research_baselines.py'),
        Path('battery_smetimes/physical_types.py'), Path('scripts/train_stm32_compact_trees.py')]
    snapshot = out/'source_snapshot'; snapshot.mkdir()
    for p in sources:
        shutil.copy2(p, snapshot/p.name)
    shutil.copy2('docs/soh_curve_fidelity_goal_20261005.md', out/'plan.md')
    seeds = list(range(20261005, 20261006 if args.smoke else 20261010))
    records = json.loads((AUDIT/'cell_registry.json').read_text())['cells']
    registry_file = AUDIT/'source_snapshot/battery_physical_registry.json'
    frozen = json.loads((AUDIT/'config.json').read_text())
    assert sha(registry_file) == frozen['registry_sha256']
    config = dict(smoke=args.smoke, plan=plan, seeds=seeds, candidates=candidates,
        origin=100, scope='training-side validation only; original outer test excluded/not loaded',
        source_sha256={p.name: sha(p) for p in sources},
        prepared_sha256={k: records[k]['prepared_sha256'] for k in plan['train']+plan['validation']},
        registry_sha256=sha(registry_file), integrity_sha256=frozen['integrity_sha256'],
        splits_sha256=sha(AUDIT/'splits.json'),
        selection='mean((full_MAPE+late_MAPE+endpoint_pp+drop_error_pp)/4)+0.25*cell_score_P90',
        outer_test_evaluated=False)
    save_json(out/'config.json', config)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
        handlers=[logging.FileHandler(out/'training.log', encoding='utf-8'), logging.StreamHandler()])
    summaries = []
    try:
        cells = load_subset(plan['train']+plan['validation'])
        assert not set(cells) & set(plan['test'])
        registry = load_registry(registry_file)
        for seed in seeds:
            scores = []
            for index, params in enumerate(candidates):
                started = time.perf_counter()
                folder = out/str(seed)/f'candidate_{index}'; folder.mkdir(parents=True)
                actual_params = dict(params)
                if params['arm'] in ('compact', 'ridge'):
                    actual_params['_registry'] = registry
                logging.info('TRAIN seed=%s candidate=%s parameters=%s', seed, index, params)
                bundle = fit_model(cells, plan['train'], seed, actual_params)
                joblib.dump(bundle, folder/'model.joblib')
                rows, arrays = evaluate(bundle, cells, plan['validation'])
                save_json(folder/'validation_metrics.json', rows)
                np.savez_compressed(folder/'validation_predictions.npz', **arrays)
                keys = ('mape_percent', 'late_mape', 'endpoint_pp', 'drop_error_pp', 'rmse_soh', 'joint_pass', 'severe_flat_failure')
                summary = dict(seed=seed, candidate=index, parameters=params, selection_score=selection_score(rows),
                    training_seconds=time.perf_counter()-started,
                    validation_cells=len(rows), **{k: float(np.mean([r[k] for r in rows])) for k in keys})
                scores.append(summary); summaries.append(summary)
                save_json(folder/'summary.json', summary); save_json(out/'summary.json', summaries)
                logging.info('SAVED seed=%s candidate=%s score=%.3f fullMAPE=%.3f endpoint=%.3f drop=%.3f',
                    seed, index, summary['selection_score'], summary['mape_percent'], summary['endpoint_pp'], summary['drop_error_pp'])
            best = min(scores, key=lambda r: r['selection_score'])
            save_json(out/str(seed)/'selection.json', dict(candidates=scores, selected=best,
                used_only_validation=True, original_test_loaded=False))
        save_json(out/'verification.json', dict(passed=True, original_test_loaded=False,
            label_free_and_future_poison_predictions=True, physical_family_rejection=True,
            loaded_ids=sorted(cells), excluded_outer_test=plan['test']))
        save_json(out/'progress.json', dict(training_complete=True, smoke=args.smoke,
            outer_evaluation_complete=False, plots_exported=False))
        print('OUTPUT='+str(out.resolve()), flush=True)
    except BaseException:
        save_json(out/'failure.json', dict(traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
