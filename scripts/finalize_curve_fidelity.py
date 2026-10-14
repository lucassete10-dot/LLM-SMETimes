"""Refit frozen validation selections, then evaluate entire outer folds.

Never selects candidate from test results. Each source run must be complete.
"""
import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import shutil
import traceback
import joblib
import numpy as np
from battery_smetimes.curve_fidelity import fit_model, evaluate, curve_metrics
from battery_smetimes.physical_types import load_registry
from scripts.train_curve_fidelity import AUDIT, load_subset
from scripts.prepare_same_type_forecast import save_json, sha

OLD = Path('experiments/stm32_compact_trees/runs/20261004T074429_087457Z')
NASA_OLD = Path('experiments/nasa_signal_trajectory/ridge/20261004T165029_391663Z')


def summarize(rows):
    grouped = {}
    for r in rows:
        grouped.setdefault(r['cell_id'], []).append(r)
    numeric = ('mape_percent', 'late_mape', 'last100_mape', 'endpoint_pp', 'drop_error_pp',
        'rmse_soh', 'mae_soh', 'truth_drop_pp', 'prediction_drop_pp')
    cells = []
    for key, values in grouped.items():
        assert len(values) == 5 and len({r['seed'] for r in values}) == 5
        means = {field: float(np.mean([r[field] for r in values])) for field in numeric}
        joint = (means['mape_percent'] <= 6 and means['late_mape'] <= 8 and
            means['endpoint_pp'] <= 8 and means['drop_error_pp'] <= 5)
        severe = means['truth_drop_pp'] >= 5 and means['prediction_drop_pp'] < .3*means['truth_drop_pp']
        cells.append(dict(cell_id=key, **means, joint_pass=bool(joint), severe_flat_failure=bool(severe)))
    seeds = sorted({r['seed'] for r in rows})
    summary = {field: dict(mean=float(np.mean([c[field] for c in cells])),
        seed_sd=float(np.std([np.mean([r[field] for r in rows if r['seed'] == seed]) for seed in seeds], ddof=1)))
        for field in numeric}
    summary.update(cells=len(cells), joint_pass_fraction=float(np.mean([c['joint_pass'] for c in cells])),
        severe_flat_fraction=float(np.mean([c['severe_flat_failure'] for c in cells])),
        endpoint_p90=float(np.quantile([c['endpoint_pp'] for c in cells], .9)),
        drop_error_p90=float(np.quantile([c['drop_error_pp'] for c in cells], .9)))
    return summary, cells


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--runs', type=Path, nargs='+', required=True)
    args = parser.parse_args()
    sources = []
    for run in args.runs:
        cfg = json.loads((run/'config.json').read_text())
        assert not cfg['smoke'] and len(cfg['seeds']) == 5
        assert json.loads((run/'progress.json').read_text())['training_complete']
        assert json.loads((run/'independent_validation_report.json').read_text())['passed']
        sources.append((run, cfg))
    datasets = {cfg['plan']['dataset'] for _, cfg in sources}; assert len(datasets) == 1
    dataset = datasets.pop()
    baseline_root = NASA_OLD if dataset == 'NASA' else OLD
    frozen = [p for p in json.loads((AUDIT/'splits.json').read_text())['plans'] if p['dataset'] == dataset]
    assert {p['name'] for p in frozen} == {cfg['plan']['name'] for _, cfg in sources}
    for _, cfg in sources:
        assert cfg['plan'] == next(p for p in frozen if p['name'] == cfg['plan']['name'])
    out = Path('experiments/curve_fidelity_outer')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True); snap = out/'source_snapshot'; snap.mkdir()
    files = [Path(__file__), Path('battery_smetimes/curve_fidelity.py'), Path('scripts/train_curve_fidelity.py'),
        Path('battery_smetimes/research_baselines.py'), Path('battery_smetimes/forecast.py'),
        Path('battery_smetimes/physical_types.py'), Path('scripts/train_stm32_compact_trees.py')]
    for p in files:
        shutil.copy2(p, snap/p.name)
    registry_file = AUDIT/'source_snapshot/battery_physical_registry.json'
    registry = load_registry(registry_file)
    save_json(out/'config.json', dict(dataset=dataset,
        validation_sources=[dict(run=str(run.resolve()), config_sha256=sha(run/'config.json'),
            selection_sha256={str(s): sha(run/str(s)/'selection.json') for s in cfg['seeds']}) for run, cfg in sources],
        source_sha256={p.name: sha(p) for p in files}, registry_sha256=sha(registry_file),
        interpretation='public development CV; already observed outer cells; no test candidate selection',
        origin=100, baseline=str(baseline_root.resolve())))
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
        handlers=[logging.FileHandler(out/'training.log', encoding='utf-8'), logging.StreamHandler()])
    rows = []; old_rows = []
    try:
        for run, cfg in sources:
            plan = cfg['plan']; cells = load_subset(plan['train']+plan['validation']+plan['test'])
            for seed in cfg['seeds']:
                selection = json.loads((run/str(seed)/'selection.json').read_text())
                expected = min(selection['candidates'], key=lambda r: r['selection_score'])
                assert expected == selection['selected']
                params = dict(expected['parameters'])
                folder = out/plan['name']/str(seed); folder.mkdir(parents=True)
                save_json(folder/'selection.json', selection)
                if params['arm'] in ('compact', 'ridge'):
                    params['_registry'] = registry
                logging.info('REFIT %s seed=%s selected=%s', plan['name'], seed, expected['parameters'])
                bundle = fit_model(cells, plan['train']+plan['validation'], seed, params)
                assert not set(bundle['train_ids']) & set(plan['test'])
                joblib.dump(bundle, folder/'model.joblib')
                result, arrays = evaluate(bundle, cells, plan['test'])
                save_json(folder/'test_metrics.json', result); np.savez_compressed(folder/'test_predictions.npz', **arrays)
                rows.extend(dict(r, fold=plan['name'], seed=seed, dataset=dataset) for r in result)
                save_json(out/'by_cell.json', rows)
                logging.info('SAVED %s seed=%s outer_cells=%s', plan['name'], seed, len(result))
            # Read OLD outer arrays only for evaluation comparison, never fit.
            for seed in range(20261004, 20261009):
                old_file = (baseline_root/plan['name']/str(seed)/'predictions.npz' if dataset == 'NASA'
                    else baseline_root/plan['name']/str(seed)/'test/predictions.npz')
                with np.load(old_file) as a:
                    for key in plan['test']:
                        cycle, y, p = [a[key+'__100__'+f] for f in ('cycle', 'truth', 'prediction')]
                        old_rows.append(dict(cell_id=key, seed=seed, **curve_metrics(cycle, y, p)))
        summary, cell_rows = summarize(rows); baseline, _ = summarize(old_rows)
        gates = dict(full_mape=summary['mape_percent']['mean'] <= 6,
            late_mape=summary['late_mape']['mean'] <= 6, rmse=summary['rmse_soh']['mean'] <= .04,
            endpoint_mean=summary['endpoint_pp']['mean'] <= 5, drop_mean=summary['drop_error_pp']['mean'] <= 4,
            joint_fraction=summary['joint_pass_fraction'] >= .9, severe_fraction=summary['severe_flat_fraction'] <= .1,
            full_no_regression=summary['mape_percent']['mean'] <= baseline['mape_percent']['mean']+.5)
        for key in ('endpoint_p90', 'drop_error_p90'):
            gates[key] = summary[key] <= (3 if baseline[key] <= 3 else .8*baseline[key])
        save_json(out/'summary.json', dict(dataset=dataset, new=summary, baseline=baseline, gates=gates,
            numeric_gates_passed=bool(all(gates.values())), curve_delivery_complete=False,
            validation_selection_isolation=True, not_blind=True))
        save_json(out/'cell_seed_means.json', cell_rows)
        save_json(out/'progress.json', dict(training_complete=True, all_outer_cells_evaluated=True,
            plots_exported=False, independent_artifact_audit_complete=False, goal_complete=False))
        print('OUTPUT='+str(out.resolve()), flush=True)
    except BaseException:
        save_json(out/'failure.json', dict(traceback=traceback.format_exc(), completed_cell_seed_rows=len(rows)))
        raise


if __name__ == '__main__':
    main()
