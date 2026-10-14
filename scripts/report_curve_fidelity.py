"""Recompute validation metrics from saved arrays and summarize seed repeats."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from scripts.prepare_same_type_forecast import save_json, sha
from scripts.train_curve_fidelity import load_subset


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--run', type=Path, required=True)
    root = parser.parse_args().run
    cfg = json.loads((root/'config.json').read_text())
    for name, digest in cfg['source_sha256'].items():
        assert sha(root/'source_snapshot'/name) == digest
    plan = cfg['plan']; cells = load_subset(plan['train']+plan['validation'])
    assert not set(cells) & set(plan['test'])
    for key, digest in cfg['prepared_sha256'].items():
        assert sha(cells[key]['prepared_file']) == digest
    checked = 0; grouped = {}; summaries = []; selected_cells = {}
    for seed in cfg['seeds']:
        selected = json.loads((root/str(seed)/'selection.json').read_text())['selected']['candidate']
        for index, params in enumerate(cfg['candidates']):
            folder = root/str(seed)/f'candidate_{index}'
            weight = joblib.load(folder/'model.joblib')
            assert set(weight['train_ids']) == set(plan['train'])
            assert not set(weight['train_ids']) & set(plan['validation']+plan['test'])
            rows = json.loads((folder/'validation_metrics.json').read_text())
            assert {r['cell_id'] for r in rows} == set(plan['validation'])
            with np.load(folder/'validation_predictions.npz') as arrays:
                for r in rows:
                    key = r['cell_id']; c = cells[key]
                    take = np.isfinite(c['soh']) & (c['soh'] > 0) & (c['label_cycle'] > 100)
                    x, y, p = [arrays[key+'__100__'+field] for field in ('cycle', 'truth', 'prediction')]
                    np.testing.assert_array_equal(x, c['label_cycle'][take])
                    np.testing.assert_array_equal(y, c['soh'][take])
                    late = x >= (x[0]+x[-1])/2
                    values = dict(mape_percent=float(100*np.mean(abs(p-y)/y)),
                        rmse_soh=float(np.sqrt(np.mean((p-y)**2))),
                        late_mape=float(100*np.mean(abs(p[late]-y[late])/y[late])),
                        endpoint_pp=float(100*abs(p[-1]-y[-1])),
                        drop_error_pp=float(100*abs((p[0]-p[-1])-(y[0]-y[-1]))))
                    for field, value in values.items():
                        np.testing.assert_allclose(value, r[field], rtol=0, atol=1e-10)
                    checked += 1
            summary = json.loads((folder/'summary.json').read_text())
            grouped.setdefault(index, []).append(summary)
            if index == selected:
                summaries.append(summary)
                for row in rows:
                    selected_cells.setdefault(row['cell_id'], []).append(row)
    fields = ('mape_percent', 'late_mape', 'endpoint_pp', 'drop_error_pp', 'rmse_soh', 'joint_pass')
    output = []
    for index, rows in grouped.items():
        output.append(dict(candidate=index, parameters=cfg['candidates'][index],
            **{key: dict(mean=float(np.mean([r[key] for r in rows])),
                seed_sd=float(np.std([r[key] for r in rows], ddof=1)) if len(rows) > 1 else None)
                for key in fields}))
    by_cell = []
    for key, rows in selected_cells.items():
        assert len(rows) == len(cfg['seeds'])
        means = {field: float(np.mean([r[field] for r in rows])) for field in fields if field != 'joint_pass'}
        joint = (means['mape_percent'] <= 6 and means['late_mape'] <= 8
            and means['endpoint_pp'] <= 8 and means['drop_error_pp'] <= 5)
        by_cell.append(dict(cell_id=key, **means, joint_pass=bool(joint)))
    report = dict(passed=True, scope='one frozen fold training-side validation, NOT full outer CV',
        original_test_loaded=False, checked_cell_seed_candidate_arrays=checked,
        seeds=len(cfg['seeds']), candidates=output,
        selected_by_validation= [dict(seed=r['seed'], candidate=r['candidate']) for r in summaries],
        selected_means={key: float(np.mean([r[key] for r in summaries])) for key in fields},
        selected_by_cell_seed_means=by_cell,
        joint_pass_fraction_on_cell_seed_means=float(np.mean([r['joint_pass'] for r in by_cell])),
        goal_complete=False, plots_exported=False,
        source_sha256=sha(__file__))
    save_json(root/'independent_validation_report.json', report)
    print(json.dumps(report['selected_means'], indent=2), flush=True)
    print('CHECKED='+str(checked), flush=True)


if __name__ == '__main__':
    main()
