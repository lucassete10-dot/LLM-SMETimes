"""All audited development-CV curves, no refit, smoothing or adoption decision."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts.audit_four_cohort_progress import SOURCES
from scripts.audit_rate_blend_metrics import metrics as arithmetic
from scripts.export_tail_curve_fidelity import curves
from scripts.prepare_same_type_forecast import save_json, sha
from scripts.save_release_soh_diagnostics import audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--figure-tools', type=Path, required=True)
    args = parser.parse_args()
    tools = args.figure_tools.resolve()
    sys.path.insert(0, str(tools))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    root = Path('results/four_cohort_goal_curves') / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    root.mkdir(parents=True)
    print('OUTPUT=' + str(root.resolve()), flush=True)
    shutil.copy2(__file__, root / Path(__file__).name)
    shutil.copy2('docs/soh_four_cohort_figure_contract_20261007.md', root / 'figure_contract.md')
    preflight = subprocess.run([sys.executable, str(tools / 'validate_figure.py'), __file__, '--json'], capture_output=True, text=True)
    save_json(root / 'source_preflight.json', dict(exit_code=preflight.returncode, stdout=preflight.stdout, stderr=preflight.stderr))
    assert preflight.returncode == 0, preflight.stdout
    matplotlib.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                               'font.size': 8, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
                               'axes.spines.right': False, 'axes.spines.top': False, 'legend.frameon': False})
    manifest, overviews = [], []

    def export(fig, base):
        require_matplotlib_panel_alignment(fig, json_out=str(base) + '.alignment.json', strict=True,
                                          tolerance_pt=1.5, gutter_tolerance_pt=1.5)
        fig.savefig(str(base) + '.pdf')
        fig.savefig(str(base) + '.svg')
        fig.savefig(str(base) + '.png', dpi=300)
        plt.close(fig)
        return audit(base, tools)

    for dataset, path, count, proof_name, _ in SOURCES:
        source = Path(path)
        cfg = json.loads((source / 'config.json').read_text())
        assert json.loads((source / proof_name).read_text())['passed']
        if 'plans' in cfg:
            plans, seeds = cfg['plans'], cfg['seeds']
        else:
            parent = Path(cfg['training_source'])
            assert sha(parent / 'config.json') == cfg['training_config_sha256']
            parent_cfg = json.loads((parent / 'config.json').read_text())
            plans = [p for p in parent_cfg['plans'] if p['dataset'] == cfg['dataset']]
            seeds = parent_cfg['seeds']
        ids = [k for p in plans for k in p['test']]
        assert len(ids) == len(set(ids)) == count
        folder = root / dataset.replace(' ', '_')
        folder.mkdir()
        local = []
        for plan in plans:
            for key in plan['test']:
                x, y, p, sd, hashes, stack = curves(source, plan['name'], seeds, key, 'test_predictions.npz')
                assert np.all(np.diff(x) > 0) and x[0] > 100
                assert np.isfinite(np.r_[x, y, p, sd]).all()
                deterministic = dataset == 'NASA'
                if deterministic:
                    assert all(np.array_equal(stack[0], item) for item in stack)
                # Historical by_cell files need not use the current five-seed
                # aggregation; independently derive displayed metrics from NPZ.
                per_seed = [arithmetic(x, y, item) for item in stack]
                row = {field: float(np.mean([r[field] for r in per_seed])) for field in per_seed[0]}
                row.update(cell_id=key)
                row['joint_pass'] = all(row[f] <= limit for f, limit in
                                        [('mape_percent', 6), ('late_mape', 8), ('endpoint_pp', 8), ('drop_error_pp', 5)])
                row['severe_flat_failure'] = row['truth_drop_pp'] >= 5 and row['prediction_drop_pp'] < .3 * row['truth_drop_pp']
                base = folder / key
                np.savez_compressed(str(base) + '.source.npz', cycle=x, truth=y,
                                    seed_predictions=np.stack(stack), mean=p, sample_seed_sd=sd)
                fig, ax = plt.subplots(figsize=(183 / 25.4, 110 / 25.4))
                fig.subplots_adjust(left=.105, right=.97, bottom=.21, top=.72)
                if not deterministic:
                    ax.fill_between(x, p - sd, p + sd, color='#2468A0', alpha=.16, linewidth=0)
                ax.plot(x, y, color='#252525', linewidth=1.5, label='Observed SOH')
                ax.plot(x, p, color='#2468A0', linewidth=1.4, label='Audited forecast')
                ax.set_xlabel('Future source cycle (input: first 100 source cycles)')
                ax.set_ylabel('SOH (0–1)')
                ax.grid(alpha=.14)
                fig.text(.105, .94, dataset + ' | ' + key, fontsize=11)
                repeat = '5 deterministic repeats; not a CI' if deterministic else 'Mean ± 1 seed SD (n=5; not a CI)'
                fig.text(.105, .89, 'Held-out-cell development CV | ' + repeat, fontsize=8)
                fig.text(.105, .84, f"Seed-mean MAPE {row['mape_percent']:.2f}% | Late {row['late_mape']:.2f}% | Endpoint {row['endpoint_pp']:.2f} pp | Drop {row['drop_error_pp']:.2f} pp", fontsize=8)
                short = len(x) < 10 or x[-1] - x[0] < 100
                status = 'Joint criteria passed' if row['joint_pass'] else 'Joint criteria NOT passed'
                if short:
                    status += f' | Short coverage: {len(x)} points, span {x[-1] - x[0]:.0f} cycles'
                fig.text(.105, .79, status, fontsize=8)
                fig.legend(*ax.get_legend_handles_labels(), loc='lower left', bbox_to_anchor=(.105, .01), ncol=2, fontsize=8)
                qa = export(fig, base)
                record = dict(dataset=dataset, cell_id=key, fold=plan['name'], figure=str(base.resolve()),
                              metrics=row, source_npz_sha256=sha(Path(str(base) + '.source.npz')),
                              prediction_files_sha256=hashes, audits=qa, short_coverage=bool(short),
                              visual_review='pending', adoption_status='unchanged; separately reviewed')
                manifest.append(record)
                local.append(record)
                save_json(root / 'progress.json', dict(terminal=False, exported_cells=len(manifest), total_cells=248, goal_complete=False))
                print('EXPORTED ' + dataset + ' ' + key, flush=True)
        fig, ax = plt.subplots(figsize=(183 / 25.4, 110 / 25.4))
        fig.subplots_adjust(left=.105, right=.97, bottom=.19, top=.76)
        means = [r['metrics']['mape_percent'] for r in local]
        deviations = []
        for record in local:
            with np.load(record['figure'] + '.source.npz') as a:
                values = 100 * np.mean(abs(a['seed_predictions'] - a['truth']) / a['truth'], axis=1)
                deviations.append(float(np.std(values, ddof=1)))
                np.testing.assert_allclose(np.mean(values), record['metrics']['mape_percent'], rtol=0, atol=1e-10)
        positions = np.arange(1, len(local) + 1)
        ax.errorbar(positions, means, yerr=None if dataset == 'NASA' else deviations,
                    fmt='o', markersize=3, color='#2468A0', linewidth=.7, capsize=1.5, label='Cell MAPE')
        ax.axhline(6, color='#777777', linestyle='--', linewidth=.8, label='6% per-cell reference')
        ax.set_xlabel('Cell index in frozen evaluation list (IDs in manifest)')
        ax.set_ylabel('Future SOH MAPE (%)')
        ax.grid(alpha=.14)
        fig.text(.105, .94, dataset + f' | All {count} held-out cells', fontsize=11)
        fig.text(.105, .89, 'Development CV; no cells removed based on prediction error', fontsize=8)
        fig.text(.105, .84, 'Deterministic repetitions; no error bars' if dataset == 'NASA' else 'Points: 5-seed cell mean; error bars: ±1 sample seed SD (not a CI)', fontsize=8)
        fig.legend(*ax.get_legend_handles_labels(), loc='lower left', bbox_to_anchor=(.105, .01), ncol=2, fontsize=8)
        base = folder / 'overview'
        qa = export(fig, base)
        overviews.append(dict(dataset=dataset, figure=str(base.resolve()), cell_order=ids, audits=qa, visual_review='pending'))
    assert len(manifest) == 248 and len(overviews) == 4
    save_json(root / 'manifest.json', dict(figures=manifest, overviews=overviews, all_cells_included=True,
              exported_cells=248, exported_overviews=4, visual_review='pending', goal_complete=False,
              source_sha256=sha(__file__), contract_sha256=sha(root / 'figure_contract.md')))
    save_json(root / 'progress.json', dict(terminal=True, passed=True, exported_cells=248, overviews=4,
                                         visual_review='pending', goal_complete=False))


if __name__ == '__main__':
    main()
