"""All held-out cells, old/new seed means and SD; no refit or test selection."""
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
from scripts.prepare_same_type_forecast import save_json, sha
from scripts.save_release_soh_diagnostics import audit
from battery_smetimes.curve_fidelity import curve_metrics


def curves(folder, fold, seeds, key, filename):
    predictions = []; hashes = {}
    for seed in seeds:
        file = folder/fold/str(seed)/filename
        with np.load(file) as a:
            x, y, p = [a[key+'__100__'+f] for f in ('cycle', 'truth', 'prediction')]
        if predictions:
            np.testing.assert_array_equal(x, reference_x); np.testing.assert_array_equal(y, reference_y)
        reference_x, reference_y = x, y
        predictions.append(p); hashes[str(file.resolve())] = sha(file)
    p = np.stack(predictions)
    return x, y, p.mean(axis=0), p.std(axis=0, ddof=1), hashes, predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--figure-tools', type=Path, required=True)
    args = parser.parse_args(); root = args.run; tools = args.figure_tools
    cfg = json.loads((root/'config.json').read_text()); old = Path(cfg['baseline'])
    assert json.loads((root/'progress.json').read_text())['all_outer_cells_evaluated']
    plans = [json.loads((Path(r['run'])/'config.json').read_text())['plan'] for r in cfg['validation_sources']]
    seeds = list(range(20261005, 20261010)); old_seeds = list(range(20261004, 20261009))
    out = Path('results/curve_fidelity')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True)
    shutil.copy2(__file__, out/Path(__file__).name)
    shutil.copy2('scripts/save_release_soh_diagnostics.py', out/'audit_helper.py')
    shutil.copy2('docs/curve_fidelity_figure_contract.md', out/'figure_contract.md')
    preflight = subprocess.run([sys.executable, str(tools/'validate_figure.py'), __file__, '--json'], capture_output=True, text=True)
    save_json(out/'source_preflight.json', dict(exit_code=preflight.returncode, stdout=preflight.stdout, stderr=preflight.stderr))
    if preflight.returncode:
        raise RuntimeError('source preflight failed: '+preflight.stdout)
    sys.path.insert(0, str(tools.resolve()))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    matplotlib.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['DejaVu Sans'],
        'font.size':8, 'svg.fonttype':'none', 'pdf.fonttype':42,
        'axes.spines.right':False, 'axes.spines.top':False, 'legend.frameon':False})
    manifest = []
    for plan in plans:
        for key in plan['test']:
            x, y, mean, sd, new_hashes, ps = curves(root, plan['name'], seeds, key, 'test_predictions.npz')
            bx, by, bmean, bsd, old_hashes, bps = curves(old, plan['name'], old_seeds, key, 'test/predictions.npz')
            np.testing.assert_array_equal(x, bx); np.testing.assert_array_equal(y, by)
            assert np.all(np.diff(x) > 0)
            np.savez_compressed(out/(key+'.source.npz'), cycle=x, truth=y, new_mean=mean,
                new_seed_sd=sd, old_mean=bmean, old_seed_sd=bsd)
            metrics = [curve_metrics(x, y, p) for p in ps]
            before = [curve_metrics(x, y, p) for p in bps]
            new = {f: float(np.mean([r[f] for r in metrics])) for f in ('mape_percent','endpoint_pp','drop_error_pp')}
            previous = {f: float(np.mean([r[f] for r in before])) for f in new}
            fig, ax = plt.subplots(figsize=(7.2047244094, 3.937007874))
            fig.subplots_adjust(left=.105, right=.97, bottom=.24, top=.73)
            ax.plot(x, y, color='#222222', linewidth=1.2, label='Observed SOH')
            ax.plot(x, bmean, color='#777777', linestyle='--', linewidth=1., label='Previous compact tree')
            ax.fill_between(x, bmean-bsd, bmean+bsd, color='#777777', alpha=.12, linewidth=0)
            ax.plot(x, mean, color='#2864A5', linewidth=1.3, label='Validation-selected model')
            ax.fill_between(x, mean-sd, mean+sd, color='#2864A5', alpha=.18, linewidth=0)
            ax.set(xlabel='Future source cycle (input: first 100 cycles)', ylabel='SOH (0–1)')
            ax.grid(alpha=.18, linewidth=.5); ax.margins(x=.015, y=.08)
            fig.text(.105, .94, f'{cfg["dataset"]} | {key}', fontsize=10)
            fig.text(.105, .88, 'Held-out-cell development CV | Mean ± 1 seed SD (n=5; not a CI)', fontsize=8)
            fig.text(.105, .82, f'MAPE: {previous["mape_percent"]:.2f}% → {new["mape_percent"]:.2f}% | '
                f'Endpoint: {previous["endpoint_pp"]:.2f} → {new["endpoint_pp"]:.2f} pp', fontsize=8)
            fig.text(.105, .765, f'Drop error: {previous["drop_error_pp"]:.2f} → {new["drop_error_pp"]:.2f} SOH pp', fontsize=8)
            fig.legend(*ax.get_legend_handles_labels(), loc='lower left', bbox_to_anchor=(.105, .01),
                ncol=2, fontsize=8, frameon=False)
            fig.canvas.draw(); base = out/key
            require_matplotlib_panel_alignment(fig, json_out=str(base)+'.alignment.json', strict=True)
            fig.savefig(str(base)+'.pdf'); fig.savefig(str(base)+'.svg')
            fig.savefig(str(base)+'.png', dpi=600); fig.savefig(str(base)+'.preview.png', dpi=300)
            plt.close(fig)
            audits = audit(base, tools)
            manifest.append(dict(cell_id=key, fold=plan['name'], new=new, previous=previous,
                new_predictions_sha256=new_hashes, baseline_predictions_sha256=old_hashes,
                source_sha256=sha(out/(key+'.source.npz')), png_sha256=sha(Path(str(base)+'.png')),
                audits=audits, visual_review='pending'))
            save_json(out/'manifest.json', dict(complete=False, run=str(root.resolve()), figures=manifest,
                all_cells_included=False, source_sha256=sha(__file__)))
            print('SAVED '+key, flush=True)
    assert len(manifest) == len({r['cell_id'] for r in manifest})
    save_json(out/'manifest.json', dict(complete=True, run=str(root.resolve()), figures=manifest,
        all_cells_included=True, source_sha256=sha(__file__), visual_review='pending'))
    print('OUTPUT='+str(out.resolve()), flush=True)


if __name__ == '__main__':
    main()
