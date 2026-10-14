"""Export every NCA held-out development curve, without smoothing or baselines."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SKILL = Path('C:/Users/Lenovo/.codex/skills/nature-figure/scripts')
sys.path.insert(0, str(SKILL))
from audit_panel_alignment import require_matplotlib_panel_alignment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    root = args.run.resolve()
    assert json.loads((root / 'verification.json').read_text())['passed']
    cfg = json.loads((root / 'config.json').read_text())
    rows = json.loads((root / 'all_rows.json').read_text())
    out = Path('deliveries') / ('NCA_main_' + root.name)
    folder = out / 'Zhu2022_NCA' / 'input_100'
    folder.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'sans-serif',
                         'font.sans-serif': ['Arial', 'DejaVu Sans'], 'font.size': 9,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none',
                         'axes.spines.top': False, 'axes.spines.right': False})
    manifest = []
    for plan in cfg['plans']:
        arrays = [dict(np.load(root / plan['name'] / '100' / str(s) / 'predictions.npz',
                               allow_pickle=False)) for s in cfg['seeds']]
        for key in plan['validation']:
            c = arrays[0][key + '__cycle']
            truth = arrays[0][key + '__truth']
            assert np.all(np.diff(c) > 0) and np.all(c > 100)
            assert np.all(np.isfinite(truth)) and np.all(truth > 0)
            for a in arrays:
                assert np.array_equal(c, a[key + '__cycle'])
                assert np.array_equal(truth, a[key + '__truth'])
            stack = np.stack([a[key + '__prediction'] for a in arrays])
            assert np.all(np.isfinite(stack))
            center, spread = stack.mean(0), stack.std(0, ddof=1)
            cell_rows = [r for r in rows if r['origin'] == 100 and
                         r['cell_id'] == key and r['horizon'] == 'operational']
            assert len(cell_rows) == 5
            mape = np.array([r['mape_percent'] for r in cell_rows])
            late = np.mean([r['late20_mae_pp'] for r in cell_rows])
            stage = np.mean([r['stage_error_pp'] for r in cell_rows])
            fig = plt.figure(figsize=(8, 5))
            ax = fig.add_axes([.12, .19, .85, .56])
            ax.plot(c, truth, color='#222222', lw=1.6, label='Observed SOH')
            ax.plot(c, center, color='#2673A6', lw=1.6, label='Forecast mean')
            ax.fill_between(c, center-spread, center+spread, color='#2673A6',
                            alpha=.15, linewidth=0, label='One seed SD (not a CI)')
            ax.set(xlabel='Future source cycle | input: first 100 source cycles',
                   ylabel='SOH (capacity / rated capacity)')
            ax.grid(alpha=.15)
            ax.margins(x=.015, y=.07)
            fig.text(.12, .96, 'Zhu2022_NCA | ' + key, fontsize=12, va='top')
            fig.text(.12, .90, 'Held-out development cell | Five seeds | No independent-test claim',
                     fontsize=8, va='top')
            fig.text(.12, .85, f'Seed-mean MAPE {mape.mean():.2f}% ± {mape.std(ddof=1):.2f}%'
                     f' | Late MAE {late:.2f} pp | Stage error {stage:.2f} pp', fontsize=8, va='top')
            fig.legend(*ax.get_legend_handles_labels(), loc='lower left',
                       bbox_to_anchor=(.105, .02), ncol=3, fontsize=8, frameon=False)
            stem = folder / key
            require_matplotlib_panel_alignment(fig, json_out=str(stem)+'.alignment.json',
                                               tolerance_pt=1.5, strict=True)
            fig.savefig(str(stem)+'.png', dpi=300)
            fig.savefig(str(stem)+'.pdf')
            fig.savefig(str(stem)+'.svg')
            plt.close(fig)
            for script, extra in [('audit_pdf_text.py', ['--min-pt', '5']),
                                  ('audit_figure_collisions.py', ['--json-out', str(stem)+'.collision.json'])]:
                subprocess.run([sys.executable, str(SKILL/script), str(stem)+'.pdf', *extra],
                               check=True, stdout=subprocess.DEVNULL)
            source = dict(cell_id=key, fold=plan['name'], input_cycles=100,
                          seeds=cfg['seeds'], cycle=c.tolist(), truth=truth.tolist(),
                          seed_predictions=stack.tolist(), seed_metrics=cell_rows)
            Path(str(stem)+'.source.json').write_text(json.dumps(source), encoding='utf-8')
            manifest.append(dict(cell_id=key, png=str(stem.resolve())+'.png',
                                 seed_mean_mape=float(mape.mean()), late_mae_pp=float(late),
                                 stage_error_pp=float(stage), automated_pdf_qa=True,
                                 individual_visual_review=False))
            print(f'Exported {len(manifest)}/44: {key}', flush=True)
    assert len(manifest) == 44 and len(set(x['cell_id'] for x in manifest)) == 44
    (out/'manifest.json').write_text(json.dumps(dict(source_run=str(root), figures=manifest,
        planned_cells=44, completed_cells=44, images=44, independent_test=False,
        provisional_input_cycles=100, goal_complete=False), indent=2), encoding='utf-8')
    print(out.resolve(), flush=True)


if __name__ == '__main__':
    main()
