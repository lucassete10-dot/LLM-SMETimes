"""Plot existing held-out SOH estimates across four public battery datasets.

These plots read saved development OOF predictions. They do not refit models
or present observed future measurements as an early-life forecast.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path('battery_models_v2')
OUT = ROOT / 'public_soh_cv_plots'
DOMAINS = ('NASA', 'CALCE', 'MIT', 'XJTU')


def load_domain(domain):
    source = ROOT / 'soh' / domain.lower()
    with np.load(source / 'development_oof_predictions.npz', allow_pickle=False) as data:
        archive = {name: data[name].copy() for name in data.files}
    ids = np.unique(archive['cell_id'])
    rows = []
    for key in ids:
        keep = archive['cell_id'] == key
        actual = archive['target'][keep]
        estimate = archive['estimate'][keep]
        if not len(actual) or np.any(actual <= 0) or not np.all(np.isfinite(estimate)):
            raise ValueError(f'{domain}/{key}: invalid scored predictions')
        folds = np.unique(archive['held_out'][keep])
        if len(folds) != 1:
            raise ValueError(f'{domain}/{key}: predictions span multiple validation folds')
        rows.append({'cell_id': str(key), 'held_out': str(folds[0]),
                     'n': int(keep.sum()),
                     'mae': float(np.mean(np.abs(estimate-actual))),
                     'mape_percent': float(100*np.mean(np.abs(estimate-actual)/actual))})
    selected = json.loads((source / 'training_manifest.json').read_text(encoding='utf-8'))
    if len(rows) != selected['cells']:
        raise ValueError(f'{domain}: OOF cell count differs from model manifest')
    return archive, rows, selected['group_cv_macro_mae'][selected['selected_model']]


def plot(kind, loaded):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=160)
    chosen = {}
    for ax, domain in zip(axes.flat, DOMAINS):
        archive, rows, macro_mae = loaded[domain]
        ranked = sorted(rows, key=lambda row: (row['mae'], row['cell_id']))
        row = ranked[len(ranked)//2] if kind == 'median' else ranked[-1]
        key = row['cell_id']
        mask = archive['cell_id'] == key
        cycle = archive['cycle'][mask]
        actual = archive['target'][mask]
        estimate = archive['estimate'][mask]
        if np.any(np.diff(cycle) <= 0):
            raise ValueError(f'{domain}/{key}: scored cycles are not increasing')
        ax.plot(cycle, actual, color='#202124', lw=2.3, label='Measured SOH')
        ax.plot(cycle, estimate, color='#087e8b', lw=1.8, label='Held-out estimate')
        ax.axhline(.8, color='#c94747', lw=1.15, linestyle='--', label='80% SOH')
        ax.set_title(f'{domain}  |  {key}  |  held out: {row["held_out"]}', fontsize=13)
        ax.set_xlabel('Source cycle / diagnostic index')
        ax.set_ylabel('SOH')
        ax.grid(alpha=.22)
        low = min(float(actual.min()), float(estimate.min()), .8)
        high = max(float(actual.max()), float(estimate.max()), 1.)
        ax.set_ylim(max(0, low-.04), high+.04)
        ax.text(.02, .03,
                f'Cell MAE {row["mae"]:.4f}  |  MAPE {row["mape_percent"]:.2f}%\n'
                f'Dataset held-out macro MAE {macro_mae:.4f}',
                transform=ax.transAxes, fontsize=9,
                bbox={'facecolor': 'white', 'edgecolor': '#dddddd', 'alpha': .9})
        chosen[domain] = row
    axes.flat[0].legend(loc='upper right', fontsize=9)
    description = ('Median-error cell per dataset' if kind == 'median' else
                   'Highest-error cell per dataset')
    fig.suptitle(f'Public battery SOH | {description} | whole-cell/batch validation',
                 fontsize=17)
    fig.text(.5, .015,
             'Each point estimates current SOH from data available through that cycle; '
             'this is not an early-life forecast of the full curve.',
             ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .035, 1, .96))
    output = OUT / f'{kind}_error_cells.png'
    fig.savefig(output, bbox_inches='tight')
    plt.close(fig)
    return output, chosen


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    loaded = {domain: load_domain(domain) for domain in DOMAINS}
    summary = {}
    for kind in ('median', 'worst'):
        path, chosen = plot(kind, loaded)
        summary[kind] = {'plot': str(path), 'cells': chosen}
        print('WROTE', path, flush=True)
    (OUT / 'selection.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
