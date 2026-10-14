"""Save all development-fit curves without refitting or touching release files."""
import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from battery_smetimes.external_soh import load_context, load_bundle
from battery_smetimes.research_baselines import evaluate_baseline, regression_metrics
from scripts.train_stm32_compact_trees import predict
from scripts.prepare_same_type_forecast import save_json, sha


def curve(row, arrays, folder, alignment):
    prefix = f'{row["cell_id"]}__100__'
    x, y, p = [arrays[prefix + field] for field in ('cycle', 'truth', 'prediction')]
    if not np.all(np.diff(x) > 0):
        raise ValueError('future observations must be strictly increasing')
    fig, ax = plt.subplots(figsize=(7.2047244094, 3.9370078740))
    fig.subplots_adjust(left=.105, right=.97, bottom=.23, top=.75)
    ax.plot(x, y, color='#333333', linewidth=1.2, label='Observed SOH')
    ax.plot(x, p, color='#2864A5', linestyle='--', linewidth=1.2,
            label=row.get('model_label', 'Release ExtraTrees'))
    ax.set(xlabel=row.get('input_axis_label','Future source cycle (input: first 100 cycles)'), ylabel='SOH (0–1)')
    ax.margins(x=.015, y=.08)
    ax.grid(alpha=.18, linewidth=.5)
    fig.text(.105, .935, f'{row["dataset"]} | {row["cell_id"]}', fontsize=10)
    fig.text(.105, .87, row.get('evaluation_label', 'Development training fit — NOT unseen-cell evaluation'), fontsize=8)
    fig.text(.105, .815, f'MAPE {row["metrics"]["mape_percent"]:.2f}% | '
             f'Late-half {row["late_half"]["mape_percent"]:.2f}% | '
             f'Endpoint {row["endpoint_abs_soh_pp"]:.2f} SOH pp', fontsize=8)
    fig.legend(*ax.get_legend_handles_labels(), loc='lower left',
               bbox_to_anchor=(.105, .015), ncol=2, frameon=False, fontsize=8)
    fig.canvas.draw()
    base = folder / row['cell_id']
    alignment(fig, json_out=str(base)+'.alignment.json', strict=True)
    fig.savefig(str(base)+'.pdf')
    fig.savefig(str(base)+'.svg')
    fig.savefig(str(base)+'.png', dpi=600)
    plt.close(fig)
    return base


def audit(base, tools):
    results = {}
    commands = {
        'text': [str(tools/'audit_pdf_text.py'), str(base)+'.pdf', '--min-pt', '5', '--json'],
        'collision': [str(tools/'audit_figure_collisions.py'), str(base)+'.pdf',
                      '--json-out', str(base)+'.collision.json',
                      '--overlay-pdf', str(base)+'.collision.pdf'],
    }
    for name, command in commands.items():
        run = subprocess.run([sys.executable, *command], capture_output=True, text=True)
        results[name] = dict(exit_code=run.returncode, stdout=run.stdout, stderr=run.stderr)
        save_json(Path(str(base)+f'.{name}-run.json'), results[name])
        if run.returncode:
            raise RuntimeError(f'{name} audit did not pass for {base}: {run.stdout} {run.stderr}')
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--figure-tools', type=Path, required=True)
    parser.add_argument('--dataset', choices=['MIT', 'XJTU', 'NASA'], action='append')
    args = parser.parse_args()
    tools = args.figure_tools.resolve()
    sys.path.insert(0, str(tools))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    matplotlib.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
        'font.size': 8, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
        'axes.spines.right': False, 'axes.spines.top': False, 'axes.linewidth': .8,
        'legend.frameon': False})
    out = Path('results/stm32_release_diagnostics') / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
        handlers=[logging.FileHandler(out/'diagnostics.log', encoding='utf-8'), logging.StreamHandler()])
    logging.getLogger('fontTools').setLevel(logging.WARNING)
    shutil.copy2(__file__, out/Path(__file__).name)
    shutil.copy2('docs/release_curve_contract.md', out/'figure_contract.md')
    for name in ('audit_panel_alignment.py', 'audit_pdf_text.py', 'audit_figure_collisions.py', 'validate_figure.py'):
        shutil.copy2(tools/name, out/name)
    save_json(out/'config.json', dict(package=str(args.package.resolve()),
        release_manifest_sha256=sha(args.package/'manifest.json'), origin=100,
        datasets=args.dataset or ['MIT','XJTU','NASA'],
        role='development_training_fit_only', independent_test=False,
        source_sha256=sha(__file__), figure_contract_sha256=sha('docs/release_curve_contract.md')))
    try:
        preflight = subprocess.run([sys.executable, str(tools/'validate_figure.py'), __file__, '--json'],
                                  capture_output=True, text=True)
        save_json(out/'source_preflight.json', dict(exit_code=preflight.returncode,
                  stdout=preflight.stdout, stderr=preflight.stderr))
        if preflight.returncode:
            raise RuntimeError('plotting source preflight failed: '+preflight.stdout)
        context = load_context(args.package)
        coverage, all_rows = [], []
        for domain in args.dataset or ['MIT', 'XJTU', 'NASA']:
            dest = out/domain
            dest.mkdir()
            bundle = load_bundle(context, domain)
            ids = sorted(k for k,c in context['cells'].items() if c['metadata']['dataset']==domain)
            if set(ids) != set(bundle['train_ids']):
                raise ValueError('diagnostic cells differ from release training membership')
            report, arrays = evaluate_baseline(bundle, context['cells'], ids, predictor=predict)
            report.update(role='development_training_fit_only', independent_test=False,
                          model_sha256=sha(args.package/domain/'model.joblib'))
            for row in report['by_cell']:
                prefix = f'{row["cell_id"]}__100__'
                x,y,p = [arrays[prefix+field] for field in ('cycle','truth','prediction')]
                row.update(dataset=domain, independent_test=False,
                    last100=regression_metrics(y[x>x[-1]-100], p[x>x[-1]-100]),
                    endpoint_abs_soh_pp=float(abs(y[-1]-p[-1])*100))
            save_json(dest/'metrics.json', report)
            np.savez_compressed(dest/'predictions.npz', **arrays)
            all_rows.extend(report['by_cell'])
            folder = dest/'curves'
            folder.mkdir()
            for i,row in enumerate(report['by_cell']):
                base = curve(row, arrays, folder, require_matplotlib_panel_alignment)
                audit(base, tools)
                coverage.append(dict(dataset=domain, cell_id=row['cell_id'],
                    numerical_role='development_training_fit_only',
                    geometry_audit='exit0', visual_review='pending', curve=str(base.relative_to(out))))
                save_json(out/'progress.json', dict(complete=False, exported=len(coverage)))
                if i%10==0: logging.info('%s curves %d/%d', domain, i+1, len(report['by_cell']))
        save_json(out/'by_cell.json', all_rows)
        save_json(out/'figure_qa.json', dict(figures=coverage,
            note='All exported figures have automated audits; panel-by-panel human visual review is separately pending. No unseen-cell accuracy claim.'))
        save_json(out/'progress.json', dict(complete=True, exported=len(coverage), visual_review='pending'))
        save_json(out/'manifest.json', dict(files_sha256={str(p.relative_to(out)):sha(p)
            for p in out.rglob('*') if p.is_file()}, release_unchanged=sha(args.package/'manifest.json'),
            figures=len(coverage), role='development_training_fit_only'))
        print('OUTPUT='+str(out.resolve()))
    except BaseException:
        save_json(out/'failure.json', dict(traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    main()
