"""Every held-out NCA curve, with all saved seeds and no model refitting."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scripts.prepare_same_type_forecast import save_json, sha
from scripts.save_release_soh_diagnostics import audit


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def render(cell_id, model, arrays, rows, base, alignment):
    key = cell_id+'__100__'
    x, y = arrays[0][key+'cycle'], arrays[0][key+'truth']
    assert np.all(np.diff(x)>0)
    for a in arrays[1:]:
        np.testing.assert_array_equal(a[key+'cycle'], x)
        np.testing.assert_array_equal(a[key+'truth'], y)
    p = np.stack([a[key+'prediction'] for a in arrays])
    deterministic = model=='ridge'
    if deterministic:
        for individual in p[1:]:
            np.testing.assert_array_equal(individual, p[0])
    fig, ax = plt.subplots(figsize=(7.2047244094, 3.9370078740))
    fig.subplots_adjust(left=.105, right=.97, bottom=.23, top=.75)
    ax.plot(x, y, color='#333333', linewidth=1.2, label='Observed SOH')
    for i, individual in enumerate(p):
        ax.plot(x, individual, color='#2864A5', linewidth=.7, alpha=.3,
                label='5 Seed Forecasts' if i==0 else None)
    ax.plot(x, p.mean(axis=0), color='#2864A5', linewidth=1.2, linestyle='--', label='Seed Mean')
    ax.set(xlabel='Future source cycle (input: first 100 source cycles)', ylabel='SOH (0–1)')
    ax.margins(x=.015, y=.08); ax.grid(alpha=.18, linewidth=.5)
    label = 'Ridge' if deterministic else 'ExtraTrees'
    mape = np.mean([r['metrics']['mape_percent'] for r in rows])
    late = np.mean([r['late_half']['mape_percent'] for r in rows])
    end = np.mean([r['endpoint_abs_soh_pp'] for r in rows])
    fig.text(.105, .935, f'NCA | {cell_id} | {label}', fontsize=10)
    note = '5 identical deterministic repetitions' if deterministic else '5 seeds; seed variation, not confidence interval'
    fig.text(.105, .87, 'Unseen-cell 5-fold — public development evaluation', fontsize=8)
    fig.text(.105, .815, f'Mean per-seed MAPE {mape:.2f}% | Late-half {late:.2f}% | Endpoint {end:.2f} pp', fontsize=8)
    fig.text(.105, .115, note, fontsize=8)
    fig.legend(*ax.get_legend_handles_labels(), loc='lower left', bbox_to_anchor=(.105,.015), ncol=3, frameon=False, fontsize=8)
    fig.canvas.draw(); alignment(fig, json_out=str(base)+'.alignment.json', strict=True)
    fig.savefig(str(base)+'.pdf'); fig.savefig(str(base)+'.svg')
    fig.savefig(str(base)+'.png', dpi=600); fig.savefig(str(base)+'.preview.png', dpi=300)
    plt.close(fig)
    return dict(mean_per_seed_mape=float(mape), mean_late_half_mape=float(late),
                mean_endpoint_soh_pp=float(end), deterministic_repetitions=deterministic)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--verification', type=Path, required=True)
    parser.add_argument('--figure-tools', type=Path, required=True)
    args=parser.parse_args(); verified=read(args.verification); assert verified['passed']
    run=Path(verified['run']); cfg=read(run/'config.json')
    assert verified['final_models']==50 and verified['cell_seed_model_evaluations']==550
    rows=read(args.verification.parent/'by_cell.json')
    tools=args.figure_tools.resolve(); sys.path.insert(0,str(tools))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    matplotlib.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
        'font.size':8,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.right':False,
        'axes.spines.top':False,'axes.linewidth':.8,'legend.frameon':False})
    out=Path('results/zhu2022_nca_heldout_curves')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True); shutil.copy2(__file__,out/Path(__file__).name)
    shutil.copy2('docs/zhu2022_nca_curve_contract.md',out/'figure_contract.md')
    save_json(out/'config.json',dict(run=str(run.resolve()), verification_sha256=sha(args.verification),
        source_sha256=sha(__file__), metric_definition='mean individual seed metrics; not ensemble metrics',
        seeds=cfg['seeds'], models=list(cfg['candidates']), visual_review='pending'))
    preflight=subprocess.run([sys.executable,str(tools/'validate_figure.py'),__file__,'--json'],capture_output=True,text=True)
    save_json(out/'source_preflight.json',dict(exit_code=preflight.returncode,stdout=preflight.stdout,stderr=preflight.stderr))
    if preflight.returncode:raise RuntimeError(preflight.stdout)
    figures=[]; sources=[]
    try:
        for model in cfg['candidates']:
            folder=out/model; folder.mkdir()
            for plan in cfg['plans']:
                arrays=[]
                for seed in cfg['seeds']:
                    path=run/model/plan['name']/str(seed)/'predictions.npz'
                    with np.load(path,allow_pickle=False) as archive:
                        arrays.append({k:archive[k] for k in archive.files})
                    sources.append(dict(path=str(path.resolve()),sha256=sha(path)))
                for cell_id in plan['test']:
                    cellrows=[r for r in rows if r['model']==model and r['fold']==plan['name'] and r['cell_id']==cell_id]
                    assert [r['seed'] for r in cellrows]==cfg['seeds']
                    base=folder/cell_id
                    info=render(cell_id,model,arrays,cellrows,base,require_matplotlib_panel_alignment)
                    audits=audit(base,tools)
                    figures.append(dict(cell_id=cell_id, model=model, fold=plan['name'], seeds=cfg['seeds'],
                        figure=str(base.resolve()),metrics=info,audits=audits,visual_review='pending',
                        png_sha256=sha(Path(str(base)+'.png'))))
                    save_json(out/'progress.json',dict(exported=len(figures),expected=110,complete=False,visual_review='pending'))
                print(model,plan['name'],'curves saved',len(figures),flush=True)
        assert len(figures)==110
        save_json(out/'manifest.json',dict(complete=True,all_eligible_cells_included=True,
            figures=figures,sources=sources,visual_review='pending',verification_sha256=sha(args.verification)))
        save_json(out/'progress.json',dict(exported=110,expected=110,complete=True,visual_review='pending'))
        print('OUTPUT='+str(out.resolve()),flush=True)
    except BaseException:
        save_json(out/'failure.json',dict(traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
