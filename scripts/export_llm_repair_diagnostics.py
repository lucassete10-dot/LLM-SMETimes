"""All raw saved TRAIN curves, fifth-example layout, no smoothing/alignment."""
import argparse,json
from pathlib import Path
import sys
import shutil
import subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from scripts.prepare_same_type_forecast import save_json
from scripts.audit_llm_curve_repair import fingerprint


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--output-name',default='figures');args=ap.parse_args();root=args.run
    if Path(args.output_name).name!=args.output_name or args.output_name in ('.','..'):
        raise ValueError('figure output must be a new child folder name')
    tools=Path('C:/Users/Lenovo/.codex/skills/nature-figure/scripts');sys.path.insert(0,str(tools))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    cfg=json.loads((root/'config.json').read_text());metrics=json.loads((root/'TRAIN_metrics.json').read_text())
    out=root/args.output_name;out.mkdir(exist_ok=False)
    (out/'figure_contract.md').write_text(
        'Question: Can the actual end-to-end pretrained LLM+SMETimes represent all twelve frozen TRAIN curves?\n'
        'Claim: These are in-sample P1 diagnostics, not unseen-cell predictions. Raw observations and saved model forecasts are both shown.\n'
        'Archetype: single-panel quantitative trajectory, one cell per page; 12 pages, no excluded cells.\n'
        'Python-only existing workflow; 8 x 5 inches, editable PDF/SVG, 300 dpi PNG, text >=8 pt.\n'
        'One seed; no uncertainty band or inferential statistics. Source cycle -> x; initial-capacity-reference SOH -> y.\n',encoding='utf-8')
    shutil.copy2(__file__,out/'plot_source.py')
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':9,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    rows=[]
    with np.load(root/'TRAIN_predictions.npz') as source,PdfPages(out/'all_TRAIN_curves.pdf') as pdf:
        for cid in cfg['train_ids']:
            prefix=f'{cid}__{cfg["origin"]}__';x,y,p=[source[prefix+k] for k in ['cycle','truth','prediction']]
            assert np.all(np.diff(x)>0) and np.isfinite(np.c_[x,y,p]).all()
            row=metrics[cid];fig=plt.figure(figsize=(8,5));ax=fig.add_axes([.12,.20,.85,.55])
            ax.plot(x,y,color='#222222',lw=1.5,label='Observed SOH')
            ax.plot(x,p,color='#2673A6',lw=1.5,label='LLM + SMETimes forecast')
            ax.set(xlabel=f'Future source cycle (input: first {cfg["origin"]} cycles)',ylabel='SOH (initial capacity reference)')
            ax.grid(alpha=.15)
            fig.text(.12,.95,'MIT | '+cid,fontsize=12)
            fig.text(.12,.89,f'{cfg["architecture"]} | {cfg["fold"]} | TRAIN in-sample diagnosis | Seed {cfg["seed"]}',fontsize=8)
            fig.text(.12,.83,f'MAPE {row["mape_percent"]:.3f}% | MAE {row["mae_pp"]:.3f} pp | Tail {row["tail_mae_pp"]:.3f} pp | Stage {row["stage_drop_max_pp"]:.3f} pp',fontsize=8)
            fig.text(.12,.78,'P1 tiny-fit gate: '+('Passed' if all(row[k]<=v for k,v in cfg['tiny_gates_pp'].items()) else 'Not passed'),fontsize=9)
            fig.legend(*ax.get_legend_handles_labels(),loc='lower left',bbox_to_anchor=(.105,.03),ncol=2,fontsize=9,frameon=False)
            stem=out/cid;require_matplotlib_panel_alignment(fig,json_out=str(stem)+'.alignment.json',strict=True)
            fig.savefig(str(stem)+'.png',dpi=300);fig.savefig(str(stem)+'.svg');fig.savefig(str(stem)+'.pdf');pdf.savefig(fig);plt.close(fig)
            rows.append({'cell_id':cid,'png':str(stem)+'.png','source_predictions_sha256':fingerprint(root/'TRAIN_predictions.npz')})
    qa=[]
    for tool,extra in [('audit_pdf_text.py',['--min-pt','5']),
        ('audit_figure_collisions.py',['--json-out',str(out/'collision_audit.json')])]:
        proc=subprocess.run([sys.executable,str(tools/tool),str(out/'all_TRAIN_curves.pdf'),*extra],capture_output=True,text=True)
        qa.append({'tool':tool,'exit_code':proc.returncode,'output':proc.stdout,'error':proc.stderr})
    save_json(out/'manifest.json',{'figures':rows,'all_tiny_cells_included':len(rows)==len(cfg['train_ids']),
        'qa':qa,'visual_review':'pending','goal_complete':False})
    if any(r['exit_code']!=0 for r in qa):raise ValueError('figure QA failed; retain raw artifacts for diagnosis')
    print('OUTPUT='+str(out.resolve()),flush=True)


if __name__=='__main__':main()
