"""All TRAIN reconstruction plots, with raw truth and explicit task labels."""
import argparse
import csv
from pathlib import Path
import subprocess
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from battery_smetimes.llm_full_life import read,load_full_life
from battery_smetimes.llm_curve_quality_v3 import curve_metrics_v3
from scripts.train_llm_full_life import save
from scripts.audit_llm_curve_repair import fingerprint


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();run=args.run;out=args.out;out.mkdir(parents=True,exist_ok=True)
    cfg=read(run/'config.json');proof=read(run/'replay_audit.json')
    if not proof['passed'] or fingerprint(run/'model.pt')!=proof['model_sha256']:raise ValueError('Actual replay required')
    _,cells,_,_=load_full_life(cfg['index'])
    skill=Path('C:/Users/Lenovo/.codex/skills/nature-figure/scripts');sys.path.insert(0,str(skill))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':9,
        'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    figures=[];records=[];folder=out/'figures';folder.mkdir(exist_ok=True)
    with np.load(run/'predictions.npz',allow_pickle=False) as arrays,PdfPages(out/'all_cells.pdf') as pdf:
        for cid in cfg['train_ids']:
            x,y,p=(arrays[cid+'__'+k] for k in ('cycle','truth','prediction'))
            cell=cells[cid];take=np.isfinite(cell['soh'])&(cell['soh']>0)
            np.testing.assert_array_equal(x,cell['label_cycle'][take]);np.testing.assert_allclose(y,cell['soh'][take],atol=6e-8,rtol=0)
            y=cell['soh'][take]  # Always draw the untouched original precision.
            m=curve_metrics_v3(x,y,p)
            # Style inherited from earlier reports; this is full-life
            # reconstruction, so no forecast or uncertainty claim is reused.
            fig=plt.figure(figsize=(8,5));ax=fig.add_axes([.12,.20,.85,.55])
            ax.plot(x,y,color='#222222',lw=1.5,marker='o' if cfg['dataset']=='UConn_ISU_ILCC' else None,markersize=2.5,label='Observed SOH')
            ax.plot(x,p,color='#2673A6',lw=1.3,label='LLM + SMETimes reconstruction')
            ax.set(xlabel='Source cycle (full-life TRAIN signals available)',ylabel='SOH (initial capacity reference)');ax.grid(alpha=.15)
            fig.text(.12,.95,cfg['dataset']+' | '+cid,fontsize=12)
            fig.text(.12,.89,'TRAIN full-life reconstruction | Not forecasting | Single seed',fontsize=8)
            fig.text(.12,.84,f"MAPE {m['mape_percent']:.3f}% | MAE {m['mae_pp']:.3f} pp | Tail {m['tail_mae_pp']:.3f} pp",fontsize=8)
            fig.text(.12,.79,f"Endpoint {m['endpoint_error_pp']:.3f} pp | TRAIN-only model selection | Raw observations retained",fontsize=8)
            fig.legend(*ax.get_legend_handles_labels(),loc='lower left',bbox_to_anchor=(.12,.025),frameon=False,ncol=2,borderaxespad=0)
            stem=folder/cid.replace('#','_');require_matplotlib_panel_alignment(fig,json_out=str(stem)+'.alignment.json',strict=True)
            fig.savefig(str(stem)+'.png',dpi=300);fig.savefig(str(stem)+'.svg');pdf.savefig(fig);plt.close(fig)
            figures.append(dict(cell_id=cid,role='TRAIN_reconstruction',png=str(stem.resolve())+'.png'))
            records.append(dict(dataset=cfg['dataset'],cell_id=cid,**m,model_sha256=proof['model_sha256']))
    qa=[]
    for tool,extra in [('audit_pdf_text.py',['--min-pt','5']),('audit_figure_collisions.py',['--json-out',str(out/'collision_audit.json')])]:
        r=subprocess.run([sys.executable,str(skill/tool),str(out/'all_cells.pdf'),*extra],capture_output=True,text=True,encoding='utf-8')
        qa.append(dict(tool=tool,exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr))
    if any(r['exit_code'] for r in qa):raise ValueError('PDF layout audit failed')
    keys=['dataset','cell_id','mape_percent','mae_pp','tail_mae_pp','endpoint_error_pp','stage_drop_max_pp','model_sha256']
    with (out/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(records)
    save(out/'figure_manifest.json',dict(figures=figures,qa=qa,reconstruction_not_forecast=True,
        model_sha256=proof['model_sha256'],pdf_sha256=fingerprint(out/'all_cells.pdf'),visual_review='pending'))
    print(dict(dataset=cfg['dataset'],figures=len(figures)),flush=True)


if __name__=='__main__':main()
