"""Development diagnostics: raw curves, no smoothing or acceptance claims."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SKILL=Path('C:/Users/Lenovo/.codex/skills/nature-figure/scripts')
sys.path.insert(0,str(SKILL))
from audit_panel_alignment import require_matplotlib_panel_alignment


def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);root=ap.parse_args().run
    proof=json.loads((root/'independent_replay_verification.json').read_text());assert proof['passed']
    cfg=json.loads((root/'config.json').read_text())
    out=root/'diagnostic_figures';out.mkdir(exist_ok=True);manifest=[]
    # Figure contract: compare frozen paired candidates, including failures.
    # One raw held-out validation cell per plot; single seed, no uncertainty band.
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
                         'font.size':9,'pdf.fonttype':42,'svg.fonttype':'none',
                         'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})
    for plan in cfg['plans']:
        base=root/plan['name']/'100'/str(cfg['seeds'][0])
        a=np.load(base/'point_VIT_leaf1'/'validation_predictions.npz',allow_pickle=False)
        b=np.load(base/'continuous_V_leaf1'/'validation_predictions.npz',allow_pickle=False)
        rows=json.loads((base/'point_VIT_leaf1'/'metrics.json').read_text())
        folder=out/plan['dataset']/'input100';folder.mkdir(parents=True,exist_ok=True)
        for r in rows:
            key=r['cell_id'];c=a[key+'__cycle'];y=a[key+'__truth'];p=a[key+'__prediction'];q=b[key+'__prediction']
            assert np.all(np.diff(c)>0) and np.isfinite(np.r_[y,p,q]).all()
            fig=plt.figure(figsize=(7.2,4.5));ax=fig.add_axes([.12,.17,.85,.60])
            ax.plot(c,y,color='#222222',lw=1.6,label='Observed SOH')
            ax.plot(c,p,color='#2673A6',lw=1.4,label='Point baseline (V/I/T)')
            ax.plot(c,q,color='#B37730',lw=1.4,ls='--',label='Continuous candidate (V)')
            ax.set(xlabel='Future source cycle (input: first 100 cycles)',ylabel='SOH (capacity / rated capacity)')
            ax.grid(alpha=.15);ax.margins(x=.015,y=.08)
            fig.text(.12,.96,plan['dataset']+' | '+key,fontsize=12,va='top')
            fig.text(.12,.895,'Inner validation | 1 fold, 1 seed | Diagnostic comparison, not final acceptance',fontsize=8,va='top')
            fig.legend(*ax.get_legend_handles_labels(),loc='upper left',bbox_to_anchor=(.105,.86),ncol=3,fontsize=8)
            fig.text(.12,.035,f'MAPE: point {100*np.mean(abs(p-y)/y):.2f}% | continuous {100*np.mean(abs(q-y)/y):.2f}% | No curve smoothing',fontsize=8)
            stem=folder/key
            require_matplotlib_panel_alignment(fig,json_out=str(stem)+'.alignment.json',strict=True)
            fig.savefig(str(stem)+'.png',dpi=300)
            fig.savefig(str(stem)+'.pdf')
            fig.savefig(str(stem)+'.svg')
            plt.close(fig)
            subprocess.run([sys.executable,str(SKILL/'audit_pdf_text.py'),str(stem)+'.pdf','--min-pt','5'],check=True,stdout=subprocess.DEVNULL)
            subprocess.run([sys.executable,str(SKILL/'audit_figure_collisions.py'),str(stem)+'.pdf','--json-out',str(stem)+'.collision.json'],check=True,stdout=subprocess.DEVNULL)
            manifest.append(dict(dataset=plan['dataset'],cell_id=key,png=str(stem.resolve())+'.png',pdf=str(stem.resolve())+'.pdf',
                origin=100,seed=cfg['seeds'][0],validation=True,automated_pdf_qa_passed=True,individual_visual_review=False))
        a.close();b.close()
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('FIGURES',len(manifest),str(out.resolve()))


if __name__=='__main__':main()
