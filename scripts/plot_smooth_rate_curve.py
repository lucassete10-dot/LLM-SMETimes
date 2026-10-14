"""All held-out curves for the fixed smooth-rate development pilot."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);root=ap.parse_args().run
    assert json.loads((root/'independent_replay_verification.json').read_text())['passed']
    cfg=json.loads((root/'config.json').read_text());out=root/'figures';out.mkdir(exist_ok=True)
    seed=cfg['seed'];base=Path(cfg['base_run'])/'100'/str(seed)/'global_point'
    sources={'Shared point':base,'Smooth shared':root/'global_0.01','Smooth local':root/'local_0.01'}
    data={k:dict(np.load(v/'validation_predictions.npz',allow_pickle=False)) for k,v in sources.items()}
    colors=['#7A7A7A','#2673A6','#B37730'];manifest=[]
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':9,
        'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    skill=Path('C:/Users/Lenovo/.codex/skills/nature-figure/scripts')
    for key in cfg['validation']:
        c=data['Shared point'][key+'__cycle'];y=data['Shared point'][key+'__truth']
        fig=plt.figure(figsize=(8,5));ax=fig.add_axes([.12,.17,.85,.60])
        ax.plot(c,y,color='#222222',lw=1.8,label='Observed SOH');scores=[]
        for (name,a),color in zip(data.items(),colors):
            np.testing.assert_allclose(c,a[key+'__cycle']);np.testing.assert_allclose(y,a[key+'__truth'])
            p=a[key+'__prediction'];ax.plot(c,p,color=color,lw=1.4,label=name)
            scores.append(f'{100*np.mean(abs(p-y)/y):.2f}%')
        ax.set(xlabel='Future source cycle | input: first 100 cycles',ylabel='SOH (capacity / rated capacity)')
        ax.grid(alpha=.15);ax.margins(x=.015,y=.08)
        fig.text(.12,.96,'XJTU | '+key,fontsize=12,va='top')
        fig.text(.12,.90,'Development validation | one fold, one seed | fixed smooth-rate candidates',fontsize=8,va='top')
        fig.legend(*ax.get_legend_handles_labels(),loc='upper left',bbox_to_anchor=(.105,.865),ncol=4,fontsize=8,frameon=False)
        fig.text(.12,.035,'MAPE (point / smooth shared / smooth local): '+' / '.join(scores),fontsize=8)
        stem=out/key
        fig.savefig(str(stem)+'.png',dpi=300)
        fig.savefig(str(stem)+'.pdf')
        fig.savefig(str(stem)+'.svg')
        plt.close(fig)
        subprocess.run([sys.executable,str(skill/'audit_pdf_text.py'),str(stem)+'.pdf','--min-pt','5'],check=True,stdout=subprocess.DEVNULL)
        subprocess.run([sys.executable,str(skill/'audit_figure_collisions.py'),str(stem)+'.pdf','--json-out',str(stem)+'.collision.json'],check=True,stdout=subprocess.DEVNULL)
        manifest.append(dict(cell_id=key,png=str(stem.resolve())+'.png',automated_pdf_qa=True,individual_visual_review=False))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('FIGURES',len(manifest))


if __name__=='__main__':main()
