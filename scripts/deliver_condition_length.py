"""Export all paired development curves and machine-readable measurements."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('run',type=Path); args=ap.parse_args(); root=args.run
    proof=json.loads((root/'independent_replay_verification.json').read_text()); assert proof['passed']
    cfg=json.loads((root/'config.json').read_text()); rows=pd.DataFrame(json.loads((root/'all_rows.json').read_text()))
    out=Path('deliveries')/('XJTU_condition_'+root.name); out.mkdir(parents=True,exist_ok=True)
    rows['mae_pp']=100*rows.mae_soh
    rows.drop(columns=['used_train','validation']).to_csv(out/'per_cell_seed_metrics.csv',index=False,encoding='utf-8-sig')
    rows.groupby(['origin','candidate','seed'])[['mape_percent','mae_pp','late20_mae_pp','screen_pass']].mean().to_csv(out/'per_seed_summary.csv',encoding='utf-8-sig')
    names={'global_point':'Shared point model','local_point':'Protocol-local point','local_rate':'Protocol-local rate'}
    colors={'global_point':'#7A7A7A','local_point':'#2673A6','local_rate':'#B37730'}
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':9,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    skill=Path('C:/Users/Lenovo/.codex/skills/nature-figure/scripts'); manifest=[]; allpoints=[]
    for origin in cfg['origins']:
        folder=out/'XJTU'/f'input_{origin:03d}'; folder.mkdir(parents=True,exist_ok=True)
        arrays={name:[dict(np.load(root/str(origin)/str(seed)/name/'validation_predictions.npz',allow_pickle=False)) for seed in cfg['seeds']] for name in names}
        ids=rows.loc[rows.origin==origin,'cell_id'].unique()
        for key in ids:
            first=arrays['global_point'][0]; c=first[key+'__cycle']; y=first[key+'__truth']
            fig=plt.figure(figsize=(8,5)); ax=fig.add_axes([.12,.17,.85,.59])
            ax.plot(c,y,color='#222222',lw=1.8,label='Observed SOH'); scores=[]
            for name in names:
                stack=np.stack([a[key+'__prediction'] for a in arrays[name]])
                for seed,p in zip(cfg['seeds'],stack):
                    allpoints.extend(dict(cell_id=key,origin=origin,candidate=name,seed=seed,cycle=float(x),truth=float(t),prediction=float(v)) for x,t,v in zip(c,y,p))
                mean=stack.mean(0); sd=stack.std(0,ddof=1)
                ax.plot(c,mean,color=colors[name],lw=1.4,label=names[name])
                ax.fill_between(c,mean-sd,mean+sd,color=colors[name],alpha=.10,linewidth=0)
                scores.append(f'{100*np.mean(abs(mean-y)/y):.2f}%')
            ax.set(xlabel=f'Future source cycle | input: first {origin} cycles',ylabel='SOH (capacity / rated capacity)')
            ax.grid(alpha=.15); ax.margins(x=.015,y=.08)
            fig.text(.12,.96,'XJTU | '+key,fontsize=12,va='top')
            fig.text(.12,.90,'Development validation: one fold, five seeds | mean +/- 1 seed SD; not a CI',fontsize=8,va='top')
            fig.legend(*ax.get_legend_handles_labels(),loc='upper left',bbox_to_anchor=(.105,.865),ncol=2,fontsize=8,frameon=False)
            fig.text(.12,.035,'Mean-curve MAPE (shared / local point / local rate): '+' / '.join(scores),fontsize=8)
            stem=folder/key
            fig.savefig(str(stem)+'.png',dpi=300)
            fig.savefig(str(stem)+'.pdf')
            fig.savefig(str(stem)+'.svg')
            plt.close(fig)
            subprocess.run([sys.executable,str(skill/'audit_pdf_text.py'),str(stem)+'.pdf','--min-pt','5'],check=True,stdout=subprocess.DEVNULL)
            subprocess.run([sys.executable,str(skill/'audit_figure_collisions.py'),str(stem)+'.pdf','--json-out',str(stem)+'.collision.json'],check=True,stdout=subprocess.DEVNULL)
            manifest.append(dict(cell_id=key,origin=origin,png=str(stem.resolve())+'.png',pdf=str(stem.resolve())+'.pdf',automated_pdf_qa=True,individual_visual_review=False))
    pd.DataFrame(allpoints).to_csv(out/'point_predictions.csv',index=False,encoding='utf-8-sig')
    for name in ('config.json','independent_replay_verification.json','train_coverage.json','validation_coverage.json'):
        shutil.copy2(root/name,out/name)
    (out/'manifest.json').write_text(json.dumps(dict(source_run=str(root.resolve()),figures=manifest,planned_figures=len(manifest),delivered_figures=len(manifest),
        independent_test=False,workbook_pending=True,goal_complete=False),indent=2),encoding='utf-8')
    print(str(out.resolve()),flush=True)


if __name__=='__main__':main()
