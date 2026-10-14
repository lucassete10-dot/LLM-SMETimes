"""QA-only contact sheets of delivered PNGs; never replace source figures."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=Path('deliveries/LLM_SMETimes_MIT_progress_20261009_v1')
    rows=json.loads((root/'evaluation_manifest.json').read_text())['figures']
    dest=Path('experiments/figure_qa/MIT_progress_20261009_v1')
    dest.mkdir(parents=True,exist_ok=True)
    for start in range(0,len(rows),9):
        fig,axes=plt.subplots(3,3,figsize=(24,15))
        fig.subplots_adjust(0,0,1,1,0,0)
        for ax,row in zip(axes.flat,rows[start:start+9]):
            ax.imshow(plt.imread(root/row['file']))
            ax.axis('off')
        fig.savefig(dest/f'contact_{start//9+1:02d}.png',dpi=100)
        plt.close(fig)
    print(dest.resolve())


if __name__=='__main__':main()
