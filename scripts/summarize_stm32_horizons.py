"""Fixed requested horizons from saved outer-test predictions, no retraining."""
import argparse
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import numpy as np
from battery_smetimes.research_baselines import regression_metrics
from scripts.prepare_same_type_forecast import save_json,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--tree',type=Path,required=True);p.add_argument('--mlp',type=Path,required=True);args=p.parse_args()
    out=Path('results/stm32_fixed_horizons')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');out.mkdir(parents=True)
    shutil.copy2(__file__,out/'source.py');rows=[];sources={}
    for model,run in [('compact_tree',args.tree),('tiny_mlp',args.mlp)]:
        if not json.loads((run/'progress.json').read_text())['complete']:raise ValueError('training incomplete')
        sources[str(run/'summary.csv')]=sha(run/'summary.csv')
        with (run/'summary.csv').open(newline='',encoding='utf-8') as f:base=list(csv.DictReader(f))
        cache={}
        for r in base:
            pred=Path(r['metric_source']).parent/'predictions.npz'
            if pred not in cache:
                with np.load(pred,allow_pickle=False) as z:cache[pred]={k:z[k] for k in z.files}
                sources[str(pred)]=sha(pred)
            arrays=cache[pred];prefix=r['cell_id']+'__100__'
            cycles,truth,forecast=[arrays[prefix+k] for k in ('cycle','truth','prediction')]
            for horizon in (50,100,200):
                take=(cycles>100)&(cycles<=100+horizon)
                status='passed' if cycles[-1]>=100+horizon and take.any() else ('record_ends_before_horizon' if cycles[-1]<100+horizon else 'no_valid_labels_in_window')
                metrics=regression_metrics(truth[take],forecast[take]) if status=='passed' else {}
                point=np.flatnonzero(cycles==100+horizon)
                rows.append(dict(model=model,mode=r.get('mode','float'),dataset=r['dataset'],fold=r['fold'],seed=int(r['seed']),cell_id=r['cell_id'],
                    horizon=horizon,status=status,observed_points=int(take.sum()),**{k:metrics.get(k) for k in ('mape_percent','mae_soh','rmse_soh','r2')},
                    exact_endpoint_available=bool(len(point)),endpoint_error_pp=float(abs(truth[point[0]]-forecast[point[0]])*100) if len(point) else None))
    with (out/'by_cell.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=[]
    for model,mode,domain,h in sorted({(r['model'],r['mode'],r['dataset'],r['horizon']) for r in rows}):
        group=[r for r in rows if (r['model'],r['mode'],r['dataset'],r['horizon'])==(model,mode,domain,h)];valid=[r for r in group if r['status']=='passed']
        seeds=sorted({r['seed'] for r in group});numbers=[np.mean([r['mape_percent'] for r in valid if r['seed']==s]) for s in seeds] if valid else []
        summary.append(dict(model=model,mode=mode,dataset=domain,horizon=h,eligible_cells=len({r['cell_id'] for r in valid}),total_cells=len({r['cell_id'] for r in group}),
            mape_mean=float(np.mean(numbers)) if numbers else None,mape_seed_sd=float(np.std(numbers,ddof=1)) if len(numbers)>1 else None,
            exact_endpoint_cells=len({r['cell_id'] for r in group if r['exact_endpoint_available']})))
    save_json(out/'summary.json',dict(summary=summary,source_sha256=sources,source_code_sha256=sha(Path(__file__)),
        definition='N=100; observed-label window (100,100+k]; require observed record extends through endpoint and at least one valid window label. Sparse diagnostic sampling remains sparse; exact endpoint coverage reported separately.',
        limitations='Different horizons can have different eligible cells; shorter-window errors are not comparable to whole-future MAPE or an improvement in model quality. No interpolation or extrapolation of ground truth. Five-seed SD is not a generalization CI.'))
    print('OUTPUT='+str(out.resolve()))


if __name__=='__main__':main()
