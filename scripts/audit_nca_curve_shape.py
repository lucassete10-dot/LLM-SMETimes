"""Descriptive post-evaluation curve diagnostics; never select or refit models."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import joblib
import numpy as np
from scripts.prepare_same_type_forecast import save_json, sha
from scripts.train_zhu2022_nca import matrix

RUN=Path('experiments/zhu2022_nca_forecast/20261004T161453_449950Z')


def shape(x,y,p):
    """Signed observed/predicted changes in raw SOH percentage points."""
    late=x>=(x[0]+x[-1])/2
    dy=float((y[-1]-y[late][0])*100)
    dp=float((p[-1]-p[late][0])*100)
    return dict(observed_full_change_pp=float((y[-1]-y[0])*100),
        predicted_full_change_pp=float((p[-1]-p[0])*100),
        observed_late_change_pp=dy,predicted_late_change_pp=dp,
        late_change_ratio=dp/dy if abs(dy)>=.5 else None,
        mean_signed_bias_pp=float(np.mean(p-y)*100),
        late_mean_signed_bias_pp=float(np.mean(p[late]-y[late])*100),
        endpoint_signed_bias_pp=float((p[-1]-y[-1])*100),
        opposite_late_direction=bool(dy*dp<0) if abs(dy)>=.5 else None)


def main():
    cfg=json.loads((RUN/'config.json').read_text());prep=Path(cfg['prepared_directory'])
    assert sha(prep/'prepared_cells.joblib')==cfg['prepared_sha256']
    cells=joblib.load(prep/'prepared_cells.joblib');rows=[]
    out=Path('experiments/nca_shape_audit')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True);shutil.copy2(__file__,out/Path(__file__).name)
    for model in cfg['candidates']:
        for plan in cfg['plans']:
            dev=plan['train']+plan['validation'];trainx=[]
            for key in dev:
                c=cells[key];take=np.unique(np.linspace(0,len(c['truth'])-1,min(96,len(c['truth']))).round().astype(int))
                trainx.append(matrix(c,c['cycle'][take]))
            raw=np.concatenate(trainx)
            train_last=max(float(cells[k]['cycle'][-1]) for k in dev)
            for seed in cfg['seeds']:
                folder=RUN/model/plan['name']/str(seed);bundle=joblib.load(folder/'model.joblib')
                assert set(bundle['train_ids'])==set(dev) and not set(dev)&set(plan['test'])
                fitted=bundle['imputer'].transform(raw);lo,hi=fitted.min(axis=0),fitted.max(axis=0)
                with np.load(folder/'predictions.npz') as arrays:
                    for key in plan['test']:
                        stem=key+'__100__';x,y,p=[arrays[stem+name] for name in ('cycle','truth','prediction')]
                        query=bundle['imputer'].transform(matrix(cells[key],x))
                        tol=1e-9*np.maximum(1,np.maximum(abs(lo),abs(hi)))
                        outside=(query<lo-tol)|(query>hi+tol)
                        prefix_len=len(cells[key]['prefix'])
                        rows.append(dict(cell_id=key,model=model,fold=plan['name'],seed=seed,
                            future_targets=len(x),future_first=int(x[0]),future_last=int(x[-1]),
                            training_last_source_cycle=train_last,
                            future_targets_beyond_training_horizon=int((x>train_last).sum()),
                            prefix_columns_outside_training_range=int(outside[0,:prefix_len].sum()),
                            query_rows_outside_any_training_range_fraction=float(outside.any(axis=1).mean()),
                            **shape(x,y,p)))
    summaries={}
    for model in cfg['candidates']:
        mr=[r for r in rows if r['model']==model];per_cell=[]
        for key in sorted(cells):
            cr=[r for r in mr if r['cell_id']==key]
            item=dict(cell_id=key)
            for name in cr[0]:
                values=[r[name] for r in cr]
                if name in ('cell_id','model','fold','seed'):continue
                if values[0] is None:item[name]=None
                else:item[name]=float(np.mean(values))
            per_cell.append(item)
        summaries[model]=dict(cells=55,by_cell=per_cell,
            cells_with_any_prefix_column_outside_training_range=sum(r['prefix_columns_outside_training_range']>0 for r in per_cell),
            cells_with_future_beyond_training_horizon=sum(r['future_targets_beyond_training_horizon']>0 for r in per_cell),
            cells_with_opposite_late_direction=sum(r['opposite_late_direction'] is not None and r['opposite_late_direction']>0 for r in per_cell),
            mean_endpoint_signed_bias_pp=float(np.mean([r['endpoint_signed_bias_pp'] for r in per_cell])))
    save_json(out/'by_cell_seed.json',rows);save_json(out/'summary.json',summaries)
    save_json(out/'config.json',dict(source_sha256=sha(__file__),run=str(RUN.resolve()),
        prepared_sha256=cfg['prepared_sha256'],role='descriptive post-evaluation diagnosis, no model selection',
        tiny_late_change_rule='ratios undefined if observed late change absolute<0.5 SOHpp',
        outside_range_tolerance='1e-9 * max(1,abs(training_min),abs(training_max))',
        caution='Coordinate min/max outside-range is not physical abnormality, causal proof, or a reason to exclude a cell.',
        unchanged_predictions=True,all_cells_retained=True))
    print('OUTPUT='+str(out.resolve()),flush=True)
    for model,s in summaries.items():
        print(model,{k:v for k,v in s.items() if k!='by_cell'},flush=True)


if __name__=='__main__':main()
