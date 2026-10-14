"""Validation-only representation diagnostic; no fitted or oracle forecast."""
import argparse,json
from pathlib import Path
import numpy as np
from scripts.prepare_same_type_forecast import save_json,sha


def span(cycle,truth,primary,rate):
    low=np.minimum(primary,rate);high=np.maximum(primary,rate)
    outside=(truth<low)|(truth>high);late=cycle>=(cycle[0]+cycle[-1])/2
    distance=np.maximum(low-truth,0)+np.maximum(truth-high,0)
    residuals=np.stack([primary-truth,rate-truth])
    correlation=float(np.corrcoef(residuals)[0,1]) if np.all(np.std(residuals,axis=1)>1e-12) else None
    return dict(outside_fraction=float(np.mean(outside)),late_outside_fraction=float(np.mean(outside[late])),endpoint_outside=bool(outside[-1]),endpoint_same_side='both_high' if truth[-1]<low[-1] else 'both_low' if truth[-1]>high[-1] else 'bracketed',endpoint_distance_to_span_pp=float(100*distance[-1]),mean_distance_to_span_pp=float(100*np.mean(distance)),component_residual_correlation=correlation)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);root=parser.parse_args().run;cfg=json.loads((root/'config.json').read_text());locks=json.loads((root/'inner_lock.json').read_text())['selections'];rows=[];evidence=[]
    assert len(locks)==25 and sha(root/'inner_lock.json')==json.loads((root/'progress.json').read_text())['inner_lock_sha256']
    for plan in cfg['plans']:
        assert not set(plan['train']+plan['validation'])&set(plan['test'])
        for seed in cfg['seeds']:
            folder=root/plan['name']/str(seed)/'inner';lock=next(r for r in locks if r['fold']==plan['name'] and r['seed']==seed);assert sha(folder/'selection.json')==lock['selection_sha256'];assert sha(folder/'components.joblib')==lock['components_sha256']
            file=folder/'validation_components.npz'
            with np.load(file) as arrays:
                expected={k+'__100__'+f for k in plan['validation'] for f in ('cycle','truth','primary','rate')};assert set(arrays.files)==expected
                for key in plan['validation']:
                    cycle,y,p,q=[arrays[key+'__100__'+f] for f in ('cycle','truth','primary','rate')]
                    assert len(cycle)==len(y)==len(p)==len(q) and np.isfinite(np.r_[cycle,y,p,q]).all() and np.all(np.diff(cycle)>0)
                    rows.append(dict(fold=plan['name'],seed=seed,cell_id=key,**span(cycle,y,p,q)))
            evidence.append(dict(file=str(file.resolve()),sha256=sha(file)))
    fields=('outside_fraction','late_outside_fraction','endpoint_distance_to_span_pp','mean_distance_to_span_pp')
    mean={f:float(np.mean([r[f] for r in rows])) for f in fields};mean.update(cell_seed_rows=len(rows),endpoint_outside_fraction=float(np.mean([r['endpoint_outside'] for r in rows])),endpoint_same_side_counts={side:sum(r['endpoint_same_side']==side for r in rows) for side in ('both_high','both_low','bracketed')})
    save_json(root/'validation_component_span_diagnostic.json',dict(summary=mean,rows=rows,evidence=evidence,source_sha256=sha(__file__),outer_loaded=False,used_for_selection=False,interpretation='Distances to two-component convex span, not forecast accuracy; validation truth used only for diagnosis. No hypothetical predictions or query-specific weights saved.',goal_complete=False))
    print(json.dumps(mean),flush=True)


if __name__=='__main__':main()
