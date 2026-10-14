"""Validation-only exact error accounting. Never produces corrected forecasts."""
import argparse,json
from pathlib import Path
import numpy as np
from scripts.prepare_same_type_forecast import sha,save_json

def decompose(cycle,truth,prediction):
    c,y,p=[np.asarray(v,dtype=float) for v in (cycle,truth,prediction)]
    if c.ndim!=1 or len(c)<2 or c.shape!=y.shape or c.shape!=p.shape or not np.isfinite(np.r_[c,y,p]).all() or not np.all(np.diff(c)>0):raise ValueError('invalid curve')
    # pp, not relative percent: residual = first-point residual + change residual.
    error=100*(p-y);offset=float(error[0]);shape=error-offset
    np.testing.assert_allclose(error,offset+shape,atol=1e-12,rtol=0)
    late=c>=(c[0]+c[-1])/2
    midpoint=(c[0]+c[-1])/2
    ym=float(np.interp(midpoint,c,y));pm=float(np.interp(midpoint,c,p))
    true_acceleration=100*(2*ym-y[0]-y[-1]);pred_acceleration=100*(2*pm-p[0]-p[-1])
    result=dict(start_signed_pp=offset,start_abs_pp=abs(offset),endpoint_signed_pp=float(error[-1]),endpoint_abs_pp=float(abs(error[-1])),change_signed_pp=float(shape[-1]),change_abs_pp=float(abs(shape[-1])),start_dominates_endpoint=bool(abs(offset)>abs(shape[-1])),endpoint_cancellation=bool(offset*shape[-1]<0),late_mean_signed_pp=float(error[late].mean()),late_shape_mean_signed_pp=float(shape[late].mean()),mse_pp2=float(np.mean(error**2)),offset_square_pp2=offset**2,shape_square_pp2=float(np.mean(shape**2)),cross_term_pp2=float(2*offset*np.mean(shape)))
    np.testing.assert_allclose(result['mse_pp2'],result['offset_square_pp2']+result['shape_square_pp2']+result['cross_term_pp2'],rtol=1e-12,atol=1e-12)
    result.update(midpoint_signed_pp=100*(pm-ym),true_secondhalf_extra_drop_pp=float(true_acceleration),pred_secondhalf_extra_drop_pp=float(pred_acceleration),acceleration_signed_error_pp=float(pred_acceleration-true_acceleration),acceleration_abs_error_pp=float(abs(pred_acceleration-true_acceleration)),underaccelerates=bool(pred_acceleration<true_acceleration))
    return result

def summarize(rows):
    # First average repeated seeds for each (fold, cell), then give cells equal weight.
    groups={}
    for r in rows:groups.setdefault((r['fold'],r['cell_id']),[]).append(r)
    fields=list(decompose([101,102],[1,.9],[1,.9]))
    cell_rows=[dict(fold=f,cell_id=k,**{field:float(np.mean([r[field] for r in records])) for field in fields}) for (f,k),records in groups.items()]
    return dict(cell_count=len(cell_rows),cell_seed_count=len(rows),**{f:float(np.mean([r[f] for r in cell_rows])) for f in fields}),cell_rows

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);root=parser.parse_args().run
    cfg=json.loads((root/'config.json').read_text());locks=json.loads((root/'inner_lock.json').read_text())['selections']
    assert sha(root/'inner_lock.json')==json.loads((root/'progress.json').read_text())['inner_lock_sha256']
    assert json.loads((root/'independent_verification.json').read_text())['passed']
    all_rows={name:[] for name in ('primary','rate','selected_blend')};evidence=[]
    for plan in cfg['plans']:
        assert not set(plan['train']+plan['validation'])&set(plan['test'])
        for seed in cfg['seeds']:
            folder=root/plan['name']/str(seed)/'inner';lock=next(r for r in locks if r['fold']==plan['name'] and r['seed']==seed)
            assert sha(folder/'selection.json')==lock['selection_sha256']
            components=folder/'validation_components.npz';selected=folder/f"candidate_{lock['selected']['candidate']}"/'validation_predictions.npz'
            before={str(f):sha(f) for f in (components,selected)}
            with np.load(components) as a,np.load(selected) as b:
                assert set(a.files)=={k+'__100__'+f for k in plan['validation'] for f in ('cycle','truth','primary','rate')}
                assert set(b.files)=={k+'__100__'+f for k in plan['validation'] for f in ('cycle','truth','prediction')}
                for key in plan['validation']:
                    prefix=key+'__100__';c,y,p,q=[a[prefix+f] for f in ('cycle','truth','primary','rate')]
                    np.testing.assert_array_equal(c,b[prefix+'cycle']);np.testing.assert_array_equal(y,b[prefix+'truth'])
                    w=lock['selected']['weight'];mix=b[prefix+'prediction'];np.testing.assert_allclose(mix,(1-w)*p+w*q,atol=1e-12,rtol=0)
                    for name,v in [('primary',p),('rate',q),('selected_blend',mix)]:all_rows[name].append(dict(fold=plan['name'],seed=seed,cell_id=key,**decompose(c,y,v)))
            for f,h in before.items():assert sha(f)==h
            evidence.append(before)
    summaries={};by_cell={}
    for name,rows in all_rows.items():summaries[name],by_cell[name]=summarize(rows)
    save_json(root/'validation_error_decomposition.json',dict(summary=summaries,by_cell=by_cell,rows=all_rows,evidence=evidence,source_sha256=sha(__file__),outer_loaded=False,used_for_selection=False,goal_complete=False,interpretation='Exact residual accounting at first available future point, not physical latent anchor attribution. Offset/shape terms can cancel; squared terms are not independent causal contributions. No corrected forecast or oracle accuracy metric produced.'))
    print(json.dumps(summaries),flush=True)

if __name__=='__main__':main()
