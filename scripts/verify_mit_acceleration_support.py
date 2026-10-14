"""Separate slope/sampling/support arithmetic on all saved TRAIN cells."""
import json,sys
from pathlib import Path
import numpy as np
from scripts.train_curve_fidelity import load_subset
from battery_smetimes.curve_fidelity import targets
from battery_smetimes.mit_segment_features import condition_key
from scripts.prepare_same_type_forecast import sha,save_json
from scripts.verify_nasa_robust_inner import close


def main(root):
    cfg=json.loads((root/'config.json').read_text());plan=cfg['plan'];cells=load_subset(plan['train'])
    assert set(cells)==set(plan['train']) and not set(cells)&set(plan['validation']+plan['test'])
    assert sha(Path(cfg['parent'])/'config.json')==cfg['parent_config_sha256']
    assert sha(Path(cfg['parent'])/'independent_verification.json')==cfg['parent_proof_sha256']
    for f,d in cfg['source_sha256'].items():assert sha(root/'source_snapshot'/Path(f).name)==d
    for k,d in cfg['prepared_sha256'].items():assert cells[k]['prepared_sha256']==d
    records=json.loads((root/'rows.json').read_text());rows=[]
    with np.load(root/'train_labels.npz',allow_pickle=False) as arrays:
        assert set(arrays.files)=={k+'__'+f for k in plan['train'] for f in ('cycle','truth')}
        for k in plan['train']:
            c,y=targets(cells[k]);np.testing.assert_array_equal(c,arrays[k+'__cycle']);np.testing.assert_array_equal(y,arrays[k+'__truth'])
            end=float(c[-1]);late=(c>end-100)&(c<=end);prev=(c>end-200)&(c<=end-100)
            def rate(mask):
                if mask.sum()<10 or np.ptp(c[mask])<25:return None
                x=c[mask]-np.mean(c[mask]);z=y[mask]-np.mean(y[mask])
                return float(-np.dot(x,z)/np.dot(x,x)*10000)
            a,b=rate(prev),rate(late);avail=a is not None and b is not None
            accelerated=bool(avail and b>=1.5*max(a,0) and b-a>=.5 and 100*(y[0]-y[-1])>=5)
            idx=np.unique(np.linspace(0,len(c)-1,min(96,len(c))).round().astype(int))
            interp=np.interp(c,c[idx],y[idx])
            rows.append(dict(cell_id=k,condition=list(condition_key(cells[k])),source_end=end,
                future_points=len(c),future_span=float(c[-1]-c[0]),previous_points=int(prev.sum()),late_points=int(late.sum()),
                previous_fade_pp100=a,late_fade_pp100=b,acceleration_available=avail,accelerated=accelerated,
                true_drop_pp=float(100*(y[0]-y[-1])),sampled_points=len(idx),sampled_late_points=int(late[idx].sum()),
                sampled_label_reconstruction_mape_percent=float(100*np.mean(abs(interp-y)/y)),
                sampled_label_reconstruction_max_error_pp=float(100*np.max(abs(interp-y)))))
    for pos,r in enumerate(rows):
        peers=[p for p in rows if p['cell_id']!=r['cell_id'] and p['condition']==r['condition']]
        actual=peers if len(peers)>=6 else [p for p in rows if p['cell_id']!=r['cell_id']]
        r.update(support_scope='same_condition' if len(peers)>=6 else 'family_fallback',peer_count=len(actual),
            peers_reaching_end=sum(p['source_end']>=r['source_end'] for p in actual),accelerated_peers=sum(p['accelerated'] for p in actual))
        close(records[pos],r)
    available=[r for r in rows if r['acceleration_available']];accelerated=[r for r in rows if r['accelerated']]
    summary=json.loads((root/'summary.json').read_text())
    calc=dict(train_cells=len(rows),available_cells=len(available),accelerated_cells=len(accelerated),
        mean_sampling_reconstruction_mape_percent=float(np.mean([r['sampled_label_reconstruction_mape_percent'] for r in rows])),
        max_sampling_reconstruction_error_pp=float(max(r['sampled_label_reconstruction_max_error_pp'] for r in rows)),
        accelerated_mean_late_sampled_points=float(np.mean([r['sampled_late_points'] for r in accelerated])) if accelerated else None,
        accelerated_low_horizon_support=sum(r['peers_reaching_end']<=2 for r in accelerated))
    close({k:summary[k] for k in calc},calc)
    assert len(rows)==83 and cfg['new_predictive_fits']==0 and not cfg['validation_loaded'] and not cfg['outer_loaded']
    save_json(root/'independent_verification.json',dict(passed=True,train_cells=83,rows=len(rows),
        scope='Separate covariance slopes/sampling/support arithmetic; shared TRAIN loading and target helper, no forecast model.',
        validation_loaded=False,outer_loaded=False,verifier_sha256=sha(Path(__file__))))
    print('PASS: 83 TRAIN cells, slopes, sampling, support, isolation and SHA')


if __name__=='__main__':main(Path(sys.argv[1]))
