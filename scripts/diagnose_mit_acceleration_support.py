"""TRAIN-only late-slope and sampling-support diagnostic; no predictor fit."""
import json, shutil
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from battery_smetimes.curve_fidelity import targets
from battery_smetimes.mit_segment_features import condition_key
from scripts.train_curve_fidelity import load_subset
from scripts.prepare_same_type_forecast import sha, save_json

PARENT=Path('experiments/mit_feature_blend_full/20261006T143017_054601Z')


def slope_pp100(c, y, mask):
    if mask.sum()<10 or np.ptp(c[mask])<25: return None
    return float(-np.polyfit(c[mask]-c[mask].mean(),y[mask],1)[0]*10000)


def describe(c,y):
    end=float(c[-1])
    late=(c>end-100)&(c<=end)
    previous=(c>end-200)&(c<=end-100)
    a=slope_pp100(c,y,previous); b=slope_pp100(c,y,late)
    available=a is not None and b is not None
    # Label-only description, never a routing feature or exclusion criterion.
    accelerated=available and b>=1.5*max(a,0) and b-a>=.5 and 100*(y[0]-y[-1])>=5
    idx=np.unique(np.rint(np.linspace(0,len(c)-1,min(96,len(c)))).astype(int))
    yi=np.interp(c,c[idx],y[idx])
    return dict(source_end=end,future_points=len(c),future_span=float(c[-1]-c[0]),
        previous_points=int(previous.sum()),late_points=int(late.sum()),
        previous_fade_pp100=a,late_fade_pp100=b,
        acceleration_available=available,accelerated=bool(accelerated),
        true_drop_pp=float(100*(y[0]-y[-1])),sampled_points=len(idx),
        sampled_late_points=int(late[idx].sum()),
        sampled_label_reconstruction_mape_percent=float(100*np.mean(abs(yi-y)/y)),
        sampled_label_reconstruction_max_error_pp=float(100*np.max(abs(yi-y))))


def main():
    cfg=json.loads((PARENT/'config.json').read_text())
    proof=json.loads((PARENT/'independent_verification.json').read_text())
    assert proof['passed'] and not proof['outer_loaded']
    plan=next(p for p in cfg['plans'] if p['name']=='MIT_fold1')
    cells=load_subset(plan['train'])
    assert set(cells)==set(plan['train']) and not set(cells)&set(plan['validation']+plan['test'])
    root=Path('experiments/mit_acceleration_support')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    root.mkdir(parents=True); print('RUN='+str(root.resolve()),flush=True)
    snap=root/'source_snapshot';snap.mkdir()
    files=[Path(__file__),Path('battery_smetimes/mit_segment_features.py'),Path('battery_smetimes/curve_fidelity.py')]
    for f in files: shutil.copy2(f,snap/f.name)
    save_json(root/'config.json',dict(parent=str(PARENT.resolve()),parent_config_sha256=sha(PARENT/'config.json'),
        parent_proof_sha256=sha(PARENT/'independent_verification.json'),plan=plan,
        source_sha256={str(f):sha(f) for f in files},prepared_sha256={k:v['prepared_sha256'] for k,v in cells.items()},
        prior_window_cycles=100,late_window_cycles=100,min_points=10,min_span=25,
        acceleration_ratio=1.5,min_acceleration_pp100=.5,min_drop_pp=5,
        validation_loaded=False,outer_loaded=False,new_predictive_fits=0))
    rows=[]; arrays={}
    for key in plan['train']:
        c,y=targets(cells[key]); d=describe(c,y)
        rows.append(dict(cell_id=key,condition=list(condition_key(cells[key])),**d))
        arrays.update({key+'__'+f:v for f,v in [('cycle',c),('truth',y)]})
    by_id={r['cell_id']:r for r in rows}
    for row in rows:
        peers=[r for r in rows if r['cell_id']!=row['cell_id'] and r['condition']==row['condition']]
        actual=peers if len(peers)>=6 else [r for r in rows if r['cell_id']!=row['cell_id']]
        row.update(support_scope='same_condition' if len(peers)>=6 else 'family_fallback',
            peer_count=len(actual),peers_reaching_end=sum(r['source_end']>=row['source_end'] for r in actual),
            accelerated_peers=sum(r['accelerated'] for r in actual))
    available=[r for r in rows if r['acceleration_available']]
    accelerated=[r for r in available if r['accelerated']]
    summary=dict(train_cells=len(rows),available_cells=len(available),accelerated_cells=len(accelerated),
        mean_sampling_reconstruction_mape_percent=float(np.mean([r['sampled_label_reconstruction_mape_percent'] for r in rows])),
        max_sampling_reconstruction_error_pp=float(max(r['sampled_label_reconstruction_max_error_pp'] for r in rows)),
        accelerated_mean_late_sampled_points=float(np.mean([r['sampled_late_points'] for r in accelerated])) if accelerated else None,
        accelerated_low_horizon_support=sum(r['peers_reaching_end']<=2 for r in accelerated),
        validation_loaded=False,outer_loaded=False,new_predictive_fits=0,goal_complete=False,
        interpretation='Own TRAIN labels describe acceleration and sampling only; not early-feature forecast accuracy, actual model route, causal proof, or cell-removal rule.')
    save_json(root/'rows.json',rows);np.savez_compressed(root/'train_labels.npz',**arrays)
    save_json(root/'summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
