"""Training-side support and observed health distribution; no new fitting."""
import json,shutil
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import joblib
from battery_smetimes.curve_fidelity import targets
from battery_smetimes.mit_segment_features import condition_key
from scripts.train_curve_fidelity import load_subset
from scripts.prepare_same_type_forecast import sha,save_json

PARENT=Path('experiments/mit_feature_blend_full/20261006T143017_054601Z')


def support(cells,ids,horizon):
    ends=[];observed=[]
    for key in ids:
        c,y=targets(cells[key]);ends.append(c[-1])
        # Select real labels only, no extrapolation or last-label carry forward.
        if c[-1]>=horizon:
            index=int(np.argmin(abs(c-horizon)))
            if abs(c[index]-horizon)<=25:observed.append(float(y[index]))
    return dict(training_cells=len(ids),cells_observed_to_horizon=int(np.sum(np.array(ends)>=horizon)),
        observed_to_horizon_fraction=float(np.mean(np.array(ends)>=horizon)),
        actual_label_count=len(observed),actual_soh_median=float(np.median(observed)) if observed else None,
        actual_soh_mean=float(np.mean(observed)) if observed else None,
        training_max_source_cycle=float(max(ends)))


def main():
    cfg=json.loads((PARENT/'config.json').read_text());proof=json.loads((PARENT/'independent_verification.json').read_text())
    assert proof['passed'] and not proof['outer_loaded']
    root=Path('experiments/mit_training_horizon_support')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    root.mkdir(parents=True);print('RUN='+str(root.resolve()),flush=True)
    snap=root/'source_snapshot';snap.mkdir();files=[Path(__file__),Path('battery_smetimes/mit_segment_features.py')]
    for f in files:shutil.copy2(f,snap/f.name)
    records=[];grid=[];dependencies={};prepared={}
    seed=cfg['seeds'][0]
    for plan in cfg['plans']:
        child=PARENT/plan['name']/str(seed);cells=load_subset(plan['train']+plan['validation'])
        assert not set(cells)&set(plan['test'])
        prepared.update({k:c['prepared_sha256'] for k,c in cells.items()})
        sf=child/'selection.json';selection=json.loads(sf.read_text());dependencies[str(sf)]=sha(sf)
        model_file=child/'feature.joblib';bundle=joblib.load(model_file);dependencies[str(model_file)]=sha(model_file)
        assert bundle['train_ids']==plan['train']
        train=plan['train']
        for horizon in range(200,1601,200):
            grid.append(dict(fold=plan['name'],source_cycle=horizon,**support(cells,train,horizon)))
        for key in plan['validation']:
            c,y=targets(cells[key]);group=condition_key(cells[key]);local=bundle['local'].get(group)
            ids=local['train_ids'] if local else train
            horizon=float(c[-1])
            records.append(dict(fold=plan['name'],cell_id=key,selected_new_weight=selection['selected']['new_weight'],
                reference='new_component_actual_route' if selection['selected']['new_weight']>0 else 'unselected_new_component_diagnostic_only',
                route='local' if local else 'family',query_end_source_cycle=horizon,
                validation_endpoint_soh=float(y[-1]),**support(cells,ids,horizon)))
    selected=[r for r in records if r['selected_new_weight']>0]
    def aggregate(rows):
        return dict(count=len(rows),mean_observed_to_horizon_fraction=float(np.mean([r['observed_to_horizon_fraction'] for r in rows])),
            zero_support_count=sum(r['cells_observed_to_horizon']==0 for r in rows),
            at_most_two_support_count=sum(r['cells_observed_to_horizon']<=2 for r in rows))
    summary=dict(all_validation_records=aggregate(records),selected_new_component_records=aggregate(selected),
        outer_read=False,new_fits=0,one_fixed_seed=True,overlapping_validation_membership=True,
        interpretation='Training-side observed-label coverage only; no causal proof of survivorship bias and no test selection.',goal_complete=False)
    save_json(root/'config.json',dict(parent=str(PARENT.resolve()),parent_config_sha256=sha(PARENT/'config.json'),
        parent_proof_sha256=sha(PARENT/'independent_verification.json'),source_sha256={str(f):sha(f) for f in files},
        input_sha256=dependencies,prepared_sha256=prepared,seed=seed,outer_read=False,new_fits=0))
    save_json(root/'validation_support.json',records);save_json(root/'training_horizon_grid.json',grid)
    save_json(root/'summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
