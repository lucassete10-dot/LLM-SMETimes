"""Read frozen INNER curves only; separate offset, decline, and horizon support."""
import json, shutil
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from battery_smetimes.curve_fidelity import condition_key, targets
from scripts.train_curve_fidelity import load_subset
from scripts.prepare_same_type_forecast import sha, save_json

PARENT = Path('experiments/xjtu_segment_full/20261006T132530_948247Z')


def decomposition(cycle, truth, prediction):
    c, y, p = [np.asarray(a, dtype=float) for a in (cycle, truth, prediction)]
    if c.ndim != 1 or len(c) < 2 or c.shape != y.shape or c.shape != p.shape:
        raise ValueError('invalid shape')
    if not all(np.isfinite(a).all() for a in (c, y, p)) or np.any(np.diff(c) <= 0):
        raise ValueError('invalid values/order')
    start = 100 * (p[0]-y[0])
    true_drop = 100 * (y[0]-y[-1])
    pred_drop = 100 * (p[0]-p[-1])
    end = 100 * (p[-1]-y[-1])
    change = true_drop-pred_drop
    np.testing.assert_allclose(end, start+change, atol=1e-12, rtol=0)
    mask = c >= c[0]+.75*(c[-1]-c[0])
    if mask.sum() < 2:
        mask[-2:] = True
    t = c[mask]; t = t-t.mean()
    late_true = -float(t@y[mask]/(t@t))*10000
    late_pred = -float(t@p[mask]/(t@t))*10000
    return dict(start_signed_pp=float(start), endpoint_signed_pp=float(end),
        true_drop_pp=float(true_drop), predicted_drop_pp=float(pred_drop),
        change_signed_pp=float(change), drop_abs_pp=float(abs(change)),
        underdecline=bool(true_drop >= 5 and pred_drop < true_drop),
        severe_underdecline=bool(true_drop >= 5 and pred_drop < .3*true_drop),
        late_true_decline_pp_per_100_cycles=late_true,
        late_pred_decline_pp_per_100_cycles=late_pred,
        late_underdecline=bool(late_true > 0 and late_pred < .5*late_true),
        mape_percent=float(np.mean(abs((p-y)/y))*100))


def summary(rows):
    if not rows:
        return dict(count=0)
    return dict(count=len(rows), **{k: float(np.mean([r[k] for r in rows]))
        for k in ('mape_percent', 'drop_abs_pp', 'underdecline',
                  'severe_underdecline', 'late_underdecline',
                  'query_end_beyond_training_max', 'support_at_query_end_fraction')},
        mean_start_abs_pp=float(np.mean([abs(r['start_signed_pp']) for r in rows])),
        mean_endpoint_abs_pp=float(np.mean([abs(r['endpoint_signed_pp']) for r in rows])))


def main():
    cfg=json.loads((PARENT/'config.json').read_text())
    proof=json.loads((PARENT/'independent_verification.json').read_text())
    assert proof['passed'] and proof['verified_tasks']==25 and not proof['outer_loaded']
    root=Path('experiments/xjtu_inner_shape_diagnostic')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    root.mkdir(parents=True); print('RUN='+str(root.resolve()), flush=True)
    snap=root/'source_snapshot'; snap.mkdir()
    files=[Path(__file__), Path('battery_smetimes/curve_fidelity.py'), Path('scripts/train_curve_fidelity.py')]
    for f in files: shutil.copy2(f,snap/f.name)
    rows=[]; dependencies={}; prepared={}
    for task in cfg['tasks']:
        child=Path(task['run']); cc=json.loads((child/'config.json').read_text())
        sel=json.loads((child/'selection.json').read_text()); plan=cc['plan']
        cells=load_subset(plan['train']+plan['validation'])
        assert not set(cells)&set(plan['test'])
        prepared.update({k:v['prepared_sha256'] for k,v in cells.items()})
        groups={}
        for k in plan['train']: groups.setdefault(condition_key(cells[k]),[]).append(k)
        for arm in ('selected','control'):
            ci=sel[arm]['candidate']; file=child/f'candidate_{ci}'/'validation_predictions.npz'
            dependencies[str(file)]=sha(file); dependencies[str(child/'selection.json')]=sha(child/'selection.json')
            with np.load(file,allow_pickle=False) as arrays:
                for key in plan['validation']:
                    stem=key+'__100__'; c,y,p=[arrays[stem+f] for f in ('cycle','truth','prediction')]
                    np.testing.assert_array_equal(c, targets(cells[key])[0])
                    np.testing.assert_array_equal(y, targets(cells[key])[1])
                    local=groups.get(condition_key(cells[key]),[])
                    # New segment arm uses this known-protocol route. Legacy control
                    # may have another structure, so no unsupported support claim.
                    support_ids=local if ci>0 and len(local)>=6 else plan['train']
                    ends=np.array([targets(cells[k])[0][-1] for k in support_ids])
                    rows.append(dict(fold=task['fold'],seed=task['seed'],cell_id=key,
                        arm=arm,candidate=ci, **decomposition(c,y,p),
                        routing='legacy_unspecified' if ci==0 else 'local' if len(local)>=6 else 'family',
                        support_interpretation='family_context_only' if ci==0 else 'actual_route',
                        support_ids=support_ids, training_max_cycle=float(ends.max()),
                        query_end_cycle=float(c[-1]), query_end_beyond_training_max=bool(c[-1]>ends.max()),
                        support_at_query_end_fraction=float(np.mean(ends>=c[-1]))))
    save_json(root/'config.json',dict(parent=str(PARENT.resolve()),parent_config_sha256=sha(PARENT/'config.json'),
        parent_proof_sha256=sha(PARENT/'independent_verification.json'),source_sha256={str(f):sha(f) for f in files},
        input_sha256=dependencies,prepared_sha256=prepared,outer_predictions_read=False,new_fits=0,
        descriptive_only=True,overlapping_validation_membership=True,goal_complete=False))
    save_json(root/'rows.json',rows)
    result=dict(arms={a:summary([r for r in rows if r['arm']==a]) for a in ('selected','control')},
        selected_by_route={a:summary([r for r in rows if r['arm']=='selected' and r['routing']==a])
            for a in ('local','family','legacy_unspecified')},
        selected_by_end_support={a:summary([r for r in rows if r['arm']=='selected' and
            ('beyond' if r['query_end_beyond_training_max'] else 'within')==a]) for a in ('beyond','within')},
        interpretation='Training-side descriptive diagnosis; not independent test accuracy or causal attribution.',
        goal_complete=False)
    save_json(root/'summary.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__': main()
