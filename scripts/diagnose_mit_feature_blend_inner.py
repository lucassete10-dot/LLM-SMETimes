"""Frozen training-side validation curves only; no selection or new fitting."""
import json, shutil
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from scripts.diagnose_xjtu_segment_inner_shape import decomposition
from scripts.prepare_same_type_forecast import sha, save_json

PARENT = Path('experiments/mit_feature_blend_full/20261006T143017_054601Z')


def summarize(rows):
    if not rows:
        return {'count': 0}
    result = {'count': len(rows)}
    for key in ('mape_percent', 'drop_abs_pp', 'underdecline',
                'severe_underdecline', 'late_underdecline',
                'query_end_beyond_training_max'):
        result[key] = float(np.mean([r[key] for r in rows]))
    result['mean_start_abs_pp'] = float(np.mean([abs(r['start_signed_pp']) for r in rows]))
    result['mean_endpoint_abs_pp'] = float(np.mean([abs(r['endpoint_signed_pp']) for r in rows]))
    result['mean_change_signed_pp'] = float(np.mean([r['change_signed_pp'] for r in rows]))
    result['drop_dominates_start_fraction'] = float(np.mean([
        r['drop_abs_pp'] > abs(r['start_signed_pp']) for r in rows]))
    return result


def main():
    cfg = json.loads((PARENT/'config.json').read_text())
    proof = json.loads((PARENT/'independent_verification.json').read_text())
    assert proof['passed'] and not proof['outer_loaded']
    root = Path('experiments/mit_inner_shape_diagnostic')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    root.mkdir(parents=True)
    print('RUN='+str(root.resolve()), flush=True)
    snap = root/'source_snapshot'; snap.mkdir()
    files = [Path(__file__), Path('scripts/diagnose_xjtu_segment_inner_shape.py')]
    for f in files:
        shutil.copy2(f, snap/f.name)
    rows, dependencies = [], {}
    for plan in cfg['plans']:
        for seed in cfg['seeds']:
            child = PARENT/plan['name']/str(seed)
            cc = json.loads((child/'config.json').read_text())
            assert cc['plan'] == plan and not cc['outer_loaded']
            assert not set(plan['test']) & set(plan['train']+plan['validation'])
            sf = child/'selection.json'
            sel = json.loads(sf.read_text()); dependencies[str(sf)] = sha(sf)
            for arm in ('selected', 'control'):
                index = sel[arm]['candidate']
                train_file = child/f'candidate_{index}'/'training_predictions.npz'
                validation_file = child/f'candidate_{index}'/'validation_predictions.npz'
                dependencies[str(train_file)] = sha(train_file)
                dependencies[str(validation_file)] = sha(validation_file)
                with np.load(train_file, allow_pickle=False) as a:
                    ends = np.array([a[k+'__100__cycle'][-1] for k in plan['train']])
                with np.load(validation_file, allow_pickle=False) as a:
                    assert set(a.files) == {k+'__100__'+f for k in plan['validation'] for f in ('cycle', 'truth', 'prediction')}
                    for key in plan['validation']:
                        c, y, p = [a[key+'__100__'+f] for f in ('cycle', 'truth', 'prediction')]
                        rows.append(dict(fold=plan['name'], seed=seed, cell_id=key,
                            arm=arm, candidate=index, new_weight=sel[arm]['new_weight'],
                            **decomposition(c,y,p),
                            query_end_beyond_training_max=bool(c[-1] > ends.max()),
                            training_max_cycle=float(ends.max()),
                            support_interpretation='whole_training_family_context_not_actual_local_route'))
    result = dict(arms={a:summarize([r for r in rows if r['arm']==a]) for a in ('selected','control')},
        by_fold={p['name']:summarize([r for r in rows if r['arm']=='selected' and r['fold']==p['name']]) for p in cfg['plans']},
        unique_validation_cells=len({r['cell_id'] for r in rows}),
        overlapping_records=True, outer_predictions_read=False, new_fits=0,
        interpretation='Descriptive inner diagnosis, not independent test accuracy or causal proof.',
        goal_complete=False)
    save_json(root/'config.json', dict(parent=str(PARENT.resolve()),
        parent_config_sha256=sha(PARENT/'config.json'), parent_proof_sha256=sha(PARENT/'independent_verification.json'),
        source_sha256={str(f):sha(f) for f in files}, input_sha256=dependencies,
        outer_predictions_read=False, new_fits=0))
    save_json(root/'rows.json', rows); save_json(root/'summary.json', result)
    print(json.dumps(result), flush=True)


if __name__=='__main__':
    main()
