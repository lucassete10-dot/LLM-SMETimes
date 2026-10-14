"""Verify instrumentation-only rerun preserves every saved candidate/forecast."""
from datetime import datetime,timezone
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from scripts.prepare_same_type_forecast import save_json,sha

OLD=Path('experiments/nasa_signal_trajectory/ridge/20261004T143636_110096Z')
NEW=Path('experiments/nasa_signal_trajectory/ridge/20261004T165029_391663Z')


def main():
    global OLD,NEW
    parser=argparse.ArgumentParser()
    parser.add_argument('--old',type=Path,default=OLD)
    parser.add_argument('--new',type=Path,default=NEW)
    parser.add_argument('--curves',type=Path,default=Path('results/nasa_heldout_curves/20261004T145215_438455Z'))
    args=parser.parse_args();OLD=args.old;NEW=args.new
    old=json.loads((OLD/'config.json').read_text());new=json.loads((NEW/'config.json').read_text())
    for key in ('plans','seeds','origin','selection'):
        assert old[key]==new[key],key
    candidates=old.get('candidates',old.get('alphas'))
    assert candidates==new.get('candidates',new.get('alphas'))
    validation_only=old.get('validation_only',False)
    assert validation_only==new.get('validation_only',False)
    assert (NEW/'training.log').stat().st_size>0
    for filename,digest in new['dependency_sha256'].items():
        assert sha(NEW/'source_snapshot'/filename)==digest
    source_matches=[p for p in NEW.glob('train_*.py') if sha(p)==new['source_sha256']]
    assert len(source_matches)==1,'saved training source hash mismatch'
    checked=[]
    for plan in old['plans']:
        for seed in old['seeds']:
            a=OLD/plan['name']/str(seed);b=NEW/plan['name']/str(seed)
            assert json.loads((a/'selection.json').read_text())==json.loads((b/'selection.json').read_text())
            for sub in ([] if validation_only else [''])+[f'candidate_{i}' for i in range(len(candidates))]:
                pa,pb=a/sub,b/sub
                filename='validation_predictions.npz' if sub else 'predictions.npz'
                with np.load(pa/filename) as left,np.load(pb/filename) as right:
                    assert set(left.files)==set(right.files)
                    for key in left.files:np.testing.assert_array_equal(left[key],right[key])
                left,right=joblib.load(pa/'model.joblib'),joblib.load(pb/'model.joblib')
                assert left['train_ids']==right['train_ids']
                assert left.get('parameters',left.get('alpha'))==right.get('parameters',right.get('alpha'))
                for obj,fields in [('model',('coef_','intercept_')),('imputer',('statistics_',)),('scaler',('mean_','var_','scale_'))]:
                    for field in fields:np.testing.assert_array_equal(getattr(left[obj],field),getattr(right[obj],field))
                checked.append(dict(fold=plan['name'],seed=seed,subdirectory=sub or 'final',
                    old_prediction_sha256=sha(pa/filename),new_prediction_sha256=sha(pb/filename)))
    out=Path('experiments/nasa_reproduction_audit')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');out.mkdir(parents=True)
    curve_dir=args.curves
    curves=[]
    for plan in [] if validation_only else old['plans']:
        for key in plan['test']:
            files={suffix:str((curve_dir/(key+suffix)).resolve()) for suffix in ('.png','.pdf','.svg')}
            curves.append(dict(cell_id=key,fold=plan['name'],seeds=new['seeds'],files=files,
                files_sha256={suffix:sha(path) for suffix,path in files.items()},
                evidence='all5 saved seed arrays exactly match the original run; previously rendered held-out curve reused unchanged'))
    save_json(NEW/'curve_reuse.json',dict(complete=True,source_run=str(OLD.resolve()),
        reused_heldout_figures=curves,validation_only=validation_only,independent_new_predictions=False,
        identical_saved_arrays_verified=True,verification_directory=str(out.resolve())))
    save_json(out/'verification.json',dict(passed=True,original=str(OLD.resolve()),repeat=str(NEW.resolve()),
        models=len(checked),units=len(old['plans'])*len(old['seeds']),all_candidate_and_final_arrays_exact=True,all_ridge_preprocessing_and_coefficients_exact=True,
        frozen_plans_candidates_seeds_unchanged=True,training_log_saved=True,dependency_snapshot_hashes_verified=True,
        validation_only=validation_only,no_outer_predictions=validation_only,
        role='instrumentation-only deterministic reproduction, not independent experimental replication',checked=checked))
    print('OUTPUT='+str(out.resolve()),flush=True)


if __name__=='__main__':main()
