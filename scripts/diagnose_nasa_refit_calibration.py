"""Training/validation-only sensitivity of inferred level to 2→3 cell refit."""
import argparse,json
from pathlib import Path
import joblib,numpy as np
from battery_smetimes.nasa_cycle_calibration import predict as calibrate
from battery_smetimes.research_baselines import predict_baseline
from scripts.train_curve_fidelity import load_subset
from scripts.train_crossfit_drift_smoke import clean
from scripts.prepare_same_type_forecast import sha,save_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);root=parser.parse_args().run;cfg=json.loads((root/'config.json').read_text());assert json.loads((root/'independent_verification.json').read_text())['passed'];screen=Path(cfg['screened']);locks=json.loads((root/'inner_lock.json').read_text())['selections'];rows=[]
    for plan in cfg['plans']:
        ids=plan['train']+plan['validation'];cells=load_subset(ids);assert not set(cells)&set(plan['test']);lock=next(r for r in locks if r['fold']==plan['name'] and r['seed']==20261005);inner_file=screen/plan['name']/f"candidate_{lock['candidate']}/model.joblib";outer_file=root/plan['name']/'20261005/model.joblib';a,b=joblib.load(inner_file),joblib.load(outer_file);assert a['train_ids']==plan['train'] and b['train_ids']==ids
        for key in ids:
            query=clean(cells[key]);oldp=float(predict_baseline(a['primary'],query,100,[101.])[0]);newp=float(predict_baseline(b['primary'],query,100,[101.])[0]);oldc=float(calibrate(a['calibration'],query,[101.])[0]) if a['calibration'] else oldp;newc=float(calibrate(b['calibration'],query,[101.])[0]) if b['calibration'] else newp
            rows.append(dict(fold=plan['name'],cell_id=key,role='original_validation' if key in plan['validation'] else 'original_train',primary_anchor_before=oldp,primary_anchor_after=newp,calibrated_anchor_before=oldc,calibrated_anchor_after=newc,primary_shift_pp=100*(newp-oldp),calibration_shift_pp=100*(newc-oldc),inner_model_sha256=sha(inner_file),refitted_model_sha256=sha(outer_file)))
    val=[r for r in rows if r['role']=='original_validation'];result=dict(rows=rows,original_validation_mean_abs_primary_shift_pp=float(np.mean([abs(r['primary_shift_pp']) for r in val])),original_validation_mean_abs_calibration_shift_pp=float(np.mean([abs(r['calibration_shift_pp']) for r in val])),original_test_loaded=False,diagnostic_not_model_selection=True,interpretation='Sensitivity to refitting on one added cell, measured only on original training/validation inputs. No query true SOH or outer input used for calibration or correction.',source_sha256=sha(__file__),goal_complete=False);save_json(root/'training_refit_sensitivity.json',result);print(json.dumps(dict(validation=val,mean_abs_calibration_shift_pp=result['original_validation_mean_abs_calibration_shift_pp'])),flush=True)


if __name__=='__main__':main()
