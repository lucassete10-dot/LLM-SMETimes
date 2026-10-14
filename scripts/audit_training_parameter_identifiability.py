"""Training-only knee-loss profiles: labels diagnose target ambiguity, never inference."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.optimize import lsq_linear
from battery_smetimes.tail_boost import labels
from scripts.train_tail_boost import load_cells
from scripts.prepare_same_type_forecast import save_json,sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    root=parser.parse_args().run;cfg=json.loads((root/'config.json').read_text())
    assert json.loads((root/'independent_verification.json').read_text())['passed']
    results=[];arrays={}
    # Fixed diagnostic tolerances, not model selection or relaxed accuracy gates.
    tolerances=(.0005,.002)
    for plan in cfg['plans']:
        cells=load_cells(dict(plan,validation=[]));assert set(cells)==set(plan['train'])
        cached=json.loads((root/(plan['name']+'_training_parameters.json')).read_text())
        for key,old in zip(plan['train'],cached):
            assert old['cell_id']==key;cycle,y=labels(cells[key]);idx=np.unique(np.linspace(0,len(y)-1,min(96,len(y))).round().astype(int))
            cycle,y=cycle[idx],y[idx];h=(cycle-100)/100;w=np.linspace(1,3,len(h));w/=w.sum()
            coefficients=[];losses=[];curves=[];knees=np.linspace(0,float(h.max()),31)
            for knee in knees:
                hinge=.1*np.logaddexp(0,(h-knee)/.1);design=np.column_stack([np.ones(len(h)),-h,-hinge**2])
                fit=lsq_linear(design*np.sqrt(w)[:,None],y*np.sqrt(w),bounds=([.5,0,0],[1.25,np.inf,np.inf]));assert fit.success
                p=design@fit.x;coefficients.append(fit.x);curves.append(p);losses.append(float(np.sum(w*(p-y)**2)))
            best=int(np.argmin(losses));coefficients=np.asarray(coefficients);curves=np.asarray(curves);rmse=np.sqrt(losses)
            np.testing.assert_allclose(coefficients[best],[old[f] for f in ('anchor','rate','acceleration')],atol=1e-12,rtol=0)
            np.testing.assert_allclose(knees[best],old['knee_h'],atol=1e-12,rtol=0)
            row=dict(dataset=plan['dataset'],cell_id=key,best_rmse_soh=float(rmse[best]),best_knee_h=float(knees[best]),profiles={})
            for tol in tolerances:
                keep=rmse<=rmse[best]+tol
                row['profiles'][str(tol)]=dict(candidates=int(keep.sum()),knee_range_cycles=float(100*np.ptp(knees[keep])),rate_range=float(np.ptp(coefficients[keep,1])),acceleration_range=float(np.ptp(coefficients[keep,2])),max_training_curve_spread_pp=float(100*np.max(np.ptp(curves[keep],axis=0))))
            results.append(row)
            arrays.update({key+'__'+name:value for name,value in [('cycle',cycle),('truth',y),('knees_h',knees),('coefficients',coefficients),('rmse',rmse),('curves',curves)]})
    summaries={}
    for dataset in {r['dataset'] for r in results}:
        rows=[r for r in results if r['dataset']==dataset];summaries[dataset]={}
        for tol in tolerances:
            profiles=[r['profiles'][str(tol)] for r in rows]
            summaries[dataset][str(tol)]=dict(cells=len(rows),median_knee_range_cycles=float(np.median([p['knee_range_cycles'] for p in profiles])),fraction_knee_range_ge50=float(np.mean([p['knee_range_cycles']>=50 for p in profiles])),median_profile_candidates=float(np.median([p['candidates'] for p in profiles])))
    np.savez_compressed(root/'training_parameter_profiles.npz',**arrays)
    save_json(root/'training_parameter_identifiability.json',dict(results=results,summary=summaries,tolerances_soh=list(tolerances),loaded_train_only=True,diagnostic_not_model_selection=True,goal_complete=False,source_sha256=sha(__file__),arrays_sha256=sha(root/'training_parameter_profiles.npz')))
    print(json.dumps(summaries),flush=True)


if __name__=='__main__':main()
