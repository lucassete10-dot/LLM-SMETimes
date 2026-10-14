"""TRAIN-label-only approximation audit of the LLM integrated trajectory head.

Fits free output parameters to already-known TRAIN curves, not an independent
forecast model. Never predicts held-out cells, routes inference, or provides
teachers. Results diagnose representational adequacy, not attainable accuracy.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from battery_smetimes.llm_curve_objective import curve_metrics,training_origin_target
from scripts.train_same_type_forecast import load_data
from scripts.prepare_same_type_forecast import save_json,sha


def numpy_trajectory(p,t):
    # Match torch.nn.functional.softplus default threshold exactly.
    soft=lambda v:np.where(np.asarray(v)>20,np.asarray(v),np.logaddexp(0.,v))
    width=.2+soft(p[4]);knee=5*soft(p[3])
    integral=width*(soft((t-knee)/width)-soft(-knee/width))
    recovery=.005*np.tanh(p[5])*(-np.expm1(-t/(.2+soft(p[6]))))
    return 1+.1*p[0]-.01*soft(p[1])*t-.02*soft(p[2])*integral+recovery


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args()
    root=Path(args.run);cfg=json.loads((root/'config.json').read_text());audit=Path(cfg['frozen_plan'])
    frozen=json.loads((audit/'config.json').read_text())
    _,cells=load_data(Path(json.loads((audit/'audit.json').read_text())['prepared_folder']),frozen['config']['integrity_audit'])
    records=[];arrays={};origin=cfg['origin']
    for cid in cfg['plan']['train']:
        cell=cells[cid];take=(cell['label_cycle']>origin)&np.isfinite(cell['soh'])&(cell['soh']>0)
        x=cell['label_cycle'][take];y=cell['soh'][take]
        if len(x)<6:continue
        t=(x-origin)/100;idx=np.unique(np.linspace(0,len(x)-1,min(96,len(x))).round().astype(int))
        tt,yy=t[idx],y[idx];label,available=training_origin_target(cell,origin)
        def residual(parameters):
            error=numpy_trajectory(parameters,tt)-yy
            if available:error=np.r_[error,(numpy_trajectory(parameters,np.array([0.]))[0]-label)*np.sqrt(len(tt))]
            return error
        solutions=[]
        for knee_guess in (.2,1.,2.):
            initial=np.array([(label if available else y[0])-1,-2.,-2.,knee_guess,-1.,0.,0.])
            initial[0]*=10
            fit=least_squares(residual,initial,bounds=([-50,-20,-20,-20,-20,-10,-20],[50,20,20,20,20,10,20]),max_nfev=500)
            solutions.append(fit)
        best=min(solutions,key=lambda r:np.dot(r.fun,r.fun));prediction=numpy_trajectory(best.x,t)
        metrics=curve_metrics(x,y,prediction)
        records.append({'cell':cid,'metrics':metrics,'parameters':best.x.tolist(),'optimizer_success':bool(best.success),
            'residual_norm':float(np.linalg.norm(best.fun))})
        arrays[cid+'__cycle']=x;arrays[cid+'__truth']=y;arrays[cid+'__approximation']=prediction
        print(cid,'TRAIN shape approximation MAE pp',round(metrics.get('mae_pp',float('nan')),3),flush=True)
    save_json(root/'head_capacity_diagnostic.json',{
        'scope':'TRAIN curves only; free head parameters fitted to known labels; not early-signal predictions or performance results',
        'inference_use':False,'heldout_labels_used':False,'source_sha256':sha(Path(__file__)),
        'settings':{'fit_points_cap':96,'max_nfev':500,'initial_knee_parameters':[.2,1.,2.],
            'objective':'raw point MSE plus exact-origin TRAIN auxiliary MSE','bounds':'finite search; no global-optimum claim'},
        'cells':records,'summary':{'count':len(records),
            'joint_approximation_passes':sum(r['metrics'].get('joint_pass',False) for r in records),
            **{k:float(np.mean([r['metrics'][k] for r in records if k in r['metrics']])) for k in ('mae_pp','tail_mae_pp','stage_drop_max_pp')}}})
    np.savez_compressed(root/'TRAIN_head_approximations_NOT_FORECAST.npz',**arrays)


if __name__=='__main__':main()
