"""TRAIN free-coefficient feasibility for bounded local residuals; never a teacher."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from scipy import sparse
from scipy.optimize import linprog
from battery_smetimes.llm_query_rate import integrate_linear_rates
from scripts.audit_llm_curve_repair import load_frozen,fingerprint
from scripts.prepare_same_type_forecast import save_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--configured-rates',action='store_true');args=parser.parse_args()
    root=Path(json.loads(Path('experiments/llm_curve_repair/latest_P0.json').read_text())['folder'])
    cfg=json.loads((root/'config.json').read_text());_,cells,plan,_,_=load_frozen(cfg)
    if 'b1c0' not in cfg['tiny_train_ids'] or 'b1c0' not in plan['train']:raise ValueError('TRAIN-only diagnostic')
    cell=cells['b1c0'];take=(cell['label_cycle']>cfg['origin'])&np.isfinite(cell['soh'])&(cell['soh']>0)
    x,y=cell['label_cycle'][take].astype(float),cell['soh'][take].astype(float);t=(x-cfg['origin'])/100;n=len(y)
    k=torch.tensor(cfg['rate_knots_horizon_100cycles'],dtype=torch.float64) if args.configured_rates else torch.arange(0,32.01,.25,dtype=torch.float64)
    design=integrate_linear_rates(torch.eye(len(k),dtype=torch.float64),k,torch.tensor(np.tile(t,(len(k),1)))).numpy().T
    records=[]
    for step in [.5,.25,.1,.05,.025]:
        knots=np.arange(0,32.001,step)
        if t.max()>knots[-1]:raise ValueError('diagnosis does not cover horizon')
        j=np.minimum(np.searchsorted(knots,t,side='right')-1,len(knots)-2);fraction=(t-knots[j])/step
        residual=sparse.coo_matrix((np.r_[1-fraction,fraction],(np.r_[np.arange(n),np.arange(n)],np.r_[j,j+1])),shape=(n,len(knots))).tocsr()
        matrix=sparse.hstack([sparse.csr_matrix(np.c_[np.ones(n),-design]),residual],format='csr');w=matrix.shape[1];eye=sparse.eye(n)
        inequalities=sparse.vstack([sparse.hstack([matrix,-eye]),sparse.hstack([-matrix,-eye])],format='csr')
        for amplitude in [.005,.01]:
            fit=linprog(np.r_[np.zeros(w),np.ones(n)/n],A_ub=inequalities,b_ub=np.r_[y,-y],
                bounds=[(.5,1.3)]+[(0,None)]*len(k)+[(0,0)]+[(-amplitude,amplitude)]*(len(knots)-1)+[(0,None)]*n,method='highs')
            if not fit.success:raise ValueError(fit.message)
            row={'residual_spacing_cycles':step*100,'pointwise_residual_bound_pp':100*amplitude,
                'best_raw_TRAIN_mae_pp':float(100*fit.fun),'success':True}
            records.append(row);print(row,flush=True)
    name='TRAIN_b1c0_configured_local_capacity_NOT_FORECAST.json' if args.configured_rates else 'TRAIN_b1c0_local_residual_capacity_NOT_FORECAST.json'
    save_json(root/name,{
        'scope':'Free coefficients on known TRAIN labels, feasibility diagnosis only; no forecasting or teacher',
        'rate_knots_horizon_100cycles':k.tolist(),'level_constrained':True,'residual_at_origin':0,'records':records,
        'global_saturating_recovery_fixed_zero':True,
        'source_sha256':fingerprint(Path(__file__)),
        'warning':'Training expressivity is not early-signal predictability of future spikes. No frozen gates, references, labels or inference values were changed.'})


if __name__=='__main__':main()
