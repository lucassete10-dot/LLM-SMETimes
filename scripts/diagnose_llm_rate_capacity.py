"""TRAIN-only free physical-rate-node approximation; never forecast/teacher.

Checks whether fixed knot spacing itself limits the tiny gate before attributing
failure to the LLM representation. Fits known TRAIN answers for diagnosis only.
"""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import lsq_linear,linprog
from scipy import sparse
from battery_smetimes.llm_query_rate import integrate_linear_rates
from battery_smetimes.llm_curve_objective import curve_metrics,training_origin_target
from scripts.audit_llm_curve_repair import load_frozen,fingerprint
from scripts.prepare_same_type_forecast import save_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--p0');ap.add_argument('--recovery-max',type=float);ap.add_argument('--loss',choices=['mse','l1'],default='mse');args=ap.parse_args()
    root=Path(args.p0 or json.loads(Path('experiments/llm_curve_repair/latest_P0.json').read_text())['folder'])
    cfg=json.loads((root/'config.json').read_text());_,cells,plan,_,_=load_frozen(cfg)
    recovery_max=cfg['recovery_max_soh'] if args.recovery_max is None else args.recovery_max
    if not 0<=recovery_max<=.02:raise ValueError('recovery diagnosis bound outside 0 to two pp')
    records=[]
    for name,knots in [('original14',np.asarray(cfg['rate_knots_horizon_100cycles'])),
        ('half_unit65',np.arange(0,32.01,.5)),('quarter_unit129',np.arange(0,32.01,.25))]:
        for cid in cfg['tiny_train_ids']:
            cell=cells[cid];origin=cfg['origin'];take=(cell['label_cycle']>origin)&np.isfinite(cell['soh'])&(cell['soh']>0)
            x,y=cell['label_cycle'][take],cell['soh'][take];time=(x-origin)/100
            k=torch.as_tensor(knots,dtype=torch.float64)
            design=integrate_linear_rates(torch.eye(len(k),dtype=torch.float64),k,torch.as_tensor(np.tile(time,(len(k),1)),dtype=torch.float64)).numpy().T
            start,available=training_origin_target(cell,origin);solutions=[]
            for tau in (.2,1.,3.,10.):
                matrix=np.c_[np.ones(len(x)),-design,-np.expm1(-time/tau)];target=y
                if available:
                    row=np.zeros((1,len(knots)+2));row[0,0]=np.sqrt(len(x));matrix=np.r_[matrix,row];target=np.r_[target,start*np.sqrt(len(x))]
                if args.loss=='mse':
                    fit=lsq_linear(matrix,target,bounds=(np.r_[.5,np.zeros(len(knots)),-recovery_max],np.r_[1.3,np.full(len(knots),np.inf),recovery_max]),tol=1e-8,max_iter=150,lsmr_tol='auto')
                else:
                    # Exact convex L1 solution for each fixed recovery tau;
                    # diagnostic parameters never become forecasting labels.
                    n,w=matrix.shape;eye=sparse.eye(n,format='csr');m=sparse.csr_matrix(matrix)
                    bound=sparse.vstack([sparse.hstack([m,-eye]),sparse.hstack([-m,-eye])],format='csr')
                    cost=np.r_[np.zeros(w),np.ones(n)]
                    fit=linprog(cost,A_ub=bound,b_ub=np.r_[target,-target],
                        bounds=[(.5,1.3)]+[(0,None)]*len(knots)+[(-recovery_max,recovery_max)]+[(0,None)]*n,method='highs')
                    if not fit.success:raise ValueError('TRAIN L1 capacity solve failed: '+fit.message)
                    fit.x=fit.x[:w]
                prediction=np.c_[np.ones(len(x)),-design,-np.expm1(-time/tau)]@fit.x
                score=float(np.mean(abs(prediction-y))) if args.loss=='l1' else float(np.mean((prediction-y)**2))
                solutions.append((score,prediction,bool(fit.success)))
            _,prediction,success=min(solutions,key=lambda v:v[0]);metrics=curve_metrics(x,y,prediction)
            gate=all(metrics[key]<=limit for key,limit in cfg['tiny_gates_pp'].items())
            records.append({'knots':name,'cell':cid,'optimizer_success':success,'metrics':metrics,'tiny_shape_approximation_pass':gate})
        local=[r for r in records if r['knots']==name]
        print(name, sum(r['tiny_shape_approximation_pass'] for r in local),'/',len(local),
              'MAE',np.mean([r['metrics']['mae_pp'] for r in local]),flush=True)
    suffix='' if args.recovery_max is None else '_recovery_'+str(recovery_max).replace('.','p')
    if args.loss!='mse':suffix+='_'+args.loss
    save_json(root/('TRAIN_rate_capacity_NOT_FORECAST'+suffix+'.json'),{'scope':'free parameters fitted to known TRAIN labels; not early-signal forecast, no teacher, no inference use',
        'recovery_max_soh':recovery_max,
        'loss':args.loss,'recovery_tau_grid':[.2,1.,3.,10.],
        'records':records,'source_sha256':fingerprint(Path(__file__)),'all_tiny_ids_are_train':set(cfg['tiny_train_ids'])<=set(plan['train'])})


if __name__=='__main__':main()
