"""Exact TRAIN-only lower bound for the nonnegative-rate/recovery curve class."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.optimize import linprog
from scripts.audit_llm_curve_repair import load_frozen,fingerprint
from scripts.prepare_same_type_forecast import save_json


def variation_mae_bound(y,budget):
    y=np.asarray(y,dtype=np.float64);n=len(y)
    if n<2 or not np.isfinite(y).all() or budget<0:raise ValueError('invalid bound inputs')
    eye=sparse.eye(n,format='csr');zero=sparse.csr_matrix((n,n-1))
    diff=sparse.diags([-np.ones(n-1),np.ones(n-1)],[0,1],shape=(n-1,n),format='csr')
    matrix=sparse.vstack([
        sparse.hstack([eye,-eye,zero]),sparse.hstack([-eye,-eye,zero]),
        sparse.hstack([diff,sparse.csr_matrix((n-1,n)),-sparse.eye(n-1)]),
        sparse.hstack([sparse.csr_matrix((1,2*n)),sparse.csr_matrix(np.ones((1,n-1)))])],format='csr')
    fit=linprog(np.r_[np.zeros(n),np.ones(n)/n,np.zeros(n-1)],A_ub=matrix,
        b_ub=np.r_[y,-y,np.zeros(n-1),budget],
        bounds=[(None,None)]*n+[(0,None)]*(2*n-1),method='highs')
    if not fit.success:raise ValueError(fit.message)
    return {'total_upward_variation_soh':budget,'global_minimum_mae_pp':float(100*fit.fun),
            'optimizer_success':True,'maximum_inequality_violation':float(max(0,np.max(matrix@fit.x-np.r_[y,-y,np.zeros(n-1),budget])))}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--p0');ap.add_argument('--cell',default='b1c0');args=ap.parse_args()
    root=Path(args.p0 or json.loads(Path('experiments/llm_curve_repair/latest_P0.json').read_text())['folder'])
    cfg=json.loads((root/'config.json').read_text());_,cells,plan,_,_=load_frozen(cfg)
    if args.cell not in cfg['tiny_train_ids'] or args.cell not in plan['train']:raise ValueError('TRAIN-only diagnostic')
    cell=cells[args.cell];take=(cell['label_cycle']>cfg['origin'])&np.isfinite(cell['soh'])&(cell['soh']>0);y=cell['soh'][take]
    budgets=sorted(set([0.,cfg['recovery_max_soh'],.02]));records=[variation_mae_bound(y,b) for b in budgets]
    report={'scope':'TRAIN mathematical representational bound; no early-input forecasting, no teacher',
        'cell':args.cell,'points':len(y),'model_recovery_cap_soh':cfg['recovery_max_soh'],
        'argument':'Nonnegative integrated rates plus one saturating recovery amplitude <= cap have total upward variation <= cap. The LP permits ANY discrete curve with that variation budget, a superset of this model class. Its global minimum bounds model MAE from below.',
        'ignores_other_gates':True,'raw_truth_sha256':__import__('hashlib').sha256(y.tobytes()).hexdigest(),
        'records':records,'source_sha256':fingerprint(Path(__file__)),'tiny_mae_gate_pp':cfg['tiny_gates_pp']['mae_pp']}
    save_json(root/('TRAIN_'+args.cell+'_shape_lower_bound.json'),report);print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
