"""Independent point-error arithmetic and raw-label replay; no model selection."""
import argparse,json
from pathlib import Path
import numpy as np
from scripts.prepare_same_type_forecast import save_json
from scripts.train_same_type_forecast import load_data


def point_metrics(x,y,p):
    if not (len(x)==len(y)==len(p) and len(x)>1 and np.all(np.diff(x)>0)):
        raise ValueError('invalid prediction coordinates')
    if not np.isfinite(np.r_[x,y,p]).all() or np.any(y<=0):
        raise ValueError('invalid prediction values')
    residual=p.astype(np.float64)-y.astype(np.float64)
    tail=x>=x[0]+.8*(x[-1]-x[0])
    answer={'mae_pp':100*np.sum(np.abs(residual))/len(x),
        'mape_percent':100*np.sum(np.abs(residual)/y)/len(x),
        'rmse_pp':100*np.sqrt(np.dot(residual,residual)/len(x)),
        'start_error_pp':100*abs(residual[0]),'endpoint_error_pp':100*abs(residual[-1]),
        'tail_mae_pp':100*np.sum(np.abs(residual[tail]))/np.count_nonzero(tail)}
    variance=np.dot(y-y.mean(),y-y.mean())
    answer['r2']=1-np.dot(residual,residual)/variance if variance>1e-12 else None
    # Supplemental continuous-cycle weighting, not a replacement pass rule.
    weights=np.r_[np.diff(x)[0]/2,(x[2:]-x[:-2])/2,np.diff(x)[-1]/2]
    weighted=100*np.sum(weights*np.abs(residual))/weights.sum()
    return answer,float(weighted)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args()
    root=Path(args.run);cfg=json.loads((root/'config.json').read_text())
    audit=Path(cfg['frozen_plan']);frozen=json.loads((audit/'config.json').read_text())
    prepared=json.loads((audit/'audit.json').read_text())['prepared_folder']
    _,cells=load_data(Path(prepared),frozen['config']['integrity_audit'])
    records=[];maximum=0.;count=0
    for candidate in cfg['candidates']:
        dest=root/candidate['name']
        for role,ids in (('validation',cfg['plan']['validation']),('heldout',cfg['plan']['test'])):
            saved_metrics=json.loads((dest/f'{role}_metrics.json').read_text())
            with np.load(dest/f'{role}_predictions.npz') as arrays:
                for cid in ids:
                    prefix=f'{cid}__{cfg["origin"]}__'
                    if prefix+'cycle' not in arrays:continue
                    x,y,p=[arrays[prefix+k] for k in ('cycle','truth','prediction')]
                    cell=cells[cid]
                    take=(cell['label_cycle']>cfg['origin']) & np.isfinite(cell['soh']) & (cell['soh']>0)
                    np.testing.assert_array_equal(x,cell['label_cycle'][take])
                    np.testing.assert_allclose(y,cell['soh'][take],atol=1e-7,rtol=0)
                    independent,weighted=point_metrics(x,y,p)
                    errors={}
                    for name,value in independent.items():
                        stored=saved_metrics[cid][name]
                        if value is None:
                            if stored is not None:raise ValueError('R2 mismatch')
                            continue
                        errors[name]=abs(float(stored)-float(value))
                    delta=max(errors.values());maximum=max(maximum,delta);count+=1
                    records.append({'candidate':candidate['name'],'role':role,'cell':cid,
                        'raw_label_replay_passed':True,'max_arithmetic_difference':delta,
                        'cycle_weighted_mae_pp':weighted})
    passed=count>0 and maximum<=1e-5
    save_json(root/'metric_verification.json',{'passed':passed,'checked_curves':count,
        'scope':'raw labels, coordinates, MAE/MAPE/RMSE/R2/start/endpoint/tail; shape detector not independently verified',
        'max_arithmetic_difference':maximum,'tolerance':1e-5,'by_cell':records})
    print('metric verification',count,'curves; max difference',maximum,'passed',passed)
    if not passed:raise ValueError('independent metric arithmetic mismatch')


if __name__=='__main__':main()
