"""Evaluate frozen forecasts against separately provided labels; never retrain."""
import argparse
import json
from pathlib import Path
import numpy as np
from battery_smetimes.research_baselines import regression_metrics
from scripts.prepare_same_type_forecast import save_json,sha


def evaluate(prediction,cycles,soh,label_info):
    if label_info.get('cell_id')!=prediction['cell_id'] or label_info.get('normalization')!=prediction['normalization']:raise ValueError('cell identity/SOH reference definition mismatch')
    if prediction.get('reference_rule') not in ('first_comparable','median_first5_comparable') or label_info.get('reference_rule')!=prediction['reference_rule']:raise ValueError('initial reference rule missing/mismatched (first measurement vs first5 median)')
    cycles=np.asarray(cycles);soh=np.asarray(soh)
    if cycles.ndim!=1 or cycles.dtype.kind not in 'iu' or soh.shape!=cycles.shape or soh.dtype.kind not in 'fiu' or not len(cycles):raise ValueError('invalid label arrays')
    if (cycles<1).any() or np.any(np.diff(cycles.astype(np.int64))<=0):raise ValueError('label cycles must be positive, strictly increasing')
    if np.isfinite(soh).any() and np.max(soh[np.isfinite(soh)])>2:raise ValueError('SOH expects ratios, not percent values or Ah')
    future=np.asarray(prediction['future_source_cycles']);forecast=np.asarray(prediction['predicted_soh'])
    if future.ndim!=1 or future.dtype.kind not in 'iu' or forecast.shape!=future.shape or not len(future) or not np.isfinite(forecast).all() or np.any(np.diff(future)<=0) or np.any(future<=100):raise ValueError('invalid prediction file')
    lookup=dict(zip(future.tolist(),forecast.tolist()));truth=[];pred=[];matched=[];excluded=[];unrequested=[]
    for n,y in zip(cycles,soh):
        if n not in lookup:unrequested.append(int(n));continue
        if not np.isfinite(y) or y<=0:excluded.append(dict(cycle=int(n),reason='nonfinite_or_nonpositive_ground_truth'));continue
        matched.append(int(n));truth.append(float(y));pred.append(lookup[int(n)])
    if not truth:raise ValueError('no valid labels at requested future cycles')
    matched=np.asarray(matched);truth=np.asarray(truth);pred=np.asarray(pred);mid=.5*(matched[0]+matched[-1]);late=matched>=mid;tail=matched>=matched[-1]-99
    return dict(cell_id=prediction['cell_id'],physical_family=prediction['physical_family'],normalization=prediction['normalization'],
        independent_new_cell=prediction['independent_new_cell'],independence_status=prediction.get('independence_status','public replay'),
        metrics=regression_metrics(truth,pred),late_half_metrics=regression_metrics(truth[late],pred[late]),last100_observed_source_cycles_metrics=regression_metrics(truth[tail],pred[tail]),
        endpoint_absolute_soh_error_pp=float(abs(truth[-1]-pred[-1])*100),matched_cycles=matched.tolist(),truth_soh=truth.tolist(),predicted_soh=pred.tolist(),
        excluded_labels=excluded,unrequested_label_cycles=unrequested,requested_points_without_valid_truth=int(len(future)-len(matched)),
        accuracy_scope='single-cell single-release forecast; not five-seed/fold CV or verified independent generalization',
        rule='No rejection based on model error; high errors stay in every metric. Only invalid ground truth excluded, with reasons.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--prediction',type=Path,required=True);p.add_argument('--labels',type=Path,required=True);p.add_argument('--label-info',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    prediction=json.loads(args.prediction.read_text(encoding='utf-8'));info=json.loads(args.label_info.read_text(encoding='utf-8'))
    with np.load(args.labels,allow_pickle=False) as z:
        if set(z.files)!={'cycle','soh'}:raise ValueError('label file must contain only cycle/soh')
        result=evaluate(prediction,z['cycle'],z['soh'],info)
    result['source_sha256']={str(file):sha(file) for file in (args.prediction,args.labels,args.label_info)}
    args.output.parent.mkdir(parents=True,exist_ok=True);save_json(args.output,result);print('OUTPUT='+str(args.output.resolve()))


if __name__=='__main__':main()
