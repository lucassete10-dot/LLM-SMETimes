"""Reload, query/future isolation, independent arithmetic and strict acceptance."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import copy
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from battery_smetimes.research_llm import load_adapter,predict_llm
from battery_smetimes.llm_curve_objective import curve_metrics
from battery_smetimes.llm_knee_supervision import training_knee_target,KNEE_RULES
from battery_smetimes.physical_types import load_registry
from scripts.audit_llm_curve_repair import load_frozen,fingerprint
from scripts.train_llm_curve_repair import create_network
from scripts.prepare_same_type_forecast import save_json
from battery_smetimes.llm_reporting import raw_curve_numeric_metrics


def strict_metrics(x,y,p,origin,thresholds):
    x,y,p=[np.asarray(v,dtype=np.float64) for v in (x,y,p)]
    if len(x)<6 or not np.isfinite(np.c_[x,y,p]).all() or np.any(np.diff(x)<=0) or np.any(y<=0):
        return {'evaluable':False,'strict_joint_pass':False,'reason':'invalid or insufficient raw curve coverage'}
    legacy=curve_metrics(x,y,p)
    if 'mae_pp' not in legacy:return {'evaluable':False,'strict_joint_pass':False,'legacy':legacy}
    error=np.abs(p-y);span=x[-1]-x[0];tail=x>=x[0]+.8*span;drops=[]
    for j in range(3):
        inside=(x>=x[0]+j*span/3)&(x<=x[0]+(j+1)*span/3);xx,yy,pp=x[inside],y[inside],p[inside]
        if len(xx)<3:return {'evaluable':False,'strict_joint_pass':False,'reason':'stage coverage'}
        width=max(1.,min(5.,(xx[-1]-xx[0])/4));a=xx<=xx[0]+width;b=xx>=xx[-1]-width
        drops.append(100*abs((np.median(yy[a])-np.median(yy[b]))-(np.median(pp[a])-np.median(pp[b]))))
    width=min(5.,span/10);fade=np.median(y[x<=x[0]+width])-np.median(y[x>=x[-1]-width])
    row={'evaluable':bool(legacy['evaluable']),'mape_percent':float(100*np.mean(error/y)),
        'mae_pp':float(100*error.mean()),'tail_mae_pp':float(100*error[tail].mean()),
        'start_error_pp':float(100*error[0]),'endpoint_error_pp':float(100*error[-1]),
        'stage_drop_max_pp':float(max(drops)),'relative_fade_mae':float(error.mean()/max(fade,.02)),
        'rmse_pp':float(100*np.sqrt(np.mean((p-y)**2))),'legacy_joint_pass':bool(legacy['joint_pass']),
        **{k:bool(legacy[k]) for k in ('false_plateau','false_rise','reversed_acceleration')}}
    truth_knee=training_knee_target({'label_cycle':x,'soh':y},origin)
    predicted_knee=training_knee_target({'label_cycle':x,'soh':p},origin)
    row['truth_knee']=truth_knee;row['predicted_knee']=predicted_knee
    knee_error=None;knee_pass=True
    if truth_knee['valid']:
        if predicted_knee['valid']:
            # Shared interval-extraction semantics for truth and prediction.
            distance=max(0.,truth_knee['lower']-predicted_knee['upper'],predicted_knee['lower']-truth_knee['upper'])*100
            knee_error=float(distance/span);knee_pass=knee_error<=.1
        else:knee_pass=False
    row['knee_interval_distance_fraction']=knee_error;row['knee_pass']=knee_pass
    row['numeric_gates']={k:row[k]<=limit for k,limit in thresholds.items()}
    row['strict_joint_pass']=bool(row['evaluable'] and all(row['numeric_gates'].values()) and knee_pass
        and not any(row[k] for k in ('false_plateau','false_rise','reversed_acceleration')))
    # Cross-check the historical implementation without copying its cached JSON.
    for key in thresholds:np.testing.assert_allclose(row[key],legacy[key],rtol=0,atol=1e-10)
    return row


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--roles',nargs='+',choices=['training','validation','test'],default=['training'])
    args=ap.parse_args();root=args.run;cfg=json.loads((root/'config.json').read_text(encoding='utf-8'))
    if cfg['mode']=='tiny' and args.roles!=['training']:raise ValueError('P1 diagnosis must not evaluate heldout/validation')
    _,cells,plan,registry_path,_=load_frozen(cfg)
    processor=joblib.load(root/'preprocessor.joblib')
    if processor.train_ids!=cfg['train_ids']:raise ValueError('processor training membership mismatch')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    network=create_network(cfg,processor,cfg['architecture'],cfg['dropout'])
    load_adapter(network,torch.load(root/'model.pt',map_location='cpu',weights_only=True));network.eval()
    bundle={'network':network,'processor':processor,'device':'cuda','registry':load_registry(registry_path),
        'physical_family':cells[cfg['train_ids'][0]]['physical_family']}
    verification=[];reports={}
    for role in args.roles:
        ids=cfg['train_ids'] if role=='training' else plan[role if role!='validation' else 'validation']
        records={};arrays={};skipped=[]
        for cid in ids:
            c=cells[cid];origin=cfg['origin'];take=(c['label_cycle']>origin)&np.isfinite(c['soh'])&(c['soh']>0)
            x,y=c['label_cycle'][take],c['soh'][take]
            if len(x)<2:skipped.append({'cell_id':cid,'reason':'fewer than two valid future measurements'});continue
            before=predict_llm(bundle,c,origin,x)
            probe=np.unique([0,len(x)//3,len(x)//2,len(x)-1]);chosen=x[probe]
            single=np.asarray([predict_llm(bundle,c,origin,np.asarray([cy]))[0] for cy in chosen])
            reordered=predict_llm(bundle,c,origin,chosen[::-1])[::-1]
            contaminated=copy.deepcopy(c)
            for field in ('soh','label_cycle','unfiltered_soh','capacity','eol','lifetime'):contaminated.pop(field,None)
            # Also add unseen future signal rows when the stored cache is N=100.
            future=np.asarray([origin+100,origin+200],dtype=c['cycle'].dtype)
            keep=contaminated['cycle']<=origin
            contaminated['cycle']=np.r_[contaminated['cycle'][keep],future]
            contaminated['raw']=np.r_[contaminated['raw'][keep],np.full((2,c['raw'].shape[1]),np.inf,dtype=c['raw'].dtype)]
            poisoned=predict_llm(bundle,contaminated,origin,chosen)
            rows={'cell_id':cid,'role':role,'singleton_query_error_soh':float(np.max(abs(single-before[probe]))),
                'reordered_query_error_soh':float(np.max(abs(reordered-before[probe]))),
                'future_and_label_poison_error_soh':float(np.max(abs(poisoned-before[probe])))}
            saved=root/('TRAIN_predictions.npz' if role=='training' else 'selected_predictions.npz')
            if role!='test' and saved.exists():
                with np.load(saved) as a:
                    key=f'{cid}__{origin}__prediction'
                    if key in a:rows['reload_max_error_soh']=float(np.max(abs(before-a[key])))
            rows['passed']=all(v<=1e-5 for k,v in rows.items() if k.endswith('_soh'))
            verification.append(rows)
            records[cid]=strict_metrics(x,y,before,origin,cfg['final_thresholds'])
            numeric=raw_curve_numeric_metrics(x,y,before)
            for key,value in numeric.items():
                if key in records[cid] and isinstance(value,float):np.testing.assert_allclose(value,records[cid][key],rtol=0,atol=1e-10)
                if key not in records[cid]:records[cid][key]=value
            prefix=f'{cid}__{origin}__';arrays.update({prefix+k:v for k,v in [('cycle',x),('truth',y),('prediction',before)]})
        reports[role]={'scope':'TRAIN in-sample diagnostic' if role=='training' else 'viewed development data',
            'by_cell':records,'skipped':skipped,'cells_requested':len(ids),'strict_passes':sum(r['strict_joint_pass'] for r in records.values()),
            'all_requested_evaluable_and_passed':bool(len(records)==len(ids) and not skipped and all(r['strict_joint_pass'] for r in records.values()))}
        np.savez_compressed(root/(role+'_audited_predictions.npz'),**arrays)
    checks={'model_sha256':fingerprint(root/'model.pt'),'per_cell':verification,'tolerance_soh':1e-5,
        'verification_source_sha256':fingerprint(Path(__file__)),
        'statistics_cards_source_sha256':fingerprint(Path('battery_smetimes/research_llm.py')),
        'passed':all(r['passed'] for r in verification),'knee_rules':KNEE_RULES,'role_scope':args.roles,
        'no_test_candidate_selection':True,'goal_complete':False}
    save_json(root/'repair_verification.json',checks);save_json(root/'strict_acceptance.json',reports)
    print(json.dumps({'verification_passed':checks['passed'],'roles':{k:{'requested':v['cells_requested'],'strict_passes':v['strict_passes']} for k,v in reports.items()}},indent=2),flush=True)
    if not checks['passed']:raise ValueError('replay/input/query isolation check failed')


if __name__=='__main__':main()
