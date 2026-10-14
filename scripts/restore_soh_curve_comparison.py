"""Restore tree curve baseline and isolate training-size versus model changes.

Uses the previous diagnostic's frozen development roles. Original fold test
predictions are read only for historical comparison, never for model selection.
"""
import argparse
import copy
import hashlib
import json
import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import torch
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from battery_smetimes.forecast import prefix_vector, query_matrix, type_key
from battery_smetimes.research_baselines import signal_prefix
from battery_smetimes.research_sequences import HistoryPreprocessor, training_contexts, pack_contexts
from scripts.diagnose_soh_curve_small import AUDIT, CurveModel, evaluate, mean_score, metrics, save


def load_cells(ids):
    registry=json.loads((AUDIT/'cell_registry.json').read_text())['cells']
    issues=json.loads(Path('data/interim/same_type_forecast/mit_discharge_integrity_20260929T060206_688121Z.json').read_text())['issues']
    cells={}
    for key in ids:
        record=registry[key];p=Path(record['prepared_file'])
        if hashlib.sha256(p.read_bytes()).hexdigest()!=record['prepared_sha256']:raise ValueError('changed prepared cell')
        with np.load(p) as d:
            c={**record,'raw':d['raw_features'].copy(),'cycle':d['cycle'].copy(),'soh':d['soh'].copy(),'label_cycle':d['label_cycle'].copy()}
        rejected=[i['cycle'] for i in issues if i['cell_id']==key]
        c['soh'][np.isin(c['label_cycle'],rejected)]=np.nan
        good=~np.isin(c['cycle'],rejected);c['cycle'],c['raw']=c['cycle'][good],c['raw'][good]
        cells[key]=c
    return cells


def tree_input(cell, future, types=None):
    prefix=signal_prefix(cell,100)
    if types is not None:
        # Fixed known batch/policy vocabulary; it is a provenance feature,
        # not a physical-family gate or a claim that each category was trained.
        prefix=np.r_[prefix[:-5],[float(type_key(cell['metadata'])==t) for t in types],prefix[-5:]]
    return query_matrix(prefix,100,future)


def fit_tree(cells,ids,seed,leaf,legacy):
    types=[f'MIT/batch{b}/{d}' for b in (1,2,3) for d in ('constant','decreasing','increasing')] if legacy else None
    xs=[];ys=[];ws=[]
    for key in ids:
        c=cells[key];take=np.flatnonzero(np.isfinite(c['soh'])&(c['soh']>0)&(c['label_cycle']>100))
        take=take[np.unique(np.linspace(0,len(take)-1,min(len(take),96)).round().astype(int))]
        xs.append(tree_input(c,c['label_cycle'][take],types));ys.append(c['soh'][take]);ws.append(np.full(len(take),1/len(take)))
    imputer=SimpleImputer(strategy='median',keep_empty_features=True)
    x=imputer.fit_transform(np.concatenate(xs));y=np.concatenate(ys)
    estimator=ExtraTreesRegressor(n_estimators=160,min_samples_leaf=leaf,max_features=.8,random_state=seed,n_jobs=2)
    estimator.fit(x,y,sample_weight=np.concatenate(ws))
    return dict(model=estimator,imputer=imputer,types=types,train_ids=ids,physical_family=cells[ids[0]]['physical_family'],origin=100)


def eval_tree(bundle,cells,ids):
    rows=[];arrays={}
    for key in ids:
        c=cells[key]
        if c['physical_family']!=bundle['physical_family']:raise ValueError('physical family mismatch')
        take=np.isfinite(c['soh'])&(c['soh']>0)&(c['label_cycle']>100)
        x=tree_input(c,c['label_cycle'][take],bundle['types'])
        p=bundle['model'].predict(bundle['imputer'].transform(x));y=c['soh'][take]
        rows.append(dict(cell_id=key,**metrics(y,p)))
        arrays.update({key+'__cycle':c['label_cycle'][take],key+'__truth':y,key+'__prediction':p})
    return rows,arrays


def summarize(rows,variant,seed,seconds,**extra):
    return dict(variant=variant,seed=seed,seconds=seconds,**extra,
                **{k:float(np.mean([r[k] for r in rows])) for k in ('mape','late_mape','endpoint_error_pp','mae_pp','rmse_pp','r2')})


def history_comparison(out):
    base=Path('experiments/soh_research_v5/runs')
    paths={'tree':base/'20260930T080008_026015Z/MIT_fold3/20260930/extra_trees',
           'llm':base/'20260930T090318_035370Z/MIT_fold3/20260930/llm_experts1'}
    loaded={k:np.load(p/'test/predictions.npz') for k,p in paths.items()}
    report={'scope':'read-only historical test comparison, never used for new checkpoint selection','by_cell':[],'source_sha256':{}}
    keysets=[{k[:-len('__100__truth')] for k in a.files if k.endswith('__100__truth')} for a in loaded.values()]
    assert keysets[0]==keysets[1]
    for key in sorted(keysets[0]):
        prefix=key+'__100__';a,b=loaded['tree'],loaded['llm']
        assert np.array_equal(a[prefix+'cycle'],b[prefix+'cycle'])
        assert np.array_equal(a[prefix+'truth'],b[prefix+'truth'],equal_nan=True)
        y=a[prefix+'truth'];valid=np.isfinite(y)&(y>0)
        # Frozen integrity masks are separately retained in saved arrays.
        if prefix+'valid_target' in a.files:valid &= a[prefix+'valid_target'].astype(bool)
        report['by_cell'].append(dict(cell_id=key,**{v:metrics(y[valid],z[prefix+'prediction'][valid]) for v,z in loaded.items()}))
    for v,p in paths.items():
        report['source_sha256'][v]={str(p/name):hashlib.sha256((p/name).read_bytes()).hexdigest() for name in ('config.json','selected.json','test/predictions.npz','test/metrics.json')}
    report['aggregate']={v:{k:float(np.mean([r[v][k] for r in report['by_cell']])) for k in ('mape','late_mape','endpoint_error_pp','rmse_pp','mae_pp','r2')} for v in paths}
    save(out/'historical_same_cells.json',report)
    for z in loaded.values():z.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--previous',type=Path,default=Path('experiments/soh_curve_diagnostics/20261001T024923_787049Z'));args=parser.parse_args()
    old=json.loads((args.previous/'config.json').read_text())
    split=next(s for s in json.loads((AUDIT/'splits.json').read_text())['plans'] if s['name']=='MIT_fold3')
    small=old['train'];inner=old['inner_validation'];val=old['diagnostic_validation']
    large=sorted(set(split['train'])-set(inner))
    assert not (set(large)&set(inner+val+split['test']))
    out=Path('experiments/soh_curve_restoration')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');out.mkdir(parents=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'training.log',encoding='utf-8'),logging.StreamHandler()])
    config=dict(previous=str(args.previous.resolve()),train_small=small,train_large=large,inner_validation=inner,
                diagnostic_validation=val,excluded_test=split['test'],seeds=old['seeds'],origin=100,
                tree_candidates=[4,16,32],tree_estimators=160,mlp_epochs=160,
                selection='inner cell-macro average of full and late-half MAPE',
                variants=['tree_small','tree_small_provenance_features','tree_large','mlp_large'],
                scope='development diagnosis; validation repeatedly inspected; independent new test still needed',
                feature_contract='early first100 source-cycle V/I/T statistics + known conditions + requested cycle; no SOH/lifetime input')
    save(out/'config.json',config)
    sources=[Path(__file__),Path('scripts/diagnose_soh_curve_small.py'),Path('battery_smetimes/research_sequences.py'),Path('battery_smetimes/research_baselines.py'),Path('battery_smetimes/forecast.py')]
    for p in sources:
        destination=out/'source_snapshot'/(Path('scripts')/p.name if p.is_absolute() else p)
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,destination)
    save(out/'source_hashes.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    cells=load_cells(large+inner+val)
    assert len({c['physical_family'] for c in cells.values()})==1
    save(out/'data_hashes.json',{k:cells[k]['prepared_sha256'] for k in cells})
    history_comparison(out)
    rows=[]
    for variant in config['variants']:
        ids=large if variant.endswith('_large') else small
        for seed in config['seeds']:
            started=time.perf_counter();folder=out/variant/str(seed);folder.mkdir(parents=True)
            if variant.startswith('tree'):
                scores=[];selected=None;best=float('inf')
                for leaf in config['tree_candidates']:
                    b=fit_tree(cells,ids,seed,leaf,legacy='provenance' in variant)
                    r,a=eval_tree(b,cells,inner);score=mean_score(r)
                    cand=folder/f'candidate_leaf{leaf}';cand.mkdir()
                    joblib.dump(b,cand/'model.joblib');save(cand/'inner_metrics.json',r);np.savez_compressed(cand/'inner_predictions.npz',**a)
                    scores.append(dict(leaf=leaf,inner_score=score))
                    if score<best:best=score;selected=b;selected_leaf=leaf
                save(folder/'selection.json',dict(candidates=scores,selected_leaf=selected_leaf))
                joblib.dump(selected,folder/'model.joblib')
                vr,arrays=eval_tree(selected,cells,val);tr,_=eval_tree(selected,cells,ids)
                extra=dict(selected_leaf=selected_leaf)
                # Executable artifact replay and inference future-label invariance.
                replay=joblib.load(folder/'model.joblib');probe=copy.deepcopy(cells[val[0]])
                future=np.arange(101,401)
                input_before=tree_input(probe,future,replay['types'])
                before=replay['model'].predict(replay['imputer'].transform(input_before))
                probe['soh'][:]=123;probe['label_cycle'][:]=987654
                input_after=tree_input(probe,future,replay['types'])
                assert np.array_equal(input_before,input_after,equal_nan=True)
                after=replay['model'].predict(replay['imputer'].transform(input_after))
                np.testing.assert_allclose(before,after,rtol=0,atol=1e-12)
                rr,ra=eval_tree(replay,cells,val)
                for k in arrays:np.testing.assert_allclose(arrays[k],ra[k],rtol=0,atol=1e-12)
                save(folder/'replay_checks.json',dict(future_label_input_invariance_exact=True,saved_model_predictions_match_atol=1e-12))
            else:
                torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(seed)
                processor=HistoryPreprocessor().fit(cells,ids,[100]);joblib.dump(processor,folder/'preprocessor.joblib')
                model=CurveModel(processor);optim=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
                context=training_contexts(cells,ids,processor);batch=pack_contexts(context,'cpu')
                ic=training_contexts(cells,inner,processor);best=float('inf');history=[]
                for epoch in range(1,161):
                    model.train();pred=model(batch)
                    loss=(((pred-batch['truth'])**2*batch['valid']).sum(1)/batch['valid'].sum(1)).mean()
                    if not torch.isfinite(loss):raise ValueError('nonfinite loss')
                    optim.zero_grad();loss.backward();nn_clip=torch.nn.utils.clip_grad_norm_(model.parameters(),1);optim.step()
                    if epoch%5==0:
                        score=mean_score(evaluate(model,ic)[0]);history.append(dict(epoch=epoch,train_mse=float(loss.detach()),inner_score=score))
                        if score<best:best=score;state=copy.deepcopy(model.state_dict());selected_epoch=epoch
                model.load_state_dict(state);torch.save(state,folder/'model.pt');save(folder/'history.json',history)
                cap=max(len(c['label_cycle']) for c in cells.values())
                vc=training_contexts(cells,val,processor,cap=cap);vr,arrays=evaluate(model,vc)
                tr,_=evaluate(model,training_contexts(cells,ids,processor,cap=cap))
                for c in vc:
                    valid=c['valid'].astype(bool);arrays[c['cell_id']+'__cycle']=c['query'][valid,0]*processor.query_scale.scaler.scale_[0]*100+processor.query_scale.scaler.mean_[0]*100
                extra=dict(selected_epoch=selected_epoch)
            save(folder/'metrics.json',dict(validation=vr,train=tr));np.savez_compressed(folder/'predictions.npz',**arrays)
            row=summarize(vr,variant,seed,time.perf_counter()-started,train_mape=float(np.mean([r['mape'] for r in tr])),**extra)
            rows.append(row);save(out/'summary.json',rows);logging.info('%s',json.dumps(row))
    save(out/'completion.json',dict(complete=True,runs=len(rows),original_test_cells_used_for_new_fitting_or_selection=0))
    print('OUTPUT='+str(out.resolve()),flush=True)


if __name__=='__main__':main()
