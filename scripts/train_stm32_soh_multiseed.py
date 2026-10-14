"""Five seeds, all frozen folds, tiny float/INT8 models including NASA."""
import argparse
import csv
from datetime import datetime,timezone
import json
import logging
import os
from pathlib import Path
import shutil
import time
import traceback
import joblib
import numpy as np
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from battery_smetimes.physical_types import load_registry,require_same_registered_family
from battery_smetimes.stm32_forecast import Scale,quantize
from battery_smetimes.stm32_multiseed import RAW_COLUMNS,compact_input,pairs,predict
from battery_smetimes.research_baselines import evaluate_baseline
from scripts.train_stm32_soh import train,export_c,c_replay
from scripts.train_stm32_compact_trees import AUDIT
from scripts.train_same_type_forecast import load_data
from scripts.prepare_same_type_forecast import save_json,sha
from scripts.train_soh_research_v5 import save_scores


def inner_preprocessing_provenance(plan,records,family,scale):
    training=set(plan['train']);validation=set(plan['validation']);test=set(plan['test'])
    if training&validation or training&test or validation&test:
        raise ValueError('overlapping cells cannot have train-only preprocessing provenance')
    return dict(train_ids=list(plan['train']),validation_ids=list(plan['validation']),
        physical_family=family,origin=100,raw_columns=list(RAW_COLUMNS[plan['dataset']]),
        train_prepared_sha256={k:records[k]['prepared_sha256'] for k in plan['train']},
        fill=scale.fill.tolist(),mean=scale.mean.tolist(),std=scale.std.tolist(),
        fit_scope='inner training cells only; validation transformed but not fitted')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cuda');parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    frozen=json.loads((AUDIT/'config.json').read_text());cfg=frozen['config']
    prepared=Path(json.loads((AUDIT/'audit.json').read_text())['prepared_folder']);registry_path=AUDIT/'source_snapshot/battery_physical_registry.json'
    for p,d in [(prepared/'manifest.json',frozen['prepared_manifest_sha256']),(Path(cfg['integrity_audit']),frozen['integrity_sha256']),(registry_path,frozen['registry_sha256'])]:
        if sha(p)!=d:raise ValueError('changed frozen data/quality/type contract')
    registry=load_registry(registry_path);manifest,cells=load_data(prepared,cfg['integrity_audit'])
    records=json.loads((AUDIT/'cell_registry.json').read_text())['cells']
    for k,c in cells.items():c.update({f:records[k][f] for f in ('physical_family','reference_established_source_cycle')})
    plans=json.loads((AUDIT/'splits.json').read_text())['plans'];seeds=list(range(20261004,20261009));epochs=400
    if args.smoke:plans=[next(p for p in plans if p['name']=='NASA_fold1')];seeds=seeds[:1];epochs=20
    out=Path('experiments/stm32_soh_multiseed/runs')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');out.mkdir(parents=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'training.log',encoding='utf-8'),logging.StreamHandler()])
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    sources=[Path(__file__),Path('battery_smetimes/stm32_multiseed.py'),Path('battery_smetimes/stm32_forecast.py'),Path('scripts/train_stm32_soh.py'),Path('battery_smetimes/forecast.py'),Path('battery_smetimes/research_baselines.py')]
    for p in sources:
        dest=out/'source_snapshot'/(Path('scripts')/p.name if p.is_absolute() else p);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    shutil.copytree(AUDIT,out/'frozen_audit')
    save_json(out/'config.json',dict(smoke=args.smoke,seeds=seeds,device=args.device,epochs=epochs,hidden_candidates=[[32,16],[64,32]],
        raw_columns=RAW_COLUMNS,origin=100,selection='validation-only cell-macro full-future MAPE; no test selection',
        source_sha256={str(p):sha(p) for p in sources},data_sha256={r['cell_id']:r['sha256'] for r in manifest['cells']}))
    save_json(out/'splits.json',dict(plans=plans));progress=dict(complete=False,completed=[]);save_json(out/'progress.json',progress)
    save_json(Path('experiments/stm32_soh_multiseed/latest.json'),dict(folder=str(out.resolve())));rows=[]
    try:
        for plan in plans:
            if set(plan['train'])&set(plan['validation']) or set(plan['train']+plan['validation'])&set(plan['test']):raise ValueError('cell roles overlap')
            ids=plan['train']+plan['validation'];family=cells[ids[0]]['physical_family']
            require_same_registered_family(family,[cells[k]['physical_family'] for k in ids+plan['test']],registry)
            for seed in seeds:
                started=time.perf_counter();dest=out/plan['name']/str(seed);dest.mkdir(parents=True)
                x,y,w=pairs(cells,plan['train']);scale=Scale().fit(x,w);x=scale.transform(x)
                validation=[]
                for key in plan['validation']:
                    c=cells[key];take=np.isfinite(c['soh'])&(c['soh']>0)&(c['label_cycle']>100)
                    validation.append((scale.transform(compact_input(c,100,c['label_cycle'][take])),c['soh'][take]))
                candidates=[]
                for hidden in ((32,16),(64,32)):
                    model,selected_epoch,score,history=train(x,y,w,hidden,seed,epochs,validation,args.device)
                    name=f'candidate_{hidden[0]}_{hidden[1]}';torch.save(model.state_dict(),dest/(name+'.pt'))
                    save_json(dest/(name+'.json'),dict(hidden=hidden,selected_epoch=selected_epoch,validation_mape=score,history=history,
                        preprocessing_provenance=inner_preprocessing_provenance(plan,records,family,scale)))
                    candidates.append(dict(hidden=hidden,selected_epoch=selected_epoch,validation_mape=score))
                selected=min(candidates,key=lambda r:r['validation_mape']);save_json(dest/'selection.json',dict(candidates=candidates,selected=selected))
                dx,dy,dw=pairs(cells,ids);scale=Scale().fit(dx,dw);transformed=scale.transform(dx)
                network,_,_,history=train(transformed,dy,dw,selected['hidden'],seed,selected['selected_epoch'],None,args.device)
                bundle=dict(network=network,scale=scale,hidden=selected['hidden'],origins=[100],train_ids=ids,physical_family=family,registry=registry,seed=seed,raw_columns=RAW_COLUMNS[plan['dataset']])
                bundle['int8_layers']=quantize(network,transformed)
                joblib.dump(bundle,dest/'model.joblib');torch.save(network.state_dict(),dest/'model.pt');save_json(dest/'refit_history.json',history)
                for mode in ('float','int8'):
                    report,arrays=evaluate_baseline(dict(bundle,quantized=mode=='int8'),cells,plan['test'],predictor=predict)
                    save_scores(dest/mode,report,arrays)
                    for r in report['by_cell']:
                        prefix=r['cell_id']+'__100__';fc,truth,pred=[arrays[prefix+n] for n in ('cycle','truth','prediction')];tail=fc>=fc[-1]-99
                        rows.append(dict(dataset=plan['dataset'],fold=plan['name'],seed=seed,mode=mode,cell_id=r['cell_id'],**r['metrics'],
                                         late_half_mape=r['late_half']['mape_percent'],last100_mape=float(np.mean(abs(pred[tail]-truth[tail])/truth[tail])*100),endpoint_error_pp=float(abs(pred[-1]-truth[-1])*100),metric_source=str((dest/mode/'metrics.json').resolve())))
                export_c(dest/'c_export',bundle);checks=[]
                for key in plan['test']:
                    c=cells[key];future=c['label_cycle'][np.isfinite(c['soh'])&(c['soh']>0)&(c['label_cycle']>100)]
                    check=c_replay(dest/'c_export',bundle,compact_input(c,100,future))
                    if check.get('status')!='passed' or check['max_absolute_soh_difference']>1e-5:raise ValueError('C replay not within goal tolerance')
                    clean={k:v for k,v in c.items() if k not in ('soh','label_cycle','unfiltered_soh')}
                    clean['cycle']=np.r_[c['cycle'],101,200];clean['raw']=np.vstack([c['raw'],np.full((2,c['raw'].shape[1]),np.inf)])
                    for mode in (False,True):
                        np.testing.assert_array_equal(predict(dict(bundle,quantized=mode),clean,100,future),predict(dict(bundle,quantized=mode),c,100,future))
                    checks.append(dict(cell_id=key,**check,future_poison_and_label_removal=True))
                save_json(dest/'c_export/replay.json',checks)
                save_json(out/'by_cell.json',rows);progress['completed'].append(dict(fold=plan['name'],seed=seed));save_json(out/'progress.json',progress)
                latest=[r['mape_percent'] for r in rows if r['fold']==plan['name'] and r['seed']==seed and r['mode']=='int8']
                logging.info('SAVED %s seed=%s INT8 MAPE=%.3f hidden=%s epochs=%s seconds=%.1f',plan['name'],seed,np.mean(latest),selected['hidden'],selected['selected_epoch'],time.perf_counter()-started)
        summary={}
        for domain in sorted({r['dataset'] for r in rows}):
            summary[domain]={}
            for mode in ('float','int8'):
                subset=[r for r in rows if r['dataset']==domain and r['mode']==mode];group={}
                for key in ('mape_percent','rmse_soh','mae_soh','r2','late_half_mape','last100_mape','endpoint_error_pp'):
                    means=[float(np.mean([r[key] for r in subset if r['seed']==s])) for s in seeds]
                    group[key]=dict(mean=float(np.mean(means)),sd=float(np.std(means,ddof=1)) if len(means)>1 else None)
                group.update(cells=len({r['cell_id'] for r in subset}),seeds=len(seeds),fraction_cell_seed_mape_le6=float(np.mean([r['mape_percent']<=6 for r in subset])))
                summary[domain][mode]=group
        save_json(out/'summary.json',summary)
        with (out/'summary.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        progress['complete']=True;save_json(out/'progress.json',progress);print('OUTPUT='+str(out.resolve()),flush=True)
    except BaseException:
        save_json(out/'failure.json',dict(traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
