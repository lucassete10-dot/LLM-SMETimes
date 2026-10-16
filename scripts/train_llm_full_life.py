"""One-dataset full-life multi-origin LLM+SMETimes MVP, with origin100 VAL selection."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import time
import traceback

import joblib
import numpy as np
import torch

from battery_smetimes.llm_full_life import (load_full_life, FullLifePreprocessor, full_life_contexts,
    visible_window, pack_full_life, predict_full_life, read)
from battery_smetimes.research_llm import adapter_state, load_adapter, cards_embedding
from battery_smetimes.llm_curve_quality_v3 import curve_metrics_v3, freeze_validation_policy, validation_score
from battery_smetimes.physical_types import load_registry
from scripts.train_llm_curve_followup import create_network, repair_optimizer, full_training_aux, objective
from scripts.audit_llm_curve_repair import fingerprint


def save(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.part')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    for attempt in range(20):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if attempt==19:
                raise
            time.sleep(.1)


def evaluate_origin100(bundle, cells, ids):
    bundle['network'].eval()
    metrics, arrays = {}, {}
    for cid in ids:
        cell = cells[cid]
        take = (cell['label_cycle']>100)&np.isfinite(cell['soh'])&(cell['soh']>0)
        cycle, truth = cell['label_cycle'][take], cell['soh'][take]
        prediction = predict_full_life(bundle,cell,100,cycle)
        metrics[cid] = curve_metrics_v3(cycle,truth,prediction)
        for name, value in (('cycle',cycle),('truth',truth),('prediction',prediction)):
            arrays[f'{cid}__100__{name}'] = value
    return metrics, arrays


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not 1<=args.epochs<=120:
        raise ValueError('MVP epoch budget outside 1..120')
    root = args.out
    root.mkdir(parents=True,exist_ok=True)
    index, cells, plan, registry = load_full_life(args.index)
    settings = dict(index['base_settings'])
    settings.update(experiment_version='full_life_multi_origin_mvp_v1',seed=20260930,
                    full_life_index=str(args.index.resolve()),full_life_index_sha256=fingerprint(args.index),
                    max_tiny_epochs=args.epochs,origin=100,history_slots=100,curriculum=False,
                    dropout=.1,weight_decay=.03,training_origin_step=100,origin100_cell_mass=.5,
                    backbone_variant='R0',selection_scope='VAL origin100 only',
                    trained_from_scratch=True,train_ids=plan['train'])
    for key in ('observation_tokens','coarse_local_head','stage_rate_head'):
        settings.pop(key,None)
    sources = [Path(__file__),Path('battery_smetimes/llm_full_life.py'),Path('scripts/prepare_llm_full_life.py'),
        Path('scripts/train_llm_curve_followup.py'),Path('battery_smetimes/research_llm.py'),
        Path('battery_smetimes/research_sequences.py'),Path('battery_smetimes/llm_query_rate.py'),
        Path('battery_smetimes/llm_local_recovery.py'),Path('battery_smetimes/llm_curve_objective_v4.py'),
        Path('battery_smetimes/llm_curve_quality_v3.py')]
    provenance = dict(sha256={str(p.resolve()):fingerprint(p) for p in sources},index_sha256=fingerprint(args.index))
    if args.resume:
        if read(root/'config.json')!=settings or read(root/'source_manifest.json')!=provenance:
            raise ValueError('Resume configuration/source drift')
    else:
        if (root/'config.json').exists():
            raise ValueError('Existing run requires explicit resume')
        save(root/'config.json',settings)
        save(root/'source_manifest.json',provenance)
        for path in sources:
            destination = root/'source_snapshot'/path.resolve().relative_to(Path.cwd().resolve())
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,destination)
    save(root/'status.json',dict(state='preparing_contexts',pid=os.getpid(),goal_complete=False))
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    processor = joblib.load(root/'preprocessor.joblib') if args.resume and (root/'preprocessor.joblib').exists() else FullLifePreprocessor().fit(cells,plan['train'])
    if not (root/'preprocessor.joblib').exists():
        joblib.dump(processor,root/'preprocessor.joblib')
    contexts = full_life_contexts(cells,processor)
    for row in contexts:
        row.update(full_training_aux(cells[row['cell_id']],row['origin'],processor))
    coverage=[]
    for cid in plan['train']:
        rows=[r for r in contexts if r['cell_id']==cid]
        used=set()
        for row in rows:
            used.update(visible_window(cells[cid],row['origin'])['cycle'].tolist())
        coverage.append(dict(cell_id=cid,full_signal_observations=len(cells[cid]['cycle']),
            distinct_input_observations_used=len(used),origins=[r['origin'] for r in rows],
            first100_target_points=len(rows[0]['truth']),contexts=len(rows),
            context_weight_sum=sum(r['weight'] for r in rows),origin100_weight=rows[0]['weight']))
    save(root/'training_context_manifest.json',dict(cells=coverage,contexts=len(contexts),
        TRAIN_cells=len(plan['train']),all_future_targets_retained=True,validation_not_in_scaler=True,
        window='last100 real observations available at source origin; source coordinates and missing masks retained'))
    started=time.perf_counter()
    network=create_network(settings,processor,'query_rate_local',.1)
    bundle=dict(network=network,processor=processor,device='cuda',origins=[100],
        registry=load_registry(registry),physical_family=cells[plan['train'][0]]['physical_family'])
    card_file=root/'card_embeddings.pt'
    if args.resume and card_file.exists():
        if read(root/'card_cache_manifest.json')['sha256']!=fingerprint(card_file):
            raise ValueError('Cached frozen text embeddings changed')
        cached=torch.load(card_file,map_location='cpu',weights_only=True)
        if len(cached)!=len(contexts):
            raise ValueError('Cached card coverage differs')
        for row, embedding in zip(contexts,cached):
            row['card_embeddings']=embedding
    else:
        for number,row in enumerate(contexts,1):
            row['card_embeddings']=cards_embedding(bundle,visible_window(cells[row['cell_id']],row['origin']),row['origin'])
            if number%25==0 or number==len(contexts):
                print(json.dumps(dict(stage='cards',completed=number,contexts=len(contexts))),flush=True)
                save(root/'status.json',dict(state='encoding_cards',pid=os.getpid(),completed=number,contexts=len(contexts)))
        torch.save([r['card_embeddings'] for r in contexts],card_file)
        save(root/'card_cache_manifest.json',dict(sha256=fingerprint(card_file),
            contexts=[dict(cell_id=r['cell_id'],origin=r['origin']) for r in contexts],
            semantics='Unchanged frozen pretrained text backbone; no learned numeric path cached'))
    optimizer=repair_optimizer(network,settings,.03)
    policy=freeze_validation_policy(cells,plan['validation'],100)
    save(root/'validation_policy.json',policy)
    history=[];best=float('inf');train_best=float('inf');first_epoch=1;best_epoch=0;gradient_records=[]
    if args.resume and (root/'resume.pt').exists():
        state=torch.load(root/'resume.pt',map_location='cuda',weights_only=False)
        load_adapter(network,state['adapter']);optimizer.load_state_dict(state['optimizer'])
        history=state['history'];best=state['best'];train_best=state['train_best'];best_epoch=state['best_epoch']
        first_epoch=state['epoch']+1
        torch.set_rng_state(state['torch_rng'].cpu());torch.cuda.set_rng_state(state['cuda_rng'].cpu())
        gradient_records=read(root/'gradient_diagnostics.json')
    try:
        for epoch in range(first_epoch,args.epochs+1):
            network.train()
            factor=.05+.95*(1+math.cos(math.pi*(epoch-1)/max(1,args.epochs-1)))/2
            for group in optimizer.param_groups:
                group['lr']=settings['learning_rate']*factor*group.get('lr_scale',1.)
            order=np.random.default_rng(settings['seed']+epoch).permutation(len(contexts))
            losses=[]
            for begin in range(0,len(contexts),settings['batch_contexts']):
                rows=[contexts[i] for i in order[begin:begin+settings['batch_contexts']]]
                batch=pack_full_life(rows,'cuda')
                embeddings=torch.stack([r['card_embeddings'] for r in rows]).cuda()
                encoded=network.encode(batch['history'],batch['length'],embeddings)
                loss,_=objective(network,encoded,batch,rows,epoch,False,settings.get('point_huber_beta_pp',.1),
                    settings.get('drop_loss_weight',.5),settings.get('tail_loss_weight',.5))
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite full-life objective')
                optimizer.zero_grad(set_to_none=True);loss.backward()
                if begin==0:
                    norms={name:float(p.grad.norm()) for name,p in network.named_parameters() if p.grad is not None}
                    if not np.isfinite(list(norms.values())).all() or norms.get('numeric_projection.0.weight',0)<=0:
                        raise ValueError('No valid gradient through real pretrained LLM input path')
                    gradient_records.append(dict(epoch=epoch,gradient_norms=norms))
                    save(root/'gradient_diagnostics.json',gradient_records)
                torch.nn.utils.clip_grad_norm_([p for p in network.parameters() if p.requires_grad],5.)
                optimizer.step();losses.append(float(loss.detach()))
            row=dict(epoch=epoch,objective=float(np.mean(losses)),contexts=len(contexts),phase='full',
                     learning_rate=optimizer.param_groups[0]['lr'])
            if epoch%3==0 or epoch==args.epochs:
                metrics,arrays=evaluate_origin100(bundle,cells,plan['validation'])
                score=validation_score(metrics,policy)
                row['validation_score']=score
                row['validation_mape_percent']=float(np.mean([r['mape_percent'] for r in metrics.values()]))
                if score<best:
                    best,best_epoch=score,epoch
                    torch.save(adapter_state(network),root/'model.pt')
                    save(root/'selected_metrics.json',metrics)
                    np.savez_compressed(root/'selected_predictions.npz',**arrays)
                train_metrics,_=evaluate_origin100(bundle,cells,plan['train'])
                train_mae=float(np.mean([r['mae_pp'] for r in train_metrics.values()]))
                row['TRAIN_origin100_mae_pp']=train_mae
                if train_mae<train_best:
                    train_best=train_mae
                    torch.save(adapter_state(network),root/'model_TRAIN_best.pt')
                    save(root/'TRAIN_best_selection.json',dict(epoch=epoch,mae_pp=train_mae,
                        scope='TRAIN-only fitting diagnostic; not used for heldout demo'))
            history.append(row);save(root/'training_history.json',history)
            temporary=root/'resume.pt.part'
            torch.save(dict(adapter=adapter_state(network),optimizer=optimizer.state_dict(),history=history,
                epoch=epoch,best=best,train_best=train_best,best_epoch=best_epoch,
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),temporary)
            temporary.replace(root/'resume.pt')
            save(root/'status.json',dict(state='training',pid=os.getpid(),epoch=epoch,best_epoch=best_epoch,
                                         elapsed_seconds=time.perf_counter()-started))
            print(json.dumps(row),flush=True)
        load_adapter(network,torch.load(root/'model.pt',map_location='cpu',weights_only=True))
        metrics,arrays=evaluate_origin100(bundle,cells,plan['train'])
        save(root/'TRAIN_metrics.json',metrics)
        np.savez_compressed(root/'TRAIN_predictions.npz',**arrays)
        save(root/'result.json',dict(selected_epoch=best_epoch,parameters=network.parameter_counts(),
            train_cells=len(plan['train']),training_contexts=len(contexts),epochs=args.epochs,
            elapsed_seconds=time.perf_counter()-started,validation_score=best,
            model_sha256=fingerprint(root/'model.pt'),goal_complete=False))
        save(root/'status.json',dict(state='complete',pid=os.getpid(),selected_epoch=best_epoch))
        print('COMPLETE='+str(root.resolve()),flush=True)
    except BaseException:
        save(root/'failure.json',dict(traceback=traceback.format_exc()))
        save(root/'status.json',dict(state='failed',pid=os.getpid()))
        raise


if __name__=='__main__':
    main()
