"""Resumable P1/P2 LLM+SMETimes experiments with TRAIN-only diagnostics.

P1 never evaluates validation/test cells or fits free target-cell parameters.
P2 keeps the frozen validation split and saves full TRAIN/validation results.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import shutil
import time
import traceback

import joblib
import numpy as np
import torch
from torch.nn import functional as F

from battery_smetimes.research_llm import FrozenSMETimesForecaster,adapter_state,load_adapter,attach_cards,adapter_optimizer
from battery_smetimes.llm_query_rate import QueryRateForecaster
from battery_smetimes.llm_local_recovery import LocalRecoveryQueryForecaster
from battery_smetimes.llm_local_refinement import local_refinement_optimizer,local_refinement_step
from battery_smetimes.research_sequences import HistoryPreprocessor,training_contexts,pack_contexts
from battery_smetimes.llm_curve_objective import curve_metrics,training_origin_target
from battery_smetimes.physical_types import load_registry
from scripts.audit_llm_curve_repair import load_frozen,fingerprint
from scripts.prepare_same_type_forecast import save_json
from scripts.train_llm_soh_mvp import evaluate
from battery_smetimes.llm_reporting import finite_metric


def create_network(settings,processor,architecture,dropout):
    common=dict(seed=settings['seed'],dropout=dropout,device='cuda',history_slots=processor.history_slots,
                normalize_router=settings.get('normalize_router',False),numeric_lora_rank=settings.get('numeric_lora_rank',0))
    if architecture in ('query_rate','query_rate_local'):
        factory=LocalRecoveryQueryForecaster if architecture=='query_rate_local' else QueryRateForecaster
        extra={'local_recovery_spacing_cycles':settings['local_recovery_spacing_cycles'],
               'local_recovery_max_soh':settings['local_recovery_max_soh'],
               'local_readout':settings.get('local_readout','query')} if architecture=='query_rate_local' else {}
        net=factory(settings['checkpoint'],2*processor.raw_width+1,
            rate_knots=settings['rate_knots_horizon_100cycles'],rate_width=settings['rate_width'],
            learned_node_queries=settings.get('learned_node_queries',False),
            recovery_max_soh=settings['recovery_max_soh'],**extra,**common)
    elif architecture=='integrated_rate':
        net=FrozenSMETimesForecaster(settings['checkpoint'],2*processor.raw_width+1,
            output_head='integrated_rate',pooling='mean',**common)
    else:raise ValueError('unknown repair architecture')
    net.bind_query_scaler(processor.query_scale.scaler)
    return net


def repair_optimizer(network,settings,weight_decay):
    scale=settings.get('local_decoder_lr_scale',1.)
    if not math.isfinite(scale) or scale<=0:raise ValueError('invalid local decoder LR scale')
    if scale==1.:return adapter_optimizer(network,settings['learning_rate'],weight_decay)
    local=[];other=[]
    for name,parameter in network.named_parameters():
        if parameter.requires_grad:
            (local if name.startswith('local_head.') or name=='local_query' else other).append(parameter)
    if not local:raise ValueError('local LR scaling requires the local LLM decoder')
    return torch.optim.AdamW([{'params':other,'lr_scale':1.,'lr':settings['learning_rate']},
        {'params':local,'lr_scale':scale,'lr':settings['learning_rate']*scale}],
        lr=settings['learning_rate'],weight_decay=weight_decay)


def full_training_aux(cell,origin,processor):
    take=(cell['label_cycle']>origin)&np.isfinite(cell['soh'])&(cell['soh']>0)
    x,y=cell['label_cycle'][take],cell['soh'][take]
    span=x[-1]-x[0];pairs=[]
    for j in range(3):
        region=(x>=x[0]+j*span/3)&(x<=x[0]+(j+1)*span/3)
        xx,yy=x[region],y[region]
        if len(xx)<3:continue
        width=max(1.,min(5.,float((xx[-1]-xx[0])/4)))
        a,b=xx<=xx[0]+width,xx>=xx[-1]-width
        pairs.append((float(xx[a].mean()),float(xx[b].mean()),float(np.median(yy[a])-np.median(yy[b])),'stage'))
    window=max(20.,span*.1)
    for begin in np.linspace(x[0],x[-1]-window,9) if span>=window else []:
        a=(x>=begin)&(x<=begin+window/4);b=(x>=begin+window*.75)&(x<=begin+window)
        if a.sum()<2 or b.sum()<2:continue
        pairs.append((float(x[a].mean()),float(x[b].mean()),float(np.median(y[a])-np.median(y[b])),'window'))
    query=np.zeros((24,3),dtype=np.float32);target=np.zeros(12,dtype=np.float32);valid=np.zeros(12,dtype=np.float32)
    horizon=np.zeros(12,dtype=np.float32);stage=np.zeros(12,dtype=np.float32)
    for j,(left,right,drop,kind) in enumerate(pairs[:12]):
        query[2*j:2*j+2]=processor.future_query(origin,np.asarray([left,right]))
        target[j]=drop;valid[j]=1;horizon[j]=right-origin;stage[j]=kind=='stage'
    start,start_valid=training_origin_target(cell,origin)
    return {'drop_query':query,'drop_truth':target,'drop_valid':valid,'drop_horizon':horizon,'drop_is_stage':stage,
        'start_truth':start,'start_valid':start_valid,
        'start_query':processor.query_scale.transform(np.asarray([[origin/100,0.,0.]]))}


def objective(network,encoding,batch,rows,epoch,curriculum,point_huber_beta_pp=1.):
    prediction=network.decode(encoding,batch['conditions'],batch['query'])
    query_physical=batch['query']*network.query_scale+network.query_mean
    phase='short' if curriculum and epoch<=25 else 'full'
    point_mask=batch['valid']*((query_physical[...,1]<=1) if phase=='short' else 1)
    # Entire contexts with no short-horizon label retain origin supervision;
    # no future labels are interpolated to manufacture missing early targets.
    residual=100*(prediction-batch['truth'])
    point=(F.smooth_l1_loss(residual,torch.zeros_like(residual),reduction='none',beta=point_huber_beta_pp)*point_mask).sum(1)/point_mask.sum(1).clamp_min(1)
    counts=batch['valid'].sum(1)
    physical_cycles=query_physical[...,0]*100
    first=torch.where(batch['valid'].bool(),physical_cycles,torch.inf).min(1).values
    last=torch.where(batch['valid'].bool(),physical_cycles,-torch.inf).max(1).values
    tail_mask=batch['valid']*(physical_cycles>=first[:,None]+.8*(last-first)[:,None])
    tail=(F.smooth_l1_loss(residual,torch.zeros_like(residual),reduction='none',beta=point_huber_beta_pp)*tail_mask).sum(1)/tail_mask.sum(1).clamp_min(1)
    start_query=torch.as_tensor(np.asarray([r['start_query'] for r in rows]),device='cuda')
    start_p=network.decode(encoding,batch['conditions'],start_query)[:,0]
    start_y=torch.as_tensor([r['start_truth'] for r in rows],device='cuda')
    start_mask=torch.as_tensor([r['start_valid'] for r in rows],device='cuda')
    start_error=100*(start_p-start_y)
    start=F.smooth_l1_loss(start_error,torch.zeros_like(start_error),reduction='none')*start_mask
    drop_query=torch.as_tensor(np.asarray([r['drop_query'] for r in rows]),device='cuda')
    drop_p=network.decode(encoding,batch['conditions'],drop_query).reshape(len(rows),12,2)
    drop_y=torch.as_tensor(np.asarray([r['drop_truth'] for r in rows]),device='cuda')
    drop_valid=torch.as_tensor(np.asarray([r['drop_valid'] for r in rows]),device='cuda')
    if phase=='short':
        drop_valid=drop_valid*(torch.as_tensor(np.asarray([r['drop_horizon'] for r in rows]),device='cuda')<=100)
    drop_error=100*(drop_p[:,:,0]-drop_p[:,:,1]-drop_y)
    drops=(F.smooth_l1_loss(drop_error,torch.zeros_like(drop_error),reduction='none')*drop_valid).sum(1)/drop_valid.sum(1).clamp_min(1)
    individual=point+start+.5*drops+(.5*tail if phase=='full' else 0)
    return (individual*batch['weight']).mean(),{'point_pp_huber':point.mean(),'start_pp_huber':start.mean(),
        'drops_pp_huber':drops.mean(),'tail_pp_huber':tail.mean(),'phase':phase}


def summary(metrics):
    values=list(metrics.values())
    keys=('mae_pp','tail_mae_pp','stage_drop_max_pp','start_error_pp','endpoint_error_pp')
    numeric={key:[v[key] for v in values if finite_metric(v,key)] for key in keys}
    return {'cells':len(values),'legacy_joint_passes':sum(v.get('joint_pass',False) for v in values),
        'metric_cells':{key:len(v) for key,v in numeric.items()},
        **{key:float(np.mean(v)) if v else None for key,v in numeric.items()},
        **{'max_'+key:float(max(numeric[key])) if numeric[key] else None for key in
           ('mae_pp','tail_mae_pp','stage_drop_max_pp')}}


def tiny_gate(metrics,settings):
    return {cid:all(finite_metric(row,k) and row[k]<=limit for k,limit in settings['tiny_gates_pp'].items()) for cid,row in metrics.items()}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--p0');ap.add_argument('--settings',type=Path);ap.add_argument('--full-training-authorization',type=Path);ap.add_argument('--architecture',choices=['integrated_rate','query_rate','query_rate_local'],required=True)
    ap.add_argument('--mode',choices=['tiny','development'],default='tiny');ap.add_argument('--resume',type=Path)
    ap.add_argument('--p1-evidence',type=Path,help='Passing TRAIN tiny checkpoint required before development training')
    ap.add_argument('--epochs',type=int);ap.add_argument('--learning-rate',type=float);ap.add_argument('--no-curriculum',action='store_true')
    args=ap.parse_args()
    p0=Path(args.p0 or json.loads(Path('experiments/llm_curve_repair/latest_P0.json').read_text())['folder'])
    settings=json.loads((args.settings or p0/'config.json').read_text(encoding='utf-8'))
    if args.epochs: settings['max_tiny_epochs']=args.epochs
    if args.learning_rate: settings['learning_rate']=args.learning_rate
    if args.mode=='tiny' and not 1<=settings['max_tiny_epochs']<=300:
        raise ValueError('P1 diagnostics require 1..300 epochs; do not bypass the frozen diagnostic budget')
    if not math.isfinite(settings.get('point_huber_beta_pp',1.)) or settings.get('point_huber_beta_pp',1.)<=0:
        raise ValueError('point Huber transition must be positive finite pp')
    refine_after=settings.get('local_refine_after_epoch')
    if refine_after is not None and (args.mode!='tiny' or args.architecture!='query_rate_local' or not isinstance(refine_after,int) or not 1<=refine_after<settings['max_tiny_epochs']):
        raise ValueError('local refinement requires a tiny local-decoder run and a phase boundary within the 300-epoch budget')
    if args.mode=='development':
        if args.p1_evidence is None:raise ValueError('P1 has not been certified: --p1-evidence required before development training')
        evidence=json.loads((args.p1_evidence/'result.json').read_text())
        previous=json.loads((args.p1_evidence/'config.json').read_text())
        authorization=json.loads(args.full_training_authorization.read_text(encoding='utf-8')) if args.full_training_authorization else None
        if previous.get('mode')!='tiny':raise ValueError('P1 evidence must be a real tiny run')
        if not evidence.get('tiny_all_pass') and not (authorization and authorization.get('advance_to_full_training') is True):raise ValueError('P1 strict tiny gate failed; explicit recorded user direction required to advance')
        if evidence['model_sha256']!=fingerprint(args.p1_evidence/'model.pt'):raise ValueError('P1 checkpoint changed')
        if not authorization and (previous['fold']!=settings['fold'] or previous['origin']!=settings['origin']):raise ValueError('P1 evidence uses another task')
        if previous.get('mit_current_units','legacy_stored_v1')!=settings.get('mit_current_units','legacy_stored_v1'):raise ValueError('P1 evidence uses another signal-unit contract')
        if not json.loads((args.p1_evidence/'repair_verification.json').read_text()).get('passed'):raise ValueError('P1 replay/isolation not verified')
    manifest,cells,plan,registry_path,prepared=load_frozen(settings)
    train_ids=settings['tiny_train_ids'] if args.mode=='tiny' else plan['train']
    if not set(train_ids)<=set(plan['train']):raise ValueError('tiny IDs outside frozen TRAIN')
    if settings.get('full_train_targets',False):
        settings['future_targets']=max(int(((cells[k]['label_cycle']>settings['origin'])&np.isfinite(cells[k]['soh'])&(cells[k]['soh']>0)).sum()) for k in train_ids)
    root=args.resume or Path('experiments/llm_curve_repair')/(args.mode+'_'+args.architecture+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    if not args.resume:root.mkdir(parents=True,exist_ok=False)
    print('RUN='+str(root.resolve()),flush=True)
    dropout=0. if args.mode=='tiny' else .1
    decay=0. if args.mode=='tiny' else .03
    curriculum=not args.no_curriculum
    config={**settings,'p0':str(p0.resolve()),'architecture':args.architecture,'mode':args.mode,'train_ids':train_ids,
            'dropout':dropout,'weight_decay':decay,'curriculum':curriculum,'evaluation_identity':'TRAIN in-sample diagnostic' if args.mode=='tiny' else 'development validation',
            'heldout_evaluated':False,'p1_evidence':str(args.p1_evidence) if args.p1_evidence else None,
            'full_training_authorization':authorization if args.mode=='development' else None,
            'p1_strict_gate_passed':bool(evidence.get('tiny_all_pass')) if args.mode=='development' else None,
            'clip_gradient_norm':5.,'loss':'pp Huber point/start/dense-label-window drops/tail; no teacher or independent predictor'}
    if args.resume:
        previous=json.loads((root/'config.json').read_text(encoding='utf-8'))
        for key in ('architecture','mode','train_ids','seed','origin','learning_rate','curriculum','rate_knots_horizon_100cycles'):
            if config[key]!=previous[key]:raise ValueError('resume config drift: '+key)
        for key,default in [('learning_rate_schedule','constant'),('mit_current_units','legacy_stored_v1')]:
            if config.get(key,default)!=previous.get(key,default):raise ValueError('resume config drift: '+key)
        if config.get('learned_node_queries',False)!=previous.get('learned_node_queries',False):raise ValueError('resume node-query drift')
        for key,default in [('normalize_router',False),('full_train_targets',False),
                            ('local_recovery_spacing_cycles',None),('local_recovery_max_soh',None),('point_huber_beta_pp',1.),('local_readout','query'),('local_decoder_lr_scale',1.),('local_refine_after_epoch',None)]:
            if config.get(key,default)!=previous.get(key,default):raise ValueError('resume config drift: '+key)
    save_json(root/'config.json',config)
    save_json(root/'status.json',{'state':'starting','goal_complete':False,'mode':args.mode,'pid':os.getpid()})
    sources=[Path(__file__),Path('scripts/audit_llm_curve_repair.py'),Path('battery_smetimes/llm_query_rate.py'),
        Path('battery_smetimes/research_llm.py'),Path('battery_smetimes/research_sequences.py'),Path('battery_smetimes/llm_local_recovery.py'),Path('battery_smetimes/llm_local_refinement.py'),Path('battery_smetimes/llm_external_schema.py'),Path('scripts/prepare_llm_external_datasets.py'),
        Path('battery_smetimes/llm_curve_objective.py'),Path('battery_smetimes/mit_units.py'),
        Path('scripts/train_llm_soh_mvp.py'),Path('configs/llm_curve_repair_v1.json')]
    for file in sources:
        relative=file.resolve().relative_to(Path.cwd().resolve())
        target=root/'source_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        if not args.resume:shutil.copy2(file,target)
        elif fingerprint(file)!=fingerprint(target):raise ValueError('source drift in resume: '+str(file))
    if args.settings:
        shutil.copy2(args.settings,root/'settings_input.json')
    if args.full_training_authorization:
        shutil.copy2(args.full_training_authorization,root/'full_training_authorization.json')
    if not args.resume:save_json(root/'source_manifest.json',{'sha256':{str(p):fingerprint(p) for p in sources},'P0_manifest_sha256':fingerprint(p0/'source_manifest.json')})
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    processor=joblib.load(root/'preprocessor.joblib') if args.resume else HistoryPreprocessor().fit(cells,train_ids,[settings['origin']],history_slots=settings['history_slots'])
    if processor.train_ids!=train_ids:raise ValueError('scaler fit identity drift')
    joblib.dump(processor,root/'preprocessor.joblib')
    contexts=training_contexts(cells,train_ids,processor,settings['future_targets'])
    for row in contexts:
        row.update(full_training_aux(cells[row['cell_id']],row['origin'],processor))
    started=time.perf_counter();network=create_network(settings,processor,args.architecture,dropout)
    bundle={'network':network,'processor':processor,'device':'cuda','origins':[settings['origin']],
            'registry':load_registry(registry_path),'physical_family':cells[train_ids[0]]['physical_family']}
    attach_cards(bundle,contexts,cells)
    optimizer=repair_optimizer(network,settings,decay)
    local_optimizer=local_refinement_optimizer(network) if refine_after is not None else None
    fixed_encoding=None;refinement_batch=None
    history=[];first_epoch=1;best_score=None;best_state=None;best_epoch=0
    if args.resume:
        resume=torch.load(root/'resume.pt',map_location='cuda',weights_only=False)
        load_adapter(network,resume['adapter']);optimizer.load_state_dict(resume['optimizer'])
        if local_optimizer is not None and resume.get('local_optimizer') is not None:local_optimizer.load_state_dict(resume['local_optimizer'])
        torch.set_rng_state(resume['torch_rng'].cpu());torch.cuda.set_rng_state(resume['cuda_rng'].cpu())
        history=resume['history'];first_epoch=resume['epoch']+1;best_score=resume['best_score'];best_epoch=resume['best_epoch']
        best_state=torch.load(root/'model.pt',map_location='cpu',weights_only=True) if (root/'model.pt').exists() else None
    diagnostics=[];bad_evaluations=0
    try:
        for epoch in range(first_epoch,settings['max_tiny_epochs']+1):
            schedule=settings.get('learning_rate_schedule','constant')
            if schedule not in ('constant','cosine'):raise ValueError('unknown LR schedule')
            factor=1. if schedule=='constant' else .05+.95*(1+math.cos(math.pi*(epoch-1)/max(1,settings['max_tiny_epochs']-1)))/2
            for group in optimizer.param_groups:group['lr']=settings['learning_rate']*factor*group.get('lr_scale',1.)
            network.train();order=np.random.default_rng(settings['seed']+epoch).permutation(len(contexts));losses=[];terms=[]
            refining=refine_after is not None and epoch>refine_after
            closure_evaluations=0
            if refining:
                # Cache only visible history representations, after freezing all
                # parameters outside the local head. All 12 TRAIN cells retain
                # the same ordinary objective and count-normalized supervision.
                network.eval()
                if fixed_encoding is None:
                    chunks=[]
                    with torch.no_grad():
                        for begin in range(0,len(contexts),settings['batch_contexts']):
                            rows=contexts[begin:begin+settings['batch_contexts']]
                            visible=pack_contexts(rows,'cuda')
                            embedding=torch.stack([r['card_embeddings'] for r in rows]).cuda()
                            chunks.append(network.encode(visible['history'],visible['length'],embedding))
                    fixed_encoding=tuple(torch.cat([chunk[i] for chunk in chunks]).detach() for i in range(2))
                    refinement_batch=pack_contexts(contexts,'cuda')
                def supervised_loss():
                    return objective(network,fixed_encoding,refinement_batch,contexts,epoch,curriculum,settings.get('point_huber_beta_pp',1.))[0]
                value,closure_evaluations=local_refinement_step(network,local_optimizer,supervised_loss)
                with torch.no_grad():
                    _,components=objective(network,fixed_encoding,refinement_batch,contexts,epoch,curriculum,settings.get('point_huber_beta_pp',1.))
                losses.append(value);terms.append({k:float(v) if torch.is_tensor(v) else v for k,v in components.items()})
                if epoch%25==0:
                    norms={k:float(p.grad.norm()) for k,p in network.named_parameters() if p.grad is not None}
                    if not norms or any(not (k.startswith('local_head.') or k=='local_query') for k in norms):raise ValueError('refinement changed frozen optimization scope')
                    diagnostics.append({'epoch':epoch,'phase':'local_only_lbfgs','gradient_norms':norms,'closure_evaluations':closure_evaluations,'visible_encoding_detached':True})
                    save_json(root/'gradient_diagnostics.json',diagnostics)
            else:
                for begin in range(0,len(order),settings['batch_contexts']):
                    rows=[contexts[i] for i in order[begin:begin+settings['batch_contexts']]];batch=pack_contexts(rows,'cuda')
                    embedding=torch.stack([r['card_embeddings'] for r in rows]).cuda()
                    encoded=network.encode(batch['history'],batch['length'],embedding)
                    forecast,components=objective(network,encoded,batch,rows,epoch,curriculum,settings.get('point_huber_beta_pp',1.))
                    # Tiny tests intentionally remove expert regularization. Full
                    # warmup keeps the additional gradient below 10% of forecast.
                    balance_coefficient=0.
                    shared=network.numeric_projection[0].weight
                    if args.mode=='development' and epoch>25:
                        gf=torch.autograd.grad(forecast,shared,retain_graph=True)[0].norm()
                        gb=torch.autograd.grad(network.balance_loss,shared,retain_graph=True)[0].norm()
                        balance_coefficient=min(.001,float(.1*gf/(gb+1e-12)))
                    loss=forecast+balance_coefficient*network.balance_loss
                    if not torch.isfinite(loss):raise ValueError('nonfinite objective')
                    optimizer.zero_grad(set_to_none=True);loss.backward()
                    if begin==0 and (epoch==1 or epoch%25==0):
                        norms={k:float(p.grad.norm()) for k,p in network.named_parameters() if p.requires_grad and p.grad is not None}
                        if not all(np.isfinite(v) for v in norms.values()) or norms.get('numeric_projection.0.weight',0)<=0:
                            raise ValueError('invalid gradient through LLM numeric path')
                        if any(p.grad is not None for p in network.backbone.parameters() if not p.requires_grad):raise ValueError('frozen backbone gradient')
                        values=encoded[0] if isinstance(encoded,tuple) else encoded
                        flat=values.detach().float().reshape(-1,values.shape[-1]);singular=torch.linalg.svdvals(flat-flat.mean(0))
                        probability=singular/singular.sum().clamp_min(1e-12)
                        diagnostics.append({'epoch':epoch,'gradient_norms':norms,'balance_coefficient':balance_coefficient,
                            'encoding_variance':float(flat.var()),'effective_encoding_rank':float(torch.exp(-(probability*probability.clamp_min(1e-12).log()).sum()))})
                        save_json(root/'gradient_diagnostics.json',diagnostics)
                    torch.nn.utils.clip_grad_norm_((p for p in network.parameters() if p.requires_grad),5.)
                    optimizer.step();losses.append(float(loss.detach()));terms.append({k:float(v.detach()) if torch.is_tensor(v) else v for k,v in components.items()})
            row={'epoch':epoch,'learning_rate':optimizer.param_groups[0]['lr'],'objective':float(np.mean(losses)), 'phase':terms[0]['phase'],
                'learning_rates_by_group':[group['lr'] for group in (local_optimizer if refining else optimizer).param_groups],
                'optimizer_phase':'local_only_lbfgs' if refining else 'joint_adamw','closure_evaluations':closure_evaluations,
                'terms':{k:float(np.mean([r[k] for r in terms])) for k in terms[0] if k!='phase'}}
            passed=False
            if epoch%settings['validation_every']==0 or epoch==settings['max_tiny_epochs']:
                ids=train_ids if args.mode=='tiny' else plan['validation']
                report,arrays,metrics,score=evaluate(bundle,cells,ids)
                summed=summary(metrics);row['evaluation']=summed
                gates=tiny_gate(metrics,settings) if args.mode=='tiny' else {}
                passed=bool(gates and all(gates.values()) and len(gates)==len(train_ids))
                # Tiny selection is diagnostic fitting only; stage dominance
                # is not a heldout quality gate or per-cell inference routing.
                if args.mode=='tiny':
                    tiny_score=max(summed['max_'+k]/v if summed['max_'+k] is not None else float('inf') for k,v in settings['tiny_gates_pp'].items())
                    selection=(tiny_score,sum(summed[k] if summed[k] is not None else float('inf') for k in settings['tiny_gates_pp']))
                else:selection=score
                if best_score is None or selection<tuple(best_score):
                    best_score=selection;best_state=adapter_state(network);best_epoch=epoch;bad_evaluations=0
                    torch.save(best_state,root/'model.pt');save_json(root/'selected_metrics.json',metrics)
                    np.savez_compressed(root/'selected_predictions.npz',**arrays)
                    save_json(root/'selection.json',{'epoch':epoch,'score':selection,'scope':config['evaluation_identity'],
                        'tiny_per_cell_pass':gates,'tiny_all_pass':passed,'summary':summed})
                else:bad_evaluations+=1
                print(json.dumps({'epoch':epoch,'phase':row['phase'],'objective':row['objective'],'evaluation':summed,'tiny_all_pass':passed}),flush=True)
            history.append(row);save_json(root/'training_history.json',history)
            if epoch%10==0 or passed or epoch==settings['max_tiny_epochs']:
                temporary=root/'resume.pt.part'
                torch.save({'adapter':adapter_state(network),'optimizer':optimizer.state_dict(),'epoch':epoch,
                    'local_optimizer':local_optimizer.state_dict() if local_optimizer is not None else None,
                    'history':history,'best_score':best_score,'best_epoch':best_epoch,'torch_rng':torch.get_rng_state(),
                    'cuda_rng':torch.cuda.get_rng_state()},temporary);temporary.replace(root/'resume.pt')
                save_json(root/'status.json',{'state':'training','epoch':epoch,'best_epoch':best_epoch,
                    'pid':os.getpid(),'elapsed_seconds':time.perf_counter()-started,'goal_complete':False})
            if passed or (args.mode=='development' and bad_evaluations>=settings.get('validation_patience',8)):break
        if best_state is None:raise ValueError('no selected checkpoint')
        load_adapter(network,best_state)
        _,arrays,metrics,_=evaluate(bundle,cells,train_ids)
        save_json(root/'TRAIN_metrics.json',metrics);np.savez_compressed(root/'TRAIN_predictions.npz',**arrays)
        gates=tiny_gate(metrics,settings)
        save_json(root/'result.json',{'selected_epoch':best_epoch,'parameters':network.parameter_counts(),
            'TRAIN_summary':summary(metrics),'tiny_per_cell_pass':gates,'tiny_all_pass':all(gates.values()),
            'cuda_peak_allocated_mb':torch.cuda.max_memory_allocated()/2**20,'elapsed_seconds':time.perf_counter()-started,
            'model_sha256':fingerprint(root/'model.pt'),'not_generalization_accuracy':args.mode=='tiny','goal_complete':False})
        save_json(root/'status.json',{'state':'complete','training_complete':True,'mode':args.mode,'pid':os.getpid(),
            'tiny_all_pass':all(gates.values()),'best_epoch':best_epoch,'goal_complete':False})
        save_json(Path('experiments/llm_curve_repair/latest_'+args.mode+'_'+args.architecture+'.json'),{'folder':str(root.resolve())})
        print('COMPLETE='+str(root.resolve()),flush=True)
    except BaseException:
        save_json(root/'failure.json',{'traceback':traceback.format_exc(),'goal_complete':False})
        save_json(root/'status.json',{'state':'failed','goal_complete':False,'pid':os.getpid()});raise


if __name__=='__main__':main()
