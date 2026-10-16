"""TRAIN-only full-life reconstruction; no validation/test model selection."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import json
import math
from pathlib import Path
import shutil
import time
import traceback
import joblib
import numpy as np
import torch

from battery_smetimes.llm_full_life import load_full_life,read
from battery_smetimes.llm_reconstruction import ReconstructionProcessor,ReconstructionSMETimes,add_targets,pack,reconstruction_loss
from battery_smetimes.research_llm import adapter_state,load_adapter
from battery_smetimes.llm_curve_quality_v3 import curve_metrics_v3
from scripts.train_llm_full_life import save
from scripts.audit_llm_curve_repair import fingerprint


def evaluate(network,rows,tokens):
    network.eval();arrays={};metrics={};by_cell={}
    with torch.no_grad():
        for begin in range(0,len(rows),4):
            rr=rows[begin:begin+4];batch=pack(rr,'cuda')
            encoded=torch.stack(tokens[begin:begin+4]).cuda()
            prediction=network.reconstruct(encoded,batch).cpu().numpy()
            for row,p in zip(rr,prediction):
                by_cell.setdefault(row['cell_id'],[]).append((row['target_cycle'],row['truth'],p[:len(row['truth'])]))
    for cid,parts in by_cell.items():
        x,y,p=(np.concatenate([r[j] for r in parts]) for j in range(3))
        order=np.argsort(x);x,y,p=x[order],y[order],p[order]
        metrics[cid]=curve_metrics_v3(x,y,p)
        for name,value in (('cycle',x),('truth',y),('prediction',p)):arrays[cid+'__'+name]=value
    score=float(np.mean([r['mae_pp'] for r in metrics.values()]))
    return score,metrics,arrays


def encode_all(network,rows):
    network.eval();tokens=[]
    with torch.no_grad():
        for begin in range(0,len(rows),4):
            b=pack(rows[begin:begin+4],'cuda');encoded,_=network.encode(b['history'],b['length'],b['cards'])
            tokens.extend(t.detach().cpu() for t in encoded)
    return tokens


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--index',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--tiny',action='store_true');ap.add_argument('--joint-epochs',type=int,default=12)
    ap.add_argument('--head-epochs',type=int,default=200);ap.add_argument('--resume',action='store_true')
    args=ap.parse_args();root=args.out;root.mkdir(parents=True,exist_ok=True)
    index,cells,plan,_=load_full_life(args.index);ids=plan['train'][:3] if args.tiny else plan['train']
    cfg=dict(dataset=index['dataset'],seed=20260930,index=str(args.index.resolve()),index_sha256=fingerprint(args.index),
        train_ids=ids,joint_epochs=args.joint_epochs,head_epochs=args.head_epochs,tiny=args.tiny,
        checkpoint=index['base_settings']['checkpoint'],task='full-life TRAIN reconstruction, NOT forecasting',
        selection='TRAIN macro MAE only',validation_test_used=False)
    sources=[Path(__file__),Path('battery_smetimes/llm_reconstruction.py'),Path('battery_smetimes/research_llm.py'),
        Path('battery_smetimes/research_sequences.py'),Path('battery_smetimes/llm_full_life.py')]
    hashes={str(p.resolve()):fingerprint(p) for p in sources}
    if (root/'config.json').exists():
        if not args.resume or read(root/'config.json')!=cfg or read(root/'source_manifest.json')!=hashes:
            raise ValueError('Explicit unchanged resume required')
    else:
        save(root/'config.json',cfg);save(root/'source_manifest.json',hashes)
        for p in sources:
            dest=root/'source_snapshot'/p.resolve().relative_to(Path.cwd().resolve())
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    processor=ReconstructionProcessor().fit(cells,ids);joblib.dump(processor,root/'processor.joblib')
    rows=[];coverage=[]
    for cid in ids:
        cell=cells[cid];inputs=processor.inputs(cell)
        stripped={k:v for k,v in cell.items() if k not in ('soh','label_cycle','capacity','eol','lifetime')}
        again=processor.inputs(stripped)
        for a,b in zip(inputs,again):
            np.testing.assert_array_equal(a['history'],b['history']);assert a['card']==b['card']
        rr=add_targets(inputs,cell)
        for r in rr:r['cell_id']=cid
        rows.extend(rr)
        y=np.asarray(cell['soh']);valid=np.isfinite(y)&(y>0)
        actual=np.concatenate([r['target_cycle'] for r in rr]);expected=cell['label_cycle'][valid]
        np.testing.assert_array_equal(np.sort(actual),np.sort(expected))
        coverage.append(dict(cell_id=cid,raw_observations=len(cell['cycle']),labels=len(actual),windows=len(rr),
            signal_only_input_verified=True,largest_nearest_signal_gap=float(max(np.max(np.abs(r['target_cycle']-r['signal_cycles'][r['observation_index']])) for r in rr))))
    # Unbiased cell mass with constant minibatch normalization.
    for r in rows:r['weight']*=len(rows)/len(ids)
    save(root/'input_audit.json',dict(passed=True,cells=coverage,total_labels=sum(r['labels'] for r in coverage),
        source_index_sha256=cfg['index_sha256'],input_labels_removed_exact=True,validation_test_used=False))
    network=ReconstructionSMETimes(cfg['checkpoint'],processor.raw_width)
    joint_parameter_counts=network.parameter_counts()
    save(root/'status.json',dict(state='encoding_cards',pid=os.getpid(),contexts=len(rows)))
    cardfile=root/'cards.pt'
    if cardfile.exists():
        if fingerprint(cardfile)!=read(root/'cards_manifest.json')['sha256']:raise ValueError('Card cache changed')
        cards=torch.load(cardfile,map_location='cpu',weights_only=True)
    else:
        cards=[]
        for begin in range(0,len(rows),4):
            groups=[[r['card']] for r in rows[begin:begin+4]]
            values=network.encode_cards(groups)[:,0].cpu()
            cards.extend(v[None].expand(20,-1).clone() for v in values)
            if begin%100==0:print(json.dumps(dict(stage='cards',contexts=len(cards),total=len(rows))),flush=True)
        torch.save(cards,cardfile);save(root/'cards_manifest.json',dict(sha256=fingerprint(cardfile),count=len(cards)))
    if len(cards)!=len(rows):raise ValueError('Card coverage differs')
    for r,c in zip(rows,cards):r['cards']=c
    state=torch.load(root/'resume.pt',map_location='cpu',weights_only=False) if args.resume and (root/'resume.pt').exists() else None
    if state:load_adapter(network,state['adapter'])
    history=[] if not state else state['history'];gradient=[] if not (root/'gradients.json').exists() else read(root/'gradients.json')
    optimizer=torch.optim.AdamW([p for p in network.parameters() if p.requires_grad],lr=3e-4,weight_decay=0.)
    if state and state['phase']=='joint':optimizer.load_state_dict(state['optimizer'])
    started=time.perf_counter()
    try:
        start=1 if not state else state['epoch']+1 if state['phase']=='joint' else args.joint_epochs+1
        for epoch in range(start,args.joint_epochs+1):
            network.train();losses=[]
            for group in optimizer.param_groups:group['lr']=3e-4*(.2+.8*(1+math.cos(math.pi*(epoch-1)/args.joint_epochs))/2)
            order=np.random.default_rng(cfg['seed']+epoch).permutation(len(rows))
            for begin in range(0,len(rows),4):
                rr=[rows[i] for i in order[begin:begin+4]];batch=pack(rr,'cuda')
                tokens,_=network.encode(batch['history'],batch['length'],batch['cards'])
                p=network.reconstruct(tokens,batch);loss=reconstruction_loss(p,batch)
                if not torch.isfinite(loss):raise ValueError('Nonfinite reconstruction objective')
                optimizer.zero_grad(set_to_none=True);loss.backward()
                if begin==0:
                    norms={n:float(p.grad.norm()) for n,p in network.named_parameters() if p.grad is not None}
                    if not all(np.isfinite(v) for v in norms.values()) or norms.get('numeric_projection.0.weight',0)<=0:
                        raise ValueError('Real LLM numeric input gradient absent')
                    gradient.append(dict(epoch=epoch,norms=norms));save(root/'gradients.json',gradient)
                torch.nn.utils.clip_grad_norm_([p for p in network.parameters() if p.requires_grad],10.)
                optimizer.step();losses.append(float(loss.detach()))
            record=dict(phase='joint',epoch=epoch,objective=float(np.mean(losses)))
            history.append(record);save(root/'history.json',history)
            torch.save(dict(phase='joint',epoch=epoch,adapter=adapter_state(network),optimizer=optimizer.state_dict(),history=history),root/'resume.pt')
            save(root/'status.json',dict(state='joint_training',pid=os.getpid(),**record));print(json.dumps(record),flush=True)
        # Cache only after the learned encoder has finished updating.
        tokenfile=root/'encoded_tokens.pt'
        if state and state['phase']=='head' and tokenfile.exists():
            cache=read(root/'encoded_manifest.json')
            if fingerprint(tokenfile)!=cache['sha256']:raise ValueError('Learned feature cache changed')
            tokens=torch.load(tokenfile,map_location='cpu',weights_only=True)
        else:
            tokens=encode_all(network,rows);torch.save(tokens,tokenfile)
            torch.save(adapter_state(network),root/'joint_model.pt')
            save(root/'encoded_manifest.json',dict(sha256=fingerprint(tokenfile),encoder_adapter_sha256=fingerprint(root/'joint_model.pt'),
                semantics='Label-free learned numeric/fusion/LLM/expert outputs; encoder frozen for head refinement'))
        for name,p in network.named_parameters():p.requires_grad=name.startswith(('reconstruction_','slot_embedding.'))
        head=torch.optim.AdamW([p for p in network.parameters() if p.requires_grad],lr=5e-4,weight_decay=0.)
        best=float('inf') if not state or state['phase']!='head' else state['best'];stale=0
        first=1 if not state or state['phase']!='head' else state['epoch']+1
        if state and state['phase']=='head':head.load_state_dict(state['optimizer'])
        for epoch in range(first,args.head_epochs+1):
            network.train();losses=[];order=np.random.default_rng(cfg['seed']+10000+epoch).permutation(len(rows))
            for begin in range(0,len(rows),16):
                indices=order[begin:begin+16];rr=[rows[i] for i in indices];batch=pack(rr,'cuda')
                encoded=torch.stack([tokens[i] for i in indices]).cuda()
                loss=reconstruction_loss(network.reconstruct(encoded,batch),batch)
                if not torch.isfinite(loss):raise ValueError('Nonfinite head objective')
                head.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_([p for p in network.parameters() if p.requires_grad],10.)
                head.step();losses.append(float(loss.detach()))
            record=dict(phase='head',epoch=epoch,objective=float(np.mean(losses)))
            if epoch%5==0 or epoch==1 or epoch==args.head_epochs:
                score,metrics,arrays=evaluate(network,rows,tokens);record['TRAIN_mae_pp']=score
                record['TRAIN_mape_percent']=float(np.mean([m['mape_percent'] for m in metrics.values()]))
                if score<best:
                    best=score;stale=0;torch.save(adapter_state(network),root/'model.pt')
                    save(root/'metrics.json',metrics);np.savez_compressed(root/'predictions.npz',**arrays)
                    save(root/'selection.json',dict(phase='head',epoch=epoch,TRAIN_mae_pp=score,scope='TRAIN-only reconstruction'))
                else:stale+=5
                print(json.dumps(record),flush=True)
            history.append(record);save(root/'history.json',history)
            save(root/'status.json',dict(state='head_refinement',pid=os.getpid(),best_TRAIN_mae_pp=best,**record))
            if epoch%5==0 or epoch==1:
                torch.save(dict(phase='head',epoch=epoch,adapter=adapter_state(network),optimizer=head.state_dict(),history=history,best=best),root/'resume.pt')
            if stale>=30:break
        # Real checkpoint reload, then recompute frozen LLM outputs; cached head
        # refinement cannot silently substitute for a runnable model.
        load_adapter(network,torch.load(root/'model.pt',map_location='cpu',weights_only=True))
        fresh=encode_all(network,rows);score,metrics,arrays=evaluate(network,rows,fresh)
        with np.load(root/'predictions.npz',allow_pickle=False) as original:
            replay=max(float(np.max(np.abs(arrays[k]-original[k]))) for k in arrays if k.endswith('__prediction'))
        if replay>1e-5:raise ValueError('Actual LLM replay differs from cached refinement: '+str(replay))
        save(root/'replay_audit.json',dict(passed=True,max_error_soh=replay,actual_LLM_recomputed=True,
            model_sha256=fingerprint(root/'model.pt'),label_inputs_absent=True))
        save(root/'result.json',dict(dataset=cfg['dataset'],TRAIN_cells=len(ids),contexts=len(rows),
            TRAIN_mae_pp=score,TRAIN_mape_percent=float(np.mean([m['mape_percent'] for m in metrics.values()])),
            selected=read(root/'selection.json'),joint_parameters=joint_parameter_counts,
            parameters_at_head_refinement=network.parameter_counts(),joint_gradient_epochs=len(gradient),
            model_sha256=fingerprint(root/'model.pt'),elapsed_seconds=time.perf_counter()-started,
            reconstruction_only=True,predictive_ability_proven=False))
        save(root/'status.json',dict(state='complete',pid=os.getpid(),TRAIN_mae_pp=score));print('COMPLETE='+str(root),flush=True)
    except BaseException:
        save(root/'failure.json',dict(traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
