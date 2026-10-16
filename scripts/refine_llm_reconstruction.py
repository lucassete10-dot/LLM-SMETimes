"""Deterministic full-TRAIN shared-head refinement after the serial first pass."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import torch
import joblib
from battery_smetimes.llm_full_life import read,load_full_life
from battery_smetimes.llm_reconstruction import ReconstructionSMETimes,add_targets,pack,reconstruction_loss
from battery_smetimes.research_llm import load_adapter,adapter_state
from scripts.train_llm_reconstruction import evaluate,encode_all
from scripts.train_llm_full_life import save
from scripts.audit_llm_curve_repair import fingerprint


def fit(parent,out):
    cfg=read(parent/'config.json');_,cells,plan,_=load_full_life(cfg['index'])
    processor=joblib.load(parent/'processor.joblib');rows=[]
    for cid in cfg['train_ids']:
        rr=add_targets(processor.inputs(cells[cid]),cells[cid])
        for r in rr:r['cell_id']=cid
        rows.extend(rr)
    for r in rows:r['weight']*=len(rows)/len(plan['train'])
    card_manifest=read(parent/'cards_manifest.json');encoded_manifest=read(parent/'encoded_manifest.json')
    if fingerprint(parent/'cards.pt')!=card_manifest['sha256'] or fingerprint(parent/'encoded_tokens.pt')!=encoded_manifest['sha256']:
        raise ValueError('Parent cache changed')
    cards=torch.load(parent/'cards.pt',map_location='cpu',weights_only=True)
    tokens=torch.load(parent/'encoded_tokens.pt',map_location='cpu',weights_only=True)
    for r,c in zip(rows,cards):r['cards']=c
    network=ReconstructionSMETimes(cfg['checkpoint'],processor.raw_width)
    load_adapter(network,torch.load(parent/'model.pt',map_location='cpu',weights_only=True))
    for name,p in network.named_parameters():p.requires_grad=name.startswith(('reconstruction_','slot_embedding.'))
    optimizer=torch.optim.LBFGS([p for p in network.parameters() if p.requires_grad],lr=1.,max_iter=1,
        max_eval=8,history_size=10,line_search_fn='strong_wolfe',tolerance_grad=1e-7,tolerance_change=1e-10)
    best,metrics,arrays=evaluate(network,rows,tokens);baseline=best
    np.testing.assert_allclose(baseline,read(parent/'result.json')['TRAIN_mae_pp'],atol=1e-5,rtol=0)
    out.mkdir(parents=True,exist_ok=True);save(out/'config.json',{**cfg,'refinement':'TRAIN-only full-batch L-BFGS shared head',
        'parent_run':str(parent.resolve()),'parent_model_sha256':fingerprint(parent/'model.pt')})
    torch.save(adapter_state(network),out/'model.pt');save(out/'metrics.json',metrics);np.savez_compressed(out/'predictions.npz',**arrays)
    history=[];save(out/'selection.json',dict(step=0,TRAIN_mae_pp=best))
    def closure():
        optimizer.zero_grad(set_to_none=True);total=0.
        for begin in range(0,len(rows),32):
            rr=rows[begin:begin+32];batch=pack(rr,'cuda');encoded=torch.stack(tokens[begin:begin+32]).cuda()
            loss=reconstruction_loss(network.reconstruct(encoded,batch),batch)*(len(rr)/len(rows))
            if not torch.isfinite(loss):raise ValueError('Nonfinite full-TRAIN objective')
            loss.backward();total+=float(loss.detach())
        return torch.tensor(total,device='cuda')
    for step in range(1,81):
        network.train();objective=float(optimizer.step(closure))
        if step%5==0 or step==1:
            score,metrics,arrays=evaluate(network,rows,tokens)
            history.append(dict(step=step,objective=objective,TRAIN_mae_pp=score))
            if score<best:
                best=score;torch.save(adapter_state(network),out/'model.pt');save(out/'metrics.json',metrics)
                np.savez_compressed(out/'predictions.npz',**arrays);save(out/'selection.json',dict(step=step,TRAIN_mae_pp=score))
            save(out/'history.json',history);print(dict(dataset=cfg['dataset'],step=step,TRAIN_mae_pp=score),flush=True)
    load_adapter(network,torch.load(out/'model.pt',map_location='cpu',weights_only=True))
    fresh=encode_all(network,rows);score,metrics,arrays=evaluate(network,rows,fresh)
    with np.load(out/'predictions.npz',allow_pickle=False) as prior:
        error=max(float(np.max(np.abs(arrays[k]-prior[k]))) for k in arrays if k.endswith('__prediction'))
    if error>1e-5:raise ValueError('Refined actual LLM replay differs')
    save(out/'replay_audit.json',dict(passed=True,max_error_soh=error,actual_LLM_recomputed=True,model_sha256=fingerprint(out/'model.pt')))
    result=dict(dataset=cfg['dataset'],TRAIN_cells=len(plan['train']),TRAIN_mae_pp=score,baseline_TRAIN_mae_pp=baseline,
        TRAIN_mape_percent=float(np.mean([m['mape_percent'] for m in metrics.values()])),adopted=score<baseline,
        parent_model_sha256=fingerprint(parent/'model.pt'),model_sha256=fingerprint(out/'model.pt'),
        encoder_cache_sha256=encoded_manifest['sha256'],steps=80,selection=read(out/'selection.json'),
        reconstruction_only=True,predictive_ability_proven=False)
    save(out/'result.json',result)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args();root=args.root
    provenance={str(p.resolve()):fingerprint(p) for p in [Path(__file__),Path('docs/LLM_SMETimes_TRAIN_reconstruction_refine_20261016.md'),
        Path('battery_smetimes/llm_reconstruction.py'),Path('scripts/train_llm_reconstruction.py')]}
    dest=root/'refinement';dest.mkdir(exist_ok=True)
    if (dest/'registration.json').exists():
        if read(dest/'registration.json')!=provenance:raise ValueError('Refinement sources changed')
    else:save(dest/'registration.json',provenance)
    while not (root/'queue.json').exists() or not read(root/'queue.json')['complete']:
        time.sleep(10)
    lock=Path('experiments/llm_curve_full/gpu_queue.lock').open('a+b');lock.seek(0)
    import msvcrt
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    results=[]
    try:
        for task in read(root/'queue.json')['tasks']:
            ds=task['dataset'];parent=root/'runs'/ds;metrics=read(parent/'metrics.json');base=read(parent/'result.json')
            needed=base['TRAIN_mae_pp']>.5 or any(m['mape_percent']>3 or m['endpoint_error_pp']>3 for m in metrics.values())
            if not needed:results.append(dict(dataset=ds,adopted=False,reason='First pass already within refinement priority thresholds'));continue
            out=dest/'runs'/ds
            if (out/'result.json').exists():result=read(out/'result.json')
            else:result=fit(parent,out)
            results.append(result);save(dest/'progress.json',results)
            if result['adopted']:
                report=dest/'reports'/ds
                if not (report/'figure_manifest.json').exists():
                    subprocess.run([sys.executable,'-m','scripts.report_llm_reconstruction','--run',str(out),'--out',str(report)],check=True)
        save(dest/'result.json',dict(complete=True,datasets=results,TRAIN_only=True))
    finally:
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()


if __name__=='__main__':main()
