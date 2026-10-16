"""Portable, label-free inference for the released full-signal SOH estimator."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from battery_smetimes.llm_reconstruction import ReconstructionProcessor, ReconstructionSMETimes, add_targets, pack
from battery_smetimes.research_llm import load_adapter


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


class ReleasedScale:
    """JSON representation of the TRAIN-only median imputer/standard scaler."""
    def __init__(self, config):
        self.median=np.asarray(config['median'],dtype=np.float64)
        self.mean=np.asarray(config['mean'],dtype=np.float64)
        self.scale=np.asarray(config['scale'],dtype=np.float64)

    def transform(self, raw):
        value=np.asarray(raw,dtype=np.float64).copy()
        value=np.where(np.isnan(value),self.median,value)
        value-=self.mean;value/=self.scale
        if not np.isfinite(value).all():raise ValueError('Nonfinite standardized features')
        return value.astype(np.float32)


def load_processor(config):
    processor=ReconstructionProcessor()
    processor.raw_width=config['raw_width']
    processor.scale=ReleasedScale(config['scaling'])
    return processor


def estimate(network,processor,signal,query):
    query=np.asarray(query)
    if query.ndim!=1 or len(query)<2 or not np.isfinite(query).all() or np.any(np.diff(query)<=0):
        raise ValueError('At least two strictly increasing finite query cycles required')
    if query[0]<signal['cycle'][0] or query[-1]>signal['cycle'][-1]:
        raise ValueError('Released model estimates SOH within observed signal range; future forecasting is unsupported')
    # Dummy values satisfy the existing batching interface. SOH never input.
    rows=add_targets(processor.inputs(signal),dict(cycle=signal['cycle'],label_cycle=query,soh=np.ones(len(query))))
    output=[]
    network.eval()
    with torch.inference_mode():
        for begin in range(0,len(rows),4):
            group=rows[begin:begin+4]
            cards=network.encode_cards([[r['card']] for r in group])[:,0].cpu()
            for row,card in zip(group,cards):row['cards']=card[None].expand(20,-1).clone()
            batch=pack(group,'cuda')
            tokens,_=network.encode(batch['history'],batch['length'],batch['cards'])
            prediction=network.reconstruct(tokens,batch).cpu().numpy()
            output.extend((r['target_cycle'],p[:len(r['truth'])]) for r,p in zip(group,prediction))
    x=np.concatenate([r[0] for r in output]);p=np.concatenate([r[1] for r in output]);order=np.argsort(x)
    np.testing.assert_array_equal(x[order],query)
    if not np.isfinite(p).all():raise ValueError('Nonfinite SOH output')
    return p[order]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bundle',type=Path,required=True,help='Dataset subfolder from the release asset')
    ap.add_argument('--checkpoint',type=Path,required=True,help='Separately obtained Llama-3.2-1B-Instruct')
    ap.add_argument('--signals',type=Path,required=True,help='NPZ: cycle [N], raw [N,D], optional query_cycle [M]')
    ap.add_argument('--out',type=Path,required=True,help='Output CSV')
    args=ap.parse_args()
    config=json.loads((args.bundle/'estimator.json').read_text(encoding='utf-8'))
    if sha256(args.bundle/'model.pt')!=config['model_sha256']:raise ValueError('Released adapter checksum mismatch')
    # Exact pretrained backbone and tokenizer are part of the model identity.
    for name,digest in config['backbone_files_sha256'].items():
        if sha256(args.checkpoint/name)!=digest:raise ValueError('Backbone/tokenizer differs: '+name)
    with np.load(args.signals,allow_pickle=False) as data:
        cycle=data['cycle'].copy();raw=data['raw'].copy()
        query=data['query_cycle'].copy() if 'query_cycle' in data else cycle.copy()
    if cycle.ndim!=1 or not np.isfinite(cycle).all() or len(cycle)<2 or np.any(np.diff(cycle)<=0):
        raise ValueError('Invalid signal cycle coordinates')
    if raw.shape!=(len(cycle),config['raw_width']) or np.isinf(raw).any():raise ValueError('Wrong signal feature schema')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    processor=load_processor(config)
    model=ReconstructionSMETimes(str(args.checkpoint),processor.raw_width,config['seed'])
    load_adapter(model,torch.load(args.bundle/'model.pt',map_location='cpu',weights_only=True))
    model.requires_grad_(False)
    prediction=estimate(model,processor,dict(cycle=cycle,raw=raw,metadata=config['metadata']),query)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    np.savetxt(args.out,np.c_[query,prediction],delimiter=',',header='source_cycle,estimated_soh',comments='')
    print(json.dumps(dict(dataset=config['dataset'],estimated_points=len(query),task='full-signal same-time SOH estimation',
                          future_forecasting=False,out=str(args.out)),ensure_ascii=False))


if __name__=='__main__':main()
