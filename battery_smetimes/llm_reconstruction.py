"""TRAIN full-life signal-to-SOH reconstruction, explicitly not forecasting."""
import numpy as np
import torch
from torch import nn

from battery_smetimes.research_sequences import TrainingScale
from battery_smetimes.research_llm import FrozenSMETimesForecaster, statistics_cards


class ReconstructionProcessor:
    def fit(self,cells,ids):
        self.train_ids=list(ids)
        self.raw_width=cells[ids[0]]['raw'].shape[1]
        raw=np.concatenate([cells[c]['raw'] for c in ids])
        weights=np.concatenate([np.full(len(cells[c]['raw']),1/len(cells[c]['raw'])) for c in ids])
        self.scale=TrainingScale().fit(raw,weights)
        return self

    def inputs(self,cell):
        # This method must remain independent of SOH/label cycles/cell identity.
        x=np.asarray(cell['cycle']);raw=np.asarray(cell['raw']);out=[]
        if not len(x) or np.any(np.diff(x)<=0) or np.isinf(raw).any():
            raise ValueError('Invalid full-life signal coordinates')
        for core in range(0,len(x),60):
            left=max(0,core-20);right=min(len(x),core+80)
            xx=x[left:right];rr=raw[left:right]
            history=np.zeros((100,2*self.raw_width+1),np.float32)
            history[:len(xx)]=np.c_[self.scale.transform(rr),np.isfinite(rr),xx/100]
            visible={**cell,'cycle':xx,'raw':rr}
            # A global signal-only statistics card provides reconstruction
            # context; its frozen LLM embedding is fused with every patch.
            card=statistics_cards(visible,float(xx[-1]),patch_size=100,history_slots=100)[0]
            out.append(dict(history=history,length=len(xx),card=card,left=left,right=right,
                core=core,core_end=min(core+60,len(x)),signal_cycles=xx.copy()))
        return out


def add_targets(inputs,cell):
    x=np.asarray(cell['cycle']);lx=np.asarray(cell['label_cycle']);y=np.asarray(cell['soh'])
    valid=np.isfinite(y)&(y>0)&np.isfinite(lx)
    lx=lx[valid];y=y[valid];order=np.argsort(lx);lx=lx[order];y=y[order]
    if len(lx)<2 or np.any(np.diff(lx)<=0):raise ValueError('Invalid reconstruction labels')
    insertion=np.searchsorted(x,lx).clip(0,len(x)-1);before=np.maximum(insertion-1,0)
    nearest=np.where(np.abs(x[before]-lx)<np.abs(x[insertion]-lx),before,insertion)
    out=[]
    for context in inputs:
        take=(nearest>=context['core'])&(nearest<context['core_end'])
        row=dict(context)
        row.update(target_cycle=lx[take],truth=y[take].copy(),
            observation_index=(nearest[take]-context['left']).astype(np.int64),
            weight=float(take.sum()/len(lx)))
        if take.any():out.append(row)
    if sum(len(r['truth']) for r in out)!=len(lx):raise ValueError('Target coverage missing')
    return out


def pack(rows,device):
    maximum=max(len(r['truth']) for r in rows)
    batch=dict(history=torch.tensor(np.stack([r['history'] for r in rows]),device=device),
        length=torch.tensor([r['length'] for r in rows],device=device),
        cards=torch.stack([r['cards'] for r in rows]).to(device))
    for name,dtype in (('observation_index',torch.long),('target_cycle',torch.float32),('truth',torch.float32)):
        array=np.stack([np.pad(r[name],(0,maximum-len(r[name]))) for r in rows])
        batch[name]=torch.tensor(array,device=device,dtype=dtype)
    batch['valid']=torch.tensor(np.stack([np.arange(maximum)<len(r['truth']) for r in rows]),device=device)
    batch['weight']=torch.tensor([r['weight'] for r in rows],device=device)
    return batch


class ReconstructionSMETimes(FrozenSMETimesForecaster):
    def __init__(self,checkpoint,raw_width,seed=20260930):
        super().__init__(checkpoint,2*raw_width+1,seed=seed,pretrained=True,use_statistics=True,
            pooling='tokens',residual_experts=True,normalize_router=True,dropout=0.,history_slots=100)
        del self.head
        d=self.backbone.config.hidden_size
        self.reconstruction_projection=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,256),nn.GELU(),nn.Linear(256,128))
        self.slot_embedding=nn.Embedding(5,8)
        self.reconstruction_head=nn.Sequential(nn.Linear(139,128),nn.GELU(),nn.Linear(128,1))
        nn.init.zeros_(self.reconstruction_head[-1].bias)
        self.to('cuda')

    def reconstruct(self,tokens,batch):
        idx=batch['observation_index'];patch=idx//5
        local=tokens.gather(1,patch[...,None].expand(-1,-1,tokens.shape[-1]))
        local=self.reconstruction_projection(local)
        nearest=batch['history'][...,-1].gather(1,idx)*100
        x=batch['target_cycle']
        physical=torch.stack((x/1000,torch.log1p(x.clamp_min(0))/10,(x-nearest)/100),-1)
        value=torch.cat((local,self.slot_embedding(idx%5),physical),-1)
        return .9+.1*self.reconstruction_head(value).squeeze(-1)


def reconstruction_loss(prediction,batch):
    # Equal physical-cell mass; each cell distributes its mass across every
    # real label, including short terminal windows. No truth interpolation.
    error=(100*(prediction-batch['truth']))**2
    point=(error*batch['valid']).sum(1)/batch['valid'].sum(1)
    return (point*batch['weight']).mean()
