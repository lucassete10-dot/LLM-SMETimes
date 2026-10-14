"""LLM/SMETimes-conditioned bounded local recovery, on fixed physical nodes.

This branch permits small local reversals that a single saturating recovery
cannot represent. It uses only the same LLM token memory and known conditions.
It does not fit target-cell coefficients or read future labels in inference.
"""
import math
import torch
from torch import nn
from battery_smetimes.llm_query_rate import QueryRateForecaster


def interpolate_fixed_nodes(values,knots,time):
    if values.ndim!=2 or time.ndim!=2 or values.shape[0]!=time.shape[0] or values.shape[1]!=len(knots):
        raise ValueError('local recovery schema mismatch')
    if knots.ndim!=1 or len(knots)<2 or float(knots[0])!=0 or torch.any(knots[1:]<=knots[:-1]):
        raise ValueError('local recovery knots must increase from zero')
    if not torch.isfinite(time).all() or torch.any(time<0):raise ValueError('invalid recovery query')
    index=torch.searchsorted(knots,time.contiguous(),right=True).sub(1).clamp(0,len(knots)-2)
    fraction=((time-knots[index])/(knots[index+1]-knots[index])).clamp(0,1)
    return values.gather(1,index)*(1-fraction)+values.gather(1,index+1)*fraction


class LocalRecoveryQueryForecaster(QueryRateForecaster):
    def __init__(self,*args,local_recovery_spacing_cycles=5.,local_recovery_max_soh=.01,local_readout='query',**kwargs):
        if local_readout not in ('query','vector'):raise ValueError('invalid local LLM readout')
        if not math.isfinite(local_recovery_spacing_cycles) or local_recovery_spacing_cycles<=0:
            raise ValueError('invalid fixed local recovery spacing')
        if not math.isfinite(local_recovery_max_soh) or not 0<local_recovery_max_soh<=.02:
            raise ValueError('local recovery must be bounded by at most two pp')
        super().__init__(*args,**kwargs)
        step=local_recovery_spacing_cycles/100;end=float(self.rate_knots[-1])
        knots=torch.arange(math.ceil(end/step)+1,device=self.rate_knots.device,dtype=torch.float32)*step
        self.register_buffer('local_recovery_knots',knots)
        self.local_recovery_max_soh=float(local_recovery_max_soh)
        self.local_readout=local_readout
        width=self.origin_query.shape[-1]
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(kwargs.get('seed',20260930)+404)
            self.local_query=nn.Parameter(torch.randn(len(knots),width)*.02) if local_readout=='query' else None
            self.local_head=nn.Sequential(nn.Linear(width,64),nn.GELU(),nn.Linear(64,1 if local_readout=='query' else len(knots)))
            nn.init.normal_(self.local_head[-1].weight,std=.002);nn.init.zeros_(self.local_head[-1].bias)
        self.to(self.rate_knots.device)

    def local_parameters(self,encoding,conditions):
        tokens,valid=encoding;memory=self.memory(tokens);time=self.local_recovery_knots
        features=torch.stack((time/10,torch.log1p(time),torch.sin(time),torch.cos(time),torch.sin(time/4),torch.cos(time/4)),dim=-1)
        query=((self.local_query+self.horizon_query(features))[None] if self.local_readout=='query'
            else self.origin_query.expand(len(tokens),-1,-1))+self.condition_query(conditions)[:,None]
        values,_=self.reader(query,memory,memory,key_padding_mask=~valid,need_weights=False)
        read=self.read_norm(values+query)
        output=self.local_head(read)
        output=output.squeeze(-1) if self.local_readout=='query' else output[:,0,:]
        residual=self.local_recovery_max_soh*torch.tanh(output)
        # Exactly zero at origin, without doubling the pointwise bound.
        return torch.cat((torch.zeros_like(residual[:,:1]),residual[:,1:]),dim=1)

    def decode(self,encoding,conditions,queries,return_parameters=False):
        base,parameters=super().decode(encoding,conditions,queries,return_parameters=True)
        values=self.local_parameters(encoding,conditions)
        time=(queries*self.query_scale+self.query_mean)[...,1].clamp_min(0)
        prediction=base+interpolate_fixed_nodes(values,self.local_recovery_knots,time)
        parameters['local_recovery_values']=values
        return (prediction,parameters) if return_parameters else prediction
