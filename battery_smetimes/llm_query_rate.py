"""SMETimes LLM-token queries -> physical rate nodes -> analytic SOH integral.

Every prediction uses the pretrained numeric/text fusion and expert path.
Knots are fixed physical horizons, never the target cell's observed endpoint.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

from battery_smetimes.research_llm import FrozenSMETimesForecaster


DEFAULT_KNOTS = (0., .1, .25, .5, 1., 2., 3., 4., 6., 8., 12., 16., 24., 32.)


def integrate_linear_rates(rates, knots, time):
    """Integral of nonnegative piecewise-linear rate, constant beyond last knot.

    Vectorized interval integrals preserve independent/unsorted query semantics
    and derivatives wrt every rate. Units: horizon/100 cycles, SOH per unit.
    """
    if rates.ndim != 2 or time.ndim != 2 or rates.shape[0] != time.shape[0]:
        raise ValueError('rate/query batch schema mismatch')
    if knots.ndim != 1 or rates.shape[1] != len(knots) or len(knots) < 2:
        raise ValueError('physical rate knot schema mismatch')
    if torch.any(knots[1:] <= knots[:-1]) or float(knots[0]) != 0.:
        raise ValueError('physical horizons must increase from zero')
    if not torch.isfinite(time).all() or torch.any(time < 0):
        raise ValueError('rate integration requires finite nonnegative horizons')
    widths = knots[1:] - knots[:-1]
    elapsed = torch.minimum((time[..., None] - knots[:-1]).clamp_min(0), widths)
    slope = (rates[:, 1:] - rates[:, :-1]) / widths
    integral = (rates[:, None, :-1] * elapsed + .5 * slope[:, None, :] * elapsed.square()).sum(-1)
    return integral + rates[:, -1:] * (time - knots[-1]).clamp_min(0)


class QueryRateForecaster(FrozenSMETimesForecaster):
    def __init__(self, checkpoint, history_width, *, rate_knots=DEFAULT_KNOTS,
                 rate_width=128, recovery_max_soh=.005, learned_node_queries=False, **kwargs):
        if 'pooling' in kwargs or 'output_head' in kwargs:
            raise ValueError('query rate architecture owns its pooling/output head')
        if rate_width < 8 or rate_width % 4:
            raise ValueError('query rate width must be a multiple of four >=8')
        super().__init__(checkpoint, history_width, pooling='tokens', output_head='integrated_rate', **kwargs)
        device=self.numeric_projection[0].weight.device
        knots=torch.as_tensor(rate_knots,dtype=torch.float32,device=device)
        if len(knots)<2 or float(knots[0])!=0 or not torch.isfinite(knots).all() or torch.any(knots[1:]<=knots[:-1]):
            raise ValueError('invalid fixed physical rate knots')
        if not math.isfinite(recovery_max_soh) or not 0<=recovery_max_soh<=.02:
            raise ValueError('recovery bound must be finite and at most two pp')
        self.register_buffer('rate_knots',knots)
        self.recovery_max_soh=float(recovery_max_soh)
        dimension=self.backbone.config.hidden_size
        del self.head
        # Isolate initialization from how many pretrained layers were loaded.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(kwargs.get('seed',20260930)+303)
            self.memory=nn.Sequential(nn.LayerNorm(dimension),nn.Linear(dimension,rate_width),nn.GELU(),nn.LayerNorm(rate_width))
            self.condition_query=nn.Linear(9,rate_width)
            self.horizon_query=nn.Sequential(nn.Linear(6,rate_width),nn.GELU(),nn.Linear(rate_width,rate_width))
            self.origin_query=nn.Parameter(torch.randn(1,1,rate_width)*.02)
            self.reader=nn.MultiheadAttention(rate_width,4,dropout=kwargs.get('dropout',.1),batch_first=True)
            self.read_norm=nn.LayerNorm(rate_width)
            self.start_head=nn.Sequential(nn.Linear(rate_width,64),nn.GELU(),nn.Linear(64,1))
            self.rate_head=nn.Sequential(nn.Linear(rate_width,64),nn.GELU(),nn.Linear(64,1))
            self.recovery_head=nn.Sequential(nn.Linear(rate_width,32),nn.GELU(),nn.Linear(32,2))
            for head in (self.start_head,self.rate_head,self.recovery_head):
                nn.init.normal_(head[-1].weight,std=.002);nn.init.zeros_(head[-1].bias)
            self.rate_head[-1].bias.data.fill_(math.log(math.expm1(1.)))
            # Optional fixed-node embeddings remove the restriction that every
            # node must share a smooth low-frequency time-feature mapping.
            # They identify physical horizons, never cells or true endpoints.
            self.node_query=nn.Parameter(torch.zeros(len(knots),rate_width)) if learned_node_queries else None
            if self.node_query is not None:nn.init.normal_(self.node_query,std=.02)
        self.to(device)

    def trajectory_parameters(self, encoding, conditions):
        tokens,valid=encoding
        if tokens.ndim!=3 or valid.shape!=tokens.shape[:2] or not valid.any(1).all():
            raise ValueError('invalid LLM token memory')
        memory=self.memory(tokens)
        time=self.rate_knots
        features=torch.stack((time/10,torch.log1p(time),torch.sin(time),torch.cos(time),
                              torch.sin(time/4),torch.cos(time/4)),dim=-1)
        condition=self.condition_query(conditions)[:,None,:]
        horizon=self.horizon_query(features)
        if self.node_query is not None:horizon=horizon+self.node_query
        query=torch.cat((self.origin_query.expand(len(tokens),-1,-1),
            horizon[None].expand(len(tokens),-1,-1)),dim=1)+condition
        values,_=self.reader(query,memory,memory,key_padding_mask=~valid,need_weights=False)
        read=self.read_norm(values+query)
        level=1+.1*self.start_head(read[:,0]).squeeze(-1)
        rates=.01*F.softplus(self.rate_head(read[:,1:]).squeeze(-1))
        recovery=self.recovery_head(read[:,0])
        return level,rates,recovery

    def decode(self, encoding, conditions, queries, return_parameters=False):
        level,rates,recovery=self.trajectory_parameters(encoding,conditions)
        physical=queries*self.query_scale+self.query_mean
        time=physical[...,1].clamp_min(0)
        prediction=level[:,None]-integrate_linear_rates(rates,self.rate_knots,time)
        amplitude=self.recovery_max_soh*torch.tanh(recovery[:,0:1])
        duration=.2+F.softplus(recovery[:,1:2])
        prediction=prediction+amplitude*(-torch.expm1(-time/duration))
        return (prediction,{'origin_soh':level,'rates':rates,'recovery':recovery}) if return_parameters else prediction
