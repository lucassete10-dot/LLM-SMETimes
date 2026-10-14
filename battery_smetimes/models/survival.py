"""A small log-normal remaining-life model with right-censoring support."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class LogNormalLifeMLP(nn.Module):
    def __init__(self, signal_width, hidden=48):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(signal_width, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Dropout(.1), nn.Linear(hidden, hidden), nn.GELU())
        self.log_life = nn.Linear(hidden, 1)
        nn.init.zeros_(self.log_life.weight)
        nn.init.zeros_(self.log_life.bias)
        self.raw_sigma = nn.Parameter(torch.tensor(-1.4))

    def forward(self, x):
        mean = math.log(500) + self.log_life(self.encoder(x)).squeeze(-1)
        sigma = F.softplus(self.raw_sigma).clamp_min(.05)
        return mean, sigma


def conditional_lognormal_nll(mean, sigma, origin, observed_time, event):
    """Likelihood of EOL at `observed_time`, conditional on survival to origin.

    `event=True` contributes an observed EOL density; `event=False` contributes
    only the constraint EOL > observed_time. This never fabricates an endpoint
    for a right-censored cell.
    """
    if torch.any(origin <= 0) or torch.any(observed_time <= origin):
        raise ValueError("origin and observed time must be positive and ordered")
    origin_z = (torch.log(origin) - mean) / sigma
    observed_z = (torch.log(observed_time) - mean) / sigma
    log_survival_origin = torch.special.log_ndtr(-origin_z)
    log_survival_observed = torch.special.log_ndtr(-observed_z)
    log_density = (-torch.log(observed_time) - torch.log(sigma) -
                   .5 * math.log(2 * math.pi) - .5 * observed_z.square())
    return -(torch.where(event, log_density, log_survival_observed) -
             log_survival_origin)


def conditional_lognormal_quantile(mean, sigma, origin, quantile):
    """EOL quantile given the battery is alive at the observation cycle."""
    if not 0 < quantile < 1 or torch.any(origin <= 0):
        raise ValueError("invalid conditional quantile or origin")
    survival_origin = torch.special.ndtr((mean - torch.log(origin)) / sigma)
    tail_probability = ((1 - quantile) * survival_origin).clamp(1e-7, 1 - 1e-7)
    return torch.exp(mean - sigma * torch.special.ndtri(tail_probability))
