"""Same SOH/physics model, with optional separation of future-loss gradients."""

import torch
from torch.nn import functional as F

from battery_smetimes.models.cycle_time import CycleTimeHealthModel


class LossSeparatedHealthModel(CycleTimeHealthModel):
    def __init__(self, *args, separate_future_gradient=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.separate_future_gradient = separate_future_gradient

    def forward(self, signal, mask, metadata, horizons=None, *, cycle_context):
        output = super().forward(signal, mask, metadata, horizons,
                                 cycle_context=cycle_context)
        if self.separate_future_gradient and horizons is not None:
            # Numerical prediction and parameter count are unchanged. Only
            # the future loss's path into current SOH and its encoder stops.
            embedding = output["embedding"].detach()
            rate = F.softplus(self.core.rate_head(embedding)).squeeze(-1) * 1e-3
            acceleration = F.softplus(self.core.acceleration_head(embedding)).squeeze(-1) * 1e-6
            h = torch.as_tensor(horizons, device=signal.device,
                                dtype=signal.dtype).reshape(1, -1)
            output["future_soh"] = (output["current_soh"].detach()[:, None] -
                                    rate[:, None] * h - acceleration[:, None] * h.square())
        return output
