"""Matched-parameter GRU/TCN timing adapter, preserving legacy checkpoints."""

import torch
from torch import nn

from battery_smetimes.models.temporal import BatteryHealthGRU, BatteryHealthTCN


class CycleTimeHealthModel(nn.Module):
    def __init__(self, signal_width, *, context_width, backbone="gru",
                 physics_future=True):
        super().__init__()
        if backbone not in ("gru", "tcn") or context_width < 2:
            raise ValueError("valid backbone and context width required")
        factory = BatteryHealthGRU if backbone == "gru" else BatteryHealthTCN
        self.core = factory(signal_width, physics_future=physics_future)
        self.backbone = backbone
        self.context_width = context_width
        self.context_projection = nn.Linear(context_width, 64, bias=False)
        # Equal initial function and identical core initialization across arms.
        nn.init.zeros_(self.context_projection.weight)

    def forward(self, signal, mask, metadata, horizons=None, *, cycle_context):
        if (cycle_context.shape != (*signal.shape[:2], self.context_width) or
                mask.shape != signal.shape[:2]):
            raise ValueError("cycle context must align with signal window")
        encoded_input = (self.core.cycle_projection(signal) +
                         self.context_projection(cycle_context)) * mask.unsqueeze(-1)
        if self.backbone == "gru":
            encoded, _ = self.core.temporal(encoded_input)
        else:
            encoded = self.core.temporal(encoded_input, mask)
        return self.core._decode(encoded[:, -1], metadata, horizons)
