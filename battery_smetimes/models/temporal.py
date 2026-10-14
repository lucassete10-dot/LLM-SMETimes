"""Small battery-aware temporal baseline with an optional physics decoder."""

import torch
from torch import nn
from torch.nn import functional as F


class BatteryHealthGRU(nn.Module):
    def __init__(self, signal_width, metadata_width=6, hidden=64,
                 physics_future=False):
        super().__init__()
        self.physics_future = physics_future
        self.cycle_projection = nn.Sequential(
            nn.Linear(signal_width, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.temporal = nn.GRU(hidden, hidden, batch_first=True)
        self.metadata_projection = nn.Sequential(
            nn.Linear(metadata_width, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.fusion_gate = nn.Linear(2 * hidden, hidden)
        self.fusion_norm = nn.LayerNorm(hidden)
        self.current_head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.current_head.weight)
        nn.init.zeros_(self.current_head.bias)
        if physics_future:
            self.rate_head = nn.Linear(hidden, 1)
            self.acceleration_head = nn.Linear(hidden, 1)
            nn.init.constant_(self.rate_head.bias, -2.0)
            nn.init.constant_(self.acceleration_head.bias, -2.0)

    def forward(self, signal, mask, metadata, horizons=None):
        if signal.ndim != 3 or mask.shape != signal.shape[:2]:
            raise ValueError("expected signal [batch, time, features] and mask")
        projected = self.cycle_projection(signal) * mask.unsqueeze(-1)
        encoded, _ = self.temporal(projected)
        # Windows are left-padded, so their most recent valid cycle is last.
        return self._decode(encoded[:, -1], metadata, horizons)

    def _decode(self, temporal, metadata, horizons):
        meta = self.metadata_projection(metadata)
        gate = torch.sigmoid(self.fusion_gate(torch.cat((temporal, meta), dim=-1)))
        fused = self.fusion_norm(gate * temporal + (1 - gate) * meta)
        current = 1.0 + self.current_head(fused).squeeze(-1)
        result = {"current_soh": current, "embedding": fused}
        if self.physics_future:
            rate = F.softplus(self.rate_head(fused)).squeeze(-1) * 1e-3
            acceleration = F.softplus(self.acceleration_head(fused)).squeeze(-1) * 1e-6
            result["degradation_rate"] = rate
            result["degradation_acceleration"] = acceleration
            if horizons is not None:
                h = torch.as_tensor(horizons, device=temporal.device,
                                    dtype=temporal.dtype).reshape(1, -1)
                result["future_soh"] = (current[:, None] - rate[:, None] * h -
                                        acceleration[:, None] * h.square())
        return result


class WaveformHealthGRU(BatteryHealthGRU):
    """Depthwise waveform Cycle Encoder plus summary-HI and metadata fusion."""

    def __init__(self, signal_width, metadata_width=6, hidden=64,
                 physics_future=False, stages=2):
        super().__init__(signal_width, metadata_width, hidden, physics_future)
        if stages not in (1, 2):
            raise ValueError("waveform stages must be charge-only or both")
        self.stages = stages
        self.waveform_encoder = nn.Sequential(
            nn.Conv1d(4, 4, kernel_size=5, stride=2, padding=2, groups=4),
            nn.Conv1d(4, 32, kernel_size=1), nn.GELU(),
            nn.Conv1d(32, 32, kernel_size=5, stride=2, padding=2, groups=32),
            nn.Conv1d(32, hidden, kernel_size=1), nn.GELU(),
            nn.AdaptiveAvgPool1d(1))
        self.waveform_projection = nn.Linear(stages * hidden, hidden)
        self.cycle_fusion_gate = nn.Linear(2 * hidden, hidden)

    def encode_cycles(self, signal, mask, waveform):
        if waveform is None or waveform.shape[:2] != signal.shape[:2] or \
           waveform.shape[-3:] != (self.stages, 4, 121):
            raise ValueError(f"expected waveform [batch, time, {self.stages}, 4, 121]")
        batch, steps = signal.shape[:2]
        flat = waveform.reshape(batch * steps * self.stages, 4, 121)
        normalized = torch.stack(((flat[:, 0] - 3.6) / .8, flat[:, 1] / 5,
                                  (flat[:, 2] - 25) / 25, flat[:, 3] / .2), dim=1)
        encoded_wave = self.waveform_encoder(normalized).squeeze(-1)
        encoded_wave = self.waveform_projection(
            encoded_wave.reshape(batch, steps, -1))
        encoded_hi = self.cycle_projection(signal)
        gate = torch.sigmoid(self.cycle_fusion_gate(
            torch.cat((encoded_wave, encoded_hi), dim=-1)))
        return (gate * encoded_wave + (1 - gate) * encoded_hi) * mask.unsqueeze(-1)

    def forward(self, signal, mask, metadata, horizons=None, waveform=None):
        fused_cycles = self.encode_cycles(signal, mask, waveform)
        encoded, _ = self.temporal(fused_cycles)
        return self._decode(encoded[:, -1], metadata, horizons)


class CausalResidualConv(nn.Module):
    """One strictly left-padded temporal convolution with a masked residual."""

    def __init__(self, hidden, dilation):
        super().__init__()
        self.dilation = dilation
        self.conv = nn.Conv1d(hidden, hidden, kernel_size=2, dilation=dilation)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(.05)

    def forward(self, x, mask):
        padded = F.pad(x.transpose(1, 2), (self.dilation, 0))
        update = self.dropout(F.gelu(self.conv(padded).transpose(1, 2)))
        return self.norm(x + update) * mask.unsqueeze(-1)


class CausalTCN(nn.Module):
    """Four layers cover the complete 16-diagnostic window (receptive field 16)."""

    def __init__(self, hidden=64):
        super().__init__()
        self.blocks = nn.ModuleList(CausalResidualConv(hidden, dilation)
                                    for dilation in (1, 2, 4, 8))

    def forward(self, x, mask):
        for block in self.blocks:
            x = block(x, mask)
        return x


class BatteryHealthTCN(BatteryHealthGRU):
    """Same signal/metadata/heads as GRU, with a causal convolution backbone."""

    def __init__(self, signal_width, metadata_width=6, hidden=64,
                 physics_future=False):
        super().__init__(signal_width, metadata_width, hidden, physics_future)
        self.temporal = CausalTCN(hidden)

    def forward(self, signal, mask, metadata, horizons=None):
        if signal.ndim != 3 or mask.shape != signal.shape[:2]:
            raise ValueError("expected signal [batch, time, features] and mask")
        projected = self.cycle_projection(signal) * mask.unsqueeze(-1)
        encoded = self.temporal(projected, mask)
        return self._decode(encoded[:, -1], metadata, horizons)


class WaveformHealthTCN(WaveformHealthGRU):
    """Raw-prefix Cycle Encoder + causal TCN + identical physics head."""

    def __init__(self, signal_width, metadata_width=6, hidden=64,
                 physics_future=False, stages=2):
        super().__init__(signal_width, metadata_width, hidden, physics_future,
                         stages=stages)
        self.temporal = CausalTCN(hidden)

    def forward(self, signal, mask, metadata, horizons=None, waveform=None):
        fused_cycles = self.encode_cycles(signal, mask, waveform)
        encoded = self.temporal(fused_cycles, mask)
        return self._decode(encoded[:, -1], metadata, horizons)
