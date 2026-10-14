"""Causal elapsed-cycle context for a controlled timing ablation."""

import numpy as np
import torch

from battery_smetimes.data.battery_dataset import CausalWindowDataset


MODES = ("baseline", "elapsed", "elapsed_rate")


def causal_cycle_slopes(cycle, signal, history=5):
    """Trailing OLS feature slopes per source cycle, never per observation."""
    cycle = np.asarray(cycle, dtype=np.float64)
    signal = np.asarray(signal, dtype=np.float64)
    if (signal.ndim != 2 or len(cycle) != len(signal) or history < 2 or
            not np.isfinite(cycle).all() or not np.isfinite(signal).all() or
            np.any(np.diff(cycle) <= 0)):
        raise ValueError("finite signals and strictly increasing cycles required")
    slopes = np.zeros_like(signal)
    for end in range(1, len(cycle)):
        start = max(0, end - history + 1)
        x = cycle[start:end + 1]
        x = x - x.mean()
        slopes[end] = x @ signal[start:end + 1] / (x @ x)
    return slopes.astype(np.float32)


class CycleTimeWindowDataset(CausalWindowDataset):
    """Keep all original features and add fixed-scale causal timing context.

    All arms have the same context width. Baseline receives zeros; elapsed
    receives only relative cycles and preceding gap; elapsed_rate additionally
    receives trailing slopes of the train-scaled raw signal columns. This
    holds the model parameterization constant between experimental arms.
    """

    def __init__(self, *args, time_mode="baseline", **kwargs):
        if time_mode not in MODES:
            raise ValueError(f"unknown time mode: {time_mode}")
        super().__init__(*args, **kwargs)
        self.time_mode = time_mode
        widths = {cell.raw.shape[1] for cell in self.cells.values()}
        if len(widths) != 1:
            raise ValueError("raw feature width must match within experiment")
        self.raw_width = widths.pop()
        self.context_width = self.raw_width + 2
        self.gaps, self.slopes = {}, {}
        for key, cell in self.cells.items():
            if self.matrices[key].shape[1] != 3 * self.raw_width:
                raise ValueError("expected raw/delta/history column contract")
            self.gaps[key] = np.r_[0, np.diff(cell.cycle)].astype(np.float32)
            self.slopes[key] = causal_cycle_slopes(
                cell.cycle, self.matrices[key][:, :self.raw_width])

    def __getitem__(self, index):
        result = super().__getitem__(index)
        key, position = self.rows[index]
        context = np.zeros((self.lookback, self.context_width), dtype=np.float32)
        start = max(0, position - self.lookback + 1)
        width = position - start + 1
        if self.time_mode != "baseline":
            # Fixed physical scales, independent of train/test distributions.
            context[:, 0] = result["relative_cycle"].numpy() / 100.0
            context[-width:, 1] = self.gaps[key][start:position + 1] / 10.0
        if self.time_mode == "elapsed_rate":
            context[-width:, 2:] = np.clip(
                self.slopes[key][start:position + 1] * 100.0, -8, 8)
        result["cycle_context"] = torch.from_numpy(context)
        return result
