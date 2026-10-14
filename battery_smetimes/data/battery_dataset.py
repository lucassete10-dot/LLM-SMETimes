"""Causal per-cell diagnostic windows for battery-aware SOH training.

The existing v2 features are already signal-only; capacity, SOH, RUL, and
endpoints are never put into the input matrix. This adapter is an interim
feature-level baseline, not the final raw-waveform Cycle Encoder.
"""

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from torch.utils.data import Dataset, Sampler


def fit_source_imputer(cells, train_ids, variant="signal_history"):
    if not train_ids:
        raise ValueError("empty training cell set")
    matrices = [cells[key].matrix(variant) for key in train_ids]
    return SimpleImputer(strategy="median", keep_empty_features=True).fit(
        np.concatenate(matrices))


def fit_source_scaler(cells, train_ids, imputer, variant="signal_history"):
    matrices = [imputer.transform(cells[key].matrix(variant)) for key in train_ids]
    scaler = RobustScaler(quantile_range=(10, 90)).fit(np.concatenate(matrices))
    scaler.scale_ = np.maximum(scaler.scale_, .02)
    return scaler


def numeric_metadata(record):
    """Physical scaling with explicit availability bits; no test-fit scaler."""
    fields = (("ambient_temperature_c", 50.0),
              ("charge_rate_c", 5.0), ("discharge_rate_c", 5.0))
    values, available = [], []
    for field, scale in fields:
        value = record.get(field)
        valid = isinstance(value, (int, float)) and np.isfinite(value)
        values.append(float(value) / scale if valid else 0.0)
        available.append(float(valid))
    return np.asarray(values + available, dtype=np.float32)


class CausalWindowDataset(Dataset):
    def __init__(self, cells, records_by_id, cell_ids, imputer, *, lookback=16,
                 variant="signal_history", labeled_only=True, scaler=None,
                 future_horizons=()):
        if lookback < 1 or len(cell_ids) != len(set(cell_ids)):
            raise ValueError("invalid lookback or duplicate cell IDs")
        self.lookback = lookback
        self.future_horizons = tuple(int(value) for value in future_horizons)
        if any(value <= 0 for value in self.future_horizons):
            raise ValueError("future horizons must be positive cycle gaps")
        self.cells = {key: cells[key] for key in cell_ids}
        self.record_by_id = {key: records_by_id[key] for key in cell_ids}
        self.matrices = {}
        self.position_by_cycle = {}
        self.rows = []
        self.index_by_domain_cell = defaultdict(lambda: defaultdict(list))
        for key in cell_ids:
            cell = self.cells[key]
            transformed = imputer.transform(cell.matrix(variant)).astype(np.float32)
            if scaler is not None:
                transformed = np.clip(scaler.transform(transformed), -8, 8).astype(np.float32)
            if np.isinf(transformed).any() or not np.isfinite(transformed).all():
                raise ValueError(f"nonfinite transformed feature for {key}")
            self.matrices[key] = transformed
            self.position_by_cycle[key] = {int(cycle): index
                                            for index, cycle in enumerate(cell.cycle)}
            for position in range(len(cell.cycle)):
                if labeled_only and not cell.soh_available[position]:
                    continue
                index = len(self.rows)
                self.rows.append((key, position))
                self.index_by_domain_cell[cell.dataset][key].append(index)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        key, position = self.rows[index]
        cell = self.cells[key]
        start = max(0, position - self.lookback + 1)
        signal = np.zeros((self.lookback, self.matrices[key].shape[1]),
                          dtype=np.float32)
        mask = np.zeros(self.lookback, dtype=np.bool_)
        elapsed = np.zeros(self.lookback, dtype=np.float32)
        width = position - start + 1
        signal[-width:] = self.matrices[key][start:position + 1]
        mask[-width:] = True
        elapsed[-width:] = (cell.cycle[start:position + 1] -
                            cell.cycle[position]).astype(np.float32)
        result = {
            "signal": torch.from_numpy(signal),
            "mask": torch.from_numpy(mask),
            "relative_cycle": torch.from_numpy(elapsed),
            "metadata": torch.from_numpy(numeric_metadata(self.record_by_id[key])),
            "target_soh": torch.tensor(float(cell.soh[position]), dtype=torch.float32),
            "target_available": bool(cell.soh_available[position]),
            "sample_weight": torch.tensor(.25 if cell.capacity_review_flag[position]
                                          else 1.0, dtype=torch.float32),
            "cell_id": key,
            "dataset_id": cell.dataset,
            "target_cycle": int(cell.cycle[position]),
        }
        if self.future_horizons:
            future = np.full(len(self.future_horizons), np.nan, dtype=np.float32)
            available = np.zeros(len(self.future_horizons), dtype=np.bool_)
            for slot, gap in enumerate(self.future_horizons):
                future_position = self.position_by_cycle[key].get(int(cell.cycle[position]) + gap)
                if future_position is not None and cell.soh_available[future_position]:
                    future[slot] = cell.soh[future_position]
                    available[slot] = True
            result["future_soh"] = torch.from_numpy(future)
            result["future_available"] = torch.from_numpy(available)
        return result


class WaveformWindowDataset(CausalWindowDataset):
    """Adds original per-diagnostic V/I/T/Q prefixes to the causal window."""

    def __init__(self, cells, records_by_id, cell_ids, imputer, *, waveform_dir,
                 lookback=16, variant="signal_history", labeled_only=True,
                 scaler=None, future_horizons=(), waveform_stage="both"):
        if waveform_stage not in ("both", "charge"):
            raise ValueError("waveform_stage must be both or charge")
        super().__init__(cells, records_by_id, cell_ids, imputer,
                         lookback=lookback, variant=variant,
                         labeled_only=labeled_only, scaler=scaler,
                         future_horizons=future_horizons)
        self.waveforms = {}
        for key in cell_ids:
            source = Path(waveform_dir) / f"{key}.npz"
            with np.load(source) as file:
                cycle = file["cycle"].astype(np.int64)
                wave = file["waveform"].astype(np.float32)
            if not np.array_equal(cycle, cells[key].cycle):
                raise ValueError(f"waveform/cell cycle mismatch: {key}")
            if wave.shape != (len(cycle), 2, 4, 121) or not np.isfinite(wave).all():
                raise ValueError(f"invalid waveform tensor: {key}")
            self.waveforms[key] = wave if waveform_stage == "both" else wave[:, :1]
        self.waveform_stages = 2 if waveform_stage == "both" else 1

    def __getitem__(self, index):
        result = super().__getitem__(index)
        key, position = self.rows[index]
        start = max(0, position - self.lookback + 1)
        width = position - start + 1
        waveform = np.zeros((self.lookback, self.waveform_stages, 4, 121),
                            dtype=np.float32)
        waveform[-width:] = self.waveforms[key][start:position + 1]
        result["waveform"] = torch.from_numpy(waveform)
        return result


class DomainCellBalancedSampler(Sampler):
    """Uniform domain -> cell -> observation sampling, seeded per epoch."""

    def __init__(self, dataset, num_samples, seed=2026):
        if num_samples < 1 or not dataset.index_by_domain_cell:
            raise ValueError("empty sampling population")
        self.population = dataset.index_by_domain_cell
        self.domains = sorted(self.population)
        self.num_samples = num_samples
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        for _ in range(self.num_samples):
            domain = self.domains[int(rng.integers(len(self.domains)))]
            cells = sorted(self.population[domain])
            cell = cells[int(rng.integers(len(cells)))]
            positions = self.population[domain][cell]
            yield positions[int(rng.integers(len(positions)))]
