"""Small sequence and statistics-only regressors with shared label-free inputs."""

from collections import Counter
import copy
import time

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.nn import functional as F

from battery_smetimes.forecast import query_matrix
from battery_smetimes.physical_types import require_same_registered_family
from battery_smetimes.research_baselines import signal_prefix


class TrainingScale:
    def fit(self, x, weight=None):
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        x = self.imputer.fit_transform(np.asarray(x, dtype=np.float64))
        offset = x[0].copy()
        self.scaler = StandardScaler().fit(x - offset, sample_weight=weight)
        self.scaler.mean_ += offset
        return self

    def transform(self, x):
        value = self.scaler.transform(self.imputer.transform(np.asarray(x, dtype=np.float64)))
        if not np.isfinite(value).all():
            raise ValueError("nonfinite train-scaled model input")
        return value.astype(np.float32)


class HistoryPreprocessor:
    def fit(self, cells, ids, origins, history_slots=100):
        if not isinstance(history_slots, int) or history_slots < 1:
            raise ValueError('invalid history storage capacity')
        self.history_slots = history_slots
        raw, raw_w, stats, stats_w, queries = [], [], [], [], []
        self.train_ids = list(ids)
        self.origins = list(origins)
        self.raw_counts = {}
        for key in ids:
            cell = cells[key]
            history = cell["raw"][cell["cycle"] <= max(origins)]
            if len(history) > self.history_slots:
                raise ValueError('observations exceed configured history capacity')
            if np.isinf(history).any():
                raise ValueError("nonfinite observed training history")
            self.raw_counts[key] = len(history)
            raw.append(history)
            raw_w.append(np.full(len(history), 1 / len(history)))
            parts = []
            for origin in origins:
                try:
                    prefix = signal_prefix(cell, origin)
                except ValueError as error:
                    if "insufficient" in str(error) or "not established" in str(error):
                        continue
                    raise
                parts.append(prefix)
                take = np.isfinite(cell["soh"]) & (cell["soh"] > 0) & (cell["label_cycle"] > origin)
                future = cell["label_cycle"][take]
                if len(future):
                    queries.append(query_matrix(prefix, origin, future)[..., -3:])
            if parts:
                stats.extend(parts)
                stats_w.extend([1 / len(parts)] * len(parts))
        self.raw_scale = TrainingScale().fit(np.concatenate(raw), np.concatenate(raw_w))
        self.stats_scale = TrainingScale().fit(np.asarray(stats), np.asarray(stats_w))
        self.query_scale = TrainingScale().fit(np.concatenate(queries))
        self.raw_width = raw[0].shape[1]
        self.stats_width = len(stats[0])
        return self

    def model_input(self, cell, origin):
        # Input path does not access label_cycle, soh, capacity or EOL.
        prefix = signal_prefix(cell, origin)
        take = cell["cycle"] <= origin
        raw = cell["raw"][take]
        cycles = cell["cycle"][take]
        slots = getattr(self, 'history_slots', 100)  # legacy saved processors
        if len(raw) > slots or np.isinf(raw).any():
            raise ValueError("invalid visible sequence")
        observed = np.c_[self.raw_scale.transform(raw), np.isfinite(raw).astype(np.float32), cycles / 100]
        padded = np.zeros((slots, 2 * self.raw_width + 1), dtype=np.float32)
        padded[:len(observed)] = observed
        statistics = self.stats_scale.transform(prefix[None])[0]
        return {"history": padded, "length": len(observed), "statistics": statistics,
                "conditions": statistics[-9:]}

    def future_query(self, origin, cycles):
        return self.query_scale.transform(query_matrix(np.zeros(1), origin, cycles)[:, -3:])


class CausalBlock(nn.Module):
    def __init__(self, width, dilation, dropout):
        super().__init__()
        self.pad = 2 * dilation
        self.conv = nn.Conv1d(width, width, 3, dilation=dilation)
        self.norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        y = self.conv(F.pad(x, (self.pad, 0)))
        y = self.norm(y.transpose(1, 2)).transpose(1, 2)
        return x + self.dropout(F.gelu(y))


class SmallForecaster(nn.Module):
    def __init__(self, architecture, history_width, statistics_width, hidden=64, dropout=.1):
        super().__init__()
        self.architecture = architecture
        self.history_projection = nn.Linear(history_width, hidden) if architecture != "statistics_mlp" else None
        if architecture == "gru":
            self.encoder = nn.GRU(hidden, hidden, batch_first=True)
        elif architecture == "lstm":
            self.encoder = nn.LSTM(hidden, hidden, batch_first=True)
        elif architecture == "tcn":
            self.encoder = nn.Sequential(*(CausalBlock(hidden, d, dropout) for d in (1, 2, 4, 8)))
        elif architecture == "transformer":
            layer = nn.TransformerEncoderLayer(hidden, 4, dim_feedforward=2 * hidden,
                dropout=dropout, batch_first=True, activation="gelu", norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        elif architecture == "statistics_mlp":
            self.encoder = nn.Sequential(nn.Linear(statistics_width, 2 * hidden), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(2 * hidden, hidden), nn.GELU())
        else:
            raise ValueError("unknown small-model architecture")
        self.head = nn.Sequential(nn.Linear(hidden + 12, hidden), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, 1))

    def encode(self, history, length, statistics):
        if self.architecture == "statistics_mlp":
            return self.encoder(statistics)
        x = self.history_projection(history)
        if self.architecture in ("gru", "lstm"):
            x, _ = self.encoder(x)
        elif self.architecture == "tcn":
            x = self.encoder(x.transpose(1, 2)).transpose(1, 2)
        else:
            padding = torch.arange(x.shape[1], device=x.device)[None] >= length[:, None]
            x = self.encoder(x, src_key_padding_mask=padding)
        return x[torch.arange(len(x), device=x.device), length - 1]

    def decode(self, encoding, conditions, queries):
        # Each context is encoded once, then used for every requested horizon.
        encoded = encoding[:, None].expand(-1, queries.shape[1], -1)
        known = conditions[:, None].expand(-1, queries.shape[1], -1)
        return .9 + .1 * self.head(torch.cat((encoded, known, queries), -1)).squeeze(-1)


def training_contexts(cells, ids, processor, cap=96):
    contexts = []
    for key in ids:
        cell = cells[key]
        parts = []
        for origin in processor.origins:
            try:
                model_input = processor.model_input(cell, origin)
            except ValueError as error:
                if "insufficient" in str(error) or "not established" in str(error):
                    continue
                raise
            take = np.flatnonzero(np.isfinite(cell["soh"]) & (cell["soh"] > 0) & (cell["label_cycle"] > origin))
            if len(take) < 2:
                continue
            if len(take) > cap:
                take = take[np.unique(np.linspace(0, len(take) - 1, cap).round().astype(int))]
            count = len(take)
            query = np.zeros((cap, 3), dtype=np.float32)
            truth, valid = np.zeros(cap, dtype=np.float32), np.zeros(cap, dtype=np.float32)
            query[:count] = processor.future_query(origin, cell["label_cycle"][take])
            truth[:count] = cell["soh"][take]
            valid[:count] = 1
            parts.append({**model_input, "cell_id": key, "origin": origin,
                          "query": query, "truth": truth, "valid": valid})
        for part in parts:
            part["weight"] = 1 / len(parts)
        contexts.extend(parts)
    if not contexts:
        raise ValueError("no neural training contexts")
    return contexts


def pack_contexts(contexts, device):
    result = {}
    for name in ("history", "statistics", "conditions", "query", "truth", "valid", "weight"):
        result[name] = torch.as_tensor(np.asarray([r[name] for r in contexts]), device=device, dtype=torch.float32)
    result["length"] = torch.as_tensor([r["length"] for r in contexts], device=device, dtype=torch.long)
    return result


def fit_epoch(network, contexts, optimizer, seed, epoch, batch_size, device):
    network.train()
    rng = np.random.default_rng(seed + epoch)
    order = rng.permutation(len(contexts))
    sum_loss, sum_weight = 0., 0.
    for start in range(0, len(order), batch_size):
        batch = pack_contexts([contexts[index] for index in order[start:start + batch_size]], device)
        encoded = network.encode(batch["history"], batch["length"], batch["statistics"])
        prediction = network.decode(encoded, batch["conditions"], batch["query"])
        individual = (((prediction - batch["truth"]) ** 2) * batch["valid"]).sum(1) / batch["valid"].sum(1)
        # Constant normalization keeps unbiased battery/origin contribution
        # across random context minibatches (not a random ratio of weights).
        loss = (individual * batch["weight"]).mean()
        if not torch.isfinite(loss):
            raise ValueError("nonfinite neural training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(network.parameters(), 1.)
        optimizer.step()
        sum_loss += float((individual.detach() * batch["weight"]).sum())
        sum_weight += float(batch["weight"].sum())
    return sum_loss / sum_weight


def predict_sequence(bundle, cell, origin, future_cycles):
    require_same_registered_family(cell["physical_family"], [bundle["physical_family"]], bundle["registry"])
    network, processor, device = bundle["network"], bundle["processor"], bundle["device"]
    values = processor.model_input(cell, origin)
    query = processor.future_query(origin, future_cycles)
    network.eval()
    with torch.inference_mode():
        history = torch.as_tensor(values["history"][None], device=device)
        statistics = torch.as_tensor(values["statistics"][None], device=device)
        conditions = torch.as_tensor(values["conditions"][None], device=device)
        length = torch.as_tensor([values["length"]], device=device)
        encoded = network.encode(history, length, statistics)
        output = [network.decode(encoded, conditions, torch.as_tensor(part[None], device=device)).cpu().numpy()[0]
                  for part in np.array_split(query, max(1, int(np.ceil(len(query) / 1024))))]
    prediction = np.concatenate(output)
    if not np.isfinite(prediction).all():
        raise ValueError("nonfinite neural prediction")
    return prediction
