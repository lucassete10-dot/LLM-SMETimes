"""Explicit shared-input neural/SMETimes adaptation for research transfer only.

No physical-family renaming or production-gate bypass. Source-only scales,
unchanged target identities, and schema-aware cards are essential: the native
51/46/44-column card extractor cannot interpret these 11-column histories.
"""

import copy

import numpy as np
import torch

from battery_smetimes.forecast import query_matrix
from battery_smetimes.research_sequences import TrainingScale, SmallForecaster, training_contexts
from battery_smetimes.research_transfer_baselines import (
    CACHED_SCHEMA, _allowed_cells, transfer_input_schema, transfer_prefix,
)
from battery_smetimes.research_transfer_signals import RESEARCH_MODE
from battery_smetimes.research_transfer_vi120 import SCHEMA as VI_SCHEMA


def _source_ids(ids, plan):
    if not ids or len(ids) != len(set(ids)) or set(ids) - set(plan["source_refit"]):
        raise ValueError("target/duplicate/empty IDs cannot enter source-only neural fitting")


def _query_identity(allowed, schema, cell):
    if transfer_input_schema(cell) != schema:
        raise ValueError("query protocol differs from trained shared-input schema")
    if allowed.get(cell["cell_id"]) != {"dataset": cell["metadata"]["dataset"], "physical_family": cell["physical_family"]}:
        raise ValueError("query cell is not in the unchanged frozen transfer cohorts")


def _visible_common(cell, origin):
    # Exactly the baseline's input validation and eligibility, not a new
    # error-dependent exclusion rule for the LLM.
    prefix = transfer_prefix(cell, origin)
    cycles = np.asarray(cell["cycle"])
    take = cycles <= origin
    observed, common = cycles[take], np.asarray(cell["common"], dtype=float)[take]
    if len(observed) > 100:
        raise ValueError("shared history exceeds frozen source-cycle budget")
    return prefix, observed, common


def shared_statistics_cards(cell, origin, patch_size=5):
    """Physical units and missingness, with no dataset/ID/type/label text.

    VI: five voltage changes relative to onset + five absolute C-rates + V0.
    Cached VIT: three voltage/temperature changes relative to 100s, respective
    absolute references, and three absolute C-rates. Neither is raw amperes.
    """
    if isinstance(patch_size, bool) or not isinstance(patch_size, int) or patch_size < 1 or 100 % patch_size:
        raise ValueError("statistics-card patch size must divide the 100-cycle budget")
    _, cycles, common = _visible_common(cell, origin)
    schema = transfer_input_schema(cell)
    if schema == VI_SCHEMA:
        voltage = common[:, :5] + common[:, 10:11]
        current = common[:, 5:10]
        temperature = None
        window = "Discharge samples within 0 to 120 s."
    elif schema == CACHED_SCHEMA:
        voltage = common[:, :3] + common[:, 9:10]
        current = common[:, 3:6]
        temperature = common[:, 6:9] + common[:, 10:11]
        window = "Discharge samples at 100, 200 and 300 s."
    else:
        raise ValueError("unsupported shared statistics-card schema")
    cards = []
    for start in range(0, len(cycles), patch_size):
        end = min(start + patch_size, len(cycles))
        text = [f"Observed cycles {int(cycles[start])} to {int(cycles[end-1])}; {end-start} observations.", window]
        for name, matrix, unit in (("Voltage", voltage, "V"), ("Absolute current normalized by rated capacity", current, "C-rate"),
                                   ("Temperature", temperature, "degC")):
            if matrix is None:
                text.append(f"{name} unavailable in this input protocol.")
                continue
            values = matrix[start:end]
            finite = values[np.isfinite(values)]
            if not len(finite):
                text.append(f"{name} unavailable.")
                continue
            means = [row[np.isfinite(row)].mean() for row in values if np.isfinite(row).any()]
            change = float(means[-1] - means[0]) if len(means) > 1 else 0.
            text.append(f"{name} mean {finite.mean():.4g} {unit}, standard deviation {finite.std():.4g}, observed change {change:.4g}.")
        cards.append(" ".join(text))
    return cards


class TransferHistoryPreprocessor:
    """All learned raw/statistical/query moments use source IDs only.

    The nine-wide existing head context is four scaled observation-clock
    statistics plus five permanently unused zero slots. They are NOT invented
    temperatures, capacities, physical type IDs or native charging conditions.
    """

    def fit(self, cells, ids, plan, origins):
        self.allowed_cells = _allowed_cells(cells, plan)  # metadata/schema only
        _source_ids(ids, plan)
        if (not origins or list(origins) != sorted(set(origins))
                or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 or n > 100 for n in origins)):
            raise ValueError("origins must be unique increasing integer source cycles within 1..100")
        self.train_ids, self.origins = list(ids), list(origins)
        self.plan = copy.deepcopy(plan)
        self.mode = RESEARCH_MODE
        self.input_schema = transfer_input_schema(cells[ids[0]])
        self.raw_counts, self.used_input_train_ids = {}, []
        raw, raw_weight, statistics, stats_weight, queries, query_weight = [], [], [], [], [], []
        for key in ids:
            cell, parts = cells[key], []
            for origin in origins:
                try:
                    prefix, cycles, values = _visible_common(cell, origin)
                except ValueError as error:
                    if "insufficient" in str(error) or "not established" in str(error):
                        continue
                    raise
                take = np.isfinite(cell["soh"]) & (cell["soh"] > 0) & (cell["label_cycle"] > origin)
                future = cell["label_cycle"][take]
                if len(future) < 2:
                    continue
                parts.append((origin, prefix, cycles, values, future))
            self.raw_counts[key] = 0
            if not parts:
                continue
            self.used_input_train_ids.append(key)
            # One common history per source cell, equal total cell weight.
            history = parts[-1][3]
            self.raw_counts[key] = len(history)
            raw.append(history)
            raw_weight.append(np.full(len(history), 1 / len(history)))
            for origin, prefix, _, _, future in parts:
                statistics.append(prefix)
                stats_weight.append(1 / len(parts))
                q = query_matrix(prefix, origin, future)[:, -3:]
                queries.append(q)
                query_weight.append(np.full(len(q), 1 / (len(parts) * len(q))))
        if not raw:
            raise ValueError("no valid source-only shared neural contexts")
        self.raw_scale = TrainingScale().fit(np.concatenate(raw), np.concatenate(raw_weight))
        self.stats_scale = TrainingScale().fit(np.asarray(statistics), np.asarray(stats_weight))
        self.query_scale = TrainingScale().fit(np.concatenate(queries), np.concatenate(query_weight))
        self.raw_width, self.stats_width = raw[0].shape[1], len(statistics[0])
        return self

    def model_input(self, cell, origin):
        _query_identity(self.allowed_cells, self.input_schema, cell)
        if origin not in self.origins:
            raise ValueError("origin outside frozen experiment")
        prefix, cycles, common = _visible_common(cell, origin)
        observed = np.c_[self.raw_scale.transform(common), np.isfinite(common).astype(np.float32), cycles / 100]
        history = np.zeros((100, 2 * self.raw_width + 1), dtype=np.float32)
        history[:len(observed)] = observed
        statistics = self.stats_scale.transform(prefix[None])[0]
        return {"history": history, "length": len(observed), "statistics": statistics,
                "conditions": np.r_[statistics[-4:], np.zeros(5, dtype=np.float32)].astype(np.float32)}

    def future_query(self, origin, cycles):
        if origin not in self.origins:
            raise ValueError("origin outside frozen experiment")
        return self.query_scale.transform(query_matrix(np.zeros(1), origin, cycles)[:, -3:])


def make_transfer_sequence_bundle(processor, architecture, parameters, seed, device, *, llm_config=None):
    """Create existing small/frozen-LLM architecture, never a native predictor.

    The executor must verify frozen model source/checkpoint hashes and main
    completion before calling this on public data. This helper is not an
    authorized generalization launcher or a smoke-training result by itself.
    """
    if processor.mode != RESEARCH_MODE:
        raise ValueError("research-only shared processor required")
    width = 2 * processor.raw_width + 1
    torch.manual_seed(seed)
    if architecture in ("gru", "statistics_mlp"):
        network = SmallForecaster(architecture, width, processor.stats_width,
            hidden=parameters["hidden"], dropout=parameters["dropout"]).to(device)
    elif architecture in ("llm_e1", "llm_e2"):
        from battery_smetimes.research_llm import FrozenSMETimesForecaster
        if llm_config is None:
            raise ValueError("shared LLM needs the explicitly frozen original configuration")
        variant = llm_config["variants"][architecture]
        if (variant["pretrained"] != (architecture == "llm_e1") or not variant["use_statistics"]
                or variant["experts"] != 4 or variant["rank"] != 64):
            raise ValueError("shared E1/E2 must preserve the frozen main comparison")
        network = FrozenSMETimesForecaster(llm_config["checkpoint"], width, seed=seed, device=device,
            dropout=parameters["dropout"], patch_size=llm_config["patch_size"],
            max_text_tokens=llm_config["max_text_tokens"], **variant)
    else:
        raise ValueError("shared neural transfer currently supports GRU/statistics_MLP/E1/E2")
    network.eval()
    return {"schema": "research_shared_transfer_sequence_v1", "mode": RESEARCH_MODE,
        "not_a_production_route": True, "network": network, "processor": processor,
        "plan": copy.deepcopy(processor.plan), "allowed_cells": copy.deepcopy(processor.allowed_cells),
        "input_schema": processor.input_schema, "source_train_ids": list(processor.train_ids),
        "source_only_preprocessor_fit_ids": list(processor.train_ids), "origins": list(processor.origins),
        "model_name": architecture, "parameters": dict(parameters), "seed": seed, "device": device,
        "llm_config": copy.deepcopy(llm_config) if architecture.startswith("llm_") else None}


def _guard_bundle(bundle, cell, origin, mode):
    if mode != RESEARCH_MODE or bundle["mode"] != RESEARCH_MODE or not bundle["not_a_production_route"]:
        raise ValueError("research-only mode required; cannot bypass production physical gate")
    _query_identity(bundle["allowed_cells"], bundle["input_schema"], cell)
    if origin not in bundle["origins"]:
        raise ValueError("origin outside frozen experiment")


def shared_cards_embedding(bundle, cell, origin):
    _guard_bundle(bundle, cell, origin, RESEARCH_MODE)
    network = bundle["network"]
    cards = shared_statistics_cards(cell, origin, network.patch_size)
    key = (bundle["input_schema"], origin, tuple(cards))
    cache = bundle.setdefault("card_cache", {})  # strictly per backbone/seed
    if key not in cache:
        cache[key] = network.encode_cards([cards])[0].detach().cpu()
    return cache[key]


def transfer_training_contexts(bundle, cells, cap=96):
    _source_ids(bundle["source_train_ids"], bundle["plan"])
    contexts = training_contexts(cells, bundle["source_train_ids"], bundle["processor"], cap)
    if bundle["model_name"].startswith("llm_"):
        for row in contexts:
            row["card_embeddings"] = shared_cards_embedding(bundle, cells[row["cell_id"]], row["origin"])
    return contexts


def predict_transfer_sequence(bundle, cell, origin, future, *, mode):
    _guard_bundle(bundle, cell, origin, mode)
    network, processor, device = bundle["network"], bundle["processor"], bundle["device"]
    values, query = processor.model_input(cell, origin), processor.future_query(origin, future)
    if not len(query):
        raise ValueError("empty requested future")
    network.eval()
    with torch.inference_mode():
        history = torch.as_tensor(values["history"][None], device=device)
        length = torch.as_tensor([values["length"]], device=device)
        if bundle["model_name"].startswith("llm_"):
            representation = shared_cards_embedding(bundle, cell, origin)[None].to(device)
        else:
            representation = torch.as_tensor(values["statistics"][None], device=device)
        encoding = network.encode(history, length, representation)
        conditions = torch.as_tensor(values["conditions"][None], device=device)
        outputs = [network.decode(encoding, conditions, torch.as_tensor(part[None], device=device)).cpu().numpy()[0]
                   for part in np.array_split(query, max(1, int(np.ceil(len(query) / 1024))))]
    prediction = np.concatenate(outputs)
    if prediction.shape != (len(future),) or not np.isfinite(prediction).all():
        raise ValueError("invalid shared neural transfer prediction")
    return prediction
