"""Leakage-controlled NASA battery SOH estimation from raw MAT curves.

Only the first ``prefix_seconds`` of each discharge cycle is used to construct
voltage, current, and temperature features. Capacity and total discharge time
are targets/metadata only and never enter the model input or semantic prompt.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Sequence

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.io import loadmat
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, LlamaForCausalLM

from battery_mvp import compute_metrics, parse_ids, set_seed
from utils.device import model_dtype, resolve_device
from utils.llm import resolve_llm_ckp_dir


@dataclass(frozen=True)
class RawBattery:
    battery_id: str
    discharge_number: np.ndarray
    source_cycle_number: np.ndarray
    features: np.ndarray
    capacity_ah: np.ndarray
    soh: np.ndarray
    source_path: str


def _as_vector(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float64).reshape(-1)


def feature_names(sample_seconds: np.ndarray) -> List[str]:
    names = []
    for channel in ("voltage_v", "current_a", "temperature_c"):
        names.extend(f"{channel}_t{int(second):04d}s" for second in sample_seconds)
    names.extend(
        [
            "voltage_drop_v",
            "voltage_mean_v",
            "voltage_std_v",
            "current_abs_mean_a",
            "current_std_a",
            "temperature_mean_c",
            "temperature_std_c",
            "temperature_rise_c",
            "temperature_max_c",
            "prefix_charge_ah",
            "prefix_energy_wh",
        ]
    )
    return names


def extract_prefix_features(data: dict, sample_seconds: np.ndarray, prefix_seconds: float) -> np.ndarray:
    time = _as_vector(data["Time"])
    voltage = _as_vector(data["Voltage_measured"])
    current = _as_vector(data["Current_measured"])
    temperature = _as_vector(data["Temperature_measured"])
    if not (len(time) == len(voltage) == len(current) == len(temperature)):
        raise ValueError("Raw discharge channels have inconsistent lengths.")
    if len(time) < 2 or time[-1] < prefix_seconds:
        raise ValueError(f"Discharge record ends at {time[-1] if len(time) else 'n.a.'} s")
    if np.any(np.diff(time) < 0):
        raise ValueError("Discharge Time is not monotonic.")

    v_grid = np.interp(sample_seconds, time, voltage)
    i_grid = np.interp(sample_seconds, time, current)
    t_grid = np.interp(sample_seconds, time, temperature)
    mask = time <= prefix_seconds
    prefix_time = np.concatenate([time[mask], [prefix_seconds]])
    prefix_voltage = np.concatenate([voltage[mask], [v_grid[-1]]])
    prefix_current = np.concatenate([current[mask], [i_grid[-1]]])
    prefix_temperature = np.concatenate([temperature[mask], [t_grid[-1]]])
    abs_current = np.abs(prefix_current)

    summary = np.asarray(
        [
            v_grid[0] - v_grid[-1],
            prefix_voltage.mean(),
            prefix_voltage.std(),
            abs_current.mean(),
            prefix_current.std(),
            prefix_temperature.mean(),
            prefix_temperature.std(),
            t_grid[-1] - t_grid[0],
            prefix_temperature.max(),
            np.trapezoid(abs_current, prefix_time) / 3600.0,
            np.trapezoid(abs_current * prefix_voltage, prefix_time) / 3600.0,
        ],
        dtype=np.float64,
    )
    return np.concatenate([v_grid, i_grid, t_grid, summary]).astype(np.float32)


def load_raw_battery(
    raw_root: Path,
    battery_id: str,
    sample_seconds: np.ndarray,
    prefix_seconds: float,
) -> RawBattery:
    path = raw_root / f"{battery_id}.mat"
    if not path.exists():
        raise FileNotFoundError(f"Missing raw NASA file: {path}")
    raw = loadmat(path, simplify_cells=True)
    if battery_id not in raw or "cycle" not in raw[battery_id]:
        raise ValueError(f"Unexpected MATLAB structure in {path}")

    features, capacities, source_cycles = [], [], []
    discharge_number = 0
    for source_cycle, record in enumerate(raw[battery_id]["cycle"], start=1):
        if record["type"] != "discharge":
            continue
        discharge_number += 1
        try:
            vector = extract_prefix_features(record["data"], sample_seconds, prefix_seconds)
        except ValueError as exc:
            raise ValueError(
                f"{battery_id} discharge {discharge_number}: cannot extract prefix features: {exc}"
            ) from exc
        capacity = float(np.asarray(record["data"]["Capacity"]).reshape(-1)[0])
        features.append(vector)
        capacities.append(capacity)
        source_cycles.append(source_cycle)

    capacity_array = np.asarray(capacities, dtype=np.float32)
    if not len(capacity_array):
        raise ValueError(f"No discharge records in {path}")
    return RawBattery(
        battery_id=battery_id,
        discharge_number=np.arange(1, len(capacity_array) + 1, dtype=np.int64),
        source_cycle_number=np.asarray(source_cycles, dtype=np.int64),
        features=np.stack(features),
        capacity_ah=capacity_array,
        soh=(capacity_array / capacity_array[0]).astype(np.float32),
        source_path=str(path.resolve()),
    )


def raw_signature(
    batteries: Dict[str, RawBattery],
    prefix_seconds: float,
    model_path: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(str(prefix_seconds).encode())
    digest.update(str(Path(model_path).resolve()).encode())
    for battery_id in sorted(batteries):
        digest.update(battery_id.encode())
        digest.update(batteries[battery_id].features.tobytes())
    return digest.hexdigest()[:16]


def feature_prompt(
    feature: np.ndarray,
    names: Sequence[str],
    prefix_seconds: float,
    feature_mode: str,
) -> str:
    values = dict(zip(names, feature.tolist()))
    if feature_mode == "delta":
        return (
            f"Change in a NASA lithium-ion discharge prefix relative to this cell's first cycle. "
            f"At 0 seconds voltage changed by {values['voltage_v_t0000s']:.5f} V; "
            f"at {prefix_seconds:.0f} seconds voltage changed by "
            f"{values[f'voltage_v_t{int(prefix_seconds):04d}s']:.5f} V; "
            f"mean voltage changed by {values['voltage_mean_v']:.5f} V; "
            f"mean absolute current changed by {values['current_abs_mean_a']:.5f} A; "
            f"temperature rise changed by {values['temperature_rise_c']:.5f} C; "
            f"prefix energy changed by {values['prefix_energy_wh']:.5f} Wh."
        )
    return (
        f"NASA lithium-ion discharge prefix from 0 to {prefix_seconds:.0f} seconds. "
        f"Voltage changed from {values['voltage_v_t0000s']:.4f} V to "
        f"{values[f'voltage_v_t{int(prefix_seconds):04d}s']:.4f} V; "
        f"mean voltage {values['voltage_mean_v']:.4f} V; "
        f"mean absolute current {values['current_abs_mean_a']:.4f} A; "
        f"temperature changed by {values['temperature_rise_c']:.4f} C; "
        f"maximum temperature {values['temperature_max_c']:.4f} C; "
        f"prefix energy {values['prefix_energy_wh']:.4f} Wh."
    )


@torch.no_grad()
def make_semantic_embeddings(
    batteries: Dict[str, RawBattery],
    names: Sequence[str],
    prefix_seconds: float,
    feature_mode: str,
    tokenizer,
    llama: LlamaForCausalLM,
    device: torch.device,
    batch_size: int,
    cache_path: Path,
) -> Dict[str, torch.Tensor]:
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        if all(
            battery_id in cached
            and cached[battery_id].shape == (len(battery.soh), llama.config.hidden_size)
            for battery_id, battery in batteries.items()
        ):
            print(f"Loaded semantic cache: {cache_path}")
            return cached

    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    llama.eval()
    result = {}
    for battery_id, battery in batteries.items():
        prompts = [
            feature_prompt(row, names, prefix_seconds, feature_mode)
            for row in battery.features
        ]
        pieces = []
        for start in range(0, len(prompts), batch_size):
            encoded = tokenizer(
                prompts[start : start + batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            ).to(device)
            hidden = llama.model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                use_cache=False,
                return_dict=True,
            ).last_hidden_state
            last = encoded["attention_mask"].sum(dim=1) - 1
            rows = torch.arange(hidden.shape[0], device=device)
            pieces.append(hidden[rows, last].detach().cpu().half())
        result[battery_id] = torch.cat(pieces)
        print(f"Embedded raw features for {battery_id}: {len(prompts)} cycles")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, cache_path)
    return result


class RawWindowDataset(Dataset):
    def __init__(
        self,
        batteries: Dict[str, RawBattery],
        battery_ids: Sequence[str],
        seq_len: int,
        scaler: StandardScaler,
        semantic_embeddings: Dict[str, torch.Tensor] | None,
    ) -> None:
        self.batteries = batteries
        self.seq_len = seq_len
        self.scaler = scaler
        self.semantic_embeddings = semantic_embeddings
        self.samples = []
        for battery_id in battery_ids:
            self.samples.extend(
                (battery_id, start)
                for start in range(len(batteries[battery_id].soh) - seq_len + 1)
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        battery_id, start = self.samples[index]
        battery = self.batteries[battery_id]
        stop = start + self.seq_len
        features = self.scaler.transform(battery.features[start:stop]).astype(np.float32)
        target = battery.soh[start:stop]
        marks = (
            torch.empty(0)
            if self.semantic_embeddings is None
            else self.semantic_embeddings[battery_id][start:stop].float()
        )
        return (
            torch.from_numpy(features.copy()),
            torch.from_numpy(target.copy()),
            marks,
            {
                "battery_id": battery_id,
                "discharge_number": int(battery.discharge_number[stop - 1]),
                "source_cycle_number": int(battery.source_cycle_number[stop - 1]),
            },
        )


class RawFeatureSMETimes(nn.Module):
    def __init__(self, llama: LlamaForCausalLM, feature_dim: int, use_semantic: bool) -> None:
        super().__init__()
        self.llama = llama
        self.use_semantic = use_semantic
        hidden = llama.config.hidden_size
        self.encoder = nn.Sequential(nn.Linear(feature_dim, 256), nn.GELU(), nn.Linear(256, hidden))
        self.numeric_norm = nn.LayerNorm(hidden)
        self.semantic_norm = nn.LayerNorm(hidden)
        self.fusion_gate = nn.Parameter(torch.zeros(())) if use_semantic else None
        self.adapter = nn.Sequential(nn.Linear(hidden, 128), nn.GELU(), nn.Linear(128, hidden))
        self.decoder = nn.Linear(hidden, 1)
        for parameter in self.llama.parameters():
            parameter.requires_grad = False

    def forward(self, features: torch.Tensor, marks: torch.Tensor | None = None) -> torch.Tensor:
        numeric = self.numeric_norm(self.encoder(features))
        if self.use_semantic:
            if marks is None or marks.numel() == 0 or marks.shape != numeric.shape:
                raise ValueError("Semantic embedding shape does not match numeric tokens.")
            alpha = torch.sigmoid(self.fusion_gate)
            semantic = self.semantic_norm(marks.to(numeric.dtype))
            inputs = alpha * numeric + (1.0 - alpha) * semantic
        else:
            inputs = numeric
        hidden = self.llama.model(
            inputs_embeds=inputs,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        hidden = hidden + self.adapter(hidden)
        # NASA SOH targets in this group lie between roughly 0.56 and 1.0.
        # Constraining the output prevents catastrophic cross-cell extrapolation.
        return 0.5 + 0.5 * torch.sigmoid(self.decoder(hidden).squeeze(-1))

    def trainable_state_dict(self):
        return {
            key: value.detach().cpu()
            for key, value in self.state_dict().items()
            if not key.startswith("llama.")
        }

    def load_trainable_state_dict(self, state) -> None:
        missing, unexpected = self.load_state_dict(state, strict=False)
        bad = [key for key in missing if not key.startswith("llama.")]
        if bad or unexpected:
            raise RuntimeError(f"Checkpoint mismatch: missing={bad}, unexpected={unexpected}")


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def evaluate(model, loader, device, use_amp):
    model.eval()
    predictions, targets, metadata = [], [], []
    for features, target, marks, batch_metadata in loader:
        features = features.to(device, non_blocking=True)
        marks = marks.to(device, non_blocking=True) if marks.numel() else None
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            output = model(features, marks)
        predictions.append(output[:, -1].float().cpu().numpy())
        targets.append(target[:, -1].numpy())
        metadata.extend(
            zip(
                batch_metadata["battery_id"],
                batch_metadata["discharge_number"].tolist(),
                batch_metadata["source_cycle_number"].tolist(),
            )
        )
    prediction = np.concatenate(predictions)
    truth = np.concatenate(targets)
    return compute_metrics(prediction, truth), prediction, truth, metadata


def fit_model(model, train_loader, val_loader, device, epochs, lr, patience, use_amp):
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=lr
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    criterion = nn.MSELoss()
    best_state, best_rmse, stale, history = None, float("inf"), 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        model.llama.eval()
        loss_sum = 0.0
        count = 0
        for features, target, marks, _ in train_loader:
            features = features.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            marks = marks.to(device, non_blocking=True) if marks.numel() else None
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                output = model(features, marks)
                loss = criterion(output, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item() * features.shape[0]
            count += features.shape[0]
        val_metrics, _, _, _ = evaluate(model, val_loader, device, use_amp)
        row = {"epoch": epoch, "train_mse": loss_sum / count, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(
            f"epoch={epoch:03d} train_mse={row['train_mse']:.7f} "
            f"val_rmse={row['val_rmse']:.6f} val_mae={row['val_mae']:.6f}"
        )
        if row["val_rmse"] < best_rmse - 1e-6:
            best_rmse = row["val_rmse"]
            best_state = copy.deepcopy(model.trainable_state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"Early stopping after {epoch} epochs")
                break
    if best_state is None:
        raise RuntimeError("No model checkpoint was produced.")
    model.load_trainable_state_dict(best_state)
    return model, history


def save_feature_table(output_path: Path, batteries: Dict[str, RawBattery], names: Sequence[str]) -> None:
    rows = []
    for battery in batteries.values():
        for index, feature in enumerate(battery.features):
            rows.append(
                {
                    "battery_id": battery.battery_id,
                    "discharge_number": int(battery.discharge_number[index]),
                    "source_cycle_number": int(battery.source_cycle_number[index]),
                    **{name: float(value) for name, value in zip(names, feature)},
                    "capacity_ah_target_only": float(battery.capacity_ah[index]),
                    "soh_target": float(battery.soh[index]),
                }
            )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def plot_result(path: Path, discharge: np.ndarray, truth: np.ndarray, predictions: Dict[str, np.ndarray]) -> None:
    plt.figure(figsize=(10, 5.5))
    plt.plot(discharge, truth, color="black", linewidth=2, label="True SOH")
    colors = {
        "cycle_linear": "#888888",
        "ridge": "#d95f02",
        "extra_trees": "#984ea3",
        "random_forest": "#e7298a",
        "numeric": "#1b9e77",
        "semantic": "#386cb0",
    }
    for name, values in predictions.items():
        plt.plot(discharge, values, linewidth=1.5, label=name, color=colors.get(name))
    plt.axhline(0.8, color="#b2182b", linestyle="--", linewidth=1, label="80% SOH")
    plt.xlabel("Discharge observation")
    plt.ylabel("SOH")
    plt.title("NASA B0018 SOH estimation from first 1000 s of each discharge")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def build_parser():
    parser = argparse.ArgumentParser(description="Raw NASA MAT SOH estimation MVP")
    parser.add_argument("--raw_root", type=Path, default=Path(r"D:\LLM\NASA\raw\extracted"))
    parser.add_argument("--llm_ckp_dir", default=r".\models\llm\llama-3.2-1b-instruct")
    parser.add_argument("--output_dir", type=Path, default=Path("battery_raw_mvp_results"))
    parser.add_argument("--train_ids", default="B0005,B0006")
    parser.add_argument("--val_ids", default="B0007")
    parser.add_argument("--test_ids", default="B0018")
    parser.add_argument("--prefix_seconds", type=float, default=1000.0)
    parser.add_argument("--sample_interval", type=float, default=100.0)
    parser.add_argument("--feature_mode", choices=["raw", "delta"], default="delta")
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--embedding_batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--variants", default="numeric,semantic")
    return parser


def main():
    args = build_parser().parse_args()
    set_seed(args.seed)
    train_ids, val_ids, test_ids = map(parse_ids, (args.train_ids, args.val_ids, args.test_ids))
    if set(train_ids) & set(val_ids + test_ids) or set(val_ids) & set(test_ids):
        raise ValueError("Train, validation, and test battery ids must be disjoint.")
    ids = list(dict.fromkeys(train_ids + val_ids + test_ids))
    sample_seconds = np.arange(0.0, args.prefix_seconds + 0.5 * args.sample_interval, args.sample_interval)
    if sample_seconds[-1] != args.prefix_seconds:
        raise ValueError("prefix_seconds must be divisible by sample_interval")
    names = feature_names(sample_seconds)
    batteries = {
        battery_id: load_raw_battery(args.raw_root, battery_id, sample_seconds, args.prefix_seconds)
        for battery_id in ids
    }
    if args.feature_mode == "delta":
        batteries = {
            battery_id: replace(battery, features=battery.features - battery.features[0])
            for battery_id, battery in batteries.items()
        }
    print("Raw discharge observations:", {key: len(value.soh) for key, value in batteries.items()})
    print(
        "Feature count:",
        len(names),
        f"mode={args.feature_mode}; capacity and total duration excluded",
    )

    scaler = StandardScaler().fit(np.concatenate([batteries[key].features for key in train_ids]))
    args.use_multi_gpu = False
    args.use_amp = not args.no_amp
    device = resolve_device(args)
    use_amp = args.use_amp and device.type == "cuda"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_feature_table(
        args.output_dir / f"extracted_prefix_features_{args.feature_mode}.csv",
        batteries,
        names,
    )

    variants = [value.strip().lower() for value in args.variants.split(",") if value.strip()]
    if any(value not in {"numeric", "semantic"} for value in variants):
        raise ValueError("--variants must contain numeric and/or semantic")

    llama = None
    semantic_embeddings = None
    if variants:
        model_path = resolve_llm_ckp_dir(args.llm_ckp_dir)
        llama = LlamaForCausalLM.from_pretrained(
            model_path, dtype=model_dtype(args, device), local_files_only=True
        ).to(device)
        llama.config.use_cache = False
        for parameter in llama.parameters():
            parameter.requires_grad = False
        if "semantic" in variants:
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            signature = raw_signature(batteries, args.prefix_seconds, model_path)
            semantic_embeddings = make_semantic_embeddings(
                batteries,
                names,
                args.prefix_seconds,
                args.feature_mode,
                tokenizer,
                llama,
                device,
                args.embedding_batch_size,
                args.output_dir / f"semantic_embeddings_{signature}.pt",
            )

    datasets = {}
    for semantic in (False, True):
        marks = semantic_embeddings if semantic else None
        key = "semantic" if semantic else "numeric"
        datasets[key] = {
            "train": RawWindowDataset(batteries, train_ids, args.seq_len, scaler, marks),
            "val": RawWindowDataset(batteries, val_ids, args.seq_len, scaler, marks),
            "test": RawWindowDataset(batteries, test_ids, args.seq_len, scaler, marks),
        }
    print("Window counts:", {key: len(value) for key, value in datasets["numeric"].items()})

    test_battery = batteries[test_ids[0]]
    test_slice = slice(args.seq_len - 1, None)
    x_train = np.concatenate([batteries[key].features for key in train_ids])
    y_train = np.concatenate([batteries[key].soh for key in train_ids])
    x_test = test_battery.features[test_slice]
    truth = test_battery.soh[test_slice]
    discharge = test_battery.discharge_number[test_slice]

    ridge = Ridge(alpha=1.0).fit(scaler.transform(x_train), y_train)
    ridge_prediction = ridge.predict(scaler.transform(x_test))
    extra_trees = ExtraTreesRegressor(
        n_estimators=300,
        min_samples_leaf=3,
        max_features=0.8,
        random_state=args.seed,
        n_jobs=-1,
    ).fit(x_train, y_train)
    extra_trees_prediction = extra_trees.predict(x_test)
    random_forest = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=3,
        max_features=0.8,
        random_state=args.seed,
        n_jobs=-1,
    ).fit(x_train, y_train)
    random_forest_prediction = random_forest.predict(x_test)
    cycle_train = np.concatenate([batteries[key].discharge_number for key in train_ids]).reshape(-1, 1)
    cycle_scaler = StandardScaler().fit(cycle_train)
    cycle_model = Ridge(alpha=1.0).fit(cycle_scaler.transform(cycle_train), y_train)
    cycle_prediction = cycle_model.predict(cycle_scaler.transform(discharge.reshape(-1, 1)))
    joblib.dump(
        {
            "model": random_forest,
            "feature_names": names,
            "feature_mode": args.feature_mode,
            "prefix_seconds": args.prefix_seconds,
            "sample_interval": args.sample_interval,
            "training_batteries": train_ids,
        },
        args.output_dir / "random_forest_soh.joblib",
    )
    joblib.dump({"model": extra_trees, "feature_names": names}, args.output_dir / "extra_trees_soh.joblib")
    joblib.dump({"model": ridge, "scaler": scaler, "feature_names": names}, args.output_dir / "ridge_soh.joblib")
    metrics = {
        "cycle_linear": compute_metrics(cycle_prediction, truth),
        "ridge": compute_metrics(ridge_prediction, truth),
        "extra_trees": compute_metrics(extra_trees_prediction, truth),
        "random_forest": compute_metrics(random_forest_prediction, truth),
    }
    predictions = {
        "cycle_linear": cycle_prediction,
        "ridge": ridge_prediction,
        "extra_trees": extra_trees_prediction,
        "random_forest": random_forest_prediction,
    }
    metadata = None

    for variant in variants:
        print(f"\n=== Training raw-feature {variant} SMETimes ===")
        set_seed(args.seed)
        model = RawFeatureSMETimes(llama, len(names), variant == "semantic").to(device)
        model, history = fit_model(
            model,
            make_loader(datasets[variant]["train"], args.batch_size, True),
            make_loader(datasets[variant]["val"], args.batch_size, False),
            device,
            args.epochs,
            args.learning_rate,
            args.patience,
            use_amp,
        )
        test_metrics, prediction, model_truth, metadata = evaluate(
            model, make_loader(datasets[variant]["test"], args.batch_size, False), device, use_amp
        )
        if not np.allclose(model_truth, truth):
            raise RuntimeError("Test target alignment failure")
        metrics[variant] = test_metrics
        predictions[variant] = prediction
        torch.save(
            {
                "variant": variant,
                "trainable_state_dict": model.trainable_state_dict(),
                "feature_names": names,
                "scaler_mean": scaler.mean_,
                "scaler_scale": scaler.scale_,
                "history": history,
                "config": vars(args),
            },
            args.output_dir / f"checkpoint_{variant}.pt",
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if metadata is None:
        metadata = list(
            zip(
                [test_ids[0]] * len(truth),
                discharge.astype(int).tolist(),
                test_battery.source_cycle_number[test_slice].astype(int).tolist(),
            )
        )

    with (args.output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["battery_id", "discharge_number", "source_cycle_number", "true_soh"] + [f"pred_{key}" for key in predictions])
        for index, item in enumerate(metadata):
            writer.writerow([item[0], item[1], item[2], truth[index]] + [predictions[key][index] for key in predictions])
    results = {
        "task": "current-cycle SOH estimation from first 1000 seconds of raw discharge curves",
        "leakage_controls": [
            "Capacity is target only and excluded from features/prompts.",
            "Total discharge duration is excluded.",
            "Feature scaler is fit on B0005+B0006 only.",
            "Battery-level train/validation/test split is used.",
        ],
        "split": {"train": train_ids, "validation": val_ids, "test": test_ids},
        "feature_count": len(names),
        "feature_mode": args.feature_mode,
        "prefix_seconds": args.prefix_seconds,
        "seed": args.seed,
        "source_files": {key: value.source_path for key, value in batteries.items()},
        "metrics": metrics,
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2, default=str)
    plot_result(args.output_dir / "b0018_raw_feature_soh.png", discharge, truth, predictions)
    print("\n=== B0018 raw-feature SOH metrics ===")
    for name, values in metrics.items():
        print(f"{name:>12} MAE={values['mae']:.6f} RMSE={values['rmse']:.6f} MAPE={values['mape_pct']:.3f}% R2={values['r2']:.4f}")
    print(f"Artifacts written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
