"""Minimal NASA battery SOH forecasting experiment for SMETimes.

The experiment intentionally treats each battery as an indivisible group:
B0005+B0006 train, B0007 validation, and B0018 test by default.  It forecasts
the next ``token_len`` SOH values from the previous ``seq_len`` values and
compares two frozen-LLM variants with persistence and linear-trend baselines.

The source Excel workbooks are read only.  Derived artifacts are written to a
separate results directory.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, LlamaForCausalLM

from utils.device import model_dtype, resolve_device
from utils.llm import resolve_llm_ckp_dir


@dataclass(frozen=True)
class BatterySeries:
    battery_id: str
    discharge_number: np.ndarray
    cycle_number: np.ndarray
    capacity_ah: np.ndarray
    soh: np.ndarray
    source_path: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_ids(text: str) -> List[str]:
    values = [item.strip().upper() for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one battery id is required.")
    return values


def load_battery(data_root: Path, battery_id: str) -> BatterySeries:
    """Load the canonical workbook, deliberately ignoring ``(1)`` duplicates."""
    path = data_root / battery_id / f"{battery_id}_data.xlsx"
    if not path.exists():
        raise FileNotFoundError(f"Missing workbook: {path}")

    frame = pd.read_excel(path, sheet_name="放电容量", engine="openpyxl")
    required = ["DischargeNumber", "CycleNumber", "Capacity_Ah"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    frame = frame[required].dropna().sort_values("DischargeNumber")
    if frame.empty:
        raise ValueError(f"{path} has no discharge-capacity observations.")
    if frame["DischargeNumber"].duplicated().any():
        raise ValueError(f"{path} contains duplicate discharge numbers.")

    capacity = frame["Capacity_Ah"].to_numpy(dtype=np.float32)
    if not np.all(np.isfinite(capacity)) or np.any(capacity <= 0):
        raise ValueError(f"{path} contains invalid capacity values.")
    soh = capacity / capacity[0]

    return BatterySeries(
        battery_id=battery_id,
        discharge_number=frame["DischargeNumber"].to_numpy(dtype=np.int64),
        cycle_number=frame["CycleNumber"].to_numpy(dtype=np.int64),
        capacity_ah=capacity,
        soh=soh.astype(np.float32),
        source_path=str(path.resolve()),
    )


def load_all_batteries(data_root: Path, ids: Iterable[str]) -> Dict[str, BatterySeries]:
    return {battery_id: load_battery(data_root, battery_id) for battery_id in ids}


def build_prompt(series: BatterySeries, start: int, token_len: int) -> str:
    stop = start + token_len
    values = series.soh[start:stop]
    capacities = series.capacity_ah[start:stop]
    return (
        f"Lithium-ion cell {series.battery_id}. "
        f"Discharge observations {int(series.discharge_number[start])} to "
        f"{int(series.discharge_number[stop - 1])}. "
        f"SOH changed from {values[0]:.5f} to {values[-1]:.5f}; "
        f"mean SOH {values.mean():.5f}; standard deviation {values.std():.5f}; "
        f"capacity changed by {capacities[-1] - capacities[0]:.5f} Ah."
    )


def series_signature(series_map: Dict[str, BatterySeries], token_len: int, model_path: str) -> str:
    digest = hashlib.sha256()
    digest.update(str(token_len).encode())
    digest.update(str(Path(model_path).resolve()).encode())
    for battery_id in sorted(series_map):
        series = series_map[battery_id]
        digest.update(battery_id.encode())
        digest.update(series.capacity_ah.tobytes())
    return digest.hexdigest()[:16]


@torch.no_grad()
def make_prompt_embeddings(
    series_map: Dict[str, BatterySeries],
    token_len: int,
    tokenizer,
    llama: LlamaForCausalLM,
    device: torch.device,
    batch_size: int,
    cache_path: Path,
) -> Dict[str, torch.Tensor]:
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        expected = {
            battery_id: len(series.soh) - token_len + 1
            for battery_id, series in series_map.items()
        }
        if all(
            battery_id in cached
            and cached[battery_id].shape[0] == count
            and cached[battery_id].shape[1] == llama.config.hidden_size
            for battery_id, count in expected.items()
        ):
            print(f"Loaded prompt embedding cache: {cache_path}")
            return cached

    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    llama.eval()
    result: Dict[str, torch.Tensor] = {}
    for battery_id, series in series_map.items():
        prompts = [
            build_prompt(series, start, token_len)
            for start in range(len(series.soh) - token_len + 1)
        ]
        pieces = []
        for offset in range(0, len(prompts), batch_size):
            encoded = tokenizer(
                prompts[offset : offset + batch_size],
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
            last_index = encoded["attention_mask"].sum(dim=1) - 1
            row_index = torch.arange(hidden.shape[0], device=device)
            pieces.append(hidden[row_index, last_index].detach().cpu().half())
        result[battery_id] = torch.cat(pieces, dim=0)
        print(f"Embedded {battery_id}: {len(prompts)} degradation prompts")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, cache_path)
    print(f"Saved prompt embedding cache: {cache_path}")
    return result


class BatteryWindowDataset(Dataset):
    def __init__(
        self,
        series_map: Dict[str, BatterySeries],
        battery_ids: Sequence[str],
        seq_len: int,
        token_len: int,
        stride: int,
        prompt_embeddings: Dict[str, torch.Tensor] | None,
    ) -> None:
        if seq_len % token_len:
            raise ValueError("seq_len must be divisible by token_len")
        self.series_map = series_map
        self.seq_len = seq_len
        self.token_len = token_len
        self.prompt_embeddings = prompt_embeddings
        self.samples = []
        for battery_id in battery_ids:
            length = len(series_map[battery_id].soh)
            last_start = length - seq_len - token_len
            if last_start < 0:
                raise ValueError(
                    f"{battery_id} has {length} observations, fewer than "
                    f"seq_len + token_len = {seq_len + token_len}."
                )
            self.samples.extend(
                (battery_id, start) for start in range(0, last_start + 1, stride)
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        battery_id, start = self.samples[index]
        series = self.series_map[battery_id]
        source = series.soh[start : start + self.seq_len]
        # Each decoded token predicts the following token-sized block.
        target = series.soh[
            start + self.token_len : start + self.seq_len + self.token_len
        ]
        if self.prompt_embeddings is None:
            marks = torch.empty(0)
        else:
            positions = range(start, start + self.seq_len, self.token_len)
            marks = self.prompt_embeddings[battery_id][list(positions)].float()
        future_start = start + self.seq_len
        metadata = {
            "battery_id": battery_id,
            "window_start": start,
            "future_discharge": torch.as_tensor(
                series.discharge_number[
                    future_start : future_start + self.token_len
                ].copy(),
                dtype=torch.long,
            ),
        }
        return (
            torch.as_tensor(source, dtype=torch.float32),
            torch.as_tensor(target, dtype=torch.float32),
            marks,
            metadata,
        )


class BatterySMETimes(nn.Module):
    def __init__(
        self,
        llama: LlamaForCausalLM,
        token_len: int,
        use_semantic: bool,
        num_experts: int = 1,
    ) -> None:
        super().__init__()
        self.llama = llama
        self.token_len = token_len
        self.use_semantic = use_semantic
        hidden_size = llama.config.hidden_size
        self.encoder = nn.Linear(token_len, hidden_size)
        self.decoder = nn.Linear(hidden_size, token_len)
        self.fusion_gate = nn.Parameter(torch.zeros(())) if use_semantic else None
        self.experts = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(num_experts)]
        )
        self.expert_gate = nn.Linear(hidden_size, num_experts)
        for parameter in self.llama.parameters():
            parameter.requires_grad = False

    def forward(self, x: torch.Tensor, marks: torch.Tensor | None = None) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True).detach()
        stdev = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + 1e-5)
        normalized = (x - mean) / stdev
        chunks = normalized.unfold(1, self.token_len, self.token_len)
        numeric = self.encoder(chunks)
        if self.use_semantic:
            if marks is None or marks.numel() == 0:
                raise ValueError("Semantic SMETimes requires prompt embeddings.")
            if marks.shape != numeric.shape:
                raise ValueError(f"Prompt shape {marks.shape} != numeric shape {numeric.shape}")
            alpha = torch.sigmoid(self.fusion_gate)
            inputs = alpha * numeric + (1.0 - alpha) * marks.to(numeric.dtype)
        else:
            inputs = numeric

        hidden = self.llama.model(
            inputs_embeds=inputs,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        weights = torch.softmax(self.expert_gate(hidden), dim=-1)
        expert_outputs = torch.stack([expert(hidden) for expert in self.experts], dim=-2)
        hidden = (weights.unsqueeze(-1) * expert_outputs).sum(dim=-2)
        decoded = self.decoder(hidden).flatten(start_dim=1)
        return decoded * stdev + mean

    def trainable_state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            name: value.detach().cpu()
            for name, value in self.state_dict().items()
            if not name.startswith("llama.")
        }

    def load_trainable_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        missing, unexpected = self.load_state_dict(state, strict=False)
        bad_missing = [name for name in missing if not name.startswith("llama.")]
        if bad_missing or unexpected:
            raise RuntimeError(f"Bad checkpoint; missing={bad_missing}, unexpected={unexpected}")


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def compute_metrics(prediction: np.ndarray, truth: np.ndarray) -> Dict[str, float]:
    prediction = prediction.reshape(-1).astype(np.float64)
    truth = truth.reshape(-1).astype(np.float64)
    error = prediction - truth
    mae = np.mean(np.abs(error))
    rmse = math.sqrt(np.mean(error**2))
    mape = np.mean(np.abs(error) / np.maximum(np.abs(truth), 1e-8)) * 100.0
    denominator = np.sum((truth - truth.mean()) ** 2)
    r2 = 1.0 - np.sum(error**2) / denominator if denominator > 0 else float("nan")
    return {"mae": float(mae), "rmse": float(rmse), "mape_pct": float(mape), "r2": float(r2)}


@torch.no_grad()
def evaluate(
    model: BatterySMETimes,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> tuple[Dict[str, float], np.ndarray, np.ndarray, list]:
    model.eval()
    predictions, truths, metadata_rows = [], [], []
    for x, target, marks, metadata in loader:
        x = x.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        marks = marks.to(device, non_blocking=True) if marks.numel() else None
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            output = model(x, marks)
        prediction = output[:, -model.token_len :].float().cpu().numpy()
        truth = target[:, -model.token_len :].float().cpu().numpy()
        predictions.append(prediction)
        truths.append(truth)
        ids = metadata["battery_id"]
        starts = metadata["window_start"].tolist()
        discharge = metadata["future_discharge"].tolist()
        metadata_rows.extend(zip(ids, starts, discharge))
    prediction = np.concatenate(predictions)
    truth = np.concatenate(truths)
    return compute_metrics(prediction, truth), prediction, truth, metadata_rows


def fit_model(
    model: BatterySMETimes,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    patience: int,
    use_amp: bool,
) -> tuple[BatterySMETimes, List[dict]]:
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    criterion = nn.MSELoss()
    best_state = None
    best_val = float("inf")
    stale = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        model.llama.eval()
        total_loss = 0.0
        total_count = 0
        for x, target, marks, _ in train_loader:
            x = x.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            marks = marks.to(device, non_blocking=True) if marks.numel() else None
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                output = model(x, marks)
                loss = criterion(output, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * x.shape[0]
            total_count += x.shape[0]

        val_metrics, _, _, _ = evaluate(model, val_loader, device, use_amp)
        row = {
            "epoch": epoch,
            "train_mse": total_loss / total_count,
            "val_rmse": val_metrics["rmse"],
            "val_mae": val_metrics["mae"],
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} train_mse={row['train_mse']:.7f} "
            f"val_rmse={row['val_rmse']:.6f} val_mae={row['val_mae']:.6f}"
        )
        if row["val_rmse"] < best_val - 1e-6:
            best_val = row["val_rmse"]
            best_state = copy.deepcopy(model.trainable_state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"Early stopping after {epoch} epochs")
                break

    if best_state is None:
        raise RuntimeError("Training produced no checkpoint.")
    model.load_trainable_state_dict(best_state)
    return model, history


def baseline_predictions(dataset: BatteryWindowDataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    persistence, linear, truths = [], [], []
    for index in range(len(dataset)):
        x, target, _, _ = dataset[index]
        x_np = x.numpy()
        truth = target[-dataset.token_len :].numpy()
        persistence.append(np.repeat(x_np[-1], dataset.token_len))
        fit_len = min(16, len(x_np))
        positions = np.arange(fit_len, dtype=np.float64)
        slope, intercept = np.polyfit(positions, x_np[-fit_len:], 1)
        future_positions = np.arange(fit_len, fit_len + dataset.token_len)
        linear.append(slope * future_positions + intercept)
        truths.append(truth)
    return np.stack(persistence), np.stack(linear), np.stack(truths)


def write_predictions(
    output_path: Path,
    metadata_rows: list,
    truth: np.ndarray,
    predictions: Dict[str, np.ndarray],
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["battery_id", "window_start", "horizon", "discharge_number", "true_soh"]
            + [f"pred_{name}" for name in predictions]
        )
        for sample_index, (battery_id, start, discharge_numbers) in enumerate(metadata_rows):
            for horizon, discharge_number in enumerate(discharge_numbers, start=1):
                writer.writerow(
                    [battery_id, start, horizon, discharge_number, truth[sample_index, horizon - 1]]
                    + [predictions[name][sample_index, horizon - 1] for name in predictions]
                )


def plot_predictions(
    output_path: Path,
    metadata_rows: list,
    truth: np.ndarray,
    predictions: Dict[str, np.ndarray],
) -> None:
    x = []
    y_true = []
    y_predictions = {name: [] for name in predictions}
    for sample_index, (_, _, discharge_numbers) in enumerate(metadata_rows):
        for horizon, discharge_number in enumerate(discharge_numbers):
            x.append(discharge_number)
            y_true.append(truth[sample_index, horizon])
            for name in predictions:
                y_predictions[name].append(predictions[name][sample_index, horizon])
    order = np.argsort(x)
    x_values = np.asarray(x)[order]
    plt.figure(figsize=(10, 5.5))
    plt.plot(x_values, np.asarray(y_true)[order], color="black", linewidth=2, label="True SOH")
    colors = {"persistence": "#888888", "linear": "#d95f02", "numeric": "#1b9e77", "semantic": "#386cb0"}
    for name, values in y_predictions.items():
        plt.plot(x_values, np.asarray(values)[order], linewidth=1.5, label=name, color=colors.get(name))
    plt.axhline(0.8, color="#b2182b", linestyle="--", linewidth=1, label="80% SOH")
    plt.xlabel("Discharge observation")
    plt.ylabel("SOH (capacity / first capacity)")
    plt.title("NASA B0018 rolling-origin 8-cycle SOH forecast")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NASA battery SMETimes MVP")
    parser.add_argument("--data_root", type=Path, default=Path(r"D:\LLM\NASA\数据集"))
    parser.add_argument("--llm_ckp_dir", default=r".\models\llm\llama-3.2-1b-instruct")
    parser.add_argument("--output_dir", type=Path, default=Path("battery_mvp_results"))
    parser.add_argument("--train_ids", default="B0005,B0006")
    parser.add_argument("--val_ids", default="B0007")
    parser.add_argument("--test_ids", default="B0018")
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--token_len", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--embedding_batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--variants", default="numeric,semantic", help="numeric, semantic, or both")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    train_ids = parse_ids(args.train_ids)
    val_ids = parse_ids(args.val_ids)
    test_ids = parse_ids(args.test_ids)
    all_ids = list(dict.fromkeys(train_ids + val_ids + test_ids))
    if set(train_ids) & set(val_ids + test_ids) or set(val_ids) & set(test_ids):
        raise ValueError("Train, validation, and test battery ids must be disjoint.")

    args.use_multi_gpu = False
    args.use_amp = not args.no_amp
    device = resolve_device(args)
    use_amp = args.use_amp and device.type == "cuda"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    series_map = load_all_batteries(args.data_root, all_ids)
    print("Battery split:", {"train": train_ids, "val": val_ids, "test": test_ids})
    print("Observations:", {key: len(value.soh) for key, value in series_map.items()})

    model_path = resolve_llm_ckp_dir(args.llm_ckp_dir)
    print(f"Loading frozen Llama from {model_path} on {device}")
    llama = LlamaForCausalLM.from_pretrained(
        model_path,
        dtype=model_dtype(args, device),
        local_files_only=True,
    ).to(device)
    llama.config.use_cache = False
    for parameter in llama.parameters():
        parameter.requires_grad = False

    variants = [value.strip().lower() for value in args.variants.split(",") if value.strip()]
    if not variants or any(value not in {"numeric", "semantic"} for value in variants):
        raise ValueError("--variants must contain numeric and/or semantic")

    prompt_embeddings = None
    if "semantic" in variants:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        signature = series_signature(series_map, args.token_len, model_path)
        cache_path = args.output_dir / f"prompt_embeddings_{signature}.pt"
        prompt_embeddings = make_prompt_embeddings(
            series_map,
            args.token_len,
            tokenizer,
            llama,
            device,
            args.embedding_batch_size,
            cache_path,
        )

    semantic_datasets = {
        "train": BatteryWindowDataset(series_map, train_ids, args.seq_len, args.token_len, 1, prompt_embeddings),
        "val": BatteryWindowDataset(series_map, val_ids, args.seq_len, args.token_len, args.token_len, prompt_embeddings),
        "test": BatteryWindowDataset(series_map, test_ids, args.seq_len, args.token_len, args.token_len, prompt_embeddings),
    }
    numeric_datasets = {
        name: BatteryWindowDataset(
            series_map,
            train_ids if name == "train" else val_ids if name == "val" else test_ids,
            args.seq_len,
            args.token_len,
            1 if name == "train" else args.token_len,
            None,
        )
        for name in ("train", "val", "test")
    }
    print("Window counts:", {name: len(ds) for name, ds in numeric_datasets.items()})

    persistence, linear, truth = baseline_predictions(numeric_datasets["test"])
    metrics = {
        "persistence": compute_metrics(persistence, truth),
        "linear": compute_metrics(linear, truth),
    }
    all_predictions: Dict[str, np.ndarray] = {
        "persistence": persistence,
        "linear": linear,
    }
    metadata_rows = None

    for variant in variants:
        print(f"\n=== Training {variant} SMETimes ===")
        set_seed(args.seed)
        datasets = semantic_datasets if variant == "semantic" else numeric_datasets
        model = BatterySMETimes(llama, args.token_len, use_semantic=(variant == "semantic")).to(device)
        model, history = fit_model(
            model,
            make_loader(datasets["train"], args.batch_size, True),
            make_loader(datasets["val"], args.batch_size, False),
            device,
            args.epochs,
            args.learning_rate,
            args.patience,
            use_amp,
        )
        test_metrics, prediction, model_truth, model_metadata = evaluate(
            model,
            make_loader(datasets["test"], args.batch_size, False),
            device,
            use_amp,
        )
        if not np.allclose(model_truth, truth):
            raise RuntimeError("Model and baseline test targets are misaligned.")
        metrics[variant] = test_metrics
        all_predictions[variant] = prediction
        metadata_rows = model_metadata
        torch.save(
            {
                "variant": variant,
                "trainable_state_dict": model.trainable_state_dict(),
                "config": vars(args),
                "history": history,
            },
            args.output_dir / f"checkpoint_{variant}.pt",
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if metadata_rows is None:
        # Build metadata without a model if only baselines were requested in a future extension.
        metadata_rows = []
        for index in range(len(numeric_datasets["test"])):
            _, _, _, metadata = numeric_datasets["test"][index]
            metadata_rows.append((metadata["battery_id"], metadata["window_start"], metadata["future_discharge"].tolist()))

    results = {
        "task": "8-cycle SOH forecasting",
        "soh_definition": "Capacity_Ah / first observed Capacity_Ah for each cell",
        "split": {"train": train_ids, "validation": val_ids, "test": test_ids},
        "seq_len": args.seq_len,
        "token_len": args.token_len,
        "seed": args.seed,
        "source_files": {key: value.source_path for key, value in series_map.items()},
        "metrics": metrics,
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2, default=str)
    write_predictions(args.output_dir / "predictions.csv", metadata_rows, truth, all_predictions)
    plot_predictions(args.output_dir / "b0018_forecast.png", metadata_rows, truth, all_predictions)

    print("\n=== B0018 test metrics ===")
    for name, values in metrics.items():
        print(
            f"{name:>12}  MAE={values['mae']:.6f}  RMSE={values['rmse']:.6f}  "
            f"MAPE={values['mape_pct']:.3f}%  R2={values['r2']:.4f}"
        )
    print(f"Artifacts written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
