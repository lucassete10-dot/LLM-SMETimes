"""Whole-cell few-shot target calibration of frozen zero-shot source models.

The percentage is a budget of measured target SOH diagnostic labels, not of
cells. Whole target cells are reserved for calibration and never scored.
The remaining target cells are scored for every method on the same split.
"""

import json
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np

from battery_smetimes.models.domain_adapter import (apply_offset,
                                                      fit_cell_balanced_offset)
from scripts.evaluate_cross_dataset_soh_v3 import load_cells


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "battery_models_v3" / "cross_dataset_shared_soh"
DOMAINS = ("MIT", "XJTU")
LABEL_FRACTIONS = (.01, .05, .10)
SEEDS = tuple(range(2026, 2036))


def choose_calibration(predicted, fraction, seed, prefix_limit=None):
    """Choose whole cells without label-value access, then exact row budget.

    `prefix_limit=100` restricts each calibration cell to its first 100 scored
    diagnostics. It uses no true-life fraction or future endpoint to locate
    those rows; more distinct calibration cells may be needed for the budget.
    """
    if not 0 < fraction < 1 or len(predicted) < 4:
        raise ValueError("need a valid fraction and at least four target cells")
    total = sum(len(target) for _, target in predicted.values())
    budget = max(1, int(round(total * fraction)))
    keys = sorted(predicted)
    order = [keys[i] for i in np.random.default_rng(seed).permutation(len(keys))]
    selected, rows = [], 0
    for key in order:
        if len(selected) >= len(keys) - 2:
            break
        selected.append(key)
        rows += min(len(predicted[key][1]), prefix_limit or len(predicted[key][1]))
        if len(selected) >= 2 and rows >= budget:
            break
    if rows < budget:
        raise ValueError("insufficient calibration rows while preserving test cells")
    pool = [(key, i) for key in selected
            for i in range(min(len(predicted[key][1]),
                               prefix_limit or len(predicted[key][1])))]
    sampled = [pool[i] for i in np.linspace(0, len(pool) - 1, budget).astype(int)]
    by_cell = defaultdict(list)
    for key, index in sampled:
        by_cell[key].append(index)
    calibration = {}
    for key in selected:
        if key in by_cell:
            index = np.asarray(by_cell[key], dtype=int)
            base, target = predicted[key]
            calibration[key] = (base[index], target[index])
    if set(calibration) != set(selected):
        raise ValueError("chosen calibration cell received no labels")
    evaluation = [key for key in keys if key not in selected]
    if set(evaluation) & set(calibration):
        raise AssertionError("calibration/evaluation cell overlap")
    return calibration, evaluation, budget, total


def score(predicted, keys, offset=None, baseline=False):
    per_cell = []
    for key in keys:
        base, target = predicted[key]
        estimate = (np.ones_like(target) if baseline else
                    base if offset is None else apply_offset(base, offset))
        absolute = np.abs(estimate - target)
        per_cell.append({"cell_id": key,
                         "mape_percent": float(np.mean(absolute / target) * 100),
                         "mae_soh_percentage_points": float(absolute.mean() * 100)})
    values = np.asarray([row["mape_percent"] for row in per_cell])
    return {"cell_macro_mape_percent": float(values.mean()),
            "cell_p90_mape_percent": float(np.percentile(values, 90)),
            "worst_cell_mape_percent": float(values.max()),
            "per_cell": per_cell}


def evaluate_domain(cells, domain):
    bundle = joblib.load(MODEL_DIR / f"heldout_{domain.lower()}.joblib")
    if bundle["held_out_dataset"] != domain or any(
            key[0] == domain for key in bundle["train_cells"]):
        raise ValueError("frozen model includes target-domain cells")
    predicted = {}
    for key in sorted(cells):
        if key[0] != domain:
            continue
        x, y, mask = cells[key]
        base = bundle["model"].predict(bundle["imputer"].transform(x[mask]))
        predicted[key[1]] = (base, y[mask])
    results = {}
    for policy_name, prefix_limit in (("full_history", None), ("first_100_diagnostics", 100)):
        result = {}
        for fraction in LABEL_FRACTIONS:
            seeds = []
            for seed in SEEDS:
                calibration, evaluation, budget, total = choose_calibration(
                    predicted, fraction, seed, prefix_limit=prefix_limit)
                offset = fit_cell_balanced_offset(calibration)
                seeds.append({
                    "seed": seed,
                    "requested_label_fraction": fraction,
                    "actual_label_fraction": budget / total,
                    "labeled_diagnostics": budget,
                    "target_diagnostics": total,
                    "calibration_cell_ids": sorted(calibration),
                    "test_cell_ids": sorted(evaluation),
                    "offset_soh_fraction": offset,
                    "zero_shot": score(predicted, evaluation),
                    "target_offset": score(predicted, evaluation, offset=offset),
                    "reference_soh_1": score(predicted, evaluation, baseline=True),
                })
            result[str(int(fraction * 100)) + "pct"] = {
                "repeats": seeds,
                "mean_zero_shot_mape_percent": float(np.mean([
                    row["zero_shot"]["cell_macro_mape_percent"] for row in seeds])),
                "mean_offset_mape_percent": float(np.mean([
                    row["target_offset"]["cell_macro_mape_percent"] for row in seeds])),
                "mean_reference_mape_percent": float(np.mean([
                    row["reference_soh_1"]["cell_macro_mape_percent"] for row in seeds])),
                "offset_better_than_zero_shot_repeats": sum(
                    row["target_offset"]["cell_macro_mape_percent"] <
                    row["zero_shot"]["cell_macro_mape_percent"] for row in seeds),
                "offset_better_than_reference_repeats": sum(
                    row["target_offset"]["cell_macro_mape_percent"] <
                    row["reference_soh_1"]["cell_macro_mape_percent"] for row in seeds),
            }
        results[policy_name] = result
    return results


def main():
    cells = load_cells()
    report = {
        "status": "development few-shot transfer on inspected public domains; not final acceptance",
        "source_models": "fixed whole-dataset-held-out zero-shot ExtraTrees models",
        "adapter": "median per-calibration-cell SOH residual; slope fixed at 1",
        "selection": "seeded random whole target cells, diagnostic labels sampled uniformly without using values",
        "evaluation": "remaining whole target cells; zero-shot and baselines rescored on identical cells",
        "label_fraction_denominator": "all available capacity-SOH diagnostic labels in the target dataset",
        "scope": "MIT and XJTU; NASA/CALCE have only 4 cells, too few for meaningful percentage-based cell-disjoint study",
        "seeds": list(SEEDS), "domains": {},
    }
    for domain in DOMAINS:
        report["domains"][domain] = evaluate_domain(cells, domain)
        print(domain, {policy: {budget: (
            round(row["mean_zero_shot_mape_percent"], 3),
            round(row["mean_offset_mape_percent"], 3),
            row["offset_better_than_zero_shot_repeats"])
            for budget, row in budgets.items()}
            for policy, budgets in report["domains"][domain].items()}, flush=True)
    output = MODEL_DIR / "few_shot_offset_diagnostic.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")


if __name__ == "__main__":
    main()
