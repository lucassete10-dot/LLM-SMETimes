"""Cell-first shared-transfer reporting; no fitting, model loading or routing."""

import numpy as np

from battery_smetimes.research_baselines import regression_metrics
from battery_smetimes.research_transfer_neural_audit import assert_coverage
from battery_smetimes.research_transfer_baselines import transfer_prefix
from battery_smetimes.research_transfer_signals import RESEARCH_MODE


METRICS = ("rmse_soh", "mae_soh", "mape_percent", "r2")
HORIZONS = ("all_observed_future", "within_10", "within_50", "within_100", "late_half")
MODELS = {"baseline": ["ridge", "extra_trees"], "neural": ["gru", "statistics_mlp", "llm_e1", "llm_e2"]}
CONTRACT_FIELDS = ("prepared", "prepared_config_sha256", "prepared_manifest_sha256", "prepared_splits_sha256",
    "main_run_bindings", "origins", "seeds", "input_schema", "feature_names")


def validate_comparison_contracts(configs, plans):
    if set(configs) != set(MODELS) or set(plans) != set(MODELS):
        raise ValueError("comparison needs exactly one explicit baseline and one neural protocol")
    reference = configs["baseline"]
    for protocol, config in configs.items():
        if (config["smoke"] or config["prepare_only"] or config["models"] != MODELS[protocol]
                or config["target_calibration"] != [] or not config["not_same_type_deployment"]
                or config["global_summary_writer"] is not False):
            raise ValueError("formal zero-shot research contract required; smoke/calibrated/deployment results stay separate")
        if any(config[field] != reference[field] for field in CONTRACT_FIELDS) or plans[protocol] != plans["baseline"]:
            raise ValueError("shared comparison input/roles/seeds/origins differ; do not pool these runs")
        if len(config["seeds"]) < 5 or len(set(config["seeds"])) != len(config["seeds"]):
            raise ValueError("formal shared comparison needs at least five unique training seeds")
        if config["origins"] != [20, 50, 100]:
            raise ValueError("shared reporting expects the frozen 20/50/100-source-cycle protocol")
    if len(plans["baseline"]) != 2 or len({p["name"] for p in plans["baseline"]}) != 2:
        raise ValueError("two unique transfer directions required")
    for plan in plans["baseline"]:
        if (plan["mode"] != RESEARCH_MODE or plan["target_calibration"] != [] or not plan["not_same_type_deployment"]
                or not plan["target_test"] or len(set(plan["target_test"])) != len(plan["target_test"])
                or set(plan["source_refit"]) & set(plan["target_test"])):
            raise ValueError("invalid whole-cell source/target research roles")


def describe(values):
    defined = [float(value) for value in values if value is not None]
    if not all(np.isfinite(value) for value in defined):
        raise ValueError("nonfinite errors cannot be silently dropped")
    return {"n_defined": len(defined), "mean": float(np.mean(defined)) if defined else None,
        "sd": float(np.std(defined, ddof=1)) if len(defined) > 1 else None}


def aggregate_cell_errors(scores, ids, seeds):
    if not ids or set(scores) != {(seed, key) for seed in seeds for key in ids}:
        raise ValueError("shared aggregation requires exact complete eligible target/seed coverage")
    for score in scores.values():
        for metric in METRICS:
            value = score[metric]
            if value is None and metric == "r2":
                continue
            if value is None or isinstance(value, bool) or not np.isfinite(value) or (metric != "r2" and value < 0):
                raise ValueError("invalid regression metric; do not drop a hard target")
    by_cell = [{"cell_id": key, **{metric: describe([scores[(seed, key)][metric] for seed in seeds]) for metric in METRICS}}
        for key in ids]
    metrics = {}
    for metric in METRICS:
        values = [describe([scores[(seed, key)][metric] for key in ids])["mean"] for seed in seeds]
        metrics[metric] = {"seed_macro_mean_and_sd": describe(values), "seed_macro_values": values,
            "between_cell_mean_and_sd_of_seed_averaged_errors": describe([row[metric]["mean"] for row in by_cell]),
            "undefined_cell_seed_pairs": sum(scores[(seed, key)][metric] is None for seed in seeds for key in ids)}
    return {"metrics": metrics, "by_cell_training_seed_variation": by_cell,
        "fraction_cells_seed_averaged_mape_le_6": float(np.mean([row["mape_percent"]["mean"] <= 6 for row in by_cell])),
        "worst_cell": max(by_cell, key=lambda row: row["mape_percent"]["mean"]),
        "statistics_scope": "cell macro errors; SD describes training seeds, not independent batteries or a confidence interval; source CV folds are not target replications"}


def eligibility(cells, plan, origin, horizon):
    ids, gaps = [], []
    for key in plan["target_test"]:
        cell = cells[key]
        if cell["physical_family"] != plan["target_physical_family"] or cell["metadata"]["dataset"] != plan["target_dataset"]:
            raise ValueError("target physical family/dataset differs from explicit research plan")
        future = cell["label_cycle"][np.isfinite(cell["soh"]) & (cell["soh"] > 0) & (cell["label_cycle"] > origin)]
        reason = "insufficient valid future targets" if len(future) < 2 else None
        if reason is None:
            try:
                transfer_prefix(cell, origin)
            except ValueError as error:
                if "insufficient" not in str(error) and "not established" not in str(error):
                    raise
                reason = str(error)
        if reason is None and horizon.startswith("within_") and not np.any(future <= origin+int(horizon.split("_")[1])):
            reason = "no comparable diagnostic target in this source-cycle horizon"
        if reason is None:
            ids.append(key)
        else:
            gaps.append({"cell_id": key, "reason": reason})
    return ids, gaps


def check_unit_partition(report, cells, plan, origins):
    expected = {(key, origin) for key in plan["target_test"] for origin in origins}
    pairs = [(row["cell_id"], row["origin"]) for row in report["by_cell"]+report["skipped"]]
    if len(pairs) != len(set(pairs)) or set(pairs) != expected:
        raise ValueError("duplicate/missing/unknown target/origin reports")
    assert_coverage(report, plan["target_test"], origins, cells)
    for row in report["by_cell"]:
        if row["physical_family"] != plan["target_physical_family"]:
            raise ValueError("metric row physical family changed")


def build_transfer_tables(configs, plans, cells, units):
    """Caller independently re-audits complete runs before supplying units.

    units[(protocol,direction,seed,model)] contains metrics and saved arrays.
    Never select runs/models/cells from their target error.
    """
    validate_comparison_contracts(configs, plans)
    origins, seeds = configs["baseline"]["origins"], configs["baseline"]["seeds"]
    grid = {(protocol, plan["name"], seed, model) for protocol, config in configs.items()
        for plan in plans[protocol] for seed in seeds for model in config["models"]}
    if set(units) != grid:
        raise ValueError("formal reporting requires the entire exact final-model grid")
    primary, paired, coverage = [], [], []
    for plan in plans["baseline"]:
        eligible = {}
        for origin in origins:
            for horizon in HORIZONS:
                ids, gaps = eligibility(cells, plan, origin, horizon)
                eligible[(origin, horizon)] = ids
                coverage.append({"direction": plan["name"], "origin_source_cycle": origin, "horizon": horizon,
                    "full_target_cells": len(plan["target_test"]), "eligible_target_cells": len(ids),
                    "coverage_fraction": len(ids)/len(plan["target_test"]), "eligible_ids": ids, "gaps": gaps})
        common = [key for key in plan["target_test"] if all(key in eligible[(origin, "all_observed_future")] for origin in origins)]
        for protocol, config in configs.items():
            for model in config["models"]:
                records = {}
                for seed in seeds:
                    unit = units[(protocol, plan["name"], seed, model)]
                    check_unit_partition(unit["metrics"], cells, plan, origins)
                    records[seed] = {(row["cell_id"], row["origin"]): row for row in unit["metrics"]["by_cell"]}
                for origin in origins:
                    base = {"direction": plan["name"], "source_dataset": plan["source_dataset"], "target_dataset": plan["target_dataset"],
                        "source_physical_family": plan["source_physical_family"], "target_physical_family": plan["target_physical_family"],
                        "evidence_level": plan["evidence_level"], "not_same_type_deployment": True, "target_calibration": [],
                        "protocol": protocol, "model": model, "origin_source_cycle": origin, "training_seeds": seeds,
                        "full_target_cells": len(plan["target_test"])}
                    for horizon in HORIZONS:
                        ids = eligible[(origin, horizon)]
                        entry = {**base, "horizon": horizon, "evaluation_cells": len(ids), "coverage_fraction": len(ids)/len(plan["target_test"])}
                        scores = {}
                        for seed in seeds:
                            for key in ids:
                                row = records[seed][(key, origin)]
                                metric = row["metrics"] if horizon == "all_observed_future" else row["late_half"] if horizon == "late_half" else row["within_horizon"][horizon.split("_")[1]]
                                if metric is None:
                                    raise ValueError("eligible target/horizon metric missing; cannot narrow denominator")
                                scores[(seed, key)] = metric
                        entry.update(aggregate_cell_errors(scores, ids, seeds) if ids else {"unavailable_reason": "no eligible targets; not a zero-error result"})
                        primary.append(entry)
                    matched = {**base, "horizon": f"matched_future_after_source{max(origins)}", "evaluation_cells": len(common),
                        "coverage_fraction": len(common)/len(plan["target_test"])}
                    scores = {}
                    for seed in seeds:
                        arrays = units[(protocol, plan["name"], seed, model)]["arrays"]
                        for key in common:
                            prefix, reference = f"{key}__{origin}__", f"{key}__{max(origins)}__"
                            future = arrays[prefix+"cycle"]
                            take = future > max(origins)
                            if (not np.array_equal(future[take], arrays[reference+"cycle"])
                                    or not np.array_equal(arrays[prefix+"truth"][take], arrays[reference+"truth"])):
                                raise ValueError("matched origins have unequal future target grids/truth")
                            truth, prediction = arrays[prefix+"truth"][take], arrays[prefix+"prediction"][take]
                            if len(truth) < 2 or prediction.shape != truth.shape or not np.isfinite(prediction).all():
                                raise ValueError("invalid matched saved predictions")
                            scores[(seed, key)] = regression_metrics(truth, prediction)
                    matched.update(aggregate_cell_errors(scores, common, seeds) if common else {"unavailable_reason": "no common eligible targets"})
                    paired.append(matched)
    return {"primary": primary, "matched_prefix_and_future": paired, "coverage": coverage,
        "interpretation": "public-development cross-physical-family zero-shot transfer, not same-type deployment or blind validation; seeds and source selection folds do not increase target battery n; R2 missing values retained; normal high errors retained"}
