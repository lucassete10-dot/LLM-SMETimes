"""Frozen cell-level folds, stratified only by pre-existing experimental metadata."""

from collections import Counter, defaultdict
import math

import numpy as np

from battery_smetimes.forecast import charge_rates
from battery_smetimes.physical_types import require_same_registered_family


def condition_stratum(metadata):
    domain = metadata["dataset"]
    if domain == "MIT":
        rates = charge_rates(metadata)
        direction = "constant" if rates[0] == rates[-1] else (
            "decreasing" if rates[0] > rates[-1] else "increasing")
        # Batch is a balancing variable, NOT physical type or a life label.
        return f'{metadata["group"]}|{direction}'
    if domain == "XJTU":
        return metadata["charge_protocol"]
    if domain == "NASA":
        return f'cutoff={metadata.get("discharge_cutoff_v")}'
    raise ValueError("unsupported research source")


def validate_splits(plans, records, registry):
    counts = defaultdict(Counter)
    for plan in plans:
        parts = [set(plan[key]) for key in ("train", "validation", "test")]
        if any(not part for part in parts):
            raise ValueError("empty train/validation/test partition")
        if any(parts[i] & parts[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError("same cell occurs in multiple partitions")
        expected = {key for key, row in records.items() if row["metadata"]["dataset"] == plan["dataset"]}
        if set.union(*parts) != expected:
            raise ValueError("fold does not account for every cell exactly once")
        for key in plan["test"] + plan["validation"]:
            require_same_registered_family(records[key]["physical_family"],
                [records[other]["physical_family"] for other in plan["train"]], registry)
        counts[plan["dataset"]].update(plan["test"])
    for domain, counts_for_domain in counts.items():
        expected = {key for key, row in records.items() if row["metadata"]["dataset"] == domain}
        if set(counts_for_domain) != expected or set(counts_for_domain.values()) != {1}:
            raise ValueError("outer cross-validation must test every cell exactly once")


def make_research_splits(records, config, registry):
    """Input is metadata-only, so no target/final-life stratification is possible."""
    plans = []
    for domain in ("MIT", "XJTU", "NASA"):
        keys = sorted(key for key, row in records.items() if row["metadata"]["dataset"] == domain)
        if not keys:
            continue
        if domain == "NASA":
            for j, key in enumerate(keys):
                val = keys[(j + 1) % len(keys)]
                plans.append({"name": f"NASA_fold{j + 1}", "dataset": domain,
                    "evidence_level": "four-cell_unverified_cohort_supplement",
                    "train": [k for k in keys if k not in (key, val)],
                    "validation": [val], "test": [key]})
            continue
        groups = defaultdict(list)
        for key in keys:
            groups[condition_stratum(records[key]["metadata"])].append(key)
        rng = np.random.default_rng(config["split_seed"])
        buckets = [[] for _ in range(config["outer_folds"])]
        for _, group in sorted(groups.items()):
            shuffled = rng.permutation(group).tolist()
            # Rotate so small strata do not always fall in fold 1.
            offset = min(range(len(buckets)), key=lambda j: len(buckets[j]))
            for j, key in enumerate(shuffled):
                buckets[(j + offset) % len(buckets)].append(key)
        for j, test in enumerate(buckets):
            train, val = [], []
            inner_rng = np.random.default_rng(config["split_seed"] + 1000 + j)
            for _, group in sorted(groups.items()):
                remaining = inner_rng.permutation(sorted(set(group) - set(test))).tolist()
                n_val = min(len(remaining) - 1,
                    math.ceil(config["validation_fraction_of_development"] * len(remaining)))
                n_val = max(0, n_val)
                val.extend(remaining[:n_val])
                train.extend(remaining[n_val:])
            plans.append({"name": f"{domain}_fold{j + 1}", "dataset": domain,
                "evidence_level": "within_physical_family_development_CV",
                "train": sorted(train), "validation": sorted(val), "test": sorted(test)})
    validate_splits(plans, records, registry)
    return plans
