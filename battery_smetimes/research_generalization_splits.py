"""Metadata-only leave-condition-out plans; never relax physical-family gates.

These are additional public development experiments, not new blind tests or
source CV folds. Cross-dataset prediction needs its own shared-signal contract;
it cannot be represented by assigning target cells the source physical family.
"""

from collections import Counter, defaultdict
import math

import numpy as np

from battery_smetimes.forecast import charge_rates
from battery_smetimes.physical_types import require_same_registered_family
from battery_smetimes.research_splits import condition_stratum


METADATA_FIELDS = (
    "dataset", "group", "charge_protocol", "charge_rate_steps_c",
    "charge_rate_c", "diagnostic_discharge_rate_c", "discharge_cutoff_v",
)


def metadata_only(records):
    """Discard reference Q, targets, lifetime and signals before any splitting."""
    return {key: {"metadata": {field: row["metadata"].get(field)
                              for field in METADATA_FIELDS},
                  "physical_family": row["physical_family"]}
            for key, row in sorted(records.items())}


def heldout_condition(metadata):
    domain = metadata["dataset"]
    if domain == "MIT":
        rates = np.asarray(charge_rates(metadata), dtype=float)
        if len(rates) != 2 or not np.isfinite(rates).all():
            raise ValueError("MIT condition requires two known charging rates")
        return "constant" if rates[0] == rates[1] else (
            "decreasing" if rates[0] > rates[1] else "increasing")
    if domain == "XJTU":
        protocol = metadata.get("charge_protocol")
        if not isinstance(protocol, str) or not protocol.strip():
            raise ValueError("XJTU condition requires a known protocol")
        return protocol
    raise ValueError("no prespecified condition holdout for this dataset")


def validate_condition_splits(plans, records, registry):
    records = metadata_only(records)
    counts, heldouts = defaultdict(Counter), defaultdict(set)
    for plan in plans:
        domain, condition = plan["dataset"], plan["heldout_condition"]
        if condition in heldouts[domain]:
            raise ValueError("duplicate held-out condition")
        heldouts[domain].add(condition)
        parts = [set(plan[key]) for key in ("train", "validation", "test")]
        if any(len(plan[key]) != len(part) for key, part in zip(("train", "validation", "test"), parts)):
            raise ValueError("duplicate cell within condition partition")
        if any(not part for part in parts) or any(parts[i] & parts[j]
                for i in range(3) for j in range(i + 1, 3)):
            raise ValueError("condition partitions must be nonempty and disjoint")
        cohort = {key for key, row in records.items() if row["metadata"]["dataset"] == domain}
        if set.union(*parts) != cohort:
            raise ValueError("condition split must account for the entire source cohort")
        expected_test = {key for key in cohort
                         if heldout_condition(records[key]["metadata"]) == condition}
        if parts[2] != expected_test:
            raise ValueError("entire held-out condition must be test-only")
        family = records[plan["train"][0]]["physical_family"]
        require_same_registered_family(family,
            [records[key]["physical_family"] for key in cohort], registry)
        if plan["physical_family"] != family:
            raise ValueError("declared condition family does not match records")
        counts[domain].update(plan["test"])
    expected_domains = {row["metadata"]["dataset"] for row in records.values()
                        if row["metadata"]["dataset"] in ("MIT", "XJTU")}
    if set(counts) != expected_domains:
        raise ValueError("condition plans omit an entire supported dataset")
    for domain, cells in counts.items():
        expected = {key for key, row in records.items() if row["metadata"]["dataset"] == domain}
        if set(cells) != expected or set(cells.values()) != {1}:
            raise ValueError("condition plans must test each source cell exactly once")


def make_condition_splits(records, config, registry):
    records = metadata_only(records)
    fraction = config["validation_fraction_of_development"]
    if not 0 < fraction < 1:
        raise ValueError("validation fraction must be strictly between zero and one")
    plans = []
    for domain in ("MIT", "XJTU"):
        keys = sorted(key for key, row in records.items() if row["metadata"]["dataset"] == domain)
        if not keys:
            continue
        conditions = defaultdict(list)
        for key in keys:
            conditions[heldout_condition(records[key]["metadata"])].append(key)
        if len(conditions) < 2:
            raise ValueError("condition generalization requires at least two known conditions")
        for index, (condition, test) in enumerate(sorted(conditions.items())):
            train, validation, strata = [], [], defaultdict(list)
            for key in sorted(set(keys) - set(test)):
                strata[condition_stratum(records[key]["metadata"])].append(key)
            rng = np.random.default_rng(config["split_seed"] + 10000 + index)
            for _, group in sorted(strata.items()):
                shuffled = rng.permutation(group).tolist()
                n_val = min(len(shuffled) - 1, math.ceil(fraction * len(shuffled)))
                validation.extend(shuffled[:n_val])
                train.extend(shuffled[n_val:])
            plans.append({"name": f"{domain}_condition{index + 1}", "dataset": domain,
                "heldout_condition": condition, "physical_family": records[keys[0]]["physical_family"],
                "train": sorted(train), "validation": sorted(validation), "test": sorted(test),
                "evidence_level": "public development leave-condition-out supplement",
                "condition_scope": "charge-rate step ordering, not an exact unseen numeric rate" if domain == "MIT"
                                   else "author-described charge/discharge usage protocol",
                "selection_contract": "source-condition validation cells only; no held-out condition labels for selection"})
    validate_condition_splits(plans, records, registry)
    return plans
