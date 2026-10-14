"""Reproducible whole-cell splits; no cycle/window may cross a split boundary."""

from collections import Counter

import numpy as np
from sklearn.model_selection import train_test_split


def cellwise_split(records, dataset_id, *, seed=2026, stratify_group=True):
    """Return 70/15/15 train/validation/test cell IDs for one dataset.

    Group stratification keeps each known protocol represented when possible.
    Cross-condition evaluation is a separate held-out-*group* experiment, not
    an inflated interpretation of this within-condition unseen-cell split.
    """
    selected = [row for row in records if row["dataset"] == dataset_id]
    ids = [row["cell_id"] for row in selected]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate cell IDs")
    if len(ids) < 7:
        raise ValueError("at least seven cells required for 70/15/15 split")
    groups = [row["group"] for row in selected]
    stratify = groups if (stratify_group and min(Counter(groups).values()) >= 3) else None
    temporary_count = max(2, round(.30 * len(ids)))
    train_ids, temporary_ids, _, temporary_groups = train_test_split(
        ids, groups, test_size=temporary_count, random_state=seed,
        stratify=stratify)
    temporary_strata = (temporary_groups if stratify is not None and
                        min(Counter(temporary_groups).values()) >= 2 else None)
    validation_ids, test_ids = train_test_split(
        temporary_ids, test_size=len(temporary_ids) // 2,
        random_state=seed + 1, stratify=temporary_strata)
    result = {"train": sorted(train_ids), "validation": sorted(validation_ids),
              "test": sorted(test_ids)}
    if set(result["train"]) & set(result["validation"]) or \
       set(result["train"]) & set(result["test"]) or \
       set(result["validation"]) & set(result["test"]):
        raise AssertionError("cell leakage across splits")
    if set().union(*map(set, result.values())) != set(ids):
        raise AssertionError("split omitted cells")
    return result
