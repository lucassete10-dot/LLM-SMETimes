"""Explicit research-only common discharge signals, not physical-type routing.

MIT and NASA have different native prefix windows and grids. Use only cached
samples at 100/200/300 s (linear regridding of MIT's 30 s samples); do not take
their incompatible whole-window statistics or make up missing NASA ratings.
"""

import numpy as np

from battery_smetimes.physical_types import require_same_registered_family


RESEARCH_MODE = "cross_dataset_research_only"
TIMES = np.array([100., 200., 300.])
NATIVE_WIDTH = {"MIT": 51, "NASA": 44}
NATIVE_GRID = {"MIT": np.arange(0., 301., 30.), "NASA": np.arange(0., 1001., 100.)}
FEATURE_NAMES = ([f"voltage_change_since_100s_v_t{int(t)}s" for t in TIMES]
    + [f"absolute_current_over_rated_ah_c_t{int(t)}s" for t in TIMES]
    + [f"temperature_change_since_100s_c_t{int(t)}s" for t in TIMES]
    + ["voltage_at_100s_v", "temperature_at_100s_c"])


def _research_mode(mode):
    if mode != RESEARCH_MODE:
        raise ValueError("shared cross-dataset signals require explicit research-only mode; not a deployment route")


def shared_discharge_history(cell, origin, registry, *, mode):
    _research_mode(mode)
    domain = cell["metadata"]["dataset"]
    if domain not in NATIVE_WIDTH:
        raise ValueError("no shared 100-300 s discharge V/I/T contract for this dataset")
    family = cell["physical_family"]
    require_same_registered_family(family, [family], registry)
    row = registry["families"][family]
    if domain not in row["datasets"]:
        raise ValueError("source dataset contradicts its unchanged registered family")
    capacity = row["specifications"]["nominal_capacity_ah"]
    if isinstance(capacity, bool) or capacity is None or not np.isfinite(capacity) or capacity <= 0:
        raise ValueError("current normalization needs documented positive RATED capacity, not measured Q")
    if isinstance(origin, bool) or not np.isfinite(origin) or origin < 1 or int(origin) != origin:
        raise ValueError("origin must be a positive source-cycle integer")
    cycles, native = np.asarray(cell["cycle"]), np.asarray(cell["raw"], dtype=float)
    if cycles.ndim != 1 or native.shape != (len(cycles), NATIVE_WIDTH[domain]):
        raise ValueError("native cached signal schema mismatch")
    take = cycles <= origin
    c, x = cycles[take], native[take]
    if len(c) < 3:
        raise ValueError("insufficient visible discharge history")
    if not np.isfinite(c).all() or np.any(np.diff(c) <= 0):
        raise ValueError("visible source cycles must be finite and increasing")
    # Validate only the used native grid support. Later observations and unused
    # 300/1000 s summary columns must never condition the shared representation.
    grid = NATIVE_GRID[domain]
    support = (grid >= (90 if domain == "MIT" else 100)) & (grid <= 300)
    channels = []
    for channel in range(3):
        values = x[:, channel * len(grid):(channel + 1) * len(grid)][:, support]
        if np.isinf(values).any() or (channel < 2 and not np.isfinite(values).all()):
            raise ValueError("invalid shared visible voltage/current/temperature support")
        # Missing temperature remains missing, not invented at room temperature.
        sampled = np.stack([np.interp(TIMES, grid[support], value)
                            if np.isfinite(value).all() else np.full(3, np.nan) for value in values])
        channels.append(sampled)
    voltage, current, temperature = channels
    common = np.column_stack((voltage - voltage[:, :1], np.abs(current) / capacity,
                              temperature - temperature[:, :1], voltage[:, 0], temperature[:, 0]))
    return c.copy(), common.astype(np.float32)


def make_transfer_plans(records, main_plans, *, mode,
                        directions=(("MIT", "NASA"), ("NASA", "MIT"))):
    """Source-only CV selection, full-source refit, disjoint target evaluation.

    No target calibration in these plans. Target-cell labels must not enter
    source feature fitting, imputation, early stopping, or model selection.
    """
    _research_mode(mode)
    if not directions or len(directions) != len(set(directions)):
        raise ValueError("transfer directions must be nonempty and unique")
    result = []
    for source, target in directions:
        if source == target or source not in ("MIT", "XJTU", "NASA") or target not in ("MIT", "XJTU", "NASA"):
            raise ValueError("unsupported cross-dataset direction")
        source_ids = sorted(k for k, r in records.items() if r["metadata"]["dataset"] == source)
        target_ids = sorted(k for k, r in records.items() if r["metadata"]["dataset"] == target)
        if not source_ids or not target_ids or set(source_ids) & set(target_ids):
            raise ValueError("transfer requires nonempty disjoint source and target cell cohorts")
        source_families = {records[k]["physical_family"] for k in source_ids}
        target_families = {records[k]["physical_family"] for k in target_ids}
        if len(source_families) != 1 or len(target_families) != 1 or source_families == target_families:
            raise ValueError("these research stress tests require two distinct isolated physical families")
        selection, tested = [], []
        for plan in main_plans:
            if plan["dataset"] != source:
                continue
            validation = list(plan["test"])
            train = sorted(plan["train"] + plan["validation"])
            if len(train) != len(set(train)) or len(validation) != len(set(validation)):
                raise ValueError("duplicate source selection IDs")
            if not train or not validation or set(train) & set(validation) or set(train + validation) != set(source_ids):
                raise ValueError("source selection split is incomplete or overlaps")
            tested.extend(validation)
            selection.append({"name": plan["name"], "source_train": train, "source_validation": validation})
        if len(tested) != len(set(tested)) or set(tested) != set(source_ids):
            raise ValueError("source selection folds must validate each source cell exactly once")
        result.append({"name": f"{source}_to_{target}", "mode": mode,
            "source_dataset": source, "target_dataset": target,
            "source_physical_family": next(iter(source_families)),
            "target_physical_family": next(iter(target_families)),
            "source_selection_folds": selection, "source_refit": source_ids, "target_test": target_ids,
            "target_calibration": [], "selection_contract": "aggregate source-cell validation errors only; target errors cannot choose parameters",
            "evidence_level": "public-development cross-physical-family research stress test; NASA chemistry/model unverified"
                if "NASA" in (source, target) else "public-development cross-physical-family transfer; XJTU model identifier missing",
            "not_same_type_deployment": True})
    return result
