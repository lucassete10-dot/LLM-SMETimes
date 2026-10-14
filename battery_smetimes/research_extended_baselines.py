"""Strong baselines and negative controls, with the exact existing input budget."""

import time
import warnings

import numpy as np
from scipy.optimize import minimize
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, DotProduct, Matern, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import StandardScaler

from battery_smetimes.physical_types import require_same_registered_family


def marginal_likelihood_optimizer(objective, initial_theta, bounds):
    """Kernel optimization uses training likelihood, never validation/test rows."""
    result = minimize(objective, initial_theta, jac=True, bounds=bounds,
                      method="L-BFGS-B", options={"maxiter": 30})
    if not result.success:
        warnings.warn(f"GP training optimizer: {result.message}", ConvergenceWarning)
    return result.x, result.fun


def balanced_coreset(cells, ids, origins, used_ids, cap):
    from battery_smetimes.research_baselines import build_training_pairs
    budget = max(1, int(cap) // len(used_ids))
    indices, detail, offset = [], [], 0
    for key in ids:
        try:
            x, _, _, _ = build_training_pairs(cells, [key], origins)
        except ValueError as error:
            if "no valid training" in str(error):
                continue
            raise
        chosen = np.unique(np.linspace(0, len(x) - 1, min(budget, len(x))).round().astype(int))
        indices.extend((chosen + offset).tolist())
        detail.append({"cell_id": key, "available_training_pairs": len(x), "selected_pairs": len(chosen)})
        offset += len(x)
    if offset == 0 or not indices:
        raise ValueError("empty GP coreset")
    return np.asarray(indices, dtype=int), detail


def fit_extended(cells, ids, family, registry, config, name, parameters, seed, smoke=False):
    from battery_smetimes.research_baselines import build_training_pairs
    require_same_registered_family(family, [cells[key]["physical_family"] for key in ids], registry)
    started = time.perf_counter()
    x, y, weight, used_ids = build_training_pairs(cells, ids, config["origins_source_cycles"])
    bundle = {"schema": "soh_research_v5_baseline", "model_name": name, "parameters": dict(parameters),
        "physical_family": family, "registry": registry, "seed": seed, "train_ids": list(ids),
        "used_train_ids": used_ids, "training_pairs": len(y), "origins": config["origins_source_cycles"],
        "smoke": smoke, "scaler": None, "imputer": None,
        "input_contract": "same audited early V/I/T and known conditions; no measured Q/SOH/EOL or batch/type one-hot"}
    if name == "physical_family_mean":
        templates = []
        for key in ids:
            cell = cells[key]
            valid = np.isfinite(cell["soh"]) & (cell["soh"] > 0)
            if valid.any():
                templates.append({"cell_id": key, "cycle": cell["label_cycle"][valid].copy(),
                                  "soh": cell["soh"][valid].copy()})
        if not templates:
            raise ValueError("no training-only family curves")
        bundle.update({"templates": templates, "model": None, "fit_seconds": time.perf_counter() - started})
        return bundle
    if name == "cycle_conditions_only":
        # Four visible-history cycle descriptors + five known condition values
        # + three horizon descriptors. No voltage/current/temperature values.
        columns = np.arange(x.shape[1] - 12, x.shape[1])
        bundle["feature_columns"] = columns
        bundle["input_contract"] = "negative control: cycle/history coverage and known conditions only; no V/I/T features"
        x = x[:, columns]
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    x = imputer.fit_transform(x.astype(np.float64))
    bundle["imputer"] = imputer
    if name in ("cycle_conditions_only", "elastic_net", "gpr"):
        offset = x[0].copy()
        scaler = StandardScaler().fit(x - offset, sample_weight=weight)
        scaler.mean_ += offset
        x = scaler.transform(x)
        bundle["scaler"] = scaler
    selected = None
    if name == "cycle_conditions_only":
        estimator = Ridge(alpha=parameters["alpha"], solver="svd")
    elif name == "elastic_net":
        estimator = ElasticNet(alpha=parameters["alpha"], l1_ratio=parameters["l1_ratio"],
            max_iter=20000, tol=1e-5, selection="cyclic")
        x = np.asfortranarray(x)
    elif name == "gpr":
        cap = min(parameters["max_training_pairs"], 256) if smoke else parameters["max_training_pairs"]
        selected, detail = balanced_coreset(cells, ids, config["origins_source_cycles"], used_ids, cap)
        bundle.update({"coreset_indices": selected, "coreset_by_cell": detail,
            "estimator_training_pairs": len(selected), "coreset_max_pairs": cap})
        dimension = x.shape[1]
        kernel = (ConstantKernel(1., (.01, 100.)) *
                  (DotProduct(sigma_0=0., sigma_0_bounds="fixed") +
                   Matern(length_scale=np.sqrt(dimension) * parameters["initial_length_factor"],
                          length_scale_bounds=(.2, 200.), nu=1.5)) +
                  WhiteKernel(parameters["initial_noise"] ** 2, noise_level_bounds=(1e-6, .1)))
        # Inverse weights express relative observation noise; every source
        # battery contributes, and reduced pair coverage is explicit.
        alpha = 1e-4 / np.maximum(weight[selected], 1e-6)
        estimator = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=True,
            optimizer=marginal_likelihood_optimizer, n_restarts_optimizer=0, random_state=seed)
    elif name == "lightgbm":
        from lightgbm import LGBMRegressor
        estimator = LGBMRegressor(n_estimators=48 if smoke else 160,
            num_leaves=parameters["num_leaves"], learning_rate=parameters["learning_rate"],
            reg_lambda=parameters["reg_lambda"], min_child_samples=20,
            subsample=1., colsample_bytree=1., deterministic=True, force_col_wise=True,
            random_state=seed, n_jobs=2, verbosity=-1)
    else:
        raise ValueError(f"unknown extended baseline: {name}")
    with warnings.catch_warnings(record=True) as messages:
        warnings.simplefilter("always", ConvergenceWarning)
        if selected is None:
            estimator.fit(x, y, sample_weight=weight)
        else:
            estimator.fit(x[selected], y[selected])
    bundle.update({"model": estimator, "fit_seconds": time.perf_counter() - started,
        "fit_warnings": [{"category": row.category.__name__, "message": str(row.message)} for row in messages]})
    if name == "gpr":
        bundle["fitted_kernel"] = str(estimator.kernel_)
    if name == "elastic_net":
        bundle["iterations"] = int(estimator.n_iter_)
        bundle["dual_gap"] = float(estimator.dual_gap_)
    return bundle
