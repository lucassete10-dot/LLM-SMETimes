"""Verify saved unseen-cell predictions with frozen source and no future inputs."""

import argparse
import copy
from datetime import datetime, timezone
import json
import importlib.util
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def resolved_device(requested, config):
    return config.get("device", "cpu") if requested == "auto" else requested


def replay_comparison(replay, original_same_device, saved, neural, causal_replay=None):
    import numpy as np
    tolerance = 1e-5 if neural else 1e-10
    causal_replay = replay if causal_replay is None else causal_replay
    invariant = bool(np.array_equal(causal_replay, original_same_device))
    saved_match = bool(np.allclose(replay, saved, rtol=tolerance, atol=tolerance))
    return {"max_absolute_soh_difference": float(np.max(np.abs(replay - saved))),
        "tolerance": tolerance, "saved_prediction_match": saved_match,
        "same_device_no_future_invariance": invariant,
        "same_device_invariance_max_absolute_difference": float(np.max(np.abs(causal_replay - original_same_device))),
        "passed": saved_match and invariant}


def causal_pair(predictor, bundle, visible, original, origin, future, replay):
    estimator = bundle.get("model")
    if estimator is not None and hasattr(estimator, "estimators_") and hasattr(estimator, "n_jobs"):
        jobs = estimator.n_jobs
        try:
            estimator.n_jobs = 1
            changed = predictor(bundle, visible, origin, future)
            intact = predictor(bundle, original, origin, future)
        finally:
            estimator.n_jobs = jobs
        return changed, intact, {"serial_tree_accumulation_control": True, "original_n_jobs": jobs,
            "control_n_jobs": 1, "original_runtime_restored": True,
            "scope": "Only the paired causal check controls summation order; saved-prediction matching retains the artifact's original runtime."}
    return replay, predictor(bundle, original, origin, future), {"serial_tree_accumulation_control": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_folder", type=Path)
    parser.add_argument("--device", default="auto", help="auto reuses the original experiment device; explicit CPU/GPU portability is separate")
    parser.add_argument("--max-cells", type=int, default=2)
    parser.add_argument("--origin", type=int, action="append")
    parser.add_argument("--condition", action="store_true", help="Explicit condition-holdout replay; do not reinterpret the main audit's CV splits")
    args = parser.parse_args()
    run = args.model_folder.resolve().parents[2]
    recorded_device = json.loads((run / "config.json").read_text()).get("device", "cpu")
    device = resolved_device(args.device, {"device": recorded_device})
    # Import the original modules under their original package names. This is
    # necessary for joblib class resolution and random-backbone exact replay.
    import battery_smetimes
    import scripts
    battery_smetimes.__path__ = [str(run / "source_snapshot/battery_smetimes"), *battery_smetimes.__path__]
    scripts.__path__ = [str(run / "source_snapshot/scripts"), *scripts.__path__]
    from scripts.prepare_same_type_forecast import save_json, sha
    if args.condition:
        configuration = json.loads((run / "config.json").read_text(encoding="utf-8"))
        for name, digest in configuration["execution_source_sha256"].items():
            path = run / "execution_snapshot" / name
            if (not path.resolve().is_relative_to((run / "execution_snapshot").resolve()) or sha(path) != digest):
                raise ValueError("condition replay execution snapshot changed")
        # Model modules remain FIRST from the original main snapshot. Only
        # the condition-aware loader/contract use the separate execution code.
        battery_smetimes.__path__.insert(1, str(run / "execution_snapshot/battery_smetimes"))
        scripts.__path__.insert(1, str(run / "execution_snapshot/scripts"))
        loader_path = run / "execution_snapshot/battery_smetimes/research_inference.py"
        spec = importlib.util.spec_from_file_location("condition_replay_loader", loader_path)
        loader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loader)
        load_model = loader.load_model
        inference_path = loader_path
    else:
        from battery_smetimes.research_inference import load_model
        inference_path = Path(sys.modules["battery_smetimes.research_inference"].__file__)
    from scripts.train_same_type_forecast import load_data
    import numpy as np
    import torch
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    destination = Path("experiments/soh_research_v5/condition_replays" if args.condition else "experiments/soh_research_v5/replays") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, destination / "replay_source.py")
    shutil.copy2(inference_path, destination / "inference_source.py")
    save_json(destination / "config.json", {"model_folder": str(args.model_folder.resolve()),
        "device": device, "requested_device": args.device, "original_training_device": recorded_device,
        "same_training_device": device == recorded_device, "max_cells": args.max_cells, "origins": args.origin,
        "python": sys.version, "torch": torch.__version__, "script_sha256": sha(Path(__file__)),
        "evaluation_protocol": "condition" if args.condition else "main",
        "runtime": {"float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cpu_mkldnn_enabled": torch.backends.mkldnn.enabled,
            "transformer_fastpath_enabled": torch.backends.mha.get_fastpath_enabled()},
        "module_resolution": "frozen source_snapshot precedes current package modules"})
    rows = []
    try:
        if args.condition:
            bundle, predictor, evidence = load_model(args.model_folder, device, evaluation_protocol="condition")
        else:
            bundle, predictor, evidence = load_model(args.model_folder, device)
        import inspect
        model_modules = {"predictor": predictor}
        if "network" in bundle:
            model_modules.update(network=type(bundle["network"]), processor=type(bundle["processor"]))
        model_sources = {key: str(Path(inspect.getfile(value)).resolve()) for key, value in model_modules.items()}
        if any(not Path(path).is_relative_to((run / "source_snapshot").resolve()) for path in model_sources.values()):
            raise ValueError("prediction implementation resolved outside frozen model source")
        evidence["model_source_paths"] = model_sources
        audit_dir = run / "frozen_audit"
        frozen = json.loads((audit_dir / "config.json").read_text())
        audit = json.loads((audit_dir / "audit.json").read_text())
        if sha(frozen["config"]["integrity_audit"]) != frozen["integrity_sha256"]:
            raise ValueError("integrity rules changed")
        prepared = Path(audit["prepared_folder"])
        if sha(prepared / "manifest.json") != frozen["prepared_manifest_sha256"]:
            raise ValueError("prepared manifest changed")
        manifest, cells = load_data(prepared, frozen["config"]["integrity_audit"])
        run_config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        if run_config["prepared_files_sha256"] != {r["cell_id"]: r["sha256"] for r in manifest["cells"]}:
            raise ValueError("replay prepared cell fingerprints differ from training")
        registered = json.loads((audit_dir / "cell_registry.json").read_text())["cells"]
        saved = np.load(args.model_folder / "test/predictions.npz")
        ids = evidence["plan"]["test"][:args.max_cells]
        if args.max_cells < 1:
            raise ValueError("max_cells must be positive")
        origins = args.origin or bundle["origins"]
        if set(origins) - set(bundle["origins"]):
            raise ValueError("origin outside the registered experiment")
        for key in ids:
            original = cells[key]
            original.update({name: registered[key][name] for name in ("physical_family", "reference_established_source_cycle")})
            for origin in origins:
                prefix = f"{key}__{origin}__"
                if prefix + "prediction" not in saved:
                    rows.append({"cell_id": key, "origin": origin, "status": "not originally eligible; retained as coverage gap"})
                    continue
                visible = copy.deepcopy(original)
                for field in ("soh", "unfiltered_soh", "label_cycle"):
                    visible.pop(field, None)
                visible["raw"][visible["cycle"] > origin] = np.inf
                future = saved[prefix + "cycle"]
                started = time.perf_counter()
                replay = predictor(bundle, visible, origin, future)
                causal_replay, original_same_device, causal_control = causal_pair(predictor, bundle, visible, original, origin, future, replay)
                # Neural deterministic FP32 heads tolerate float roundoff, not
                # percent-scale SOH changes. Tabular replay uses tighter bounds.
                comparison = replay_comparison(replay, original_same_device, saved[prefix + "prediction"], "network" in bundle, causal_replay)
                rows.append({"cell_id": key, "origin": origin, "future_points": len(future),
                    **comparison, "causal_control": causal_control,
                    "seconds": time.perf_counter() - started, "labels_removed": True,
                    "all_future_raw_signals_replaced_by_infinity": True})
                save_json(destination / "progress.json", {"evidence": evidence, "rows": rows})
                if not comparison["passed"]:
                    raise ValueError("saved future-SOH prediction could not be reproduced")
        if not any(row.get("passed") for row in rows):
            raise ValueError("replay selected no originally evaluated observations")
        report = {"status": "passed saved-checkpoint no-future-input replay", "evidence": evidence,
            "rows": rows, "scope": "checked cells/origins only; no training or accuracy tuning; not a deployment benchmark"}
        save_json(destination / "report.json", report)
        print(json.dumps({"folder": str(destination.resolve()), "checks": sum("passed" in r for r in rows),
                          "status": report["status"]}, indent=2))
    except BaseException:
        save_json(destination / "failure.json", {"traceback": traceback.format_exc(), "rows": rows})
        raise


if __name__ == "__main__":
    main()
