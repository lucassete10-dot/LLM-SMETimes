"""Label-free saved-model latency/resource measurements, with an idle guard.

Single-point queries are FUTURE SOH, not the abandoned current-SOH task.
Warm text-card caching is reported separately from a new input's first call.
"""

import argparse
import copy
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
for _thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_thread_variable, "2")

import numpy as np


def latency_summary(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("latency samples must be nonempty, finite and nonnegative")
    return {"technical_repeats": len(values), "mean_ms": float(values.mean()),
        "sd_ms": float(values.std(ddof=1)) if len(values) > 1 else None,
        "median_ms": float(np.median(values)), "p95_ms": float(np.quantile(values, .95)),
        "min_ms": float(values.min()), "max_ms": float(values.max())}


def visible_input(cell, origin):
    take = np.asarray(cell["cycle"]) <= origin
    return {"metadata": copy.deepcopy(cell["metadata"]), "physical_family": cell["physical_family"],
        "reference_established_source_cycle": cell["reference_established_source_cycle"],
        "cycle": np.asarray(cell["cycle"])[take].copy(), "raw": np.asarray(cell["raw"])[take].copy()}


def model_description(bundle):
    if "network" in bundle:
        network = bundle["network"]
        dtypes = {}
        for parameter in network.parameters():
            key = str(parameter.dtype)
            dtypes[key] = dtypes.get(key, 0) + parameter.numel()
        return {"kind": "neural", "total_parameters": sum(p.numel() for p in network.parameters()),
            "trainable_parameters": sum(p.numel() for p in network.parameters() if p.requires_grad),
            "frozen_parameters": sum(p.numel() for p in network.parameters() if not p.requires_grad),
            "parameter_storage_bytes": sum(p.numel() * p.element_size() for p in network.parameters()),
            "parameters_by_dtype": dtypes, "statistics_card_cache_applicable": hasattr(network, "backbone") and network.use_statistics}
    estimator = bundle["model"]
    result = {"kind": "tabular", "estimator_class": type(estimator).__name__,
        "total_parameters": None, "trainable_parameters": None, "frozen_parameters": None,
        "statistics_card_cache_applicable": False,
        "caveat": "Tree nodes/kernel support are not neural parameter counts; no fictitious equivalent count."}
    result["estimator_inference_n_jobs"] = getattr(estimator, "n_jobs", None)
    if hasattr(estimator, "coef_"):
        result["estimator_coefficient_count_not_all_pipeline_state"] = int(np.size(estimator.coef_) + np.size(estimator.intercept_))
    if hasattr(estimator, "estimators_"):
        trees = [value for value in np.asarray(estimator.estimators_, dtype=object).reshape(-1) if hasattr(value, "tree_")]
        result["tree_count"] = len(trees)
        result["tree_nodes_not_neural_parameters"] = sum(tree.tree_.node_count for tree in trees)
    return result


def other_project_workers():
    import psutil
    own = psutil.Process()
    exclude = {own.pid, *(parent.pid for parent in own.parents())}
    records = []
    for process in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
        try:
            info = process.info
            command = " ".join(info["cmdline"] or [])
            if info["pid"] not in exclude and "python" in (info["name"] or "").lower() and "soh_research_v5" in command:
                records.append({"pid": info["pid"], "create_time": info["create_time"], "command": command})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return records


def gpu_compute_workers():
    import psutil
    own = psutil.Process()
    exclude = {own.pid, *(parent.pid for parent in own.parents())}
    try:
        result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=5)
        if result.returncode:
            return {"auditable": False, "stderr": result.stderr, "workers": []}
        rows = []
        for line in result.stdout.splitlines():
            pid, _, name = line.partition(",")
            if pid.strip().isdigit() and int(pid) not in exclude:
                rows.append({"pid": int(pid), "name": name.strip()})
        return {"auditable": True, "workers": rows}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"auditable": False, "error": str(error), "workers": []}


def enforce_idle(project_workers, gpu_state, device, allow_contended):
    contended = bool(project_workers or (device.startswith("cuda") and gpu_state["workers"]))
    unknown = device.startswith("cuda") and not gpu_state["auditable"]
    if not allow_contended and (contended or unknown):
        raise ValueError("idle preflight failed: competing workers or GPU inspection unavailable; only explicit --allow-contended permits engineering timing")
    return contended or unknown


def process_memory():
    import psutil
    memory = psutil.Process().memory_info()
    return {"rss_bytes_at_observation": memory.rss,
        "windows_peak_working_set_bytes_process_lifetime": getattr(memory, "peak_wset", None)}


def validate_query_equivalence(prediction, cache_reference, singleton, neural):
    cache_equal = (np.array_equal(prediction, cache_reference) if neural else
                   np.allclose(prediction, cache_reference, rtol=1e-10, atol=1e-10)) if cache_reference is not None else True
    if not cache_equal:
        raise ValueError("cold/warm cache modes changed the model's prediction")
    tolerance = 1e-5 if neural else 1e-10
    difference = float(abs(prediction[0] - singleton)) if singleton is not None else None
    if singleton is not None and not np.isclose(prediction[0], singleton, rtol=tolerance, atol=tolerance):
        raise ValueError("future query depends on the requested curve length")
    return difference


def timed_prediction(predictor, bundle, cell, origin, queries, device, cold, torch):
    if cold:
        bundle.pop("card_cache", None)
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
        baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    else:
        baseline = None
    started = time.perf_counter()
    output = predictor(bundle, cell, origin, queries)
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    elapsed = (time.perf_counter() - started) * 1000
    memory = process_memory()
    if device.startswith("cuda"):
        peak = torch.cuda.max_memory_allocated(device)
        memory.update(cuda_allocated_before_call_bytes=baseline, cuda_peak_allocated_bytes=peak,
            cuda_incremental_peak_allocated_bytes=max(0, peak - baseline),
            cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    return output, elapsed, memory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_folder", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--origin", type=int, default=100)
    parser.add_argument("--max-cells", type=int, default=2)
    parser.add_argument("--query-points", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--allow-contended", action="store_true")
    args = parser.parse_args()
    if min(args.origin, args.max_cells, args.query_points, args.repeats) < 1 or args.warmups < 0:
        raise ValueError("positive counts and nonnegative warmups required")
    model = args.model_folder.resolve()
    run = model.parents[2]
    config = json.loads((run / "config.json").read_text())
    device = config.get("device", "cpu") if args.device == "auto" else args.device
    if "sequence_config" not in config and device != "cpu":
        raise ValueError("tabular artifact cannot be silently rerouted to GPU")
    if not (model / "resources.json").is_file():
        raise ValueError("benchmark only finalized saved artifacts")
    import battery_smetimes
    import scripts
    battery_smetimes.__path__ = [str(run / "source_snapshot/battery_smetimes"), *battery_smetimes.__path__]
    scripts.__path__ = [str(run / "source_snapshot/scripts"), *scripts.__path__]
    from battery_smetimes.research_inference import load_model
    from scripts.prepare_same_type_forecast import save_json, sha
    from scripts.train_same_type_forecast import load_data
    import psutil
    import torch
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    output = Path("experiments/soh_research_v5/resource_benchmarks") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output / "benchmark_source.py")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[
        logging.FileHandler(output / "benchmark.log", encoding="utf-8"), logging.StreamHandler()])
    project_before, gpu_before = other_project_workers(), gpu_compute_workers()
    save_json(output / "config.json", {"model_folder": str(model), "device": device, "origin": args.origin,
        "max_cells": args.max_cells, "query_points": args.query_points, "repeats": args.repeats, "warmups": args.warmups,
        "allow_contended": args.allow_contended, "script_sha256": sha(Path(__file__)),
        "python": sys.version, "torch": torch.__version__, "numpy": np.__version__, "psutil": psutil.__version__,
        "platform": platform.platform(), "processor": platform.processor(), "logical_cpu_count": psutil.cpu_count(),
        "torch_threads": torch.get_num_threads(), "gpu": torch.cuda.get_device_name(device) if device.startswith("cuda") else None,
        "thread_environment": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "runtime": {"float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32, "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32},
        "project_workers_before": project_before, "gpu_compute_before": gpu_before,
        "query_contract": "source N+1 through N+query_points, supplied explicitly without true lifetime or future diagnostic/label information",
        "input_contract": "already extracted early V/I/T features, not raw signal acquisition/extraction time",
        "unit_contract": "technical timing repeats within batteries, not extra independent batteries; no confidence intervals or significance tests"})
    rows, skipped, case_checks, predictions_by_query = [], [], [], {}
    try:
        contended = enforce_idle(project_before, gpu_before, device, args.allow_contended)
        started = time.perf_counter()
        bundle, predictor, evidence = load_model(model, device)
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        model_load_seconds = time.perf_counter() - started
        description = model_description(bundle)
        if args.origin not in bundle["origins"]:
            raise ValueError("observation origin outside frozen experiment")
        audit_dir = run / "frozen_audit"
        frozen = json.loads((audit_dir / "config.json").read_text())
        prepared = Path(json.loads((audit_dir / "audit.json").read_text())["prepared_folder"])
        integrity = frozen["config"]["integrity_audit"]
        if sha(prepared / "manifest.json") != frozen["prepared_manifest_sha256"] or sha(integrity) != frozen["integrity_sha256"]:
            raise ValueError("benchmark's audited input or integrity rules changed")
        started = time.perf_counter()
        _, cells = load_data(prepared, integrity)
        data_setup_seconds = time.perf_counter() - started
        registered = json.loads((audit_dir / "cell_registry.json").read_text())["cells"]
        ids = evidence["plan"]["test"][:args.max_cells]
        for cell_id in ids:
            original = cells[cell_id]
            original.update({name: registered[cell_id][name] for name in ("physical_family", "reference_established_source_cycle")})
            visible = visible_input(original, args.origin)
            for count in sorted({1, args.query_points}):
                queries = np.arange(args.origin + 1, args.origin + count + 1)
                for cache_mode in ("cold_statistics_card_cache", "warm_statistics_card_cache"):
                    cold = cache_mode.startswith("cold")
                    try:
                        # Warmups are excluded; first-call text cache is still cleared on every cold repetition.
                        for _ in range(args.warmups):
                            timed_prediction(predictor, bundle, visible, args.origin, queries, device, cold, torch)
                        samples, memory_samples, reference, repeat_drifts, repeat_bitwise = [], [], None, [], []
                        for repeat in range(args.repeats):
                            prediction, elapsed, memory = timed_prediction(predictor, bundle, visible, args.origin, queries, device, cold, torch)
                            if prediction.shape != (count,) or not np.isfinite(prediction).all():
                                raise ValueError("invalid finite query output")
                            if reference is None:
                                reference = prediction.copy()
                            drift = float(np.max(np.abs(prediction - reference)))
                            identical = bool(np.array_equal(prediction, reference))
                            # Forest's threaded accumulation can reorder floating additions;
                            # retain drift instead of inventing bitwise reproducibility.
                            stable = identical if "network" in bundle else bool(np.allclose(prediction, reference, rtol=1e-10, atol=1e-10))
                            if not stable:
                                raise ValueError(f"same-device repeat changed output beyond declared stability contract: max difference={drift:.12g}")
                            repeat_drifts.append(drift)
                            repeat_bitwise.append(identical)
                            samples.append(elapsed)
                            memory_samples.append(memory)
                            save_json(output / "in_progress.json", {"completed_cases": rows, "cell_id": cell_id,
                                "query_points": count, "cache_mode": cache_mode, "timing_ms": samples, "memory": memory_samples,
                                "repeat_output_max_absolute_differences": repeat_drifts, "repeat_output_bitwise_identical": repeat_bitwise})
                        difference = validate_query_equivalence(reference, predictions_by_query.get((cell_id, count)),
                            predictions_by_query[(cell_id, 1)][0] if (cell_id, 1) in predictions_by_query else None,
                            "network" in bundle)
                        predictions_by_query[(cell_id, count)] = reference.copy()
                        row = {"cell_id": cell_id, "dataset": visible["metadata"]["dataset"], "origin_source_cycle": args.origin,
                            "visible_observations": len(visible["cycle"]), "future_query_points": count, "cache_mode": cache_mode,
                            "statistics_cache_applicable": description["statistics_card_cache_applicable"],
                            "raw_timing_ms": samples, "memory_by_repeat": memory_samples, **latency_summary(samples),
                            "repeat_output_max_absolute_differences": repeat_drifts,
                            "repeat_outputs_bitwise_identical": all(repeat_bitwise),
                            "repeat_stability_contract": "bitwise identical neural output" if "network" in bundle else "existing tabular checkpoint tolerance rtol=atol=1e-10; float addition ordering may differ",
                            "labels_absent": True, "future_raw_rows_absent": True, "prediction_min": float(reference.min()),
                            "prediction_max": float(reference.max()), "single_query_vs_curve_first_point_absolute_difference": difference}
                        rows.append(row)
                        project_now, gpu_now = other_project_workers(), gpu_compute_workers()
                        contended = enforce_idle(project_now, gpu_now, device, args.allow_contended) or contended
                        case_checks.append({"cell_id": cell_id, "points": count, "cache_mode": cache_mode,
                            "project_workers": project_now, "gpu_state": gpu_now})
                        save_json(output / "progress.json", {"evidence": evidence, "description": description, "cases": rows,
                            "skipped": skipped, "contention_checks": case_checks})
                        logging.info("MEASURED %s points=%s %s median_ms=%.4f", cell_id, count, cache_mode, row["median_ms"])
                    except ValueError as error:
                        if "insufficient" in str(error) or "not established" in str(error):
                            skipped.append({"cell_id": cell_id, "points": count, "cache_mode": cache_mode, "reason": str(error)})
                        else:
                            raise
        if not rows:
            raise ValueError("no requested battery/origin was measurable; do not report empty latency as zero")
        project_after, gpu_after = other_project_workers(), gpu_compute_workers()
        contended = enforce_idle(project_after, gpu_after, device, args.allow_contended) or contended
        files = {name: (model / name).stat().st_size for name in evidence["artifact_sha256"]}
        backbone_bytes = None
        if (model / "llm.json").is_file():
            llm = json.loads((model / "llm.json").read_text())
            backbone_bytes = sum((Path(llm["checkpoint"]) / name).stat().st_size for name in llm["checkpoint_files_sha256"])
        report = {"status": "completed engineering measurement under contention" if contended else "completed idle-checked measurement",
            "formal_resource_comparison_eligible": not contended, "evidence": evidence, "model": description,
            "model_load_seconds_including_source_checkpoint_verification": model_load_seconds,
            "audited_full_dataset_setup_seconds_not_single_battery_ingestion": data_setup_seconds,
            "saved_local_artifact_bytes": files, "required_shared_backbone_artifact_bytes": backbone_bytes,
            "process_memory_after_measurements": process_memory(), "cases": rows, "skipped": skipped,
            "selected_test_batteries": ids, "full_test_partition_batteries": len(evidence["plan"]["test"]),
            "project_workers_after": project_after, "gpu_compute_after": gpu_after, "contention_checks": case_checks,
            "limits": "Timing of cached extracted inputs only; finite synthetic requested future grid is not ground truth and not an accuracy evaluation. Repeats are technical. Not a full-fold/full-seed resource ranking. Warm card-cache results do not represent a new battery's first call. GPU peaks are this process's PyTorch allocator, not whole-device occupancy; Windows peak RSS is process lifetime including load."}
        save_json(output / "report.json", report)
        print(json.dumps({"folder": str(output.resolve()), "cases": len(rows), "formal_eligible": not contended}, indent=2))
    except BaseException:
        save_json(output / "failure.json", {"traceback": traceback.format_exc(), "completed_cases": rows, "skipped": skipped})
        raise


if __name__ == "__main__":
    main()
