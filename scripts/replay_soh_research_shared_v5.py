"""Fresh frozen-source shared neural/baseline replay; all units by default."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import traceback

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(name, "2")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
WORKSPACE = Path.cwd().resolve()
sys.path.insert(0, str(WORKSPACE))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--baseline", action="store_true", help="explicit Ridge/ExtraTrees protocol; never inferred from folder contents")
    parser.add_argument("--device", default="auto", help="recorded backend by default; explicit portability is separately labelled")
    parser.add_argument("--direction", action="append")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--model", action="append")
    parser.add_argument("--origin", type=int, action="append")
    parser.add_argument("--max-cells", type=int, help="default all cells; a subset never certifies unchecked cells")
    args = parser.parse_args()
    run = args.run.resolve()
    protocol = "baseline" if args.baseline else "neural"
    replay_name = f"research_transfer_{protocol}_replay"
    root = WORKSPACE / f"experiments/soh_research_v5/transfer_{protocol}_runs"
    if not run.is_relative_to(root) or run == root or not (WORKSPACE / "battery_smetimes").is_dir():
        raise ValueError("shared replay requires an explicit trusted-local run inside the original workspace")
    output = WORKSPACE / f"experiments/soh_research_v5/shared_{protocol}_replays" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output / "replay_entry.py")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[
        logging.FileHandler(output / "replay.log", encoding="utf-8"), logging.StreamHandler()])
    rows, unit_evidence = [], []
    save(output / "config.json", {"run": str(run), "protocol": protocol, "requested_device": args.device,
        "directions": args.direction, "seeds": args.seed, "models": args.model, "origins": args.origin,
        "max_cells": args.max_cells, "python": sys.version, "entry_sha256": digest(__file__),
        "training_performed": False, "target_calibration": [], "not_same_type_deployment": True})
    try:
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        state = json.loads((run / "progress.json").read_text(encoding="utf-8"))
        if (config["prepare_only"] or not state["complete"] or not state["model_training_started"]
                or (run / "failure.json").exists() or (run / "launcher_failure.json").exists()):
            raise ValueError("shared replay requires successful completed real fitting; no weight loading from prepared/partial/failed runs")
        if args.max_cells is not None and args.max_cells < 1:
            raise ValueError("shared replay max_cells must be positive")
        if args.origin is not None and (len(args.origin) != len(set(args.origin)) or set(args.origin)-set(config["origins"])):
            raise ValueError("shared replay origins duplicate or outside frozen experiment")
        snapshot = run / "source_snapshot"
        for name, expected in config["source_sha256"].items():
            path = snapshot / name
            if not path.resolve().is_relative_to(snapshot.resolve()) or digest(path) != expected:
                raise ValueError("shared replay frozen source changed")
        required = {"scripts/replay_soh_research_shared_v5.py", f"battery_smetimes/{replay_name}.py",
            "battery_smetimes/research_transfer_replay_checks.py"}
        if not required.issubset(config["source_sha256"]):
            raise ValueError("this run has no frozen shared replay implementation; do not upgrade its source in place")
        if digest(__file__) != config["source_sha256"]["scripts/replay_soh_research_shared_v5.py"]:
            raise ValueError("use this run's own frozen replay entrypoint, not a later implementation")
        sys.path.insert(0, str(snapshot))
        import battery_smetimes
        import scripts
        battery_smetimes.__path__ = [str(snapshot / "battery_smetimes"), *battery_smetimes.__path__]
        scripts.__path__ = [str(snapshot / "scripts"), *scripts.__path__]
        import importlib
        replay = importlib.import_module("battery_smetimes." + replay_name)
        if Path(replay.__file__).resolve() != (snapshot / f"battery_smetimes/{replay_name}.py").resolve():
            raise ValueError("shared replay verifier resolved outside its frozen source")
        shutil.copy2(replay.__file__, output / "replay_verifier.py")
        import numpy as np
        runtime = {"backend": "original CPU sklearn package versions", "torch_imported": False}
        if not args.baseline:
            import torch
            torch.set_num_threads(2)
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
            runtime = {"torch": torch.__version__, "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "cpu_mkldnn_enabled": torch.backends.mkldnn.enabled,
                "transformer_fastpath_enabled": torch.backends.mha.get_fastpath_enabled()}
        verified = replay.verify_replay_run(run, progress_callback=lambda key, count, total:
            logging.info("SHARED REPLAY INPUT %s %s/%s", key, count, total))
        save(output / "verified_contract.json", {**verified["evidence"], "protocol": protocol, "runtime": runtime})
        selected = {}
        for name, requested, available in (("direction", args.direction, [p["name"] for p in verified["plans"]]),
                                          ("seed", args.seed, config["seeds"]), ("model", args.model, config["models"])):
            chosen = available if requested is None else requested
            if not chosen or len(chosen) != len(set(chosen)) or set(chosen)-set(available):
                raise ValueError("shared replay selector empty/duplicate/outside frozen grid: " + name)
            selected[name] = chosen
        for plan in verified["plans"]:
            if plan["name"] not in selected["direction"]:
                continue
            for seed in selected["seed"]:
                for model in selected["model"]:
                    bundle, evidence = replay.load_verified_unit(verified, plan, seed, model, device=args.device)
                    unit = run / plan["name"] / str(seed) / model
                    checked = output / plan["name"] / str(seed) / model
                    checked.mkdir(parents=True, exist_ok=False)
                    save(checked / "config.json", {"evidence": evidence, "origins": args.origin,
                        "max_cells": args.max_cells, "training_performed": False})
                    unit_evidence.append({**evidence, "replay_complete": False})
                    def progress(current_rows, arrays):
                        save(checked / "progress.json", {"evidence": evidence, "rows": current_rows})
                        np.savez_compressed(checked / "replayed_predictions.npz", **arrays)
                    with np.load(unit / "target_test/predictions.npz", allow_pickle=False) as saved:
                        checked_rows, _ = replay.replay_unit(bundle, verified["cells"], plan, saved,
                            origins=args.origin, max_cells=args.max_cells, progress_callback=progress)
                    save(checked / "report.json", {"evidence": evidence, "rows": checked_rows,
                        "scope": "selected target cells/origins only; fixed weights; no accuracy tuning or deployment claim"})
                    unit_evidence[-1]["replay_complete"] = True
                    rows.extend({"direction": plan["name"], "seed": seed, "model": model, **row} for row in checked_rows)
                    save(output / "progress.json", {"units": unit_evidence, "rows": rows})
                    del bundle
                    if not args.baseline and evidence["device"].startswith("cuda"):
                        torch.cuda.empty_cache()
        report = {"status": "passed shared saved-checkpoint causal replay", "verified_contract": verified["evidence"],
            "units": unit_evidence, "rows": rows, "checked_observations": sum(r.get("passed", False) for r in rows),
            "scope": "explicitly checked final models/cells/origins only; source validation candidates not replayed; not an independent blind test or inference benchmark"}
        save(output / "report.json", report)
        print(json.dumps({"folder": str(output), "status": report["status"], "checked_final_models": len(unit_evidence),
            "checked_observations": report["checked_observations"]}, indent=2))
    except BaseException:
        save(output / "failure.json", {"traceback": traceback.format_exc(), "units": unit_evidence, "rows": rows,
            "training_performed": False})
        raise


if __name__ == "__main__":
    main()
