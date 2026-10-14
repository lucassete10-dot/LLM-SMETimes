"""Independent complete shared neural/baseline re-audit; never load/train weights."""

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import shutil
import sys
import traceback

sys.path.insert(0, str(Path.cwd().resolve()))

from battery_smetimes.research_inference import verify_source_snapshot
from battery_smetimes.research_transfer_neural_audit import audit_shared_neural_run
from scripts.train_soh_research_transfer_v5 import load_shared_cells
from scripts.prepare_same_type_forecast import sha, save_json


SOURCES = (
    "scripts/audit_soh_research_shared_transfer_v5.py", "battery_smetimes/research_transfer_neural_audit.py",
    "scripts/audit_soh_research_artifacts_v5.py", "scripts/train_soh_research_transfer_v5.py",
    "scripts/train_soh_research_transfer_neural_v5.py", "battery_smetimes/research_transfer_baselines.py",
    "battery_smetimes/research_transfer_sequences.py", "battery_smetimes/research_transfer_signals.py",
    "battery_smetimes/research_transfer_vi120.py", "battery_smetimes/research_phase_gate.py",
    "battery_smetimes/research_inference.py", "battery_smetimes/research_conditions_execution.py",
    "battery_smetimes/research_generalization_splits.py", "battery_smetimes/research_baselines.py",
    "battery_smetimes/research_sequences.py", "battery_smetimes/research_llm.py",
    "battery_smetimes/forecast.py", "battery_smetimes/physical_types.py",
    "scripts/train_same_type_forecast.py", "scripts/prepare_same_type_forecast.py", "prepare_mit.py",
    "scripts/train_soh_research_transfer_baselines_v5.py", "battery_smetimes/research_transfer_baseline_execution.py",
    "battery_smetimes/research_transfer_baseline_audit.py", "battery_smetimes/research_transfer_neural_recovery.py",
    "prepare_xjtu.py", "adapt_nasa_mat_v2.py", "audit_mit_label_free_extraction_v2.py", "battery_v2_features.py",
    "battery_smetimes/research_splits.py", "scripts/summarize_soh_research_v5.py",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--baseline", action="store_true", help="Explicit new shared Ridge/ExtraTrees run, not legacy transfer_runs")
    args = parser.parse_args()
    run = args.run.resolve()
    output = Path("experiments/soh_research_v5/shared_baseline_audits" if args.baseline else "experiments/soh_research_v5/shared_neural_audits") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[
        logging.FileHandler(output / "audit.log", encoding="utf-8"), logging.StreamHandler()])
    hashes = {}
    for name in SOURCES:
        destination = output / "source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(name, destination)
        hashes[name] = sha(destination)
    save_json(output / "config.json", {"run": str(run), "run_config_sha256": sha(run / "config.json"),
        "source_sha256": hashes, "model_training_performed": False, "weights_deserialized": False,
        "evaluation_protocol": "shared_baseline" if args.baseline else "shared_neural",
        "scope": "complete same-source shared run, raw/prepared input and static numerical artifact recheck; no predictive weight replay"})
    try:
        expected_root = Path("experiments/soh_research_v5/transfer_baseline_runs" if args.baseline else "experiments/soh_research_v5/transfer_neural_runs").resolve()
        if not run.is_relative_to(expected_root) or run == expected_root:
            raise ValueError("shared audit requires the explicit matching local experiment protocol/root")
        config = verify_source_snapshot(run)
        state = json.loads((run / "progress.json").read_text(encoding="utf-8"))
        if config["prepare_only"] or not state["complete"] or not state["model_training_started"]:
            raise ValueError("shared run has not completed real fitting; prepared/partial runs are not numerical results")
        cells = load_shared_cells(Path(config["prepared"]), progress_callback=lambda key, count, total:
            logging.info("SHARED AUDIT INPUT %s %s/%s", key, count, total))
        if args.baseline:
            from battery_smetimes.research_transfer_baseline_audit import audit_baseline_run
            report = audit_baseline_run(run, cells)
        else:
            report = audit_shared_neural_run(run, cells)
        if json.loads((run / "static_run_audit.json").read_text(encoding="utf-8")) != report:
            raise ValueError("original run finalization audit differs from current complete audit")
        save_json(output / "report.json", report)
        print(json.dumps({"folder": str(output.resolve()), "status": report["status"],
                          "finalized_units": report["finalized_units"]}, indent=2))
    except BaseException:
        save_json(output / "failure.json", {"traceback": traceback.format_exc(), "model_training_performed": False,
            "weights_deserialized": False})
        raise


if __name__ == "__main__":
    main()
