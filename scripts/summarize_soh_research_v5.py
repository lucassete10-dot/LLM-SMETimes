"""Rebuild experiment metric logs, retaining failed and partial runs separately."""

import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from scripts.prepare_same_type_forecast import save_json


def summarize():
    rows, statuses = [], []
    for run in sorted(Path("experiments/soh_research_v5/runs").glob("*")):
        if not (run / "config.json").exists():
            continue
        config = json.loads((run / "config.json").read_text())
        state = json.loads((run / "progress.json").read_text())
        status = "failed_partial" if (run / "failure.json").exists() else "complete" if state["complete"] else "in_progress"
        statuses.append({"run_id": run.name, "status": status, "smoke": config["smoke"],
            "completed_models": state["completed"], "folder": str(run.resolve())})
        for file in sorted(run.glob("*/*/*/test/metrics.json")):
            fold, seed, model = file.relative_to(run).parts[:3]
            report = json.loads(file.read_text())
            model_config = json.loads((file.parent.parent / "config.json").read_text())
            evidence = config["evidence_level"] if config["smoke"] else model_config["plan"]["evidence_level"]
            for cell in report["by_cell"]:
                measures = {"all_observed_future": cell["metrics"], "late_half": cell["late_half"],
                    **{f"within_{h}": metric for h, metric in cell["within_horizon"].items() if metric is not None}}
                for horizon, metric in measures.items():
                    rows.append({"run_id": run.name, "run_status": status, "evidence_level": evidence,
                        "dataset": fold.split("_fold")[0], "physical_family": cell["physical_family"],
                        "fold": fold, "seed": int(seed), "model": model,
                        "cell_id": cell["cell_id"], "origin_source_cycle": cell["origin"],
                        "horizon": horizon, **metric, "metric_source": str(file.resolve())})
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    archive = Path("results/soh_research_v5") / stamp
    archive.mkdir(parents=True, exist_ok=False)
    fields = ["run_id", "run_status", "evidence_level", "dataset", "physical_family", "fold", "seed", "model",
              "cell_id", "origin_source_cycle", "horizon", "rmse_soh", "mae_soh", "mape_percent", "r2", "metric_source"]
    csv_path = archive / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    save_json(archive / "experiment_status.json", {"runs": statuses,
        "note": "Do not pool smoke/failed/partial runs into five-fold five-seed results. Deterministic Ridge repetitions are not independent stochastic training evidence."})
    # A standard latest entry point plus immutable historical snapshots.
    destination = Path("results/summary.csv")
    if destination.exists():
        with destination.open(encoding="utf-8") as handle:
            if next(csv.reader(handle)) != fields:
                raise ValueError("refusing to overwrite a pre-existing incompatible results table")
    temporary = destination.with_suffix(".csv.part")
    temporary.write_bytes(csv_path.read_bytes())
    temporary.replace(destination)
    save_json(Path("results/latest_soh_research_v5.json"), {"folder": str(archive.resolve())})
    return {"metric_rows": len(rows), "runs": len(statuses), "folder": str(archive.resolve())}


if __name__ == "__main__":
    print(json.dumps(summarize(), indent=2))
