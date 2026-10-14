"""All-battery diagnostic atlas from completed, static-audited predictions.

Fixed linear/tree/sequence diagnostics, not target-error-selected winners.
No smoothing, interpolation, clipping, fitting, sampling or weight loading.
"""

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import logging
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import ScaledTranslation
import numpy as np

from battery_smetimes.research_baselines import regression_metrics
from battery_smetimes.research_inference import verify_source_snapshot
from battery_smetimes.research_transfer_vi120 import stream_sha
from scripts.prepare_same_type_forecast import save_json


MODELS = ("ridge", "extra_trees", "gru")
ORIGINS = (20, 50, 100)
LABELS = {"ridge": "Ridge", "extra_trees": "ExtraTrees", "gru": "GRU"}
COLORS = {"ridge": "#8B929B", "extra_trees": "#386A93", "gru": "#8A648D"}
STYLES = {"ridge": "--", "extra_trees": "-", "gru": "-."}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def prepare_atlas(horizon_path, audit_path, workspace):
    horizon, audit = read(horizon_path), read(audit_path)
    if audit["audit_contract_schema_version"] != 2 or audit["failed_units"] or audit["failed_runs"]:
        raise ValueError("atlas requires a successful schema-2 static numerical artifact audit")
    rows = {}
    for row in horizon["rows"]:
        if row["model"] in MODELS and row["horizon"] == "all_observed_future":
            key = (row["dataset"], row["model"], row["origin_source_cycle"])
            if key in rows or not row["complete_model_dataset_primary"]:
                raise ValueError("atlas cannot select duplicate/incomplete diagnostic result rows")
            if len(row["training_seeds"]) < 5 or len(set(row["training_seeds"])) != len(row["training_seeds"]):
                raise ValueError("atlas requires at least five unique recorded training seeds")
            rows[key] = row
    if set(rows) != {(dataset, model, origin) for dataset in ("MIT", "XJTU", "NASA") for model in MODELS for origin in ORIGINS}:
        raise ValueError("atlas requires every cohort/model/origin; no selective battery plotting")
    units = {}
    for unit in audit["units"]:
        key = (unit["run"], unit["fold"], unit["seed"], unit["model"])
        if key in units:
            raise ValueError("duplicate audited unit")
        units[key] = unit
    runs, members, provenance = {}, {}, {}
    for run_id in sorted({row["run_id"] for row in rows.values()}):
        run = workspace / "experiments/soh_research_v5/runs" / run_id
        if not run.resolve().is_relative_to((workspace / "experiments/soh_research_v5/runs").resolve()):
            raise ValueError("atlas run outside trusted local root")
        config = verify_source_snapshot(run)
        state = read(run / "progress.json")
        if config["smoke"] or not state["complete"] or (run / "failure.json").exists():
            raise ValueError("atlas only consumes successful complete main runs")
        matches = [record for record in audit["runs"] if record["run"] == str(run.resolve())]
        if len(matches) != 1 or not matches[0]["run_claimed_complete"]:
            raise ValueError("atlas run absent from complete numerical audit")
        recorded = matches[0]
        if stream_sha(run / "config.json") != recorded["run_config_sha256"] or stream_sha(run / "progress.json") != recorded["progress_sha256_at_snapshot"]:
            raise ValueError("atlas audited parent changed")
        for name, digest in recorded["frozen_audit_sha256"].items():
            if stream_sha(run / "frozen_audit" / name) != digest:
                raise ValueError("atlas frozen data/physical/split audit changed")
        plans = read(run / "frozen_audit/splits.json")["plans"]
        current = {(plan["name"].split("_fold")[0], key): plan["name"] for plan in plans for key in plan["test"]}
        if len(current) != sum(len(plan["test"]) for plan in plans):
            raise ValueError("battery occurs in duplicate frozen outer folds")
        if members and current != members:
            raise ValueError("diagnostic models have different frozen test memberships")
        members = current
        registered = read(run / "frozen_audit/cell_registry.json")["cells"]
        for dataset, key in current:
            cell = registered[key]
            if cell["metadata"]["dataset"] != dataset or stream_sha(cell["prepared_file"]) != cell["prepared_sha256"]:
                raise ValueError("atlas original prepared cell changed")
        runs[run_id] = run
        provenance[run_id] = {"config_sha256": recorded["run_config_sha256"], "progress_sha256": recorded["progress_sha256_at_snapshot"],
            "frozen_audit_sha256": recorded["frozen_audit_sha256"]}
    for dataset in ("MIT", "XJTU", "NASA"):
        for origin in ORIGINS:
            sets = [{item["cell_id"] for item in rows[(dataset, model, origin)]["by_cell_training_seed_variation"]} for model in MODELS]
            if any(ids != sets[0] for ids in sets[1:]):
                raise ValueError("diagnostic models disagree on independently eligible batteries")
    return {"rows": rows, "units": units, "runs": runs, "members": members, "run_provenance": provenance,
        "cache": {}, "prediction_sources": {}, "horizon_sha256": stream_sha(horizon_path), "audit_sha256": stream_sha(audit_path)}


def curves_for_cell(atlas, dataset, key, origin, model):
    row = atlas["rows"][(dataset, model, origin)]
    eligible = {item["cell_id"]: item for item in row["by_cell_training_seed_variation"]}
    if key not in eligible:
        return None
    own = eligible[key]
    if own["fold"] != atlas["members"][(dataset, key)]:
        raise ValueError("curve cell does not match frozen test fold")
    predictions, cycle, truth = [], None, None
    prefix = f"{key}__{origin}__"
    for seed in row["training_seeds"]:
        unit = atlas["units"][(row["run_id"], own["fold"], seed, model)]
        if unit["status"] != "passed static artifact/metric audit":
            raise ValueError("curve unit did not pass static numerical audit")
        path = atlas["runs"][row["run_id"]] / own["fold"] / str(seed) / model / "test/predictions.npz"
        expected = unit["artifact_sha256"][str(Path("test") / "predictions.npz")]
        if path not in atlas["cache"]:
            if stream_sha(path) != expected:
                raise ValueError("curve predictions changed after numerical audit")
            with np.load(path, allow_pickle=False) as saved:
                arrays = {name: saved[name].copy() for name in saved.files}
            if stream_sha(path) != expected:
                raise ValueError("curve predictions changed during read")
            atlas["cache"][path] = arrays
            atlas["prediction_sources"][str(path.resolve())] = expected
        arrays = atlas["cache"][path]
        current_cycle, current_truth, prediction = [arrays[prefix+name] for name in ("cycle", "truth", "prediction")]
        if (current_cycle.ndim != 1 or len(current_cycle) < 2 or current_truth.shape != current_cycle.shape or prediction.shape != current_cycle.shape
                or not all(np.isfinite(value).all() for value in (current_cycle, current_truth, prediction))
                or not np.all(np.diff(current_cycle) > 0) or np.any(current_cycle <= origin) or np.any(current_truth <= 0)):
            raise ValueError("invalid observed future curve")
        if cycle is None:
            cycle, truth = current_cycle, current_truth
        elif not np.array_equal(cycle, current_cycle) or not np.array_equal(truth, current_truth):
            raise ValueError("training seed curves use different target grids or truth")
        predictions.append(prediction)
    values = np.asarray(predictions)
    mapes = [regression_metrics(truth, prediction)["mape_percent"] for prediction in values]
    if not np.isclose(np.mean(mapes), own["mape_percent"]["mean"], rtol=1e-10, atol=1e-10):
        raise ValueError("plot curve metrics differ from audited result table")
    return {"cycle": cycle, "truth": truth, "predictions": values, "mean": values.mean(0), "sd": values.std(0, ddof=1),
        "seeds": row["training_seeds"], "mape_mean": float(np.mean(mapes)), "mape_seed_sd": float(np.std(mapes, ddof=1))}


def draw_cell(dataset, key, curves):
    available = [value for value in curves.values() if value is not None]
    if not available:
        raise ValueError("no eligible origin for a frozen battery; record a coverage page instead of dropping it")
    fig, axes = plt.subplots(1, 3, figsize=(183/25.4, 98/25.4), sharex=True, sharey=True)
    fig.subplots_adjust(left=.075, right=.98, top=.76, bottom=.28, wspace=.19)
    low = min(float(np.min(np.minimum(value["truth"], value["mean"]-value["sd"]))) for value in available)
    high = max(float(np.max(np.maximum(value["truth"], value["mean"]+value["sd"]))) for value in available)
    margin = max(.015, .07*(high-low))
    last = max(float(value["cycle"][-1]) for value in available)
    for axis, origin, label in zip(axes, ORIGINS, "abc"):
        panel = [curves[(origin, model)] for model in MODELS if curves[(origin, model)] is not None]
        axis.axvspan(0, origin, color="#E6E9ED", alpha=.55, linewidth=0)
        axis.axvline(origin, color="#ADB3BA", linewidth=.7, linestyle=":")
        if panel:
            reference = panel[0]
            axis.plot(reference["cycle"], reference["truth"], color="#202B34", linewidth=1.05, zorder=5)
            for model in MODELS:
                value = curves[(origin, model)]
                if value is None or not np.array_equal(value["cycle"], reference["cycle"]) or not np.array_equal(value["truth"], reference["truth"]):
                    raise ValueError("model comparisons have different future targets")
                axis.plot(value["cycle"], value["mean"], color=COLORS[model], linestyle=STYLES[model], linewidth=.9)
                axis.fill_between(value["cycle"], value["mean"]-value["sd"], value["mean"]+value["sd"], color=COLORS[model], alpha=.12, linewidth=0)
            note = " / ".join(f'{curves[(origin, model)]["mape_mean"]:.2f}' for model in MODELS)
            axis.set_title(f"N = {origin}\nMAPE (%): {note}", fontsize=7, pad=8)
        else:
            axis.set_title(f"N = {origin}\nNot eligible", fontsize=7, pad=8)
            axis.text(.5, .5, "Input/reference\ncoverage gap", transform=axis.transAxes, ha="center", va="center", fontsize=7)
        axis.set_xlim(0, last)
        axis.set_ylim(low-margin, high+margin)
        axis.set_xlabel("Source cycle", fontsize=7)
        axis.xaxis.set_major_locator(MaxNLocator(4))
        axis.yaxis.set_major_locator(MaxNLocator(4))
        if panel:
            axis.grid(axis="y", alpha=.12)
        transform = axis.transAxes+ScaledTranslation(-10/72, 5/72, fig.dpi_scale_trans)
        axis.text(0, 1, label, transform=transform, fontsize=8, fontweight="bold", va="bottom")
    axes[0].set_ylabel("SOH (initial reference)", fontsize=7)
    fig.suptitle(f"{dataset} {key}: unseen-cell future SOH", y=.97, fontsize=9)
    subtitle = "Physically unverified four-cell NASA supplement" if dataset == "NASA" else (
        "XJTU within-cohort validation; temperature unavailable" if dataset == "XJTU" else "Public-development within-cohort grouped cell validation")
    fig.text(.52, .91, subtitle, ha="center", fontsize=6.5)
    handles = [Line2D([], [], color="#202B34", label="Observed SOH")]
    handles.extend(Line2D([], [], color=COLORS[model], linestyle=STYLES[model], label=LABELS[model]) for model in MODELS)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.52, .13), ncol=4, frameon=False, fontsize=6.5)
    fig.text(.52, .103, "Forecast lines: five-seed means; bands: seed SD, not confidence intervals.", ha="center", fontsize=6)
    fig.text(.52, .068, "Shading: available early-cycle signals; observed SOH is not an input.", ha="center", fontsize=6)
    fig.text(.52, .032, "Future intervals differ. Seed-mean MAPE order: Ridge / ExtraTrees / GRU.", ha="center", fontsize=6)
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon-report", required=True, type=Path)
    parser.add_argument("--artifact-audit", required=True, type=Path)
    parser.add_argument("--qa-scripts", required=True, type=Path)
    args = parser.parse_args()
    output = Path("results/soh_research_v5_cell_atlas") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output / "plot_source.py")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[logging.FileHandler(output / "atlas.log", encoding="utf-8"), logging.StreamHandler()])
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    qa = args.qa_scripts.resolve()
    config = {"horizon_report": str(args.horizon_report.resolve()), "artifact_audit": str(args.artifact_audit.resolve()),
        "models": MODELS, "origins": ORIGINS, "source_sha256": stream_sha(__file__), "python": sys.version,
        "numpy": np.__version__, "matplotlib": matplotlib.__version__, "backend": "Python",
        "qa_helpers_sha256": {name: stream_sha(qa / name) for name in ("validate_figure.py", "audit_panel_alignment.py", "audit_pdf_text.py", "audit_figure_collisions.py")},
        "contract": {"question": "For every held-out battery, where do fixed linear/tree/sequence forecasts depart from observed future SOH?",
            "claim": "All frozen eligible cells, including difficult cells and input-coverage gaps, are visible; no aggregate superiority claim.",
            "archetype": "quantitative diagnostic grid", "panel_roles": "earlier / intermediate / longer visible prefix; future intervals differ",
            "scope": "fixed diagnostic model subset; full-model main comparison and LLM completion remain separate",
            "uncertainty": "five training seeds, one sample SD; no CI, p values or population inference",
            "transformations": "no smoothing/interpolation/clipping/monotonic correction or extrapolation; actual saved future targets",
            "selection": "every frozen test cell; fixed Ridge/ExtraTrees/GRU, not winners selected from test error",
            "dimensions_mm": [183, 98], "minimum_glyph_pt": 5}, "weights_loaded": False, "training_performed": False}
    save_json(output / "config.json", config)
    completed = []
    try:
        preflight = subprocess.run([sys.executable, str(qa / "validate_figure.py"), __file__, "--json"], capture_output=True, text=True)
        (output / "source-preflight.json").write_text(preflight.stdout+preflight.stderr, encoding="utf-8")
        if preflight.returncode:
            raise ValueError("atlas source preflight failed")
        atlas = prepare_atlas(args.horizon_report, args.artifact_audit, Path.cwd().resolve())
        save_json(output / "verified_sources.json", {"run_provenance": atlas["run_provenance"], "horizon_sha256": atlas["horizon_sha256"],
            "audit_sha256": atlas["audit_sha256"], "full_cohort_cells": len(atlas["members"]),
            "static_not_executable_weight_replay": True})
        spec = importlib.util.spec_from_file_location("audit_panel_alignment", qa / "audit_panel_alignment.py")
        alignment = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = alignment
        spec.loader.exec_module(alignment)
        matplotlib.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 7,
            "xtick.labelsize": 6, "ytick.labelsize": 6, "pdf.fonttype": 42, "svg.fonttype": "none",
            "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .65})
        for (dataset, key), fold in sorted(atlas["members"].items()):
            curves = {(origin, model): curves_for_cell(atlas, dataset, key, origin, model) for origin in ORIGINS for model in MODELS}
            folder = output / dataset / key
            folder.mkdir(parents=True, exist_ok=False)
            arrays, numbers = {}, []
            for (origin, model), value in curves.items():
                numbers.append({"origin": origin, "model": model, "eligible": value is not None,
                    "mape_mean": value["mape_mean"] if value is not None else None,
                    "mape_seed_sd": value["mape_seed_sd"] if value is not None else None})
                if value is not None:
                    arrays.update({f"{origin}__{model}__{name}": item for name, item in value.items() if isinstance(item, np.ndarray)})
            np.savez_compressed(folder / "source_data.npz", **arrays)
            save_json(folder / "numbers.json", {"dataset": dataset, "cell_id": key, "fold": fold, "numbers": numbers})
            fig = draw_cell(dataset, key, curves)
            alignment.require_matplotlib_panel_alignment(fig, json_out=str(folder / "alignment.json"), tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
            fig.savefig(folder / "curves.pdf")
            fig.savefig(folder / "curves.svg")
            fig.savefig(folder / "curves.png", dpi=400)
            plt.close(fig)
            for command, log in (([sys.executable, str(qa / "audit_pdf_text.py"), str(folder / "curves.pdf"), "--min-pt", "5", "--json"], "text-audit.json"),
                ([sys.executable, str(qa / "audit_figure_collisions.py"), str(folder / "curves.pdf"), "--json-out", str(folder / "collision-audit.json"), "--overlay-pdf", str(folder / "collision-audit.pdf")], "collision-audit.log")):
                result = subprocess.run(command, capture_output=True, text=True)
                (folder / log).write_text(result.stdout+result.stderr, encoding="utf-8")
                if result.returncode:
                    raise ValueError("atlas rendered geometry QA failed: " + dataset + " " + key)
            completed.append({"dataset": dataset, "cell_id": key, "fold": fold, "folder": str(folder.resolve()),
                "automated_geometry_qa_completed": True, "visual_review_completed": False})
            save_json(output / "progress.json", {"complete": False, "expected_cells": len(atlas["members"]), "completed": completed})
            logging.info("ATLAS %s %s %s/%s", dataset, key, len(completed), len(atlas["members"]))
        for path, digest in atlas["prediction_sources"].items():
            if stream_sha(path) != digest:
                raise ValueError("atlas source changed during full rendering")
        save_json(output / "prediction_sources.json", atlas["prediction_sources"])
        save_json(output / "progress.json", {"complete": True, "expected_cells": len(atlas["members"]), "completed": completed,
            "all_visual_reviews_completed": False, "not_full_goal_completion": True})
        print(json.dumps({"folder": str(output.resolve()), "rendered_batteries": len(completed), "status": "automated QA complete; per-panel visual review still required"}, indent=2))
    except BaseException:
        save_json(output / "failure.json", {"traceback": traceback.format_exc(), "completed": completed,
            "weights_loaded": False, "training_performed": False})
        raise


if __name__ == "__main__":
    main()
