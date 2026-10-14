"""Regenerate a conservative coverage report from the v2 manifest and registry.

Only ingested cells count as training coverage. Candidate records are listed
separately and are not silently added to the training total.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def describe_values(cells, field, suffix=""):
    counts = Counter(str(cell[field]) if cell.get(field) is not None else "unknown"
                     for cell in cells)
    return ", ".join(f"{value}{suffix}: {count}" for value, count in sorted(counts.items()))


def render(manifest, registry, sample_audit=None, exact80_audit=None):
    by_dataset = defaultdict(list)
    for cell in manifest["cells"]:
        by_dataset[cell["dataset"]].append(cell)
    lines = [
        "# Battery data coverage (ingested v2 only)",
        "",
        "Generated from `battery_data_v2/manifest.json`,",
        "`metadata/dataset_registry.yaml`, and the separately named endpoint audit",
        "when available. Candidate data are **not** counted as",
        "training cells. An observation is a retained capacity diagnostic, not",
        "necessarily a full cycling event.",
        "",
        "| Dataset | Chemistry (cells) | Cells | Capacity observations | SOH labels | Source-contract endpoints (not unified 80% EOL) |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for dataset, cells in sorted(by_dataset.items()):
        lines.append(
            f"| {dataset} | {describe_values(cells, 'chemistry')} | {len(cells)} | "
            f"{sum(c['rows'] for c in cells)} | {sum(c['soh_examples'] for c in cells)} | "
            f"{sum(bool(c['endpoint_observed']) for c in cells)} |"
        )
    lines += [
        "",
        f"Total: {len(manifest['cells'])} cells, "
        f"{sum(c['rows'] for c in manifest['cells'])} retained capacity observations.",
        "",
        "The source-contract endpoint field follows each source's original",
        "label convention; it does **not** count confirmed 80%-SOH threshold",
        "events. In particular, the 55 XJTU entries must not be interpreted",
        "as 55 observed project-defined RUL-to-80% labels.",
        "",
    ]
    if exact80_audit:
        lines += [
            "## Confirmed per-cell 80%-SOH endpoints (audited separately)",
            "",
            "| Dataset | Confirmed 80% events | Right-censored for 80% | Source-defined endpoints |",
            "| --- | ---: | ---: | ---: |",
        ]
        for dataset, row in sorted(exact80_audit["domains"].items()):
            lines.append(
                f"| {dataset} | {row['confirmed_event_cells']} | "
                f"{row['right_censored_for_confirmed_event']} | "
                f"{row['source_endpoint_cells']} |")
        lines += [
            "",
            "These are observed events under the audited three-consecutive-",
            "diagnostic confirmation rule, not necessarily precise physical",
            "crossing times. A censored cell has no known 80% RUL point target.",
            "",
        ]
    lines += [
        "## Condition and waveform coverage by dataset",
        "",
        "| Dataset | Ambient temperature (cells) | Charge C-rate (cells) | Discharge C-rate (cells) | DoD | Measured temperature waveform (cells) |",
        "| --- | --- | --- | --- | --- | ---: |",
    ]
    for dataset, cells in sorted(by_dataset.items()):
        lines.append(
            f"| {dataset} | {describe_values(cells, 'ambient_temperature_c')} | "
            f"{describe_values(cells, 'charge_rate_c')} | "
            f"{describe_values(cells, 'discharge_rate_c')} | "
            f"not structured in v2 | {sum(bool(c.get('temperature_available')) for c in cells)}/{len(cells)} |"
        )
    lines += [
        "",
        "V/I waveforms are available through the existing per-dataset raw adapters,",
        "but this report does not assert their sampling completeness. CALCE's",
        "accepted Step 7 diagnostics do not contain measured temperature. The",
        "missing chemistry for NASA is intentionally not guessed. The v2 manifest",
        "does not structure per-cell DoD or MIT multi-stage charge rate into one",
        "scalar C-rate; `unknown` is not evidence of a zero rate.",
        "",
        "## Candidate domains, not yet ingested",
        "",
        "| Candidate | Chemistry | Claimed cells | Temperature conditions | Raw V/I/T claimed | Access status |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    for row in registry["datasets"]:
        vit = "/".join("yes" if row[f"raw_{name}_available"] is True else
                       "no" if row[f"raw_{name}_available"] is False else "?"
                       for name in "VIT")
        lines.append(
            f"| {row['dataset_id']} | {row['chemistry'] or 'unknown'} | "
            f"{row['cell_count'] or 'unknown'} | {row['temperature_conditions'] or 'unknown'} | "
            f"{vit} | {row['access_status']} |"
        )
    if sample_audit:
        lines += [
            "",
            "## Panasonic raw intake sample (excluded from training)",
            "",
            "| Cell | Aging profile | Aging mode | Qualified 25°C RPTs | 80% crossing evidence | Invalid temperature points |",
            "| --- | --- | --- | ---: | --- | ---: |",
        ]
        for cell in sample_audit["cells"]:
            crossing = cell["eol80_rpt_interval"]
            if crossing["status"] == "interval_observed_not_exact_cycle":
                crossing_text = (f"between tests {crossing['last_above_test_id']} and "
                                 f"{crossing['first_below_test_id']}; not exact")
                mode = (cell["aging_profile_metadata"] or {}).get("aging_mode")
                if mode == "cyclic" and crossing.get("last_above_equivalent_full_cycles_candidate") is not None:
                    crossing_text += (
                        f"; EFC {crossing['last_above_equivalent_full_cycles_candidate']:.1f}"
                        f"–{crossing['first_below_equivalent_full_cycles_candidate']:.1f}")
                elif mode in ("calendar", "near_calendar"):
                    crossing_text += "; calendar, no cycle-RUL"
            else:
                crossing_text = "not observed/confirmed"
            lines.append(
                f"| {cell['cell_id']} | {cell['aging_profile_code'] or 'unknown'} | "
                f"{(cell['aging_profile_metadata'] or {}).get('aging_mode', 'unknown')} | "
                f"{cell['qualified_rpt_count']} | {crossing_text} | "
                f"{cell['invalid_temperature_samples']} |"
            )
        lines += [
            "",
            "RPT count is not cycling count. Provisional capacities are the median of",
            "two protocol-qualified 1C discharges; calendar-aging cells cannot be",
            "assigned cycle-RUL. Source EOL was 70% or project end, whereas the",
            "project target is a separately audited 80% crossing.",
        ]
    lines += [
        "",
        "## Gaps that matter for the model",
        "",
        "- No additional source has passed raw-file QA, label harmonization, or",
        "  cell-level deduplication yet; cross-dataset claims remain four-source.",
        "- Ambient temperature and DoD are largely unstructured in the current",
        "  manifest. The dataset cannot yet support a fair temperature/DoD ablation.",
        "- The four sources use different observed endpoint rules; raw RUL cycle",
        "  errors must not be pooled until a common 80%-SOH EOL task is audited.",
        "- Panasonic would add within-format condition diversity; HUST would add",
        "  LFP discharge protocols; RWTH would add NMC replicate-cell diversity;",
        "  the large-format LFP source would add calendar-aging stress.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "battery_data_v2/manifest.json")
    parser.add_argument("--registry", type=Path, default=ROOT / "metadata/dataset_registry.yaml")
    parser.add_argument("--panasonic-audit", type=Path,
                        default=ROOT / "data/interim/panasonic_ncr18650b/intake_qa.json")
    parser.add_argument("--exact80-audit", type=Path,
                        default=ROOT / "battery_models_v2/exact80_target_audit.json")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/data_coverage_report.md")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    registry = yaml.safe_load(args.registry.read_text(encoding="utf-8"))
    sample_audit = (json.loads(args.panasonic_audit.read_text(encoding="utf-8"))
                    if args.panasonic_audit.exists() else None)
    exact80_audit = (json.loads(args.exact80_audit.read_text(encoding="utf-8"))
                     if args.exact80_audit.exists() else None)
    output = render(manifest, registry, sample_audit, exact80_audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
