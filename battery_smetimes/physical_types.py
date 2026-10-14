"""Evidence-based physical gates. No clustering, labels or cycle-life inputs."""

import json
import math
from pathlib import Path
import re

REGISTRY = Path(__file__).resolve().parents[1] / "configs/battery_physical_registry.json"
TEXT_FIELDS = ("chemistry", "manufacturer", "model", "form_factor")
NUMERIC_FIELDS = ("nominal_capacity_ah", "nominal_voltage_v",
                  "rated_voltage_min_v", "rated_voltage_max_v")


def load_registry(path=REGISTRY):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical(field, value):
    if value is None or not str(value).strip():
        return None
    if field in NUMERIC_FIELDS:
        if isinstance(value, bool):
            raise ValueError(f"invalid {field}: boolean")
        number = float(value)
        if not math.isfinite(number) or number <= 0:
            raise ValueError(f"invalid {field}: expected finite positive value")
        return number
    compact = re.sub(r"[^a-z0-9]", "", str(value).lower())
    aliases = {
        "chemistry": {"lfpgraphite": "lfpgraphite", "lifepo4graphite": "lfpgraphite",
                      "ncm532": "ncm532", "nmc532": "ncm532"},
        "manufacturer": {"a123systems": "a123", "a123systemsllc": "a123"},
        "form_factor": {"cylindrical18650": "18650", "18650cylindrical": "18650"},
    }
    return aliases.get(field, {}).get(compact, compact)


def compare_specifications(left, right, required=None):
    """Unknown never means equal. Numeric tolerance is for rounded ratings only."""
    required = tuple(required or TEXT_FIELDS + NUMERIC_FIELDS)
    conflicts, missing = [], []
    try:
        # Validate even fields not explicitly required by a caller.
        normalized = [{f: canonical(f, side.get(f)) for f in TEXT_FIELDS + NUMERIC_FIELDS}
                      for side in (left, right)]
        for side in normalized:
            lo, hi = side["rated_voltage_min_v"], side["rated_voltage_max_v"]
            if lo is not None and hi is not None and lo >= hi:
                raise ValueError("invalid rated voltage interval")
        for field in TEXT_FIELDS + NUMERIC_FIELDS:
            a, b = normalized[0][field], normalized[1][field]
            if a is None or b is None:
                if field in required:
                    missing.append(field)
                continue
            if field in TEXT_FIELDS:
                equal = a == b
            else:
                tolerance = max(.01, .03 * min(a, b)) if field == "nominal_capacity_ah" else .05
                equal = math.isclose(a, b, rel_tol=0, abs_tol=tolerance + 1e-12)
            if not equal:
                conflicts.append(field)
    except (ValueError, TypeError) as error:
        return {"status": "invalid_specs", "conflicts": [], "missing": [], "error": str(error)}
    return {"status": "incompatible" if conflicts else "needs_specs" if missing else "compatible",
            "conflicts": conflicts, "missing": missing}


def match_external_specifications(specifications, registry=None):
    """Dataset names, profile IDs and a similar V/I/T curve cannot bypass this gate."""
    registry = registry or load_registry()
    candidates = {key: compare_specifications(specifications, row["specifications"],
                  registry["external_required_fields"])
                  for key, row in registry["families"].items()}
    matched = [key for key, row in candidates.items() if row["status"] == "compatible"]
    if any(row["status"] == "invalid_specs" for row in candidates.values()):
        status = "invalid_specs"
    elif len(matched) == 1:
        status = "matched"
    elif len(matched) > 1:
        status = "ambiguous_specs"
    elif any(row["status"] == "needs_specs" for row in candidates.values()):
        status = "needs_specs"
    else:
        status = "unseen_type"
    return {"status": status, "physical_family": matched[0] if status == "matched" else None,
            "candidates": candidates}


def family_for_registered_record(record, registry=None):
    """Internal source registration only; caller must verify its manifest/data hash.

    This is not a classifier for external batteries. The NASA group is explicitly
    unverified, and the XJTU group does not pretend to know its part number.
    """
    registry = registry or load_registry()
    metadata = record["metadata"]
    matches = [(key, row) for key, row in registry["families"].items()
               if metadata["dataset"] in row["datasets"]]
    if len(matches) != 1:
        raise ValueError("source has no unique registered physical family")
    key, row = matches[0]
    result = compare_specifications(metadata, row["specifications"])
    if result["status"] in ("incompatible", "invalid_specs"):
        raise ValueError(f"registered source contradicts physical specifications: {result}")
    if metadata["dataset"] == "NASA" and record["cell_id"] not in ("B0005", "B0006", "B0007", "B0018"):
        raise ValueError("NASA cell outside the audited isolated cohort")
    return key


def require_same_registered_family(query_family, training_families, registry=None):
    registry = registry or load_registry()
    if query_family not in registry["families"]:
        raise ValueError("unknown registered physical family")
    if not training_families or any(family != query_family for family in training_families):
        raise ValueError("incompatible physical-family routing; prediction rejected")
    return registry["families"][query_family]["status"]
