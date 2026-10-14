"""One actual pretrained-Llama gradient step on training cells, not accuracy."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import joblib
import numpy as np
import torch

from battery_smetimes.physical_types import load_registry, require_same_registered_family
from battery_smetimes.research_llm import FrozenSMETimesForecaster, statistics_cards
from battery_smetimes.research_sequences import HistoryPreprocessor, pack_contexts, training_contexts
from scripts.prepare_same_type_forecast import save_json, sha
from scripts.train_same_type_forecast import load_data
from smoke_original_smetimes_battery_v2 import CHECKPOINT, checkpoint_sha256


def main():
    folder = Path("experiments/soh_research_v5/llm_smokes") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    folder.mkdir(parents=True, exist_ok=False)
    audit_folder = Path(json.loads(Path("experiments/soh_research_v5/latest_audit.json").read_text())["folder"])
    audit = json.loads((audit_folder / "audit.json").read_text())
    registered = json.loads((audit_folder / "cell_registry.json").read_text())["cells"]
    registry = load_registry(audit_folder / "source_snapshot/battery_physical_registry.json")
    frozen = json.loads((audit_folder / "config.json").read_text())
    manifest, cells = load_data(Path(audit["prepared_folder"]), frozen["config"]["integrity_audit"])
    for key, cell in cells.items():
        cell.update({field: registered[key][field] for field in ("physical_family", "reference_established_source_cycle")})
    plan = next(p for p in json.loads((audit_folder / "splits.json").read_text())["plans"] if p["name"] == "XJTU_fold1")
    ids = plan["train"][:2]
    family = registered[ids[0]]["physical_family"]
    require_same_registered_family(family, [registered[k]["physical_family"] for k in ids], registry)
    save_json(folder / "config.json", {"status": "gradient smoke, NOT an accuracy experiment", "train_ids": ids,
        "plan": plan, "seed": 20260930, "checkpoint": str(CHECKPOINT), "checkpoint_sha256": checkpoint_sha256(),
        "physical_family": family, "experts": 4, "low_rank": 64, "patch_cycles": 5, "balance_coefficient": .01,
        "dtype": "bfloat16", "torch": torch.__version__,
        "source_sha256": {str(p): sha(p) for p in [Path(__file__), Path("battery_smetimes/research_llm.py"),
            Path("battery_smetimes/research_sequences.py"), Path("battery_smetimes/research_baselines.py")]}})
    try:
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.cuda.reset_peak_memory_stats()
        processor = HistoryPreprocessor().fit(cells, ids, [20, 50, 100])
        joblib.dump(processor, folder / "preprocessor.joblib")
        contexts = [r for r in training_contexts(cells, ids, processor) if r["origin"] == 100]
        batch = pack_contexts(contexts, "cuda")
        card_groups = [statistics_cards(cells[r["cell_id"]], r["origin"]) for r in contexts]
        save_json(folder / "statistics_cards.json", {r["cell_id"]: card_groups[j] for j, r in enumerate(contexts)})
        model = FrozenSMETimesForecaster(CHECKPOINT, 2 * processor.raw_width + 1, device="cuda")
        if any(p.requires_grad for p in model.backbone.parameters()):
            raise ValueError("language backbone not frozen")
        embeddings = model.encode_cards(card_groups)
        model.train()
        encoding = model.encode(batch["history"], batch["length"], embeddings)
        prediction = model.decode(encoding, batch["conditions"], batch["query"])
        mse = (((prediction - batch["truth"]) ** 2) * batch["valid"]).sum() / batch["valid"].sum()
        loss = mse + .01 * model.balance_loss
        before = model.numeric_projection[0].weight.detach().cpu().clone()
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = model.numeric_projection[0].weight.grad
        if gradient is None or not torch.isfinite(gradient).all() or gradient.norm() == 0:
            raise ValueError("no finite nonzero input-projection gradient through frozen LLM")
        if any(p.grad is not None for p in model.backbone.parameters()):
            raise ValueError("unexpected frozen-backbone gradient")
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.)
        optimizer.step()
        if torch.equal(before, model.numeric_projection[0].weight.detach().cpu()):
            raise ValueError("numeric adapter did not update")
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items() if not k.startswith("backbone.")},
                   folder / "adapter_heads.pt")
        report = {"status": "passed actual pretrained-Llama SOH-head gradient smoke; no held-out accuracy claim",
            "training_cells": ids, "training_contexts": len(contexts), "statistics_cards": sum(map(len, card_groups)),
            "parameter_counts": model.parameter_counts(), "loss": float(loss.detach()), "mse": float(mse.detach()),
            "balance_loss": float(model.balance_loss.detach()), "numeric_gradient_norm": float(gradient.norm()),
            "backbone_frozen": True, "decoder": "deterministic pointwise SOH regression, not text sampling",
            "cuda_peak_allocated_mb": torch.cuda.max_memory_allocated() / 2 ** 20}
        save_json(folder / "report.json", report)
        shutil.copy2(__file__, folder / "smoke_source.py")
        shutil.copy2("battery_smetimes/research_llm.py", folder / "llm_source.py")
        save_json(Path("experiments/soh_research_v5/latest_llm_smoke.json"), {"folder": str(folder.resolve())})
        print(json.dumps(report, indent=2), flush=True)
    except BaseException:
        save_json(folder / "failure.json", {"traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
