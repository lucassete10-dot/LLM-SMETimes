"""Reload trusted local v5 artifacts without fitting or reading target labels."""

import json
from pathlib import Path

import joblib

from battery_smetimes.physical_types import require_same_registered_family
from scripts.prepare_same_type_forecast import sha


def run_folder(model_folder):
    folder = Path(model_folder).resolve()
    if len(folder.parents) < 3:
        raise ValueError("not a v5 model artifact path")
    run = folder.parents[2]
    if not (run / "frozen_audit/splits.json").is_file():
        raise ValueError("model has no frozen whole-cell audit")
    return run


def verify_source_snapshot(run):
    config = json.loads((run / "config.json").read_text())
    for original, expected in config["source_sha256"].items():
        path = Path(original)
        snapshot = run / "source_snapshot" / (path.name if path.is_absolute() else path)
        if not snapshot.is_file() or sha(snapshot) != expected:
            raise ValueError(f"source snapshot missing or changed: {original}")
    return config


def load_model(model_folder, device="cpu", *, evaluation_protocol="main"):
    """Only load trusted local joblib/torch artifacts, not arbitrary uploads."""
    folder = Path(model_folder).resolve()
    run = run_folder(folder)
    config = verify_source_snapshot(run)
    recorded = json.loads((folder / "config.json").read_text())
    plan = recorded["plan"]
    condition_evidence = None
    if evaluation_protocol == "condition":
        from battery_smetimes.research_conditions_execution import verify_condition_model_contract
        condition_evidence = verify_condition_model_contract(folder, config, recorded)
        if sha(Path(__file__)) != config["execution_source_sha256"]["battery_smetimes/research_inference.py"]:
            raise ValueError("condition model loader differs from frozen execution source")
    elif evaluation_protocol == "main":
        if config.get("not_main_grouped_cv"):
            raise ValueError("condition model requires explicit condition evaluation protocol")
        frozen_plans = json.loads((run / "frozen_audit/splits.json").read_text())["plans"]
        if plan != next(p for p in frozen_plans if p["name"] == plan["name"]):
            raise ValueError("model split differs from the frozen plan")
    else:
        raise ValueError("unknown model evaluation protocol")
    if (folder / "model.joblib").is_file():
        bundle = joblib.load(folder / "model.joblib")
        from battery_smetimes.research_baselines import predict_baseline
        predictor = predict_baseline
        artifacts = [folder / "model.joblib"]
    else:
        import torch
        from battery_smetimes.research_sequences import SmallForecaster, predict_sequence
        info = json.loads((folder / "model.json").read_text())
        processor = joblib.load(folder / "preprocessor.joblib")
        if (folder / "llm.json").is_file():
            from battery_smetimes.research_llm import FrozenSMETimesForecaster, load_adapter, predict_llm
            llm = json.loads((folder / "llm.json").read_text())
            checkpoint = Path(llm["checkpoint"])
            for name, expected in llm["checkpoint_files_sha256"].items():
                if sha(checkpoint / name) != expected:
                    raise ValueError(f"frozen LLM checkpoint changed: {name}")
            if config["torch"] != torch.__version__:
                raise ValueError("torch version differs; random-backbone replay is not certified")
            parameters = config["sequence_config"]
            network = FrozenSMETimesForecaster(checkpoint, info["history_width"], seed=info["seed"],
                device=device, dropout=info["parameters"]["dropout"], patch_size=parameters["patch_size"],
                max_text_tokens=parameters["max_text_tokens"], **llm["variant"])
            load_adapter(network, torch.load(folder / "model.pt", map_location="cpu", weights_only=True))
            predictor = predict_llm
        else:
            torch.manual_seed(info["seed"])
            network = SmallForecaster(info["architecture"], info["history_width"], info["statistics_width"],
                hidden=info["parameters"]["hidden"], dropout=info["parameters"]["dropout"]).to(device)
            network.load_state_dict(torch.load(folder / "model.pt", map_location="cpu", weights_only=True))
            predictor = predict_sequence
        network.eval()
        bundle = {"network": network, "processor": processor, "physical_family": info["physical_family"],
            "registry": info["registry"], "device": device, "model_name": info["architecture"],
            "train_ids": info["train_ids"], "origins": info["origins"], "seed": info["seed"]}
        artifacts = [folder / name for name in ("model.pt", "model.json", "preprocessor.joblib")]
    if set(bundle["train_ids"]) != set(plan["train"] + plan["validation"]):
        raise ValueError("saved final model train IDs do not match refit partition")
    if len(bundle["train_ids"]) != len(plan["train"] + plan["validation"]):
        raise ValueError("duplicate saved training IDs")
    if set(bundle["train_ids"]) & set(plan["test"]):
        raise ValueError("test battery entered saved training model")
    if bundle["physical_family"] != recorded["physical_family"]:
        raise ValueError("saved model physical family changed")
    if bundle["seed"] != recorded["seed"] or bundle["model_name"] != recorded["model"]:
        raise ValueError("saved model identity differs from its experiment configuration")
    frozen_configuration = json.loads((run / "frozen_audit/config.json").read_text())
    registry_path = run / "frozen_audit/source_snapshot/battery_physical_registry.json"
    if sha(registry_path) != frozen_configuration["registry_sha256"]:
        raise ValueError("frozen physical registry fingerprint changed")
    frozen_registry = json.loads(registry_path.read_text())
    if bundle["registry"] != frozen_registry:
        raise ValueError("saved model registry differs from frozen specifications")
    registered = json.loads((run / "frozen_audit/cell_registry.json").read_text())["cells"]
    require_same_registered_family(bundle["physical_family"],
        [registered[key]["physical_family"] for key in bundle["train_ids"]], frozen_registry)
    if "processor" in bundle and bundle["processor"].train_ids != bundle["train_ids"]:
        raise ValueError("saved scaler training IDs disagree with the model")
    if bundle["origins"] != config["config"]["origins_source_cycles"]:
        raise ValueError("saved origins differ from frozen observation budget")
    return bundle, predictor, {"run": str(run), "plan": plan,
        "artifact_sha256": {path.name: sha(path) for path in artifacts},
        "source_snapshot_verified": True, "no_fit": True,
        "evaluation_protocol": evaluation_protocol, "condition_contract": condition_evidence}
