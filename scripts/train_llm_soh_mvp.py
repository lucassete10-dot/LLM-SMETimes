"""LLM-only MVP: train E1 and one curve-supervision change on frozen MIT cells."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import copy
import json
from pathlib import Path
import shutil
import time
import traceback
from datetime import datetime, timezone
import joblib
import numpy as np
import torch
from battery_smetimes.research_llm import FrozenSMETimesForecaster, attach_cards, adapter_state, load_adapter, predict_llm, adapter_optimizer
from battery_smetimes.research_sequences import HistoryPreprocessor, training_contexts, pack_contexts
from battery_smetimes.research_baselines import evaluate_baseline
from battery_smetimes.llm_curve_objective import curve_loss, curve_metrics, training_origin_target
from battery_smetimes.llm_reporting import raw_curve_numeric_metrics, finite_metric
from battery_smetimes.llm_knee_supervision import training_knee_target, knee_interval_loss, KNEE_RULES
from battery_smetimes.llm_acceleration_supervision import sustained_acceleration_loss, ACCELERATION_RULES
from battery_smetimes.physical_types import load_registry
from scripts.train_same_type_forecast import load_data
from scripts.prepare_same_type_forecast import sha, save_json


def evaluation_curve_metrics(curve):
    try:metric=curve_metrics(*curve)
    except ValueError:metric={"evaluable":False,"reason":"insufficient curve coverage"}
    if 'mae_pp' not in metric:metric.update(raw_curve_numeric_metrics(*curve),joint_pass=False)
    return metric


def evaluate(bundle, cells, ids):
    report, arrays = evaluate_baseline(bundle, cells, ids, predictor=predict_llm)
    metrics = {}
    for row in report["by_cell"]:
        prefix = f'{row["cell_id"]}__{row["origin"]}__'
        curve=tuple(arrays[prefix+k] for k in ("cycle", "truth", "prediction"))
        metrics[row["cell_id"]]=evaluation_curve_metrics(curve)
    if not metrics:
        raise ValueError("no validation curves")
    values = list(metrics.values())
    score = (-np.mean([r.get("joint_pass", False) for r in values]),
             np.mean([sum(r[k] for k in ("mae_pp","tail_mae_pp","stage_drop_max_pp") if finite_metric(r,k)) for r in values]))
    return report, arrays, metrics, tuple(map(float, score))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/llm_soh_mvp.json")
    args = parser.parse_args()
    settings = json.loads(Path(args.config).read_text())
    settings['history_slots'] = settings.get('history_slots', max(100, 5*int(np.ceil(settings['origin']/5))))
    audit_dir = Path(settings["frozen_plan"])
    frozen = json.loads((audit_dir/"config.json").read_text())
    audit = json.loads((audit_dir/"audit.json").read_text())
    registry_path = audit_dir/"source_snapshot/battery_physical_registry.json"
    if sha(registry_path) != frozen["registry_sha256"]:
        raise ValueError("registry drift")
    data = Path(audit["prepared_folder"])
    if sha(data/"manifest.json") != frozen["prepared_manifest_sha256"]:
        raise ValueError("prepared manifest drift")
    if sha(frozen["config"]["integrity_audit"]) != frozen["integrity_sha256"]:
        raise ValueError("integrity rules drift")
    manifest, cells = load_data(data, frozen["config"]["integrity_audit"])
    registered = json.loads((audit_dir/"cell_registry.json").read_text())["cells"]
    for key, cell in cells.items():
        cell.update({field: registered[key][field] for field in ("physical_family", "reference_established_source_cycle")})
    plan = next(p for p in json.loads((audit_dir/"splits.json").read_text())["plans"] if p["name"] == settings["fold"])
    for a,b in (("train","validation"),("train","test"),("validation","test")):
        if set(plan[a]) & set(plan[b]):
            raise ValueError("cell leakage")
    family = cells[plan["train"][0]]["physical_family"]
    if any(cells[k]["physical_family"] != family for role in ("train","validation","test") for k in plan[role]):
        raise ValueError("mixed physical families")
    folder = Path("experiments/llm_soh_mvp")/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    folder.mkdir(parents=True)
    save_json(folder/"config.json", {**settings,"plan":plan,"status":"running"})
    sources = [Path(__file__),Path(args.config),Path("battery_smetimes/research_llm.py"),
               Path("battery_smetimes/research_sequences.py"),Path("battery_smetimes/llm_curve_objective.py"),
               Path("battery_smetimes/llm_knee_supervision.py"),
               Path("battery_smetimes/llm_acceleration_supervision.py"),
               Path("battery_smetimes/research_baselines.py"),Path("battery_smetimes/forecast.py"),
               Path("scripts/train_same_type_forecast.py")]
    for source in sources:
        relative = source.resolve().relative_to(Path.cwd().resolve())
        target = folder/"source_snapshot"/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
    save_json(folder/"provenance.json", {"sources":{str(p):sha(p) for p in sources},
        "data_manifest_sha256":sha(data/"manifest.json"),"prepared_cells_sha256":{r["cell_id"]:r["sha256"] for r in manifest["cells"]},
        "checkpoint_sha256":{p.name:sha(p) for p in Path(settings["checkpoint"]).iterdir() if p.suffix in (".json",".safetensors")}})
    save_json(Path("experiments/llm_soh_mvp/latest.json"),{"folder":str(folder.resolve())})
    print(f"RUN {folder.resolve()}",flush=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    processor = HistoryPreprocessor().fit(cells,plan["train"],[settings["origin"]], history_slots=settings['history_slots'])
    contexts = training_contexts(cells,plan["train"],processor,settings["future_targets"])
    if any(c.get('acceleration_weight',0) for c in settings['candidates']):
        save_json(folder/'acceleration_supervision_rules.json', {
            'scope': 'TRAIN curve supervision only; no parameter centre label',
            'rules': ACCELERATION_RULES, 'train_cells': plan['train']})
    if any(c.get('knee_weight',0) for c in settings['candidates']):
        if any(c.get('knee_weight',0) and c.get('output_head') != 'integrated_rate' for c in settings['candidates']):
            raise ValueError('knee supervision requires integrated-rate head')
        for row in contexts:
            assert row['cell_id'] in plan['train']
            row['knee_target'] = training_knee_target(cells[row['cell_id']], row['origin'])
        save_json(folder/'knee_supervision_coverage.json', {
            'scope': 'TRAIN-only auxiliary labels; never inferred from validation or heldout labels',
            'rules': KNEE_RULES, 'available': sum(r['knee_target']['valid'] for r in contexts),
            'contexts': len(contexts),
            'by_cell': {r['cell_id']: r['knee_target'] for r in contexts}})
    if any(c.get("start_weight",0) for c in settings["candidates"]):
        for row in contexts:
            assert row['cell_id'] in plan['train']
            row['start_truth'],row['start_valid']=training_origin_target(cells[row['cell_id']],row['origin'])
            # Auxiliary TRAIN query at zero horizon; public future_query keeps
            # its strict >origin contract for actual forecasting requests.
            row['start_query']=processor.query_scale.transform(np.array([[row['origin']/100,0.,0.]]))
        save_json(folder/'start_supervision_coverage.json',{
            'scope':'training contexts only; exact-origin labels, no future interpolation',
            'contexts':len(contexts),'available':sum(int(r['start_valid']) for r in contexts),
            'train_cell_ids':sorted({r['cell_id'] for r in contexts if r['start_valid']})})
    for row in contexts:
        row["source_cycles"] = np.zeros(settings["future_targets"],dtype=np.float32)
        count = int(row["valid"].sum())
        # Recover cycle coordinates from the training-fitted query scaler.
        inverse = processor.query_scale.scaler.inverse_transform(row["query"][:count])
        row["source_cycles"][:count] = inverse[:,0]*100
    results=[]
    try:
        for candidate in settings["candidates"]:
            dest=folder/candidate["name"]
            dest.mkdir()
            save_json(dest/"config.json",{**settings,"candidate":candidate,"plan":plan})
            started=time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            network=FrozenSMETimesForecaster(settings["checkpoint"],2*processor.raw_width+1,
                seed=settings["seed"],dropout=settings["dropout"],device="cuda",
                normalize_router=candidate.get("normalize_router",False),
                output_head=candidate.get("output_head","point"),
                pooling=candidate.get("pooling","mean"),numeric_dynamics=candidate.get("numeric_dynamics",False),
                adaptive_fusion=candidate.get("adaptive_fusion",False),
                summary_width=processor.stats_width if candidate.get('summary_encoding',False) else 0,
                residual_experts=candidate.get('residual_experts',False), history_slots=processor.history_slots,
                initial_knee_bias=candidate.get('initial_knee_bias',1.), numeric_lora_rank=candidate.get('numeric_lora_rank',0))
            network.bind_query_scaler(processor.query_scale.scaler)
            bundle={"network":network,"processor":processor,"physical_family":family,
                "registry":load_registry(registry_path),"device":"cuda","origins":[settings["origin"]]}
            attach_cards(bundle,contexts,cells)
            optimizer=adapter_optimizer(network,settings['learning_rate'],settings['weight_decay'],
                candidate.get('numeric_lora_lr_scale',1.))
            save_json(dest/'optimizer_groups.json',[
                {'learning_rate':group['lr'],'weight_decay':group['weight_decay'],
                 'parameters':sum(p.numel() for p in group['params'])} for group in optimizer.param_groups])
            history=[]; best=None; best_state=None; bad=0; best_epoch=0
            for epoch in range(1,settings["max_epochs"]+1):
                network.train(); losses=[]; components=[]
                order=np.random.default_rng(settings["seed"]+epoch).permutation(len(contexts))
                for start in range(0,len(order),settings["batch_contexts"]):
                    rows=[contexts[i] for i in order[start:start+settings["batch_contexts"]]]
                    batch=pack_contexts(rows,"cuda")
                    emb=torch.stack([r["card_embeddings"] for r in rows]).to("cuda")
                    encoding=network.encode(batch["history"],batch["length"],emb,
                        batch['statistics'] if network.summary_width else None)
                    if candidate.get('knee_weight',0):
                        prediction,curve_parameters=network.decode(encoding,batch['conditions'],batch['query'],return_parameters=True)
                    else:
                        prediction=network.decode(encoding,batch["conditions"],batch["query"])
                    cycles=torch.as_tensor(np.array([r["source_cycles"] for r in rows]),device="cuda")
                    individual, terms=curve_loss(prediction,batch["truth"],cycles,batch["valid"],candidate["stage_weight"],candidate["tail_weight"],candidate.get('rate_weight',0.))
                    if candidate.get('acceleration_weight',0):
                        auxiliary,coverage = sustained_acceleration_loss(prediction,batch['truth'],cycles,batch['valid'])
                        individual = individual + candidate['acceleration_weight'] * auxiliary
                        terms['acceleration_aux'] = auxiliary
                        terms['acceleration_windows'] = torch.as_tensor(coverage,device='cuda',dtype=torch.float32)
                    if candidate.get('knee_weight',0):
                        target = [r['knee_target'] for r in rows]
                        lower,upper,mask = [torch.as_tensor([t[k] for t in target],device='cuda',dtype=torch.float32)
                                            for k in ('lower','upper','valid')]
                        auxiliary = knee_interval_loss(curve_parameters,lower,upper,mask)
                        individual = individual + candidate['knee_weight'] * auxiliary
                        terms['knee_aux'] = auxiliary
                    if candidate.get('start_weight',0):
                        start_queries=torch.as_tensor(np.array([r['start_query'] for r in rows]),device='cuda')
                        start_prediction=network.decode(encoding,batch['conditions'],start_queries)[:,0]
                        start_truth=torch.as_tensor([r['start_truth'] for r in rows],device='cuda')
                        start_valid=torch.as_tensor([r['start_valid'] for r in rows],device='cuda')
                        auxiliary=(start_prediction-start_truth).square()*start_valid
                        individual=individual+candidate['start_weight']*auxiliary
                        terms['start_aux']=auxiliary
                    loss=(individual*batch["weight"]).mean()+settings["balance_coefficient"]*network.balance_loss
                    terms['balance_weighted']=settings['balance_coefficient']*network.balance_loss.expand_as(individual)
                    if not torch.isfinite(loss): raise ValueError("nonfinite loss")
                    optimizer.zero_grad(set_to_none=True); loss.backward()
                    if epoch==1 and start==0:
                        gradients={k:float(p.grad.norm()) for k,p in network.named_parameters() if p.requires_grad and p.grad is not None}
                        if any(not np.isfinite(v) for v in gradients.values()) or gradients.get("numeric_projection.0.weight",0)<=0 or any(p.grad is not None for p in network.backbone.parameters() if not p.requires_grad):
                            raise ValueError("LLM gradient path failed")
                        if candidate.get('numeric_lora_rank',0) and not any(v>0 for k,v in gradients.items() if k.endswith('.lora_B')):
                            raise ValueError('numeric LoRA gradient path failed')
                        if candidate.get('numeric_dynamics',False) and gradients.get('dynamics_projection.weight',0)<=0:
                            raise ValueError('dynamics adapter gradient path failed')
                        if candidate.get('adaptive_fusion',False) and gradients.get('fusion_gate.weight',0)<=0:
                            raise ValueError('adaptive fusion gradient path failed')
                        if candidate.get('pooling')=='dual_scale' and gradients.get('temporal_contrast.2.weight',0)<=0:
                            raise ValueError('temporal contrast gradient path failed')
                        if candidate.get('summary_encoding',False) and gradients.get('summary_projection.weight',0)<=0:
                            raise ValueError('historical summary gradient path failed')
                        save_json(dest/"gradient_check.json",gradients)
                    torch.nn.utils.clip_grad_norm_((p for p in network.parameters() if p.requires_grad),1.)
                    optimizer.step(); losses.append(float(loss.detach()))
                    components.append({k:float(v.detach().mean()) for k,v in terms.items()})
                record={"epoch":epoch,"train_objective":float(np.mean(losses)),
                    "terms":{k:float(np.mean([v[k] for v in components])) for k in components[0]}}
                if epoch%settings["validation_every"]==0 or epoch==settings["max_epochs"]:
                    report,arrays,metrics,score=evaluate(bundle,cells,plan["validation"])
                    record["validation_score"]=score
                    if best is None or score<best:
                        best=score; best_state=adapter_state(network); best_epoch=epoch; bad=0
                        torch.save(best_state,dest/"model.pt")
                        save_json(dest/"validation_metrics.json",metrics)
                        np.savez_compressed(dest/"validation_predictions.npz",**arrays)
                    else: bad+=1
                history.append(record); save_json(dest/"training_history.json",history)
                print(candidate["name"],record,flush=True)
                if bad>=settings["patience"]: break
            load_adapter(network,best_state)
            joblib.dump(processor,dest/"preprocessor.joblib")
            report,arrays,metrics,score=evaluate(bundle,cells,plan["test"])
            save_json(dest/"heldout_metrics.json",metrics);save_json(dest/"heldout_report.json",report)
            np.savez_compressed(dest/"heldout_predictions.npz",**arrays)
            result={"name":candidate["name"],"selected_epoch":best_epoch,"validation_score":best,
                "elapsed_seconds":time.perf_counter()-started,"parameters":network.parameter_counts(),
                "cuda_peak_allocated_mb":torch.cuda.max_memory_allocated()/2**20}
            results.append(result);save_json(dest/"result.json",result);save_json(folder/"results.json",results)
            del network,optimizer,bundle,best_state
            torch.cuda.empty_cache()
        save_json(folder/"status.json",{"training_complete":True,"delivery_complete":False,"results":results})
    except BaseException:
        save_json(folder/"failure.json",{"traceback":traceback.format_exc(),"completed":results})
        raise


if __name__=="__main__": main()
