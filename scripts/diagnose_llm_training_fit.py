"""Evaluate the actual saved LLM on its TRAIN cells to distinguish underfit.

This is in-sample diagnosis, never a generalization score or model-selection
gate. No free per-cell curve parameters, new teacher or model fitting is used.
"""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import torch
from battery_smetimes.research_llm import FrozenSMETimesForecaster, load_adapter
from battery_smetimes.physical_types import load_registry
from scripts.train_llm_soh_mvp import evaluate
from scripts.train_same_type_forecast import load_data
from scripts.prepare_same_type_forecast import save_json, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    root = Path(args.run)
    cfg = json.loads((root/'config.json').read_text())
    audit = Path(cfg['frozen_plan'])
    frozen = json.loads((audit/'config.json').read_text())
    prepared = json.loads((audit/'audit.json').read_text())['prepared_folder']
    _, cells = load_data(Path(prepared), frozen['config']['integrity_audit'])
    registry = json.loads((audit/'cell_registry.json').read_text())['cells']
    for cid, cell in cells.items():
        cell.update({k: registry[cid][k] for k in ('physical_family', 'reference_established_source_cycle')})
    torch.set_num_threads(2)
    for candidate in cfg['candidates']:
        dest = root/candidate['name']
        processor = joblib.load(dest/'preprocessor.joblib')
        if processor.train_ids != cfg['plan']['train']:
            raise ValueError('TRAIN identity drift')
        network = FrozenSMETimesForecaster(cfg['checkpoint'], 2*processor.raw_width+1,
            seed=cfg['seed'], dropout=cfg['dropout'], device='cuda',
            normalize_router=candidate.get('normalize_router', False),
            output_head=candidate.get('output_head', 'point'), pooling=candidate.get('pooling', 'mean'),
            numeric_dynamics=candidate.get('numeric_dynamics', False),
            adaptive_fusion=candidate.get('adaptive_fusion', False),
            summary_width=processor.stats_width if candidate.get('summary_encoding', False) else 0,
            residual_experts=candidate.get('residual_experts', False), history_slots=getattr(processor,'history_slots',100),
            numeric_lora_rank=candidate.get('numeric_lora_rank',0))
        load_adapter(network, torch.load(dest/'model.pt', map_location='cpu', weights_only=True))
        bundle = {'network': network, 'processor': processor, 'device': 'cuda', 'origins': [cfg['origin']],
                  'registry': load_registry(audit/'source_snapshot/battery_physical_registry.json'),
                  'physical_family': cells[processor.train_ids[0]]['physical_family']}
        report, arrays, metrics, _ = evaluate(bundle, cells, processor.train_ids)
        values = list(metrics.values())
        summary = {'cells': len(values), 'joint_passes': sum(v.get('joint_pass', False) for v in values),
                   **{k: float(np.mean([v[k] for v in values])) for k in
                      ('mape_percent', 'mae_pp', 'tail_mae_pp', 'stage_drop_max_pp', 'start_error_pp')}}
        save_json(dest/'TRAIN_fit_diagnostic.json', {
            'scope': 'In-sample actual saved LLM predictions; NOT heldout accuracy or selection',
            'adapter_sha256': sha(dest/'model.pt'), 'summary': summary, 'by_cell': metrics})
        np.savez_compressed(dest/'TRAIN_fit_diagnostic_predictions.npz', **arrays)
        print(candidate['name'], summary, flush=True)
        del network, bundle
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
