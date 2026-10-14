"""Real pretrained-LLM capacity check, not a longer-N forecast experiment."""
import argparse
import copy
import json
from pathlib import Path
import joblib
import numpy as np
import torch
from battery_smetimes.research_llm import FrozenSMETimesForecaster, load_adapter, predict_llm, cards_embedding
from battery_smetimes.physical_types import load_registry
from scripts.train_same_type_forecast import load_data
from scripts.prepare_same_type_forecast import save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--slots', type=int, required=True)
    parser.add_argument('--prepared', type=Path, help='Immutable signal extension; labels and old prefix must match')
    args = parser.parse_args()
    root = Path(args.run)
    cfg = json.loads((root/'config.json').read_text())
    audit = Path(cfg['frozen_plan'])
    frozen = json.loads((audit/'config.json').read_text())
    prepared = json.loads((audit/'audit.json').read_text())['prepared_folder']
    _, cells = load_data(Path(prepared), frozen['config']['integrity_audit'])
    if args.prepared:
        _, extended = load_data(args.prepared, frozen['config']['integrity_audit'])
        for cid, cell in cells.items():
            other = extended[cid]
            np.testing.assert_array_equal(cell['label_cycle'], other['label_cycle'])
            np.testing.assert_array_equal(cell['soh'], other['soh'])
            take = other['cycle'] <= 100
            np.testing.assert_array_equal(cell['cycle'], other['cycle'][take])
            np.testing.assert_array_equal(cell['raw'], other['raw'][take])
        cells = extended
    registry = json.loads((audit/'cell_registry.json').read_text())['cells']
    for cid, cell in cells.items():
        cell.update({k: registry[cid][k] for k in ('physical_family', 'reference_established_source_cycle')})
    torch.set_num_threads(2)
    results = []
    for candidate in cfg['candidates']:
        dest = root/candidate['name']
        processor = joblib.load(dest/'preprocessor.joblib')
        processor.history_slots = args.slots
        cid = cfg['plan']['train'][0]  # predefined TRAIN identity, no result sorting
        cell = cells[cid]
        network = FrozenSMETimesForecaster(cfg['checkpoint'], 2*processor.raw_width+1,
            seed=cfg['seed'], dropout=cfg['dropout'], device='cuda', history_slots=args.slots,
            normalize_router=candidate.get('normalize_router', False),
            output_head=candidate.get('output_head', 'point'), pooling=candidate.get('pooling', 'mean'),
            numeric_dynamics=candidate.get('numeric_dynamics', False), adaptive_fusion=candidate.get('adaptive_fusion', False),
            summary_width=processor.stats_width if candidate.get('summary_encoding', False) else 0,
            residual_experts=candidate.get('residual_experts', False),
            numeric_lora_rank=candidate.get('numeric_lora_rank',0))
        load_adapter(network, torch.load(dest/'model.pt', map_location='cpu', weights_only=True))
        bundle = {'network': network, 'processor': processor, 'device': 'cuda',
                  'registry': load_registry(audit/'source_snapshot/battery_physical_registry.json'),
                  'physical_family': cell['physical_family']}
        with np.load(dest/'TRAIN_fit_diagnostic_predictions.npz') as saved:
            prefix = f'{cid}__{cfg["origin"]}__'
            x, expected = saved[prefix+'cycle'], saved[prefix+'prediction']
            positions = [0, len(x)//2, len(x)-1]
            actual = predict_llm(bundle, cell, cfg['origin'], x[positions])
            deviation = float(np.max(abs(actual-expected[positions])))
        # Longer visible prefix is a runtime/gradient diagnostic using the old
        # scaler only. It is explicitly NOT evidence for longer-N accuracy.
        origin = args.slots
        future = origin+np.array([1., 50.])
        before = predict_llm(bundle, cell, origin, future)
        poisoned = copy.deepcopy(cell)
        poisoned['raw'][poisoned['cycle'] > origin] = np.inf
        del poisoned['soh'], poisoned['label_cycle']
        after = predict_llm(bundle, poisoned, origin, future)
        poison_delta = float(np.max(abs(before-after)))
        values = processor.model_input(poisoned, origin)
        network.train()
        encoding = network.encode(torch.as_tensor(values['history'][None], device='cuda'),
            torch.as_tensor([values['length']], device='cuda'), cards_embedding(bundle, poisoned, origin)[None].to('cuda'),
            torch.as_tensor(values['statistics'][None], device='cuda') if network.summary_width else None)
        output = network.decode(encoding, torch.as_tensor(values['conditions'][None], device='cuda'),
                                torch.as_tensor(processor.future_query(origin, future)[None], device='cuda'))
        output.square().mean().backward()
        gradient = float(network.numeric_projection[0].weight.grad.norm())
        passed = deviation <= 1e-5 and poison_delta <= 1e-5 and np.isfinite(gradient) and gradient > 0
        results.append({'candidate': candidate['name'], 'TRAIN_cell': cid, 'capacity': args.slots,
                        'longer_prefix_observations': values['length'], 'padding_reload_max_soh_error': deviation,
                        'future_poison_max_soh_error': poison_delta, 'numeric_projection_gradient_norm': gradient,
                        'backbone_parameters_remain_frozen': all(p.grad is None for p in network.backbone.parameters()),
                        'scope': 'Runtime, padding equivalence and gradients only; old scaler, no longer-N accuracy claim',
                        'passed': bool(passed)})
        print(results[-1], flush=True)
        del network, bundle
        torch.cuda.empty_cache()
    suffix = '_extended_trimmed' if args.prepared else ''
    save_json(root/f'history_capacity_{args.slots}{suffix}_verification.json', results)
    if not all(r['passed'] and r['backbone_parameters_remain_frozen'] for r in results):
        raise ValueError('history capacity verification failed')


if __name__ == '__main__':
    main()
