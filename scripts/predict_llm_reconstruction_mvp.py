"""Frozen reconstruction models on five preselected validation cells.

Two distinct tasks: full-signal SOH estimation and explicitly untrained
prefix-only head extrapolation. No fitting, cell reselection or postprocessing.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
from pathlib import Path
import gc
import shutil
import joblib
import numpy as np
import torch
from battery_smetimes.llm_full_life import read
from battery_smetimes.llm_reconstruction import ReconstructionSMETimes, add_targets, pack
from battery_smetimes.research_llm import load_adapter
from battery_smetimes.llm_curve_quality_v3 import curve_metrics_v3
from scripts.audit_llm_curve_repair import load_frozen, fingerprint
from scripts.train_llm_full_life import save


def infer(network, processor, signal, queries):
    # Only dummy targets reach batching. Actual SOH is used after inference.
    contexts = processor.inputs(signal)
    rows = add_targets(contexts, dict(cycle=signal['cycle'],
                       label_cycle=queries, soh=np.ones(len(queries))))
    predictions = []
    with torch.inference_mode():
        for begin in range(0, len(rows), 4):
            group = rows[begin:begin+4]
            cards = network.encode_cards([[r['card']] for r in group])[:, 0].cpu()
            for row, card in zip(group, cards):
                row['cards'] = card[None].expand(20, -1).clone()
            batch = pack(group, 'cuda')
            tokens, _ = network.encode(batch['history'], batch['length'], batch['cards'])
            values = network.reconstruct(tokens, batch).cpu().numpy()
            for row, value in zip(group, values):
                predictions.append((row['target_cycle'], value[:len(row['truth'])]))
    x = np.concatenate([r[0] for r in predictions])
    p = np.concatenate([r[1] for r in predictions])
    order = np.argsort(x)
    np.testing.assert_array_equal(x[order], queries)
    return p[order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path('experiments/llm_reconstruction/20261016_TRAIN'))
    ap.add_argument('--out', type=Path, default=Path('experiments/llm_reconstruction/20261016_heldout_MVP'))
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    audit = read(args.root/'selected_delivery_audit.json')
    assert audit['passed']
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    selections = []
    for item in audit['chosen']:
        run = Path(item['run']); cfg = read(run/'config.json')
        original_index = read(cfg['index'])
        selected = read(Path('data/interim/llm_reconstruction_mvp/20261016_heldout')/item['dataset']/'index.json')
        cid = original_index['physical_plan']['validation'][0]
        assert selected['selected_ids'] == [cid] and cid not in cfg['train_ids']
        selections.append(dict(dataset=item['dataset'], cell_id=cid, model=run.as_posix(),
                               model_sha256=item['model_sha256'], role='validation'))
    registration = dict(selection_rule='First validation ID in each frozen split; no metric-based selection',
        cells=selections, no_training=True, no_model_selection=True,
        full_signal_task='Same-time SOH estimation with all released signal observations',
        prefix_task='Untrained head extrapolation: <=100-cycle signals only; future query uses final observed patch',
        source_sha256=fingerprint(__file__))
    if (args.out/'registration.json').exists():
        assert read(args.out/'registration.json') == registration
    else:
        save(args.out/'registration.json', registration)
        shutil.copy2(__file__, args.out/'source_snapshot.py')
    results = []
    for item in audit['chosen']:
        ds = item['dataset']; run = Path(item['run']); cfg = read(run/'config.json')
        folder = args.out/ds; folder.mkdir(exist_ok=True)
        if (folder/'result.json').exists():
            results.append(read(folder/'result.json')); continue
        assert fingerprint(run/'model.pt') == item['model_sha256']
        index = read(cfg['index'])
        _, cells, plan, _, _ = load_frozen(index['base_settings'])
        cid = plan['validation'][0]; cell = cells[cid]
        extracted = read(Path('data/interim/llm_reconstruction_mvp/20261016_heldout')/ds/'index.json')['cells'][0]
        assert extracted['cell_id'] == cid and fingerprint(extracted['path']) == extracted['sha256']
        with np.load(extracted['path'], allow_pickle=False) as data:
            cell['cycle'], cell['raw'] = data['cycle'].copy(), data['raw'].copy()
        # Strip identities, labels, lifetime, capacity and all unneeded fields.
        signal = {k:cell[k] for k in ('cycle','raw','metadata')}
        processor_run = Path(cfg.get('parent_run', str(run)))
        processor = joblib.load(processor_run/'processor.joblib')
        assert processor.train_ids == cfg['train_ids'] and cid not in processor.train_ids
        model = ReconstructionSMETimes(cfg['checkpoint'], processor.raw_width, cfg['seed'])
        load_adapter(model, torch.load(run/'model.pt', map_location='cpu', weights_only=True))
        model.eval(); model.requires_grad_(False)
        take = np.isfinite(cell['soh']) & (cell['soh'] > 0) & (cell['label_cycle'] > 100)
        x = cell['label_cycle'][take]; truth = cell['soh'][take]
        assert len(x) >= 2
        complete = infer(model, processor, signal, x)
        prefix_take = signal['cycle'] <= 100
        prefix = {**signal, 'cycle':signal['cycle'][prefix_take], 'raw':signal['raw'][prefix_take]}
        assert len(prefix['cycle']) and prefix['cycle'].max() <= 100
        extrapolated = infer(model, processor, prefix, x)
        # Query horizon must not affect an earlier prediction. Labels never input.
        replay = infer(model, processor, prefix, x[:max(2, min(10, len(x)))])
        np.testing.assert_allclose(replay, extrapolated[:len(replay)], atol=1e-6, rtol=0)
        assert fingerprint(run/'model.pt') == item['model_sha256']
        np.savez_compressed(folder/'predictions.npz', cycle=x, truth=truth,
                            full_signal_prediction=complete, prefix_only_extrapolation=extrapolated)
        result = dict(dataset=ds, cell_id=cid, role='validation', single_seed=cfg['seed'],
            model_sha256=item['model_sha256'], raw_source=extracted,
            full_signal_metrics=curve_metrics_v3(x, truth, complete),
            prefix_only_metrics=curve_metrics_v3(x, truth, extrapolated),
            prefix_observations=int(prefix_take.sum()), prefix_last_cycle=float(prefix['cycle'][-1]),
            full_observations=len(signal['cycle']), evaluated_labels=len(x),
            evaluated_cycles_min=float(x[0]), evaluated_cycles_max=float(x[-1]),
            labels_used_as_model_input=False, no_training=True, train_cell_overlap=False,
            query_horizon_independence_verified=True, prefix_future_signals_absent=True,
            prefix_extrapolation_not_trained=True, predictions_sha256=fingerprint(folder/'predictions.npz'))
        save(folder/'result.json', result); results.append(result)
        print(dict(dataset=ds, cell_id=cid, full_mape=result['full_signal_metrics']['mape_percent'],
                   prefix_mape=result['prefix_only_metrics']['mape_percent']), flush=True)
        del model; gc.collect(); torch.cuda.empty_cache()
    save(args.out/'result.json', dict(complete=True, cells=results, registration_sha256=fingerprint(args.out/'registration.json')))


if __name__ == '__main__':
    main()
