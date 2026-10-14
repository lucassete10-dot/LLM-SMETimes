"""Small local-checkpoint smoke test of the original Llama SMETimes backbone.

This does not train a battery SOH model or assess predictive performance. It
only checks that two audited label-free diagnostic sequences reach the actual
frozen Llama backbone and produce finite token features/reconstructions.
"""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from battery_smetimes_signal_v2 import build_sequence
from battery_v2_features import load_manifest
from models.SMETimes_Llama import Model


CHECKPOINT = Path('models/llm/llama-3.2-1b-instruct')
OUT = Path('battery_models_v2/original_smetimes_smoke.json')


def checkpoint_sha256():
    digest = hashlib.sha256()
    with (CHECKPOINT / 'model.safetensors').open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def run():
    if not torch.cuda.is_available():
        raise RuntimeError('original 1B SMETimes smoke requires available CUDA GPU')
    record = next(row for row in load_manifest()['cells'] if row['dataset'] == 'XJTU')
    with np.load(record['feature_source'], allow_pickle=False) as archive:
        raw = archive[record['feature_array']][:2]
    sequence, context = build_sequence('XJTU', raw)
    if sequence.shape != (2, 10, 9) or context.shape != (2, 4):
        raise ValueError('battery/SMETimes adapter shape mismatch')
    config = SimpleNamespace(token_len=5, device='cuda:0',
        llm_ckp_dir=str(CHECKPOINT), mix_embeds=False, use_amp=True,
        local_files_only=True, num_experts=2, lambda_reg=0.,
        mlp_hidden_layers=0, use_multi_gpu=False, local_rank=0)
    model = Model(config).to(device='cuda:0', dtype=torch.float16).eval()
    frozen_backbone = all(not parameter.requires_grad
                          for parameter in model.llama.parameters())
    if not frozen_backbone:
        raise ValueError('original SMETimes backbone is unexpectedly trainable')
    captured = []

    def capture_backbone(_module, _inputs, output):
        captured.append(output[0].detach())

    hook = model.llama.model.register_forward_hook(capture_backbone)
    try:
        x = torch.as_tensor(sequence, dtype=torch.float16, device='cuda:0')
        with torch.inference_mode():
            reconstruction = model(x, None, None, None)
    finally:
        hook.remove()
    if (len(captured) != 1 or captured[0].shape != (2*9, 2, 2048) or
            reconstruction.shape != x.shape or
            not torch.isfinite(captured[0]).all() or
            not torch.isfinite(reconstruction).all()):
        raise ValueError('original SMETimes backbone feature/reconstruction smoke failed')
    return {'status': 'original Llama SMETimes forward smoke only; no SOH/RUL accuracy claim',
            'cell_id': record['cell_id'], 'samples': 2,
            'input_shape': list(sequence.shape),
            'backbone_hidden_shape': list(captured[0].shape),
            'reconstruction_shape': list(reconstruction.shape),
            'backbone_frozen': frozen_backbone,
            'checkpoint': str(CHECKPOINT),
            'checkpoint_sha256': checkpoint_sha256(),
            'cuda_peak_allocated_gb': round(torch.cuda.max_memory_allocated()/2**30, 3),
            'note': 'tokenizer/decoder are untrained; this is not a usable SOH feature yet'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--verify', action='store_true')
    actions.add_argument('--replace', action='store_true',
                         help='replace this smoke-only, non-performance report')
    args = parser.parse_args()
    report = run()
    if args.verify:
        saved = json.loads(OUT.read_text(encoding='utf-8'))
        checked = set(report)-{'cuda_peak_allocated_gb'}
        if checked != set(saved)-{'cuda_peak_allocated_gb'} or any(
                report[key] != saved[key] for key in checked):
            raise ValueError('original SMETimes smoke differs from saved report')
        print('VERIFIED', OUT, report['backbone_hidden_shape'], flush=True)
    else:
        if OUT.exists() and not args.replace:
            raise FileExistsError(OUT)
        OUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print('WROTE', OUT, report['backbone_hidden_shape'], flush=True)
