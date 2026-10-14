"""Original frozen-Llama SMETimes as an XJTU SOH feature extractor.

Per outer strategy: train the original SMETimes tokenizer/MoE/decoder only to
reconstruct label-free diagnostic curves from *other* strategies. Extract
hidden Llama tokens, fit PCA and SOH regression heads on outer-training cells,
then score untouched held-out cells. Fixed settings; no held-out labels enter
representation fitting, PCA, scaling, or regression.

This is a development ablation, not a selected or externally validated model.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler

from battery_smetimes_signal_v2 import build_sequence
from battery_v2_features import load_all
from models.SMETimes_Llama import Model
from smoke_original_smetimes_battery_v2 import CHECKPOINT, checkpoint_sha256
from train_soh_v2 import outer_splits, score


OUT = Path('battery_models_v2/original_smetimes_xjtu')
EPOCHS = 3
SAMPLES_PER_TRAIN_CELL = 24
BATCH_SIZE = 16
EMBED_BATCH_SIZE = 64
PCA_COMPONENTS = 32
SEED = 2026


def sample_indices(length, count):
    return np.unique(np.round(np.linspace(0, length-1, min(length, count))).astype(int))


def instantiate():
    config = SimpleNamespace(token_len=5, device='cuda:0',
        llm_ckp_dir=str(CHECKPOINT), mix_embeds=False, use_amp=True,
        local_files_only=True, num_experts=2, lambda_reg=0.,
        mlp_hidden_layers=0, use_multi_gpu=False, local_rank=0)
    model = Model(config).to('cuda:0')
    model.llama.config.use_cache = False
    if any(parameter.requires_grad for parameter in model.llama.parameters()):
        raise ValueError('pretrained Llama backbone must be frozen')
    return model


def pretrain(model, sequences, ids):
    selected = np.concatenate([sequences[key][sample_indices(len(sequences[key]),
                                                              SAMPLES_PER_TRAIN_CELL)]
                               for key in ids]).astype('float32')
    channel_scale = np.maximum(np.std(selected, axis=(0, 1)), .1).astype('float32')
    data = torch.as_tensor(selected, device='cuda:0')
    scale = torch.as_tensor(channel_scale, device='cuda:0')
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=.001, weight_decay=.03)
    grad_scaler = torch.amp.GradScaler('cuda')
    generator = torch.Generator(device='cpu').manual_seed(SEED)
    losses = []
    for epoch in range(EPOCHS):
        model.train()
        order = torch.randperm(len(data), generator=generator)
        batch_losses = []
        for positions in order.split(BATCH_SIZE):
            batch = data[positions.to('cuda:0')]
            with torch.autocast('cuda', dtype=torch.float16):
                reconstructed = model(batch, None, None, None)
                loss = (((reconstructed.float()-batch)/scale)**2).mean()
            if not torch.isfinite(loss):
                raise ValueError('nonfinite original SMETimes reconstruction loss')
            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters()
                                            if p.requires_grad), 1.)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            batch_losses.append(float(loss.detach()))
        losses.append(float(np.mean(batch_losses)))
        print('RECON_EPOCH', epoch+1, round(losses[-1], 6), flush=True)
    model.eval()
    return {'sampled_train_diagnostics': len(selected),
            'reconstruction_loss_by_epoch': losses,
            'cuda_peak_allocated_gb': round(torch.cuda.max_memory_allocated()/2**30, 3)}


def encode_batch(model, x):
    """Follow the original SMETimes tokenizer and frozen Llama path to tokens."""
    b, steps, channels = x.shape
    if steps != 10 or channels != 9:
        raise ValueError('SMETimes diagnostic sequence shape differs')
    means = x.mean(1, keepdim=True)
    centered = x-means
    stdev = torch.sqrt(torch.var(centered, dim=1, keepdim=True, unbiased=False)+1e-5)
    normalized = centered/stdev
    folded = normalized.permute(0, 2, 1).reshape(b*channels, steps).unfold(
        -1, size=5, step=5)
    embedded = model.encoder(folded)
    hidden = model.llama.model(inputs_embeds=embedded, use_cache=False)[0]
    if hidden.shape != (b*channels, 2, 2048):
        raise ValueError('unexpected original SMETimes hidden token shape')
    return hidden.reshape(b, channels, 2, 2048)[:, :3].mean(dim=(1, 2))


def encode_cell(model, sequence):
    pieces = []
    with torch.inference_mode():
        for part in np.array_split(sequence, max(1, int(np.ceil(len(sequence)/EMBED_BATCH_SIZE)))):
            x = torch.as_tensor(part, dtype=torch.float32, device='cuda:0')
            with torch.autocast('cuda', dtype=torch.float16):
                value = encode_batch(model, x)
            pieces.append(value.float().cpu().numpy())
    result = np.concatenate(pieces).astype('float32')
    if result.shape != (len(sequence), 2048) or not np.isfinite(result).all():
        raise ValueError('nonfinite original SMETimes cell embedding')
    return result


def historical_features(projected, reference_count):
    anchor = np.median(projected[:reference_count], axis=0)
    delta = projected-anchor
    history = np.stack([np.mean(delta[max(0, i-4):i+1], axis=0)
                        for i in range(len(delta))])
    return np.column_stack((projected, delta, history)).astype('float32')


def fit_head(cells, features, ids, kind):
    x_parts, y_parts, w_parts = [], [], []
    for key in ids:
        cell = cells[key]
        keep = cell.soh_available
        x_parts.append(features[key][keep])
        y_parts.append(cell.soh[keep])
        weight = np.where(cell.capacity_review_flag[keep], .25, 1.).astype('float64')
        w_parts.append(weight/weight.sum()/len(ids))
    x, y, weight = map(np.concatenate, (x_parts, y_parts, w_parts))
    imputer = SimpleImputer(strategy='median', keep_empty_features=True).fit(x)
    x = imputer.transform(x)
    if kind == 'ridge':
        scaler = RobustScaler(quantile_range=(10, 90)).fit(x)
        scaler.scale_ = np.maximum(scaler.scale_, .02)
        x = np.clip(scaler.transform(x), -6, 6)
        model = Ridge(alpha=10).fit(x, y, sample_weight=weight*len(x))
    else:
        scaler = None
        model = ExtraTreesRegressor(n_estimators=160, min_samples_leaf=5,
                                    max_features=.8, random_state=SEED, n_jobs=-1)
        model.fit(x, y, sample_weight=weight)
    return imputer, scaler, model


def predict_head(bundle, features):
    imputer, scaler, model = bundle
    x = imputer.transform(features)
    if scaler is not None:
        x = np.clip(scaler.transform(x), -6, 6)
    return model.predict(x)


def evaluate_fold(held_out):
    if not torch.cuda.is_available():
        raise RuntimeError('original 1B SMETimes requires CUDA for this experiment')
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(4)
    manifest, all_cells = load_all()
    cells = {key: cell for key, cell in all_cells.items() if cell.dataset == 'XJTU'}
    splits = {group: (train, val) for group, train, val in outer_splits(
        manifest['cells'], 'XJTU')}
    if held_out not in splits:
        raise ValueError(f'unknown XJTU held-out strategy: {held_out}')
    train, val = splits[held_out]
    sequences = {key: build_sequence('XJTU', cells[key].raw)[0]
                 for key in cells}
    model = instantiate()
    unsupervised = pretrain(model, sequences, train)
    embeddings = {key: encode_cell(model, sequences[key]) for key in train+val}
    pca = PCA(n_components=PCA_COMPONENTS, svd_solver='randomized',
              random_state=SEED).fit(np.concatenate([embeddings[key] for key in train]))
    features = {key: historical_features(pca.transform(embeddings[key]),
                                          cells[key].reference_count)
                for key in train+val}
    candidates = {}
    for kind in ('ridge', 'extra_trees'):
        head = fit_head(cells, features, train, kind)
        prediction = {key: predict_head(head, features[key]) for key in val}
        candidates[kind] = score(cells, prediction, val)
        print('SOH', held_out, kind,
              round(candidates[kind]['cell_macro_mae'], 5), flush=True)
    prior = next(row['extra_trees_history'] for row in json.loads(Path(
        'battery_models_v2/soh/xjtu/group_cv.json').read_text(encoding='utf-8'))
        if row['held_out'] == held_out)
    return {'status': 'development-only original Llama SMETimes SOH feature ablation',
            'held_out': held_out, 'train_cell_ids': train, 'validation_cell_ids': val,
            'encoder_pretraining': 'label-free reconstruction on outer training strategies only',
            'head_training': 'PCA and Ridge/ExtraTrees fit on outer training cells only',
            'checkpoint_sha256': checkpoint_sha256(),
            'fixed_settings': {'epochs': EPOCHS,
                               'samples_per_train_cell': SAMPLES_PER_TRAIN_CELL,
                               'batch_size': BATCH_SIZE,
                               'pca_components': PCA_COMPONENTS,
                               'seed': SEED},
            'encoder_diagnostic': unsupervised,
            'baseline_extra_trees_history': prior,
            'original_smetimes_ridge': candidates['ridge'],
            'original_smetimes_extra_trees': candidates['extra_trees'],
            'warning': 'one/few development folds do not establish generalization; '
                       'the baseline remains selected until strict whole-strategy '
                       'comparison and independent testing'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--held-out', choices=[f'batch{i}' for i in range(1, 7)],
                        required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / f'{args.held_out}.json'
    if destination.exists():
        raise FileExistsError(destination)
    report = evaluate_fold(args.held_out)
    destination.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('WROTE', destination, flush=True)
