"""Small-data SMETimes derivative: learned numerical + frozen Llama features.

Nested battery splits select epochs without outer-test labels. This is an
architecture adaptation, not a reproduction of the original SMETimes backbone.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

from battery_raw_mvp import (RawBattery, extract_prefix_features, feature_names,
                             make_semantic_embeddings)
from battery_mvp import compute_metrics, set_seed

IDS = ['B0005', 'B0006', 'B0007', 'B0018']


def strict_prefix(data):
    times = np.asarray(data['Time']).reshape(-1)
    mask = times <= 1000
    if times[-1] < 1000 or mask.sum() < 2:
        raise ValueError('Insufficient prefix')
    prefix = {}
    for channel in ['Time', 'Voltage_measured', 'Current_measured', 'Temperature_measured']:
        values = np.asarray(data[channel]).reshape(-1)[mask]
        if not np.isfinite(values).all():
            raise ValueError(f'Nonfinite prefix: {channel}')
        prefix[channel] = np.r_[values, 1000 if channel == 'Time' else values[-1]]
    return extract_prefix_features(prefix, np.arange(0, 1001, 100), 1000)


def load_data(root):
    """Strict observation cutoff: never interpolate with a sample after 1000 s."""
    batteries = {}
    for key in IDS:
        path = root / (key + '.mat')
        features, capacities, cycles = [], [], []
        for idx, record in enumerate(loadmat(path, simplify_cells=True)[key]['cycle'], 1):
            if record['type'] != 'discharge':
                continue
            data = record['data']
            features.append(strict_prefix(data))
            capacities.append(float(data['Capacity']))
            cycles.append(idx)
        x = np.asarray(features, dtype=np.float32)
        y = np.asarray(capacities, dtype=np.float32)
        if not len(y) or not np.isfinite(y).all() or np.any(y <= 0):
            raise ValueError(f'{key}: invalid capacity labels')
        batteries[key] = RawBattery(key, np.arange(1, len(y)+1), np.array(cycles),
                                     x-x[0], y, y/y[0], str(path.resolve()))
    return batteries


class Features:
    def fit(self, batteries, semantic, ids):
        x = np.concatenate([batteries[k].features for k in ids])
        self.scaler = StandardScaler().fit(x)
        # A floor avoids exploding near-constant current channels on a new cell.
        self.scaler.scale_ = np.maximum(self.scaler.scale_, 0.02)
        self.pca = PCA(n_components=8, svd_solver='full').fit(np.concatenate([semantic[k] for k in ids]))
        self.semantic_scaler = StandardScaler().fit(self.pca.transform(np.concatenate([semantic[k] for k in ids])))
        self.ridge = Ridge(alpha=10).fit(self.scaler.transform(x), np.concatenate([batteries[k].soh for k in ids]))
        return self

    def transform(self, battery, semantic):
        x = self.scaler.transform(battery.features).astype('float32')
        s = self.semantic_scaler.transform(self.pca.transform(semantic)).astype('float32')
        # A causal trailing window; no future cycles or historical SOH labels.
        history = np.stack([x[max(0, i-7):i+1].mean(0) for i in range(len(x))])
        return [torch.tensor(a, dtype=torch.float32) for a in
                (np.clip(x, -6, 6), np.clip(history, -6, 6), np.clip(s, -6, 6),
                 self.ridge.predict(x), battery.soh)]


class RegularizedSMETimes(nn.Module):
    def __init__(self, semantic=True, feature_dim=44):
        super().__init__()
        self.semantic = semantic
        self.encoder = nn.Sequential(nn.Linear(feature_dim, 24), nn.GELU(), nn.Dropout(.15), nn.Linear(24, 8))
        self.reconstruct = nn.Linear(8, feature_dim)
        self.semantic_encoder = nn.Linear(8, 8)
        self.gate = nn.Parameter(torch.tensor(-2.0))
        self.head = nn.Sequential(nn.Linear(16, 16), nn.GELU(), nn.Dropout(.15), nn.Linear(16, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x, history, semantic, baseline):
        z = self.encoder(x)
        h = self.encoder(history)
        fused = z
        if self.semantic:
            fused = z + torch.sigmoid(self.gate)*torch.tanh(self.semantic_encoder(semantic))
        correction = .15*torch.tanh(self.head(torch.cat([fused, h], -1)).squeeze(-1))
        return baseline + correction, self.reconstruct(z), z


def join(data, ids):
    return [torch.cat([data[k][i] for k in ids]) for i in range(5)]


def train(data, ids, val_id, semantic, seed, epochs, patience=15):
    set_seed(seed)
    model = RegularizedSMETimes(semantic, feature_dim=data[ids[0]][0].shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.03)
    x, h, s, base, y = join(data, ids)
    weights = torch.cat([torch.full((len(data[k][0]),), 1/len(data[k][0])/len(ids)) for k in ids])
    best, best_epoch, stale, history = None, 0, 0, []
    best_loss = float('inf')
    for epoch in range(1, epochs+1):
        model.train()
        pred, reconstruction, _ = model(x + .015*torch.randn_like(x), h, s, base)
        supervised = ((pred-y).square()*weights).sum()
        loss = supervised + .0002*(reconstruction-x).square().mean() + .01*(pred-base).square().mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        opt.step()
        if val_id is not None:
            model.eval()
            with torch.no_grad():
                val = model(*data[val_id][:4])[0]
                score = (val[15:]-data[val_id][4][15:]).square().mean().item()
            history.append({'epoch': epoch, 'train_mse': supervised.item(), 'val_mse': score})
            if score < best_loss - 1e-7:
                best_loss, best_epoch, stale = score, epoch, 0
                best = copy.deepcopy(model.state_dict())
            else:
                stale += 1
            if stale >= patience:
                break
    if val_id is not None:
        model.load_state_dict(best)
    else:
        best_epoch = epochs
    model.eval()
    return model, best_epoch, history


def predict_from_features(fold, delta_features, llama_embeddings, semantic=True):
    """Inference needs observed prefix features, not capacities or SOH labels.

    Rows must start at the reference discharge and be in chronological order.
    Embeddings must use the same frozen Llama/prompt and prefix configuration.
    """
    fold = Path(fold)
    prep = Features()
    prep.__dict__.update(joblib.load(fold / 'preprocessing.joblib'))
    x = np.asarray(delta_features, dtype=np.float32)
    marks = np.asarray(llama_embeddings, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 44 or len(x) != len(marks) or not np.isfinite(x).all() or not np.isfinite(marks).all():
        raise ValueError('Expected finite aligned arrays: [cycles,44] and [cycles,llama_hidden]')
    n = len(x)
    dummy = RawBattery('inference', np.arange(n)+1, np.arange(n)+1, x,
                       np.zeros(n), np.zeros(n), '')
    data = prep.transform(dummy, marks)
    name = 'smetimes_fusion' if semantic else 'numeric_ablation'
    runs = []
    for seed in (2026, 2027, 2028):
        checkpoint = torch.load(fold / f'{name}_{seed}.pt', weights_only=True)
        model = RegularizedSMETimes(semantic)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        with torch.no_grad():
            runs.append(model(*data[:4])[0].numpy())
    return np.mean(runs, axis=0)


def same_split(args, batteries, semantic):
    """Direct comparison with previous experiment, keeping B0007 validation-only."""
    folder = args.output / 'original_split'
    folder.mkdir(exist_ok=True)
    prep = Features().fit(batteries, semantic, ['B0005', 'B0006'])
    joblib.dump(vars(prep), folder / 'preprocessing.joblib')
    data = {k: prep.transform(batteries[k], semantic[k]) for k in IDS}
    metrics, predictions = {}, {}
    for use_semantic in (False, True):
        name = 'smetimes_fusion' if use_semantic else 'numeric_ablation'
        runs, seed_metrics = [], []
        for seed in (2026, 2027, 2028):
            model, epoch, history = train(data, ['B0005', 'B0006'], 'B0007', use_semantic, seed, args.epochs)
            with torch.no_grad():
                pred = model(*data['B0018'][:4])[0].numpy()
            runs.append(pred)
            seed_metrics.append({'seed': seed, 'epoch': epoch, **compute_metrics(pred[15:], batteries['B0018'].soh[15:])})
            torch.save({'state_dict': model.state_dict(), 'semantic': use_semantic, 'seed': seed,
                        'train_ids': ['B0005', 'B0006'], 'epochs': epoch}, folder / f'{name}_{seed}.pt')
            (folder / f'{name}_{seed}_history.json').write_text(json.dumps(history, indent=2))
        predictions[name] = np.mean(runs, axis=0)
        metrics[name] = {'ensemble': compute_metrics(predictions[name][15:], batteries['B0018'].soh[15:]), 'seeds': seed_metrics}
    (folder / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    np.savez(folder / 'predictions.npz', true_soh=batteries['B0018'].soh, **predictions)
    plt.figure(figsize=(10, 5))
    cycles = np.arange(16, len(batteries['B0018'].soh)+1)
    plt.plot(cycles, batteries['B0018'].soh[15:], 'k', label='True SOH', linewidth=2)
    for name, values in predictions.items():
        plt.plot(cycles, values[15:], label=name)
    old = Path('battery_raw_mvp_results_delta/predictions.csv')
    if old.exists():
        import pandas as pd
        frame = pd.read_csv(old)
        for col in ['pred_semantic', 'pred_random_forest']:
            plt.plot(frame.discharge_number, frame[col], '--', alpha=.7, label='Previous '+col.replace('pred_', ''))
    plt.xlabel('Discharge observation')
    plt.ylabel('SOH')
    plt.title('B0018: train B0005+B0006, validation B0007 (retrospective)')
    plt.legend()
    plt.grid(alpha=.2)
    plt.tight_layout()
    plt.savefig(folder / 'comparison.png', dpi=170)
    print('ORIGINAL SPLIT', json.dumps(metrics), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--raw-root', type=Path, default=Path(r'D:\LLM\NASA\raw\extracted'))
    p.add_argument('--output', type=Path, default=Path('battery_smetimes_regularized_results'))
    p.add_argument('--epochs', type=int, default=120)
    p.add_argument('--same-split-only', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    batteries = load_data(args.raw_root)
    configuration = {
        'architecture': 'SMETimes derivative with frozen Llama semantic extraction; not original backbone',
        'prefix_seconds': 1000, 'reference': 'first observed discharge curve and capacity',
        'features': feature_names(np.arange(0, 1001, 100)),
        'max_epochs': args.epochs, 'seeds': [2026, 2027, 2028],
        'trainable_parameters': sum(p.numel() for p in RegularizedSMETimes().parameters()),
        'protocol': 'outer leave-one-cell-out; inner leave-one-cell-out epoch selection; median epoch refit',
        'source_sha256': {k: hashlib.sha256(Path(b.source_path).read_bytes()).hexdigest() for k,b in batteries.items()},
        'limitations': ['four previously examined NASA cells; retrospective assessment',
                        'no Maryland/MIT/XJTU validation', 'requires first-cycle reference measurement',
                        'current-cycle SOH estimation, not future SOH forecast or RUL'],
    }
    (args.output / 'configuration.json').write_text(json.dumps(configuration, indent=2))
    signature = hashlib.sha256(b''.join(b.features.tobytes() for b in batteries.values())).hexdigest()[:16]
    cache = args.output / f'llama_prefix_{signature}.pt'
    if not cache.exists():
        from transformers import AutoTokenizer, LlamaForCausalLM
        path = 'models/llm/llama-3.2-1b-instruct'
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        llama = LlamaForCausalLM.from_pretrained(path, local_files_only=True,
                    dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32).to(device)
        make_semantic_embeddings(batteries, feature_names(np.arange(0,1001,100)), 1000, 'delta',
              AutoTokenizer.from_pretrained(path, local_files_only=True), llama, device, 32, cache)
        del llama
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    semantic = {k: v.float().numpy() for k,v in torch.load(cache, weights_only=True).items()}
    if args.same_split_only:
        same_split(args, batteries, semantic)
        return
    metrics, rows = {}, []
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, test in zip(axes.flat, IDS):
        development = [k for k in IDS if k != test]
        fold = args.output / test
        fold.mkdir(exist_ok=True)
        selected = {}
        # All architecture choices fixed before this loop; only epoch is selected.
        for use_semantic in (False, True):
            name = 'smetimes_fusion' if use_semantic else 'numeric_ablation'
            epochs = []
            for val in development:
                train_ids = [k for k in development if k != val]
                prep = Features().fit(batteries, semantic, train_ids)
                data = {k: prep.transform(batteries[k], semantic[k]) for k in development}
                _, epoch, history = train(data, train_ids, val, use_semantic, 2026, args.epochs)
                epochs.append(epoch)
                (fold / f'{name}_validate_{val}.json').write_text(json.dumps(history, indent=2))
            selected[name] = max(1, int(np.median(epochs)))
        prep = Features().fit(batteries, semantic, development)
        data = {k: prep.transform(batteries[k], semantic[k]) for k in IDS}
        joblib.dump(vars(prep), fold / 'preprocessing.joblib')
        predictions = {'ridge': data[test][3].numpy()}
        rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, max_features=.8, random_state=2026, n_jobs=-1)
        rf.fit(np.concatenate([batteries[k].features for k in development]), np.concatenate([batteries[k].soh for k in development]))
        predictions['random_forest'] = rf.predict(batteries[test].features)
        joblib.dump(rf, fold / 'random_forest.joblib')
        seed_results = {}
        for use_semantic in (False, True):
            name = 'smetimes_fusion' if use_semantic else 'numeric_ablation'
            runs = []
            seed_results[name] = []
            for seed in (2026, 2027, 2028):
                model, _, _ = train(data, development, None, use_semantic, seed, selected[name])
                with torch.no_grad():
                    pred, _, latent = model(*data[test][:4])
                runs.append(pred.numpy())
                seed_results[name].append(compute_metrics(pred.numpy()[15:], batteries[test].soh[15:]))
                torch.save({'state_dict': model.state_dict(), 'semantic': use_semantic,
                     'train_ids': development, 'epochs': selected[name], 'seed': seed}, fold / f'{name}_{seed}.pt')
                np.save(fold / f'{name}_features_{seed}.npy', latent.numpy())
            predictions[name] = np.mean(runs, axis=0)
        truth = batteries[test].soh
        metrics[test] = {'training_batteries': development, 'selected_epochs': selected,
                        'metrics': {k: compute_metrics(v[15:], truth[15:]) for k,v in predictions.items()},
                        'seed_metrics': seed_results}
        ax.plot(np.arange(16, len(truth)+1), truth[15:], 'k', label='True SOH', linewidth=2)
        for name, pred in predictions.items():
            ax.plot(np.arange(16, len(truth)+1), pred[15:], label=name, alpha=.85)
        ax.set(title=f'Outer held-out cell: {test}', xlabel='Discharge observation', ylabel='SOH')
        ax.grid(alpha=.2)
        for i in range(15, len(truth)):
            rows.append({'battery': test, 'cycle': i+1, 'true_soh': float(truth[i]), **{k: float(v[i]) for k,v in predictions.items()}})
        print(test, json.dumps(metrics[test]['metrics']), 'epochs', selected, flush=True)
        (args.output/'metrics.json').write_text(json.dumps(metrics, indent=2))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output / 'cross_battery_soh.png', dpi=170)
    import csv
    with (args.output / 'predictions.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {name: {metric: float(np.mean([metrics[k]['metrics'][name][metric] for k in IDS]))
                 for metric in ('mae', 'rmse', 'r2')} for name in predictions}
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2))
    print('MACRO AVERAGE', json.dumps(summary), flush=True)
    same_split(args, batteries, semantic)


if __name__ == '__main__':
    main()
