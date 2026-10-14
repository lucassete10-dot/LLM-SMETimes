"""First physics/domain-alignment ablation on causal MIT current SOH.

This is deliberately smaller than the proposed Battery-SMETimes architecture:
it isolates the value of a physics penalty and source-source CORAL alignment
before adding a temporal backbone or MoE. No held-out batch labels, capacity,
future signal, or endpoint enter fitting, preprocessing, or inference.
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from battery_v2_features import load_cell, load_manifest
from evaluate_soh_age_signal_v2 import age_matrix
from train_soh_v2 import outer_splits


ROOT = Path('battery_models_v2/mit_physics_da_challenger')
BASELINE = Path('battery_models_v2/soh_age_signal_challenger/mit_development_oof_predictions.npz')
VARIANTS = {
    'tiny_mlp': (0., 0.),
    'physics_only': (.2, 0.),
    'coral_only': (0., .02),
    'physics_and_coral': (.2, .02),
}
EPOCHS = 12
SAMPLES_PER_CELL_PER_EPOCH = 96
BATCH_SIZE = 512
SEED = 2026


class TinyHealthNet(torch.nn.Module):
    def __init__(self, input_width):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_width, 64), torch.nn.LayerNorm(64),
            torch.nn.GELU(), torch.nn.Linear(64, 32),
            torch.nn.LayerNorm(32), torch.nn.GELU())
        self.head = torch.nn.Linear(32, 1)

    def forward(self, x):
        hidden = self.encoder(x)
        return self.head(hidden).squeeze(-1), hidden


def coral_penalty(hidden, batch_index):
    """Source-source moment alignment; no target/test distribution is used."""
    unique = torch.unique(batch_index)
    if len(unique) != 2:
        return hidden.new_zeros(())
    parts = [hidden[batch_index == value] for value in unique]
    if min(len(part) for part in parts) < 2:
        return hidden.new_zeros(())
    means = [part.mean(dim=0) for part in parts]
    centered = [part - mean for part, mean in zip(parts, means)]
    covariances = [part.T @ part / (len(part) - 1) for part in centered]
    return ((means[0]-means[1]).square().mean()
            + (covariances[0]-covariances[1]).square().mean())


def physics_penalty(current, previous, cycle_gap, regeneration_per_cycle=.002):
    """Allow small regeneration, penalize larger predicted upward jumps."""
    excess = torch.relu(current-previous-regeneration_per_cycle*cycle_gap)
    return excess.mean()


def prepare_fold(cells, matrices, training, validation):
    # Fit all signal preprocessing on source cells only. No target-batch
    # moments are used, even without labels.
    source_x = np.concatenate([matrices[key] for key in training])
    imputer = SimpleImputer(strategy='median', keep_empty_features=True).fit(source_x)
    scaler = StandardScaler().fit(imputer.transform(source_x))
    keys = [*training, *validation]
    scaled = {key: np.clip(scaler.transform(imputer.transform(matrices[key])),
                           -8., 8.).astype(np.float32) for key in keys}
    return scaled


def draw_epoch(cells, scaled, training, group_number, rng):
    current, previous, target, gap, group = [], [], [], [], []
    for key in training:
        cell = cells[key]
        eligible = np.flatnonzero(cell.soh_available)
        # Equal examples per cell prevent long-lived cells from dominating.
        sampled = rng.choice(eligible, size=SAMPLES_PER_CELL_PER_EPOCH,
                             replace=len(eligible) < SAMPLES_PER_CELL_PER_EPOCH)
        current.append(scaled[key][sampled])
        previous.append(scaled[key][sampled-1])
        target.append(cell.soh[sampled])
        gap.append(cell.cycle[sampled]-cell.cycle[sampled-1])
        group.append(np.full(len(sampled), group_number[cell.group], dtype=np.int64))
    arrays = [np.concatenate(part) for part in (current, previous, target, gap, group)]
    order = rng.permutation(len(arrays[0]))
    return [part[order] for part in arrays]


def fit_variant(cells, scaled, training, variant, device):
    physics_weight, coral_weight = VARIANTS[variant]
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    model = TinyHealthNet(next(iter(scaled.values())).shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    group_number = {group: i for i, group in enumerate(sorted({
        cells[key].group for key in training}))}
    if len(group_number) != 2:
        raise ValueError('expected two source MIT batches per outer fold')
    rng = np.random.default_rng(SEED)
    training_log = []
    model.train()
    for epoch in range(EPOCHS):
        arrays = draw_epoch(cells, scaled, training, group_number, rng)
        losses = []
        for start in range(0, len(arrays[0]), BATCH_SIZE):
            part = [torch.as_tensor(a[start:start+BATCH_SIZE], device=device)
                    for a in arrays]
            x, x_prev, y, gap, group = part
            current, hidden = model(x)
            supervised = torch.nn.functional.huber_loss(current, y, delta=.03)
            loss = supervised
            if physics_weight:
                previous, _ = model(x_prev)
                loss = loss + physics_weight*physics_penalty(
                    current, previous, gap.float())
            if coral_weight:
                loss = loss + coral_weight*coral_penalty(hidden, group)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        training_log.append(float(np.mean(losses)))
    return model, training_log


def predict(model, matrix, device):
    model.eval()
    pieces = []
    with torch.no_grad():
        for start in range(0, len(matrix), 4096):
            x = torch.as_tensor(matrix[start:start+4096], device=device)
            estimate, _ = model(x)
            pieces.append(estimate.cpu().numpy())
    return np.concatenate(pieces)


def score(cells, validation, predictions):
    rows = []
    for key in validation:
        cell = cells[key]
        keep = cell.soh_available
        truth = cell.soh[keep]
        estimate = predictions[key][keep]
        rows.append({'cell_id': key, 'mape_percent': float(np.mean(
            100*np.abs(estimate-truth)/truth)),
                     'bias_soh': float(np.mean(estimate-truth))})
    return {'cells': len(rows),
            'cell_macro_mape_percent': float(np.mean([r['mape_percent'] for r in rows])),
            'cell_macro_bias_soh': float(np.mean([r['bias_soh'] for r in rows])),
            'by_cell': rows}


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    records = [row for row in load_manifest()['cells'] if row['dataset'] == 'MIT']
    cells = {row['cell_id']: load_cell(row) for row in records}
    matrices = {key: age_matrix(cell) for key, cell in cells.items()}
    with np.load(BASELINE) as archive:
        baseline = {(str(key), int(cycle)): float(value)
                    for key, cycle, value in zip(archive['cell_id'], archive['cycle'],
                                                 archive['gated_with_age'])}
    folds = []
    oof = {name: {} for name in VARIANTS}
    for held_out, training, validation in outer_splits(records, 'MIT'):
        scaled = prepare_fold(cells, matrices, training, validation)
        baseline_pred = {key: np.asarray([baseline[key, int(cycle)]
                           for cycle in cells[key].cycle]) for key in validation}
        metrics = {'age_extratrees': score(cells, validation, baseline_pred)}
        logs = {}
        for name in VARIANTS:
            model, logs[name] = fit_variant(cells, scaled, training, name, device)
            predictions = {key: predict(model, scaled[key], device) for key in validation}
            oof[name].update(predictions)
            metrics[name] = score(cells, validation, predictions)
        folds.append({'held_out': held_out, 'training_cells': len(training),
                      'validation_cells': len(validation), 'metrics': metrics,
                      'training_loss_by_epoch': logs})
        print(held_out, {key: round(value['cell_macro_mape_percent'], 3)
                         for key, value in metrics.items()}, flush=True)
    names = ['age_extratrees', *VARIANTS]
    summary = {name: {'cell_macro_mape_percent': float(np.mean([
        row['mape_percent'] for fold in folds
        for row in fold['metrics'][name]['by_cell']]))}
        for name in names}
    report = {'status': 'unpromoted development ablation, not independent test',
              'scope': 'MIT current SOH with first 300 s causal signals and observed age',
              'method': 'small MLP; optional monotonicity and source-source CORAL losses',
              'held_out_batch_preprocessing_used': False,
              'source_cell_sampling': 'equal examples per cell per epoch',
              'settings': {'epochs': EPOCHS, 'samples_per_cell_per_epoch':
                           SAMPLES_PER_CELL_PER_EPOCH, 'batch_size': BATCH_SIZE,
                           'seed': SEED, 'weights': VARIANTS, 'device': str(device)},
              'summary': summary, 'folds': folds}
    (ROOT/'development_cv.json').write_text(json.dumps(report, indent=2),
                                             encoding='utf-8')
    ids = [row['cell_id'] for row in records]
    np.savez_compressed(ROOT/'development_oof.npz',
                        cell_id=np.concatenate([np.full(len(cells[key].cycle), key)
                                                for key in ids]),
                        cycle=np.concatenate([cells[key].cycle for key in ids]),
                        target=np.concatenate([cells[key].soh for key in ids]),
                        soh_available=np.concatenate([cells[key].soh_available for key in ids]),
                        **{name: np.concatenate([oof[name][key] for key in ids])
                           for name in VARIANTS})
    print('SUMMARY', summary, flush=True)
    print('WROTE', ROOT, flush=True)


if __name__ == '__main__':
    main()
