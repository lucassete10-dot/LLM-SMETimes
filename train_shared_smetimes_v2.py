"""Cross-dataset numeric SMETimes-style encoder with domain-specific SOH heads.

Discharge voltage/current/temperature traces are resampled to five relative
positions within each *observed* prefix. XJTU charge channels are included;
other domains have zero charge channels. Absolute, calibration-relative, and
causal-history channels form a common 90-D physical representation. This is a
compact numeric derivative, not the original SMETimes LLM backbone.
"""

import json
from pathlib import Path
import warnings

import joblib
import numpy as np
import torch
from torch import nn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler

from battery_v2_features import load_all
from train_soh_v2 import outer_splits, score


OUT = Path('battery_models_v2/shared_smetimes')
DOMAINS = ('NASA', 'CALCE', 'MIT', 'XJTU')
EPOCHS = 20


def five_positions(block):
    block = np.asarray(block, dtype='float32')
    position = np.linspace(0, block.shape[1]-1, 5)
    lower = np.floor(position).astype(int)
    upper = np.ceil(position).astype(int)
    fraction = (position-lower).astype('float32')
    return block[:, lower]*(1-fraction) + block[:, upper]*fraction


def physical_features(cell):
    raw = cell.raw
    n = len(raw)
    if cell.dataset == 'XJTU':
        discharge = (raw[:, 23:28], raw[:, 28:33], raw[:, 33:38])
        charge = (raw[:, 0:5], raw[:, 5:10], raw[:, 10:15])
    elif cell.dataset in ('NASA', 'MIT'):
        discharge = (raw[:, 0:11], raw[:, 11:22], raw[:, 22:33])
        charge = None
    elif cell.dataset == 'CALCE':
        discharge = (raw[:, 0:11], raw[:, 11:22], np.full((n, 11), np.nan))
        charge = None
    else:
        raise ValueError(cell.dataset)
    parts = [five_positions(discharge[0]), np.abs(five_positions(discharge[1])),
             five_positions(discharge[2])]
    if charge is None:
        parts.extend([np.zeros((n, 5), dtype='float32') for _ in range(3)])
    else:
        parts.extend([five_positions(charge[0]), np.abs(five_positions(charge[1])),
                      five_positions(charge[2])])
    absolute = np.column_stack(parts).astype('float32')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        anchor = np.nanmedian(absolute[:cell.reference_count], axis=0)
        delta = absolute-anchor
        history = np.stack([np.nanmean(delta[max(0, i-4):i+1], axis=0)
                            for i in range(n)])
    return np.column_stack((absolute, delta, history)).astype('float32')


class DomainPrep:
    def fit(self, cells, features, ids):
        self.domains = {}
        for domain in DOMAINS:
            keys = [key for key in ids if cells[key].dataset == domain]
            if not keys:
                raise ValueError(f'no training cells for {domain}')
            x = np.concatenate([features[key] for key in keys])
            imputer = SimpleImputer(strategy='median', keep_empty_features=True).fit(x)
            imputed = imputer.transform(x)
            scaler = RobustScaler(quantile_range=(10, 90)).fit(imputed)
            scaler.scale_ = np.maximum(scaler.scale_, .02)
            x_parts, y_parts, w_parts = [], [], []
            for key in keys:
                cell = cells[key]
                keep = cell.soh_available
                x_parts.append(features[key][keep])
                y_parts.append(cell.soh[keep])
                w = np.where(cell.capacity_review_flag[keep], .25, 1.).astype('float64')
                w_parts.append(w / w.sum() / len(keys))
            supervised = np.concatenate(x_parts)
            sx = np.clip(scaler.transform(imputer.transform(supervised)), -6, 6)
            ridge = Ridge(alpha=10).fit(sx, np.concatenate(y_parts),
                                        sample_weight=np.concatenate(w_parts)*len(sx))
            self.domains[domain] = (imputer, scaler, ridge)
        return self

    def transform(self, cell, features):
        imputer, scaler, ridge = self.domains[cell.dataset]
        x = np.clip(scaler.transform(imputer.transform(features)), -6, 6).astype('float32')
        history = np.stack([np.mean(x[max(0, i-7):i+1], axis=0)
                            for i in range(len(x))]).astype('float32')
        base = ridge.predict(x).astype('float32')
        return torch.from_numpy(x), torch.from_numpy(history), torch.from_numpy(base)


class SharedNumericSMETimes(nn.Module):
    def __init__(self, input_dim=90):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 24), nn.GELU(),
                                     nn.Dropout(.15), nn.Linear(24, 8))
        self.reconstruct = nn.Linear(8, input_dim)
        self.heads = nn.ModuleDict({domain: nn.Sequential(nn.Linear(16, 16),
            nn.GELU(), nn.Dropout(.15), nn.Linear(16, 1)) for domain in DOMAINS})
        for head in self.heads.values():
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def forward(self, x, history, base, domain_index):
        z = self.encoder(x)
        h = self.encoder(history)
        fused = torch.cat((z, h), dim=1)
        correction = torch.zeros_like(base)
        for index, domain in enumerate(DOMAINS):
            value = .15*torch.tanh(self.heads[domain](fused).squeeze(1))
            correction = correction + torch.where(domain_index == index, value,
                                                   torch.zeros_like(value))
        return base+correction, self.reconstruct(z), z


def fit(cells, features, ids, seed=2026):
    torch.manual_seed(seed)
    prep = DomainPrep().fit(cells, features, ids)
    model = SharedNumericSMETimes()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.03)
    x_parts, h_parts, base_parts, domain_parts, label_parts, weight_parts = [], [], [], [], [], []
    for domain_index, domain in enumerate(DOMAINS):
        keys = [key for key in ids if cells[key].dataset == domain]
        for key in keys:
            cell = cells[key]
            keep = np.flatnonzero(cell.soh_available)[::5]
            x, history, base = prep.transform(cell, features[key])
            x_parts.append(x[keep])
            h_parts.append(history[keep])
            base_parts.append(base[keep])
            domain_parts.append(torch.full((len(keep),), domain_index))
            label_parts.append(torch.from_numpy(cell.soh[keep]))
            weight = np.where(cell.capacity_review_flag[keep], .25, 1.).astype('float32')
            weight_parts.append(torch.from_numpy(weight/weight.sum()/len(keys)/len(DOMAINS)))
    x, history, base, domain_index, label, weight = (torch.cat(parts) for parts in
        (x_parts, h_parts, base_parts, domain_parts, label_parts, weight_parts))
    for _ in range(EPOCHS):
        model.train()
        prediction, reconstruction, _ = model(x+.01*torch.randn_like(x), history,
                                              base, domain_index)
        error = (prediction-label).abs()
        huber = torch.where(error <= .025, .5*error.square(), .025*(error-.0125))
        loss = (huber*weight).sum() + .0002*(reconstruction-x).square().mean() + \
               .01*(prediction-base).square().mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
    model.eval()
    return prep, model


def predict(prep, model, cell, features):
    x, history, base = prep.transform(cell, features)
    domain = torch.full((len(x),), DOMAINS.index(cell.dataset))
    with torch.no_grad():
        return model(x, history, base, domain)[0].numpy()


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    manifest, cells = load_all()
    records = manifest['cells']
    features = {key: physical_features(cell) for key, cell in cells.items()}
    report = {'architecture': 'shared numeric encoder with four domain-specific heads',
              'not_original_llm_backbone': True, 'input_dimension': 90,
              'data_contract_version': 2, 'epochs': EPOCHS, 'domains': {}}
    for domain in DOMAINS:
        folder = OUT / domain.lower()
        folder.mkdir(parents=True, exist_ok=True)
        folds = []
        for group, target_train, val in outer_splits(records, domain):
            train = [row['cell_id'] for row in records if row['dataset'] != domain] + target_train
            prep, model = fit(cells, features, train)
            metric = score(cells, {key: predict(prep, model, cells[key], features[key])
                                   for key in val}, val)
            ridge_metric = score(cells, {key: prep.transform(cells[key], features[key])[2].numpy()
                                         for key in val}, val)
            folds.append({'held_out': group, 'train_cells': len(train),
                          'shared_smetimes': metric, 'ridge_common': ridge_metric})
            print(domain, group, 'shared', round(metric['cell_macro_mae'], 5),
                  'ridge', round(ridge_metric['cell_macro_mae'], 5), flush=True)
        (folder / 'group_cv.json').write_text(json.dumps(folds, indent=2), encoding='utf-8')
        report['domains'][domain] = {'group_cv_macro_mae': float(np.mean(
            [fold['shared_smetimes']['cell_macro_mae'] for fold in folds])),
            'ridge_common_macro_mae': float(np.mean(
                [fold['ridge_common']['cell_macro_mae'] for fold in folds])),
            'folds': len(folds)}
        print('SUMMARY', domain, report['domains'][domain], flush=True)
    ids = [row['cell_id'] for row in records]
    prep, model = fit(cells, features, ids)
    joblib.dump(prep, OUT / 'preprocessor.joblib')
    torch.save({'state_dict': model.state_dict(), 'domains': DOMAINS, 'epochs': EPOCHS,
                'training_cell_ids': ids}, OUT / 'model.pt')
    (OUT / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
