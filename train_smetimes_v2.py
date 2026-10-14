"""Masked v2 SOH comparison using the repository's numeric SMETimes derivative.

This is not the original language-model SMETimes backbone. Its compact learned
encoder is evaluated against the non-neural v2 SOH baselines on identical groups.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler

from battery_smetimes_regularized import RegularizedSMETimes
from battery_v2_features import load_all
from train_soh_v2 import outer_splits, score


OUT = Path('battery_models_v2/soh_numeric_smetimes')
EPOCHS = 20
INPUT_VARIANT = 'signal_history'


class Preprocessor:
    def fit(self, cells, ids):
        x = np.concatenate([cells[key].matrix(INPUT_VARIANT) for key in ids])
        self.imputer = SimpleImputer(strategy='median', keep_empty_features=True).fit(x)
        x = self.imputer.transform(x)
        self.scaler = RobustScaler(quantile_range=(10, 90)).fit(x)
        self.scaler.scale_ = np.maximum(self.scaler.scale_, .02)
        supervised_x, supervised_y, weights = [], [], []
        for key in ids:
            cell = cells[key]
            keep = cell.soh_available
            supervised_x.append(cell.matrix(INPUT_VARIANT)[keep])
            supervised_y.append(cell.soh[keep])
            w = np.where(cell.capacity_review_flag[keep], .25, 1.).astype('float64')
            weights.append(w / w.sum() / len(ids))
        sx = np.concatenate(supervised_x)
        sy = np.concatenate(supervised_y)
        weights = np.concatenate(weights)
        sx = np.clip(self.scaler.transform(self.imputer.transform(sx)), -6, 6)
        self.ridge = Ridge(alpha=10).fit(sx, sy, sample_weight=weights*len(sx))
        return self

    def transform(self, cell):
        x = np.clip(self.scaler.transform(self.imputer.transform(cell.matrix(INPUT_VARIANT))),
                    -6, 6).astype('float32')
        history = np.stack([np.mean(x[max(0, i-7):i+1], axis=0)
                            for i in range(len(x))]).astype('float32')
        base = self.ridge.predict(x).astype('float32')
        return tuple(torch.from_numpy(v) for v in
                     (x, history, np.zeros((len(x), 8), dtype='float32'), base))


def fit(cells, ids, seed=2026):
    torch.manual_seed(seed)
    prep = Preprocessor().fit(cells, ids)
    data = {key: prep.transform(cells[key]) for key in ids}
    model = RegularizedSMETimes(False, feature_dim=data[ids[0]][0].shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.03)
    x, history, semantic, base, label, weight = [], [], [], [], [], []
    for key in ids:
        cell = cells[key]
        keep = np.flatnonzero(cell.soh_available)[::5]
        if not len(keep):
            continue
        parts = data[key]
        for holder, tensor in zip((x, history, semantic, base), parts):
            holder.append(tensor[keep])
        label.append(torch.from_numpy(cell.soh[keep]))
        w = np.where(cell.capacity_review_flag[keep], .25, 1.).astype('float32')
        weight.append(torch.from_numpy(w / w.sum() / len(ids)))
    x, history, semantic, base = (torch.cat(item) for item in
                                  (x, history, semantic, base))
    label, weight = torch.cat(label), torch.cat(weight)
    for epoch in range(EPOCHS):
        model.train()
        pred, reconstructed, _ = model(x+.01*torch.randn_like(x), history,
                                       semantic, base)
        error = (pred-label).abs()
        huber = torch.where(error <= .025, .5*error.square(), .025*(error-.0125))
        loss = (huber*weight).sum() + .0002*(reconstructed-x).square().mean() + \
               .01*(pred-base).square().mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
    model.eval()
    return prep, model


def predict(prep, model, cell):
    with torch.no_grad():
        return model(*prep.transform(cell))[0].numpy()


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    manifest, cells = load_all()
    report = {'architecture': 'numeric RegularizedSMETimes derivative; not full LLM backbone',
              'input_variant': INPUT_VARIANT,
              'epochs': EPOCHS, 'data_contract_version': 2, 'domains': {}}
    for domain in ('MIT', 'XJTU'):
        folder = OUT / domain.lower()
        folder.mkdir(parents=True, exist_ok=True)
        folds = []
        for group, train, val in outer_splits(manifest['cells'], domain):
            prep, model = fit(cells, train)
            predictions = {key: predict(prep, model, cells[key]) for key in val}
            metric = score(cells, predictions, val)
            folds.append({'held_out': group, 'score': metric})
            print(domain, group, round(metric['cell_macro_mae'], 5), flush=True)
        (folder / 'group_cv.json').write_text(json.dumps(folds, indent=2), encoding='utf-8')
        ids = [row['cell_id'] for row in manifest['cells'] if row['dataset'] == domain]
        prep, model = fit(cells, ids)
        joblib.dump(prep, folder / 'preprocessor.joblib')
        torch.save({'state_dict': model.state_dict(), 'feature_dim': model.encoder[0].in_features,
                    'epochs': EPOCHS, 'training_cell_ids': ids}, folder / 'model.pt')
        macro = float(np.mean([row['score']['cell_macro_mae'] for row in folds]))
        report['domains'][domain] = {'group_cv_macro_mae': macro,
                                     'training_cells': len(ids), 'folds': len(folds)}
        print('SELECTED', domain, round(macro, 5), flush=True)
    (OUT / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
