"""Development-only MIT right-censored RUL survival-likelihood experiment.

An observed endpoint contributes a log-normal event likelihood. A censored
cell contributes only P(remaining life > observed lower bound), never a
fabricated end-of-life label. All preprocessing is fitted on outer training
batches, and only observed-endpoint validation cells get RUL MAE scores.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

from battery_v2_features import load_all
from train_rul_v2 import MIN_HISTORY, assess, signal_features
from train_soh_v2 import outer_splits


OUT = Path('battery_models_v2/rul/mit/censored_aft_diagnostic.json')
PRIOR = Path('battery_models_v2/rul/mit/group_cv.json')
EPOCHS = 40
STRIDE = 10
SIGMA = .8  # Fixed before the held-out-batch comparison; no outer-fold tuning.


class LogNormalAFT(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, 32), nn.GELU(), nn.Dropout(.1),
                                 nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(1)


def censored_likelihood_loss(mu, lower_bound, event, weight, sigma=SIGMA):
    """Weighted negative log likelihood of T=RUL+1 or its censoring lower bound."""
    log_time = torch.log1p(lower_bound)
    z = (log_time-mu)/sigma
    observed_nll = log_time + np.log(sigma) + .5*np.log(2*np.pi) + .5*z.square()
    censored_nll = -torch.special.log_ndtr(-z)
    return torch.sum(weight*torch.where(event, observed_nll, censored_nll))


def training_arrays(cells, ids):
    features, bounds, events, weights = [], [], [], []
    for key in ids:
        cell = cells[key]
        index = np.flatnonzero(np.arange(len(cell.cycle)) >= MIN_HISTORY)
        if cell.endpoint_cycle is not None:
            index = index[cell.cycle[index] <= cell.endpoint_cycle]
            lower = cell.endpoint_cycle-cell.cycle[index]
            observed = np.ones(len(index), dtype=bool)
        else:
            # Last observed cycle is a lower bound on end-of-life, not an event.
            lower = cell.cycle[-1]-cell.cycle[index]
            observed = np.zeros(len(index), dtype=bool)
        if not len(index):
            continue
        index, lower, observed = index[::STRIDE], lower[::STRIDE], observed[::STRIDE]
        features.append(signal_features(cell)[index])
        bounds.append(lower.astype('float32'))
        events.append(observed)
        weight = np.where(cell.capacity_review_flag[index], .25, 1.).astype('float64')
        weights.append(weight/weight.sum()/len(ids))
    return (np.concatenate(features), np.concatenate(bounds),
            np.concatenate(events), np.concatenate(weights))


def fit(cells, ids, seed=2026):
    torch.manual_seed(seed)
    x, bound, event, weight = training_arrays(cells, ids)
    imputer = SimpleImputer(strategy='median', keep_empty_features=True).fit(x)
    scaler = RobustScaler(quantile_range=(10, 90)).fit(imputer.transform(x))
    scaler.scale_ = np.maximum(scaler.scale_, .02)
    normalized = np.clip(scaler.transform(imputer.transform(x)), -6, 6)
    features = torch.as_tensor(normalized, dtype=torch.float32)
    bound = torch.as_tensor(bound, dtype=torch.float32)
    event = torch.as_tensor(event, dtype=torch.bool)
    weight = torch.as_tensor(weight, dtype=torch.float32)
    model = LogNormalAFT(features.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.03)
    for _ in range(EPOCHS):
        model.train()
        mu = model(features)
        loss = censored_likelihood_loss(mu, bound, event, weight)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
    model.eval()
    return imputer, scaler, model, {'training_rows': len(x),
                                    'censored_training_rows': int((~event).sum()),
                                    'observed_training_rows': int(event.sum())}


def predict(bundle, cell):
    imputer, scaler, model, _ = bundle
    x = signal_features(cell)
    x = np.clip(scaler.transform(imputer.transform(x)), -6, 6)
    with torch.no_grad():
        mu = model(torch.as_tensor(x, dtype=torch.float32)).numpy()
    result = np.maximum(0, np.expm1(np.clip(mu, -10, 12)))
    result[:MIN_HISTORY] = np.nan
    return result


def summary(folds, name):
    values = [fold[name] for fold in folds]
    return {'equal_fold_cell_macro_mae_cycles': float(np.mean([
                row['cell_macro_mae_cycles'] for row in values])),
            'worst_cell_mae_cycles': float(max(cell['mae_cycles'] for row in values
                                                for cell in row['by_cell'])),
            'phase_equal_fold_macro_mae_cycles': {phase: float(np.mean([
                row['phase_macro_mae_cycles'][phase] for row in values
                if row['phase_macro_mae_cycles'][phase] is not None]))
                for phase in ('early', 'middle', 'late')}}


def evaluate():
    torch.set_num_threads(4)
    contract, all_cells = load_all()
    cells = {key: cell for key, cell in all_cells.items() if cell.dataset == 'MIT'}
    records = [row for row in contract['cells'] if row['dataset'] == 'MIT']
    if len(cells) != 134 or sum(cell.endpoint_cycle is None for cell in cells.values()) != 12:
        raise ValueError('MIT censoring cohort differs from audited 134/12-cell contract')
    prior = {fold['held_out']: fold for fold in json.loads(
        PRIOR.read_text(encoding='utf-8'))}
    selected_model = json.loads(Path('battery_models_v2/rul/mit/training_manifest.json')
                                .read_text(encoding='utf-8'))['selected_model']
    if selected_model not in ('direct', 'assisted'):
        raise ValueError('unexpected selected MIT RUL route')
    folds = []
    for group, training, validation in outer_splits(contract['cells'], 'MIT'):
        bundle = fit(cells, training)
        observed_val = [key for key in validation if cells[key].endpoint_cycle is not None]
        censored_val = [key for key in validation if cells[key].endpoint_cycle is None]
        estimates = {key: predict(bundle, cells[key]) for key in validation}
        observed_train = [key for key in training if cells[key].endpoint_cycle is not None]
        median_life = float(np.median([cells[key].endpoint_cycle for key in observed_train]))
        metric = assess(cells, observed_val, estimates, median_life)
        censored_checks = []
        for key in censored_val:
            cell = cells[key]
            index = np.arange(MIN_HISTORY, len(cell.cycle))
            lower_bound = cell.cycle[-1]-cell.cycle[index]
            censored_checks.append({'cell_id': key,
                'fraction_predicted_below_known_survival_bound': float(np.mean(
                    estimates[key][index] < lower_bound))})
        folds.append({'held_out': group, 'censored_aft': metric,
                      'selected_rul': prior[group][selected_model],
                      'direct_rul': prior[group]['direct'],
                      'training': bundle[3],
                      'censored_validation_checks': censored_checks})
        print(group, 'AFT', round(metric['cell_macro_mae_cycles'], 1),
              'selected', round(folds[-1]['selected_rul']['cell_macro_mae_cycles'], 1),
              'censored train', bundle[3]['censored_training_rows'], flush=True)
    result = {'status': 'development-only censored-likelihood MIT RUL diagnostic',
              'method': f'fixed-sigma log-normal AFT; {EPOCHS} epochs; every {STRIDE}th '
                        'eligible row; cell-balanced weights; signal and age only',
              'warning': 'no fabricated endpoint for censored cells; only 3 batches; '
                         'not an independent accuracy or interval claim',
              'summary': {name: summary(folds, name) for name in
                          ('censored_aft', 'selected_rul', 'direct_rul')},
              'folds': folds}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    result = evaluate()
    if args.verify:
        saved = json.loads(OUT.read_text(encoding='utf-8'))
        for name in ('censored_aft', 'selected_rul', 'direct_rul'):
            if not np.isclose(result['summary'][name]['equal_fold_cell_macro_mae_cycles'],
                              saved['summary'][name]['equal_fold_cell_macro_mae_cycles'],
                              atol=1e-5, rtol=1e-8):
                raise ValueError(f'{name}: saved MIT AFT score differs')
        print('REPRODUCED', OUT)
    else:
        if OUT.exists():
            raise FileExistsError(OUT)
        OUT.write_text(json.dumps(result, indent=2), encoding='utf-8')
        print('WROTE', OUT)
