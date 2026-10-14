"""Bounded linear MIT RUL experiment with and without censor lower-bound loss.

Both branches share identical signal/age inputs, scaler, 3000-cycle output
bound, and whole-batch folds. The censored branch adds only a one-sided loss:
its RUL must be at least as long as the observed follow-up. No endpoint is
fabricated. This is exploratory development work, not a calibrated survival
model or a replacement for the selected RUL route.
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
from evaluate_mit_censored_aft_v2 import training_arrays
from train_rul_v2 import MIN_HISTORY, assess, signal_features
from train_soh_v2 import outer_splits


OUT = Path('battery_models_v2/rul/mit/censored_linear_diagnostic.json')
MAX_RUL_CYCLES = 3000.
EPOCHS = 200


class BoundedLinearRUL(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.linear = nn.Linear(width, 1)

    def forward(self, x):
        return np.log1p(MAX_RUL_CYCLES)*torch.sigmoid(self.linear(x).squeeze(1))


def objective(mu, lower_bound, event, weight, use_censor):
    log_time = torch.log1p(lower_bound)
    observed_error = (mu-log_time).square()
    censor_error = torch.relu(log_time-mu).square() if use_censor else torch.zeros_like(mu)
    return torch.sum(weight*torch.where(event, observed_error, censor_error))


def fit(cells, ids, use_censor, seed=2026):
    torch.manual_seed(seed)
    x, bound, event, weight = training_arrays(cells, ids)
    keep = np.ones(len(event), dtype=bool) if use_censor else event
    x, bound, event, weight = x[keep], bound[keep], event[keep], weight[keep]
    weight = weight/weight.sum()
    imputer = SimpleImputer(strategy='median', keep_empty_features=True).fit(x)
    scaler = RobustScaler(quantile_range=(10, 90)).fit(imputer.transform(x))
    scaler.scale_ = np.maximum(scaler.scale_, .02)
    x = np.clip(scaler.transform(imputer.transform(x)), -6, 6)
    feature = torch.as_tensor(x, dtype=torch.float32)
    bound = torch.as_tensor(bound, dtype=torch.float32)
    event = torch.as_tensor(event, dtype=torch.bool)
    weight = torch.as_tensor(weight, dtype=torch.float32)
    model = BoundedLinearRUL(feature.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.05)
    for _ in range(EPOCHS):
        model.train()
        mu = model(feature)
        loss = objective(mu, bound, event, weight, use_censor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    return imputer, scaler, model, {'training_rows': len(x),
                                    'censored_rows': int((~event).sum()),
                                    'final_objective': float(loss.detach())}


def predict(bundle, cell):
    imputer, scaler, model, _ = bundle
    x = signal_features(cell)
    x = np.clip(scaler.transform(imputer.transform(x)), -6, 6)
    with torch.no_grad():
        mu = model(torch.as_tensor(x, dtype=torch.float32)).numpy()
    estimate = np.maximum(0, np.expm1(mu))
    estimate[:MIN_HISTORY] = np.nan
    return estimate


def summarize(folds, name):
    values = [fold[name] for fold in folds]
    return {'equal_fold_cell_macro_mae_cycles': float(np.mean([
                value['cell_macro_mae_cycles'] for value in values])),
            'worst_cell_mae_cycles': float(max(cell['mae_cycles'] for value in values
                                                for cell in value['by_cell'])),
            'phase_equal_fold_macro_mae_cycles': {phase: float(np.mean([
                value['phase_macro_mae_cycles'][phase] for value in values
                if value['phase_macro_mae_cycles'][phase] is not None]))
                for phase in ('early', 'middle', 'late')}}


def evaluate():
    torch.set_num_threads(4)
    contract, all_cells = load_all()
    cells = {key: cell for key, cell in all_cells.items() if cell.dataset == 'MIT'}
    if len(cells) != 134 or sum(cell.endpoint_cycle is None for cell in cells.values()) != 12:
        raise ValueError('MIT censor cohort differs from audited 134/12-cell contract')
    prior = {row['held_out']: row for row in json.loads(Path(
        'battery_models_v2/rul/mit/group_cv.json').read_text(encoding='utf-8'))}
    selected = json.loads(Path('battery_models_v2/rul/mit/training_manifest.json')
                          .read_text(encoding='utf-8'))['selected_model']
    folds = []
    for group, train, val in outer_splits(contract['cells'], 'MIT'):
        observed_val = [key for key in val if cells[key].endpoint_cycle is not None]
        observed_train = [key for key in train if cells[key].endpoint_cycle is not None]
        median_life = float(np.median([cells[key].endpoint_cycle for key in observed_train]))
        candidates = {}
        training = {}
        for name, use_censor in (('observed_only', False), ('with_censor_bounds', True)):
            model = fit(cells, train, use_censor)
            prediction = {key: predict(model, cells[key]) for key in observed_val}
            candidates[name] = assess(cells, observed_val, prediction, median_life)
            training[name] = model[3]
        folds.append({'held_out': group, **candidates,
                      'selected_rul': prior[group][selected],
                      'training': training})
        print(group, 'observed', round(candidates['observed_only']['cell_macro_mae_cycles'], 1),
              'censored', round(candidates['with_censor_bounds']['cell_macro_mae_cycles'], 1),
              'selected', round(prior[group][selected]['cell_macro_mae_cycles'], 1), flush=True)
    return {'status': 'development-only bounded linear MIT RUL censoring ablation',
            'method': f'linear log-RUL, 0-{int(MAX_RUL_CYCLES)} cycle bound, '
                      f'{EPOCHS} fixed epochs, cell-balanced weights, ten-row stride',
            'warning': 'bounded linear family was proposed after seeing an unstable '
                       'nonlinear AFT run; development CV is not independent evidence',
            'summary': {name: summarize(folds, name) for name in
                        ('observed_only', 'with_censor_bounds', 'selected_rul')},
            'folds': folds}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    current = evaluate()
    if args.verify:
        stored = json.loads(OUT.read_text(encoding='utf-8'))
        for name in ('observed_only', 'with_censor_bounds', 'selected_rul'):
            if not np.isclose(current['summary'][name]['equal_fold_cell_macro_mae_cycles'],
                              stored['summary'][name]['equal_fold_cell_macro_mae_cycles'],
                              atol=1e-5, rtol=1e-8):
                raise ValueError(f'{name}: stored linear MIT score differs')
        print('REPRODUCED', OUT)
    else:
        if OUT.exists():
            raise FileExistsError(OUT)
        OUT.write_text(json.dumps(current, indent=2), encoding='utf-8')
        print('WROTE', OUT)
