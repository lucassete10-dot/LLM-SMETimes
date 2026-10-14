"""Compact, label-free future SOH predictor and portable INT8 inference."""
import numpy as np
import torch
from torch import nn
from battery_smetimes.forecast import query_matrix
from battery_smetimes.research_baselines import signal_prefix

# Fixed physical signal columns, selected before looking at new test predictions.
RAW_COLUMNS = {'MIT': [0, 5, 10, 33, 35, 36, 40],
               'XJTU': [0, 4, 23, 27, 38, 41, 45]}


def compact_input(cell, origin, future):
    domain = cell['metadata']['dataset']
    width = cell['raw'].shape[1]
    prefix = signal_prefix(cell, origin)
    columns = [block * width + column for block in range(4) for column in RAW_COLUMNS[domain]]
    columns.extend(range(4 * width, len(prefix)))
    value = query_matrix(prefix, origin, future)[:, columns + list(range(len(prefix), len(prefix) + 3))]
    if value.shape[1] != 40:
        raise ValueError('expected exactly 40 compact features')
    return value


def pairs(cells, ids):
    parts, targets, weights = [], [], []
    for key in ids:
        cell = cells[key]
        take = np.flatnonzero(np.isfinite(cell['soh']) & (cell['soh'] > 0) & (cell['label_cycle'] > 100))
        if len(take) < 2:
            raise ValueError(f'{key}: insufficient targets')
        if len(take) > 128:
            take = take[np.unique(np.linspace(0, len(take) - 1, 128).round().astype(int))]
        parts.append(compact_input(cell, 100, cell['label_cycle'][take]))
        targets.append(cell['soh'][take])
        weights.append(np.full(len(take), 1 / len(ids) / len(take)))
    return np.concatenate(parts), np.concatenate(targets), np.concatenate(weights)


class Scale:
    def fit(self, x, weights):
        self.fill = np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0
                              for c in x.T], dtype=np.float32)
        value = np.where(np.isfinite(x), x, self.fill).astype(np.float64)
        self.mean = np.average(value, axis=0, weights=weights).astype(np.float32)
        variance = np.average((value - self.mean) ** 2, axis=0, weights=weights)
        self.std = np.where(variance > 1e-12, np.sqrt(variance), 1).astype(np.float32)
        return self

    def transform(self, x):
        if np.isinf(x).any():
            raise ValueError('infinite input')
        value = np.where(np.isfinite(x), x, self.fill)
        return np.clip((value - self.mean) / self.std, -8, 8).astype(np.float32)


class TinySOH(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(40, hidden[0]), nn.ReLU(),
                                    nn.Linear(hidden[0], hidden[1]), nn.ReLU(),
                                    nn.Linear(hidden[1], 1))
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, x):
        return 1.0 + self.layers(x).squeeze(-1)


def quantize(model, calibration):
    value = calibration.copy()
    result = []
    for index, layer in enumerate([model.layers[0], model.layers[2], model.layers[4]]):
        weight = layer.weight.detach().cpu().numpy()
        scale = np.maximum(np.max(np.abs(weight), axis=1) / 127, 1e-10)
        qw = np.clip(np.rint(weight / scale[:, None]), -127, 127).astype(np.int8)
        activation_scale = max(float(np.max(np.abs(value))) / 127, 1e-10)
        bias = layer.bias.detach().cpu().numpy()
        result.append({'weight': qw, 'weight_scale': scale.astype(np.float32),
                       'input_scale': activation_scale, 'bias': bias})
        qx = np.clip(np.rint(value / activation_scale), -127, 127).astype(np.int32)
        value = (qx @ qw.astype(np.int32).T) * (activation_scale * scale) + bias
        if index < 2:
            value = np.maximum(value, 0)
    return result


def predict_int8(layers, x):
    for index, layer in enumerate(layers):
        qx = np.clip(np.rint(x / layer['input_scale']), -127, 127).astype(np.int32)
        x = (qx @ layer['weight'].astype(np.int32).T).astype(np.float32)
        x = x * (np.float32(layer['input_scale']) * layer['weight_scale']) + layer['bias']
        if index < 2:
            x = np.maximum(x, 0)
    return 1 + x[:, 0]


def predict(bundle, cell, origin, future):
    if cell.get('physical_family') != bundle['physical_family']:
        raise ValueError('battery physical family is incompatible with this model')
    if origin != 100:
        raise ValueError('this MCU model was trained only at source cycle 100')
    x = bundle['scale'].transform(compact_input(cell, origin, future))
    if bundle.get('quantized'):
        return predict_int8(bundle['int8_layers'], x)
    with torch.no_grad():
        return bundle['network'](torch.from_numpy(x)).cpu().numpy()
