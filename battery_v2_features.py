"""Join v2 labels to causal, capacity-free signal features."""

import json
from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np


MANIFEST = Path('battery_data_v2/manifest.json')


@dataclass(frozen=True)
class CellFeatures:
    cell_id: str
    dataset: str
    group: str
    cycle: np.ndarray
    soh: np.ndarray
    soh_available: np.ndarray
    rul_cycles: np.ndarray
    rul_available: np.ndarray
    capacity_review_flag: np.ndarray
    raw: np.ndarray
    delta: np.ndarray
    history: np.ndarray
    reference_count: int
    endpoint_cycle: int | None
    reference_capacity_ah: float

    def matrix(self, variant):
        if variant == 'delta':
            return self.delta
        if variant == 'signal_history':
            return np.concatenate((self.raw, self.delta, self.history), axis=1)
        if variant == 'charge_signal_history':
            if self.dataset != 'XJTU' or self.raw.shape[1] != 46:
                raise ValueError('charge-only 23-column contract currently applies to XJTU')
            return np.concatenate((self.raw[:, :23], self.delta[:, :23],
                                   self.history[:, :23]), axis=1)
        raise ValueError(f'unknown feature variant: {variant}')


@dataclass(frozen=True)
class InferenceFeatures:
    cycle: np.ndarray
    raw: np.ndarray
    delta: np.ndarray
    history: np.ndarray
    reference_count: int

    def matrix(self, variant):
        if variant == 'delta':
            return self.delta
        if variant == 'signal_history':
            return np.concatenate((self.raw, self.delta, self.history), axis=1)
        if variant == 'charge_signal_history':
            if self.raw.shape[1] != 46:
                raise ValueError('charge-only 23-column contract currently applies to XJTU')
            return np.concatenate((self.raw[:, :23], self.delta[:, :23],
                                   self.history[:, :23]), axis=1)
        raise ValueError(f'unknown feature variant: {variant}')


def derive_signal_features(raw, reference_count):
    raw = np.asarray(raw, dtype='float32')
    if raw.ndim != 2 or not 1 <= reference_count <= len(raw):
        raise ValueError('expected [observations, features] and valid reference_count')
    if np.isinf(raw).any():
        raise ValueError('infinite signal feature')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        anchor = np.nanmedian(raw[:reference_count], axis=0)
        delta = raw-anchor
        history = np.stack([np.nanmean(delta[max(0, i-4):i+1], axis=0)
                            for i in range(len(delta))])
    return delta.astype('float32'), history.astype('float32')


def make_inference_features(raw, cycle, reference_count):
    cycle = np.asarray(cycle, dtype='int64')
    raw = np.asarray(raw, dtype='float32')
    if len(cycle) != len(raw) or np.any(np.diff(cycle) <= 0):
        raise ValueError('cycle must align with signal observations and increase strictly')
    delta, history = derive_signal_features(raw, reference_count)
    return InferenceFeatures(cycle, raw, delta, history, reference_count)


def load_manifest(path=MANIFEST):
    manifest = json.loads(Path(path).read_text(encoding='utf-8'))
    if manifest['schema_version'] != 2:
        raise ValueError('expected v2 battery data contract')
    return manifest


def load_cell(record):
    with np.load(record['labels_file']) as labels:
        cycle = labels['cycle'].astype('int64')
        capacity = labels['capacity_ah']
        soh = labels['soh'].astype('float32')
        soh_available = labels['soh_available'].astype(bool)
        rul_cycles = labels['rul_cycles'].astype('float32')
        rul_available = labels['rul_available'].astype(bool)
        review = labels['capacity_review_flag'].astype(bool)

    if record['dataset'] == 'NASA':
        from battery_raw_mvp import load_raw_battery
        battery = load_raw_battery(Path(record['measurement_source']).parent,
                                   record['cell_id'], np.arange(0, 1001, 100), 1000)
        raw = battery.features.astype('float32')
        feature_capacity = battery.capacity_ah
    else:
        with np.load(record['feature_source']) as source:
            raw = source[record['feature_array']].astype('float32')
            feature_capacity = source['capacity'].astype('float32')
            feature_cycle = source['observation'] if record['dataset'] == 'CALCE' else source['cycle']
            if not np.array_equal(feature_cycle.astype('int64'), cycle):
                raise ValueError(f"{record['cell_id']}: source/contract cycle mismatch")
    if raw.ndim != 2 or len(raw) != len(cycle):
        raise ValueError(f"{record['cell_id']}: source/contract feature shape mismatch")
    if not np.allclose(feature_capacity, capacity, rtol=1e-5, atol=1e-5):
        raise ValueError(f"{record['cell_id']}: source/contract capacity mismatch")
    if np.isinf(raw).any():
        raise ValueError(f"{record['cell_id']}: infinite signal feature")

    reference_count = record['reference_measurement_count']
    delta, history = derive_signal_features(raw, reference_count)
    return CellFeatures(record['cell_id'], record['dataset'], record['group'],
                        cycle, soh, soh_available, rul_cycles, rul_available,
                        review, raw, delta, history, reference_count,
                        record['endpoint_cycle'], record['reference_capacity_ah'])


def load_all(path=MANIFEST):
    manifest = load_manifest(path)
    cells = {record['cell_id']: load_cell(record) for record in manifest['cells']}
    if len(cells) != len(manifest['cells']):
        raise ValueError('duplicate cell identifier')
    return manifest, cells
