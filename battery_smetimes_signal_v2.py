"""Label-free physical-sequence adapter for the original SMETimes backbone.

The audited raw-feature arrays are converted to two five-point slots:
charge then diagnostic discharge. NASA/CALCE/MIT have no accepted charge
prefix, so its values and availability masks are zero; CALCE has no measured
temperature. This adapter never receives capacity, SOH, RUL, or endpoint data.
"""

import numpy as np


CHANNELS = ('voltage_v', 'current_magnitude_a', 'temperature_c',
            'voltage_available', 'current_available', 'temperature_available',
            'charge_stage', 'discharge_stage', 'relative_stage_time')
CONTEXT = ('charge_prefix_seconds', 'discharge_prefix_seconds',
           'charge_available', 'temperature_available')
SCHEMAS = {'NASA': (44, 1000.), 'CALCE': (29, 1000.),
           'MIT': (51, 300.), 'XJTU': (46, 120.)}


def _five_positions(block):
    """Interpolate one channel inside its measured prefix, retaining availability."""
    if block is None:
        return None, None
    block = np.asarray(block, dtype='float32')
    n, width = block.shape
    source = np.linspace(0., 1., width)
    target = np.linspace(0., 1., 5)
    value = np.zeros((n, 5), dtype='float32')
    available = np.zeros((n, 5), dtype='float32')
    for index, row in enumerate(block):
        valid = np.isfinite(row)
        if valid.sum() < 2:
            continue
        value[index] = np.interp(target, source[valid], row[valid]).astype('float32')
        available[index] = 1.
    return value, available


def _stage(raw, voltage, current, temperature, stage):
    n = len(raw)
    block = np.zeros((n, 5, len(CHANNELS)), dtype='float32')
    if voltage is None:
        return block
    for channel, source in enumerate((voltage, current, temperature)):
        if source is None:
            continue
        value, availability = _five_positions(raw[:, source])
        if channel == 1:
            value = np.abs(value)
        block[:, :, channel] = value
        block[:, :, channel+3] = availability
    if not np.all(block[:, :, 3]) or not np.all(block[:, :, 4]):
        raise ValueError('accepted diagnostic stage needs usable voltage and current')
    block[:, :, 6 if stage == 'charge' else 7] = 1.
    block[:, :, 8] = np.linspace(0., 1., 5, dtype='float32')
    return block


def build_sequence(dataset, raw):
    """Return [observations, 10, 9] physical sequence and [observations, 4] context."""
    dataset = str(dataset).upper()
    if dataset not in SCHEMAS:
        raise ValueError(f'unsupported SMETimes battery schema: {dataset}')
    raw = np.asarray(raw, dtype='float32')
    width, seconds = SCHEMAS[dataset]
    if raw.ndim != 2 or raw.shape[1] != width or not len(raw) or np.isinf(raw).any():
        raise ValueError(f'{dataset}: invalid raw signal array for SMETimes')
    if dataset == 'XJTU':
        charge = _stage(raw, slice(0, 5), slice(5, 10), slice(10, 15), 'charge')
        discharge = _stage(raw, slice(23, 28), slice(28, 33), slice(33, 38),
                           'discharge')
    else:
        charge = _stage(raw, None, None, None, 'charge')
        discharge = _stage(raw, slice(0, 11), slice(11, 22),
                           slice(22, 33) if dataset != 'CALCE' else None,
                           'discharge')
    sequence = np.concatenate((charge, discharge), axis=1)
    context = np.column_stack((
        np.full(len(raw), seconds if dataset == 'XJTU' else 0., dtype='float32'),
        np.full(len(raw), seconds, dtype='float32'),
        np.full(len(raw), float(dataset == 'XJTU'), dtype='float32'),
        np.any(sequence[:, :, 5] > 0, axis=1).astype('float32'),
    )).astype('float32')
    if sequence.shape != (len(raw), 10, len(CHANNELS)) or not np.isfinite(sequence).all():
        raise ValueError(f'{dataset}: nonfinite SMETimes sequence')
    return sequence, context
