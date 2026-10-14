"""TRAIN-label-only sustained transition extraction; never an inference input."""
import numpy as np
import torch


KNEE_RULES = {'bin_cycles': 20., 'minimum_bins': 9, 'minimum_bin_points': 3,
              'minimum_relative_fit_gain': .5, 'minimum_accelerated_drop_soh': .02,
              'near_best_mse_factor': 1.1, 'maximum_localization_fraction': .2}


def training_knee_target(cell, origin):
    """A masked horizon interval, not a fabricated endpoint/knee for linear cells.

    Fixed source-cycle median bins suppress isolated spikes. A continuous hinge
    fit is used only to extract a supervised label, not to forecast any cell.
    Require sustained slopes, fit gain, observable acceleration and localization.
    """
    x, y = np.asarray(cell['label_cycle']), np.asarray(cell['soh'])
    take = (x > origin) & np.isfinite(x) & np.isfinite(y) & (y > 0)
    x, y = x[take], y[take]
    missing = {'valid': False, 'reason': 'insufficient sustained coverage', 'lower': 0., 'upper': 0.}
    if len(x) < 27 or np.any(np.diff(x) <= 0):
        return missing
    bins = np.floor((x - x[0]) / KNEE_RULES['bin_cycles']).astype(int)
    bx, by = [], []
    for b in np.unique(bins):
        mask = bins == b
        if mask.sum() >= KNEE_RULES['minimum_bin_points']:
            bx.append(np.median(x[mask])); by.append(np.median(y[mask]))
    x, y = np.asarray(bx), np.asarray(by)
    if len(x) < KNEE_RULES['minimum_bins'] or np.max(np.diff(x)) > 2 * KNEE_RULES['bin_cycles']:
        return missing
    t = (x - origin) / 100.
    linear = np.column_stack((np.ones(len(t)), t))
    linear_fit = np.linalg.lstsq(linear, y, rcond=None)[0]
    linear_mse = np.mean((linear @ linear_fit - y) ** 2)
    if linear_mse < 1e-10:
        return {**missing, 'reason': 'no observed nonlinear transition'}
    fits = []
    for knee in t[3:-3]:
        design = np.column_stack((linear, np.maximum(t - knee, 0)))
        coef = np.linalg.lstsq(design, y, rcond=None)[0]
        mse = float(np.mean((design @ coef - y) ** 2))
        # Never supervise a slowing or recovering curve as accelerated decline.
        if coef[1] <= 0 and coef[2] < 0 and -coef[2] * (t[-1] - t[0]) >= KNEE_RULES['minimum_accelerated_drop_soh']:
            fits.append((mse, float(knee)))
    if not fits:
        return {**missing, 'reason': 'no sustained accelerated decline'}
    fits.sort()
    best = fits[0][0]
    gain = 1 - best / linear_mse
    if gain < KNEE_RULES['minimum_relative_fit_gain']:
        return {**missing, 'reason': 'transition fit gain insufficient'}
    plausible = [k for mse, k in fits if mse <= max(best * KNEE_RULES['near_best_mse_factor'], 1e-10)]
    lower, upper = min(plausible), max(plausible)
    if upper - lower > KNEE_RULES['maximum_localization_fraction'] * (t[-1] - t[0]):
        return {**missing, 'reason': 'transition location ambiguous'}
    return {'valid': True, 'reason': 'localized sustained acceleration',
            'lower': lower, 'upper': upper, 'fit_gain': float(gain),
            'binned_points': len(x), 'units': 'future horizon / 100 source cycles'}


def knee_interval_loss(parameters, lower, upper, valid):
    """Log-horizon interval loss on the SAME head parameters used for SOH.

    Invalid labels have zero gradient; log compression avoids long-lived cells
    dominating. Targets are training-only and do not enter forward inference.
    """
    knee = 5 * torch.nn.functional.softplus(parameters[:, 3])
    log_knee = torch.log1p(knee)
    distance = torch.relu(torch.log1p(lower) - log_knee) + torch.relu(log_knee - torch.log1p(upper))
    return torch.nn.functional.smooth_l1_loss(distance, torch.zeros_like(distance), reduction='none') * valid
