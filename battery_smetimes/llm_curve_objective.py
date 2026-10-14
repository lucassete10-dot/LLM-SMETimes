"""Source-cycle-aware supervision and validation for the LLM-only MVP."""
import numpy as np
import torch


def training_origin_target(cell, origin):
    """Optional exact-origin label for TRAIN-only auxiliary supervision.

    Never used by model_input/predict. Missing exact-origin measurements are
    not filled from future labels or assigned a fabricated SOH target.
    """
    x,y=np.asarray(cell['label_cycle']),np.asarray(cell['soh'])
    take=np.flatnonzero((x <= origin) & np.isclose(x,origin,rtol=0,atol=1e-8) & np.isfinite(y) & (y>0))
    return (float(y[take[-1]]),1.) if len(take) else (0.,0.)


def curve_loss(prediction, truth, cycles, valid, stage_weight=0., tail_weight=0., rate_weight=0.):
    """Per-cell losses; future coordinates/labels are TRAINING supervision only.

    End-window averages avoid differentiating isolated capacity spikes. Windows
    and stage boundaries depend on training target coverage, never inference.
    """
    point = ((prediction - truth).square() * valid).sum(1) / valid.sum(1)
    stages, tails, rates = [], [], []
    for i in range(len(prediction)):
        mask = valid[i].bool()
        x, y, p = cycles[i][mask], truth[i][mask], prediction[i][mask]
        span = x[-1] - x[0]
        tail = x >= x[0] + .8 * span
        tails.append((p[tail] - y[tail]).square().mean())
        errors = []
        for j in range(3):
            inside = (x >= x[0] + j * span / 3) & (x <= x[0] + (j + 1) * span / 3)
            xx, yy, pp = x[inside], y[inside], p[inside]
            if len(xx) < 2:
                continue
            width = max(1., min(5., float((xx[-1] - xx[0]) / 4)))
            left, right = xx <= xx[0] + width, xx >= xx[-1] - width
            errors.append(((pp[left].mean() - pp[right].mean()) -
                           (yy[left].mean() - yy[right].mean())).square())
        stages.append(torch.stack(errors).mean() if errors else point[i] * 0)
        local_errors = []
        if rate_weight:
            # Coarse sustained slopes, never raw adjacent/high-order differences.
            # Target coverage defines TRAIN supervision only, not inference time.
            window = max(20., float(span) * .1)
            starts = torch.linspace(x[0], x[-1] - window, 9, device=x.device) if span >= window else []
            for start in starts:
                left = (x >= start) & (x <= start + window / 4)
                right = (x >= start + 3 * window / 4) & (x <= start + window)
                if left.sum() < 2 or right.sum() < 2:
                    continue
                gap = x[right].mean() - x[left].mean()
                observed_drop = y[left].median() - y[right].median()
                predicted_drop = p[left].mean() - p[right].mean()
                local_errors.append(((predicted_drop - observed_drop) * 100 / gap).square())
        rates.append(torch.stack(local_errors).mean() if local_errors else point[i] * 0)
    stage, tail = torch.stack(stages), torch.stack(tails)
    rate = torch.stack(rates)
    return point + stage_weight * stage + tail_weight * tail + rate_weight * rate, {
        "point": point, "stage": stage, "tail": tail, "rate": rate}


def curve_metrics(cycles, truth, prediction):
    x, y, p = map(np.asarray, (cycles, truth, prediction))
    span = x[-1] - x[0]
    if len(x) < 6 or span <= 0:
        raise ValueError("insufficient curve coverage")
    error = np.abs(p-y)
    tail = x >= x[0] + .8 * span
    drops, slopes_y, slopes_p = [], [], []
    for j in range(3):
        m = (x >= x[0] + j*span/3) & (x <= x[0] + (j+1)*span/3)
        xx, yy, pp = x[m], y[m], p[m]
        if len(xx) < 3:
            return {"evaluable": False, "reason": "stage coverage"}
        width = max(1., min(5., (xx[-1]-xx[0])/4))
        a, b = xx <= xx[0]+width, xx >= xx[-1]-width
        dy, dp = np.median(yy[a])-np.median(yy[b]), np.median(pp[a])-np.median(pp[b])
        drops.append(abs(dy-dp)*100)
        slopes_y.append(dy/(xx[-1]-xx[0]))
        slopes_p.append(dp/(xx[-1]-xx[0]))
    width = min(5., span/10)
    fade = np.median(y[x <= x[0]+width])-np.median(y[x >= x[-1]-width])
    window = max(20., .1*span)
    plateau = rise = False
    shape_evaluable = span >= window and np.max(np.diff(x)) <= window/2
    if shape_evaluable:
        for start in np.arange(x[0], x[-1]-window+1, max(1., window/4)):
            a = (x >= start) & (x <= start+width)
            b = (x >= start+window-width) & (x <= start+window)
            if not a.any() or not b.any():
                continue
            dy, dp = np.median(y[b])-np.median(y[a]), np.median(p[b])-np.median(p[a])
            plateau |= bool(dy <= -.01 and abs(dp) <= .002)
            rise |= bool(dp >= .01 and dy < .005)
    # Fixed conservative acceleration threshold; synthetic counterexample tests
    # precede training. This is not a knee-location estimate.
    reversed_acceleration = bool((slopes_y[-1]-slopes_y[0])*span > .01 and
                                 (slopes_p[-1]-slopes_p[0])*span < -.01)
    result = {"evaluable": bool(shape_evaluable), "mape_percent": float(100*np.mean(error/y)),
              "mae_pp": float(100*error.mean()), "rmse_pp": float(100*np.sqrt(np.mean(error**2))),
              "tail_mae_pp": float(100*error[tail].mean()), "start_error_pp": float(100*error[0]),
              "endpoint_error_pp": float(100*error[-1]), "stage_drop_max_pp": float(max(drops)),
              "relative_fade_mae": float(error.mean()/max(fade,.02)),
              "false_plateau": plateau, "false_rise": rise, "reversed_acceleration": reversed_acceleration}
    sst = np.sum((y-y.mean())**2)
    result["r2"] = float(1-np.sum((p-y)**2)/sst) if sst > 1e-12 else None
    result["joint_pass"] = bool(shape_evaluable and result["mape_percent"] <= 6 and
        result["mae_pp"] <= 1 and result["tail_mae_pp"] <= 2 and result["start_error_pp"] <= 1 and
        result["stage_drop_max_pp"] <= 2 and result["relative_fade_mae"] <= .25 and
        not (plateau or rise or reversed_acceleration))
    return result
