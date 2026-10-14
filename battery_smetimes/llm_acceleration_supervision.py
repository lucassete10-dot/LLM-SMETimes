"""Sustained TRAIN curve changes, with no decoder-parameter knee assumption."""
import torch


ACCELERATION_RULES = {'window_cycles': [100., 200.], 'anchor_width_fraction': .125,
                      'stride_fraction': .5, 'minimum_anchor_points': 2,
                      'maximum_gap_fraction': .25}


def sustained_acceleration_loss(prediction, truth, cycles, valid):
    """Cell-balanced multiscale second differences of robust local levels.

    These are training targets, never inference inputs or boundaries. Slopes
    use actual anchor coordinates; a fixed window/2 multiplier gives SOH units.
    No adjacent noisy derivatives, inferred lifetime or forced acceleration.
    Both observed recovery and deceleration may have negative targets.
    """
    losses, counts = [], []
    for i in range(len(prediction)):
        mask = valid[i].bool()
        x, y, p = cycles[i][mask], truth[i][mask], prediction[i][mask]
        scales, count = [], 0
        if len(x) < 6 or torch.any(x[1:] <= x[:-1]):
            losses.append(p.sum() * 0)
            counts.append(0)
            continue
        for window in ACCELERATION_RULES['window_cycles']:
            errors = []
            start = float(x[0])
            while start + window <= float(x[-1]) + 1e-5:
                region = (x >= start) & (x <= start + window)
                xx = x[region]
                if len(xx) < 6 or torch.any(xx[1:] - xx[:-1] > window * ACCELERATION_RULES['maximum_gap_fraction']):
                    start += window * ACCELERATION_RULES['stride_fraction']
                    continue
                anchors = []
                width = window * ACCELERATION_RULES['anchor_width_fraction']
                for left, right in ((start, start + width),
                                    (start + window/2 - width/2, start + window/2 + width/2),
                                    (start + window - width, start + window)):
                    take = (x >= left) & (x <= right)
                    if int(take.sum()) < ACCELERATION_RULES['minimum_anchor_points']:
                        break
                    # Identical robust aggregation for truth and predictions.
                    anchors.append((x[take].median(), y[take].median(), p[take].median()))
                if len(anchors) == 3:
                    def change(index):
                        first = (anchors[0][index] - anchors[1][index]) / (anchors[1][0] - anchors[0][0])
                        second = (anchors[1][index] - anchors[2][index]) / (anchors[2][0] - anchors[1][0])
                        return (second - first) * window / 2
                    errors.append((change(2) - change(1)).square())
                start += window * ACCELERATION_RULES['stride_fraction']
            if errors:
                scales.append(torch.stack(errors).mean())
                count += len(errors)
        losses.append(torch.stack(scales).mean() if scales else p.sum() * 0)
        counts.append(count)
    return torch.stack(losses), counts
