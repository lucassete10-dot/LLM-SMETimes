"""Oracle diagnostic: can early *measured* SOH slope explain XJTU EOL?

This deliberately uses capacity labels through the origin and is NOT a
capacity-free model. It measures whether slope extrapolation is worth trying
with estimated SOH. No held-out future capacity enters the prediction.
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import theilslopes

from battery_v2_features import load_cell, load_manifest
from train_trajectory_v2 import threshold_soh


OUT = Path('battery_models_v2/xjtu_early_soh_slope_oracle.json')
MAX_LIFE = 2500


def estimate_endpoint(cycle, soh, threshold, origin, window):
    """Robust linear extrapolation from measured SOH up to origin only."""
    if window < 3 or not 0 <= origin < len(cycle):
        raise ValueError('invalid slope window or origin')
    begin = max(0, origin-window+1)
    indices = np.arange(begin, origin+1)
    indices = indices[np.isfinite(soh[indices])]
    if len(indices) < 3:
        return None, None
    slope = float(theilslopes(soh[indices], cycle[indices])[0])
    if soh[origin] <= threshold:
        return int(cycle[origin]), slope
    if slope >= 0:
        return MAX_LIFE, slope
    projected = int(np.ceil(cycle[origin] + (soh[origin]-threshold)/(-slope)))
    return int(np.clip(projected, cycle[origin], MAX_LIFE)), slope


def main():
    manifest = load_manifest()
    rows = []
    for record in manifest['cells']:
        if record['dataset'] != 'XJTU':
            continue
        cell = load_cell(record)
        origin = max(15, cell.reference_count)
        if origin >= len(cell.cycle) or cell.endpoint_cycle is None:
            continue
        if cell.cycle[origin] >= cell.endpoint_cycle:
            continue
        estimates = {}
        for window in (5, 10, 16):
            endpoint, slope = estimate_endpoint(cell.cycle, cell.soh,
                                                threshold_soh(cell), origin, window)
            estimates[str(window)] = {
                'predicted_endpoint_cycle': endpoint,
                'early_soh_slope_per_source_cycle': slope,
                'eol_ape_percent': (float(100*abs(endpoint-cell.endpoint_cycle)/
                                          cell.endpoint_cycle)
                                    if endpoint is not None else None)}
        rows.append({'cell_id': cell.cell_id, 'group': cell.group,
                     'origin_cycle': int(cell.cycle[origin]),
                     'source_endpoint_cycle': int(cell.endpoint_cycle),
                     'true_soh_at_origin': float(cell.soh[origin]),
                     'threshold_soh': float(threshold_soh(cell)),
                     'windows': estimates})
    summary = {}
    for window in ('5', '10', '16'):
        scored = [r['windows'][window]['eol_ape_percent'] for r in rows
                  if r['windows'][window]['eol_ape_percent'] is not None]
        summary[window] = {'scored_cells': len(scored),
                           'cell_macro_eol_mape_percent': float(np.mean(scored)),
                           'by_group': {group: float(np.mean([
                               r['windows'][window]['eol_ape_percent'] for r in rows
                               if r['group'] == group and
                               r['windows'][window]['eol_ape_percent'] is not None]))
                               for group in sorted({r['group'] for r in rows})}}
    report = {'status': 'oracle measured-prefix SOH diagnostic, not deployable model',
              'input': 'true capacity/SOH through origin only; no future capacity',
              'target': 'source-defined XJTU EOL, not universally exact 80% crossing',
              'max_projected_cycle': MAX_LIFE, 'summary': summary, 'by_cell': rows}
    OUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print({window: value['cell_macro_eol_mape_percent'] for window, value
           in summary.items()})
    print('WROTE', OUT)


if __name__ == '__main__':
    main()
