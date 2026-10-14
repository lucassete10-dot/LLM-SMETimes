"""MIT age-blended EOL evaluation using two existing whole-batch OOF routes.

The 100-cycle blend scale is transferred unchanged from the XJTU experiment,
not fitted on MIT labels. This is still development data because the
component models were selected/assessed on these MIT batch folds.
"""

import json
from pathlib import Path

import numpy as np

from battery_v2_features import load_manifest
from evaluate_xjtu_age_blend_v2 import blend_endpoints, read_oof


BASE = Path('battery_models_v2/rul/mit/development_oof_predictions.npz')
FRACTION = Path('battery_models_v2/mit_fractional_life_challenger/development_oof_endpoint.npz')
OUT = Path('battery_models_v2/mit_fractional_life_challenger/age_blend_audit.json')
LANDMARKS = (50, 100, 200, 500)


def main():
    records = [row for row in load_manifest()['cells'] if row['dataset'] == 'MIT'
               and row['endpoint_cycle'] is not None]
    direct = read_oof(BASE, 'estimate')
    fraction = read_oof(FRACTION, 'fractional_endpoint')
    if set(direct) != set(fraction):
        raise ValueError('MIT OOF component cycles do not align')
    rows = []
    for record in records:
        key = record['cell_id']
        cycle = np.array(sorted(number for cell_id, number in direct if cell_id == key))
        if not len(cycle) or cycle[0] >= record['endpoint_cycle']:
            continue
        d = np.array([direct[key, int(number)] for number in cycle])
        f = np.array([fraction[key, int(number)] for number in cycle])
        endpoints = {'selected_direct': cycle+d,
                     'fraction': f,
                     'blend': blend_endpoints(cycle, d, f)}
        true = record['endpoint_cycle']
        row = {'cell_id': key, 'group': record['group'],
               'origin_cycle': int(cycle[0]), 'true_endpoint_cycle': true,
               'landmarks': {}}
        for name, value in endpoints.items():
            row[f'{name}_origin_ape_percent'] = float(100*abs(value[0]-true)/true)
            row[f'{name}_all_cycle_mae'] = float(np.mean(abs(value-true)))
        for landmark in LANDMARKS:
            if true <= landmark:
                continue
            candidates = np.flatnonzero(cycle <= landmark)
            if not len(candidates):
                continue
            i = int(candidates[-1])
            row['landmarks'][str(landmark)] = {
                'observed_cycle': int(cycle[i]),
                **{name+'_ape_percent': float(100*abs(value[i]-true)/true)
                   for name, value in endpoints.items()}}
        rows.append(row)
    names = ('selected_direct', 'fraction', 'blend')
    summary = {'scored_cells': len(rows)}
    for metric in ('origin_ape_percent', 'all_cycle_mae'):
        summary[metric] = {name: float(np.mean([
            row[f'{name}_{metric}'] for row in rows])) for name in names}
    summary['landmarks'] = {}
    for landmark in LANDMARKS:
        subset = [row['landmarks'][str(landmark)] for row in rows
                  if str(landmark) in row['landmarks']]
        summary['landmarks'][str(landmark)] = {
            'scored_cells': len(subset),
            **{name: float(np.mean([r[name+'_ape_percent'] for r in subset]))
               if subset else None for name in names}}
    report = {'status': 'unpromoted MIT whole-batch development OOF blend',
              'rule': 'fraction_weight = observed_cycle/(observed_cycle+100)',
              'scale_source': '100-cycle age scale transferred unchanged from XJTU',
              'endpoint': 'MIT source 0.88 Ah; 12 right-censored cells unscored',
              'summary': summary, 'by_cell': rows}
    OUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('SUMMARY', summary)
    print('WROTE', OUT)


if __name__ == '__main__':
    main()
