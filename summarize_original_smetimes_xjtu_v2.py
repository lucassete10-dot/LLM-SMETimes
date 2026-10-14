"""Aggregate six untouched-strategy original-SMETimes SOH diagnostics."""

import argparse
import json
from pathlib import Path

import numpy as np

from battery_v2_features import load_manifest
from smoke_original_smetimes_battery_v2 import checkpoint_sha256


ROOT = Path('battery_models_v2/original_smetimes_xjtu')
OUT = ROOT / 'summary.json'
NAMES = ('baseline_extra_trees_history', 'flat_smetimes_input',
         'original_smetimes_ridge', 'original_smetimes_extra_trees')


def summarize():
    records = [row for row in load_manifest()['cells'] if row['dataset'] == 'XJTU']
    all_ids = {row['cell_id'] for row in records}
    by_group = {group: {row['cell_id'] for row in records if row['group'] == group}
                for group in sorted({row['group'] for row in records})}
    if len(records) != 55 or len(by_group) != 6:
        raise ValueError('XJTU cohort differs from audited 55-cell/6-strategy set')
    input_folds = json.loads((ROOT / 'input_only_cv.json').read_text(encoding='utf-8'))['folds']
    flat = {row['held_out']: row['flat_smetimes_input'] for row in input_folds}
    flat_baseline = {row['held_out']: row['baseline_120s'] for row in input_folds}
    if len(input_folds) != len(flat) or set(flat) != set(by_group):
        raise ValueError('flat-input folds differ from six audited strategies')
    folds = []
    hashes = set()
    settings = []
    for group, val_ids in by_group.items():
        row = json.loads((ROOT / f'{group}.json').read_text(encoding='utf-8'))
        if (row['held_out'] != group or set(row['validation_cell_ids']) != val_ids or
                set(row['train_cell_ids']) != all_ids-val_ids or group not in flat):
            raise ValueError(f'{group}: strict held-out-cell alignment differs')
        if not np.isclose(row['baseline_extra_trees_history']['cell_macro_mae'],
                          flat_baseline[group]['cell_macro_mae'], atol=1e-9):
            raise ValueError(f'{group}: stored baseline differs from flat-input audit')
        for name in ('baseline_extra_trees_history', 'original_smetimes_ridge',
                     'original_smetimes_extra_trees'):
            if {cell['cell_id'] for cell in row[name]['by_cell']} != val_ids:
                raise ValueError(f'{group}: {name} evaluation cells differ')
        if {cell['cell_id'] for cell in flat[group]['by_cell']} != val_ids:
            raise ValueError(f'{group}: flat-input evaluation cells differ')
        hashes.add(row['checkpoint_sha256'])
        settings.append(row['fixed_settings'])
        folds.append({'held_out': group,
                      'baseline_extra_trees_history': row['baseline_extra_trees_history'],
                      'flat_smetimes_input': flat[group],
                      'original_smetimes_ridge': row['original_smetimes_ridge'],
                      'original_smetimes_extra_trees': row['original_smetimes_extra_trees'],
                      'encoder_diagnostic': row['encoder_diagnostic']})
    if len(hashes) != 1 or next(iter(hashes)) != checkpoint_sha256() or any(
            item != settings[0] for item in settings):
        raise ValueError('SMETimes fold checkpoint or fixed settings differ')
    summary = {name: {
        'equal_fold_macro_mae': float(np.mean([fold[name]['cell_macro_mae']
                                              for fold in folds])),
        'equal_cell_macro_mae': float(np.mean([cell['mae'] for fold in folds
                                               for cell in fold[name]['by_cell']])),
        'worst_cell_mae': float(max(cell['mae'] for fold in folds
                                    for cell in fold[name]['by_cell'])),
        'folds_better_than_selected_baseline': sum(
            fold[name]['cell_macro_mae'] <
            fold['baseline_extra_trees_history']['cell_macro_mae'] for fold in folds),
    } for name in NAMES}
    return {'status': 'development-only original Llama SMETimes whole-strategy SOH ablation',
            'checkpoint_sha256': next(iter(hashes)),
            'fixed_settings': settings[0],
            'comparison': 'same 55 cells and six held-out strategies; encoder, PCA, '
                          'imputer and regression heads trained without outer validation data',
            'warning': 'one improved strategy does not justify replacing the baseline; '
                       'architecture and input were explored on this development corpus; '
                       'new independent batteries are still needed',
            'summary': summary, 'folds': folds}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    current = summarize()
    if args.verify:
        if current != json.loads(OUT.read_text(encoding='utf-8')):
            raise ValueError('saved original SMETimes summary differs from fold artifacts')
        print('VERIFIED', OUT)
    else:
        if OUT.exists():
            raise FileExistsError(OUT)
        OUT.write_text(json.dumps(current, indent=2), encoding='utf-8')
        print('WROTE', OUT, flush=True)
