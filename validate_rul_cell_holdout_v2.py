"""Diagnostic: unseen cells under known MIT batches and XJTU strategies."""

import json
from pathlib import Path

import numpy as np

from battery_v2_features import load_all
from train_rul_v2 import SOH_MODEL, assess, fit_rul, oof_soh, predict_rul
from train_soh_v2 import cell_holdout, fit as fit_soh, predict as predict_soh


def main():
    manifest, cells = load_all()
    for domain in ('MIT', 'XJTU'):
        training, validation = cell_holdout(manifest['cells'], domain)
        train = [key for key in training if cells[key].endpoint_cycle is not None]
        val = [key for key in validation if cells[key].endpoint_cycle is not None]
        base = float(np.median([cells[key].endpoint_cycle for key in train]))
        direct = fit_rul(cells, train, 'direct')
        oof = oof_soh(cells, manifest['cells'], train)
        assisted = fit_rul(cells, train, 'assisted', oof)
        soh = fit_soh(cells, train, SOH_MODEL)
        report = {'validation_design': 'last two whole cells per known batch/strategy',
                  'train_cells': len(train), 'validation_cells': len(val),
                  'censored_validation_cells': len(validation)-len(val),
                  'direct': assess(cells, val,
                                   {key: predict_rul(direct, cells[key]) for key in val}, base),
                  'assisted': assess(cells, val,
                                     {key: predict_rul(assisted, cells[key],
                                           predict_soh(soh, cells[key])) for key in val}, base)}
        path = Path('battery_models_v2/rul') / domain.lower() / 'same_group_cell_holdout.json'
        path.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(domain, {kind: round(report[kind]['cell_macro_mae_cycles'], 1)
                       for kind in ('direct', 'assisted')}, flush=True)


if __name__ == '__main__':
    main()
