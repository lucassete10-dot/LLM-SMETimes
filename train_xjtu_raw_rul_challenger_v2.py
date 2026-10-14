"""Fit an optional XJTU raw-cycle RUL challenger on the 55 development cells.

The selected log-target RUL model remains unchanged. The raw-target variant
was proposed after inspecting development errors and requires an independent,
paired blind test before any model-selection claim.
"""

import json
from pathlib import Path

import joblib
import numpy as np

from battery_v2_features import load_all
from evaluate_xjtu_rul_target_v2 import fit_raw_target
from train_rul_v2 import MIN_HISTORY, signal_features


ROOT = Path('battery_models_v2/rul/xjtu')
MODEL = ROOT / 'raw_cycle_target_challenger.joblib'
MANIFEST = ROOT / 'raw_cycle_target_challenger_manifest.json'


def train():
    if MODEL.exists() or MANIFEST.exists():
        raise FileExistsError('raw-cycle RUL challenger already exists')
    source = json.loads((ROOT / 'target_transform_diagnostic.json').read_text(
        encoding='utf-8'))
    nested = json.loads((ROOT / 'target_transform_nested_cv.json').read_text(
        encoding='utf-8'))
    contract, all_cells = load_all()
    ids = [row['cell_id'] for row in contract['cells'] if row['dataset'] == 'XJTU']
    cells = {key: all_cells[key] for key in ids}
    if len(ids) != 55 or any(cell.endpoint_cycle is None for cell in cells.values()):
        raise ValueError('raw-cycle challenger requires the audited 55 observed XJTU cells')
    if (source['summary']['raw_improved_folds'] != 3 or
            nested['summary']['outer_folds_selecting_raw'] != 5):
        raise ValueError('development target-transform audit differs from registration')
    imputer, model = fit_raw_target(cells, ids)
    bundle = {'dataset': 'XJTU', 'feature_schema': 'xjtu_diagnostic_prefix_120s_v1',
              'reference_count': 5, 'min_history': MIN_HISTORY,
              'target_transform': 'raw_cycle_count',
              'imputer': imputer, 'model': model, 'training_cell_ids': ids,
              'data_contract_version': 2,
              'status': 'post-hoc development challenger; independent accuracy unknown'}
    smoke = np.maximum(0, model.predict(imputer.transform(signal_features(cells[ids[0]]))))
    if len(smoke) != len(cells[ids[0]].cycle) or not np.isfinite(smoke).all():
        raise ValueError('raw-cycle challenger prediction smoke test failed')
    joblib.dump(bundle, MODEL)
    MANIFEST.write_text(json.dumps({
        'status': bundle['status'], 'feature_schema': bundle['feature_schema'],
        'target_transform': bundle['target_transform'], 'training_cell_ids': ids,
        'development_equal_fold_mae_cycles': source['summary']['raw_cycle_target']
            ['equal_fold_macro_mae_cycles'],
        'development_improved_folds': source['summary']['raw_improved_folds'],
        'nested_selection_equal_fold_mae_cycles': nested['summary']['nested_selected']
            ['equal_fold_macro_mae_cycles'],
        'warning': 'not the default RUL model; no calibrated interval; development '
                   'improvement is post-hoc and mixed across strategies',
        'model_file': str(MODEL),
    }, indent=2), encoding='utf-8')
    print('TRAINED', MODEL, len(ids), 'cells; independent accuracy unknown')


if __name__ == '__main__':
    train()
