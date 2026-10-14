"""Fit a preregistered 300-second XJTU SOH challenger on all development cells.

The frozen 120-second production candidate is unchanged. This challenger was
proposed after development error analysis and requires independent testing.
"""

import json
from pathlib import Path

import joblib
import numpy as np

from battery_v2_features import load_all
from evaluate_xjtu_long_prefix_v2 import ROOT, fit, matrix


MODEL = ROOT / 'extra_trees_300s_challenger.joblib'
MANIFEST = ROOT / 'challenger_manifest.json'


def train():
    extraction = json.loads((ROOT / 'extraction_audit.json').read_text(encoding='utf-8'))
    validation = json.loads((ROOT / 'group_cv.json').read_text(encoding='utf-8'))
    if extraction['total_diagnostics'] != 10389 or len(extraction['cells']) != 55:
        raise ValueError('XJTU 300s extraction coverage differs from audited training population')
    if validation['summary']['300s']['improved_folds_vs_120s'] != 6:
        raise ValueError('300s challenger no longer matches reported outer-fold check')
    data, cells = load_all()
    ids = [row['cell_id'] for row in data['cells'] if row['dataset'] == 'XJTU']
    features = {}
    with np.load(ROOT / 'features.npz') as archive:
        for key in ids:
            if not np.array_equal(archive[f'{key}_cycle'], cells[key].cycle):
                raise ValueError(f'{key}: cycle alignment mismatch')
            raw = archive[f'{key}_300s'].astype('float32')
            if raw.shape != (len(cells[key].cycle), 82):
                raise ValueError(f'{key}: 300s raw shape mismatch')
            features[key] = matrix(raw, cells[key].reference_count)
    imputer, model = fit(cells, features, ids)
    bundle = {
        'dataset': 'XJTU', 'feature_schema': 'xjtu_diagnostic_prefix_300s_v1',
        'window_s': 300, 'raw_columns': 82, 'reference_count': 5,
        'variant': 'signal_history', 'imputer': imputer, 'model': model,
        'training_cell_ids': ids, 'data_contract_version': 2,
        'status': 'development challenger; independent accuracy unknown',
    }
    ROOT.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL)
    reloaded = joblib.load(MODEL)
    smoke = reloaded['model'].predict(reloaded['imputer'].transform(features[ids[0]]))
    if len(smoke) != len(cells[ids[0]].cycle) or not np.isfinite(smoke).all():
        raise ValueError('saved challenger failed prediction smoke test')
    MANIFEST.write_text(json.dumps({
        'status': bundle['status'], 'feature_schema': bundle['feature_schema'],
        'development_selection_warning': validation['selection_warning'],
        'training_cell_ids': ids, 'training_diagnostics': extraction['total_diagnostics'],
        'whole_strategy_cv': validation['summary'],
        'model_file': str(MODEL),
    }, indent=2), encoding='utf-8')
    print('TRAINED', MODEL, len(ids), 'cells', extraction['total_diagnostics'],
          'diagnostics; independent accuracy unknown', flush=True)


if __name__ == '__main__':
    train()
