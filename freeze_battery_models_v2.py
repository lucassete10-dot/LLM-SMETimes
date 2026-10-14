"""Hash the selected development SOH/RUL candidates before independent testing.

Run `python freeze_battery_models_v2.py` only after training/evaluation is done.
Run `python freeze_battery_models_v2.py --verify` before scoring new data. A
mismatch means the candidate changed and the evaluation is not a frozen test.
"""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile


ROOT = Path('battery_models_v2')
FREEZE = ROOT / 'frozen_candidate_manifest.json'
ARCHIVE = ROOT / 'freeze_history'
DOMAINS = ('NASA', 'CALCE', 'MIT', 'XJTU')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def selected_files():
    files = [Path('battery_data_v2/manifest.json'),
             Path('battery_data_v2/operating_metadata_audit.json'),
             ROOT / 'development_intervals.json',
             Path('battery_v2_features.py'), Path('predict_battery_v2.py'),
             Path('train_soh_v2.py'), Path('train_rul_v2.py'),
             Path('train_trajectory_v2.py'), Path('external_test_v2.py'),
             Path('evaluate_interval_robustness_v2.py'),
             Path('battery_smetimes_signal_v2.py'),
             Path('audit_smetimes_signal_v2.py'),
             Path('smoke_original_smetimes_battery_v2.py'),
             Path('evaluate_original_smetimes_xjtu_v2.py'),
             Path('evaluate_smetimes_input_baseline_v2.py'),
             Path('summarize_original_smetimes_xjtu_v2.py'),
             Path('models/SMETimes_Llama.py'), Path('utils/device.py'),
             Path('utils/llm.py'), Path('layers/mlp.py'),
             Path('prepare_xjtu.py'), Path('adapt_xjtu_mat_v2.py'),
             Path('prepare_mit.py'), Path('audit_mit_label_free_extraction_v2.py'),
             Path('adapt_mit_mat_v2.py'),
             Path('battery_raw_mvp.py'), Path('adapt_nasa_mat_v2.py'),
             Path('prepare_calce.py'), Path('audit_calce_label_free_extraction_v2.py'),
             Path('adapt_calce_xlsx_v2.py'),
             Path('battery_data_contract.py'), Path('build_battery_data_v2.py'),
             Path('audit_capacity_reference_v2.py'),
             Path('audit_operating_metadata_v2.py'),
             Path('audit_xjtu_soh_residuals_v2.py'),
             Path('audit_xjtu_prefix_coverage_v2.py'),
             Path('extract_xjtu_long_prefix_v2.py'),
             Path('evaluate_xjtu_long_prefix_v2.py'),
             Path('evaluate_xjtu_group_balance_v2.py'),
             Path('evaluate_xjtu_300_nested_intervals_v2.py'),
             Path('evaluate_xjtu_stage_ablation_v2.py'),
             Path('train_xjtu_300_challenger_v2.py'),
             Path('adapt_xjtu_300_mat_v2.py'),
             Path('predict_xjtu_300_challenger_v2.py'),
             Path('evaluate_xjtu_300_rul_direct_v2.py'),
             Path('evaluate_xjtu_300_rul_assisted_v2.py'),
             Path('evaluate_xjtu_rul_target_v2.py'),
             Path('evaluate_xjtu_rul_target_nested_v2.py'),
             Path('evaluate_mit_censored_aft_v2.py'),
             Path('evaluate_mit_censored_linear_v2.py'),
             Path('train_xjtu_raw_rul_challenger_v2.py'),
             Path('predict_xjtu_raw_rul_challenger_v2.py'),
             Path('xjtu_applicability_v2.py'),
             Path('evaluate_xjtu_applicability_cv_v2.py'),
             ROOT / 'xjtu_applicability_reference.npz',
             ROOT / 'xjtu_applicability_group_cv.json',
             ROOT / 'interval_robustness_diagnostic.json',
             ROOT / 'smetimes_signal_audit.json',
             ROOT / 'original_smetimes_smoke.json',
             ROOT / 'original_smetimes_xjtu' / 'input_only_cv.json',
             ROOT / 'original_smetimes_xjtu' / 'summary.json',
             *(ROOT / 'original_smetimes_xjtu' / f'batch{i}.json'
               for i in range(1, 7)),
             Path('extract_xjtu_usage_history_v2.py'),
             Path('evaluate_xjtu_usage_rul_v2.py'),
             ROOT / 'xjtu_usage_history' / 'features.npz',
             ROOT / 'xjtu_usage_history' / 'extraction_audit.json',
             ROOT / 'xjtu_usage_history' / 'direct_rul_group_cv.json',
             ROOT / 'rul' / 'xjtu' / 'target_transform_diagnostic.json',
             ROOT / 'rul' / 'xjtu' / 'target_transform_nested_cv.json',
             ROOT / 'rul' / 'xjtu' / 'raw_cycle_target_challenger.joblib',
             ROOT / 'rul' / 'xjtu' / 'raw_cycle_target_challenger_manifest.json',
             ROOT / 'rul' / 'mit' / 'censored_aft_diagnostic.json',
             ROOT / 'rul' / 'mit' / 'censored_linear_diagnostic.json',
             ROOT / 'xjtu_long_prefix' / 'features.npz',
             ROOT / 'xjtu_long_prefix' / 'extraction_audit.json',
             ROOT / 'xjtu_long_prefix' / 'group_cv.json',
             ROOT / 'xjtu_group_balance' / 'group_cv.json',
             ROOT / 'xjtu_long_prefix' / 'nested_interval_diagnostic.json',
             ROOT / 'xjtu_long_prefix' / 'stage_ablation.json',
             ROOT / 'xjtu_long_prefix' / 'direct_rul_300s_diagnostic.json',
             ROOT / 'xjtu_long_prefix' / 'assisted_rul_300s_diagnostic.json',
             ROOT / 'xjtu_long_prefix' / 'challenger_manifest.json',
             ROOT / 'xjtu_long_prefix' / 'extra_trees_300s_challenger.joblib',
             Path('freeze_battery_models_v2.py'), ROOT / 'trajectory' / 'summary.json']
    selected = {}
    for domain in DOMAINS:
        soh_folder = ROOT / 'soh' / domain.lower()
        rul_folder = ROOT / 'rul' / domain.lower()
        soh_info = json.loads((soh_folder / 'training_manifest.json').read_text())
        rul_info = json.loads((rul_folder / 'training_manifest.json').read_text())
        selected[domain] = {'soh': soh_info['selected_model'],
                            'rul': rul_info['selected_model']}
        files.extend((soh_folder / 'training_manifest.json',
                      rul_folder / 'training_manifest.json',
                      soh_folder / f"{soh_info['selected_model']}.joblib",
                      rul_folder / f"{rul_info['selected_model']}.joblib",
                      ROOT / 'trajectory' / domain.lower() / 'rate_model.joblib'))
        if rul_info['selected_model'] == 'assisted':
            files.append(soh_folder / 'extra_trees_history.joblib')
    return selected, sorted(set(files), key=str)


def create(replace=False):
    selected, files = selected_files()
    payload = {'data_contract_version': 2, 'purpose': 'frozen development candidate for first independent test',
               'selected': selected, 'sha256': {str(path): sha256(path) for path in files},
               'registered_challenger': {'XJTU_SOH': 'xjtu_diagnostic_prefix_300s_v1',
                                          'XJTU_RUL': 'xjtu_raw_cycle_rul_v1'},
               'warning': 'frozen does not mean externally validated or safe for deployment'}
    encoded = json.dumps(payload, indent=2).encode('utf-8')
    if FREEZE.exists():
        previous = FREEZE.read_bytes()
        if previous == encoded:
            print('UNCHANGED', len(files), 'frozen files:', FREEZE)
            return
        if not replace:
            raise FileExistsError('freeze already exists and differs; preserve any first blind '
                                  'score before explicitly using --replace')
        ARCHIVE.mkdir(parents=True, exist_ok=True)
        old_hash = hashlib.sha256(previous).hexdigest()
        archived = ARCHIVE / f'{old_hash}.json'
        if archived.exists() and archived.read_bytes() != previous:
            raise ValueError(f'freeze archive hash collision: {archived}')
        if not archived.exists():
            archived.write_bytes(previous)
        print('ARCHIVED previous freeze:', archived)
    with tempfile.NamedTemporaryFile(dir=FREEZE.parent, prefix='.freeze-', suffix='.tmp',
                                     delete=False) as stream:
        stream.write(encoded)
        staged = Path(stream.name)
    staged.replace(FREEZE)
    print('FROZEN', len(files), 'files:', FREEZE)


def verify():
    payload = json.loads(FREEZE.read_text(encoding='utf-8'))
    selected, files = selected_files()
    if (selected != payload['selected'] or
            payload.get('registered_challenger') !=
            {'XJTU_SOH': 'xjtu_diagnostic_prefix_300s_v1',
             'XJTU_RUL': 'xjtu_raw_cycle_rul_v1'} or
            set(map(str, files)) != set(payload['sha256'])):
        raise ValueError('selected-model set differs from freeze manifest')
    changed = [str(path) for path in files if sha256(path) != payload['sha256'][str(path)]]
    if changed:
        raise ValueError(f'frozen candidate changed: {changed}')
    print('VERIFIED', len(files), 'frozen files')


def main():
    parser = argparse.ArgumentParser()
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--verify', action='store_true')
    actions.add_argument('--replace', action='store_true',
                         help='archive and replace an existing freeze after preserving any first score')
    args = parser.parse_args()
    verify() if args.verify else create(replace=args.replace)


if __name__ == '__main__':
    main()
