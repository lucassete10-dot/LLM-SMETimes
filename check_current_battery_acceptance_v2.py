"""Audit the existing four datasets before considering new training data.

This is a development-data scorecard, not an independent test certificate.
The chosen challenger routes were selected on development folds, so their
reported scores may be optimistic. In particular, the future-SOH report only
scores rolling 50/100-cycle horizons, not the full remaining life curve.
"""

import json
from pathlib import Path


ROOT = Path('battery_models_v2')
DOMAINS = ('NASA', 'CALCE', 'MIT', 'XJTU')
DEVELOPMENT_SOH_LIMITS = {'NASA': 1.5, 'CALCE': 1.5, 'MIT': 2.0, 'XJTU': 2.5}
FUTURE_MAPE_LIMIT = 6.0
EARLY_EOL_MAPE_LIMIT = 20.0


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding='utf-8'))


def audit():
    soh = read('paper_inspired_soh_challenger/development_cv.json')
    nasa = read('nasa_boundary_soh_challenger/development_cv.json')
    future = read('trajectory/future_soh_mape.json')
    rul = read('paper_inspired_soh_challenger/rul_development_cv.json')
    nasa_rate = read('nasa_rate_decay_challenger/development_cv.json')
    mit_policy = read('mit_policy_rul_challenger/development_cv.json')
    mit_diagnostic = read('mit_checkpoint_rul_challenger/development_cv.json')
    xjtu_combined = read('xjtu_combined_rul_challenger/development_cv.json')
    xjtu_joint = read('joint_future_curve_challenger/development_cv_xjtu.json')
    xjtu_invariant = read('xjtu_invariant_rul_challenger/development_cv.json')
    xjtu_router = read('xjtu_ood_routing_challenger/development_cv.json')
    xjtu_throughput = read('xjtu_throughput_rul_challenger/checkpoint_audit.json')
    mit_exact80 = read('mit_exact80_endpoint_audit.json')
    mit_signal_checkpoint = read('mit_checkpoint_signal_rul_challenger/development_cv.json')
    knee_curve = read('knee_trajectory_challenger/development_cv.json')
    xjtu_selected_curve = read('xjtu_selected_soh_curve_audit.json')
    xjtu_throughput_soh = read('xjtu_throughput_soh_challenger/development_cv.json')
    xjtu_joint_selected = read('xjtu_selected_joint_curve_audit.json')
    selected_joint = read('selected_joint_curve_all_audit.json')
    full_curve = read('full_curve_practical_audit.json')
    soh_route = {'NASA': 'boundary_extrapolation', 'CALCE': 'fusion',
                 'MIT': 'gated', 'XJTU': 'fusion'}
    rul_route = {'NASA': 'baseline', 'CALCE': 'gated_raw',
                 'MIT': 'baseline', 'XJTU': 'gated_log'}
    curve_route = {'NASA': 'local_linear', 'CALCE': 'direct_linear',
                   'MIT': 'fixed_half', 'XJTU': 'bounded_half_local_rate'}
    domains = {}
    for domain in DOMAINS:
        route = soh_route[domain]
        current = (nasa['challenger_cell_macro_mape_percent'] if domain == 'NASA'
                   else soh['domains'][domain]['summary'][route]['cell_macro_mape_percent'])
        horizons = future['domains'][domain]['summary']
        horizon_rows = {}
        for name in ('50', '100'):
            result = horizons[name]
            horizon_rows[name] = {
                'mape_percent': result['equal_cell_mape_percent'],
                'scored_cells': result['scored_cells'],
                'scored_pairs': result['scored_pairs'],
                'below_6_percent': result['equal_cell_mape_percent'] <= FUTURE_MAPE_LIMIT,
            }
        early = rul['domains'][domain]['summary'][rul_route[domain]]
        first_curve = full_curve['domains'][domain]['first_available']
        curve_method = curve_route[domain]
        curve_eol = first_curve['curve_derived_eol'][curve_method]
        unphysical_cells = sum(
            any(cell['metrics'][curve_method][field] > 0
                for field in ('upward_steps', 'below_zero_points', 'above_origin_points'))
            for cell in first_curve['by_cell'])
        domains[domain] = {
            'current_soh': {'route': route, 'cell_macro_mape_percent': current,
                            'development_limit_percent': DEVELOPMENT_SOH_LIMITS[domain],
                            'below_development_limit': current <= DEVELOPMENT_SOH_LIMITS[domain],
                            'below_aspirational_0_5_percent': current <= .5},
            'rolling_future_soh': horizon_rows,
            'early_eol_rul': {'route': rul_route[domain],
                              'mape_percent': early['eol_mape_at_50_percent'],
                              'scored_cells': early['cells_at_50'],
                              'endpoint_cells': early['cells'],
                              'below_20_percent': early['eol_mape_at_50_percent'] <=
                                                  EARLY_EOL_MAPE_LIMIT},
            'full_curve_from_first_available': {
                'route': curve_method,
                'consistent_with_selected_RUL': False,
                'soh_source': 'fusion' if domain == 'XJTU' else 'baseline',
                'curve_mape_percent': (xjtu_selected_curve['summary'][
                    'selected_curve_mape_percent'] if domain == 'XJTU' else
                    first_curve['equal_cell_curve_mape_percent'][curve_method]),
                'scored_cells': first_curve['eligible_cells'],
                'endpoint_cells': first_curve['endpoint_cells'],
                'unphysical_curve_cells': unphysical_cells,
                'curve_derived_eol_mape_percent': curve_eol['mape_percent'],
                'curve_derived_eol_scored_cells': curve_eol['scored_cells']},
            'coherent_selected_joint_curve': {
                'soh_route': selected_joint['domains'][domain]['soh_route'],
                'rul_route': selected_joint['domains'][domain]['rul_route'],
                'curve_mape_percent': selected_joint['domains'][domain]['summary'][
                    'cell_macro_curve_mape_percent'],
                'eol_mape_percent': selected_joint['domains'][domain]['summary'][
                    'cell_macro_eol_mape_percent'],
                'scored_cells': selected_joint['domains'][domain]['summary'][
                    'scored_cells'],
                'endpoint_cells': first_curve['endpoint_cells']},
        }
    gaps = []
    for domain, row in domains.items():
        if not row['current_soh']['below_development_limit']:
            gaps.append(f'{domain}: current SOH above development limit')
        if not row['current_soh']['below_aspirational_0_5_percent']:
            gaps.append(f'{domain}: current SOH above requested 0.5% MAPE')
        for horizon in ('50', '100'):
            score = row['rolling_future_soh'][horizon]
            if not score['below_6_percent']:
                gaps.append(f'{domain}: {horizon}-cycle future SOH above 6%')
            if score['scored_cells'] < 3:
                gaps.append(f'{domain}: {horizon}-cycle validation has fewer than 3 cells')
        early = row['early_eol_rul']
        if not early['below_20_percent']:
            gaps.append(f'{domain}: early EOL/RUL above 20%')
        if early['scored_cells'] < .8 * early['endpoint_cells']:
            gaps.append(f'{domain}: early EOL/RUL covers fewer than 80% of cells')
        curve = row['full_curve_from_first_available']
        if curve['curve_mape_percent'] > FUTURE_MAPE_LIMIT:
            gaps.append(f'{domain}: full future curve above 6% MAPE')
        if curve['scored_cells'] < .8 * curve['endpoint_cells']:
            gaps.append(f'{domain}: full future curve covers fewer than 80% of cells')
        if curve['unphysical_curve_cells']:
            gaps.append(f'{domain}: selected full curves violate physical bounds')
        if (curve['curve_derived_eol_mape_percent'] is None or
                curve['curve_derived_eol_mape_percent'] > EARLY_EOL_MAPE_LIMIT):
            gaps.append(f'{domain}: curve-derived declared EOL crossing above 20% MAPE')
        if curve['curve_derived_eol_scored_cells'] < .8 * curve['endpoint_cells']:
            gaps.append(f'{domain}: curve-derived crossing covers fewer than 80% of cells')
        joint = row['coherent_selected_joint_curve']
        if joint['curve_mape_percent'] > FUTURE_MAPE_LIMIT:
            gaps.append(f'{domain}: coherent selected full curve above 6% MAPE')
        if joint['eol_mape_percent'] > EARLY_EOL_MAPE_LIMIT:
            gaps.append(f'{domain}: coherent selected EOL above 20% MAPE')
        if joint['scored_cells'] < .8 * joint['endpoint_cells']:
            gaps.append(f'{domain}: coherent selected full curve covers fewer than 80% of cells')
    if (xjtu_joint_selected['summary']['throughput_bounded_direct_curve_mape_percent'] >
            FUTURE_MAPE_LIMIT):
        gaps.append('XJTU: best tested coherent SOH/RUL curve above 6% MAPE')
    if (xjtu_joint_selected['summary']['throughput_eol_ape_percent'] >
            EARLY_EOL_MAPE_LIMIT):
        gaps.append('XJTU: best tested first-origin EOL remains above 20% MAPE')
    return {'status': 'ready_for_new_data' if not gaps else 'not_ready_for_new_data',
            'evaluation': 'development whole-cell NASA/CALCE and whole-batch MIT/XJTU',
            'selection_warning': 'challenger and curve routes chosen on these development folds',
            'metric_consistency_warning': 'non-coherent low-MAPE full-curve routes are '
                                          'reported separately from selected SOH+RUL routes',
            'endpoint_definition_warning': 'NASA/CALCE use per-cell 80% SOH; '
                                           'MIT uses source 0.88 Ah and XJTU uses source nominal EOL',
            'MIT_exact80_observability': {
                'observed_events': mit_exact80['observed_exact80_events'],
                'right_censored_cells': mit_exact80['right_censored_exact80'],
                'events_by_batch': {group: row['observed_exact80_events']
                                    for group, row in mit_exact80['by_group'].items()},
                'warning': 'exact 80% SOH leave-batch-out validation has no observed '
                           'events in Batch 1 or 3'},
            'full_future_curve_evaluated': True,
            'independent_test_evaluated': False,
            'gaps': gaps,
            'unpromoted_experiments': {
                'NASA_100_cycle_rate_decay': {
                    'pre_eol_mape_percent': nasa_rate['pre_eol_corrected_mape_percent'],
                    'pre_eol_scored_cells': nasa_rate['pre_eol_validation_cells'],
                    'all_observed_baseline_mape_percent': nasa_rate[
                        'all_observed_baseline_mape_percent'],
                    'all_observed_challenger_mape_percent': nasa_rate[
                        'all_observed_corrected_mape_percent'],
                    'reason': 'only one pre-EOL validation cell; all-observed score worsened'},
                'MIT_policy_signal_only': {
                    'baseline_eol_mape_at_50_percent': mit_policy['summary'][
                        'signal_only']['eol_mape_at_50_percent'],
                    'challenger_eol_mape_at_50_percent': mit_policy['summary'][
                        'signal_policy']['eol_mape_at_50_percent'],
                    'reason': 'known policy features worsened cross-batch error'},
                'MIT_diagnostic_assisted': {
                    'cycle_50_equal_batch_mape_percent': mit_diagnostic['checkpoints'][
                        '50']['equal_batch_mape_percent'],
                    'cycle_100_equal_batch_mape_percent': mit_diagnostic['checkpoints'][
                        '100']['equal_batch_mape_percent'],
                    'reason': 'requires complete measured capacity; worse at cycle 100 and '
                              'does not meet online-input requirement'},
                'MIT_predicted_SOH_checkpoint_RUL': {
                    'candidate_eol_mape_at_50_percent': mit_signal_checkpoint['summary'][
                        'candidate_cell_macro_mape_percent'],
                    'baseline_eol_mape_at_50_percent': mit_signal_checkpoint['summary'][
                        'baseline_cell_macro_mape_percent'],
                    'scored_cells': mit_signal_checkpoint['summary']['scored_cells'],
                    'reason': 'causal estimated-SOH history worsened whole-batch endpoint accuracy; '
                              'no full model promoted'},
                'fixed_cubic_late_knee_curve': {
                    'curve_mape_percent': {domain: knee_curve['domains'][domain][
                        'equal_cell_curve_mape_percent'] for domain in DOMAINS},
                    'reason': 'predeclared physical shape worsened curve accuracy because '
                              'predicted endpoint errors misplaced the knee'},
                'XJTU_combined_300s_usage': {
                    'baseline_all_cycle_mae': xjtu_combined['summary'][
                        'baseline']['cell_macro_mae_cycles'],
                    'challenger_all_cycle_mae': xjtu_combined['summary'][
                        'gated_raw']['cell_macro_mae_cycles'],
                    'improved_strategy_folds': xjtu_combined['summary'][
                        'gated_raw']['improved_strategy_folds_vs_baseline'],
                    'early_eol_mape_at_50_percent': xjtu_combined['summary'][
                        'gated_raw']['eol_mape_at_50_percent'],
                    'early_eol_scored_cells': xjtu_combined['summary'][
                        'gated_raw']['cells_at_50'],
                    'reason': 'all-cycle MAE improved, but early EOL MAPE and coverage still fail'},
                'XJTU_joint_curve_and_eol': {
                    'curve_mape_percent': xjtu_joint['domains']['XJTU']['summary'][
                        'equal_cell_curve_mape_percent'],
                    'eol_mape_percent': xjtu_joint['domains']['XJTU']['summary'][
                        'eol_mape_percent'],
                    'reason': 'single coherent curve and crossing, but both metrics fail'},
                'XJTU_invariant_rul': {
                    'all_cycle_mae_cycles': xjtu_invariant['summary']['relative_only'][
                        'cell_macro_mae_cycles'],
                    'early_eol_mape_at_50_percent': xjtu_invariant['summary'][
                        'relative_only']['eol_mape_at_50_percent'],
                    'reason': 'average MAE improved, but early EOL got worse'},
                'XJTU_ood_routed_rul': {
                    'all_cycle_mae_cycles': xjtu_router['summary']['routed'][
                        'cell_macro_mae_cycles'],
                    'early_eol_mape_at_50_percent': xjtu_router['summary'][
                        'routed']['eol_mape_at_50_percent'],
                    'reason': 'routing improves average MAE but still fails early EOL; '
                              'threshold is not independently calibrated'},
                'XJTU_throughput_aligned_rul': {
                    'cycle_50_eol_mape_percent': xjtu_throughput['checkpoints'][
                        'cycle_50']['throughput_full']['cell_macro_mape_percent'],
                    'cycle_50_scored_cells': xjtu_throughput['checkpoints'][
                        'cycle_50']['throughput_full']['scored_cells'],
                    'first_available_eol_mape_percent': xjtu_throughput['checkpoints'][
                        'first_available']['throughput_full']['cell_macro_mape_percent'],
                    'first_available_scored_cells': xjtu_throughput['checkpoints'][
                        'first_available']['throughput_full']['scored_cells'],
                    'reason': 'fixed-cycle-50 subset below 20%, but broader early checkpoints fail'},
                'XJTU_throughput_aligned_SOH': {
                    'selected_fusion_mape_percent': xjtu_throughput_soh['summary'][
                        'selected_fusion']['cell_macro_mape_percent'],
                    'throughput_gated_mape_percent': xjtu_throughput_soh['summary'][
                        'throughput_gated']['cell_macro_mape_percent'],
                    'throughput_gated_full_curve_mape_percent': xjtu_throughput_soh[
                        'practical_curve']['throughput_gated']['curve_cell_macro_mape_percent'],
                    'reason': 'equal-throughput features worsened SOH and full-curve accuracy'},
            },
            'selected_SOH_curve_recheck': {
                'XJTU': {
                    'old_baseline_curve_mape_percent': xjtu_selected_curve['summary'][
                        'baseline_curve_mape_percent'],
                    'selected_fusion_curve_mape_percent': xjtu_selected_curve['summary'][
                        'selected_curve_mape_percent'],
                    'warning': 'SOH improves, but RUL and 80%-crossing remain inadequate'}},
            'XJTU_joint_curve_recheck': {
                'selected_SOH_plus_selected_RUL_bounded_curve_mape_percent':
                    xjtu_joint_selected['summary'][
                        'selected_bounded_direct_curve_mape_percent'],
                'selected_SOH_plus_throughput_RUL_bounded_curve_mape_percent':
                    xjtu_joint_selected['summary'][
                        'throughput_bounded_direct_curve_mape_percent'],
                'throughput_RUL_first_origin_eol_mape_percent': xjtu_joint_selected[
                    'summary']['throughput_eol_ape_percent'],
                'warning': 'development selection on same held-out groups; not an independent test'},
            'domains': domains}


def main():
    report = audit()
    target = ROOT / 'current_data_acceptance.json'
    target.write_text(json.dumps(report, indent=2), encoding='utf-8')
    for domain, row in report['domains'].items():
        print(domain, 'SOH', round(row['current_soh']['cell_macro_mape_percent'], 2),
              'future50', round(row['rolling_future_soh']['50']['mape_percent'], 2),
              'future100', round(row['rolling_future_soh']['100']['mape_percent'], 2),
              'RUL@50%', round(row['early_eol_rul']['mape_percent'], 2))
    print('New-data gate:', report['status'])


if __name__ == '__main__':
    main()
