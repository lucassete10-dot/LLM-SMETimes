"""Validation-only saved predictions on identical cells and future coordinates.

Does not fit a model or inspect heldout predictions. Full-coverage scores and
common-interval scores have distinct interpretations; neither proves optimal N.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from battery_smetimes.llm_curve_objective import curve_metrics
from scripts.prepare_same_type_forecast import save_json, sha
from scripts.verify_llm_curve_metrics import point_metrics


def aligned_curves(first, second, after):
    x1, y1, p1 = first
    x2, y2, p2 = second
    for x in (x1, x2):
        if not np.all(np.diff(x) > 0):
            raise ValueError('source coordinates must be unique and increasing')
    x, a, b = np.intersect1d(x1, x2, return_indices=True)
    keep = x > after
    x, a, b = x[keep], a[keep], b[keep]
    np.testing.assert_allclose(y1[a], y2[b], rtol=0, atol=1e-7)
    return x, y1[a], p1[a], p2[b]


def aggregate(records):
    if not records:
        return {'cells': 0}
    names = ('mape_percent', 'mae_pp', 'tail_mae_pp', 'stage_drop_max_pp', 'start_error_pp')
    return {'cells': len(records), 'evaluable_cells': sum(r.get('evaluable', False) for r in records),
            'joint_passes': sum(r.get('joint_pass', False) for r in records),
            **{k: float(np.mean([r[k] for r in records if k in r])) for k in names}}


def fixed_horizons(curve, origin):
    x, y, p = curve
    results = {}
    for horizon in (50, 100, 200):
        inside = (x > origin) & (x <= origin+horizon)
        record = {'requested_source_horizon': horizon, 'observed_points': int(inside.sum()),
                  'coverage_reaches_requested_end': bool(x[-1] >= origin+horizon),
                  'observed_last_source_cycle': float(x[inside][-1]) if inside.any() else None}
        if inside.sum() >= 2:
            metrics, weighted = point_metrics(x[inside], y[inside], p[inside])
            record.update(metrics=metrics, cycle_weighted_mae_pp=weighted)
        else:
            record['reason'] = 'insufficient observations in requested distance'
        results[str(horizon)] = record
    return results


def verify_signal_extension(cfg, provenance):
    """Allow a proven label-preserving signal extension, not arbitrary drift."""
    parent, extended = [Path(c['frozen_plan']) for c in cfg]
    binding = json.loads((extended/'signal_extension_binding.json').read_text())
    if Path(binding['parent_plan']).resolve() != parent.resolve():
        raise ValueError('different frozen plan is not a bound signal extension')
    frozen = [json.loads((p/'config.json').read_text()) for p in (parent, extended)]
    if frozen[1]['signal_extension_parent_config_sha256'] != sha(parent/'config.json'):
        raise ValueError('extension parent plan fingerprint mismatch')
    for name in ('splits.json', 'cell_registry.json', 'source_snapshot/battery_physical_registry.json'):
        if sha(parent/name) != sha(extended/name):
            raise ValueError('extension changed identities or splits: '+name)
    data = [Path(json.loads((p/'audit.json').read_text())['prepared_folder']) for p in (parent, extended)]
    manifests = []
    for i, folder in enumerate(data):
        if sha(folder/'manifest.json') != provenance[i]['data_manifest_sha256']:
            raise ValueError('run data manifest fingerprint mismatch')
        manifests.append({r['cell_id']: r for r in json.loads((folder/'manifest.json').read_text())['cells']})
    if manifests[0].keys() != manifests[1].keys():
        raise ValueError('extension changed cell coverage')
    for cid, a in manifests[0].items():
        b = manifests[1][cid]
        if a['metadata'] != b['metadata']:
            raise ValueError('extension changed physical conditions')
        for i, record in enumerate((a, b)):
            if sha(record['file']) != provenance[i]['prepared_cells_sha256'][cid]:
                raise ValueError('run cell fingerprint mismatch')
        with np.load(a['file']) as x, np.load(b['file']) as y:
            for key in x.files:
                v = y[key][y['cycle'] <= 100] if key in ('cycle', 'raw_features') else y[key]
                np.testing.assert_array_equal(x[key], v)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--control', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--control-name', required=True)
    parser.add_argument('--candidate-name', required=True)
    args = parser.parse_args()
    roots = [Path(args.control), Path(args.candidate)]
    cfg = [json.loads((r/'config.json').read_text()) for r in roots]
    if cfg[0]['plan'] != cfg[1]['plan']:
        raise ValueError('unmatched physical-cell plans')
    provenance = [json.loads((r/'provenance.json').read_text()) for r in roots]
    if cfg[0]['seed'] != cfg[1]['seed']:
        raise ValueError('unmatched seed')
    # Human-readable reference notes can differ in wording. Require identical
    # actual prepared data and physical-cell file hashes, then check all shared
    # labels point by point below; do not rewrite historical run metadata.
    if any(provenance[0][field] != provenance[1][field]
           for field in ('data_manifest_sha256', 'prepared_cells_sha256')):
        verify_signal_extension(cfg, provenance)
    names = [args.control_name, args.candidate_name]
    arrays = [np.load(r/n/'validation_predictions.npz') for r, n in zip(roots, names)]
    common, full = [], [[], []]
    after = max(c['origin'] for c in cfg)
    coverage = [[], []]
    fixed_distance = [[], []]
    try:
        for cid in cfg[0]['plan']['validation']:
            curves = []
            for i in range(2):
                prefix = f'{cid}__{cfg[i]["origin"]}__'
                if prefix+'cycle' not in arrays[i]:
                    curves.append(None)
                    continue
                curve = tuple(arrays[i][prefix+k] for k in ('cycle', 'truth', 'prediction'))
                curves.append(curve)
                coverage[i].append(cid)
                full[i].append(curve_metrics(*curve))
                fixed_distance[i].append({'cell': cid, 'horizons': fixed_horizons(curve, cfg[i]['origin'])})
            if any(c is None for c in curves):
                continue
            x, y, p1, p2 = aligned_curves(*curves, after)
            if len(x) < 9:
                common.append({'cell': cid, 'status': 'insufficient common coverage', 'points': len(x)})
                continue
            common.append({'cell': cid, 'points': len(x), 'first_source_cycle': float(x[0]),
                           'last_source_cycle': float(x[-1]),
                           'control': curve_metrics(x, y, p1), 'candidate': curve_metrics(x, y, p2)})
    finally:
        for a in arrays:
            a.close()
    output = {'scope': 'selection validation only; no heldout N selection or optimal-N claim',
              'control': str(roots[0]), 'candidate': str(roots[1]),
              'origins': [c['origin'] for c in cfg], 'common_source_cycles_strictly_after': after,
              'common_interval_summary': {role: aggregate([r[role] for r in common if role in r])
                                          for role in ('control', 'candidate')},
              'full_coverage_summary': {role: aggregate(full[i]) for i, role in enumerate(('control', 'candidate'))},
              'validation_coverage': coverage, 'by_cell': common,
              'fixed_distance_by_budget': fixed_distance,
              'note': 'Common-interval start error is error at its first common point, not at each model input origin. Full-coverage stages/tails differ in coordinates; use common scores for path comparisons.'}
    save_json(roots[1]/'input_budget_validation_comparison.json', output)
    print(json.dumps(output['common_interval_summary'], indent=2))


if __name__ == '__main__':
    main()
