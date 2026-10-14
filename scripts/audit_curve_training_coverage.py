"""Diagnose early-feature coverage using TRAIN/VALIDATION cells only.

Future labels are descriptive validation targets, never classifier inputs or
filters. Distance is a diagnostic, not a deployable confidence threshold.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from battery_smetimes.curve_fidelity import condition_key, targets
from battery_smetimes.research_baselines import signal_prefix
from scripts.train_curve_fidelity import load_subset
from scripts.prepare_same_type_forecast import save_json, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    args = parser.parse_args()
    out = Path('experiments/curve_training_coverage') / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True)
    shutil.copy2(__file__, out/'audit_source.py')
    evidence = []
    for run in args.runs:
        cfg = json.loads((run/'config.json').read_text())
        plan = cfg['plan']
        cells = load_subset(plan['train']+plan['validation'])
        assert not set(cells) & set(plan['test'])
        x = np.stack([signal_prefix(cells[k], 100) for k in plan['train']]).astype(float)
        imputer = SimpleImputer(strategy='median', keep_empty_features=True)
        z = imputer.fit_transform(x)
        scaler = StandardScaler().fit(z)
        active = np.ptp(z, axis=0) > 1e-6
        z_scaled = scaler.transform(z)[:, active]
        record = dict(fold=plan['name'], train_ids=plan['train'], validation_ids=plan['validation'],
            outer_excluded=plan['test'], config_sha256=sha(run/'config.json'),
            train_protocols=[list(condition_key(cells[k])) for k in plan['train']],
            active_dimensions=int(active.sum()), validation=[])
        for key in plan['validation']:
            cell = cells[key]
            prefix = signal_prefix(cell, 100).astype(float)
            clean = {k:v for k,v in cell.items() if k not in ('soh', 'label_cycle', 'reference_capacity_ah')}
            np.testing.assert_array_equal(prefix, signal_prefix(clean, 100))
            v = imputer.transform(prefix[None])[0]
            lo, hi = z.min(axis=0), z.max(axis=0)
            outside = active & ((v < lo-1e-6) | (v > hi+1e-6))
            distance = np.sqrt(np.mean((z_scaled-scaler.transform(v[None])[:, active])**2, axis=1))
            cyc, y = targets(cell)
            validation = dict(cell_id=key, protocol=list(condition_key(cell)),
                exact_protocol_seen=condition_key(cell) in {condition_key(cells[k]) for k in plan['train']},
                active_feature_range_exceed_fraction=float(outside.sum()/max(active.sum(),1)),
                nearest_train_cell=plan['train'][int(np.argmin(distance))],
                nearest_train_standardized_rms=float(distance.min()),
                future_span=float(cyc[-1]-100), future_count=len(cyc),
                truth_first_soh=float(y[0]), truth_last_soh=float(y[-1]),
                truth_drop_pp=float(100*(y[0]-y[-1])))
            record['validation'].append(validation)
        record['training_label_descriptions'] = []
        for key in plan['train']:
            cyc, y = targets(cells[key])
            record['training_label_descriptions'].append(dict(cell_id=key,
                future_span=float(cyc[-1]-100), truth_first_soh=float(y[0]),
                truth_last_soh=float(y[-1]), truth_drop_pp=float(100*(y[0]-y[-1]))))
        evidence.append(record)
    save_json(out/'coverage.json', dict(folds=evidence, outer_labels_loaded=False,
        note='Train-only scaling; future labels descriptive, not used in distances. Tiny training cohorts make distances unstable. No exclusion or model selection from this diagnostic.'))
    print('OUTPUT='+str(out.resolve()))
    for fold in evidence:
        for row in fold['validation']:
            print(fold['fold'], json.dumps(row, ensure_ascii=False))


if __name__ == '__main__':
    main()
