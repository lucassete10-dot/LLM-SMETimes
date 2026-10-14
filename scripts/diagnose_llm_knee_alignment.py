"""TRAIN-only audit: hinge labels versus logistic transition centres.

Free parameters from an existing capacity audit are diagnostic approximations,
not teachers or forecast results. No fitting or heldout data is used here.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from battery_smetimes.llm_knee_supervision import training_knee_target
from scripts.prepare_same_type_forecast import save_json, sha


def compare_interval(parameter, target, origin):
    centre = 5 * np.logaddexp(0., parameter[3])
    width = .2 + np.logaddexp(0., parameter[4])
    valid = bool(target['valid'])
    distance = max(target['lower'] - centre, centre - target['upper'], 0.) if valid else None
    return {'centre_source_cycle': float(origin + 100 * centre),
            'transition_width_cycles': float(100 * width),
            'label_valid': valid,
            'outside_interval_cycles': float(100 * distance) if valid else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--capacity-run', required=True)
    args = parser.parse_args()
    root, capacity = Path(args.run), Path(args.capacity_run)
    cfg = json.loads((root/'config.json').read_text())
    reference = json.loads((capacity/'config.json').read_text())
    if cfg['plan']['train'] != reference['plan']['train'] or cfg['origin'] != reference['origin']:
        raise ValueError('TRAIN identities or origin mismatch')
    fits = json.loads((capacity/'head_capacity_diagnostic.json').read_text())
    assert fits['heldout_labels_used'] is False and fits['inference_use'] is False
    coverage = json.loads((root/'knee_supervision_coverage.json').read_text())['by_cell']
    arrays = np.load(capacity/'TRAIN_head_approximations_NOT_FORECAST.npz')
    rows = []
    for row in fits['cells']:
        cid = row['cell']
        assert cid in cfg['plan']['train']
        synthetic = training_knee_target({'label_cycle': arrays[cid+'__cycle'],
                                         'soh': arrays[cid+'__approximation']}, cfg['origin'])
        rows.append({'cell': cid, 'approximation_mae_pp': row['metrics']['mae_pp'],
                     'original_label': coverage[cid], 'approximation_label': synthetic,
                     'original_alignment': compare_interval(row['parameters'], coverage[cid], cfg['origin']),
                     'approximation_alignment': compare_interval(row['parameters'], synthetic, cfg['origin'])})
    summary = {}
    for name in ('original_alignment', 'approximation_alignment'):
        values = [r[name]['outside_interval_cycles'] for r in rows if r[name]['label_valid']]
        summary[name] = {'valid_count': len(values),
                         'median_outside_cycles': float(np.median(values)),
                         'count_outside_over_20_cycles': sum(v > 20 for v in values)}
    output = {'scope': 'TRAIN-only parameter/auxiliary-definition compatibility; not forecast accuracy',
              'heldout_labels_used': False, 'inference_use': False,
              'capacity_source_sha256': sha(capacity/'head_capacity_diagnostic.json'),
              'summary': summary, 'cells': rows}
    save_json(root/'TRAIN_knee_alignment_diagnostic.json', output)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
