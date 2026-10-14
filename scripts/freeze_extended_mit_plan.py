"""Bind the immutable signal extension to unchanged whole-cell split plans."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from scripts.prepare_same_type_forecast import save_json, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((args.parent/'config.json').read_text())
    audit = json.loads((args.parent/'audit.json').read_text())
    original = Path(audit['prepared_folder'])
    old = json.loads((original/'manifest.json').read_text())
    new = json.loads((args.prepared/'manifest.json').read_text())
    assert sha(original/'manifest.json') == config['prepared_manifest_sha256']
    records = {r['cell_id']: r for r in new['cells']}
    assert set(records) == {r['cell_id'] for r in old['cells']}
    for record in old['cells']:
        other = records[record['cell_id']]
        assert record['metadata'] == other['metadata']
        assert sha(record['file']) == record['sha256']
        assert sha(other['file']) == other['sha256']
        with np.load(record['file']) as a, np.load(other['file']) as b:
            for key in a.files:
                value = b[key][b['cycle'] <= 100] if key in ('cycle','raw_features') else b[key]
                np.testing.assert_array_equal(a[key], value)
    if not json.loads((args.prepared/'extension_audit.json').read_text())['passed']:
        raise ValueError('extension audit failed')
    shutil.copytree(args.parent, args.output)
    config['prepared_manifest_sha256'] = sha(args.prepared/'manifest.json')
    config['signal_extension_parent_config_sha256'] = sha(args.parent/'config.json')
    audit['prepared_folder'] = str(args.prepared.resolve())
    audit['signal_extension_note'] = 'Only MIT signal budget extended; labels, identity, quality rules and splits unchanged'
    save_json(args.output/'config.json', config)
    save_json(args.output/'audit.json', audit)
    save_json(args.output/'signal_extension_binding.json', {
        'parent_plan': str(args.parent.resolve()), 'prepared': str(args.prepared.resolve()),
        'prefix_and_labels_replay_passed': True,
        'unchanged_splits_sha256': sha(args.output/'splits.json'),
        'unchanged_registry_sha256': sha(args.output/'cell_registry.json')})
    assert sha(args.output/'splits.json') == sha(args.parent/'splits.json')
    print('FROZEN', args.output.resolve())


if __name__ == '__main__':
    main()
