"""Re-extract all 193 MIT/XJTU/NASA early inputs from original MAT files."""
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from scripts.extract_forecast_prefix import extract
from scripts.train_curve_fidelity import AUDIT, load_subset
from scripts.prepare_same_type_forecast import sha, save_json


def file_hash(file):
    digest = hashlib.sha256()
    with open(file, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    root = Path('experiments/goal_v2_raw_features') / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    root.mkdir(parents=True)
    print('RUN=' + str(root.resolve()), flush=True)
    shutil.copy2(__file__, root / 'verification_source.py')
    records_file = AUDIT / 'cell_registry.json'
    records = json.loads(records_file.read_text())['cells']
    assert len(records) == 193
    rows, raw_sources = [], {}
    sources = ['scripts/extract_forecast_prefix.py', 'adapt_nasa_mat_v2.py',
               'prepare_mit.py', 'prepare_xjtu.py', 'scripts/audit_mit_discharge_integrity.py',
               'scripts/train_curve_fidelity.py']
    save_json(root / 'config.json', dict(cells=sorted(records), origin=100,
              registry_sha256=sha(records_file), operation='raw reconstruction, no training or selection',
              source_sha256={f:sha(f) for f in sources}, goal_complete=False))
    try:
        for key in sorted(records):
            record = records[key]
            cell = load_subset([key])[key]
            domain = cell['metadata']['dataset']
            options = dict(cell_index=int(key.split('c')[1])) if domain == 'MIT' else (
                      dict(variable=key) if domain == 'NASA' else {})
            raw_file = Path(record['raw_source'])
            cycle, raw = extract(raw_file, cell['metadata'], 100, **options)
            # The same predeclared integrity exclusions apply on both paths.
            config = json.loads((AUDIT / 'config.json').read_text())
            issues_file = Path(config['config']['integrity_audit'])
            assert sha(issues_file) == config['integrity_sha256']
            issues = json.loads(issues_file.read_text())['issues']
            excluded = [i['cycle'] for i in issues if i['cell_id'] == key]
            good = ~np.isin(cycle, excluded)
            cycle, raw = cycle[good], raw[good]
            take = cell['cycle'] <= 100
            np.testing.assert_array_equal(cycle, cell['cycle'][take])
            np.testing.assert_allclose(raw, cell['raw'][take], rtol=0, atol=1e-7, equal_nan=True)
            if str(raw_file) not in raw_sources:
                raw_sources[str(raw_file)] = dict(file=str(raw_file), size_bytes=raw_file.stat().st_size,
                                                sha256=file_hash(raw_file))
            row = dict(cell_id=key, dataset=domain, points=len(cycle), max_cycle=int(cycle[-1]),
                       feature_columns=raw.shape[1], raw_source=str(raw_file),
                       prepared_sha256=record['prepared_sha256'], matches=True)
            rows.append(row)
            save_json(root / 'progress.json', dict(terminal=False, cells_completed=len(rows), last_cell=key))
            if len(rows) % 10 == 0 or domain == 'NASA':
                print(f'VERIFIED {len(rows)}/193 {key}', flush=True)
        save_json(root / 'rows.json', rows)
        save_json(root / 'raw_source_sha256.json', list(raw_sources.values()))
        save_json(root / 'report.json', dict(passed=True, cells=len(rows),
                  source_sha256=sha(__file__), all_features_reconstructed=True,
                  label_access='extractor reads source cycle/protocol indexing, not capacity/SOH/EOL for feature values',
                  limits='NCA is separate and not covered here; this does not prove every historical preprocessing fit',
                  goal_complete=False))
        save_json(root / 'progress.json', dict(terminal=True, passed=True, cells_completed=len(rows)))
    except BaseException:
        import traceback
        save_json(root / 'failure.json', dict(traceback=traceback.format_exc(), cells_completed=len(rows)))
        save_json(root / 'rows.json', rows)
        save_json(root / 'progress.json', dict(terminal=True, passed=False, cells_completed=len(rows)))
        raise


if __name__ == '__main__':
    main()
