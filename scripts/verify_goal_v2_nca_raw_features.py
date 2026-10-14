"""Rebuild NCA early predictors without opening the raw capacity column."""
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scripts.train_tail_boost import NCA
from scripts.prepare_zhu2022_nca import completed_groups, signal_features, prefix
from scripts.prepare_same_type_forecast import sha, save_json
from scripts.verify_goal_v2_raw_features import file_hash


def main():
    root = Path('experiments/goal_v2_nca_raw_features') / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    root.mkdir(parents=True)
    print('RUN=' + str(root.resolve()), flush=True)
    shutil.copy2(__file__, root / 'verification_source.py')
    index = json.loads((NCA / 'index.json').read_text())
    prepared_file = Path(index['source'])
    assert sha(prepared_file) == index['source_sha256']
    original = joblib.load(prepared_file)
    assert set(original) == set(index['cells']) and len(original) == 55
    audit = json.loads(prepared_file.with_name('audit.json').read_text())
    cfg = json.loads(prepared_file.with_name('config.json').read_text())
    raw_dir = Path(audit['raw_directory'])
    intake = json.loads((raw_dir / 'intake_manifest.json').read_text())
    archive = raw_dir / intake['file']
    save_json(root / 'config.json', dict(origin=100, cells=sorted(original),
              input_columns=['time/s', 'Ecell/V', '<I>/mA', 'cycle number'],
              capacity_column_read=False, index_sha256=sha(NCA/'index.json'),
              source_sha256={f:sha(f) for f in ('scripts/prepare_zhu2022_nca.py',
                                               'scripts/verify_goal_v2_nca_raw_features.py')},
              prepared_source_sha256=sha(prepared_file), goal_complete=False))
    rows = []
    try:
        digest = file_hash(archive)
        assert digest == intake['sha256'] == cfg['raw_sha256'] == audit['raw_sha256']
        with zipfile.ZipFile(archive) as zipped:
            members = {Path(n).stem:n for n in zipped.namelist() if n.lower().endswith('.csv')}
            for key in sorted(original):
                entry = index['cells'][key]
                file = NCA / entry['file']
                assert sha(file) == entry['sha256']
                stored = joblib.load(file)
                assert set(stored) == set(original[key])
                for field, value in stored.items():
                    np.testing.assert_array_equal(value, original[key][field])
                cycles, features, rejected = [], [], []
                with zipped.open(members[key]) as stream:
                    reader = pd.read_csv(stream, usecols=['time/s','Ecell/V','<I>/mA','cycle number'], chunksize=200000)
                    for number, group in completed_groups(reader):
                        if number > 100:
                            break
                        if number != int(number) or not np.isfinite(number):
                            raise ValueError('invalid source cycle')
                        try:
                            vector = signal_features(group)
                        except ValueError as error:
                            rejected.append(dict(cycle=int(number), reason=str(error)))
                            continue
                        cycles.append(int(number))
                        features.append(vector)
                cycles, features = np.asarray(cycles), np.asarray(features)
                np.testing.assert_array_equal(cycles, stored['early_cycle'])
                np.testing.assert_array_equal(features, stored['early_raw'])
                np.testing.assert_array_equal(prefix(cycles, features, stored['conditions']), stored['prefix'])
                rows.append(dict(cell_id=key, early_cycles=len(cycles), feature_columns=features.shape[1],
                                 input_column_capacity_absent=True, rejected_input_records=rejected,
                                 prepared_sha256=entry['sha256'], matches=True))
                save_json(root / 'progress.json', dict(terminal=False, cells_completed=len(rows), last_cell=key))
                if len(rows) % 5 == 0:
                    print(f'VERIFIED {len(rows)}/55 {key}', flush=True)
        save_json(root / 'rows.json', rows)
        save_json(root / 'report.json', dict(passed=True, cells=len(rows), raw_file=str(archive.resolve()),
                  raw_sha256=digest, early_inputs_rebuilt_without_capacity_column=True,
                  passive_conversion_exact=True, source_sha256=sha(__file__), goal_complete=False))
        save_json(root / 'progress.json', dict(terminal=True, passed=True, cells_completed=len(rows)))
    except BaseException:
        import traceback
        save_json(root / 'failure.json', dict(traceback=traceback.format_exc(), cells_completed=len(rows)))
        save_json(root / 'rows.json', rows)
        save_json(root / 'progress.json', dict(terminal=True, passed=False, cells_completed=len(rows)))
        raise


if __name__ == '__main__':
    main()
