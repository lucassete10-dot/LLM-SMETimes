import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import freeze_battery_models_v2 as freeze


class FreezeCandidateTests(unittest.TestCase):
    def test_changed_candidate_requires_explicit_archived_replacement(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'model.bin'
            source.write_bytes(b'first candidate')
            manifest = root / 'frozen_candidate_manifest.json'
            archive = root / 'freeze_history'
            selected = {'XJTU': {'soh': 'test', 'rul': 'test'}}
            with (patch.object(freeze, 'FREEZE', manifest),
                  patch.object(freeze, 'ARCHIVE', archive),
                  patch.object(freeze, 'selected_files', return_value=(selected, [source]))):
                freeze.create()
                first = manifest.read_bytes()
                freeze.create()
                self.assertEqual(manifest.read_bytes(), first)
                source.write_bytes(b'second candidate')
                with self.assertRaisesRegex(FileExistsError, 'already exists and differs'):
                    freeze.create()
                self.assertEqual(manifest.read_bytes(), first)
                freeze.create(replace=True)
                self.assertNotEqual(manifest.read_bytes(), first)
                archived = archive / f'{hashlib.sha256(first).hexdigest()}.json'
                self.assertEqual(archived.read_bytes(), first)
                freeze.verify()


if __name__ == '__main__':
    unittest.main()
