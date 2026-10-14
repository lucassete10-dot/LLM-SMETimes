import tempfile
import unittest
from pathlib import Path
from scripts.train_neighbor_nested_formal import baseline_config_path


class BaselineConfigTests(unittest.TestCase):
    def test_historical_predictor_uses_parent_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            model=root/'extra_trees'
            model.mkdir()
            (root/'config.json').touch()
            self.assertEqual(baseline_config_path(model),root/'config.json')

    def test_local_config_has_priority(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            model=root/'model'
            model.mkdir()
            (root/'config.json').touch()
            (model/'config.json').touch()
            self.assertEqual(baseline_config_path(model),model/'config.json')

    def test_missing_config_is_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                baseline_config_path(Path(directory)/'missing')


if __name__=='__main__':
    unittest.main()
