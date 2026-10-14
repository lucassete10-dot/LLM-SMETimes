import json
import unittest
from pathlib import Path
from urllib.parse import urlparse

import yaml

from scripts.build_data_coverage_report import render


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {
    "dataset_name", "official_url", "publication", "publication_year",
    "download_url_or_access_method", "license", "citation_requirement",
    "chemistry", "form_factor", "nominal_capacity", "cell_count",
    "temperature_conditions", "charge_protocols", "discharge_protocols",
    "EOL_definition", "raw_V_available", "raw_I_available",
    "raw_T_available", "capacity_available", "DCIR_available",
    "EIS_available", "estimated_size", "access_status", "notes",
}


class DatasetRegistryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = yaml.safe_load((ROOT / "metadata/dataset_registry.yaml").read_text(encoding="utf-8"))
        cls.licenses = yaml.safe_load((ROOT / "metadata/licenses.yaml").read_text(encoding="utf-8"))

    def test_every_candidate_has_required_fields_and_original_host(self):
        ids = []
        for row in self.registry["datasets"]:
            self.assertFalse(REQUIRED - set(row), row["dataset_id"])
            ids.append(row["dataset_id"])
            self.assertNotIn("kaggle", urlparse(row["official_url"]).netloc.lower())
            self.assertNotIn("github", urlparse(row["official_url"]).netloc.lower())
            self.assertNotEqual(row["access_status"], "ingested")
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), set(self.licenses["licenses"]))

    def test_license_and_technical_eligibility_are_separate(self):
        for row in self.registry["datasets"]:
            status = self.licenses["licenses"][row["dataset_id"]]
            self.assertFalse(status["training_eligible"])
            if status["license"] == "unverified":
                self.assertFalse(status["training_allowed"])
            if row["cell_count"] is not None:
                self.assertLessEqual(status["raw_data_downloaded_cells"], row["cell_count"])

    def test_coverage_excludes_untrained_candidates(self):
        manifest = json.loads((ROOT / "battery_data_v2/manifest.json").read_text(encoding="utf-8"))
        exact80 = json.loads((ROOT / "battery_models_v2/exact80_target_audit.json")
                             .read_text(encoding="utf-8"))
        output = render(manifest, self.registry, exact80_audit=exact80)
        self.assertIn("Total: 197 cells, 124681 retained capacity observations.", output)
        self.assertIn("| MIT | 43 | 91 | 122 |", output)
        self.assertIn("| XJTU | 0 | 55 | 55 |", output)
        self.assertIn("Candidate domains, not yet ingested", output)
        self.assertNotIn("Total: 508", output)


if __name__ == "__main__":
    unittest.main()
