import unittest

from audit_operating_metadata_v2 import audit
from battery_v2_features import load_manifest


class OperatingMetadataAuditTests(unittest.TestCase):
    def test_audited_training_contract_has_expected_rates_and_units(self):
        report = audit(load_manifest()['cells'])
        self.assertEqual(report['cells'], 197)
        self.assertEqual(report['domains']['MIT']['diagnostic_discharge_rates_c'], [4.0])
        self.assertEqual(report['domains']['XJTU']['diagnostic_discharge_rates_c'],
                         [0.5, 1.0])
        self.assertEqual(report['domains']['NASA']['diagnostic_discharge_rates_c'], [])
        self.assertEqual(report['domains']['XJTU']['groups']['batch6']['cells'], 8)


if __name__ == '__main__':
    unittest.main()
