import io
from pathlib import Path
import unittest
import zipfile

import numpy as np

from parsers.inl_nmc_44 import archive_index, first_confirmed_candidate, parse_archive


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "inl_nmc_44"


class InlNmcParserTest(unittest.TestCase):
    def test_candidate_is_not_guaranteed_terminal_event(self):
        cycle = np.arange(1, 15)
        soh = np.array([1, .95, .90, .79, .77, .76, .74, .73, .72, .70,
                        .88, .87, .86, .85])
        self.assertEqual(first_confirmed_candidate(cycle, soh), 4)
        self.assertTrue(np.any(soh[4:] > .82))

    def test_missing_archive_channels_rejected(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("Battery raw data/P462_NMC532_R2 design/Capacity_Cell04.csv", "x")
        buffer.seek(0)
        with zipfile.ZipFile(buffer) as archive:
            with self.assertRaisesRegex(ValueError, "missing channel"):
                archive_index(archive)

    @unittest.skipUnless((RAW / "Battery raw data.zip").exists() and
                         (RAW / "Pouch cell_summary.xlsx").exists(),
                         "official checksum-verified download not present")
    def test_official_population_and_no_trainable_rul(self):
        cells, rows = parse_archive(RAW / "Battery raw data.zip",
                                    RAW / "Pouch cell_summary.xlsx")
        self.assertEqual((len(cells), len(rows)), (44, 24525))
        self.assertEqual(len({row["cell_id"] for row in cells}), 44)
        self.assertEqual(sum(row["candidate80_median5_reference_cycle"] is not None
                             for row in cells), 6)
        self.assertEqual(sum(row["recovered_above82pct_after_candidate"]
                             for row in cells), 4)
        self.assertTrue(all(row["rul80_cycles"] is None and
                            row["soh_label_approved"] is False for row in rows))


if __name__ == "__main__":
    unittest.main()
