import unittest

from train_xjtu_stage_b import protocol_split


class XjtuProtocolSplitTest(unittest.TestCase):
    def test_pre_registered_groups_are_disjoint_and_complete(self):
        rows = [{"group": f"batch{group}", "cell_id": f"b{group}_{cell}"}
                for group in range(1, 7) for cell in range(2)]
        split = protocol_split(rows)
        self.assertEqual([len(split[part]) for part in
                          ("train", "validation", "test")], [8, 2, 2])
        self.assertEqual(split["test"], ["b2_0", "b2_1"])
        self.assertEqual(split["validation"], ["b3_0", "b3_1"])
        self.assertEqual(len(set().union(*map(set, split.values()))), len(rows))

    def test_unknown_group_must_fail(self):
        rows = [{"group": f"batch{group}", "cell_id": f"b{group}"}
                for group in range(1, 8)]
        with self.assertRaisesRegex(ValueError, "differ"):
            protocol_split(rows)


if __name__ == "__main__":
    unittest.main()
