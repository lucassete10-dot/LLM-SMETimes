import unittest
from scripts.screen_nasa_calibration_ensemble import specifications,summarize

class EnsembleBudgetTests(unittest.TestCase):
    def test_fixed_budget_and_members(self):
        parent=[{'candidate':i} for i in range(61)]
        specs=specifications(parent)
        self.assertEqual(len(specs),79)
        self.assertEqual(specs[:61],parent)
        self.assertEqual(len({(tuple(s['members']),s['reducer'],s['weight']) for s in specs[61:]}),18)
        for s in specs[61:]:
            self.assertTrue(all(1<=i<=12 for i in s['members']))
            self.assertIn(s['weight'],(0.,.25,.5))

    def test_no_improvement_does_not_expand(self):
        metrics=dict(score=2.,mape_percent=2.,endpoint_pp=2.,drop_error_pp=2.)
        records=[dict(selected=dict(candidate=37,**metrics),control=dict(candidate=37,**metrics)) for _ in range(4)]
        self.assertFalse(summarize(records)['expand'])
        self.assertEqual(summarize(records)['wins'],0)

if __name__=='__main__':unittest.main()
