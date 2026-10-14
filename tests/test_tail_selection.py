import unittest
from battery_smetimes.tail_selection import winner


class TailSelectionTests(unittest.TestCase):
    def test_tie_keeps_original(self):
        new=dict(score=1.,mode='tail');old=dict(score=1.,mode='reference')
        self.assertIs(winner(new,old),old)

    def test_only_validation_score_matters(self):
        new=dict(score=.8,test_mape=100.,mode='tail')
        old=dict(score=1.,test_mape=0.,mode='reference')
        self.assertIs(winner(new,old),new)
        self.assertIs(winner(dict(score=1.2),old),old)


if __name__=='__main__':unittest.main()
