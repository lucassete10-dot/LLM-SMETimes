import copy
import unittest
import torch
import numpy as np

from battery_smetimes.llm_curve_objective_v4 import combine_curve_losses
from battery_smetimes.llm_v4_workflow import RATE_KNOTS, screen_winner, accepted_confirmation
from battery_smetimes.llm_curve_quality_v3 import freeze_validation_policy
from tests.test_llm_query_rate import QueryDecoderTests


class DenserRateQueryTests(QueryDecoderTests):
    def model(self, **kwargs):
        return super().model(rate_knots=RATE_KNOTS, **kwargs)


class LossTests(unittest.TestCase):
    def test_future_poison_preserves_order_and_visible_prefix_with_full_external_cache(self):
        from scripts.diagnose_llm_curve_v4 import poisoned_visible_cell
        cell={'cycle':np.arange(1,301), 'raw':np.arange(600,dtype=float).reshape(300,2),
              'soh':np.ones(300),'label_cycle':np.arange(1,301)}
        poisoned=poisoned_visible_cell(cell,100)
        self.assertTrue(np.all(np.diff(poisoned['cycle'])>0))
        np.testing.assert_array_equal(poisoned['raw'][:100],cell['raw'][:100])
        self.assertTrue(np.isinf(poisoned['raw'][-1]).all())
        self.assertNotIn('soh',poisoned)
        self.assertEqual(len(cell['cycle']),300)

    def test_phase_weights_and_tail_gradient(self):
        point,start,drop,tail=[torch.tensor([v],requires_grad=True) for v in (2.,1.,3.,4.)]
        base=combine_curve_losses(point,start,drop,tail,'full')
        torch.testing.assert_close(base,torch.tensor([6.5]))
        candidate=combine_curve_losses(point,start,drop,tail,'full',drop_weight=1.,tail_weight=1.5)
        torch.testing.assert_close(candidate,torch.tensor([12.]))
        candidate.sum().backward()
        torch.testing.assert_close(drop.grad,torch.tensor([1.]))
        torch.testing.assert_close(tail.grad,torch.tensor([1.5]))
        short=combine_curve_losses(point,start,drop,tail,'short',drop_weight=1.,tail_weight=1.5)
        torch.testing.assert_close(short,torch.tensor([6.]))
        for weight in (-1.,float('inf'),float('nan')):
            with self.assertRaises(ValueError):combine_curve_losses(point,start,drop,tail,'full',tail_weight=weight)


class DecisionTests(unittest.TestCase):
    def evidence(self,scale=1.,membership=('a','b')):
        policy={'cell_ids':list(membership),'active_metrics':['mae_pp'],'scales':{'mae_pp':2.},'origin':100}
        rows={cid:dict(mae_pp=2*scale,mape_percent=2*scale) for cid in membership}
        return dict(policy=policy,by_cell=rows)

    def test_screen_rejects_small_gain_and_worse_worst_cell(self):
        base=self.evidence()
        self.assertIsNone(screen_winner({'K':self.evidence(.98),'L':self.evidence(1.1)},base))
        self.assertEqual(screen_winner({'K':self.evidence(.8),'L':self.evidence(.7)},base),'L')
        bad=self.evidence(.1);bad['by_cell']['b']['mae_pp']=3.
        self.assertIsNone(screen_winner({'K':bad,'L':self.evidence(1.1)},base))

    def test_three_seed_confirmation_checks_exact_membership_and_protection(self):
        refs=[self.evidence() for _ in range(3)]
        candidates=[self.evidence(.8) for _ in range(3)]
        self.assertTrue(accepted_confirmation(candidates,refs))
        candidates[2]=self.evidence(1.2)
        self.assertFalse(accepted_confirmation(candidates,refs))
        with self.assertRaises(ValueError):accepted_confirmation([self.evidence(.8,('x','y'))]*3,refs)
        with self.assertRaises(ValueError):accepted_confirmation(candidates[:2],refs)


if __name__=='__main__':unittest.main()
