import copy,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from battery_smetimes.llm_external_schema import external_signal_prefix
from battery_smetimes.research_llm import statistics_cards
from battery_smetimes.llm_local_refinement import local_refinement_optimizer,local_refinement_step


class ExternalContractTests(unittest.TestCase):
    def cell(self):
        return {'cycle':np.array([0.,30.,60.,120.]),'raw':np.arange(16,dtype=np.float32).reshape(4,4),
            'reference_established_source_cycle':0,'soh':np.array([1.,.99,.98,.97]),
            'metadata':{'dataset':'UConn_ISU_ILCC','known_conditions':[2,None,None,None,None],
                'signal_schema':{'width':4,'minimum_observations':1,'text_channel_columns':{'Voltage':[0,1,2,3],'Current':[],'Temperature':[]}}}}

    def test_future_and_labels_cannot_change_input_or_text(self):
        cell=self.cell();poison=copy.deepcopy(cell);poison['raw'][-1]=np.inf;poison.pop('soh')
        np.testing.assert_equal(external_signal_prefix(cell,60),external_signal_prefix(poison,60))
        self.assertEqual(statistics_cards(cell,60),statistics_cards(poison,60))
        self.assertIn('Current unavailable.',statistics_cards(cell,60)[0])
        self.assertIn('Temperature unavailable.',statistics_cards(cell,60)[0])

    def test_single_sparse_observation_retains_actual_cycle(self):
        row=external_signal_prefix(self.cell(),10)
        self.assertEqual(float(row[-8]),0.)
        self.assertAlmostEqual(float(row[-7]),.01,places=7)
        self.assertTrue(np.isfinite(row[:8]).all())

    def test_refinement_freezes_everything_outside_local_head_and_restores_optimizer(self):
        torch.manual_seed(11);net=torch.nn.Module();net.local_head=torch.nn.Linear(2,1);net.other=torch.nn.Linear(2,2)
        inputs=net.other(torch.tensor([[1.,0.],[0.,1.]])).detach();target=torch.tensor([[.4],[-.3]])
        before={k:v.detach().clone() for k,v in net.other.state_dict().items()}
        optimizer=local_refinement_optimizer(net)
        def loss():return ((net.local_head(inputs)-target)**2).mean()
        initial=float(loss().detach())
        for _ in range(5):value,count=local_refinement_step(net,optimizer,loss);self.assertGreaterEqual(count,1)
        self.assertLess(value,initial)
        for k,v in net.other.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)
        self.assertTrue(all(p.grad is None for p in net.other.parameters()))
        restored=local_refinement_optimizer(net);restored.load_state_dict(optimizer.state_dict())
        self.assertEqual(restored.state_dict()['state'][0]['n_iter'],optimizer.state_dict()['state'][0]['n_iter'])


if __name__=='__main__':unittest.main()
