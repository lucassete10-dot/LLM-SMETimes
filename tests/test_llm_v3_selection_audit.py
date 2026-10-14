import json,tempfile,unittest
from pathlib import Path
import numpy as np
from battery_smetimes.llm_curve_quality_v3 import freeze_validation_policy,curve_metrics_v3,validation_score
from battery_smetimes.llm_v3_selection_audit import audit_selection


class SelectionAuditTests(unittest.TestCase):
    def fixture(self,root,sparse=False):
        x=np.asarray([101,140] if sparse else np.arange(101,141))
        y=np.ones(len(x));p=y*.99
        cells={'a':{'label_cycle':x,'soh':y}};plan={'validation':['a']}
        policy=freeze_validation_policy(cells,['a'],100)
        row=curve_metrics_v3(x,y,p);score=validation_score({'a':row},policy)
        items={'config.json':{'selection_version':'curve_quality_v3','origin':100,'validation_selection_policy':policy},
            'validation_policy.json':policy,'selected_metrics.json':{'a':row},
            'selection.json':{'epoch':20,'score':[score],'scope':'development validation'},
            'result.json':{'selected_epoch':20},
            'training_history.json':[{'epoch':10,'validation_quality_score':score+1},
                {'epoch':20,'validation_quality_score':score},{'epoch':30,'validation_quality_score':score}]}
        for name,value in items.items():(root/name).write_text(json.dumps(value),encoding='utf-8')
        np.savez(root/'selected_predictions.npz',a__100__cycle=x,a__100__truth=y,a__100__prediction=p)
        return cells,plan

    def test_dense_arithmetic_and_earliest_minimum(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);cells,plan=self.fixture(root)
            self.assertTrue(audit_selection(root,cells,plan)['earliest_minimum_score_verified'])
            selection=json.loads((root/'selection.json').read_text());selection['epoch']=30
            (root/'selection.json').write_text(json.dumps(selection))
            (root/'result.json').write_text(json.dumps({'selected_epoch':30}))
            with self.assertRaisesRegex(ValueError,'earliest minimum'):audit_selection(root,cells,plan)

    def test_sparse_missing_stage_cannot_be_replaced_by_zero(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);cells,plan=self.fixture(root,True)
            self.assertFalse(audit_selection(root,cells,plan)['stage_evidence_complete'])
            metrics=json.loads((root/'selected_metrics.json').read_text());metrics['a']['stage_drop_max_pp']=0.
            (root/'selected_metrics.json').write_text(json.dumps(metrics))
            with self.assertRaisesRegex(ValueError,'missing stage'):audit_selection(root,cells,plan)

    def test_policy_and_cached_score_tampering_detected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);cells,plan=self.fixture(root)
            policy=json.loads((root/'validation_policy.json').read_text());policy['scales']['mae_pp']=20.
            (root/'validation_policy.json').write_text(json.dumps(policy))
            with self.assertRaisesRegex(ValueError,'policy changed'):audit_selection(root,cells,plan)
            self.fixture(root)
            selection=json.loads((root/'selection.json').read_text());selection['score']=[0.]
            (root/'selection.json').write_text(json.dumps(selection))
            with self.assertRaises(AssertionError):audit_selection(root,cells,plan)


if __name__=='__main__':unittest.main()
