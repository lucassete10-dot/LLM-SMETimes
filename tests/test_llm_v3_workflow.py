"""CPU state-machine fixtures do not establish real model accuracy."""
import copy
import unittest
from battery_smetimes.llm_v3_workflow import (new_dataset_state,next_round_robin_action,
    trial_key,confirmation_pass,work_goal,summarize_validation)


def evidence(error=1.):
    rows={str(i):{'mae_pp':error,'tail_mae_pp':error,'start_error_pp':error,
        'endpoint_error_pp':error,'stage_drop_max_pp':error,'mape_percent':2.,
        'late_mape_percent':2.} for i in range(10)}
    return {'by_cell':rows,'policy':{'cell_ids':list(rows),
        'active_metrics':['mae_pp','tail_mae_pp','start_error_pp','endpoint_error_pp','stage_drop_max_pp'],
        'scales':{'mae_pp':2.,'tail_mae_pp':3.,'start_error_pp':1.,'endpoint_error_pp':3.,'stage_drop_max_pp':3.}}}


class WorkflowTests(unittest.TestCase):
    def simulate(self,improve):
        workflow={'sequence':['MIT','XJTU'],'datasets':{}}
        for ds in workflow['sequence']:
            workflow['datasets'][ds]=new_dataset_state(ds,[ds+'_fold1',ds+'_fold2'],evidence(),ds+'_R0',{})
        fits=[];audits=[]
        for _ in range(100):
            action=next_round_robin_action(workflow)
            if action is None:break
            state=workflow['datasets'][action['dataset']]
            if action['kind']=='fit':
                self.assertNotIn('test',action['roles'])
                self.assertNotIn(action['key'],state['trials'])
                good=improve and action['config']=='A'
                state['trials'][action['key']]={'state':'complete','run':'fixture',
                    'evidence':evidence(.8 if good else 1.1 if action['config']!='R0' else 1.)}
                fits.append(action)
            else:
                self.assertIsNotNone(state['selection_lock'])
                state['formal_slots'][action['key']]={'test_verified':True}
                audits.append(action)
            workflow['last_dataset_index']=action['dataset_index']
        self.assertTrue(all(s['phase']=='complete' for s in workflow['datasets'].values()))
        self.assertEqual(len(audits),12)
        return workflow,fits

    def test_winner_round_robin_and_three_confirmation_fits_reused(self):
        workflow,fits=self.simulate(True)
        self.assertTrue(all(s['winner']=='A' for s in workflow['datasets'].values()))
        self.assertEqual(len(fits),2*(2+6+3))
        self.assertEqual([a['dataset'] for a in fits[:4]],['MIT','XJTU','MIT','XJTU'])
        self.assertFalse(any(a['config'] in ('C','D') for a in fits))
        self.assertIsNone(next_round_robin_action(workflow))

    def test_no_winner_keeps_reference_and_still_covers_all_folds(self):
        workflow,fits=self.simulate(False)
        self.assertTrue(all(s['winner']=='R0' for s in workflow['datasets'].values()))
        self.assertEqual(len(fits),2*(2+3+3))

    def test_premature_test_state_cannot_bypass_lock(self):
        state=new_dataset_state('MIT',['fold1'],evidence(),'fixture',{})
        state['phase']='formal';state['winner']='R0'
        with self.assertRaises(ValueError):
            next_round_robin_action({'sequence':['MIT'],'datasets':{'MIT':state}})

    def test_sparse_unknown_shape_never_counts_as_reference_quality(self):
        item=evidence()
        for row in item['by_cell'].values():row['stage_drop_max_pp']=None
        item['policy']['active_metrics'].remove('stage_drop_max_pp')
        self.assertFalse(work_goal([item]))
        self.assertGreater(summarize_validation(item)['score'],0)

    def test_conditional_candidates_have_evidence_and_bound(self):
        state=new_dataset_state('MIT',['f'],evidence(),'fixture',
            {'training_validation_mae_ratio':2.1,'noisy_training_cell_fraction':.3})
        self.assertEqual(state['screen_order'],['A','B','C','D'])

    def test_early_reference_confirmation_failure_reuses_reference_fits(self):
        early=evidence()
        for row in early['by_cell'].values():row['mape_percent']=1.
        state=new_dataset_state('MIT',['f'],early,'fixture',{})
        workflow={'sequence':['MIT'],'datasets':{'MIT':state}}
        requests=[]
        for _ in range(20):
            action=next_round_robin_action(workflow)
            if action['kind']=='evaluate_test':break
            self.assertNotIn(action['key'],state['trials'])
            # R0's three-seed quality confirmation fails. A improves its score;
            # those same R0 fits must be used for the later paired comparison.
            value=2. if action['config']=='R0' and action['epochs']==240 else .8 if action['config']=='A' else 1.1
            state['trials'][action['key']]={'state':'complete','run':'fixture','evidence':evidence(value)}
            requests.append(action)
            workflow['last_dataset_index']=action['dataset_index']
        self.assertEqual(state['winner'],'A')
        self.assertEqual(sum(a['config']=='R0' and a['epochs']==240 for a in requests),3)
        self.assertEqual(len(requests),8)

    def test_early_reference_success_skips_candidate_search(self):
        early=evidence()
        for row in early['by_cell'].values():row['mape_percent']=1.
        state=new_dataset_state('MIT',['f'],early,'fixture',{})
        workflow={'sequence':['MIT'],'datasets':{'MIT':state}}
        for _ in range(3):
            action=next_round_robin_action(workflow)
            self.assertEqual(action['config'],'R0')
            state['trials'][action['key']]={'state':'complete','run':'fixture','evidence':copy.deepcopy(early)}
            workflow['last_dataset_index']=action['dataset_index']
        self.assertEqual(next_round_robin_action(workflow)['kind'],'evaluate_test')
        self.assertEqual(state['winner'],'R0')
        self.assertTrue(state['selection_lock']['curve_work_goal_reached'])

    def test_missing_seed_coverage_and_worst_protection(self):
        a=evidence();b=copy.deepcopy(a);b['by_cell'].pop('0')
        with self.assertRaises(ValueError):work_goal([a,b,a])
        reference=summarize_validation(a)
        candidate=dict(reference,score=reference['score']*.8,worst_mae_pp=3.)
        self.assertFalse(confirmation_pass([candidate]*3,[reference]*3))


if __name__=='__main__':unittest.main()
