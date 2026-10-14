import copy
from datetime import datetime,timezone
from pathlib import Path
import unittest
import shutil
import numpy as np
from battery_smetimes.external_soh import load_context,read_signals,route,forecast
from scripts.prepare_same_type_forecast import save_json,sha
from scripts.evaluate_release_soh import evaluate


class ExternalSOHTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context=load_context(Path('deployment/stm32_release_trees/20261004T083706_476897Z'))
        cls.cell=cls.context['cells']['b1c11'];cls.family=cls.cell['physical_family']
        cls.request=dict(cell_id='external_example_not_an_actual_new_test',specifications=copy.deepcopy(cls.context['registry']['families'][cls.family]['specifications']),
            operating_conditions=dict(charge_rate_steps_c=[5.4,3.0],charge_switch_soc=.5,diagnostic_discharge_rate_c=4.,ambient_temperature_c=30.))
        cls.out=Path('experiments/external_soh_contract_tests')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');cls.out.mkdir(parents=True);cls.results=[]
        for file in [Path(__file__),Path('battery_smetimes/external_soh.py'),Path('scripts/evaluate_release_soh.py')]:shutil.copy2(file,cls.out/file.name)

    def tearDown(self):
        failures={case.id() for case,_ in self._outcome.result.failures+self._outcome.result.errors}
        self.results.append(dict(test=self.id(),status='failed' if self.id() in failures else 'passed'))
        save_json(self.out/'results.json',dict(results=self.results,source_sha256={p.name:sha(p) for p in self.out.glob('*.py')},scope='interface synthetic/public tests, not new-cell accuracy evaluation'))

    def route(self,request=None,cycle=None,raw=None):
        return route(self.context,self.cell['cycle'] if cycle is None else cycle,self.cell['raw'] if raw is None else raw,self.request if request is None else request)

    def test_renamed_training_signals_not_independent(self):
        cell,provenance=self.route();self.assertFalse(provenance['independent_new_cell']);self.assertEqual(provenance['public_signal_match'],'b1c11')
        result=forecast(self.context,cell,provenance,200);self.assertEqual(len(result['predicted_soh']),100);self.assertTrue(result['warnings'])

    def test_specs_cannot_be_replaced_by_dataset_name(self):
        bad=copy.deepcopy(self.request);bad['physical_family']=self.family
        with self.assertRaises(ValueError):self.route(bad)
        bad=copy.deepcopy(self.request);bad['specifications']['chemistry']='NCM532';bad['specifications']['nominal_capacity_ah']=2.
        with self.assertRaises(ValueError):self.route(bad)

    def test_xjtu_unverified_part_number_not_automatically_matched(self):
        bad=copy.deepcopy(self.request);bad['specifications']=copy.deepcopy(self.context['registry']['families']['LISHEN_18650_NCM532_2Ah_XJTU_cohort']['specifications'])
        bad['specifications']['model']='unsourced_part_number'
        with self.assertRaises(ValueError):self.route(bad)

    def test_unknown_conditions_rejected(self):
        for key,value in [('ambient_temperature_c',25.),('charge_rate_steps_c',[99.,3.])]:
            bad=copy.deepcopy(self.request);bad['operating_conditions'][key]=value
            with self.assertRaises(ValueError):self.route(bad)

    def test_public_replay_requires_exact_signal_match(self):
        cell,provenance=route(self.context,self.cell['cycle'],self.cell['raw'],public_cell='b1c11');self.assertEqual(provenance['mode'],'public_replay')
        altered=self.cell['raw'].copy();altered[0,0]+=.001
        with self.assertRaises(ValueError):route(self.context,self.cell['cycle'],altered,public_cell='b1c11')

    def test_future_history_and_label_keys_rejected(self):
        path=self.out/'label_contaminated.npz';np.savez_compressed(path,cycle=self.cell['cycle'],raw_features=self.cell['raw'],soh=np.ones(len(self.cell['cycle'])))
        with self.assertRaises(ValueError):read_signals(path)
        path=self.out/'future_history.npz';np.savez_compressed(path,cycle=np.r_[self.cell['cycle'],101],raw_features=np.vstack([self.cell['raw'],self.cell['raw'][-1]]))
        with self.assertRaises(ValueError):read_signals(path)

    def test_insufficient_external_history_rejected(self):
        with self.assertRaises(ValueError):self.route(cycle=self.cell['cycle'][:3],raw=self.cell['raw'][:3])

    def test_extrapolation_warns_not_lifetime_classification(self):
        cell,provenance=self.route();result=forecast(self.context,cell,provenance,2300,100)
        self.assertTrue(any('exceed development support' in w for w in result['warnings']));self.assertNotIn('predicted_total_life',result)

    def test_large_errors_not_deleted_from_evaluation(self):
        cell,provenance=self.route();result=forecast(self.context,cell,provenance,103)
        labels=np.full(3,.5);info=dict(cell_id=result['cell_id'],normalization='initial_comparable_capacity',reference_rule='first_comparable')
        scores=evaluate(result,np.array([101,102,103]),labels,info)
        self.assertGreater(scores['metrics']['mape_percent'],50);self.assertEqual(len(scores['matched_cycles']),3);self.assertFalse(scores['excluded_labels'])

    def test_label_reference_mismatch_rejected(self):
        cell,provenance=self.route();result=forecast(self.context,cell,provenance,103)
        with self.assertRaises(ValueError):evaluate(result,np.array([101,102,103]),np.ones(3),dict(cell_id=result['cell_id'],normalization='nominal_capacity'))

    def test_unit_error_and_nan_label_reporting(self):
        cell,provenance=self.route();result=forecast(self.context,cell,provenance,103);info=dict(cell_id=result['cell_id'],normalization='initial_comparable_capacity',reference_rule='first_comparable')
        with self.assertRaises(ValueError):evaluate(result,np.array([101,102,103]),np.full(3,80.),info)
        report=evaluate(result,np.array([101,102,103]),np.array([1.,np.nan,1.]),info)
        self.assertEqual(len(report['excluded_labels']),1);self.assertEqual(report['matched_cycles'],[101,103])


if __name__=='__main__':unittest.main()
