import json
from datetime import datetime,timezone
from pathlib import Path
import shutil
import unittest
import numpy as np
from battery_smetimes.stm32_forecast import Scale
from scripts.audit_stm32_preprocessing import medians,moments
from scripts.train_stm32_soh_multiseed import inner_preprocessing_provenance
from scripts.prepare_same_type_forecast import save_json,sha


class PreprocessingAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out=Path('experiments/stm32_preprocessing_contract_tests')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        cls.out.mkdir(parents=True);cls.results=[]
        for path in (Path(__file__),Path('scripts/audit_stm32_preprocessing.py'),Path('scripts/train_stm32_soh_multiseed.py')):
            shutil.copy2(path,cls.out/path.name)

    def tearDown(self):
        failures={case.id() for case,_ in self._outcome.result.failures+self._outcome.result.errors}
        self.results.append(dict(test=self.id(),passed=self.id() not in failures))
        save_json(self.out/'results.json',dict(results=self.results,
            source_sha256={p.name:sha(p) for p in self.out.glob('*.py')},
            scope='synthetic preprocessing/provenance contract tests, not training or independent-cell evaluation'))

    def test_empty_column_and_weighted_statistics(self):
        x=np.array([[1.,np.nan,8.],[2.,np.nan,8.],[9.,np.nan,np.nan]],dtype=np.float32)
        w=np.array([.1,.2,.7]);scale=Scale().fit(x,w)
        fill,mean,std=moments(x,w)
        np.testing.assert_array_equal(fill,scale.fill)
        np.testing.assert_allclose(mean,scale.mean,rtol=1e-7,atol=1e-7)
        np.testing.assert_allclose(std,scale.std,rtol=1e-7,atol=1e-7)
        self.assertEqual(fill[1],0);self.assertEqual(std[2],1)
        self.assertAlmostEqual(float(mean[0]),6.8,places=5)
        np.testing.assert_array_equal(medians(x,np.float64),[2.,0.,8.])

    def test_provenance_serializes_actual_parameters_and_train_hashes(self):
        plan=dict(dataset='MIT',train=['a'],validation=['b'],test=['c'])
        records={k:dict(prepared_sha256=k+'-hash') for k in ('a','b','c')}
        scale=Scale().fit(np.arange(80,dtype=np.float32).reshape(2,40),np.ones(2))
        result=json.loads(json.dumps(inner_preprocessing_provenance(plan,records,'family',scale)))
        self.assertEqual(result['train_prepared_sha256'],{'a':'a-hash'})
        self.assertEqual(result['train_ids'],['a']);self.assertEqual(result['validation_ids'],['b'])
        np.testing.assert_array_equal(result['mean'],scale.mean)
        self.assertNotIn('test',result)

    def test_overlap_cannot_be_claimed_train_only(self):
        plan=dict(dataset='MIT',train=['a'],validation=['b'],test=['a'])
        with self.assertRaises(ValueError):inner_preprocessing_provenance(plan,{},'family',None)


if __name__=='__main__':unittest.main()
