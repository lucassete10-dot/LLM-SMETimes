import unittest
from unittest.mock import patch
import numpy as np
from threadpoolctl import threadpool_limits
from battery_smetimes.tail_selection import fit
from scripts.verify_neighbor_nested_formal import independent_primary


class IndependentRefitTest(unittest.TestCase):
    def test_independent_equal_cell_supervision_matches_trainer(self):
        cells={key:dict(physical_family='NCA',key=key,early_raw=np.zeros((1,1))) for key in ('a','b','c')}
        def labels(cell):
            c=np.arange(101,101+{'a':120,'b':80,'c':150}[cell['key']],dtype=float)
            return c,1-({'a':1,'b':2,'c':3}[cell['key']]*.0005)*(c-100)
        def inputs(cell,future):
            return np.column_stack([future/100,np.full(len(future),{'a':1,'b':2,'c':3}[cell['key']]),np.full(len(future),np.nan)])
        choice=dict(mode='tail',parameters=dict(model='hist_gradient_boosting',max_leaves=15,tail_weight=1,local=False));ids=['a','b','c'];seed=20261005
        with threadpool_limits(limits=2),patch('battery_smetimes.tail_boost.labels',side_effect=labels),patch('battery_smetimes.tail_boost.inputs',side_effect=inputs),patch('scripts.verify_neighbor_nested_formal.labels',side_effect=labels),patch('scripts.verify_neighbor_nested_formal.inputs',side_effect=inputs):
            trained=fit(cells,ids,choice,seed);replay=independent_primary(cells,ids,seed,choice)
        a,b=trained['inner'],replay['inner'];np.testing.assert_array_equal(a['imputer'].statistics_,b['imputer'].statistics_)
        x=inputs(cells['b'],np.arange(101,180));np.testing.assert_allclose(a['model'].predict(a['imputer'].transform(x)),b['model'].predict(b['imputer'].transform(x)),rtol=0,atol=1e-12)
        self.assertEqual(replay['train_ids'],ids)


if __name__=='__main__':
    unittest.main()
