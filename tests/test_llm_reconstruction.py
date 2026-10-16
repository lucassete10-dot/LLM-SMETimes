import unittest
import numpy as np
from battery_smetimes.llm_reconstruction import ReconstructionProcessor,add_targets


class ReconstructionDataTest(unittest.TestCase):
    def test_all_labels_and_signal_only_inputs(self):
        x=np.arange(1,238,dtype=np.float32)
        raw=np.tile(np.linspace(3,4,33,dtype=np.float32),(len(x),1))
        cell=dict(cycle=x,raw=raw,label_cycle=x,soh=1-x*.0002,metadata={'dataset':'MIT'})
        processor=ReconstructionProcessor().fit({'a':cell},['a'])
        inputs=processor.inputs(cell)
        stripped={k:v for k,v in cell.items() if k not in ('label_cycle','soh')}
        for first,second in zip(inputs,processor.inputs(stripped)):
            np.testing.assert_array_equal(first['history'],second['history']);self.assertEqual(first['card'],second['card'])
        contexts=add_targets(inputs,cell)
        np.testing.assert_array_equal(np.concatenate([r['target_cycle'] for r in contexts]),x)
        self.assertAlmostEqual(sum(r['weight'] for r in contexts),1)
        used=set()
        for r in inputs:used.update(r['signal_cycles'].tolist())
        self.assertEqual(used,set(x.tolist()))

    def test_sparse_real_coordinates(self):
        x=np.array([3,25,100,450,900],np.float32)
        raw=np.ones((len(x),33),np.float32)
        cell=dict(cycle=x,raw=raw,label_cycle=np.array([1,20,101,500,901]),soh=np.array([1,.99,.98,.9,.8]),metadata={'dataset':'MIT'})
        p=ReconstructionProcessor().fit({'a':cell},['a']);inputs=p.inputs(cell);rows=add_targets(inputs,cell)
        self.assertEqual(inputs[0]['length'],5)
        np.testing.assert_array_equal(inputs[0]['history'][:5,-1],x/100)
        np.testing.assert_array_equal(rows[0]['target_cycle'],cell['label_cycle'])


if __name__=='__main__':unittest.main()
