import unittest
import numpy as np
from prepare_mit import GRID,feature_vector,ENHANCED_NAMES
from battery_smetimes.mit_units import mit_prefix_to_amperes,CURRENT_DERIVED_COLUMNS


class MITCurrentUnitTests(unittest.TestCase):
    def test_conversion_agrees_with_reextracting_raw_current_in_amperes(self):
        t=np.arange(0,301,5)/60;current=-4.+.03*np.sin(t)
        voltage=3.5-.03*t;temperature=25+.2*t
        stored=feature_vector(t,current,voltage,temperature,GRID)[None]
        physical=feature_vector(t,1.1*current,voltage,temperature,GRID)[None]
        converted=mit_prefix_to_amperes(stored)
        np.testing.assert_allclose(converted,physical,rtol=2e-7,atol=1e-7)
        unrelated=sorted(set(range(len(ENHANCED_NAMES)))-set(CURRENT_DERIVED_COLUMNS))
        np.testing.assert_array_equal(converted[:,unrelated],stored[:,unrelated])
        np.testing.assert_array_equal(stored,feature_vector(t,current,voltage,temperature,GRID)[None])

    def test_wrong_schema_is_rejected(self):
        with self.assertRaises(ValueError):mit_prefix_to_amperes(np.ones((3,46)))


if __name__=='__main__':unittest.main()
