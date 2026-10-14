import unittest
import numpy as np
from tests.test_mit_segment_features import cell
from battery_smetimes.mit_pace_forecast import pace_label,head,pace_predict


class PaceTests(unittest.TestCase):
    def labeled(self,j=0):
        c=cell();c['raw']=c['raw']*(1+j*.01)
        c['label_cycle']=np.arange(101,301)
        c['soh']=1-np.arange(200)/1000
        return c

    def test_pace_is_not_last_observed_cycle(self):
        c=self.labeled();p=pace_label(c)
        self.assertGreater(p,1);self.assertLess(p,100)
        c['soh']=np.ones(200)
        self.assertIsNone(pace_label(c))

    def test_single_spike_not_pace(self):
        c=self.labeled();c['soh']=np.ones(200);c['soh'][30]=.7
        self.assertIsNone(pace_label(c))

    def test_query_labels_life_future_and_q_unused(self):
        cells={str(j):self.labeled(j) for j in range(12)}
        h=head(cells,list(cells),10);c=self.labeled();before=pace_predict(h,c)
        c['label_cycle']=np.inf;c['soh']=np.inf;c['cycle_life']=np.inf
        c['raw'][:,[38,39]]=np.inf
        c['raw']=np.vstack([c['raw'],np.full((1,51),np.inf)])
        c['cycle']=np.r_[c['cycle'],9999]
        self.assertEqual(before,pace_predict(h,c))

    def test_missing_supervision_not_fake_lifetime(self):
        cells={str(j):self.labeled(j) for j in range(12)}
        cells['missing']=self.labeled();cells['missing']['soh']=np.ones(200)
        h=head(cells,list(cells),10)
        self.assertIn('missing',h['train_ids']);self.assertIn('missing',h['unobserved_ids'])
        self.assertNotIn('missing',h['labeled_ids'])
