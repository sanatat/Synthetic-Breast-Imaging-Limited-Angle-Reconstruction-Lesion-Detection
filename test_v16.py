import unittest
import numpy as np
from run_v16 import METHODS,candidate,confusions
class TestV16(unittest.TestCase):
    def test_methods(self):
        a=np.zeros((12,18,18));a[6,9,9]=1
        for m in METHODS:
            c=candidate(a,m)
            self.assertEqual(len(c['center']),3)
            self.assertTrue(np.isfinite(c['score']))
    def test_uniform(self):
        for m in METHODS:
            self.assertTrue(np.isfinite(candidate(np.ones((12,18,18)),m)['score']))
    def test_unknown_method(self):
        with self.assertRaises(ValueError):candidate(np.zeros((12,18,18)),'bad')
    def test_localization_is_required(self):
        rows=[{'label':'positive','candidates':{'x':{'score':2.,'distance_to_lesion_vox':8}}},
              {'label':'negative','candidates':{'x':{'score':2.,'distance_to_lesion_vox':None}}}]
        d=confusions(rows,'x',1.)
        self.assertEqual(d['localized_true_positives'],0)
        self.assertEqual(d['false_positive_negative_scans'],1)
if __name__=='__main__':unittest.main()
