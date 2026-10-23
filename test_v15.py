import unittest
import numpy as np
from scipy.sparse import eye
from run_v15 import stress_pair,sirt_blind,classification_metrics

class TestV15(unittest.TestCase):
    def test_lesion_is_present_and_pair_shapes(self):
        a,b,mask,info=stress_pair(2)
        self.assertEqual(a.shape,b.shape)
        self.assertGreater(mask.sum(),0)
        self.assertTrue(np.all(a>=b))
        self.assertEqual(info['lesion_voxels'],int(mask.sum()))
    def test_deterministic_phantoms(self):
        a,b,m,_=stress_pair(3); aa,bb,mm,_=stress_pair(3)
        np.testing.assert_array_equal(a,aa)
        np.testing.assert_array_equal(b,bb)
        np.testing.assert_array_equal(m,mm)
    def test_blind_reconstruction_no_mask(self):
        a=eye(8,format='csr');x=sirt_blind(a,np.ones(8,dtype=np.float32),(2,2,2),steps=8)
        self.assertEqual(x.shape,(2,2,2));self.assertTrue(np.isfinite(x).all())
    def test_metrics_counts(self):
        rows=[{'label':'positive','candidate_score':.8,'center_error_vox':1.},
              {'label':'negative','candidate_score':.2}]
        r=classification_metrics(rows,.5)
        self.assertEqual(r['true_positive'],1);self.assertEqual(r['false_positive'],0)
    def test_no_mask_in_blind_reconstruction(self):
        with self.assertRaises(TypeError):sirt_blind(eye(8),np.ones(8),(2,2,2),mask=np.ones(8))

if __name__=='__main__':unittest.main()
