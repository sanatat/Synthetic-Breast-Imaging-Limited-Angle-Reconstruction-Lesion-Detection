import unittest
import numpy as np
from run_v14 import poisson_line_integrals,blind_candidate,remove_lesion

class TestV14(unittest.TestCase):
    def test_counts_finite(self):
        from scipy.sparse import eye
        y,c=poisson_line_integrals(eye(8,format='csr'),np.ones((2,2,2),np.float32),np.random.default_rng(1),30000,.035)
        self.assertTrue(np.all(np.isfinite(y)))
        self.assertEqual(c.shape,(8,))
    def test_photons_invalid(self):
        from scipy.sparse import eye
        with self.assertRaises(ValueError):poisson_line_integrals(eye(8),np.zeros((2,2,2)),np.random.default_rng(1),0)
    def test_candidate_without_mask(self):
        x=np.zeros((12,18,18),np.float32);x[6,9,10]=10
        c=blind_candidate(x)
        self.assertEqual(c['center'],[6,9,10])
    def test_negative_removes_added_contrast(self):
        a=np.ones((4,4,4),np.float32)*.75;m=np.zeros_like(a,dtype=bool);m[1,1,1]=True
        b=remove_lesion(a,m)
        self.assertAlmostEqual(float(b[1,1,1]),.2,places=5)
        self.assertEqual(float(b[0,0,0]),.75)

if __name__=='__main__':unittest.main()
