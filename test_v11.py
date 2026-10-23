import unittest
import numpy as np
from scipy.sparse import csr_matrix
from run_v11 import sensitivity_volume, slice_coverage, run

class TestV11(unittest.TestCase):
    def test_visibility_known_matrix(self):
        A=csr_matrix([[1.,0.,2.,0.],[0.,0.,1.,0.]])
        s,mask=sensitivity_volume(A,(1,2,2))
        self.assertEqual(mask.tolist(),[[[True,False],[True,False]]])
        self.assertEqual(float(s.sum()),4.)
    def test_shape_mismatch(self):
        with self.assertRaises(ValueError): sensitivity_volume(csr_matrix((2,3)),(2,2,2))
    def test_slice_profiles(self):
        mask=np.zeros((2,3,4),bool);mask[0]=True
        profiles=slice_coverage(mask)
        self.assertEqual(profiles['y'],[1.,0.])
        self.assertEqual(profiles['z'],[.5,.5,.5])
    def test_invalid_detector_pairing(self):
        with self.assertRaises(ValueError):run(detector_widths=(10,12),detector_heights=(12,))

if __name__=='__main__': unittest.main()
