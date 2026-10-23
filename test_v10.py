import unittest
import numpy as np
from scipy import sparse
from run_v10 import audit_operator, sirt_trace

class TestV10(unittest.TestCase):
    def test_adjoint(self):
        A=sparse.csr_matrix(np.array([[1.,2.,0.],[0.,3.,4.]],dtype=np.float32))
        self.assertLess(audit_operator(A)['adjoint_relative_error'],1e-5)
    def test_zero_data_convergence(self):
        A=sparse.identity(8,format='csr',dtype=np.float32)
        truth=np.zeros((2,2,2),dtype=np.float32)
        lesion=np.zeros_like(truth,dtype=bool);lesion[0,0,0]=True
        # avoid oracle-shell empty in evaluate for tiny shape: test with larger volume instead
        A=sparse.identity(512,format='csr',dtype=np.float32)
        truth=np.zeros((8,8,8),dtype=np.float32)
        lesion=np.zeros_like(truth,dtype=bool);lesion[4,4,4]=True
        trace,pred=sirt_trace(A,np.zeros(512,dtype=np.float32),truth,lesion,(1,2,4))
        self.assertEqual(len(trace),3)
        self.assertTrue(np.allclose(pred,0))
    def test_sensitivity_for_unseen_voxel(self):
        A=sparse.csr_matrix(np.array([[1.,0.,2.]],dtype=np.float32))
        self.assertAlmostEqual(audit_operator(A)['voxel_coverage_fraction'],2/3)

if __name__=='__main__': unittest.main()
