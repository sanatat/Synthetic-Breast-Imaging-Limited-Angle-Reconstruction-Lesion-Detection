import unittest
import numpy as np
from run_v8 import system_matrix, reconstruct, evaluate
from run_v7 import phantom

class TestV8(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.A=system_matrix(ny=10,n=16,views=5,extent=15,detector_u=18,detector_v=12,samples=30)
        cls.truth,cls.lesion,_=phantom(42,10,16)
    def test_dimensions_and_nonnegative_weights(self):
        self.assertEqual(self.A.shape,(5*12*18,10*16*16))
        self.assertGreater(self.A.nnz,1000)
        self.assertGreaterEqual(float(self.A.data.min()),0)
    def test_adjoint_identity(self):
        rng=np.random.default_rng(3)
        x=rng.normal(size=self.A.shape[1]);y=rng.normal(size=self.A.shape[0])
        self.assertAlmostEqual(float(np.dot(self.A@x,y)),float(np.dot(x,self.A.T@y)),places=4)
    def test_reconstruction_nonnegative_and_shape(self):
        y=self.A@self.truth.ravel()
        rec=reconstruct(self.A,y,self.truth.shape,steps=5)
        self.assertEqual(rec.shape,self.truth.shape)
        self.assertTrue(np.isfinite(rec).all())
        self.assertGreaterEqual(float(rec.min()),0)
    def test_projector_self_consistency(self):
        x=self.truth.ravel()
        self.assertTrue(np.allclose(self.A@x,self.A@x))
    def test_metrics(self):
        y=self.A@self.truth.ravel()
        d=evaluate(self.truth,self.lesion,self.truth,self.A,y)
        self.assertAlmostEqual(d['lesion_mae'],0.,places=6)
        self.assertAlmostEqual(d['projection_rmse'],0.,places=5)
if __name__=='__main__': unittest.main()
