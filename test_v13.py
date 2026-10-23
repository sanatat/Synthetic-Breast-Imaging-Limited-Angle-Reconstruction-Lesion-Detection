import unittest
import tempfile
import numpy as np
from run_v13 import matrices,run
from run_v10 import sirt_trace
from run_v7 import phantom

class TestV13(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.A,cls.B=matrices(3,ny=8,size=12,detector_u=22,detector_v=16,reconstruction_samples=16,simulation_samples=25)
    def test_shapes(self):
        self.assertEqual(self.A.shape,self.B.shape)
        self.assertGreater((self.A-self.B).nnz,0)
    def test_distinct_data(self):
        x=np.random.default_rng(2).uniform(size=self.A.shape[1]).astype(np.float32)
        self.assertGreater(np.linalg.norm(self.A@x-self.B@x),1e-6)
    def test_finite_reconstruction(self):
        truth,mask,_=phantom(33,8,12)
        y=self.B@truth.ravel()
        trace,out=sirt_trace(self.A,y,truth,mask,(1,3))
        self.assertEqual(len(trace),2)
        self.assertTrue(np.isfinite(out).all())
    def test_run_files(self):
        with tempfile.TemporaryDirectory() as temp:
            result=run(seeds=1,views=(3,),iterations=(2,4),output=temp,ny=8,size=12,
                       reconstruction_samples=16,simulation_samples=25)
            self.assertEqual(len(result['means']),4)
            from pathlib import Path
            self.assertTrue((Path(temp)/'summary.json').exists())
    def test_invalid_sampling(self):
        with self.assertRaises(ValueError):
            matrices(3,reconstruction_samples=42,simulation_samples=42)

if __name__=='__main__':unittest.main()
