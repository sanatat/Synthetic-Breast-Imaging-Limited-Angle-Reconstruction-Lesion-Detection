import unittest
import numpy as np
from run_v2 import edge_diffusion, reconstruct, evaluate
from run_experiment import phantom
from skimage.transform import radon

class TestV2(unittest.TestCase):
    def test_diffusion_constant(self):
        a=np.full((16,16),.5)
        np.testing.assert_allclose(edge_diffusion(a,a),a)
    def test_reconstruction_shapes(self):
        a,_=phantom(32)
        theta=np.linspace(-25,25,9)
        s=radon(a,theta=theta,circle=True)
        out=reconstruct(s,theta,np.zeros_like(a),a,outer=1)
        self.assertEqual(out.shape,a.shape)
        self.assertTrue(np.isfinite(out).all())
    def test_metrics_finite(self):
        a,lesion=phantom(32)
        theta=np.linspace(-25,25,9)
        s=radon(a,theta=theta,circle=True)
        m=evaluate(a,a,lesion,s,theta)
        self.assertLess(m['projection_rmse'],1e-10)
        self.assertAlmostEqual(m['lesion_mae'],0)

if __name__=='__main__': unittest.main()
