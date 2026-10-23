import unittest
import numpy as np
from run_v7 import phantom,forward,reconstruct,metrics
class TestV7(unittest.TestCase):
    def test_phantom_inside_circle(self):
        v,lesion,_=phantom(3,16,32)
        self.assertTrue(lesion.any())
        z,x=np.ogrid[:32,:32];outside=(z-15.5)**2+(x-15.5)**2>(16-1)**2
        self.assertTrue(np.all(v[:,outside]==0))
    def test_forward_shape_and_linearity(self):
        vol,_,_=phantom(9,12,24)
        th=np.linspace(-20,20,7)
        a=forward(vol,th)
        self.assertEqual(a.shape,(12,24,7))
        self.assertTrue(np.allclose(forward(vol*.5,th),a*.5,atol=1e-5))
    def test_recon_finite_and_metrics(self):
        v,lesion,_=phantom(11,12,24)
        th=np.linspace(-18,18,7);s=forward(v,th)
        r=reconstruct(s,th,'tv',iterations=1)
        self.assertEqual(r.shape,v.shape)
        self.assertTrue(np.isfinite(r).all())
        self.assertTrue(all(np.isfinite(list(metrics(v,lesion,r,th,s).values()))))
if __name__=='__main__':unittest.main()
