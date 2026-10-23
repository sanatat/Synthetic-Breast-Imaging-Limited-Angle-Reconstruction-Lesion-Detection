import unittest
import numpy as np
from run_v5 import guided_step, reconstruct_v5, lesion_metrics
from run_v4 import phantom_v4, circular_mask
from skimage.transform import radon

class TestV5(unittest.TestCase):
    def test_constant_preservation(self):
        x=np.full((32,32),.4)
        self.assertTrue(np.allclose(guided_step(x,np.zeros_like(x)),x))
    def test_directional_suppresses_across_edge(self):
        x=np.zeros((32,32));x[:,16:]=1
        edge=x.copy()
        a=guided_step(x,edge,edge_scale=.05)
        b=guided_step(x,edge,directional=False)
        self.assertGreater(a[:,16:].mean()-a[:,:16].mean(),b[:,16:].mean()-b[:,:16].mean())
    def test_shape_finite(self):
        truth,lesion,_=phantom_v4(48,15)
        theta=np.linspace(-25,25,9);sino=radon(truth,theta=theta,circle=True);mask=circular_mask(48)
        out=reconstruct_v5(sino,theta,np.zeros_like(truth),truth,mask,outer=1)
        self.assertEqual(out.shape,truth.shape);self.assertTrue(np.isfinite(out).all())
        self.assertTrue(all(np.isfinite(v) for v in lesion_metrics(truth,lesion,out,theta,sino,mask).values()))
if __name__=='__main__':unittest.main()
