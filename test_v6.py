import unittest
import numpy as np
from skimage.transform import radon
from run_v3 import circular_mask
from run_v4 import phantom_v4
from run_v6 import reconstruct,evaluate
class TestV6(unittest.TestCase):
 def test_tv_shape_finite(self):
  n=48; mask=circular_mask(n); gt,lesion,_=phantom_v4(n,500); theta=np.linspace(-25,25,11); sino=radon(gt,theta=theta,circle=True)
  for method in ('tv','sart'):
   x=reconstruct(sino,theta,n,mask,method,iterations=2)
   self.assertEqual(x.shape,gt.shape);self.assertTrue(np.isfinite(x).all())
   self.assertGreaterEqual(x.min(),0);self.assertLessEqual(x.max(),1)
   self.assertTrue(np.isfinite(evaluate(gt,lesion,x,sino,theta,mask)['ssim']))
 def test_mask_zero_boundary(self):
  n=48;gt,_,_=phantom_v4(n,509);mask=circular_mask(n)
  self.assertTrue(np.all(gt[~mask]==0))
if __name__=='__main__':unittest.main()
