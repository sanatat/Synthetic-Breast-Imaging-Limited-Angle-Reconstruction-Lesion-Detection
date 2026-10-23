import unittest
import numpy as np
from run_v4 import phantom_v4, synthesize, metrics, circular_mask

class TestV4(unittest.TestCase):
    def test_variable_lesions(self):
        a,la,da=phantom_v4(64,10);b,lb,db=phantom_v4(64,11)
        self.assertTrue(la.any() and lb.any());self.assertFalse(np.array_equal(la,lb))
        self.assertNotEqual(da['amplitude'],db['amplitude'])
    def test_radon_domain(self):
        truth,_,_=phantom_v4(64,10)
        self.assertTrue(np.all(truth[~circular_mask(64)]==0))
    def test_independent_us(self):
        *_,us,shifted,unrelated=synthesize(15,64,21,25,.015)
        self.assertFalse(np.allclose(us,unrelated))
        self.assertFalse(np.allclose(us,shifted))
    def test_metrics(self):
        t,l,_=phantom_v4(64,10)
        from skimage.transform import radon
        theta=np.linspace(-25,25,21);sin=radon(t,theta=theta,circle=True)
        m=metrics(t,l,t,sin,theta)
        self.assertAlmostEqual(m['lesion_mae'],0)
        self.assertTrue(all(np.isfinite(v) for v in m.values()))
if __name__=='__main__':unittest.main()
