import unittest
import numpy as np
from run_experiment import phantom, ultrasound_surrogate, xray_recon, diffuse, metrics

class PilotTests(unittest.TestCase):
    def test_phantom_and_lesion(self):
        img,lesion=phantom(64,12)
        self.assertEqual(img.shape,(64,64))
        self.assertGreater(lesion.sum(),0)
        self.assertGreaterEqual(img.min(),0)
        self.assertLessEqual(img.max(),1)

    def test_projection_reconstruction(self):
        img,_=phantom(64,12)
        angles,sinogram,fbp,sart=xray_recon(img,np.random.default_rng(3))
        self.assertEqual(sart.shape,img.shape)
        self.assertEqual(fbp.shape,img.shape)
        self.assertTrue(np.isfinite(sart).all())
        self.assertEqual(sinogram.shape[1],len(angles))

    def test_ultrasound_misregistration(self):
        img,_=phantom(64,12)
        a=ultrasound_surrogate(img,np.random.default_rng(0))
        b=ultrasound_surrogate(img,np.random.default_rng(0),misregistration=5)
        self.assertGreater(np.mean(np.abs(a-b)),0.01)

    def test_diffuse_no_artificial_range(self):
        img,lesion=phantom(64,12)
        pred=diffuse(img,img)
        self.assertTrue(np.isfinite(pred).all())
        self.assertGreaterEqual(pred.min(),0)
        self.assertLessEqual(pred.max(),1)
        self.assertTrue(np.isfinite(metrics(img,pred,lesion)['ssim']))

if __name__=='__main__': unittest.main()
