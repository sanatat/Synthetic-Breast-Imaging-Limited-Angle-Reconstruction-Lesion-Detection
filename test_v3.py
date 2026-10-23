import unittest
import numpy as np
from run_v3 import circular_mask, synthesize, lesion_statistics, paired_stats

class TestV3(unittest.TestCase):
    def test_mask_inside_circle(self):
        m = circular_mask(64)
        self.assertFalse(m[0,0]);self.assertTrue(m[32,32])
    def test_phantom_and_us_shape(self):
        t,lesion,mask,theta,sino,sart,us,shifted,unrelated=synthesize(0,64,11,25,.015)
        self.assertEqual(t.shape,us.shape)
        self.assertTrue(np.all(t[~mask]==0))
        self.assertTrue(np.any(lesion))
        self.assertGreater(float(np.mean(np.abs(us-shifted))),0)
        self.assertGreater(float(np.mean(np.abs(us-unrelated))),0)
        score=lesion_statistics(t,sart,lesion,sino,theta,mask)
        self.assertTrue(all(np.isfinite(v) for v in score.values()))
    def test_paired_direction(self):
        rec=[{'method':m,'ssim':v,'lesion_mae':1-v} for m,v in [('xray_only',.6),('ultrasound',.7),('xray_only',.5),('ultrasound',.6)]]
        # Include all listed methods to meet input contract; others can be empty.
        s=paired_stats(rec,'ssim',boots=30)
        self.assertAlmostEqual(s['mean_improvement'],.1)
        e=paired_stats(rec,'lesion_mae',boots=30)
        self.assertAlmostEqual(e['mean_improvement'],.1)

if __name__=='__main__': unittest.main()
