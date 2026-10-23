import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from scipy.sparse import csr_matrix
from run_v12 import coverage_fraction, run

class TestV12(unittest.TestCase):
    def test_coverage(self):
        self.assertAlmostEqual(coverage_fraction(csr_matrix([[1,0,2],[0,0,1]])),2/3)
    def test_invalid_seeds(self):
        with self.assertRaises(ValueError):run(seeds=0)
    def test_invalid_iterations(self):
        with self.assertRaises(ValueError):run(iterations=(20,5))
    def test_small_smoke(self):
        with TemporaryDirectory() as d:
            s=run(seeds=1,views=(3,),iterations=(1,),ny=8,size=12,
                  detector_u=24,detector_v=20,samples=15,output=d)
            self.assertTrue(Path(d,'summary.json').exists())
            self.assertEqual(len(s['means']),1)
            self.assertGreater(s['coverage']['3']['fraction'],0)
if __name__=='__main__':unittest.main()
