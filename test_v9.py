import unittest
import numpy as np
from run_v9 import lesion_diagnostics,select_weight
from run_v7 import phantom
class TestV9(unittest.TestCase):
    def test_perfect_truth_oracle_volume(self):
        truth,mask,_=phantom(9,12,18)
        d=lesion_diagnostics(truth,mask,truth)
        self.assertGreater(d['oracle_detected_volume_ratio'],0)
        self.assertIsNotNone(d['oracle_binary_centroid_error'])
    def test_empty_prediction(self):
        truth,mask,_=phantom(9,12,18)
        d=lesion_diagnostics(truth,mask,np.zeros_like(truth))
        self.assertEqual(d['oracle_detected_volume_ratio'],0)
    def test_constraint_disqualifies_tv(self):
        rows=[{'weight':0.,'lesion_mae':.2,'projection_rmse':.1}, {'weight':.01,'lesion_mae':.1,'projection_rmse':.4}]
        selected,_=select_weight(rows,[0.,.01]);self.assertEqual(selected,0.)
    def test_valid_weight_selection(self):
        rows=[{'weight':0.,'lesion_mae':.2,'projection_rmse':.1}, {'weight':.01,'lesion_mae':.1,'projection_rmse':.11}]
        selected,_=select_weight(rows,[0.,.01]);self.assertEqual(selected,.01)
if __name__=='__main__':unittest.main()
