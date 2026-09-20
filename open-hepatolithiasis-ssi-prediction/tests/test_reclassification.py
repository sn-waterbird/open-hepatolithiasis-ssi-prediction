"""Synthetic examples only; no patient records."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'analysis'))
from reclassification import estimates,compare,pair

class ReclassificationTests(unittest.TestCase):
    def test_hand_calculated_counts_and_slope(self):
        # Events: up, down, tie; non-events: up, down, down.
        y=[1,1,1,0,0,0];old=[.5]*6;new=[.7,.4,.5,.6,.3,.4]
        np.testing.assert_allclose(estimates(y,old,new),[1/3,.1],atol=1e-14)
    def test_model_swap_changes_sign(self):
        y=[1,1,0,0];old=[.1,.4,.5,.7];new=[.7,.2,.6,.3]
        np.testing.assert_allclose(estimates(y,old,new),-estimates(y,new,old))
    def test_identical_predictions_have_zero_improvement(self):
        r=compare([1,1,0,0],[.1,.6,.2,.5],[.1,.6,.2,.5],100)
        for k in ['NRI','IDI']:
            self.assertEqual(r[k]['estimate'],0)
            self.assertEqual(r[k]['CI'],[0,0])
            self.assertIsNone(r[k]['p'])  # undefined estimate / zero SE
    def test_bootstrap_reproducible(self):
        args=([1,1,0,0,0],[.1,.4,.3,.2,.7],[.3,.2,.1,.4,.6])
        self.assertEqual(compare(*args,100),compare(*args,100))
    def test_invalid_probabilities_and_outcomes(self):
        for y,old,new in [([1,1],[.2,.3],[.3,.4]),([1,0],[.2,.3],[1.2,.4]),([1,0],[.2,.3],[float('nan'),.4])]:
            with self.assertRaises(ValueError):estimates(y,old,new)
    def test_pair_by_id_and_reject_mismatch(self):
        final=[{'ID':'b','y_true':'0','prob_uncalib':'.2'},{'ID':'a','y_true':'1','prob_uncalib':'.7'}]
        full=[{'ID':'a','y_true':'1','pred_prob':'.6'},{'ID':'b','y_true':'0','pred_prob':'.3'}]
        split=[{'ID':'a','split':'VAL','Infection':'1'},{'ID':'b','split':'VAL','Infection':'0'}]
        self.assertEqual([r['ID'] for r in pair(final,full,split)],['b','a'])
        self.assertEqual(pair(final,full,split)[0]['p_full'],.3)
        with self.assertRaises(ValueError):pair(final,full+full,split)
        full[0]['y_true']='0'
        with self.assertRaises(ValueError):pair(final,full,split)

if __name__=='__main__':unittest.main()
