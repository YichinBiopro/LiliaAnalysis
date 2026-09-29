"""M3 controls detect cross-gap contamination and incompatible power units."""
import unittest

import numpy as np

from lilia.goertzel import goertzel_power
from tools.compare_goertzel_methods import compare_smoothing, power_measures, reference_power, smooth
from tools.freeze_goertzel_profiles import check_case


class GoertzelMethodComparisonTests(unittest.TestCase):
    def test_segment_candidate_has_no_cross_boundary_influence(self):
        values=np.array([0.,0.,100.,100.]);groups=np.array([0,0,1,1]);good=np.ones(4,bool)
        old,new,summary=compare_smoothing(values,good,groups)
        np.testing.assert_array_equal(old,[0,50,50,100])
        np.testing.assert_array_equal(new,values)
        self.assertEqual(summary['changed_indices'],[1,2])
        values[2:]=[-1000,2000]
        np.testing.assert_array_equal(smooth(values,good,groups,segmented=True)[:2],[0,0])

    def test_continuous_candidate_retains_masked_positions_and_even_window_semantics(self):
        values=np.array([1.,2.,np.nan,100.,200.]);good=np.array([True,True,False,True,False])
        groups=np.zeros(5,int)
        for win in (1,2,5,10):
            old,new,summary=compare_smoothing(values,good,groups,win)
            np.testing.assert_array_equal(old,new)
            self.assertTrue(np.isnan(new[~good]).all())
            self.assertEqual(summary['changed'],0)

    def test_repeated_segment_labels_do_not_reconnect_disjoint_runs(self):
        out=smooth([1,100,3],[True]*3,[0,1,0],segmented=True)
        np.testing.assert_array_equal(out,[1,100,3])
        np.testing.assert_array_equal(smooth([],[],[],segmented=True),[])
        self.assertTrue(np.isnan(smooth([1,2],[False,False],[0,1],segmented=True)).all())

    def test_bad_shape_and_window_are_rejected(self):
        for win in (0,1.5,True):
            with self.assertRaises(ValueError):smooth([1],[True],[0],win)
        with self.assertRaises(ValueError):smooth([1,2],[True],[0,1])

    def test_arbitrary_frequency_matches_direct_dtft_and_detects_wrong_power(self):
        t=np.arange(2500)/500
        for target in (60.,60.3):
            x=2*np.sin(2*np.pi*target*t)
            power=goertzel_power(x,target,500.)
            np.testing.assert_allclose(power,reference_power(x,target,500.),rtol=1e-8,atol=1e-8)
            row=power_measures(x,power,target=target)
            self.assertAlmostEqual(row['linear']['coherent_mean_square'],2.,places=4)
            with self.assertRaises(AssertionError):power_measures(x,power*2,target=target)

    def test_units_and_floor_remain_explicit(self):
        n=2500;x=2*np.sin(2*np.pi*60*np.arange(n)/500)
        row=power_measures(x,goertzel_power(x,60,500))
        self.assertAlmostEqual(row['db_offsets']['per_n_squared'],-20*np.log10(n))
        self.assertNotEqual(row['db']['legacy'],row['db']['one_sided_density'])
        zero=power_measures(np.ones(250),0.)
        self.assertTrue(all(zero['floor_active'].values()))
        self.assertEqual(set(zero['db'].values()),{-120.})
        with self.assertRaises(ValueError):reference_power([1,np.nan])
        with self.assertRaises(ValueError):power_measures([1,2],0.)

    def test_empty_and_unexpected_failure_cannot_pass_equal_outputs(self):
        check_case(dict(name='short',status='empty'),'empty',0)
        check_case(dict(name='bad',status='nonfinite_rejected'),'nonfinite_rejected',0)
        for status,count in [('empty',0),('complete',0),('nonfinite_rejected',0)]:
            with self.assertRaises(AssertionError):check_case(dict(name='real'),status,count)
