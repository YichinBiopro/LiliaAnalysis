"""Independent references and failure-detection checks for M4 calibration."""
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.feature_selection import mutual_info_classif

import spectral_entropy as spectral
from tools.compare_mi_methods import (brute_mixed_mi, capture_event_estimates,
                                      check_histogram, check_null, circular_reference, mixed_radius_audit)
from tools.freeze_mi_profiles import require_outcome


class MIComparisonTests(unittest.TestCase):
    def test_known_discrete_information_and_corruption(self):
        x = np.tile([0., 0., 1., 1.], 64)
        for y, target in [(x, 1.), (np.tile([0., 1., 0., 1.], 64), 0.)]:
            result = spectral.compute_joint_probability(x, y, bins=8)
            self.assertAlmostEqual(result['mutual_information'], target)
            self.assertEqual(check_histogram(result), 'normalized_pmf')
            with self.assertRaises(AssertionError):
                check_histogram({**result, 'mutual_information_mm': 9.})
        result = spectral.compute_joint_probability(np.ones(32), np.ones(32), binning='quantile')
        self.assertEqual(check_histogram(result), 'degenerate_zero_sentinel')

    def test_mixed_reference_matches_public_sklearn_in_one_dimension(self):
        raw = np.random.default_rng(12).normal(size=(80, 1))
        y = np.repeat([0, 1], 40)
        raw[40:] += .7
        for k in (1, 3, 5):
            x = spectral._preprocess_continuous_features(raw, np.random.RandomState(0))
            expected = mutual_info_classif(raw, y, n_neighbors=k, random_state=0)[0]
            self.assertAlmostEqual(brute_mixed_mi(x, y, k), expected, places=12)
            self.assertAlmostEqual(spectral.compute_mi_cd_multivariate(x, y, k), expected, places=12)

    def test_mixed_multivariate_singleton_and_invalid_input(self):
        x = np.random.default_rng(24).normal(size=(31, 3))
        y = np.r_[np.zeros(15), np.ones(15), 2]
        for k in (1, 3):
            self.assertAlmostEqual(brute_mixed_mi(x, y, k), spectral.compute_mi_cd_multivariate(x, y, k), places=12)
        self.assertAlmostEqual(mixed_radius_audit(x, y, 40), spectral.compute_mi_cd_multivariate(x, y, 40), places=12)
        # Retain the discovered numerical boundary witness instead of relaxing tolerance.
        self.assertGreater(abs(brute_mixed_mi(x, y, 40) - mixed_radius_audit(x, y, 40)), 1e-4)
        self.assertEqual(brute_mixed_mi([[0.], [1.]], [0, 1]), 0.)
        with self.assertRaises(ValueError):
            brute_mixed_mi([[np.nan]], [0])
        with self.assertRaises(ValueError):
            brute_mixed_mi(x, y[:-1])

    def test_capture_detects_wrong_estimator_and_keeps_actual_input(self):
        x = np.random.default_rng(7).normal(size=(20, 3))
        y = np.repeat([0, 1], 10)
        arrays = {}
        with capture_event_estimates(spectral, arrays, 'probe'):
            spectral.compute_mi_cd_multivariate(x, y)
        np.testing.assert_array_equal(arrays['probe_features'], x)
        with patch.object(spectral, 'compute_mi_cd_multivariate', return_value=9.):
            with self.assertRaises(AssertionError):
                with capture_event_estimates(spectral, {}, 'bad'):
                    spectral.compute_mi_cd_multivariate(x, y)

    def test_circular_runs_do_not_merge_repeated_labels(self):
        x, y = np.random.default_rng(18).normal(size=(2, 80))
        ids = np.repeat([0, 1, 0], [20, 40, 20])
        expected = circular_reference(spectral, x, y, ids, bins=8, binning='quantile', count=19, seed=7)
        result = spectral.compute_joint_mi_significance(x, y, segment_ids=ids, bins=8,
                                                       binning='quantile', n_surrogates=19, seed=7)
        np.testing.assert_allclose(expected, result['surrogates'], rtol=0, atol=1e-12)
        check_null(result['mutual_information'], expected, result)
        with self.assertRaises(AssertionError):
            check_null(result['mutual_information'], expected, {**result, 'p_value': -1.})
        for count, groups in [(0, ids), (19, np.r_[np.zeros(65), np.ones(15)])]:
            self.assertEqual(len(circular_reference(spectral, x, y, groups, bins=8,
                                                   binning='quantile', count=count, seed=0)), 0)

    def test_null_ddof_and_zero_variance(self):
        check_null(1., [0., 2.], dict(surrogate_mean=1., surrogate_std=1., p_value=2/3, z=0.))
        check_null(1., [0., 2.], dict(surrogate_mean_bits=1., surrogate_std_bits=np.sqrt(2),
                                    surrogate_p_value=2/3, surrogate_z=0.), event=True)
        check_null(0., [0.], dict(surrogate_mean_bits=0., surrogate_std_bits=0.,
                                 surrogate_p_value=1., surrogate_z=np.nan), event=True)

    def test_unexpected_empty_or_failure_cannot_pass_baseline_comparison(self):
        require_outcome('valid', 'complete', 'complete', 1)
        require_outcome('short', 'short_rejected', 'short_rejected', 0)
        for status, count in [('complete', 0), ('short_rejected', 0)]:
            with self.assertRaises(AssertionError):
                require_outcome('invalid', 'complete', status, count)
