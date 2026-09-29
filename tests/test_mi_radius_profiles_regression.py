"""M4-R1 numerical boundary and plot-provenance regression witnesses."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

import spectral_entropy as spectral
from tools.compare_mi_methods import brute_mixed_mi, mixed_radius_audit
from tools.mi_radius_profiles import direct_mixed_mi, null_summary
from tools.freeze_mi_profiles import compare_metadata
from tools.validate_mi_radius_profiles import replay


class RadiusProfileTests(unittest.TestCase):
    def test_metadata_comparison_rejects_method_and_identity_changes(self):
        from copy import deepcopy
        from lilia.entropy_io import config_id
        def capture(code):
            params = dict(code_sha256=code, fs=500., mi_bins=16)
            identity = config_id(params)
            return dict(table=dict(frame=dict(config_id=[identity], value=[.2]),
                metadata=dict(kind='joint_mi', parameters=params, code_sha256=code,
                              config_id=identity, table_sha256=code)))
        old, new = capture('old'), capture('new')
        compare_metadata(old, new, 'old', 'new')
        bad = deepcopy(new)
        bad['table']['frame']['value'] = [.3]
        with self.assertRaises(AssertionError):
            compare_metadata(old, bad, 'old', 'new')
        bad = deepcopy(new)
        bad['table']['metadata']['parameters']['mi_bins'] = 32
        with self.assertRaises(AssertionError):
            compare_metadata(old, bad, 'old', 'new')
        with self.assertRaises(AssertionError):
            compare_metadata(old, new, 'old', 'unexpected')

    def test_small_samples_ties_singletons_and_k_caps(self):
        cases = [
            (np.random.default_rng(24).normal(size=(31, 3)), np.r_[np.zeros(15), np.ones(15), 2]),
            (np.array([[0., 0.], [1., 0.], [-1., 0.], [0., 1.], [0., -1.], [9., 9.]]), [0, 0, 0, 1, 1, 2]),
            (np.zeros((6, 3)), [0, 0, 0, 1, 1, 1]),
            (np.array([[0.], [0.], [1.], [1.], [2.]]), [0, 0, 1, 1, 2]),
            (np.ones((2, 3)), [0, 1]),
        ]
        for x, y in cases:
            for k in (1, 3, 5, 40):
                with self.subTest(shape=x.shape, k=k):
                    np.testing.assert_allclose(direct_mixed_mi(x, y, k), brute_mixed_mi(x, y, k),
                                               atol=1e-12, rtol=1e-12)
                    permutation = np.random.default_rng(7).permutation(len(x))
                    self.assertAlmostEqual(direct_mixed_mi(x[permutation], np.array(y)[permutation], k),
                                           direct_mixed_mi(x, y, k), places=12)

    def test_search_algorithm_boundary_counterexample_is_preserved(self):
        x = np.random.default_rng(24).normal(size=(31, 3))
        y = np.r_[np.zeros(15), np.ones(15), 2]
        results = {a: mixed_radius_audit(x, y, 40, algorithm=a)
                   for a in ('auto', 'brute', 'kd_tree', 'ball_tree')}
        self.assertGreater(abs(results['auto'] - direct_mixed_mi(x, y, 40)), 1e-4)
        self.assertEqual(results['auto'], results['brute'])
        for algorithm in ('kd_tree', 'ball_tree'):
            self.assertAlmostEqual(results[algorithm], direct_mixed_mi(x, y, 40), places=12)
        self.assertEqual(spectral.compute_mi_cd_multivariate(x, y, 40), results['auto'])

    def test_invalid_inputs_and_null_boundaries(self):
        for x, y, k in [([[np.nan]], [0], 1), ([[1.]], [0, 1], 1),
                        ([[1.]], [0], 0), ([[1.]], [0], 1.5), (np.empty((2, 0)), [0, 1], 1)]:
            with self.assertRaises(ValueError):
                direct_mixed_mi(x, y, k)
        self.assertTrue(np.isnan(null_summary([0.])['p']))
        self.assertEqual(null_summary([0., 0.])['p'], 1.)
        self.assertTrue(np.isnan(null_summary([0., 0.])['z']))
        self.assertAlmostEqual(null_summary([1., 0., 2.])['std'], np.sqrt(2))
        with self.assertRaises(ValueError):
            null_summary([])

    def test_complete_null_replay_detects_missing_or_corrupted_null(self):
        x = np.random.default_rng(24).normal(size=(31, 3))
        y = np.r_[np.zeros(15), np.ones(15), 2]
        rng = np.random.RandomState(7)
        rng.standard_normal(size=x.shape)
        labels = [y] + [rng.permutation(y) for _ in range(3)]
        legacy = np.array([spectral.compute_mi_cd_multivariate(x, d, 40) for d in labels])
        summary = null_summary(legacy / np.log(2))
        result = dict(n_neighbors=40, joint_mi_bits=summary['observed'],
                      surrogate_mean_bits=summary['mean'], surrogate_std_bits=summary['std'],
                      surrogate_p_value=summary['p'], surrogate_z=summary['z'])
        row, arrays = replay(x, y, legacy, result, seed=7, count=3)
        self.assertEqual(row['count'], 3)
        np.testing.assert_array_equal(arrays['labels'], labels)
        with self.assertRaises(AssertionError):
            replay(x, y, legacy[:-1], result, seed=7, count=3)
        candidate = [direct_mixed_mi(x, d, 40) for d in labels]
        candidate[-1] += .1
        with patch('tools.validate_mi_radius_profiles.direct_mixed_mi', side_effect=candidate):
            with self.assertRaises(AssertionError):
                replay(x, y, legacy, result, seed=7, count=3)

    def test_plot_uses_actual_run_context_and_leaves_curves_unchanged(self):
        fs = 200.
        raw = np.random.default_rng(6).normal(size=4000)
        with patch.dict(spectral._PROV, {'no_bandpass': True}):
            frame = spectral.run_band_event_mi_pipeline({'ch1': raw}, [2000], fs=fs,
                windows_sec=(2., 4.), sub_sec=.5, sub_step_sec=.25, n_neighbors=1,
                n_surrogates=1, random_state=7, front_bandpass=True)
            figures = []
            with tempfile.TemporaryDirectory() as out, patch.object(Figure, 'savefig',
                    autospec=True, side_effect=lambda fig, *a, **kw: figures.append(fig)):
                spectral.plot_band_event_mi(frame, 'context witness', str(Path(out)/'plot.png'))
            fig = figures[0]
            text = '\n'.join(t.get_text() for t in fig.texts)
            for expected in ('0.5-45 Hz', 'fs=200 Hz', 'Sub-epoch=0.5 s / step=0.25 s',
                             'k requested=1', 'seed=7', '1 label shuffles', 'quality disabled'):
                self.assertIn(expected, text)
            self.assertNotIn('smooth=', text)
            self.assertIn('half-window', fig.axes[0].get_xlabel())
            self.assertIn('amplitude envelopes', fig.axes[0].get_ylabel())
            for line, column in zip(fig.axes[0].lines, ('Joint_MI_KSG_Bits', 'Joint_MI_Sum_Bits'), strict=True):
                np.testing.assert_array_equal(line.get_ydata(), frame[column])
            historical = pd.DataFrame(frame.to_dict('records'))
            self.assertIn('unknown', spectral.event_mi_provenance(historical))


if __name__ == '__main__':
    unittest.main()
