"""Legacy numeric fidelity and explicit invalid quality diagnostics."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import warnings

import numpy as np
from lilia import quality
from lilia.provenance import file_sha256

FIXTURES = Path(__file__).parent / 'fixtures'


class QualityDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.data = np.random.default_rng(18018).normal(size=(4, 1000))

    def score(self, data=None, **kwargs):
        return quality.get_eeg_quality_index_v2_parametric(self.data if data is None else data, **kwargs)

    def only(self, component):
        return {key + '_weight': float(key == component) for key in ('flat', 'spectrum', 'kurtosis', 'corr')}

    def test_all_frozen_arrays_exact_including_invalid_values(self):
        meta = json.loads((FIXTURES / 'quality_stage18_reference.json').read_text())
        for name, key in [('quality_stage18_inputs.npz', 'input_sha256'),
                          ('quality_stage18_reference.npz', 'reference_sha256')]:
            self.assertEqual(file_sha256(FIXTURES / name), meta[key])
        with np.load(FIXTURES / 'quality_stage18_inputs.npz') as inputs, \
             np.load(FIXTURES / 'quality_stage18_reference.npz') as expected, \
             warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for case in meta['cases']:
                for preset, params in meta['presets'].items():
                    with self.subTest(case=case['key'], preset=preset):
                        result = self.score(inputs[case['key']], fs=case['fs'], params=params, stage=case['stage'])
                        prefix = case['key'] + '__' + preset + '__'
                        np.testing.assert_array_equal(result['overall'], expected[prefix + 'overall'])
                        for component, values in result['detail'].items():
                            np.testing.assert_array_equal(values, expected[prefix + component])
                        self.assertEqual(result['context']['preset'], preset)
                        json.dumps(result['context'], allow_nan=False)
                        json.dumps(result['invalid_reasons'])

    def test_activity_boundary_preserves_legacy_exclusive_stop(self):
        for length, valid in [(99, False), (100, False), (101, True)]:
            result = self.score(self.data[:, :length], params=self.only('flat'))
            self.assertEqual(result['valid'].tolist(), [valid] * 4)
            if not valid:
                self.assertIn('flat:no_activity_subwindows', result['invalid_reasons'][0])
                self.assertTrue(np.isnan(result['usable_overall']).all())
                self.assertTrue(np.isfinite(result['overall']).all())

    def test_empty_single_sample_and_no_channels(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = self.score(np.empty((2, 0)))
        self.assertFalse(result['valid'].any())
        self.assertIn('empty_input', result['invalid_reasons'][0])
        result = self.score(np.ones((2, 1)), params=self.only('flat'))
        self.assertFalse(result['valid'].any())
        with self.assertRaisesRegex(ValueError, 'one channel'):
            self.score(np.empty((0, 1000)))

    def test_nonfinite_pollution_and_correlation_peer_effect(self):
        data = self.data.copy()
        data[0, 100] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            without_corr = self.score(data)
            with_corr = self.score(data, params=quality.BEST_EEG_QUALITY_V2_MEAN_ABS_CORR_PARAMS)
        self.assertEqual(without_corr['valid'].tolist(), [False, True, True, True])
        self.assertIn('nonfinite_input', without_corr['invalid_reasons'][0])
        self.assertFalse(with_corr['valid'].any())
        self.assertIn('corr:nonfinite_correlation', with_corr['invalid_reasons'][1])
        np.testing.assert_array_equal(without_corr['overall'][1:], self.score()['overall'][1:])

    def test_spectrum_exception_retains_fallback_but_masks_usable(self):
        with patch.object(quality.sp_signal, 'welch', side_effect=RuntimeError('injected')):
            result = self.score(params=self.only('spectrum'))
        np.testing.assert_array_equal(result['overall'], np.full(4, .5))
        self.assertFalse(result['valid'].any())
        self.assertIn('spectrum:exception:RuntimeError', result['invalid_reasons'][0])
        self.assertTrue(np.isnan(result['usable_overall']).all())

    def test_insufficient_bins_and_nonfinite_fit_are_distinct(self):
        with patch.object(quality.sp_signal, 'welch', return_value=(np.array([0., 1.]), np.ones(2))):
            result = self.score(params=self.only('spectrum'))
        self.assertIn('spectrum:insufficient_fit_bins', result['invalid_reasons'][0])
        with patch.object(quality.np, 'polyfit', return_value=(np.nan, 0.)):
            result = self.score(params=self.only('spectrum'))
        self.assertIn('spectrum:nonfinite_slope', result['invalid_reasons'][0])
        self.assertFalse(result['valid'].any())

    def test_kurtosis_exception_and_constant_floor(self):
        with patch.object(quality, 'kurtosis', side_effect=ArithmeticError('injected')):
            result = self.score(params=self.only('kurtosis'))
        np.testing.assert_array_equal(result['overall'], np.full(4, .5))
        self.assertIn('kurtosis:exception:ArithmeticError', result['invalid_reasons'][0])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = self.score(np.zeros_like(self.data), params=self.only('kurtosis'))
        self.assertIn('kurtosis:nonfinite_kurtosis', result['invalid_reasons'][0])
        np.testing.assert_allclose(result['overall'], .2)

    def test_correlation_exception_and_single_channel_identity(self):
        with patch.object(quality.np, 'corrcoef', side_effect=RuntimeError('injected')):
            result = self.score(params=self.only('corr'))
        self.assertIn('corr:exception:RuntimeError', result['invalid_reasons'][0])
        self.assertFalse(result['valid'].any())
        result = self.score(self.data[:1], params=self.only('corr'))
        self.assertTrue(result['valid'][0])
        self.assertEqual(result['overall'][0], 1.)
        self.assertEqual(result['context']['corr_single_channel_policy'], 'identity')

    def test_disabled_components_do_not_invalidate(self):
        with patch.object(quality.sp_signal, 'welch', side_effect=AssertionError('must not call')):
            result = self.score(params=self.only('flat'))
        self.assertTrue(result['valid'].all())
        self.assertEqual(list(result['component_valid']), ['flat'])
        # A computable low score is valid, not an artifact acceptance decision.
        result = self.score(np.zeros_like(self.data), params=self.only('flat'))
        self.assertTrue(result['valid'].all())
        self.assertTrue(np.all(result['overall'] < .5))

    def test_context_stage_preset_and_configuration_identity(self):
        default = self.score()
        self.assertEqual(default['context']['stage'], 'unspecified')
        self.assertEqual(default['context']['preset'], 'default')
        raw, bp = self.score(stage='raw'), self.score(stage='bandpass')
        self.assertNotEqual(raw['context']['config_id'], bp['context']['config_id'])
        np.testing.assert_array_equal(raw['overall'], bp['overall'])
        changed = self.score(params={'flat_activity_k': .4})
        self.assertEqual(changed['context']['preset'], 'custom')
        self.assertNotEqual(changed['context']['config_id'], default['context']['config_id'])
        with self.assertRaises(ValueError):
            self.score(stage='')
        for fs in (0, -1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                self.score(fs=fs)


if __name__ == '__main__':
    unittest.main()
