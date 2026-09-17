"""Independent PSD controls and failures that equal old/new captures can conceal."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from tools.compare_psd_sensitivity import assert_reference, measurements, reference_welch
from tools.freeze_psd_profiles import bind_model, check_coverage, normalize


class PsdProfileFreezeTests(unittest.TestCase):
    def test_explicit_model_overrides_definition_time_frozen_default(self):
        def inference(values, tflite_path='/frozen/missing.tflite'):
            return values, tflite_path
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp)/'model.tflite'
            model.write_bytes(b'model path witness')
            pipeline = SimpleNamespace(apply_tflite_windowed=inference)
            bind_model(pipeline, model)
            values = np.arange(4)
            actual, used = pipeline.apply_tflite_windowed(values)
            self.assertIs(actual, values)
            self.assertEqual(used, str(model.resolve()))
            with self.assertRaises(FileNotFoundError):
                bind_model(pipeline, Path(tmp)/'missing')

    def test_missing_duplicate_and_invented_panels_cannot_pass_by_equality(self):
        case = dict(name='continuous', model_expected='complete', quality_expected='complete')
        arrays = {f'quality_welch_{i}_density': np.ones(2) for i in range(16)}
        panels = [dict(channel=ch, branch=branch, segment_id=0) for ch in range(1, 5)
                  for branch in (('before', 'after') if ch <= 2 else ('before',))]
        meta = dict(quality_status='complete', model_status='complete', panels=panels,
                    segments=[dict(segment_id=0, status='retained')], jen_display_count=6)
        check_coverage(case, arrays, meta)
        for bad_panels in (panels[:-1], panels+[panels[0]],
                           panels+[dict(channel=3, branch='after', segment_id=0)]):
            bad = dict(meta, panels=bad_panels)
            with self.assertRaises(AssertionError):
                check_coverage(case, arrays, bad)
        for key, value in [('jen_display_count', 5), ('quality_status', 'insufficient_samples')]:
            with self.assertRaises(AssertionError):
                check_coverage(case, arrays, dict(meta, **{key: value}))
        del arrays['quality_welch_15_density']
        with self.assertRaises(AssertionError):
            check_coverage(case, arrays, meta)

    def test_short_case_rejects_silent_model_success(self):
        case = dict(name='short', model_expected='no_complete_window', quality_expected='insufficient_samples')
        meta = dict(quality_status='insufficient_samples', model_status='no_complete_window', panels=[])
        check_coverage(case, {}, meta)
        with self.assertRaises(AssertionError):
            check_coverage(case, {}, dict(meta, model_status='complete'))

    def test_odd_and_even_one_sided_density_preserves_known_energy(self):
        for n in (200, 201):
            values = 2*np.sin(2*np.pi*10*np.arange(n)/n)
            f, p = reference_welch(values, n, n)
            self.assertAlmostEqual(float(p.sum()*(f[1]-f[0])), 2., places=12)
            self.assertEqual(f[np.argmax(p)], 10.)
            assert_reference(f, p, values, n, n)
            with self.assertRaises(AssertionError):
                assert_reference(f, p*2, values, n, n)
        _, p = reference_welch(np.full(200, 3.), 200, 200)
        np.testing.assert_array_equal(p, 0.)

    def test_single_bin_policy_and_nominal_window_capping_are_observable(self):
        values = 2*np.sin(2*np.pi*25*np.arange(8)/200)
        _, _, one = measurements(values, 200, 1)
        _, _, four = measurements(values, 200, 4)
        self.assertEqual(one, four)
        self.assertEqual(one['half_band'][2], 0.)
        self.assertGreater(one['entropy_band'][2], 0.)
        self.assertEqual(one['nperseg'], 8)

    def test_boundary_and_denominator_changes_remain_distinct(self):
        values = 2*np.sin(2*np.pi*8*np.arange(400)/200)
        _, _, row = measurements(values, 200, 1)
        self.assertGreater(row['boundary_max_abs'], .8)
        self.assertGreater(row['denominator_max_abs'], .1)
        self.assertEqual(row['singleton_max_abs'], 0.)

    def test_normalization_keeps_hashes_indices_and_diagnostics(self):
        metadata = dict(source='/source/record.csv', model_sha256='model', selected=[1, 2],
                        quality=dict(stage='filtered', valid=False), output='/run/case/result.csv')
        original = copy.deepcopy(metadata)
        normalized = normalize(metadata, Path('/run/case'))
        self.assertEqual(metadata, original)
        self.assertEqual(normalized.pop('output'), '<output>/result.csv')
        original.pop('output')
        self.assertEqual(normalized, original)
