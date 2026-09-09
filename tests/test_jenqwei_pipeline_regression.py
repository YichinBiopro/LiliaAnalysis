"""Stage-16 model baselines and independent raw/Before/After index contracts."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from scipy.signal import stft, welch

import analyze_jenqwei_pipeline as jenqwei
from lilia.provenance import file_sha256

FIXTURES = Path(__file__).parent / 'fixtures'


def identity(data):
    return data[:len(data) // 400 * 400, :2].copy()


class JenqweiPipelineTests(unittest.TestCase):
    def test_five_real_and_two_synthetic_cases_match_frozen_model_and_spectra(self):
        paths = sorted(FIXTURES.glob('jenqwei_*_reference.npz'))
        self.assertEqual(len(paths), 7)
        for path in paths:
            with self.subTest(case=path.stem), np.load(path) as old:
                metadata = json.loads(path.with_suffix('.json').read_text())
                self.assertEqual(file_sha256(path), metadata['npz_sha256'])
                self.assertEqual(file_sha256(Path(jenqwei.__file__).parent / 'tiny_v4_optimized.tflite'), metadata['model_sha256'])
                result = jenqwei.process_segments(old['source_time_us'], old['source_raw'], metadata['max_samples'])
                for key in ('time_us_500', 'data_raw', 'data_filt_500', 'time_us_200', 'pre_data_200', 'tfl_data_200'):
                    np.testing.assert_array_equal(result[key], old[key])
                self.assertEqual(len(result['tfl_data_200']) % 400, 0)
                np.testing.assert_array_equal(result['tfl_time_us_200'], old['time_us_200'][:len(old['tfl_data_200'])])
                np.testing.assert_array_equal(result['model_to_before_idx'], np.arange(len(old['tfl_data_200'])))
                for ch in range(4):
                    for branch, values in (('before', result['pre_data_200'][:, ch]),
                                           ('after', result['tfl_data_200'][:, min(ch, 1)])):
                        prefix = f'ch{ch}_{branch}'
                        f, power = welch(values.astype(np.float64), fs=200,
                                         nperseg=min(800, len(result['pre_data_200']), len(values)), scaling='density')
                        np.testing.assert_array_equal(f, old[prefix + '_psd_f'])
                        np.testing.assert_array_equal(power, old[prefix + '_psd'])
                        f, t, z = stft(values[:12000].astype(np.float64), fs=200,
                                       nperseg=256, noverlap=128, window='hann')
                        mask = f <= 50
                        np.testing.assert_array_equal(f[mask], old[prefix + '_stft_f'])
                        np.testing.assert_array_equal(t, old[prefix + '_stft_t'])
                        np.testing.assert_array_equal(20 * np.log10(np.abs(z[mask]) + 1e-8), old[prefix + '_stft_db'])

    def test_gaps_preserve_distinct_before_and_after_tails_and_source_positions(self):
        lengths = (10, 1503, 2001, 997)
        t = 9000000000000001 + np.concatenate([i * 10000000 + np.arange(n) * 2000 for i, n in enumerate(lengths)])
        raw = np.random.default_rng(16).normal(size=(len(t), 4)).astype(np.float32)
        with patch.object(jenqwei, 'apply_tflite_windowed', side_effect=identity):
            result = jenqwei.process_segments(t, raw)
            independent = [jenqwei.run_pipeline(t[a:b], raw[a:b]) for a, b in ((10, 1513), (1513, 3514))]
        self.assertEqual(result['pre_data_200'].shape, (1403, 4))
        self.assertEqual(result['tfl_data_200'].shape, (1200, 2))
        self.assertEqual([s['trimmed_samples'] for s in result['segments']], [4, 202, 1, 399])
        np.testing.assert_array_equal(result['pre_data_200'], np.concatenate([p['pre_data_200'] for p in independent]))
        np.testing.assert_array_equal(result['tfl_data_200'], np.concatenate([p['tfl_data_200'] for p in independent]))
        np.testing.assert_array_equal(result['model_to_before_idx'], np.r_[0:400, 602:1402])
        np.testing.assert_array_equal(result['raw_sample_idx'], np.arange(10, 3514))
        self.assertEqual(result['before_raw_fractional_idx'][602], 1513.)
        self.assertEqual(result['model_raw_fractional_idx'][400], 1513.)
        np.testing.assert_array_equal(result['tfl_time_us_200'], result['time_us_200'][result['model_to_before_idx']])
        self.assertEqual(result['timeline'].source_epoch_us, t[0])
        self.assertEqual(result['quality_state'], 'disabled')
        for window in result['timeline'].model_windows:
            row = result['segments'][window['segment_id']]
            self.assertGreaterEqual(window['raw_start_idx'], row['raw_start_idx'])
            self.assertLessEqual(window['raw_end_idx'], row['raw_end_idx'])

    def test_segment_filter_and_model_are_isolated_from_other_segments(self):
        t = np.r_[np.arange(1503) * 2000, 10000000 + np.arange(1503) * 2000]
        raw = np.random.default_rng(161).normal(size=(len(t), 4)).astype(np.float32)
        with patch.object(jenqwei, 'apply_tflite_windowed', side_effect=identity):
            original = jenqwei.process_segments(t, raw)
            raw[:1503] *= 10000
            changed = jenqwei.process_segments(t, raw)
        np.testing.assert_array_equal(original['pre_data_200'][602:], changed['pre_data_200'][602:])
        np.testing.assert_array_equal(original['tfl_data_200'][400:], changed['tfl_data_200'][400:])

    def test_small_gap_is_not_lost_at_output_rate_and_tail_positions_extrapolate(self):
        t = np.r_[np.arange(998) * 2000, 2002000 + np.arange(998) * 2000]
        with patch.object(jenqwei, 'apply_tflite_windowed', side_effect=identity):
            result = jenqwei.process_segments(t, np.ones((len(t), 4)))
        self.assertEqual(result['pre_data_200'].shape, (800, 4))
        np.testing.assert_array_equal(result['before_segment_ids'][[399, 400]], [0, 1])
        np.testing.assert_array_equal(result['timeline'].segment_ids[[399, 400]], [0, 1])
        np.testing.assert_array_equal(result['tfl_time_us_200'][[399, 400]], [1995000, 2002000])
        self.assertEqual(result['before_raw_fractional_idx'][399], 997.5)
        self.assertEqual(result['before_raw_fractional_idx'][400], 998.)

    def test_max_samples_filters_full_segment_before_truncating(self):
        t = np.r_[np.arange(1503) * 2000, 10000000 + np.arange(2503) * 2000]
        raw = np.random.default_rng(162).normal(size=(len(t), 4)).astype(np.float32)
        with patch.object(jenqwei, 'apply_tflite_windowed', side_effect=identity):
            result = jenqwei.process_segments(t, raw, max_samples=3006)
            first = jenqwei.run_pipeline(t[:1503], raw[:1503])
            second = jenqwei.run_pipeline(t[1503:], raw[1503:], max_samples=1503)
            truncated_first = jenqwei.run_pipeline(t[1503:3006], raw[1503:3006])
        self.assertEqual(result['full_source_samples'], 4006)
        self.assertEqual(result['used_source_samples'], 3006)
        self.assertEqual(result['segments'][1]['filter_context_end_idx'], 4006)
        self.assertEqual(result['segments'][1]['raw_end_idx'], 3006)
        np.testing.assert_array_equal(result['pre_data_200'], np.concatenate([first['pre_data_200'], second['pre_data_200']]))
        self.assertGreater(float(np.max(np.abs(second['pre_data_200'] - truncated_first['pre_data_200']))), .01)

    def test_all_short_and_polluted_short_segments_fail_before_inference(self):
        with patch.object(jenqwei, 'apply_tflite_windowed') as model:
            with self.assertRaisesRegex(ValueError, 'No source segment'):
                jenqwei.process_segments(np.arange(997) * 2000, np.ones((997, 4)))
            t = np.r_[np.arange(10) * 2000, 10000000 + np.arange(998) * 2000]
            for bad in (np.nan, np.inf, -np.inf):
                raw = np.ones((len(t), 4))
                raw[0, 2] = bad
                with self.assertRaisesRegex(ValueError, 'non-finite'):
                    jenqwei.process_segments(t, raw)
            model.assert_not_called()

    def test_failed_model_in_second_segment_has_source_context(self):
        t = np.r_[np.arange(998) * 2000, 2002000 + np.arange(998) * 2000]
        for failure in (RuntimeError('missing model'), np.zeros((399, 2)), np.full((400, 2), np.nan)):
            with patch.object(jenqwei, 'apply_tflite_windowed', side_effect=[np.zeros((400, 2)), failure]):
                with self.assertRaisesRegex(ValueError, r'segment 1 raw \[998:1996\] failed'):
                    jenqwei.process_segments(t, np.ones((len(t), 4)))

    def test_invalid_shape_timestamp_limit_and_legacy_gap_guard(self):
        t = np.arange(1000) * 2000
        for shape in ((1000, 3), (1000, 5), (999, 4)):
            with self.assertRaisesRegex(ValueError, 'exactly four'):
                jenqwei.process_segments(t, np.ones(shape))
        for limit in (0, -1, 2.5, True):
            with self.assertRaisesRegex(ValueError, 'max_samples'):
                jenqwei.process_segments(t, np.ones((1000, 4)), limit)
        for times in (t.astype(float), np.zeros(1000, dtype=np.int64)):
            with self.assertRaises(ValueError):
                jenqwei.process_segments(times, np.ones((1000, 4)))
        t[500:] += 10000000
        with self.assertRaises(ValueError):
            jenqwei.run_pipeline(t, np.ones((1000, 4)))


if __name__ == '__main__':
    unittest.main()
