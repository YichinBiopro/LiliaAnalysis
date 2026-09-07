import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

import spectral_entropy as spectral
from lilia.entropy_io import load_entropy_table, load_joint_mi_table
from lilia.provenance import file_sha256
from lilia.windowing import build_window_grid


class JointMIWindowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.t = np.r_[np.arange(800) * 5000, 20000000 + np.arange(800) * 5000]
        self.x = np.random.default_rng(427).normal(size=(len(self.t), 2))

    def recording(self, t=None, x=None, name='recording.csv'):
        t = self.t if t is None else t
        x = self.x if x is None else x
        frame = pd.DataFrame(x, columns=['ch1', 'ch2'])
        frame.insert(0, 'Time[us]', t)
        path = self.root / name
        path.write_text('File Name,test\nAmp Gain,500,Abs Time Offset[us],0\n'
                        'Channels,1,2\nSample Rate,200,200\n' + frame.to_csv(index=False))
        return path

    def run_cli(self, path, *extra):
        args = ['spectral_entropy.py', '--csv', str(path), '--out', str(self.root / 'out'),
                '--fs', '200', '--joint-mi', '--win', '2', '--step', '1',
                '--mi-bins', '8', '--mi-surrogates', '3', *extra]
        with patch('sys.argv', args), contextlib.redirect_stdout(io.StringIO()), \
             patch.object(spectral, 'plot_joint_distribution'), patch.object(spectral, 'plot_joint_excess'), \
             patch.object(Figure, 'savefig'):
            spectral.main()
        return self.root / 'out/recording_joint_mi_ch1_ch2_timeseries.csv'

    def test_continuous_mi_quality_and_surrogates_match_reference(self):
        reference = json.loads((Path(__file__).parent / 'fixtures/joint_mi_continuous_reference.json').read_text())
        x = spectral.bandpass_filter(np.random.default_rng(reference['seed']).normal(size=reference['shape']),
                                     fs=200, lo=.5, hi=45)
        grid = build_window_grid(np.arange(len(x)) * 5000, 200, 2, 1)
        actual = spectral.compute_joint_mi_windowed(x[:, 0], x[:, 1], fs=200, win_sec=2,
                    step_sec=1, bins=8, binning='quantile', windows=grid)
        significance = spectral.compute_joint_mi_significance(x[:, 0], x[:, 1], bins=8,
                    binning='quantile', n_surrogates=12, segment_ids=np.zeros(len(x)))
        joint = spectral.compute_joint_probability(x[:, 0], x[:, 1], bins=8, binning='quantile')
        for got, expected in [(actual, reference['window']), (significance, reference['significance']),
                              (joint, reference['joint'])]:
            for key, value in expected.items():
                np.testing.assert_allclose(got[key], value, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(spectral.compute_quality_windowed_aligned(x.astype(float), fs=200,
            win_sec=2, step_sec=1, windows=grid), reference['quality'], rtol=1e-12, atol=1e-12)

    def test_joint_windows_use_actual_timestamps_and_original_sample_grid(self):
        t = np.r_[np.arange(851) * 5000, 20000000 + np.arange(800) * 5000]
        t[1:] += np.random.default_rng(4).integers(-30, 31, len(t) - 1)
        x = np.random.default_rng(7).normal(size=(len(t), 2))
        grid = build_window_grid(t, 200, 2, 1)
        result = spectral.compute_joint_mi_windowed(x[:, 0], x[:, 1], fs=200, win_sec=2,
                                                   step_sec=1, windows=grid)
        np.testing.assert_array_equal(grid.starts, [0, 200, 400, 1000, 1200])
        np.testing.assert_array_equal(result['time'], t[grid.starts + 200] / 1e6)
        for i, start in enumerate(grid.starts):
            expected = spectral.compute_joint_probability(x[start:start + 400, 0], x[start:start + 400, 1])
            self.assertEqual(result['joint_mi'][i], expected['mutual_information'])
        with self.assertRaisesRegex(ValueError, 'equal sample counts'):
            spectral.compute_joint_mi_windowed(x[:-1, 0], x[:, 1], fs=200, windows=grid)
        with self.assertRaisesRegex(ValueError, 'does not match'):
            spectral.compute_joint_mi_windowed(x[:, 0], x[:, 1], fs=200, win_sec=1, windows=grid)

    def test_surrogate_rolls_each_segment_independently(self):
        x = np.random.default_rng(9).normal(size=(140, 2))
        groups = np.r_[np.zeros(60), np.ones(80)]
        result = spectral.compute_joint_mi_significance(x[:, 0], x[:, 1], bins=8,
                         binning='quantile', n_surrogates=7, seed=5, segment_ids=groups)
        rng = np.random.default_rng(5)
        left = rng.integers(1, 60, size=7)
        right = rng.integers(1, 80, size=7)
        expected = [spectral._histogram_mutual_information(x[:, 0],
                    np.r_[np.roll(x[:60, 1], a), np.roll(x[60:, 1], b)], bins=8, binning='quantile')
                    for a, b in zip(left, right)]
        np.testing.assert_array_equal(result['surrogates'], expected)
        self.assertEqual(result['null_method'], 'within_segment_circular_shift')
        self.assertEqual(result['null_state'], 'computed')
        short = spectral.compute_joint_mi_significance(x[:, 0], x[:, 1],
                            segment_ids=np.r_[np.zeros(10), np.ones(130)])
        self.assertEqual(short['null_state'], 'segment_too_short')
        self.assertEqual(short['n_surrogates'], 0)
        self.assertTrue(np.isnan(short['p_value']))
        with self.assertRaisesRegex(ValueError, 'finite paired'):
            spectral.compute_joint_mi_significance(np.r_[np.nan, x[1:, 0]], x[:, 1], segment_ids=groups)

    def test_cli_quality_nan_mask_and_metadata_keep_all_window_rows(self):
        source = self.recording()
        scored = []
        scores = iter([1, np.nan, .1, 1, 1, 1])
        def quality(data, **kwargs):
            scored.append(data.copy())
            return {'overall': np.array([next(scores)] * 2)}
        with patch.object(spectral, '_eeg_quality_v2', side_effect=quality):
            table = self.run_cli(source, '--no-bandpass')
        frame, metadata = load_joint_mi_table(table, source, [1, 2])
        self.assertEqual(metadata['kind'], 'joint_mi')
        self.assertEqual(metadata['parameters']['channels'], [1, 2])
        self.assertEqual(frame.quality_valid.tolist(), [True, False, False, True, True, True])
        self.assertEqual(frame.time_s.tolist(), [1, 2, 3, 21, 22, 23])
        self.assertTrue(frame.joint_mi.iloc[1:3].isna().all())
        for seen, start in zip(scored, frame.window_start_idx):
            np.testing.assert_array_equal(seen, self.x[start:start + 400].astype(np.float32).T)
        summary = pd.read_csv(self.root / 'out/recording_joint_mi_ch1_ch2_summary.csv')
        self.assertEqual(summary.population_quality_state.iloc[0], 'disabled')
        self.assertEqual(summary.population_runs.iloc[0], 2)

    def test_missing_quality_or_count_mismatch_fails(self):
        source = self.recording()
        with patch.object(spectral, '_QC_AVAILABLE', False), self.assertRaisesRegex(RuntimeError, 'explicitly'):
            self.run_cli(source)
        with patch.object(spectral, 'compute_quality_windowed_aligned', return_value=np.ones(1)), \
             self.assertRaisesRegex(ValueError, 'count does not match'):
            self.run_cli(source)
        with patch.object(spectral, '_QC_AVAILABLE', False):
            table = self.run_cli(source, '--no-quality-mask')
        frame, metadata = load_joint_mi_table(table, source)
        self.assertFalse(metadata['parameters']['quality_enabled'])
        self.assertEqual(frame.quality_state.unique().tolist(), ['disabled'])

    def test_metadata_rejects_wrong_source_pair_kind_or_rehashed_indexes(self):
        source = self.recording()
        table = self.run_cli(source, '--no-quality-mask')
        with self.assertRaisesRegex(ValueError, 'channels differ'):
            load_joint_mi_table(table, source, [2, 1])
        wrong = self.recording(x=self.x + 1, name='wrong.csv')
        with self.assertRaisesRegex(ValueError, 'Raw recording differs'):
            load_joint_mi_table(table, wrong)
        with self.assertRaisesRegex(ValueError, 'fingerprint mismatch'):
            load_entropy_table(table, source)
        frame = pd.read_csv(table)
        frame.loc[0, 'window_start_idx'] += 1
        frame.to_csv(table, index=False)
        sidecar = Path(str(table) + '.meta.json')
        meta = json.loads(sidecar.read_text())
        meta['table_sha256'] = file_sha256(table)
        sidecar.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, 'does not match raw timestamps'):
            load_joint_mi_table(table, source)
        sidecar.unlink()
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            load_joint_mi_table(table, source)

    def test_short_segment_excluded_from_summary_and_filtering_is_independent(self):
        t = np.r_[np.arange(10) * 5000, 20000000 + np.arange(800) * 5000]
        x = np.random.default_rng(5).normal(size=(len(t), 2))
        source = self.recording(t=t, x=x)
        table = self.run_cli(source, '--no-quality-mask')
        frame, _ = load_joint_mi_table(table, source)
        expected = spectral.bandpass_filter(x[10:], fs=200, lo=.5, hi=45)
        for row in frame.itertuples():
            a, b = row.window_start_idx - 10, row.window_end_idx - 10
            joint = spectral.compute_joint_probability(expected[a:b, 0], expected[a:b, 1], bins=8, binning='quantile')
            self.assertAlmostEqual(row.joint_mi, joint['mutual_information'], places=12)
        summary = pd.read_csv(self.root / 'out/recording_joint_mi_ch1_ch2_summary.csv')
        self.assertEqual(summary.n_samples.iloc[0], 800)
        self.assertEqual(summary.excluded_samples.iloc[0], 10)
        self.assertEqual(summary.population_runs.iloc[0], 1)

    def test_peri_event_interpolation_does_not_fill_recording_or_quality_gaps(self):
        result = {'time': np.array([0., 1, 2, 3, 20, 21]),
                  'joint_mi': np.array([0., 0, np.nan, 7, 10, 10]),
                  'segment_id': np.array([0, 0, 0, 0, 1, 1])}
        grid = np.arange(22, dtype=float)
        series, mean, _ = spectral._peri_event_series_on_grid(result, [('event', 0)], grid, 0, 21)
        expected = np.r_[0, 0, np.nan, 7, np.full(16, np.nan), 10, 10]
        np.testing.assert_allclose(series[0][1], expected, equal_nan=True)
        np.testing.assert_allclose(mean, expected, equal_nan=True)
        plot_grid = spectral._peri_event_plot_grid(result, [('event', 0)], 0, 21, 20)
        self.assertIn(2., plot_grid)
        self.assertIn(11.5, plot_grid)
        captured = []
        def capture(fig, *args, **kwargs):
            captured.append(fig.axes[0].lines[0].get_ydata())
        with patch.object(Figure, 'savefig', capture), contextlib.redirect_stdout(io.StringIO()):
            spectral.plot_peri_event_mi(result, [('event', 0)], 'test', str(self.root / 'peri.png'), pre_sec=0, post_sec=21)
        np.testing.assert_allclose(captured[0], [0, 0, np.nan, 7, np.nan, 10, 10], equal_nan=True)

    def test_series_plot_preserves_first_point_after_gap_and_smooths_per_run(self):
        source = self.recording()
        result = {'time': np.array([1., 2, 3, 21, 22, 23]),
                  'joint_mi': np.array([0., 0, np.nan, 10, 10, 10]),
                  'joint_mi_norm': np.array([0., 0, np.nan, 1, 1, 1]),
                  'segment_id': np.array([0, 0, 0, 1, 1, 1])}
        figures = []
        original = spectral.plt.subplots
        def capture(*args, **kwargs):
            fig, axes = original(*args, **kwargs)
            figures.append(fig)
            return fig, axes
        with patch.object(spectral, 'compute_joint_mi_windowed', return_value=result), \
             patch.object(spectral.plt, 'subplots', side_effect=capture):
            self.run_cli(source, '--no-quality-mask')
        for line in figures[0].axes[0].lines[:2]:
            np.testing.assert_allclose(line.get_ydata(), [0, 0, np.nan, np.nan, 10, 10, 10], equal_nan=True)

    def test_nonfinite_samples_invalidate_windows_without_dropping_rows(self):
        x = self.x.copy()
        x[500, 0] = np.nan
        grid = build_window_grid(self.t, 200, 2, 1)
        result = spectral.compute_joint_mi_windowed(x[:, 0], x[:, 1], fs=200,
                                                    win_sec=2, step_sec=1, windows=grid)
        np.testing.assert_array_equal(np.isfinite(result['joint_mi']), [True, False, False, True, True, True])
        np.testing.assert_array_equal(result['time'], grid.time_s)

    def test_baseline_and_joint_mi_modes_cannot_be_combined(self):
        source = self.recording()
        with self.assertRaisesRegex(ValueError, 'cannot be combined'):
            self.run_cli(source, '--baseline', '0', '2', '--event', '20', '22')


if __name__ == '__main__':
    unittest.main()
