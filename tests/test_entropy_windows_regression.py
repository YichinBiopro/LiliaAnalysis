import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

import spectral_entropy as entropy
import plot_index_vs_raw as plot
from lilia.entropy_io import load_entropy_table, write_entropy_table
from lilia.provenance import file_sha256
from lilia.signal import bandpass
from lilia.windowing import build_window_grid, plot_breaks, transform_runs


class EntropyWindowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def recording(self, time_us, data=None, name='recording.csv'):
        if data is None:
            data = np.random.default_rng(3).normal(size=(len(time_us), 2))
        frame = pd.DataFrame(data, columns=['ch1', 'ch2'])
        frame.insert(0, 'Time[us]', time_us)
        path = self.root / name
        with path.open('w') as handle:
            handle.write('File Name,test\nAmp Gain,500,Abs Time Offset[us],0\nChannels,1,2\nSample Rate,500,500\n')
            frame.to_csv(handle, index=False)
        return path

    def run_cli(self, source, *extra):
        argv = ['spectral_entropy.py', '--csv', str(source), '--out', str(self.root / 'output'),
                '--win', '2', '--step', '1', *extra]
        with patch('sys.argv', argv), contextlib.redirect_stdout(io.StringIO()), \
             patch.object(entropy, 'plot_band_entropy'), patch.object(entropy, 'plot_band_composition'), \
             patch.object(entropy, 'plot_band_ternary'), patch.object(entropy, 'plot_focus_relax_scatter'):
            entropy.main()
        tables = list((self.root / 'output').glob('*band_entropy*.csv'))
        return tables[0]

    def versioned_table(self, fs=500):
        t = np.r_[np.arange(fs * 4) * int(1e6 / fs),
                  20000000 + np.arange(fs * 4) * int(1e6 / fs)]
        source = self.recording(t)
        grid = build_window_grid(t, fs, 2, 1)
        frame = pd.DataFrame({'time_s': grid.time_s, 'p_theta': .2, 'p_alpha': .3, 'p_beta': .5,
                              'quality': .9, 'band_entropy_norm': np.linspace(.9, .1, len(grid.starts))})
        parameters = {'fs': fs, 'win_sec': 2, 'step_sec': 1, 'channel': 1, 'quality_enabled': True}
        table = self.root / 'entropy.csv'
        write_entropy_table(table, source, frame, grid, parameters, 'test-revision')
        return table, source, grid

    def test_continuous_numerics_match_pre_refactor_fixture(self):
        reference = json.loads((Path(__file__).parent / 'fixtures/entropy_continuous_reference.json').read_text())
        x = np.random.default_rng(reference['seed']).normal(size=reference['shape'])
        grid = build_window_grid(np.arange(len(x)) * 2000, 500, 2, 1)
        actual = entropy.compute_band_entropy_windowed(x[:, 0], fs=500, win_sec=2, step_sec=1, windows=grid)
        sync = entropy.compute_lagged_interhemispheric_sync_windowed(x[:, 0], x[:, 1], fs=500,
                    win_sec=2, step_sec=1, tau_ms_list=[10, 20], bins=8, apply_bandpass=False, windows=grid)
        for key, expected in reference['entropy'].items():
            np.testing.assert_allclose(actual[key], expected, atol=1e-12, rtol=1e-12)
        for key, expected in reference['sync'].items():
            np.testing.assert_allclose(sync[key], expected, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(entropy.compute_quality_windowed_aligned(x, fs=500,
                                   win_sec=2, step_sec=1, windows=grid), reference['quality'], atol=1e-12, rtol=1e-12)

    def test_non_grid_gap_and_jitter_keep_true_sample_centres(self):
        epoch = 1778566248498593
        t = epoch + np.r_[np.arange(1251) * 2000, 20000000 + np.arange(2500) * 2000]
        t[1:] += np.random.default_rng(4).integers(-30, 31, len(t) - 1)
        grid = build_window_grid(t, 500, 1, .5)
        expected = [i for i in range(0, len(t) - 500 + 1, 250) if np.diff(t[i:i + 500]).max() <= 6000]
        np.testing.assert_array_equal(grid.starts, expected)
        np.testing.assert_array_equal(grid.columns['window_center_us'], t[grid.starts + 250])
        np.testing.assert_array_equal(grid.time_s, (t[grid.starts + 250] - t[0]) / 1e6)
        x = np.column_stack([np.arange(len(t)) / len(t)] * 2)
        with patch.object(entropy, '_eeg_quality_v2', side_effect=lambda seg, **kw: {'overall': seg[:, 0]}):
            q = entropy.compute_quality_windowed_aligned(x, fs=500, win_sec=1, step_sec=.5, windows=grid)
        np.testing.assert_array_equal(q, x[grid.starts, 0])
        result = entropy.compute_band_entropy_windowed(x[:, 0], fs=500, win_sec=1, step_sec=.5, windows=grid)
        np.testing.assert_array_equal(result['time'], grid.time_s)
        for row, start in enumerate(grid.starts):
            expected_entropy = entropy.compute_band_entropy(x[start:start + 500, 0], fs=500)
            self.assertAlmostEqual(result['band_entropy'][row], expected_entropy['band_entropy'])

    def test_grid_mismatch_and_no_complete_segment_are_rejected(self):
        grid = build_window_grid(np.arange(2000) * 2000, 500, 2)
        with self.assertRaisesRegex(ValueError, 'grid does not match'):
            entropy.compute_band_entropy_windowed(np.ones(2000), fs=500, win_sec=1, windows=grid)
        with self.assertRaisesRegex(ValueError, 'No complete analysis window'):
            build_window_grid(np.r_[np.arange(600) * 2000, 20000000 + np.arange(600) * 2000], 500, 2)
        with self.assertRaisesRegex(ValueError, 'same sample count'):
            entropy.compute_lagged_interhemispheric_sync_windowed(np.ones(2000), np.ones(2001), windows=grid)

    def test_short_segment_skipped_before_filter_without_poisoning_other_segments(self):
        t = np.r_[np.arange(10) * 2000, 20000000 + np.arange(2500) * 2000]
        x = np.random.default_rng(9).normal(size=(len(t), 2))
        path = self.recording(t, x)
        table = self.run_cli(path, '--no-quality-mask')
        frame, meta = load_entropy_table(table, path, 1)
        self.assertTrue((frame.segment_id == 1).all())
        # The recording loader deliberately converts EEG to float32 first.
        filtered = bandpass(x[10:].astype(np.float32), fs=500).astype(np.float32)
        for _, row in frame.iterrows():
            start = int(row.window_start_idx) - 10
            expected = entropy.compute_band_entropy(filtered[start:start + 1000, 0])
            self.assertAlmostEqual(row.band_entropy, expected['band_entropy'], places=10)
        self.assertFalse(meta['parameters']['quality_enabled'])
        self.assertTrue((frame.quality_state == 'disabled').all())

    def test_invalid_quality_masks_metrics_but_retains_metadata_rows(self):
        t = np.r_[np.arange(2000) * 2000, 20000000 + np.arange(2000) * 2000]
        path = self.recording(t)
        scores = iter([.9, np.nan, .2, .9, .9, .9])
        with patch.object(entropy, '_eeg_quality_v2', side_effect=lambda *a, **kw: {'overall': [next(scores)]}):
            table = self.run_cli(path, '--sync-pair', '1', '2', '--tau-ms', '10', '--mi-bins', '8')
        frame, _ = load_entropy_table(table, path, 1)
        self.assertEqual(len(frame), 6)
        self.assertTrue(frame.loc[[1, 2], ['band_entropy', 'p_alpha', 'lagged_mi_mean']].isna().all().all())
        self.assertFalse(frame.loc[[1, 2], 'quality_valid'].any())
        self.assertTrue(np.isfinite(frame.window_start_idx).all())
        np.testing.assert_allclose(frame.time_s, [1, 2, 3, 21, 22, 23])

    def test_metadata_rejects_wrong_source_channel_and_partial_schema(self):
        table, source, _ = self.versioned_table()
        with self.assertRaisesRegex(ValueError, 'channel differs'):
            load_entropy_table(table, source, 2)
        original = source.read_text()
        source.write_text(original.replace('File Name,test', 'File Name,changed'))
        with self.assertRaisesRegex(ValueError, 'Raw recording differs'):
            load_entropy_table(table, source, 1)
        source.write_text(original)
        Path(str(table) + '.meta.json').unlink()
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            load_entropy_table(table, source, 1)

    def test_metadata_checks_indexes_against_timestamps_even_if_table_rehashed(self):
        table, source, _ = self.versioned_table()
        frame = pd.read_csv(table)
        frame.loc[0, 'window_start_idx'] += 1
        frame.to_csv(table, index=False)
        sidecar = Path(str(table) + '.meta.json')
        meta = json.loads(sidecar.read_text())
        meta['table_sha256'] = file_sha256(table)
        sidecar.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, 'window_start_idx does not match'):
            load_entropy_table(table, source, 1)

    def test_gap_break_retains_first_post_gap_point_and_smoothing_is_separate(self):
        groups = np.array([0, 0, 1, 1])
        x, y = plot_breaks([1, 2, 21, 22], [0, 0, 10, 10], groups)
        np.testing.assert_array_equal(x, [1, 2, 21, 21, 22])
        np.testing.assert_allclose(y, [0, 0, np.nan, 10, 10], equal_nan=True)
        np.testing.assert_array_equal(plot._rolling_median(np.array([0., 0., 10., 10.]), 5, groups), [0, 0, 10, 10])
        np.testing.assert_allclose(transform_runs([0, 0, np.nan, 10, 10], entropy._smooth_series),
                                   [0, 0, np.nan, 10, 10], equal_nan=True)

    def test_session_renderer_uses_metadata_instead_of_cli_defaults_or_row_positions(self):
        table, source, grid = self.versioned_table(fs=200)
        captured = []
        def capture(fig, *args, **kwargs):
            captured.append((fig.axes[0].lines[0].get_xdata(), fig.axes[0].lines[0].get_ydata()))
        with patch.object(plot, '_real_window_times', side_effect=AssertionError('must use metadata')), \
             patch.object(Figure, 'savefig', capture), contextlib.redirect_stdout(io.StringIO()):
            plot.plot_index_vs_raw_session(str(table), str(source), 'Demo', 1, str(self.root / 'plot.png'),
                                            use_minutes=False, fs=500, win_sec=5)
        actual_t, y = captured[0]
        self.assertEqual(np.isfinite(y).sum(), len(grid.starts))
        self.assertEqual(np.isnan(y).sum(), 1)
        self.assertIn(21, actual_t)

    def test_disabled_quality_requires_explicit_plot_override(self):
        q = np.array([np.nan])
        signals = {'_metadata': {'parameters': {'quality_enabled': False}}}
        with self.assertRaisesRegex(ValueError, 'quality scoring was disabled'):
            plot._quality_good(q, signals, .5)
        self.assertTrue(plot._quality_good(q, signals, -1).all())

    def test_legacy_gap_guards_stay_for_unmigrated_modes_and_tables(self):
        t = np.r_[np.arange(1000) * 2000, 20000000 + np.arange(1000) * 2000]
        path = self.recording(t)
        with self.assertRaisesRegex(ValueError, 'requires a continuous recording'):
            plot._real_window_times(2, str(path), 2, 2, 500)

    def test_renderer_smoothing_and_lines_do_not_join_segments(self):
        table, source, grid = self.versioned_table()
        result = {'time': grid.time_s, 'segment_id': grid.columns['segment_id'],
                  'p_theta': np.array([.1] * 3 + [.9] * 3), 'p_alpha': np.ones(6) * .05,
                  'p_beta': np.array([.85] * 3 + [.05] * 3),
                  'band_entropy': np.ones(6), 'band_entropy_norm': np.ones(6)}
        captured = []
        def capture(fig, *args, **kwargs):
            captured.append(fig.axes[0].lines[1].get_ydata())
        with patch.object(Figure, 'savefig', capture), contextlib.redirect_stdout(io.StringIO()):
            entropy.plot_band_entropy(result, 'Demo', str(self.root / 'plot.png'))
        np.testing.assert_allclose(captured[0], [.1, .1, .1, np.nan, .9, .9, .9], equal_nan=True)
