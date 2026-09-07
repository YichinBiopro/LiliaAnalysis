import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure

import data_analysis as da
import spectral_entropy as spectral
from lilia.event_windows import select_event_windows
from lilia.neural import build_inference_timeline, denoise_with_time, model_provenance
from lilia.neural_io import load_denoised_joint_mi_table
from lilia.entropy_io import load_joint_mi_table
from lilia.provenance import file_sha256
from lilia.signal import resample_polyphase, resample_segment_time_us
from lilia.windowing import continuous_slices


class Identity(torch.nn.Module):
    def forward(self, x):
        return x[:, :2]


class NeuralTimelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_real_model_matches_correct_resampling_old_ola_reference(self):
        ref = json.loads((Path(__file__).parent / 'fixtures/denoise_continuous_reference.json').read_text())
        try:
            model = da.load_model()
        except (ModuleNotFoundError, FileNotFoundError) as exc:
            self.skipTest(str(exc))
        self.assertEqual(model_provenance()['checkpoint_sha256'], ref['model_sha256'])
        self.assertEqual(model_provenance()['architecture_sha256'], ref['architecture_sha256'])
        x = np.random.default_rng(ref['seed']).normal(size=ref['shape'])
        timeline, actual = denoise_with_time(np.arange(len(x)) * 2000, x, model=model)
        expected = np.load(Path(__file__).parent / 'fixtures/denoise_resampled_reference.npz')['output']
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)
        self.assertEqual(actual.shape, (602, 2))
        np.testing.assert_array_equal(timeline.time_us, np.arange(602) * 5000)
        # Historical wrapper incorrectly kept 1503 rows while reporting 200 Hz.
        old = np.load(Path(__file__).parent / 'fixtures/denoise_continuous_reference.npz')['output']
        self.assertEqual(old.shape, (1503, 2))

    def test_overlap_add_identity_preserves_real_edges_and_rejects_short(self):
        for n in (400, 401, 602, 799):
            x = np.random.default_rng(n).normal(size=(n, 4)).astype(np.float32)
            pred = da.run_model(Identity(), x)
            np.testing.assert_allclose(pred, x[:, :2], atol=5e-7, rtol=5e-7)
        with self.assertRaisesRegex(ValueError, 'complete model window'):
            da.run_model(Identity(), np.ones((100, 4)))
        with self.assertRaisesRegex(ValueError, 'window'):
            da.run_model(Identity(), np.ones((400, 4)), hop=0)
        class Bad(torch.nn.Module):
            def forward(self, x):
                return x[:, :1] * np.nan
        with self.assertRaisesRegex(ValueError, 'output'):
            da.run_model(Bad(), np.ones((400, 4)))

    def test_small_gap_survives_resampling_and_grid_restarts_per_segment(self):
        t = np.r_[np.arange(1001) * 2000, 2008000 + np.arange(1001) * 2000]
        timeline = build_inference_timeline(t, 500)
        self.assertEqual(len(continuous_slices(timeline.time_us, 200)), 1)
        self.assertEqual(len(continuous_slices(timeline.time_us, 200, segment_ids=timeline.segment_ids)), 2)
        grid = timeline.grid(2)
        np.testing.assert_array_equal(grid.starts, [0, 401])
        np.testing.assert_array_equal(grid.columns['segment_id'], [0, 1])
        windows, rows = select_event_windows(timeline.time_us, 200, [('edge', 2008000)], 1,
                                             segment_ids=timeline.segment_ids)
        self.assertEqual(windows, [])
        self.assertEqual(rows[0]['Reason'], 'pre_crosses_gap')
        for record in timeline.model_windows:
            seg = timeline.segments[record['segment_id']]
            self.assertGreaterEqual(record['output_start_idx'], seg['output_start_idx'])
            self.assertLessEqual(record['output_end_idx'], seg['output_end_idx'])
            self.assertEqual(record['input_end_idx'] - record['input_start_idx'], 400)

    def test_short_segment_preserves_raw_epoch_and_original_id(self):
        t = np.r_[np.arange(100) * 2000, 20000000 + np.arange(1200) * 2000]
        x = np.random.default_rng(1).normal(size=(len(t), 4))
        timeline, pred = denoise_with_time(t, x, model=Identity())
        self.assertEqual(pred.shape, (480, 2))
        self.assertEqual(timeline.segments[0]['reason'], 'shorter_than_model_window')
        grid = timeline.grid(2)
        np.testing.assert_array_equal(grid.time_s, [21])
        np.testing.assert_array_equal(grid.columns['segment_id'], [1])
        self.assertEqual(timeline.segments[1]['raw_start_idx'], 100)
        with self.assertRaisesRegex(ValueError, 'No continuous segment'):
            denoise_with_time(t[:100], x[:100], model=Identity())

    def test_segment_filtering_inference_and_input_fs_are_independent(self):
        fs = 250
        t = np.r_[np.arange(600) * 4000, 10000000 + np.arange(650) * 4000]
        x = np.random.default_rng(6).normal(size=(len(t), 4))
        timeline, actual = denoise_with_time(t, x, fs, model=Identity())
        expected = []
        for sl in continuous_slices(t, fs):
            filtered = da.apply_filters(x[sl], fs=fs)
            expected.append(da.run_model(Identity(), resample_polyphase(filtered, fs, 200)))
        np.testing.assert_array_equal(actual, np.concatenate(expected))
        changed = x.copy()
        changed[:600] *= 1000
        _, other = denoise_with_time(t, changed, fs, model=Identity())
        cut = timeline.segments[1]['output_start_idx']
        np.testing.assert_array_equal(actual[cut:], other[cut:])

    def test_fractional_rate_jitter_mapping_and_end_cap(self):
        fs = 512.5
        t = 1778566248498593 + np.rint(np.arange(1024) * 1e6 / fs).astype(np.int64)
        t[1:] += np.random.default_rng(5).integers(-20, 21, len(t) - 1)
        timeline = build_inference_timeline(t, fs)
        np.testing.assert_array_equal(timeline.time_us, resample_segment_time_us(t, fs, 200))
        self.assertEqual(len(timeline.time_us), 400)
        positions = np.arange(400) * fs / 200
        expected = np.interp(positions, np.arange(len(t)), t - t[0])
        np.testing.assert_array_equal(timeline.time_us, t[0] + np.rint(expected).astype(np.int64))
        end = timeline.grid(2).columns['window_end_us'][0]
        self.assertEqual(end, t[-1] + round(1e6 / fs))
        _, rows = select_event_windows(timeline.time_us, 200, [('tail', int(t[0]) + 1000000)], 1,
                    segment_ids=timeline.segment_ids, segment_end_us={0: int(end)})
        self.assertEqual(rows[0]['Reason'], 'post_outside_recording')

    def test_invalid_time_and_failed_model_output_are_rejected(self):
        t = np.arange(1000) * 2000
        with self.assertRaisesRegex(ValueError, 'integer'):
            build_inference_timeline(t.astype(float), 500)
        bad = t.copy()
        bad[50] = bad[49]
        with self.assertRaisesRegex(ValueError, 'strictly increasing'):
            build_inference_timeline(bad, 500)
        x = np.ones((1000, 4))
        with patch.object(da, 'run_model', return_value=np.zeros((399, 2))), \
             self.assertRaisesRegex(ValueError, 'inference timestamps'):
            denoise_with_time(t, x, model=Identity())

    def cli(self, *extra):
        t = np.r_[np.arange(100) * 2000, 20000000 + np.arange(1200) * 2000,
                  30000000 + np.arange(1000) * 2000]
        x = np.random.default_rng(8).normal(size=(len(t), 4))
        source = self.root / 'source.csv'
        frame = pd.DataFrame(x, columns=['ch1', 'ch2', 'ch3', 'ch4'])
        frame.insert(0, 'Time[us]', t)
        source.write_text('File Name,test\nAmp Gain,500,Abs Time Offset[us],0\n'
                          'Channels,1,2,3,4\nSample Rate,500,500,500,500\n' + frame.to_csv(index=False))
        args = ['spectral_entropy.py', '--csv', str(source), '--joint-mi', '--denoise',
                '--mi-surrogates', '0', '--out', str(self.root / 'out'), *extra]
        with patch('sys.argv', args), patch.object(da, 'load_model', return_value=Identity()), \
             patch.object(spectral, 'plot_joint_distribution'), patch.object(spectral, 'plot_joint_excess'), \
             patch.object(Figure, 'savefig'), contextlib.redirect_stdout(io.StringIO()):
            spectral.main()
        return source, self.root / 'out/source_joint_mi_denoised_ch1_ch2_timeseries.csv'

    def test_cli_output_metadata_verifies_inference_axis_not_raw_grid(self):
        source, table = self.cli()
        frame, meta = load_denoised_joint_mi_table(table, source, da.MODEL_PATH)
        self.assertEqual(meta['kind'], 'denoised_joint_mi')
        self.assertEqual(meta['parameters']['index_space'], 'resampled_model_output')
        self.assertEqual(meta['source_samples'], 2300)
        self.assertEqual(frame.time_s.tolist(), [21, 31])
        self.assertEqual(frame.segment_id.tolist(), [1, 2])
        self.assertEqual(frame.quality_state.unique().tolist(), ['disabled'])
        self.assertEqual(frame.window_start_idx.tolist(), [0, 480])
        self.assertEqual(meta['inference']['segments'][0]['status'], 'excluded')
        with self.assertRaisesRegex(ValueError, 'fingerprint mismatch'):
            load_joint_mi_table(table, source)
        fake_model = self.root / 'wrong.pth'
        fake_model.write_text('not the checkpoint')
        with self.assertRaisesRegex(ValueError, 'checkpoint differs'):
            load_denoised_joint_mi_table(table, source, fake_model)

    def test_metadata_rejects_mapping_or_rehashed_window_changes(self):
        source, table = self.cli()
        sidecar = Path(str(table) + '.meta.json')
        meta = json.loads(sidecar.read_text())
        original = sidecar.read_text()
        meta['inference']['segments'][1]['raw_start_idx'] += 1
        sidecar.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, 'mapping does not match'):
            load_denoised_joint_mi_table(table, source)
        sidecar.write_text(original)
        f = pd.read_csv(table)
        f.loc[0, 'window_start_idx'] += 1
        f.to_csv(table, index=False)
        meta = json.loads(original)
        meta['table_sha256'] = file_sha256(table)
        sidecar.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, 'does not match inference timestamps'):
            load_denoised_joint_mi_table(table, source)

    def test_denoise_rejects_disabling_required_filters(self):
        with self.assertRaisesRegex(ValueError, 'incompatible'):
            self.cli('--no-bandpass')


if __name__ == '__main__':
    unittest.main()
