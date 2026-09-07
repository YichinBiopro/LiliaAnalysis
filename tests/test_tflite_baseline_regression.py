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

import plot_tflite_summary as summary
from lilia.tflite import build_tflite_timeline, run_tflite_recording, apply_tflite_with_time
from lilia.tflite_baseline import score_baseline_windows, select_tflite_baseline, KEYS
from lilia.tflite_io import load_tflite_table
from lilia.provenance import file_sha256
from lilia.qeeg import compute_qeeg_indices
from lilia.signal import resample_with_time
from lilia.windowing import continuous_slices


def good_quality(data, **kwargs):
    return {'overall': np.ones(data.shape[0])}


class TFLiteBaselineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def catalog(self, t, data, timeline, scorer=good_quality):
        return score_baseline_windows(t, data, data, timeline, scorer=scorer, quality_params={})

    def test_real_continuous_inference_matches_reference(self):
        ref = json.loads((Path(__file__).parent / 'fixtures/tflite_continuous_reference.json').read_text())
        self.assertEqual(file_sha256(summary.TFLITE_MODEL_PATH), ref['model_sha256'])
        x = np.random.default_rng(ref['seed']).normal(size=ref['shape']).astype(np.float32)
        t = np.arange(len(x)) * 2000
        filt = summary.bandpass_filter(x, fs=500, lo=.5, hi=45)
        timeline, _, output = run_tflite_recording(t, filt, summary.TFLITE_MODEL_PATH)
        expected = np.load(Path(__file__).parent / 'fixtures/tflite_continuous_reference.npz')
        np.testing.assert_allclose(output, expected['output'], atol=1e-6, rtol=1e-6)
        np.testing.assert_array_equal(timeline.time_us, expected['time_us'])

    def test_each_source_segment_trims_separately_and_keeps_small_gaps(self):
        t = np.r_[np.arange(1001) * 2000, 2008000 + np.arange(1001) * 2000]
        x = np.random.default_rng(2).normal(size=(len(t), 4))
        with patch('lilia.tflite.apply_tflite_windowed', side_effect=lambda x, *_: x[:, :2]) as infer:
            timeline, before, after = run_tflite_recording(t, x, 'unused')
        self.assertEqual(infer.call_count, 1)
        self.assertEqual(before.shape, (800, 4))
        np.testing.assert_array_equal(before[:, :2], after)
        self.assertEqual([r['trimmed_samples'] for r in timeline.segments], [1, 1])
        self.assertEqual(len(continuous_slices(timeline.time_us, 200)), 1)
        grid = timeline.grid(2)
        np.testing.assert_array_equal(grid.starts, [0, 400])
        np.testing.assert_array_equal(grid.columns['segment_id'], [0, 1])
        for row in timeline.model_windows:
            self.assertLessEqual(row['raw_end_idx'], timeline.segments[row['segment_id']]['raw_end_idx'])

    def test_resampling_propagates_ids_to_existing_tflite_adapter(self):
        t = np.r_[np.arange(1001) * 2000, 2008000 + np.arange(1001) * 2000]
        x = np.random.default_rng(4).normal(size=(len(t), 4))
        t200, x200, ids = resample_with_time(t, x, 500, 200, return_segment_ids=True)
        with patch('lilia.tflite.apply_tflite_windowed', side_effect=lambda x, *_: x[:, :2]):
            out_t, out = apply_tflite_with_time(t200, x200, 'unused', segment_ids=ids)
        np.testing.assert_array_equal(out_t, t200[np.r_[0:400, 401:801]])
        np.testing.assert_array_equal(out, x200[np.r_[0:400, 401:801], :2])

    def test_baseline_uses_complete_output_blocks_without_inference_or_psd_concat(self):
        t = np.arange(8000) * 2000
        raw = np.random.default_rng(8).normal(size=(len(t), 4))
        timeline = build_tflite_timeline(t)
        output = np.random.default_rng(9).normal(size=(len(timeline.time_us), 2))
        catalog = self.catalog(t, raw, timeline)
        with patch('lilia.tflite.apply_tflite_windowed', side_effect=AssertionError('no baseline inference')):
            ref, meta = select_tflite_baseline(output, timeline, catalog, 0, 16000000,
                                               required_sec=6, seed=42)
        self.assertEqual(meta['n_selected'], 3)
        self.assertEqual(meta['selected_nominal_sec'], 6)
        for key in KEYS:
            expected = []
            for row in meta['selected']:
                block = output[row['output_start_idx']:row['output_end_idx']]
                self.assertEqual(len(block), 400)
                expected.append([compute_qeeg_indices(block[:, ch], fs=200)[key] for ch in range(2)])
            np.testing.assert_allclose(ref[key], np.mean(expected, axis=0), atol=1e-12)
        again = select_tflite_baseline(output, timeline, catalog, 0, 16000000, required_sec=6, seed=42)[1]
        self.assertEqual(meta['selected'], again['selected'])
        with self.assertRaisesRegex(ValueError, 'Packed raw baseline'):
            summary._tflite_qeeg_reference(raw)

    def test_quality_invalid_or_saturated_half_rejects_whole_model_window(self):
        t = np.arange(3000) * 2000
        x = np.zeros((len(t), 4))
        x[500:1000] = 2048
        timeline = build_tflite_timeline(t)
        scores = iter([1, np.nan, 1, 1])
        def scorer(data, **kwargs):
            return {'overall': np.array([next(scores)] * 4)}
        catalog = self.catalog(t, x, timeline, scorer)
        self.assertEqual([r['reason'] for r in catalog], ['raw_saturation', 'nonfinite_quality', ''])
        with self.assertRaisesRegex(ValueError, 'Insufficient complete'):
            select_tflite_baseline(np.zeros((1200, 2)), timeline, catalog, 0, 6000000, required_sec=4)

    def test_pre_event_search_requires_entire_window_and_retains_buffer(self):
        t = np.arange(10000) * 2000
        x = np.zeros((len(t), 4))
        timeline = build_tflite_timeline(t)
        catalog = self.catalog(t, x, timeline)
        _, meta = select_tflite_baseline(np.zeros((4000, 2)), timeline, catalog,
                                         5000000, 17000000, required_sec=10)
        self.assertEqual([r['window_start_us'] for r in meta['selected']], [6000000, 8000000, 10000000, 12000000, 14000000])
        with self.assertRaisesRegex(ValueError, 'Insufficient'):
            select_tflite_baseline(np.zeros((4000, 2)), timeline, catalog, 5000000, 16000000, required_sec=12)
        with self.assertRaisesRegex(ValueError, 'must divide'):
            score_baseline_windows(t, x, x, timeline, scorer=good_quality, quality_params={}, epoch_sec=.7)

    def test_short_segments_and_bad_model_outputs_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, 'No complete'):
            run_tflite_recording(np.arange(100) * 2000, np.ones((100, 4)), 'unused')
        with patch('lilia.tflite.apply_tflite_windowed', return_value=np.ones((399, 2))), \
             self.assertRaisesRegex(ValueError, 'retained timeline'):
            run_tflite_recording(np.arange(1000) * 2000, np.ones((1000, 4)), 'unused')

    def run_plot(self, scorer=good_quality, **kwargs):
        folder = self.root / 'Demo'
        folder.mkdir(exist_ok=True)
        t = np.r_[np.arange(16000) * 2000, 100000000 + np.arange(16000) * 2000]
        x = np.random.default_rng(10).normal(size=(len(t), 4)).astype(np.float32)
        frame = pd.DataFrame(x, columns=['ch1', 'ch2', 'ch3', 'ch4'])
        frame.insert(0, 'Time[us]', t)
        source = folder / 'merged.csv'
        source.write_text('File Name,test\nAmp Gain,500,Abs Time Offset[us],0\n'
                          'Channels,1,2,3,4\nSample Rate,500,500,500,500\n' + frame.to_csv(index=False))
        figures = []
        def capture(fig, *args, **kwargs):
            figures.append(fig)
        with patch('lilia.tflite.apply_tflite_windowed', side_effect=lambda x, *_: x[:, :2]), \
             patch.object(summary, 'get_eeg_quality_index_v2_parametric', side_effect=scorer), \
             patch.object(summary, 'EVENTS', []), patch.object(summary, 'CONE_STAGES', []), \
             patch.object(Figure, 'savefig', capture), contextlib.redirect_stdout(io.StringIO()):
            paths = summary.plot_subject_tflite_summary('Demo', {'dir': 'Demo', 'sn': 'S1'},
                        str(self.root / 'out'), base_dir=str(self.root), **kwargs)
        return source, paths, figures

    def test_summary_gaps_metadata_and_heatmap_physical_bounds(self):
        source, paths, figures = self.run_plot(heatmap_baseline_mode='both', on_insufficient='skip')
        self.assertEqual(len(paths), 1)
        table = self.root / 'out/Demo_S1_tflite_metrics.csv'
        frame, meta = load_tflite_table(table, source, summary.TFLITE_MODEL_PATH)
        self.assertEqual(meta['kind'], 'tflite_qeeg')
        self.assertEqual(frame.segment_id.tolist(), [0] * 6 + [1] * 6)
        self.assertEqual(frame.time_s.tolist(), [2.5, 7.5, 12.5, 17.5, 22.5, 27.5,
                                                102.5, 107.5, 112.5, 117.5, 122.5, 127.5])
        self.assertTrue(np.isnan(figures[0].axes[0].lines[0].get_ydata()[6]))
        meshes = figures[0].axes[5].collections
        self.assertEqual(len(meshes), 2)
        for mesh in meshes:
            coords = mesh.get_coordinates()
            self.assertAlmostEqual((coords[0, -1, 0] - coords[0, 0, 0]) * 86400, 30)
        audit = json.loads((self.root / 'out/Demo_S1_tflite_analysis.json').read_text())
        self.assertEqual(audit['baselines'][0]['n_selected'], 5)
        self.assertEqual([r['center_us'] for r in audit['heatmap_bins']], [15000000, 115000000])
        for row in audit['heatmap_bins']:
            self.assertEqual(row['window_end_us'] - row['window_start_us'], 30000000)
            self.assertEqual(len(row['metric_rows']), 6)
        sidecar = Path(str(table) + '.meta.json')
        metadata = json.loads(sidecar.read_text())
        metadata['inference']['model_windows'][0]['raw_start_idx'] += 1
        sidecar.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, 'mapping does not match'):
            load_tflite_table(table, source)

    def test_all_invalid_quality_cannot_produce_a_zero_baseline(self):
        def invalid(data, **kwargs):
            return {'overall': np.full(data.shape[0], np.nan)}
        with self.assertRaisesRegex(ValueError, 'No usable TFLite baseline'):
            self.run_plot(invalid, on_insufficient='skip')
        audit = json.loads((self.root / 'out/Demo_S1_tflite_analysis.json').read_text())
        self.assertEqual(audit['baselines'][0]['status'], 'excluded')
        frame = pd.read_csv(self.root / 'out/Demo_S1_tflite_metrics.csv')
        self.assertFalse(frame.quality_valid.any())
        self.assertTrue(frame.focus_ch1.isna().all())


if __name__ == '__main__':
    unittest.main()
