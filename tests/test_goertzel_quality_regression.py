import contextlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import plot_goertzel_vs_raw as plot
from lilia import quality
from lilia.goertzel_io import load_goertzel_table
from lilia.provenance import file_sha256
from lilia.quality_audit import COLUMNS
from lilia.quality_policy import valid_goertzel_rows


def write_source(path, *, gap=False, n=2000):
    t = np.arange(n, dtype=np.int64) * 2000 + 1700000000000000
    if gap:
        t[n//2:] += 2000000
    x = np.random.default_rng(527).normal(0, 5, (n, 2)).astype(np.float32)
    x[:min(n, 250), 0] += 2000  # raw artifact attenuated by filtering
    frame = pd.DataFrame({'Time[us]': t, 'ch1': x[:, 0], 'ch2': x[:, 1]})
    path.write_text('File Name,synthetic\nAmp Gain,500,Abs Time Offset[us],0\n'
                    'Channels,1,2\nSample Rate,500,500\n' + frame.to_csv(index=False))
    return t, x


class GoertzelQualityRegressionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source.csv'
        write_source(self.source)
        self.table = self.root / 'metrics.csv'

    def generate(self, **kwargs):
        args = dict(merged_csv=str(self.source), out_png=str(self.root / 'plot.png'),
            out_csv=str(self.table), ch=1, fs=500., target_freq=60., win_sec=1., step_sec=1.,
            smooth_win=1, quality_threshold=.5, raw_ylim=(-150., 150.),
            exclude_hard_artifact=True, sat_uv=1950., sat_frac_threshold=.12,
            step_ptp_threshold=1000., bp_shift_sec=1., bp_shift_threshold=80.)
        args.update(kwargs)
        with patch.object(plot, '_render_plot'):
            plot._plot_subject(**args)
        return load_goertzel_table(self.table, self.source)

    def test_source_roundtrip_and_cache_rerender_preserve_diagnostics(self):
        frame, meta = self.generate()
        self.assertEqual(set(frame.quality_stage), {'filtered'})
        self.assertEqual(meta['feature_sources']['max_abs_diff_uv'], 'raw')
        self.assertEqual(meta['feature_sources']['bp_edge_shift_uv'], 'filtered')
        self.assertGreater(frame.sat_frac_1950.iloc[0], 0)
        self.assertEqual(frame.quality_final.iloc[0], 0)
        self.assertEqual(meta['quality_audit'][0]['quality_diagnostics']['request']['n_channels'], 2)
        with patch.object(plot, '_render_plot') as render:
            plot._plot_subject_from_csv(str(self.source), str(self.table), str(self.root/'again.png'),
                1, 500., 60., 1, .5, (-150., 150.), True, True)
        self.assertEqual(render.call_args.kwargs['result'].quality_audit, meta['quality_audit'])

    def test_rehashed_forgery_rejected(self):
        self.generate()
        original = self.table.read_bytes()
        sidecar = Path(str(self.table) + '.meta.json')
        original_meta = sidecar.read_text()
        for kind in ('stage', 'component', 'quality', 'raw_feature', 'edge_feature', 'hard',
                     'mapping', 'drop_row', 'missing_column', 'undeclared', 'feature_source'):
            with self.subTest(kind=kind):
                self.table.write_bytes(original)
                meta = json.loads(original_meta)
                frame = pd.read_csv(self.table, float_precision='round_trip')
                record = meta['quality_audit'][0]['quality_diagnostics']
                if kind == 'stage':
                    record['request']['stage'] = 'raw'
                    frame.loc[0, 'quality_stage'] = 'raw'
                elif kind == 'component':
                    record['result']['detail']['flat'][0] = .123
                elif kind == 'quality':
                    frame.loc[0, 'quality'] = .123
                elif kind == 'raw_feature':
                    frame.loc[0, 'max_abs_diff_uv'] += 1
                elif kind == 'edge_feature':
                    frame.loc[0, 'bp_edge_shift_uv'] += 1
                elif kind == 'hard':
                    frame.loc[0, 'artifact_hard_clip'] = 0
                elif kind == 'mapping':
                    frame['window_start_idx'] = frame.window_start_idx.astype(float)
                    frame.loc[0, 'window_start_idx'] += .5
                elif kind == 'drop_row':
                    frame = frame.iloc[1:]
                    meta['quality_audit'] = meta['quality_audit'][1:]
                elif kind == 'missing_column':
                    frame = frame.drop(columns=['quality_stage'])
                elif kind == 'undeclared':
                    del meta['quality_diagnostics_version']
                else:
                    meta['feature_sources']['bp_edge_shift_uv'] = 'raw'
                frame.to_csv(self.table, index=False)
                meta['table_sha256'] = file_sha256(self.table)
                sidecar.write_text(json.dumps(meta))
                with self.assertRaises(ValueError):
                    load_goertzel_table(self.table, self.source)

    def test_legacy_table_and_injected_scorer_remain_explicit(self):
        def old_scorer(data, fs, params):
            return {'overall': np.full(data.shape[0], .7)}
        with patch.object(plot, 'get_eeg_quality_index_v2_parametric', side_effect=old_scorer):
            frame, meta = self.generate()
        self.assertEqual(set(frame.quality_diagnostic_state), {'unavailable'})
        for key in ('quality_diagnostics_version', 'quality_audit', 'feature_sources', 'kind'):
            del meta[key]
        frame.drop(columns=list(COLUMNS)).to_csv(self.table, index=False)
        meta['table_sha256'] = file_sha256(self.table)
        Path(str(self.table) + '.meta.json').write_text(json.dumps(meta))
        reread, legacy = load_goertzel_table(self.table, self.source)
        self.assertEqual(len(reread), len(frame))
        self.assertNotIn('quality_audit', legacy)

    def test_finite_fallback_does_not_change_selection(self):
        params = {key+'_weight': float(key == 'spectrum') for key in ('flat', 'spectrum', 'kurtosis', 'corr')}
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(plot, 'get_ibrain_device_eeg_quality_v2_params', return_value=params))
            stack.enter_context(patch.object(quality.sp_signal, 'welch', side_effect=RuntimeError('forced')))
            frame, _ = self.generate()
        self.assertEqual(set(frame.quality_diagnostic_state), {'invalid'})
        np.testing.assert_array_equal(frame.quality, .5)
        self.assertTrue(valid_goertzel_rows(frame, .49, exclude_hard=False).all())
        self.assertFalse(valid_goertzel_rows(frame, .5, exclude_hard=False).any())
        load_goertzel_table(self.table, self.source)  # historical fallback after injection is removed

    def test_half_second_gap_and_no_candidate_windows(self):
        write_source(self.source, gap=True)
        frame, _ = self.generate(win_sec=.5, step_sec=.5)
        self.assertEqual(len(frame), 8)
        self.assertEqual(set(frame.quality_diagnostic_state), {'invalid'})
        self.assertGreater(frame.time_s.iloc[4] - frame.time_s.iloc[3], 2)
        frame, meta = self.generate(win_sec=5.)
        self.assertEqual(len(frame), 0)
        self.assertEqual(meta['quality_audit'], [])

    def test_nonfinite_helper_diagnostic_and_source_rejection(self):
        t, raw = write_source(self.source)
        raw[50, 0] = np.nan
        with np.errstate(all='ignore'):
            result = plot._compute_window_metrics(t, raw, raw, 500., 1, 60., .5, .5,
                quality.get_ibrain_device_eeg_quality_v2_params(), 1950., .12, 1000., 1., 80.)
        self.assertEqual(result.quality_audit[0]['quality_diagnostics']['state'], 'invalid')
        text = self.source.read_text().splitlines()
        text[55] = text[55].split(',')[0] + ',nan,1'
        self.source.write_text('\n'.join(text) + '\n')
        with self.assertRaisesRegex(ValueError, 'Non-finite EEG samples'):
            self.generate()

    def test_scorer_exception_propagates(self):
        with patch.object(plot, 'get_eeg_quality_index_v2_parametric', side_effect=RuntimeError('sentinel')):
            with self.assertRaisesRegex(RuntimeError, 'sentinel'):
                self.generate()
        self.assertFalse(self.table.exists())

    def test_plot_marks_invalid_low_and_hard(self):
        t, raw = write_source(self.source)
        result = plot._compute_window_metrics(t, raw, raw, 500., 1, 60., .5, .5,
            quality.get_ibrain_device_eeg_quality_v2_params(), 1950., .12, 1000., 1., 80.)
        with patch.object(plot.plt, 'close'), patch('matplotlib.figure.Figure.savefig'):
            plot._render_plot(str(self.root/'plot.png'), result, (t-t[0])/1e6, raw[:, 0],
                (t-t[0])/1e6, raw[:, 0], 'synthetic', 1, 60., 1, .5, (-150., 150.), True, [])
            ax = plot.plt.gcf().axes[1]
            labels = ax.get_legend_handles_labels()[1]
            self.assertIn('Diagnostic invalid (legacy scores)', labels)
            self.assertIn('At/below threshold', labels)
            self.assertTrue(any(label.startswith('Hard artifact') for label in labels))
        plot.plt.close('all')

    def test_gap_plot_breaks_all_four_panels(self):
        t, raw = write_source(self.source, gap=True)
        result = plot._compute_window_metrics(t, raw, raw, 500., 1, 60., 1., 1.,
            quality.get_ibrain_device_eeg_quality_v2_params(), 1950., .12, 1000., 1., 80.)
        tx, y = plot._load_raw_decimated(t, raw[:, 0])
        np.testing.assert_array_equal(result.segment_ids, [0, 0, 1, 1])
        with patch.object(plot.plt, 'close'), patch('matplotlib.figure.Figure.savefig'):
            plot._render_plot(str(self.root/'plot.png'), result, tx, y, tx, y,
                'gap', 1, 60., 1, .5, (-150., 150.), True, [])
            for ax in plot.plt.gcf().axes:
                self.assertTrue(np.isnan(ax.lines[0].get_ydata()).any())
        plot.plt.close('all')


if __name__ == '__main__':
    unittest.main()
