"""Raw qEEG CLI: frozen legacy numbers, gaps, failure audit and source round-trip."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from lilia import qeeg
from lilia.entropy_io import config_id
from lilia.provenance import file_sha256
from lilia.qeeg_raw import analyze_raw_qeeg, METRIC_KEYS
from lilia.qeeg_io import load_qeeg_table

FIXTURES = Path(__file__).parent / 'fixtures'


class RawQeegTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.epoch = 1750000000000001

    def source(self, t, raw):
        path = self.root / 'source.csv'
        frame = pd.DataFrame(raw, columns=[f'ch{i+1}' for i in range(raw.shape[1])])
        frame.insert(0, 'Time[us]', t)
        path.write_text('File Name,test\nAmp Gain,500,Abs Time Offset[us],0\n'
                        'Channels,1,2\nSample Rate,500,500\n' + frame.to_csv(index=False))
        return path

    def gapped(self):
        # A short prefix, small (>3 periods) and large gaps, and a short tail.
        t = self.epoch + np.arange(17600, dtype=np.int64) * 2000
        t[100:] += 20000000
        t[5100:] += 6000
        t[12600:] += 10000000
        raw = np.random.default_rng(14).normal(size=(len(t), 2))
        return t, raw

    def cli(self, source, *args, expect=0):
        def save(fig, path, **kwargs):
            Path(path).write_text('unit test figure placeholder')
        output = io.StringIO()
        with patch.object(sys, 'argv', ['qeeg', '--csv', str(source), '--out', str(self.root / 'out'), *args]), \
                patch.object(Figure, 'savefig', save), contextlib.redirect_stdout(output), \
                contextlib.redirect_stderr(output):
            if expect:
                with self.assertRaises(SystemExit) as exc:
                    qeeg.main()
                self.assertEqual(exc.exception.code, expect)
            else:
                qeeg.main()
        return output.getvalue()

    def test_frozen_continuous_numbers_odd_and_truncated_windows(self):
        ref = json.loads((FIXTURES / 'qeeg_cli_continuous_reference.json').read_text())
        with np.load(FIXTURES / 'qeeg_cli_continuous_reference.npz') as arrays:
            for i, case in enumerate(ref['cases']):
                r = analyze_raw_qeeg(arrays['time_us'], arrays['raw'], win_sec=case['win_sec'])
                self.assertEqual(r['grid'].win, case['window_samples'])
                for key in METRIC_KEYS:
                    np.testing.assert_allclose(r['metrics'][key], arrays[f'case{i}_{key}'], rtol=1e-12, atol=1e-12)
                np.testing.assert_array_equal(r['grid'].time_s, arrays[f'case{i}_time'])
                np.testing.assert_array_equal(r['grid'].columns['window_center_us'],
                                              arrays['time_us'][r['grid'].starts+r['grid'].win//2])

    def test_timestamp_free_helper_and_shim_retain_legacy_contract(self):
        import qeeg_indices as shim
        with np.load(FIXTURES / 'qeeg_cli_continuous_reference.npz') as arrays:
            x = arrays['raw'][:, 0]
            for helper in (qeeg.compute_qeeg_indices_windowed, shim.compute_qeeg_indices_windowed):
                actual = helper(x, win_sec=1.0019)
                self.assertEqual(set(actual), {'time', *METRIC_KEYS})
                for key, values in actual.items():
                    np.testing.assert_array_equal(values, arrays[f'case1_{key}'])
                self.assertEqual(len(helper(x[:7])['time']), 0)
            overlap = qeeg.compute_qeeg_indices_windowed(x[:10000], win_sec=5, step_sec=2.5)
            np.testing.assert_array_equal(overlap['time'], [2.5, 5., 7.5, 10., 12.5, 15., 17.5])

    def test_cli_keeps_integer_microseconds_and_old_summary(self):
        ref = json.loads((FIXTURES / 'qeeg_cli_continuous_reference.json').read_text())
        with np.load(FIXTURES / 'qeeg_cli_continuous_reference.npz') as arrays:
            source = self.source(arrays['time_us'], arrays['raw'])
        t, _ = qeeg._load_eeg_csv(source)
        self.assertEqual(t.dtype, np.dtype('int64'))
        self.assertEqual(t[0], self.epoch)
        output = self.cli(source)
        for line in ref['cli_stdout'].splitlines():
            if line.startswith('  '):
                self.assertIn(line, output)
        frame, meta = load_qeeg_table(self.root / 'out/source_qeeg_ch1.csv', source, channel=1)
        self.assertEqual(meta['source_epoch_us'], self.epoch)
        self.assertTrue((frame.quality_state == 'disabled').all())

    def test_gaps_preserve_original_grid_segments_and_excluded_coverage(self):
        t, raw = self.gapped()
        r = analyze_raw_qeeg(t, raw)
        np.testing.assert_array_equal(r['grid'].starts, [2500, 7500, 10000, 15000])
        self.assertEqual(r['grid'].columns['segment_id'].tolist(), [1, 2, 2, 3])
        self.assertEqual([w['window_start_idx'] for w in r['analysis']['excluded_grid_windows']], [0, 5000, 12500])
        for row in r['analysis']['segments']:
            covered = (row['excluded_prefix_end_idx'] - row['excluded_prefix_start_idx']
                       + row['complete_windows']*2500 + row['tail_samples'])
            self.assertEqual(covered, row['raw_end_idx']-row['raw_start_idx'])
        self.assertEqual(r['analysis']['segments'][0]['status'], 'short_segment')
        self.assertEqual(r['analysis']['segments'][2]['gap_before_us'], 6000)
        self.assertEqual(r['grid'].time_s[-1], (t[16250]-t[0])/1e6)
        for row, start in enumerate(r['grid'].starts):
            expected = qeeg.compute_qeeg_indices(raw[start:start+2500, 0])
            for key in METRIC_KEYS:
                self.assertEqual(r['metrics'][key][row], expected[key])

    def test_nonfinite_window_does_not_shift_rows_or_poison_other_channels(self):
        t, raw = self.gapped()
        raw[:, 1] = np.inf
        raw[7501, 0] = np.nan
        r = analyze_raw_qeeg(t, raw)
        self.assertEqual(r['valid'].tolist(), [True, False, True, True])
        self.assertEqual(r['analysis']['window_audit'][1]['status'], 'nonfinite_window')
        for key in METRIC_KEYS:
            self.assertTrue(np.isnan(r['metrics'][key][1]))
        changed = raw.copy(); changed[100:5100, 0] *= 100
        other = analyze_raw_qeeg(t, changed)
        for key in METRIC_KEYS:
            np.testing.assert_array_equal(r['metrics'][key][2:], other['metrics'][key][2:])
        source = self.source(t, raw)
        self.cli(source)
        frame, _ = load_qeeg_table(self.root / 'out/source_qeeg_ch1.csv', source)
        self.assertEqual(len(frame), 4)

    def test_nonfinite_tail_and_prefix_are_audited_without_excluding_valid_window(self):
        t, raw = self.gapped()
        raw[101, 0] = np.nan
        raw[-1, 0] = np.inf
        r = analyze_raw_qeeg(t, raw)
        self.assertTrue(r['valid'].all())
        self.assertEqual([s['nonfinite_samples'] for s in r['analysis']['segments']], [0, 1, 0, 1])

    def test_exact_minimum_short_and_unaligned_segment_boundaries(self):
        raw = np.ones((8, 1))
        t = self.epoch + np.arange(8)*2000
        self.assertEqual(analyze_raw_qeeg(t, raw, win_sec=.016)['grid'].win, 8)
        self.assertEqual(analyze_raw_qeeg(t[:7], raw[:7], win_sec=.016)['analysis']['status'], 'no_complete_windows')
        t = self.epoch + np.arange(15)*2000; t[7:] += 20000
        r = analyze_raw_qeeg(t, np.ones((15, 1)), win_sec=.016)
        self.assertEqual(r['analysis']['segments'][1]['status'], 'no_grid_window')
        self.assertIsNone(r['grid'])

    def test_gap_threshold_and_timestamp_jitter_use_actual_centres(self):
        raw = np.ones((16, 1))
        t = self.epoch + np.arange(16)*2000; t[8:] += 4000
        self.assertEqual(len(analyze_raw_qeeg(t, raw, win_sec=.016)['analysis']['segments']), 1)
        t[8:] += 1
        r = analyze_raw_qeeg(t, raw, win_sec=.016)
        self.assertEqual(len(r['analysis']['segments']), 2)
        self.assertEqual(r['grid'].columns['window_center_us'][1], t[12])
        jitter = self.epoch + np.arange(16)*2000; jitter[4] += 1
        self.assertEqual(analyze_raw_qeeg(jitter, raw, win_sec=.016)['grid'].time_s[0], .008001)

    def test_invalid_settings_shape_and_time_fail(self):
        t = self.epoch + np.arange(2500)*2000; raw = np.ones((2500, 1))
        for kwargs in ({'fs':0}, {'fs':np.inf}, {'win_sec':np.nan}, {'win_sec':.01},
                       {'channel':0}, {'channel':2}, {'channel':1.5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                analyze_raw_qeeg(t, raw, **kwargs)
        for times, data in ((t.astype(float), raw), (t[:-1], raw), (t[::-1], raw),
                            (np.zeros_like(t), raw), (t, raw[:,0]), (t[:0], raw[:0])):
            with self.assertRaises(ValueError):
                analyze_raw_qeeg(times, data)

    def test_all_short_and_all_excluded_cli_save_failure_audit(self):
        for size, value, status in ((7, 1., 'no_complete_windows'), (2500, np.nan, 'all_windows_excluded')):
            source = self.source(self.epoch + np.arange(size)*2000, np.full((size,1), value))
            self.cli(source, expect=1)
            audit = json.loads((self.root / 'out/source_qeeg_ch1_analysis_audit.json').read_text())
            self.assertEqual(audit['status'], 'failed')
            self.assertEqual(audit['analysis']['status'], status)
            self.assertEqual(audit['analysis']['summary']['focus']['mean'], None)
            if size == 2500:
                frame, _ = load_qeeg_table(self.root / 'out/source_qeeg_ch1.csv', source)
                self.assertFalse(frame.metric_valid.any())
            else:
                self.assertEqual(audit['artifacts'], {})

    def test_invalid_channel_and_settings_cli_audit(self):
        source = self.source(self.epoch + np.arange(2500)*2000, np.ones((2500, 1)))
        self.cli(source, '--ch', '2', expect=1)
        audit = json.loads((self.root / 'out/source_qeeg_ch2_analysis_audit.json').read_text())
        self.assertIn('channel 2 not found', audit['error'])
        self.cli(source, '--fs', 'nan', expect=1)
        audit = json.loads((self.root / 'out/source_qeeg_ch1_analysis_audit.json').read_text())
        self.assertIn('finite and positive', audit['error'])

    def test_metric_exception_and_nonfinite_result_leave_no_partial_values(self):
        t = self.epoch + np.arange(2500)*2000; raw = np.ones((2500,1))
        for kwargs, reason in (({'side_effect':RuntimeError('broken')}, 'metric_error'),
                               ({'return_value':dict.fromkeys(METRIC_KEYS,np.inf)}, 'nonfinite_metric')):
            with patch('lilia.qeeg_raw.compute_qeeg_indices', **kwargs):
                r = analyze_raw_qeeg(t,raw)
            self.assertEqual(r['analysis']['window_audit'][0]['status'], reason)
            self.assertFalse(r['valid'].any())
            self.assertTrue(all(np.isnan(r['metrics'][key]).all() for key in METRIC_KEYS))

    def test_two_panels_break_at_gaps_and_keep_isolated_windows(self):
        t, raw = self.gapped(); raw[7501,0] = np.nan
        r = analyze_raw_qeeg(t, raw)
        figures=[]
        with patch.object(Figure, 'savefig', lambda fig, *a, **kw: figures.append(fig)), \
                contextlib.redirect_stdout(io.StringIO()):
            qeeg.plot_qeeg_indices({'time':r['grid'].time_s, **r['metrics']}, 'test', self.root/'plot.png',
                                  segment_ids=r['grid'].columns['segment_id'])
        for ax, count in zip(figures[0].axes, (3, 4)):
            for line in ax.lines[:count]:
                self.assertEqual(line.get_marker(), '.')
                self.assertEqual(len(line.get_xdata()), 6)
                self.assertTrue(np.isnan(line.get_ydata()[1:3]).all())
                self.assertTrue(np.isfinite(line.get_ydata()[0]))
                self.assertTrue(np.isfinite(line.get_ydata()[-1]))

    def test_table_rejects_source_channel_metadata_and_rehashed_tampering(self):
        t, raw = self.gapped(); source = self.source(t,raw)
        self.cli(source)
        path = self.root/'out/source_qeeg_ch1.csv'
        sidecar = Path(str(path)+'.meta.json')
        original_csv, original_meta = path.read_bytes(), sidecar.read_bytes()
        with self.assertRaises(ValueError):
            load_qeeg_table(path, source, channel=2)
        for field, value in [('window_start_idx',2501), ('window_center_us',self.epoch),
                             ('segment_id',0), ('channel',2), ('theta', .99),
                             ('quality_state','passed'), ('metric_valid',False)]:
            frame = pd.read_csv(io.BytesIO(original_csv)); frame.loc[0,field] = value
            frame.to_csv(path,index=False)
            meta = json.loads(original_meta); meta['table_sha256'] = file_sha256(path)
            sidecar.write_text(json.dumps(meta))
            with self.subTest(field=field), self.assertRaises(ValueError):
                load_qeeg_table(path,source)
        path.write_bytes(original_csv)
        for part in ('segments', 'summary', 'policy'):
            meta = json.loads(original_meta)
            if part == 'segments':
                meta['analysis']['segments'][0]['tail_samples'] += 1
            elif part == 'summary':
                meta['analysis']['summary']['focus']['mean'] += .1
            else:
                meta['parameters']['baseline'] = 'session_start'
            meta['analysis_id'] = config_id(meta['analysis'])
            sidecar.write_text(json.dumps(meta))
            with self.subTest(part=part), self.assertRaises(ValueError):
                load_qeeg_table(path, source)
        sidecar.write_bytes(original_meta)
        changed = raw.copy(); changed[0,1] += .1; self.source(t,changed)
        with self.assertRaises(ValueError):
            load_qeeg_table(path,source)
        sidecar.unlink()
        with self.assertRaises(ValueError):
            load_qeeg_table(path,source)

    def test_empty_plot_keeps_source_time_and_labels_missing_values(self):
        figures = []
        with patch.object(Figure, 'savefig', lambda fig, *a, **kw: figures.append(fig)), \
                contextlib.redirect_stdout(io.StringIO()):
            qeeg.plot_qeeg_indices({'time': np.array([2.5]),
                                   **{key: np.array([np.nan]) for key in METRIC_KEYS}},
                                  'test', self.root/'empty.png', segment_ids=np.array([0]),
                                  time_limits=(0., 5.))
        for ax in figures[0].axes:
            self.assertEqual(ax.get_xlim(), (0., 5.))
            self.assertIn('No finite qEEG windows', [text.get_text() for text in ax.texts])


if __name__ == '__main__':
    unittest.main()
