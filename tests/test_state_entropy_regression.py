import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import spectral_entropy as entropy
from lilia.io import bandpass_filter
from lilia.entropy_io import config_id
from lilia.provenance import file_sha256
from lilia.state_entropy_io import load_state_entropy_table
from lilia.state_windows import select_state_windows


class StateEntropyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def source(self, t, x):
        path = self.root / 'raw.csv'
        frame = pd.DataFrame(x, columns=[f'ch{i+1}' for i in range(x.shape[1])])
        frame.insert(0, 'Time[us]', t)
        with path.open('w') as handle:
            handle.write('File Name,test\nAmp Gain,500,Abs Time Offset[us],0\nChannels,1,2\nSample Rate,500,500\n')
            frame.to_csv(handle, index=False)
        return path

    def run_cli(self, source, *extra):
        args = ['spectral_entropy.py', '--csv', str(source), '--out', str(self.root),
                '--baseline', '0', '6', '--event', '20', '26', *extra]
        with patch('sys.argv', args), contextlib.redirect_stdout(io.StringIO()):
            entropy.main()
        return self.root / 'raw_baseline_event_entropy_ch1.csv'

    def gapped(self):
        t = 1778566248498593 + np.r_[np.arange(3000)*2000, 20000000+np.arange(3000)*2000]
        x = np.random.default_rng(29).normal(size=(len(t), 4)).astype(np.float32)
        return t, x

    def test_continuous_ordinary_and_real_quality_match_saved_reference(self):
        r = json.loads((Path(__file__).parent / 'fixtures/state_entropy_continuous_reference.json').read_text())
        x = np.random.default_rng(r['seed']).normal(size=r['shape']).astype(np.float32)
        t = r['epoch_us'] + np.arange(len(x))*2000
        y = bandpass_filter(x)
        result = entropy.compare_baseline_event(y[:, 0], *r['ranges'], step_sec=1, time_us=t)
        def check_state(actual, expected):
            for key, value in expected.items():
                if isinstance(value, dict):
                    for band, number in value.items():
                        np.testing.assert_allclose(actual[key][band], number, atol=1e-12, rtol=1e-12)
                else:
                    np.testing.assert_allclose(actual[key], value, atol=1e-12, rtol=1e-12)
        for label, (lo, hi) in zip(('baseline', 'event'), r['ranges']):
            check_state(result[label], r['ordinary'][label])
            epochs, audit = entropy.collect_clean_epochs(t, y, x, t[0]+lo*1000000, t[0]+hi*1000000, 0)
            check_state(entropy.compute_state_entropy_from_epochs(epochs), r['clean'][label]['state'])
            for key, value in r['clean'][label]['meta'].items():
                self.assertEqual(audit[key], value)
        for key in ('delta_pooled_entropy', 'delta_mean_window_entropy', 'mannwhitneyu_u', 'mannwhitneyu_p'):
            self.assertAlmostEqual(result[key], r['ordinary'][key], places=12)
        source = self.source(t, x)
        table = self.run_cli(source, '--baseline', '1', '15', '--event', '18', '31', '--step', '1')
        frame, _ = load_state_entropy_table(table, source, 1)
        np.testing.assert_allclose(frame.pooled_entropy, [r['ordinary'][s]['pooled_entropy'] for s in ('baseline', 'event')], atol=1e-12)
        table = self.run_cli(source, '--baseline', '1', '15', '--event', '18', '31', '--clean')
        frame, _ = load_state_entropy_table(table, source, 1)
        np.testing.assert_allclose(frame.pooled_entropy, [r['clean'][s]['state']['pooled_entropy'] for s in ('baseline', 'event')], atol=1e-12)

    def test_gapped_cli_matches_independent_segment_filter_and_psds(self):
        t, x = self.gapped()
        source = self.source(t, x)
        table = self.run_cli(source)
        frame, meta = load_state_entropy_table(table, source, 1)
        self.assertEqual(frame.n_windows.tolist(), [3, 3])
        for i, label in enumerate(('baseline', 'event')):
            y = bandpass_filter(x[i*3000:(i+1)*3000])
            expected = entropy.compute_state_entropy(y[:, 0])
            self.assertAlmostEqual(frame.iloc[i].pooled_entropy, expected['pooled_entropy'], places=12)
            rows = meta['states'][label]['windows']
            self.assertTrue(all(row['segment_id'] == i for row in rows))
            self.assertEqual(rows[0]['window_start_us'], int(t[i*3000]))
            self.assertTrue(all(row['quality_state'] == 'disabled' for row in rows))

    def test_interval_crossing_gap_pools_only_independent_complete_windows(self):
        t, x = self.gapped()
        result = entropy.compare_baseline_event(x[:, 0], (0, 26), (20, 26), time_us=t)
        state = result['baseline']
        expected = entropy.compute_state_entropy_from_epochs([x[s:s+1000, 0] for s in range(0, 6000, 1000)])
        np.testing.assert_allclose(state['mean_psd'], expected['mean_psd'])
        self.assertEqual(state['n_windows'], 6)
        self.assertEqual(state['audit']['missing_spans'], [{'start_us': int(t[0]+6000000), 'end_us': int(t[3000]), 'reason': 'no_recorded_samples'}])

    def test_half_open_fractional_boundaries_and_short_segments(self):
        t = np.r_[np.arange(5)*2000, 10000000+np.arange(2001)*2000]
        a = select_state_windows(t, 500, 1, 14000001, 2, 1)
        complete = [r for r in a['windows'] if r['complete']]
        self.assertEqual([r['window_start_idx'] for r in complete], [5, 505, 1005])
        self.assertTrue(all(r['window_start_us'] >= 1 and r['window_end_us'] <= 14000001 for r in complete))
        self.assertEqual(a['windows'][0]['status'], 'incomplete_window')
        edge = select_state_windows(np.arange(2000)*2000, 500, 1, 2000001, 2)
        self.assertFalse(any(r['complete'] for r in edge['windows']))
        for lo, hi in [(1, 1), (2, 1), (0, np.nan)]:
            with self.assertRaises(ValueError):
                select_state_windows(t, 500, lo, hi, 2)

    def test_clean_audits_quality_nonfinite_saturation_and_no_cross_gap(self):
        t, x = self.gapped()
        raw = x.copy()
        raw[1000:2000] = np.nan
        with patch.object(entropy, '_saturation_frac', side_effect=[1., 0., 0., 0., 0.]), \
             patch.object(entropy, '_eeg_quality_v2', side_effect=[{'overall': [np.nan]*4}, {'overall': [.1]*4}, {'overall': [.9]*4}, {'overall': [.9]*4}]):
            epochs, audit = entropy.collect_clean_epochs(t, x, raw, t[0], t[-1]+2000, 0)
        self.assertEqual([r['status'] for r in audit['windows']], ['raw_saturation', 'nonfinite_signal', 'invalid_quality', 'low_quality', 'accepted', 'accepted'])
        self.assertEqual(len(epochs), 2)
        np.testing.assert_array_equal(epochs[0], x[4000:5000, 0])
        self.assertEqual(audit['n_total_epochs'], 6)

    def test_all_excluded_and_empty_interval_still_save_verifiable_audit(self):
        t, x = self.gapped()
        source = self.source(t, x)
        with patch.object(entropy, '_eeg_quality_v2', return_value={'overall': [np.nan]*4}), \
             self.assertRaisesRegex(ValueError, 'audit saved'):
            self.run_cli(source, '--clean')
        table = self.root / 'raw_baseline_event_entropy_ch1.csv'
        frame, meta = load_state_entropy_table(table, source)
        self.assertEqual(frame.n_windows.tolist(), [0, 0])
        self.assertEqual(meta['states']['baseline']['n_invalid_quality_epochs'], 3)
        with self.assertRaisesRegex(ValueError, 'audit saved'):
            self.run_cli(source, '--baseline', '7', '9')
        frame, meta = load_state_entropy_table(table, source)
        self.assertEqual(frame.n_windows.tolist(), [0, 3])
        self.assertEqual(meta['states']['baseline']['windows'], [])
        self.assertEqual(len(meta['states']['baseline']['missing_spans']), 1)

    def test_metadata_rejects_source_channel_config_and_rehashed_geometry(self):
        t, x = self.gapped()
        source = self.source(t, x)
        table = self.run_cli(source)
        with self.assertRaisesRegex(ValueError, 'channel differs'):
            load_state_entropy_table(table, source, 2)
        sidecar = Path(str(table)+'.meta.json')
        original = sidecar.read_text()
        meta = json.loads(original)
        meta['states']['event']['windows'][0]['window_start_idx'] += 1
        meta['audit_id'] = config_id(meta['states'])
        sidecar.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, 'mapping differs'):
            load_state_entropy_table(table, source)
        sidecar.write_text(original)
        frame = pd.read_csv(table)
        frame.loc[0, 'n_windows'] = 4
        frame.to_csv(table, index=False)
        meta = json.loads(original)
        meta['table_sha256'] = file_sha256(table)
        sidecar.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, 'counts differ'):
            load_state_entropy_table(table, source)
        source.write_text(source.read_text().replace('File Name,test', 'File Name,changed'))
        with self.assertRaisesRegex(ValueError, 'Raw recording differs'):
            load_state_entropy_table(table, source)

    def test_filter_only_contributing_segments_and_keep_finite_channel(self):
        t, x = self.gapped()
        t = np.r_[t[0]-1000000 + np.arange(4)*2000, t]
        x = np.vstack([np.ones((4, 4)), x])
        x[:, 1] = np.nan
        source = self.source(t, x)
        with patch.object(entropy, 'bandpass_filter', wraps=bandpass_filter) as filt:
            table = self.run_cli(source, '--baseline', '1', '7', '--event', '21', '27')
        self.assertEqual([call.args[0].shape for call in filt.call_args_list], [(3000, 1), (3000, 1)])
        frame, _ = load_state_entropy_table(table, source)
        self.assertEqual(frame.n_windows.tolist(), [3, 3])

    def test_failed_scorer_is_audited_and_clean_step_is_explicit(self):
        t, x = self.gapped()
        with patch.object(entropy, '_eeg_quality_v2', side_effect=RuntimeError('scorer unavailable')):
            epochs, audit = entropy.collect_clean_epochs(t, x, x, t[0], t[-1]+2000, 0)
        self.assertEqual(epochs, [])
        self.assertTrue(all(r['status'] == 'quality_error' for r in audit['windows']))
        with self.assertRaisesRegex(ValueError, 'non-overlapping'):
            self.run_cli(self.source(t, x), '--clean', '--step', '1')
        with self.assertRaisesRegex(ValueError, 'equal-length'):
            entropy.compute_state_entropy_from_epochs([np.ones(1000), np.ones(900)])

    def test_nonfinite_filter_context_and_too_short_filter_are_audited(self):
        t, x = self.gapped()
        x[0, 0] = np.nan
        source = self.source(t, x)
        with self.assertRaisesRegex(ValueError, 'audit saved'):
            self.run_cli(source)
        table = self.root / 'raw_baseline_event_entropy_ch1.csv'
        frame, meta = load_state_entropy_table(table, source)
        self.assertEqual(frame.n_windows.tolist(), [0, 3])
        self.assertEqual(meta['states']['baseline']['n_nonfinite_epochs'], 3)
        # Without a filter, one corrupt window need not discard its whole segment.
        table = self.run_cli(source, '--no-bandpass')
        frame, _ = load_state_entropy_table(table, source)
        self.assertEqual(frame.n_windows.tolist(), [2, 3])
        source = self.source(np.arange(8)*2000, np.ones((8, 4)))
        with self.assertRaisesRegex(ValueError, 'audit saved'):
            self.run_cli(source, '--baseline', '0', '.016', '--event', '0', '.016', '--win', '.016')
        frame, meta = load_state_entropy_table(table, source)
        self.assertEqual(meta['states']['baseline']['windows'][0]['status'], 'filter_error')


if __name__ == '__main__':
    unittest.main()
