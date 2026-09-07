import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import joint_mi
import spectral_entropy as spectral
from lilia.event_windows import select_event_windows


class EventWindowTests(unittest.TestCase):
    def setUp(self):
        self.fs = 200
        self.t = np.r_[np.arange(4000) * 5000, 40000000 + np.arange(8000) * 5000]
        self.x = np.random.default_rng(314).normal(size=(len(self.t), 2))

    def test_half_open_true_time_after_gap_and_between_samples(self):
        windows, rows = select_event_windows(self.t, self.fs, [('event', 50002500)], 5)
        self.assertEqual(rows[0]['Status'], 'accepted')
        self.assertEqual((windows[0].pre.start, windows[0].pre.stop), (5001, 6001))
        self.assertEqual((windows[0].post.start, windows[0].post.stop), (6001, 7001))
        self.assertEqual(windows[0].segment_id, 1)
        np.testing.assert_array_equal(self.t[windows[0].pre],
                                      self.t[(self.t >= 45002500) & (self.t < 50002500)])

    def test_gap_onset_pre_post_and_incomplete_intervals(self):
        events = [('onset_gap', 30000000), ('pre_gap', 42000000),
                  ('post_gap', 18000000), ('pre_short', 2000000),
                  ('post_short', 78000000), ('outside', 90000000),
                  ('gap_boundary', 40000000), ('end_exact', 75000000)]
        windows, rows = select_event_windows(self.t, self.fs, events, 5)
        self.assertEqual([row['Reason'] for row in rows], [
            'onset_in_gap', 'pre_crosses_gap', 'post_crosses_gap',
            'pre_outside_recording', 'post_outside_recording', 'onset_outside_recording',
            'pre_crosses_gap', ''])
        self.assertEqual([w.name for w in windows], ['end_exact'])
        self.assertEqual(windows[0].post.stop, len(self.t))

    def test_continuous_numerics_match_pre_refactor(self):
        ref = json.loads((Path(__file__).parent / 'fixtures/band_event_continuous_reference.json').read_text())
        x = np.random.default_rng(ref['seed']).normal(size=ref['shape'])
        actual = spectral.run_band_event_mi_pipeline(
            {'ch1': x[:, 0], 'ch2': x[:, 1]}, [], fs=ref['fs'],
            windows_sec=ref['windows_sec'], n_surrogates=ref['n_surrogates'],
            time_us=np.arange(len(x)) * 5000, events=[('event', 15000000)], front_bandpass=True)
        for got, expected in zip(actual.to_dict('records'), ref['records'], strict=True):
            for key, value in expected.items():
                if isinstance(value, str):
                    self.assertEqual(got[key], value)
                else:
                    np.testing.assert_allclose(got[key], value, rtol=1e-12, atol=1e-12)

    def test_filter_hilbert_never_borrow_across_gap(self):
        env = spectral.extract_band_envelopes(self.x[:, 0], fs=self.fs,
                                              time_us=self.t, front_bandpass=True)
        for lo, hi in [(0, 4000), (4000, len(self.t))]:
            expected = spectral.extract_band_envelopes(self.x[lo:hi, 0], fs=self.fs,
                                                       front_bandpass=True)
            for key in env:
                np.testing.assert_array_equal(env[key][lo:hi], expected[key])
        changed = self.x[:, 0].copy()
        changed[:4000] *= 1000
        actual = spectral.extract_band_envelopes(changed, fs=self.fs, time_us=self.t,
                                                segment_ids={1}, front_bandpass=True)
        for key in env:
            self.assertTrue(np.isnan(actual[key][:4000]).all())
            np.testing.assert_array_equal(actual[key][4000:], env[key][4000:])

    def test_short_unused_segment_and_per_duration_eligibility(self):
        t = np.r_[np.arange(5) * 5000, 40000000 + np.arange(8000) * 5000]
        x = np.random.default_rng(2).normal(size=len(t))
        audit = []
        frame = spectral.run_band_event_mi_pipeline({'ch1': x}, [], fs=self.fs,
            time_us=t, events=[('event', 47000000)], windows_sec=(5, 10),
            front_bandpass=True, audit=audit)
        self.assertEqual(frame.Window_Size.tolist(), [5])
        self.assertEqual([r['Reason'] for r in audit], ['', 'pre_crosses_gap'])

    def test_nonfinite_channel_is_excluded_and_quality_not_claimed(self):
        x = self.x.copy()
        x[4500, 0] = np.nan
        audit = []
        frame = spectral.run_band_event_mi_pipeline({'ch1': x[:, 0], 'ch2': x[:, 1]}, [],
            fs=self.fs, time_us=self.t, events=[('event', 50000000)],
            windows_sec=(5,), front_bandpass=True, audit=audit)
        self.assertEqual(frame.Channel.tolist(), ['ch2'])
        self.assertEqual(frame.Quality_State.tolist(), ['disabled'])
        self.assertEqual(audit[0]['Reason'], 'nonfinite_signal_or_envelope')

    def test_participation_and_legacy_index_interfaces_use_real_time(self):
        events = [('restricted', '12:00', 1, ['Hardy']), ('all', '12:01', 1, None)]
        args = argparse.Namespace(ibrain_events=True, subject='Ann', fs=self.fs)
        with patch.object(spectral, '_IBRAIN_EVENTS', events), \
             patch.object(spectral, '_hhmm_to_us', return_value=50000000), \
             patch.object(joint_mi, 'EVENTS', events), \
             patch.object(joint_mi, 'hhmm_to_us', return_value=50000000):
            audit = []
            self.assertEqual(spectral._resolve_event_onsets(args, self.t, len(self.t),
                                                           audit=audit), [6000])
            self.assertEqual(audit[0]['Reason'], 'not_participant')
            self.assertEqual(joint_mi.resolve_events(self.t, len(self.t), self.fs, 1000,
                                                     subject='Ann'), [('all', 6000)])

    def test_pre_onset_histograms_use_complete_physical_intervals(self):
        events = [('valid', '12:00', 1, None), ('gap', '12:01', 1, None)]
        with patch.object(spectral, '_IBRAIN_EVENTS', events), \
             patch.object(spectral, '_hhmm_to_us', side_effect=[50000000, 30000000]):
            audit = []
            result = spectral.compute_event_pre_onset_joint_mi(
                self.x[:, 0], self.x[:, 1], 0, fs=self.fs, pre_sec=5, onset_sec=5,
                n_surrogates=0, time_us=self.t, audit=audit)
        self.assertEqual([row['name'] for row in result], ['valid'])
        expected = spectral.compute_joint_probability(self.x[5000:6000, 0],
                    self.x[5000:6000, 1], bins=spectral.DEFAULT_MI_BINS, binning='quantile')
        self.assertEqual(result[0]['pre_mi'], expected['mutual_information'])
        self.assertEqual(audit[1]['Reason'], 'onset_in_gap')

    def test_invalid_timestamps_and_lengths_rejected(self):
        with self.assertRaisesRegex(ValueError, 'strictly increasing'):
            select_event_windows(np.array([0, 5000, 5000]), self.fs, [], 5)
        with self.assertRaisesRegex(ValueError, 'integer microseconds'):
            select_event_windows(self.t.astype(float), self.fs, [], 5)
        with self.assertRaisesRegex(ValueError, 'lengths differ'):
            spectral.extract_band_envelopes(self.x[:-1, 0], fs=self.fs, time_us=self.t)

    def test_cli_gap_success_and_exclusion_audit_on_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'recording.csv'
            frame = pd.DataFrame({'Time[us]': self.t, 'ch1': self.x[:, 0], 'ch2': self.x[:, 1]})
            source.write_text('File Name,test\nAmp Gain,500,Abs Time Offset[us],0\n'
                              'Channels,1,2\nSample Rate,200,200\n' + frame.to_csv(index=False))
            base = ['spectral_entropy.py', '--csv', str(source), '--fs', '200',
                    '--band-event-mi', '--mi-windows', '5', '--mi-surrogates', '0']
            with patch('sys.argv', base + ['--event-onset', '50', '--out', str(root / 'good')]), \
                 patch.object(spectral, 'plot_band_event_mi'), contextlib.redirect_stdout(io.StringIO()):
                spectral.main()
            result = pd.read_csv(root / 'good/recording_band_event_mi.csv')
            self.assertEqual(len(result), 2)
            audit = pd.read_csv(root / 'good/recording_band_event_mi_events.csv')
            self.assertEqual(audit.Pre_Start_Idx.tolist(), [5000, 5000])
            with patch('sys.argv', base + ['--event-onset', '30', '--out', str(root / 'bad')]), \
                 contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                spectral.main()
            audit = pd.read_csv(root / 'bad/recording_band_event_mi_events.csv')
            self.assertEqual(audit.Reason.tolist(), ['onset_in_gap', 'onset_in_gap'])


if __name__ == '__main__':
    unittest.main()
