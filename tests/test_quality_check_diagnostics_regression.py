import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import quality_check as entry
from lilia.quality_check_io import (load_quality_samples_table,
                                    load_quality_anomalies_table)
from lilia.provenance import file_sha256


def source(path, n=4000, gap=False):
    t = np.arange(n, dtype=np.int64)*2000 + 1700000000000000
    if gap:
        t[n//2+251:] += 7000000
    x = np.random.default_rng(729).normal(0, 6, (n, 4)).astype(np.float32)
    x[:500, 0] += 2050
    frame = pd.DataFrame(x, columns=[f'ch{i+1}' for i in range(4)])
    frame.insert(0, 'Time[us]', t)
    path.write_text('File Name,validation copy\nAmp Gain,500,Abs Time Offset[us],0\n'
                    'Channels,1,2,3,4\nSample Rate,500,500\n'+frame.to_csv(index=False))
    return t, x


class QualityCheckDiagnosticsRegressionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.subject = self.root/'Demo(SN999)'
        self.subject.mkdir()
        self.raw = self.subject/'merged.csv'
        source(self.raw)
        self.out = self.root/'out'
        self.out.mkdir()

    def samples(self, scorer=None):
        with patch.object(entry, 'SEG_SEC', 4.), patch.object(entry, 'SEG_SAMPLES', 2000), \
             patch.object(entry, 'QEEG_WIN_SEC', 2.), \
             patch('matplotlib.figure.Figure.savefig'):
            if scorer is None:
                entry.plot_segments(str(self.raw), 'Demo', 'Subject', str(self.out),
                                    np.random.default_rng(4))
            else:
                with patch.object(entry, 'get_eeg_quality_index_v2_parametric', side_effect=scorer):
                    entry.plot_segments(str(self.raw), 'Demo', 'Subject', str(self.out),
                                        np.random.default_rng(4))
        return self.out/'Demo_Subject_sample_quality.csv'

    def anomalies(self, *, win_sec=1., scorer=None):
        kwargs = dict(name='Demo', info={'dir':'Demo(SN999)','sn':'SN999'},
                      outdir=str(self.out), base_dir=str(self.root), win_sec=win_sec)
        with patch('matplotlib.figure.Figure.savefig'):
            if scorer is None:
                entry.plot_quality_anomaly_report(**kwargs)
            else:
                with patch.object(entry, 'get_eeg_quality_index_v2_parametric', side_effect=scorer):
                    entry.plot_quality_anomaly_report(**kwargs)
        return self.out/'Demo_SN999_quality_anomalies.csv'

    def test_nan_flagged_and_ranked_without_changing_finite_rules(self):
        rows = [dict(qmed=np.nan, clip=0., flat=0., gap=0.),
                dict(qmed=.8, clip=0., flat=0., gap=0.),
                dict(qmed=.3, clip=0., flat=0., gap=0.)]
        actual = entry.flag_anomalies(rows)
        self.assertEqual(actual[0]['reasons'], ['invalid-Q non-finite'])
        self.assertTrue(np.isfinite(actual[0]['severity']))
        self.assertEqual(actual[1]['reasons'], [])
        self.assertEqual(actual[2]['reasons'], ['low-Q 0.30', 'jump -0.50'])

    def test_samples_raw_filtered_source_roundtrip(self):
        path = self.samples()
        frame, meta = load_quality_samples_table(path, self.raw)
        self.assertEqual(len(frame), 4)
        self.assertEqual(frame.quality_stage.tolist(), ['raw','filtered','raw','filtered'])
        self.assertEqual(meta['quality_diagnostics_version'], 1)
        self.assertEqual(len(meta['parameters']['starts']), 2)
        self.assertEqual(meta['quality_audit'][0]['quality_diagnostics']['request']['n_channels'], 4)

    def test_anomalies_source_roundtrip_and_reasons(self):
        source(self.raw, gap=True)
        path = self.anomalies()
        frame, meta = load_quality_anomalies_table(path, self.raw)
        self.assertEqual(len(frame), 8)
        self.assertEqual(set(frame.quality_stage), {'raw'})
        self.assertEqual(meta['parameters']['quality_threshold'], entry.QUALITY_THRESHOLD)
        self.assertTrue(frame.reasons.str.contains('clip').any())
        self.assertTrue(frame.reasons.str.contains('time-gap').any())

    def test_external_scorer_is_unavailable_and_nan_is_visible_in_output(self):
        def old(data, fs, params):
            return {'overall': np.full(data.shape[0], np.nan)}
        path = self.anomalies(scorer=old)
        frame, meta = load_quality_anomalies_table(path, self.raw)
        self.assertEqual(set(frame.quality_diagnostic_state), {'unavailable'})
        self.assertTrue(frame.reasons.str.contains('invalid-Q').all())
        self.assertTrue(np.isfinite(frame.severity).all())
        self.assertEqual(len(meta['quality_audit']), len(frame))
        sample = self.samples(scorer=old)
        frame, _ = load_quality_samples_table(sample, self.raw)
        self.assertEqual(set(frame.quality_diagnostic_state), {'unavailable'})

    def test_rehashed_anomaly_forgery_rejected(self):
        path = self.anomalies()
        sidecar = Path(str(path)+'.meta.json')
        original, original_meta = path.read_bytes(), sidecar.read_text()
        for kind in ('stage','score','component','clip','gap','severity','reasons',
                     'mapping','drop','missing','undeclared','params'):
            with self.subTest(kind=kind):
                path.write_bytes(original)
                meta = json.loads(original_meta)
                frame = pd.read_csv(path, float_precision='round_trip')
                record = meta['quality_audit'][0]
                if kind == 'stage':
                    record['quality_diagnostics']['request']['stage'] = 'filtered'
                    frame.loc[0,'quality_stage'] = 'filtered'
                elif kind == 'score':
                    frame.loc[0,'qmed'] = .123
                elif kind == 'component':
                    record['quality_diagnostics']['result']['detail']['flat'][0] = .123
                elif kind == 'clip':
                    frame.loc[0,'clip'] += .1
                elif kind == 'gap':
                    frame.loc[0,'gap'] += .1
                elif kind == 'severity':
                    frame.loc[0,'severity'] += .1
                elif kind == 'reasons':
                    frame.loc[0,'reasons'] = '[]'
                elif kind == 'mapping':
                    frame.loc[0,'window_center_us'] += 2000
                elif kind == 'drop':
                    frame = frame.iloc[1:]
                    meta['quality_audit'] = meta['quality_audit'][1:]
                elif kind == 'missing':
                    frame = frame.drop(columns=['quality_stage'])
                elif kind == 'undeclared':
                    del meta['quality_diagnostics_version']
                else:
                    meta['parameters']['quality_threshold'] = .1
                frame.to_csv(path, index=False)
                meta['table_sha256'] = file_sha256(path)
                sidecar.write_text(json.dumps(meta))
                with self.assertRaises(ValueError):
                    load_quality_anomalies_table(path, self.raw)

    def test_rehashed_samples_forgery_rejected(self):
        path = self.samples()
        sidecar = Path(str(path)+'.meta.json')
        original, original_meta = path.read_bytes(), sidecar.read_text()
        for kind in ('stage','score','component','source_mapping','filter_params',
                     'overlap','missing','undeclared'):
            with self.subTest(kind=kind):
                path.write_bytes(original)
                meta = json.loads(original_meta)
                frame = pd.read_csv(path, float_precision='round_trip')
                record = meta['quality_audit'][0]
                if kind == 'stage':
                    record['quality_diagnostics']['request']['stage'] = 'filtered'
                    frame.loc[0,'quality_stage'] = 'filtered'
                elif kind == 'score':
                    frame.loc[0,'quality_ch1'] = .123
                elif kind == 'component':
                    record['quality_diagnostics']['result']['detail']['flat'][0] = .123
                elif kind == 'source_mapping':
                    frame.loc[0,'window_start_us'] += 2000
                elif kind == 'filter_params':
                    meta['parameters']['bp_high'] = 30.
                elif kind == 'overlap':
                    meta['parameters']['starts'][1] = meta['parameters']['starts'][0]
                elif kind == 'missing':
                    frame = frame.drop(columns=['quality_stage'])
                else:
                    del meta['quality_diagnostics_version']
                frame.to_csv(path, index=False)
                meta['table_sha256'] = file_sha256(path)
                sidecar.write_text(json.dumps(meta))
                with self.assertRaises(ValueError):
                    load_quality_samples_table(path, self.raw)

    def test_short_source_and_nonfinite_rejection_keep_entry_policy(self):
        source(self.raw, n=800)
        path = self.anomalies(win_sec=5.)
        frame, _ = load_quality_anomalies_table(path, self.raw)
        self.assertEqual(len(frame), 0)
        with patch.object(entry, 'SEG_SAMPLES', 1000):
            with patch('matplotlib.figure.Figure.savefig'):
                self.assertIsNone(entry.plot_segments(str(self.raw), 'Demo','Short',
                                   str(self.out), np.random.default_rng(4)))
        self.assertFalse((self.out/'Demo_Short_sample_quality.csv').exists())
        text = self.raw.read_text().splitlines()
        text[10] = text[10].split(',')[0]+',nan,0,0,0'
        self.raw.write_text('\n'.join(text)+'\n')
        with self.assertRaisesRegex(ValueError, 'Non-finite EEG samples'):
            self.anomalies()

    def test_scorer_error_propagates_without_table(self):
        def fail(*args, **kwargs):
            raise RuntimeError('sentinel')
        with self.assertRaisesRegex(RuntimeError,'sentinel'):
            self.anomalies(scorer=fail)
        self.assertFalse((self.out/'Demo_SN999_quality_anomalies.csv').exists())

    def test_anomaly_raw_panels_break_at_gap_and_empty_axis_is_bounded(self):
        source(self.raw, gap=True)
        with patch('matplotlib.figure.Figure.savefig'), patch.object(entry.plt, 'close'):
            entry.plot_quality_anomaly_report('Demo',
                {'dir':'Demo(SN999)','sn':'SN999'}, str(self.out),
                base_dir=str(self.root), win_sec=1.)
            fig = entry.plt.gcf()
            self.assertTrue(any(np.isnan(line.get_ydata()).any()
                                for ax in fig.axes[1:] for line in ax.lines[:4]))
            entry.plot_quality_anomaly_report('Demo',
                {'dir':'Demo(SN999)','sn':'SN999'}, str(self.out),
                base_dir=str(self.root), win_sec=20.)
            empty = entry.plt.gcf().axes[0]
            self.assertLess(empty.get_xlim()[1]-empty.get_xlim()[0], 1/1440)
        entry.plt.close('all')


if __name__ == '__main__':
    unittest.main()
