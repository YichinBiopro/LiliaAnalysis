import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

import spectral_entropy as entropy
from lilia import quality
from lilia.entropy_io import config_id, load_entropy_table, load_joint_mi_table
from lilia.provenance import file_sha256
from lilia.quality_audit import diagnostic_columns
from lilia.state_entropy_io import load_state_entropy_table


class EntropyQualityDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'raw.csv'
        self.t = np.r_[np.arange(2000)*2000, 20000000+np.arange(2000)*2000]
        self.x = np.random.default_rng(812).normal(size=(len(self.t), 2)).astype(np.float32)
        self.write_source()
        self.params = {key+'_weight': float(key == 'spectrum') for key in ('flat', 'spectrum', 'kurtosis', 'corr')}

    def write_source(self):
        frame = pd.DataFrame({'Time[us]': self.t, 'ch1': self.x[:, 0], 'ch2': self.x[:, 1]})
        self.source.write_text('File Name,test\nAmp Gain,500,Abs Time Offset[us],0\n'
                               'Channels,1,2\nSample Rate,500,500\n' + frame.to_csv(index=False))

    def cli(self, mode, *extra, scorer=None):
        args = ['spectral_entropy.py', '--csv', str(self.source), '--out', str(self.root / mode),
                '--win', '2', '--step', '2']
        if mode == 'joint':
            args += ['--joint-mi', '--mi-surrogates', '0']
        elif mode == 'state':
            args += ['--baseline', '0', '4', '--event', '20', '24', '--clean']
        args += list(extra)
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch('sys.argv', args))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(patch.object(Figure, 'savefig'))
            stack.enter_context(patch.object(entropy, '_QUALITY_PARAMS', self.params))
            if scorer is not None:
                stack.enter_context(patch.object(entropy, '_eeg_quality_v2', side_effect=scorer))
            entropy.main()
        pattern = '*timeseries.csv' if mode == 'joint' else '*entropy_ch1.csv'
        return next((self.root / mode).glob(pattern))

    def fallback(self, data, **kwargs):
        with patch.object(quality.sp_signal, 'welch', side_effect=RuntimeError('forced spectrum failure')):
            return quality.get_eeg_quality_index_v2_parametric(data, **kwargs)

    def test_finite_fallback_remains_accepted_and_reader_preserves_reason_and_stage(self):
        for mode, reader in [('joint', load_joint_mi_table), ('entropy', load_entropy_table),
                             ('state', load_state_entropy_table)]:
            with self.subTest(mode=mode):
                table = self.cli(mode, scorer=self.fallback)
                frame, meta = reader(table, self.source)
                rows = meta['states']['baseline']['windows'] if mode == 'state' else meta['quality_audit']
                for row in rows:
                    record = row['quality_diagnostics']
                    self.assertEqual(record['state'], 'invalid')
                    self.assertEqual(record['request']['stage'], 'filtered')
                    self.assertEqual(record['result']['overall'], [.5, .5])
                    self.assertIn('spectrum:exception:RuntimeError', record['reasons'])
                if mode == 'state':
                    self.assertEqual(frame.n_windows.tolist(), [2, 2])
                    self.assertTrue(all(r['status'] == 'accepted' for r in rows))
                else:
                    self.assertTrue(frame.quality_valid.all())
                    self.assertEqual(frame.quality.tolist(), [.5]*4)

    def test_rehashed_diagnostic_csv_stage_components_and_score_are_rejected(self):
        table = self.cli('joint')
        frame, original = load_joint_mi_table(table, self.source)
        sidecar = Path(str(table) + '.meta.json')
        for change in ('stage', 'component', 'score', 'column', 'missing'):
            with self.subTest(change=change):
                meta, changed = copy.deepcopy(original), frame.copy()
                record = meta['quality_audit'][0]['quality_diagnostics']
                if change == 'stage':
                    record['request']['stage'] = 'raw'
                elif change == 'component':
                    record['result']['detail']['spectrum'][0] = .123
                elif change == 'score':
                    changed.loc[0, 'quality'] = .95
                    changed.loc[0, 'quality_valid'] = True
                elif change == 'column':
                    changed.loc[0, 'quality_diagnostic_state'] = 'unavailable'
                else:
                    del meta['quality_audit'][0]['quality_diagnostics']
                if change not in ('column', 'missing'):
                    for key, value in diagnostic_columns(meta['quality_audit']).items():
                        changed[key] = value
                changed.to_csv(table, index=False)
                meta['table_sha256'] = file_sha256(table)
                sidecar.write_text(json.dumps(meta))
                with self.assertRaises(ValueError):
                    load_joint_mi_table(table, self.source)

    def test_raw_and_disabled_and_external_scorer_are_distinct(self):
        table = self.cli('entropy', '--no-bandpass')
        frame, meta = load_entropy_table(table, self.source)
        self.assertEqual(frame.quality_stage.unique().tolist(), ['raw'])
        self.assertTrue(all(r['quality_diagnostics']['result'] is not None for r in meta['quality_audit']))
        table = self.cli('entropy', '--no-quality-mask')
        frame, meta = load_entropy_table(table, self.source)
        self.assertEqual(frame.quality_diagnostic_state.unique().tolist(), ['not_scored'])
        self.assertTrue(all(r['quality_diagnostics']['reasons'] == ['quality_disabled'] for r in meta['quality_audit']))
        table = self.cli('entropy', scorer=lambda *a, **kw: {'overall': [.9, .9]})
        frame, _ = load_entropy_table(table, self.source)
        self.assertEqual(frame.quality_diagnostic_state.unique().tolist(), ['unavailable'])

    def test_state_precheck_and_exception_diagnostics_survive_all_excluded_failure(self):
        self.x[:1000] = 2047
        self.write_source()
        def fail(*args, **kwargs):
            raise RuntimeError('external scorer failed')
        with self.assertRaisesRegex(ValueError, 'audit saved'):
            self.cli('state', '--no-bandpass', scorer=fail)
        table = self.root / 'state/raw_baseline_event_entropy_ch1.csv'
        frame, meta = load_state_entropy_table(table, self.source)
        self.assertEqual(frame.n_windows.tolist(), [0, 0])
        rows = meta['states']['baseline']['windows']
        self.assertEqual(rows[0]['quality_diagnostics']['reasons'], ['raw_saturation'])
        self.assertEqual(rows[1]['quality_diagnostics']['reasons'], ['scorer_exception:RuntimeError'])
        rows[0]['quality_diagnostics']['request']['stage'] = 'filtered'
        meta['audit_id'] = config_id(meta['states'])
        Path(str(table) + '.meta.json').write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, 'source request'):
            load_state_entropy_table(table, self.source)

    def test_half_second_and_nonfinite_helper_preserve_scores_and_expose_invalidity(self):
        signal = np.ones((250, 2))
        signal[10, 0] = np.nan
        rows = []
        with patch.object(entropy, '_QUALITY_PARAMS', self.params):
            expected = entropy.compute_quality_windowed_aligned(signal, win_sec=.5)
            actual = entropy.compute_quality_windowed_aligned(signal, win_sec=.5,
                        quality_audit=rows, quality_stage='raw')
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(rows[0]['quality_diagnostics']['state'], 'invalid')
        self.assertIn('nonfinite_input', rows[0]['quality_diagnostics']['reasons'])

    def test_empty_mi_plot_keeps_real_time_extent_and_message(self):
        figures = []
        def capture(fig, path, *args, **kwargs):
            if str(path).endswith('_timeseries.png'):
                figures.append((fig.axes[0].get_xlim(), [t.get_text() for t in fig.axes[0].texts]))
        # cli replaces savefig; intercept its plot factory to inspect after close instead.
        original = entropy.plt.close
        def close(fig=None):
            if isinstance(fig, Figure) and fig.axes and 'Zero-lag' in fig.axes[0].get_title():
                capture(fig, '_timeseries.png')
            original(fig)
        with patch.object(entropy.plt, 'close', side_effect=close):
            self.cli('joint', scorer=lambda *a, **kw: {'overall': [0., 0.]})
        self.assertEqual(figures[0][0], (1., 23.))
        self.assertIn('No usable MI windows (quality/signal excluded)', figures[0][1])


if __name__ == '__main__':
    unittest.main()
