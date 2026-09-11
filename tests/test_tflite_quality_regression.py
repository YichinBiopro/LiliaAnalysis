"""Keep baseline selection stable while distinguishing three quality stages."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

import plot_tflite_summary as entry
from lilia import quality
from lilia.entropy_io import config_id
from lilia.provenance import file_sha256
from lilia.tflite import build_tflite_timeline
from lilia.tflite_baseline import score_baseline_windows
from lilia.tflite_io import load_tflite_table


def spectrum_fallback(data, fs, params):
    with patch.object(quality.sp_signal, 'welch', side_effect=RuntimeError('injected spectrum failure')):
        return quality.get_eeg_quality_index_v2_parametric(data, fs=fs, params=params)


class TFLiteQualityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.t = np.arange(16000, dtype=np.int64) * 2000
        self.raw = np.random.default_rng(18102).normal(size=(len(self.t), 4)).astype(np.float32)
        self.params = dict(flat_weight=0., spectrum_weight=1., kurtosis_weight=0., corr_weight=0.)

    def run_plot(self, scorer=None):
        folder = self.root / 'Demo'
        folder.mkdir()
        source = folder / 'merged.csv'
        with source.open('w') as handle:
            handle.write('header\n' * 4)
            pd.DataFrame({'Time[us]': self.t, **{f'ch{i+1}': self.raw[:, i] for i in range(4)}}).to_csv(handle, index=False)
        figures = []
        with patch('lilia.tflite.apply_tflite_windowed', side_effect=lambda x, *_: x[:, :2]), \
                patch.object(entry, 'QUALITY_PARAMS', self.params), \
                patch.object(entry, 'get_eeg_quality_index_v2_parametric',
                             side_effect=scorer or quality.get_eeg_quality_index_v2_parametric), \
                patch.object(entry, 'EVENTS', []), patch.object(entry, 'CONE_STAGES', []), \
                patch.object(Figure, 'savefig', lambda fig, *_a, **_k: figures.append(fig)), \
                contextlib.redirect_stdout(io.StringIO()):
            entry.plot_subject_tflite_summary('Demo', {'dir': 'Demo', 'sn': 'Q'},
                str(self.root / 'out'), base_dir=str(self.root), quality_ratio=.5 if scorer else 0.)
        table = self.root / 'out/Demo_Q_tflite_metrics.csv'
        return source, table, figures

    def test_baseline_records_filtered_stage_and_preserves_short_circuit(self):
        t, raw = self.t[:3000], self.raw[:3000].copy()
        filtered = raw.copy()
        raw[500:1000] = 2048
        filtered[1000] = np.nan
        calls = []

        def scorer(data, fs, params):
            calls.append(data.shape)
            return spectrum_fallback(data, fs, params)

        rows = score_baseline_windows(t, filtered, raw, build_tflite_timeline(t),
                                      scorer=scorer, quality_params=self.params)
        self.assertEqual([r['reason'] for r in rows], ['raw_saturation', 'nonfinite_signal', ''])
        self.assertEqual(len(calls), 3)
        self.assertEqual([len(r['quality_subepochs']) for r in rows], [2, 1, 2])
        scored = rows[2]['quality_subepochs'][0]
        self.assertEqual(scored['quality_diagnostics']['request']['stage'], 'filtered')
        self.assertEqual(scored['quality_diagnostics']['state'], 'invalid')
        self.assertEqual(scored['quality_median'], .5)
        self.assertTrue(rows[2]['eligible'])
        self.assertEqual(rows[0]['quality_min'], .5)

    def test_short_subepoch_is_not_scored_and_scorer_errors_still_propagate(self):
        t, raw = self.t[:1000], self.raw[:1000]
        timeline = build_tflite_timeline(t)
        with patch.object(quality, 'get_eeg_quality_index_v2_parametric') as scorer:
            rows = score_baseline_windows(t, raw, raw, timeline, scorer=scorer,
                                           quality_params=self.params, epoch_sec=.01)
            scorer.assert_not_called()
        sub = rows[0]['quality_subepochs'][0]
        self.assertEqual(sub['quality_diagnostics']['state'], 'not_scored')
        self.assertEqual(rows[0]['reason'], 'insufficient_raw_subepoch')
        with self.assertRaisesRegex(RuntimeError, 'scorer failed'):
            score_baseline_windows(t, raw, raw, timeline, scorer=lambda *_a, **_k: (_ for _ in ()).throw(
                RuntimeError('scorer failed')), quality_params=self.params)

    def test_summary_three_stages_fallback_markers_and_source_reader(self):
        source, table, figures = self.run_plot(spectrum_fallback)
        frame, meta = load_tflite_table(table, source, entry.TFLITE_MODEL_PATH)
        analysis = meta['quality_analysis']
        self.assertTrue(frame.quality_valid.all())
        np.testing.assert_array_equal(frame.quality_after, .5)
        for branch, stage in [('before', 'filtered_resampled'), ('after', 'model_output')]:
            row = analysis[branch][0]
            self.assertEqual(row['quality_diagnostics']['request']['stage'], stage)
            self.assertEqual(row['quality_diagnostics']['state'], 'invalid')
        self.assertEqual(analysis['baseline_catalog'][0]['quality_subepochs'][0][
            'quality_diagnostics']['request']['stage'], 'filtered')
        ax = figures[0].axes[0]
        self.assertIn('spectrum [custom]', ax.get_title(loc='left'))
        self.assertEqual(len(ax.collections), 2)
        self.assertEqual(analysis['baselines'][0]['n_selected'], 5)

    def test_rehashed_diagnostic_stage_preset_coverage_and_baseline_tampering_fail(self):
        source, table, _ = self.run_plot()
        _, original = load_tflite_table(table, source, entry.TFLITE_MODEL_PATH)
        sidecar = Path(str(table) + '.meta.json')
        for case in ('stage', 'preset', 'coverage', 'baseline_index', 'baseline_missing', 'selection'):
            with self.subTest(case=case):
                meta = copy.deepcopy(original)
                analysis = meta['quality_analysis']
                if case in ('stage', 'preset'):
                    context = analysis['after'][0]['quality_diagnostics']['result']['context']
                    context[case] = 'raw' if case == 'stage' else 'default'
                    context['config_id'] = config_id({k: context[k] for k in ('profile', 'preset', 'stage', 'fs', 'parameters')})
                elif case == 'coverage':
                    analysis['before'].pop()
                elif case == 'baseline_index':
                    analysis['baseline_catalog'][0]['quality_subepochs'][0]['raw_start_idx'] += 1
                elif case == 'baseline_missing':
                    analysis['baseline_catalog'][0]['quality_subepochs'].pop()
                else:
                    analysis['baselines'][0]['selected'][0] = analysis['baseline_catalog'][-1]
                sidecar.write_text(json.dumps(meta, allow_nan=False))
                with self.assertRaises(ValueError):
                    load_tflite_table(table, source, entry.TFLITE_MODEL_PATH)

    def test_csv_score_tampering_rejected_after_rehash_and_legacy_tables_readable(self):
        source, table, _ = self.run_plot()
        frame, meta = load_tflite_table(table, source, entry.TFLITE_MODEL_PATH)
        sidecar = Path(str(table) + '.meta.json')
        changed = frame.copy()
        changed.loc[0, 'quality_before'] = .123
        changed.to_csv(table, index=False)
        meta['table_sha256'] = file_sha256(table)
        sidecar.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, 'median'):
            load_tflite_table(table, source)
        frame = frame.drop(columns=[c for c in frame if c.startswith(('before_quality_', 'after_quality_'))])
        frame.to_csv(table, index=False)
        meta.pop('quality_diagnostics_version')
        meta.pop('quality_analysis')
        meta['table_sha256'] = file_sha256(table)
        sidecar.write_text(json.dumps(meta))
        loaded, _ = load_tflite_table(table, source)
        self.assertEqual(len(loaded), len(frame))

    def test_rehashed_channel_mapping_and_reduction_are_rejected(self):
        source, table, _ = self.run_plot()
        original_frame, original = load_tflite_table(table, source, entry.TFLITE_MODEL_PATH)
        for key, value in [('channels', [3, 4]), ('input_channels', [4, 3, 2, 1]),
                           ('quality_reduction', 'maximum')]:
            with self.subTest(key=key):
                meta = copy.deepcopy(original)
                frame = original_frame.copy()
                meta['parameters'][key] = value
                meta['config_id'] = config_id(meta['parameters'])
                frame['config_id'] = meta['config_id']
                frame.to_csv(table, index=False)
                meta['table_sha256'] = file_sha256(table)
                Path(str(table) + '.meta.json').write_text(json.dumps(meta))
                with self.assertRaisesRegex(ValueError, 'channel mapping or reduction'):
                    load_tflite_table(table, source, entry.TFLITE_MODEL_PATH)

    def test_legacy_helper_preserves_tuple_and_sampler_selection(self):
        with patch.object(entry, 'QUALITY_PARAMS', self.params), \
                patch.object(entry, 'get_eeg_quality_index_v2_parametric', side_effect=spectrum_fallback):
            old = entry.compute_quality_windowed_fs(self.t, self.raw, 500)
            audited = entry.compute_quality_windowed_fs(self.t, self.raw, 500, return_audit=True)
            np.testing.assert_array_equal(old[1], audited[1])
            self.assertEqual(old[0], audited[0])
            self.assertEqual(audited[2][0]['quality_diagnostics']['request']['stage'], 'unspecified')
            sampled, meta = entry.build_session_baseline(self.t, self.raw, required_sec=4)
        selected = np.sort(np.random.default_rng(42).choice(32, 4, replace=False))
        self.assertEqual(meta['selected_starts_us'], (selected * 1000000).tolist())
        np.testing.assert_array_equal(sampled, np.concatenate([self.raw[i*500:(i+1)*500] for i in selected]))
        self.assertTrue(all(r['accepted'] for r in meta['quality_audit']))
        self.assertTrue(all(r['quality_diagnostics']['state'] == 'invalid' for r in meta['quality_audit']))


if __name__ == '__main__':
    unittest.main()
