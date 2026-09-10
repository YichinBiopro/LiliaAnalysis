"""Source-aware Jenqwei tables, CLI failure audits and segmented plot contracts."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from matplotlib.figure import Figure
import numpy as np

import analyze_jenqwei_pipeline as entry
from lilia.jenqwei_io import load_signal_table, write_signal_tables
from lilia.jenqwei_plot import compute_panels, plot_result, plot_metadata
from lilia.provenance import file_sha256
from tests.test_jenqwei_pipeline_regression import identity


class JenqweiOutputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source.csv'
        self.t = 9000000000000001 + np.r_[np.arange(10) * 2000,
            10000000 + np.arange(1503) * 2000, 20000000 + np.arange(2001) * 2000,
            30000000 + np.arange(997) * 2000]
        self.raw = np.random.default_rng(166).normal(size=(len(self.t), 4)).astype(np.float32)
        self.recording()

    def recording(self):
        with self.source.open('w') as handle:
            handle.write('File Name,test\nAmp Gain,500\nChannels,1,2,3,4\nSample Rate,500,500,500,500\n'
                         'Time[us],ch1,ch2,ch3,ch4\n')
            for t, row in zip(self.t, self.raw):
                handle.write(','.join([str(int(t)), *map(str, row)]) + '\n')

    def process(self):
        with patch.object(entry, 'apply_tflite_windowed', side_effect=identity):
            return entry.process_segments(self.t, self.raw)

    def tables(self):
        result = self.process()
        write_signal_tables(self.root, self.source, result, entry.TFLITE_MODEL_PATH, {})
        return result

    def load(self, branch):
        return load_signal_table(self.root / f'source_{branch}.csv', self.source, model_path=entry.TFLITE_MODEL_PATH)

    def test_roundtrip_preserves_distinct_indexes_tails_epoch_and_complete_mapping(self):
        result = self.tables()
        before, bm = self.load('before')
        after, am = self.load('after')
        self.assertEqual(len(before), 1403)
        self.assertEqual(len(after), 1200)
        np.testing.assert_array_equal(before['Time[us]'], result['time_us_200'])
        np.testing.assert_array_equal(after['Time[us]'], result['tfl_time_us_200'])
        np.testing.assert_array_equal(before[['ch1','ch2','ch3','ch4']], result['pre_data_200'])
        np.testing.assert_array_equal(after[['ch1','ch2']], result['tfl_data_200'])
        np.testing.assert_array_equal(after['before_idx'], np.r_[0:400, 602:1402])
        self.assertEqual(am['index_space'], 'packed_after')
        self.assertEqual(bm['segments'][0]['status'], 'excluded')
        self.assertEqual(am['quality_state'], 'disabled')

    def test_table_tampering_is_rejected_even_when_table_hash_is_refreshed(self):
        self.tables()
        path = self.root / 'source_after.csv'
        meta_path = Path(str(path) + '.meta.json')
        original_bytes = path.read_bytes()
        original, metadata = original_bytes.decode(), meta_path.read_text()
        for column, value in [('Time[us]', '9000000000000002'), ('before_idx', '602'),
                              ('segment_id', '0'), ('raw_fractional_idx', '9.5'), ('ch1', 'nan')]:
            with self.subTest(column=column):
                rows = original.splitlines()
                columns, row = rows[0].split(','), rows[1].split(',')
                row[columns.index(column)] = value
                rows[1] = ','.join(row)
                path.write_text('\n'.join(rows) + '\n')
                meta = json.loads(metadata)
                meta['table_sha256'] = file_sha256(path)
                meta_path.write_text(json.dumps(meta))
                with self.assertRaises(ValueError):
                    self.load('after')
        path.write_bytes(original_bytes)
        meta_path.write_text(metadata)
        self.load('after')

    def test_truncated_table_rebuilds_full_filter_context_and_used_prefix(self):
        with patch.object(entry,'apply_tflite_windowed',side_effect=identity):
            result=entry.process_segments(self.t,self.raw,max_samples=3006)
        write_signal_tables(self.root,self.source,result,entry.TFLITE_MODEL_PATH,{})
        after,meta=self.load('after')
        np.testing.assert_array_equal(after['Time[us]'],result['tfl_time_us_200'])
        self.assertEqual(meta['used_source_samples'],3006)
        self.assertEqual(meta['segments'][2]['filter_context_end_idx'],3514)
        self.assertEqual(meta['segments'][2]['raw_end_idx'],3006)

    def test_metadata_source_model_and_header_tampering_fail(self):
        self.tables()
        path = self.root / 'source_before.csv.meta.json'
        original = path.read_text()
        for key, value in [('quality_state', 'valid'), ('index_space','packed_after'),
                           ('source_id', 'wrong'), ('full_source_samples', 1), ('inference', {})]:
            meta = json.loads(original)
            meta[key] = value
            path.write_text(json.dumps(meta))
            with self.assertRaises(ValueError):
                self.load('before')
        path.write_text(original)
        wrong_model = self.root / 'wrong.tflite'
        wrong_model.write_bytes(b'wrong model')
        with self.assertRaisesRegex(ValueError, 'model or configuration'):
            load_signal_table(self.root / 'source_before.csv', self.source, model_path=wrong_model)
        raw = self.source.read_text()
        self.source.write_text(raw.replace('Amp Gain,500', 'Amp Gain,1'))
        with self.assertRaisesRegex(ValueError, 'source fingerprint'):
            self.load('before')

    def test_multiple_display_channels_share_one_processing_result(self):
        def save(fig, path, **kwargs):
            path.write(b'test figure placeholder; real rendering validated by CLI')
        with patch.object(entry, 'apply_tflite_windowed', side_effect=identity) as model, \
                patch.object(entry, 'process_segments', wraps=entry.process_segments) as process, \
                patch.object(Figure, 'savefig', save):
            audit = entry.analyze_recording(self.source, self.root / 'run', [0,1,2,3])
        process.assert_called_once()
        self.assertEqual(model.call_count, 2)  # one call per retained segment, not per plotted channel
        self.assertTrue(audit['tables_verified'])
        self.assertEqual(audit['plotting_status'], 'complete')
        self.assertEqual(len(audit['artifacts']), 8)
        self.assertIsNone(audit['plot_timeline']['channels'][2]['after_source_channel'])
        self.assertEqual(audit['before_samples'], 1403)
        self.assertEqual(audit['after_samples'], 1200)

    def test_model_and_plot_failures_save_context_without_reporting_success(self):
        with patch.object(entry, 'apply_tflite_windowed', side_effect=[np.zeros((400,2)), RuntimeError('model failed')]):
            with self.assertRaisesRegex(ValueError, 'segment 2'):
                entry.analyze_recording(self.source, self.root / 'model')
        audit = json.loads((self.root / 'model/source_analysis_audit.json').read_text())
        self.assertEqual(audit['status'], 'failed')
        self.assertEqual(audit['stage'], 'process_segments')
        self.assertEqual(audit['artifacts'], [])
        with patch.object(entry, 'apply_tflite_windowed', side_effect=identity), \
                patch.object(entry, 'plot_result', side_effect=OSError('disk full')):
            with self.assertRaisesRegex(OSError, 'disk full'):
                entry.analyze_recording(self.source, self.root / 'plot')
        audit = json.loads((self.root / 'plot/source_analysis_audit.json').read_text())
        self.assertTrue(audit['tables_verified'])
        self.assertEqual(audit['status'], 'failed')
        self.assertEqual(audit['plotting_status'], 'in_progress')
        self.assertEqual(len(audit['artifacts']), 4)

    def test_collision_does_not_overwrite_prior_outputs_or_audit(self):
        out = self.root / 'collision'
        out.mkdir()
        previous = {out / name: b'previous research' for name in ('source_before.csv','source_analysis_audit.json')}
        for path, data in previous.items():
            path.write_bytes(data)
        with self.assertRaises(FileExistsError):
            entry.analyze_recording(self.source, out)
        for path, data in previous.items():
            self.assertEqual(path.read_bytes(), data)
        failure, = out.glob('source_analysis_audit_*.json')
        self.assertEqual(json.loads(failure.read_text())['stage'], 'preflight')

    def test_missing_invalid_short_and_polluted_short_fail_with_audit(self):
        cases = ('missing', 'channels', 'display', 'short', 'polluted', 'missing_model')
        for case in cases:
            with self.subTest(case=case):
                source = self.root/'missing.csv' if case=='missing' else self.source
                channels = [4] if case == 'channels' else [0,1]
                max_sec = float('nan') if case == 'display' else 60.
                if case == 'short':
                    self.t, self.raw = np.arange(997)*2000, np.ones((997,4))
                    self.recording()
                if case == 'polluted':
                    self.raw[0,0] = np.inf
                    self.recording()
                model_path = str(self.root/'missing.tflite') if case == 'missing_model' else entry.TFLITE_MODEL_PATH
                with patch.object(entry, 'TFLITE_MODEL_PATH', model_path), \
                        patch.object(entry, 'apply_tflite_windowed') as model:
                    with self.assertRaises((ValueError, FileNotFoundError)):
                        entry.analyze_recording(source, self.root/case, channels, max_sec)
                    model.assert_not_called()
                audit = json.loads((self.root/case/f'{source.stem}_analysis_audit.json').read_text())
                self.assertEqual(audit['status'],'failed')
                self.assertFalse(audit['tables_verified'])

    def test_main_processes_each_source_once_and_returns_failure_when_any_source_fails(self):
        args = argparse.Namespace(csv=['first.csv', 'second.csv'], channels=[0,1], max_sec=60., outdir=str(self.root))
        with patch('argparse.ArgumentParser.parse_args', return_value=args), \
                patch.object(entry, 'analyze_recording', side_effect=[{}, ValueError('failed')]) as run:
            self.assertEqual(entry.main(), 1)
        self.assertEqual(run.call_count, 2)


class JenqweiPlotTests(unittest.TestCase):
    def test_continuous_panels_match_all_frozen_psd_and_stft_arrays(self):
        fixture = Path(__file__).parent/'fixtures'
        for path in sorted(fixture.glob('jenqwei_*_reference.npz')):
            with self.subTest(case=path.stem), np.load(path) as ref:
                # Signal baseline is independently tested with the real model;
                # this check focuses on spectra produced by the new plot path.
                meta = json.loads(path.with_suffix('.json').read_text())
                with patch.object(entry, 'apply_tflite_windowed', return_value=ref['tfl_data_200']):
                    result = entry.process_segments(ref['source_time_us'], ref['source_raw'], meta['max_samples'])
                for ch in range(4):
                    panels = compute_panels(result, ch)
                    self.assertEqual(len(panels), 2 if ch<2 else 1)
                    for panel in panels:
                        prefix = f'ch{ch}_{panel["branch"]}'
                        np.testing.assert_array_equal(panel['psd'], ref[prefix+'_psd'])
                        np.testing.assert_array_equal(panel['psd_frequency'], ref[prefix+'_psd_f'])
                        f,t,db = panel['stft']
                        np.testing.assert_array_equal(f,ref[prefix+'_stft_f'])
                        np.testing.assert_allclose(t,ref[prefix+'_stft_t'],rtol=0,atol=1e-12)
                        np.testing.assert_array_equal(db,ref[prefix+'_stft_db'])

    def result(self):
        t = 9000000000000001 + np.r_[np.arange(10)*2000,10000000+np.arange(1503)*2000,
                                    13012000+np.arange(2001)*2000,20000000+np.arange(997)*2000]
        raw = np.random.default_rng(167).normal(size=(len(t),4)).astype(np.float32)
        with patch.object(entry,'apply_tflite_windowed',side_effect=identity):
            return entry.process_segments(t,raw)

    def test_psd_and_stft_are_independent_per_segment_and_exclude_gap_from_spectrum(self):
        result = self.result()
        original = compute_panels(result,0)
        self.assertEqual(len(original),4)
        result['pre_data_200'][:602] *= 1000
        changed = compute_panels(result,0)
        for old,new in zip(original[2:],changed[2:]):
            np.testing.assert_array_equal(old['psd'],new['psd'])
            np.testing.assert_array_equal(old['stft'][2],new['stft'][2])
        meta = plot_metadata(result,[0],60.)
        self.assertEqual(meta['xlim_s'],[0.,21.994])
        self.assertEqual(meta['after_tails'][0]['start_s'],12.)
        self.assertEqual(meta['after_tails'][0]['end_s'],13.006)
        self.assertEqual(meta['missing_spans'][1],{'start_s':13.006,'end_s':13.012})

    def test_rendered_source_channel3_has_no_fabricated_after_and_short_views_are_explicit(self):
        result = self.result()
        figures = []
        def inspect(fig, path, **kwargs):
            fig.canvas.draw()
            figures.append(fig.number)
            self.assertIn('source Ch3',fig._suptitle.get_text())
            self.assertIn('Before only',fig._suptitle.get_text())
            after = next(ax for ax in fig.axes if ax.get_title()=='After STFT')
            self.assertEqual(len(after.collections),0)
            self.assertIn('No model output',after.texts[0].get_text())
            td = next(ax for ax in fig.axes if ax.get_title().startswith('Time domain'))
            self.assertEqual(len([line for line in td.lines if line.get_label().startswith('Before')]),2)
            self.assertFalse(any(line.get_label().startswith('After') for line in td.lines))
        with tempfile.TemporaryDirectory() as tmp, patch.object(Figure,'savefig',inspect):
            plot_result(result,tmp,'source',2)
        self.assertFalse(any(entry.matplotlib.pyplot.fignum_exists(n) for n in figures))
        panels = compute_panels(result,0,10.5)
        self.assertTrue(all(p['stft'] is None for p in panels))
        self.assertEqual(panels[0]['stft_status'],'too_short_for_128_overlap')
        self.assertEqual(panels[2]['stft_status'],'outside_display')

    def test_jittered_stft_centers_follow_source_time_without_changing_spectrum(self):
        t=9000000000000001+np.arange(1503)*2000
        t[500:]+=1000
        raw=np.random.default_rng(168).normal(size=(1503,4)).astype(np.float32)
        with patch.object(entry,'apply_tflite_windowed',side_effect=identity):
            result=entry.process_segments(t,raw)
        from scipy.signal import stft
        for panel in compute_panels(result,0,60.):
            f,local,z=stft(panel['values'].astype(np.float64),fs=200,nperseg=256,noverlap=128,window='hann')
            np.testing.assert_array_equal(panel['stft'][2],20*np.log10(np.abs(z[f<=50])+1e-8))
            indices=np.rint(local*200).astype(int)
            inside=indices<len(panel['elapsed'])
            # Elapsed coordinates are floats; integer source timestamps remain exact.
            np.testing.assert_allclose(panel['stft'][1][inside],panel['elapsed'][indices[inside]],rtol=0,atol=1e-12)

    def test_real_save_failure_closes_figure(self):
        from matplotlib import pyplot as plt
        previous=plt.get_fignums()
        with tempfile.TemporaryDirectory() as tmp, patch.object(Figure,'savefig',side_effect=OSError('disk full')):
            with self.assertRaisesRegex(OSError,'disk full'):
                plot_result(self.result(),tmp,'failure',0)
        self.assertEqual(plt.get_fignums(),previous)

    def test_rendered_td_breaks_and_stft_clips_before_after_tails_separately(self):
        result = self.result()
        def inspect(fig,path,**kwargs):
            fig.canvas.draw()
            td=next(ax for ax in fig.axes if ax.get_title().startswith('Time domain'))
            signals=[line for line in td.lines if line.get_label().startswith(('Before','After'))]
            self.assertEqual([len(line.get_xdata()) for line in signals],[602,400,801,800])
            self.assertEqual(signals[2].get_xdata()[0],13.012)
            before=next(ax for ax in fig.axes if ax.get_title()=='Before STFT')
            after=next(ax for ax in fig.axes if ax.get_title()=='After STFT')
            self.assertEqual(len(before.collections),2)
            for ax,end in ((before,13.006),(after,12.)):
                clip=ax.collections[0].get_clip_box().transformed(ax.transData.inverted())
                np.testing.assert_allclose(clip.extents[[0,2]],[10.,end],atol=1e-12,rtol=0)
            np.testing.assert_allclose(before.get_position().width,after.get_position().width,atol=1e-12)
        with tempfile.TemporaryDirectory() as tmp, patch.object(Figure,'savefig',inspect):
            plot_result(result,tmp,'source',0)


if __name__ == '__main__':
    unittest.main()
