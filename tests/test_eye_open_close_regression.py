import json
import argparse
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from data_analysis import downsample_data

import process_lilia_eye_open_close as eye
from lilia.neural import model_provenance
from lilia.eye_io import load_signal_table, signal_parameters, write_signal_table
from lilia.provenance import file_sha256
from lilia.windowing import continuous_slices

FIXTURES = Path(__file__).parent / 'fixtures'


class Identity(torch.nn.Module):
    def forward(self, x):
        return x[:, :2]


class EyeProcessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_real_and_synthetic_models_match_frozen_continuous_baseline(self):
        # Deliberately fail if the required real checkpoint is unavailable.
        model = eye.load_model()
        provenance = model_provenance()
        for name in ('real', 'synthetic'):
            with self.subTest(case=name):
                path = FIXTURES / f'eye_{name}_continuous_reference.npz'
                meta = json.loads(path.with_suffix('.json').read_text())
                self.assertEqual(file_sha256(path), meta['npz_sha256'])
                for key in ('checkpoint_sha256', 'architecture_sha256'):
                    self.assertEqual(provenance[key], meta['model'][key])
                with np.load(path) as ref:
                    timeline, before, actual = eye.process_segments(ref['raw_time_us'], ref['raw'], model=model)
                    np.testing.assert_array_equal(timeline.time_us, ref['time_us'])
                    np.testing.assert_array_equal(before, ref['before'])
                    np.testing.assert_allclose(actual, ref['processed'], atol=1e-6, rtol=1e-6)
                    self.assertEqual(timeline.segments[0]['output_end_idx'], len(actual))
                    for col, channel in enumerate((1, 2, 5, 6)):
                        for stage, signal in (('before', before[:, channel - 1]), ('after', actual[:, col])):
                            f, t, db = eye.compute_stft_db(signal, 200, 50.)
                            np.testing.assert_array_equal(f, ref[f'{stage}_ch{channel}_f'])
                            np.testing.assert_array_equal(t, ref[f'{stage}_ch{channel}_t'])
                            np.testing.assert_allclose(db, ref[f'{stage}_ch{channel}_db'], atol=1e-6, rtol=1e-6)

    def test_group_mapping_and_gap_processing_equal_independent_segments(self):
        lengths = (1503, 2001)
        time_us = np.r_[np.arange(lengths[0]) * 2000, 10000000 + np.arange(lengths[1]) * 2000]
        raw = np.random.default_rng(15).normal(size=(sum(lengths), 8)).astype(np.float32)
        timeline, before, actual = eye.process_segments(time_us, raw, model=Identity())
        expected = []
        for sl in continuous_slices(time_us, 500):
            filtered = eye.bandpass_filter(raw[sl], fs=500, lo=.5, hi=45.)
            _, part = downsample_data(time_us[sl].astype(float) / 1e6, filtered)
            expected.append(part)
        np.testing.assert_array_equal(before, np.concatenate(expected))
        np.testing.assert_allclose(actual, before[:, [0, 1, 4, 5]], atol=1e-6, rtol=1e-6)
        np.testing.assert_array_equal(timeline.segment_ids, np.r_[np.zeros(602), np.ones(801)])
        self.assertEqual(timeline.segments[1]['raw_start_idx'], 1503)
        for window in timeline.model_windows:
            record = timeline.segments[window['segment_id']]
            self.assertGreaterEqual(window['output_start_idx'], record['output_start_idx'])
            self.assertLessEqual(window['output_end_idx'], record['output_end_idx'])
        # Changing a different segment cannot contaminate the retained result.
        raw[:1503] *= 1000
        _, changed_input, changed_output = eye.process_segments(time_us, raw, model=Identity())
        np.testing.assert_array_equal(changed_input[602:], before[602:])
        np.testing.assert_array_equal(changed_output[602:], actual[602:])

    def test_short_segments_keep_source_id_epoch_and_complete_resampled_tail(self):
        epoch = 1750000000000001
        time_us = epoch + np.r_[np.arange(10) * 2000, 10000000 + np.arange(1503) * 2000,
                                20000000 + np.arange(997) * 2000]
        timeline, before, actual = eye.process_segments(time_us, np.ones((len(time_us), 8)), model=Identity())
        self.assertEqual([s['status'] for s in timeline.segments], ['excluded', 'retained', 'excluded'])
        self.assertEqual(timeline.source_epoch_us, epoch)
        self.assertEqual(timeline.segments[1]['raw_start_idx'], 10)
        np.testing.assert_array_equal(timeline.segment_ids, np.ones(602))
        np.testing.assert_array_equal(timeline.time_us, epoch + 10000000 + np.arange(602) * 5000)
        self.assertEqual(before.shape, (602, 8))
        self.assertEqual(actual.shape, (602, 4))
        self.assertEqual(timeline.segments[0]['reason'], 'shorter_than_model_window')

    def test_minimum_model_length_and_small_raw_gap_are_explicit(self):
        time_us = np.r_[np.arange(998) * 2000, 2002000 + np.arange(998) * 2000]
        timeline, _, actual = eye.process_segments(time_us, np.ones((1996, 8)), model=Identity())
        self.assertEqual(actual.shape, (800, 4))
        self.assertEqual(len(continuous_slices(timeline.time_us, 200)), 1)
        self.assertEqual(len(continuous_slices(timeline.time_us, 200, segment_ids=timeline.segment_ids)), 2)
        np.testing.assert_array_equal(timeline.time_us[[399, 400]], [1995000, 2002000])

    def test_all_short_fails_before_loading_model(self):
        with patch.object(eye, 'load_model') as loader:
            with self.assertRaisesRegex(ValueError, 'No continuous segment'):
                eye.process_segments(np.arange(997) * 2000, np.ones((997, 8)))
            loader.assert_not_called()

    def test_nonfinite_even_in_short_segments_fails_before_inference(self):
        for bad in (np.nan, np.inf, -np.inf):
            raw = np.ones((1010, 8))
            raw[0, 7] = bad
            t = np.r_[np.arange(10) * 2000, 10000000 + np.arange(1000) * 2000]
            with patch.object(eye, 'run_model') as run:
                with self.assertRaisesRegex(ValueError, 'non-finite'):
                    eye.process_segments(t, raw, model=Identity())
                run.assert_not_called()

    def test_bad_channel_shape_and_timestamps_fail(self):
        t = np.arange(1000) * 2000
        for shape in ((1000, 7), (1000, 9), (999, 8), (1000,)):
            with self.assertRaisesRegex(ValueError, 'exactly eight'):
                eye.process_segments(t, np.ones(shape), model=Identity())
        for time_us, message in ((t.astype(float), 'integer'), (np.zeros(1000, dtype=np.int64), 'strictly increasing')):
            with self.assertRaisesRegex(ValueError, message):
                eye.process_segments(time_us, np.ones((1000, 8)), model=Identity())

    def test_failed_second_group_reports_segment_and_never_returns_partial_output(self):
        for failure in (RuntimeError('inference failure'), np.zeros((399, 2)), np.full((400, 2), np.nan)):
            with patch.object(eye, 'run_model', side_effect=[np.zeros((400, 2)), failure]):
                with self.assertRaisesRegex(ValueError, r'segment 0 raw \[0:1000\].*input channels 5-8'):
                    eye.process_segments(np.arange(1000) * 2000, np.ones((1000, 8)), model=Identity())


class EyeOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source.csv'
        self.output = self.root / 'output.csv'
        self.sidecar = Path(str(self.output) + '.meta.json')
        self.parameters = signal_parameters()

    def recording(self, time_us, raw=None):
        if raw is None:
            raw = np.random.default_rng(151).normal(size=(len(time_us), 8)).astype(np.float32)
        with self.source.open('w') as handle:
            handle.write('File Name,source\nAmp Gain,500\nChannels,1,2,3,4,5,6,7,8\n'
                         'Sample Rate (per channel),500,500,500,500,500,500,500,500\n'
                         'Time[us],ch1,ch2,ch3,ch4,ch5,ch6,ch7,ch8\n')
            for timestamp, row in zip(time_us, raw):
                handle.write(','.join([str(int(timestamp)), *map(str, row)]) + '\n')
        return raw

    def table(self, time_us):
        raw = self.recording(time_us)
        timeline, _, output = eye.process_segments(time_us, raw, model=Identity())
        write_signal_table(self.output, self.source, timeline, output, self.parameters, {})
        return timeline, output

    def read(self):
        return load_signal_table(self.output, self.source, model_path=eye.MODEL_PATH)

    def save_meta(self, meta):
        self.sidecar.write_text(json.dumps(meta))

    def test_gapped_table_roundtrip_keeps_short_segments_epoch_and_all_sample_maps(self):
        epoch = 9000000000000001
        t = epoch + np.r_[np.arange(10) * 2000, 10000000 + np.arange(1503) * 2000,
                          20000000 + np.arange(998) * 2000]
        timeline, output = self.table(t)
        frame, meta = self.read()
        np.testing.assert_array_equal(frame['Time[us]'], timeline.time_us)
        np.testing.assert_array_equal(frame.iloc[:, 1:].to_numpy(), output)
        self.assertEqual(meta['inference']['segments'][0]['status'], 'excluded')
        self.assertEqual(meta['sample_mapping']['segment_id'][602], 2)
        self.assertEqual(meta['sample_mapping']['raw_fractional_idx'][602], 1513)
        self.assertEqual(meta['inference']['model_windows'][0]['input_start_idx'], -200)
        self.assertEqual(meta['parameters']['model']['filters'], {'bandpass': [.5, 45.]})
        self.assertEqual(meta['quality_state'], 'disabled')

    def test_metadata_tampering_rejected_even_with_valid_table_hash(self):
        self.table(np.arange(1503) * 2000)
        original = json.loads(self.sidecar.read_text())
        mutations = [
            lambda m: m.update(kind='app_nuc_signal'),
            lambda m: m.update(quality_state='passed'),
            lambda m: m['sample_mapping']['raw_fractional_idx'].__setitem__(1, 3.),
            lambda m: m['sample_mapping']['segment_id'].__setitem__(0, 1),
            lambda m: m['sample_mapping']['output_idx'].__setitem__(0, 1),
            lambda m: m['inference']['model_windows'][0].update(input_start_idx=0),
            lambda m: m['inference']['segments'][0].update(raw_end_idx=1502),
            lambda m: m['parameters'].update(output_channels=[1, 2, 3, 4]),
            lambda m: m['parameters']['model'].update(architecture_sha256='bad'),
        ]
        for mutate in mutations:
            meta = copy.deepcopy(original)
            mutate(meta)
            self.save_meta(meta)
            with self.assertRaises(ValueError):
                self.read()

    def test_table_timestamp_header_width_and_nonfinite_tampering_rejected(self):
        self.table(np.arange(1503) * 2000)
        original = self.output.read_text()
        meta = json.loads(self.sidecar.read_text())
        for kind in ('timestamp', 'header', 'width', 'nonfinite', 'fractional_timestamp'):
            with self.subTest(kind=kind):
                lines = original.splitlines()
                row = lines[5].split(',')
                if kind == 'timestamp':
                    row[0] = '1'
                elif kind == 'fractional_timestamp':
                    row[0] = '0.5'
                elif kind == 'header':
                    lines[2] = 'Channels,1,2,3,4'
                elif kind == 'width':
                    row.append('99')
                else:
                    row[1] = 'nan'
                lines[5] = ','.join(row)
                self.output.write_text('\n'.join(lines) + '\n')
                meta['table_sha256'] = file_sha256(self.output)
                self.save_meta(meta)
                with self.assertRaises(ValueError):
                    self.read()

    def test_source_model_and_signal_hash_mismatch_rejected(self):
        self.table(np.arange(1503) * 2000)
        bad_model = self.root / 'bad.pth'
        bad_model.write_bytes(b'wrong checkpoint')
        with self.assertRaisesRegex(ValueError, 'checkpoint'):
            load_signal_table(self.output, self.source, model_path=bad_model)
        self.source.write_text(self.source.read_text().replace('Amp Gain,500', 'Amp Gain,501'))
        with self.assertRaisesRegex(ValueError, 'source'):
            self.read()
        self.output.write_text(self.output.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            self.read()

    def test_writer_preserves_existing_table_and_rejects_nonfinite(self):
        timeline, output = self.table(np.arange(1503) * 2000)
        before = self.output.read_bytes()
        with self.assertRaises(FileExistsError):
            write_signal_table(self.output, self.source, timeline, output, self.parameters, {})
        self.assertEqual(self.output.read_bytes(), before)
        output[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'finite'):
            write_signal_table(self.root / 'bad.csv', self.source, timeline, output, self.parameters, {})
        self.assertFalse((self.root / 'bad.csv').exists())

    def args(self, name='run'):
        return argparse.Namespace(csv=str(self.source), outdir=str(self.root / name), fmax=50.)

    def test_main_uses_adapter_exact_timestamps_and_verifies_outputs(self):
        t = 9000000000000001 + np.arange(1503) * 2000
        raw = self.recording(t)
        timeline, _, expected = eye.process_segments(t, raw, model=Identity())
        # Plot behavior is outside this increment; separately exercise real CLI.
        def plot(time_s, data, outdir, stem, fmax, offset, suffix):
            p = Path(outdir) / f'plot_{suffix}.txt'
            p.write_text('test plot')
            return p
        with patch.object(eye, 'load_model', return_value=Identity()), \
                patch.object(eye, 'process_segments', wraps=eye.process_segments) as adapter, \
                patch.object(eye, 'plot_output_channels', side_effect=plot), \
                patch.object(eye, 'plot_before_after_channels', side_effect=lambda t, b, a, *args: plot(t, a, *args)):
            audit = eye.run_analysis(self.args())
        adapter.assert_called_once()
        self.assertEqual(audit['status'], 'success')
        self.assertTrue(audit['table_verified'])
        frame, _ = load_signal_table(self.root / 'run/source_tinyv4_output.csv', self.source,
                                    model_path=eye.MODEL_PATH)
        np.testing.assert_array_equal(frame['Time[us]'], timeline.time_us)
        np.testing.assert_array_equal(frame.iloc[:, 1:], expected)

    def test_main_failure_audits_preserve_guard_short_nonfinite_and_model_context(self):
        cases = ('gap', 'short', 'nonfinite', 'model', 'missing', 'invalid_fmax')
        for case in cases:
            with self.subTest(case=case):
                t = np.arange(1503) * 2000
                if case == 'gap':
                    t[1000:] += 10000000
                elif case == 'short':
                    t = t[:997]
                raw = np.ones((len(t), 8), dtype=np.float32)
                if case == 'nonfinite':
                    raw[0, 7] = np.nan
                self.recording(t, raw)
                if case == 'missing':
                    self.source.unlink()
                args = self.args(case)
                if case == 'invalid_fmax':
                    args.fmax = float('nan')
                with patch.object(eye, 'load_model', return_value=Identity()), \
                        patch.object(eye, 'run_model', side_effect=[np.zeros((602, 2)), RuntimeError('model failed')]):
                    with self.assertRaises((ValueError, FileNotFoundError)):
                        eye.run_analysis(args)
                audit = json.loads((Path(args.outdir) / 'source_analysis_audit.json').read_text())
                self.assertEqual(audit['status'], 'failed')
                self.assertFalse(audit['table_verified'])
                self.assertEqual(audit['artifacts'], [])
                self.assertFalse((Path(args.outdir) / 'source_tinyv4_output.csv').exists())
                if case == 'gap':
                    self.assertEqual(audit['stage'], 'continuity_guard')
                    self.assertEqual(len(audit['inference']['segments']), 2)
                elif case == 'short':
                    self.assertEqual(audit['inference']['segments'][0]['status'], 'excluded')
                elif case == 'nonfinite':
                    self.assertEqual(audit['nonfinite_input'], [{'segment_id': 0, 'input_channel': 8, 'count': 1}])
                elif case == 'model':
                    self.assertIn('segment 0 raw [0:1503]', audit['error']['message'])
                    self.assertIn('input channels 5-8', audit['error']['message'])

    def test_main_collision_preserves_previous_artifacts_and_audit(self):
        self.recording(np.arange(1503) * 2000)
        args = self.args()
        directory = Path(args.outdir)
        directory.mkdir()
        previous = {directory / name: b'previous research' for name in (
            'source_tinyv4_output.csv', 'source_analysis_audit.json',
            'source_tinyv4_output_ch5_6_time_stft.png')}
        for path, contents in previous.items():
            path.write_bytes(contents)
        with self.assertRaises(FileExistsError):
            eye.run_analysis(args)
        for path, contents in previous.items():
            self.assertEqual(path.read_bytes(), contents)
        failure, = directory.glob('source_analysis_audit_*.json')
        self.assertEqual(json.loads(failure.read_text())['stage'], 'preflight')


if __name__ == '__main__':
    unittest.main()
