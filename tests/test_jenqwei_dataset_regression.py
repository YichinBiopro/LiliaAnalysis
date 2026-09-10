"""Dataset split/window, source mapping and publication regression tests."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from lilia.io import bandpass_filter
from lilia.jenqwei_dataset import (
    DatasetPlanError, dataset_plan, load_fragment, load_manifest, parameters,
    process_source, write_manifest,
)
from lilia.provenance import file_sha256
from lilia.signal import resample_with_time

ROOT = Path(__file__).resolve().parents[1]


def write_source(path, t, raw):
    with Path(path).open('w') as handle:
        handle.write('header\n' * 4)
        pd.DataFrame({'Time[us]': t, **{f'ch{i+1}': raw[:, i] for i in range(raw.shape[1])}}).to_csv(handle, index=False)


class DatasetRegression(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source.csv'
        self.t = 1700000000000001 + np.arange(7503, dtype=np.int64) * 2000
        self.raw = np.random.default_rng(17017).normal(size=(len(self.t), 4)).astype(np.float32)
        write_source(self.source, self.t, self.raw)

    def generate(self, **kwargs):
        p = parameters(**kwargs)
        rows = process_source(self.source, self.root / 'out', p)
        manifest = write_manifest(self.root / 'out', rows, [self.source], p)
        return rows, manifest

    def test_frozen_continuous_and_fractional_both_modes(self):
        for name, rate in [('synthetic', 200.), ('fractional', 199.5)]:
            fixture = ROOT / 'tests/fixtures' / f'dataset_jenqwei_{name}_reference.npz'
            meta = json.loads(fixture.with_suffix('.json').read_text())
            self.assertEqual(file_sha256(fixture), meta['npz_sha256'])
            with np.load(fixture) as old:
                for single in (False, True):
                    with self.subTest(name=name, single=single):
                        mode = 'windows' if single else 'splits'
                        p = parameters(fs_out=rate, one_window_per_file=single)
                        out = self.root / f'{name}_{mode}'
                        rows = process_source(self.source, out, p)
                        manifest = write_manifest(out, rows, [self.source], p)
                        self.assertEqual(len(load_manifest(manifest)[0]), len(rows))
                        for i, (row, previous) in enumerate(zip(rows, meta['modes'][mode])):
                            for key in previous.keys() - {'out_csv', 'source_csv'}:
                                self.assertEqual(getattr(row, key), previous[key])
                            frame, _ = load_fragment(row.out_csv, self.source)
                            np.testing.assert_array_equal(frame.time_us, old[f'{mode}_{i}_time_us'])
                            np.testing.assert_array_equal(frame.iloc[:, 1:].to_numpy(dtype=np.float32), old[f'{mode}_{i}_values'])

    def test_splits_before_trimming_and_true_window_count(self):
        p = parameters(n_splits=3)
        plan = dataset_plan(self.t, p)
        self.assertEqual([f['resampled_start_idx'] for f in plan['fragments']], [0, 1000, 2001])
        self.assertEqual([f['model_window_count'] for f in plan['fragments']], [2, 2, 2])
        self.assertEqual([f['window_count_in_split'] for f in plan['fragments']], [1, 1, 1])
        self.assertEqual([s['excluded_samples'] for s in plan['segments'][0]['splits']], [200, 201, 201])

    def test_segment_filter_independence_and_window_coordinates(self):
        t = self.t.copy()
        t[3503:] += 500000
        write_source(self.source, t, self.raw)
        for single in (False, True):
            p = parameters(n_splits=2, one_window_per_file=single)
            rows = process_source(self.source, self.root / str(single), p)
            for row in rows:
                a, b = row.raw_start_idx, row.raw_end_idx
                filtered = bandpass_filter(self.raw[a:b], fs=500., lo=.5, hi=45.)
                rt, data = resample_with_time(t[a:b], filtered, 500., 200.)
                frame, meta = load_fragment(row.out_csv, self.source)
                start, end = row.segment_local_start_idx, row.segment_local_end_idx
                np.testing.assert_array_equal(frame.time_us, rt[start:end])
                # CSV decimal formatting preserves float32 round trips.
                np.testing.assert_array_equal(frame.iloc[:, 1:].to_numpy(dtype=np.float32), data[start:end])
                self.assertLess(row.raw_fractional_last_idx, b)
                self.assertGreaterEqual(row.raw_fractional_first_idx, a)
                self.assertEqual(row.model_window_count * 400, len(frame))
                self.assertEqual(len(meta['plan']['segments']), 2)
        # An impulse in the next segment cannot affect the first segment.
        changed = self.raw.copy()
        changed[3503] = 1e8
        write_source(self.source, t, changed)
        new = process_source(self.source, self.root / 'changed', parameters(n_splits=2))
        original = sorted((self.root / 'False').rglob('*_seg0000_*.csv'))
        for row, old_path in zip([r for r in new if r.segment_id == 0], original):
            self.assertEqual(file_sha256(row.out_csv), file_sha256(old_path))

    def test_gap_threshold_and_nonmonotonic(self):
        t = self.t.copy()
        t[3503:] += 4000  # exactly 3 nominal periods: same shared policy
        self.assertEqual(len(dataset_plan(t, parameters())['segments']), 1)
        t[3503:] += 1
        self.assertEqual(len(dataset_plan(t, parameters())['segments']), 2)
        for value in [t[10], t[10] - 1]:
            bad = t.copy()
            bad[11] = value
            with self.assertRaisesRegex(ValueError, 'strictly increasing'):
                dataset_plan(bad, parameters())

    def test_short_split_strict_rejects_before_writing(self):
        with self.assertRaises(DatasetPlanError) as error:
            self.generate(n_splits=10)
        self.assertTrue(error.exception.plan['segments'])
        self.assertFalse((self.root / 'out').exists())

    def test_short_segment_drop_and_filter_pad_boundary(self):
        for length in (27, 28):
            t = self.t.copy()
            t[length:] += 500000
            write_source(self.source, t, self.raw)
            rows = process_source(self.source, self.root / str(length), parameters(allow_short_drop=True))
            _, meta = load_fragment(rows[0].out_csv, self.source)
            first = meta['plan']['segments'][0]
            self.assertEqual(first['splits'][0]['reason'], 'filter_too_short' if length == 27 else 'short_split')
            self.assertEqual({r.segment_id for r in rows}, {1})
            self.assertEqual(rows[0].raw_start_idx, length)
            self.assertEqual(rows[0].resampled_start_idx, (length * 2 + 4) // 5)

    def test_all_dropped_is_error(self):
        with self.assertRaisesRegex(DatasetPlanError, 'all splits excluded'):
            self.generate(n_splits=100, allow_short_drop=True)

    def test_partial_short_splits_are_audited(self):
        write_source(self.source, self.t[:2998], self.raw[:2998])  # 1200 resampled, three exact windows
        rows, _ = self.generate(n_splits=3)
        self.assertEqual(len(rows), 3)
        write_source(self.source, self.t[:2996], self.raw[:2996])  # 1199: [399,400,400]
        rows = process_source(self.source, self.root / 'partial', parameters(n_splits=3, allow_short_drop=True))
        self.assertEqual([r.split_index for r in rows], [2, 3])
        self.assertEqual(load_fragment(rows[0].out_csv, self.source)[1]['plan']['segments'][0]['splits'][0]['reason'], 'short_split')

    def test_pollution_in_dropped_and_unused_channels_rejected(self):
        for channel, index in [(0, 0), (4, 0), (0, -1)]:
            raw = np.column_stack([self.raw, np.zeros(len(self.raw), dtype=np.float32)])
            raw[index, channel] = np.nan
            t = self.t.copy()
            t[20:] += 500000
            write_source(self.source, t, raw)
            with self.assertRaisesRegex(ValueError, 'Non-finite'):
                self.generate(allow_short_drop=True)

    def test_channel_contract_and_invalid_parameters(self):
        for kwargs in [dict(n_ch=5), dict(n_ch=0), dict(n_ch=True), dict(n_splits=0),
                       dict(tflite_win=0), dict(fs_in=float('nan')), dict(fs_out=0),
                       dict(bp_high=250.), dict(bp_low=float('nan')), dict(allow_short_drop='yes')]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.generate(**kwargs)

    def test_repeat_does_not_overwrite(self):
        rows, manifest = self.generate()
        before = file_sha256(rows[0].out_csv), file_sha256(manifest)
        with self.assertRaises(FileExistsError):
            self.generate()
        self.assertEqual(before, (file_sha256(rows[0].out_csv), file_sha256(manifest)))

    def test_source_change_rejected(self):
        rows, _ = self.generate()
        write_source(self.source, self.t, self.raw * 2)
        with self.assertRaisesRegex(ValueError, 'source mismatch'):
            load_fragment(rows[0].out_csv, self.source)

    def test_fragment_timestamp_tamper_even_with_updated_hash(self):
        rows, _ = self.generate()
        path = Path(rows[0].out_csv)
        frame = pd.read_csv(path)
        frame.loc[1, 'time_us'] += 1
        frame.to_csv(path, index=False)
        audit_path = path.parent / 'dataset.json'
        meta = json.loads(audit_path.read_text())
        meta['files'][0]['sha256'] = file_sha256(path)
        audit_path.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, 'timestamps'):
            load_fragment(path, self.source)

    def test_missing_extra_fragment_and_mapping_tamper(self):
        rows, _ = self.generate()
        path = Path(rows[0].out_csv)
        extra = path.parent / 'extra.csv'
        extra.write_text(path.read_text())
        with self.assertRaisesRegex(ValueError, 'extra'):
            load_fragment(path, self.source)
        extra.unlink()
        audit_path = path.parent / 'dataset.json'
        original = audit_path.read_text()
        meta = json.loads(original)
        meta['plan']['fragments'][0]['raw_fractional_first_idx'] += 1
        audit_path.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, 'plan differs'):
            load_fragment(path, self.source)
        audit_path.write_text(original)
        Path(rows[-1].out_csv).unlink()
        with self.assertRaisesRegex(ValueError, 'Missing'):
            load_fragment(path, self.source)

    def test_manifest_missing_duplicate_and_reordered_rows_even_rehashed(self):
        _, manifest = self.generate()
        original = pd.read_csv(manifest, float_precision='round_trip')
        meta_path = manifest.with_suffix('.meta.json')
        for frame in [original.iloc[:-1], pd.concat([original, original.iloc[:1]]), original.iloc[::-1]]:
            frame.to_csv(manifest, index=False)
            meta = json.loads(meta_path.read_text())
            meta['manifest_sha256'] = file_sha256(manifest)
            meta_path.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, 'manifest rows'):
                load_manifest(manifest)

    def test_source_mutation_during_processing_detected(self):
        original = bandpass_filter
        def mutate(*args, **kwargs):
            write_source(self.source, self.t, self.raw * 2)
            return original(*args, **kwargs)
        with patch('lilia.jenqwei_dataset.bandpass_filter', side_effect=mutate):
            with self.assertRaisesRegex(ValueError, 'Source changed'):
                self.generate()
        self.assertFalse(list((self.root / 'out').rglob('dataset.json')))

    def test_custom_window_channel_and_upsampling(self):
        # Smallest filterable segment, retaining all positions including the final
        # fractional centre beyond the last raw centre, still inside [0, 28).
        write_source(self.source, self.t[:28], self.raw[:28])
        rows, manifest = self.generate(n_splits=1, n_ch=2, fs_out=1000., tflite_win=1)
        self.assertEqual(rows[0].model_window_count, 56)
        self.assertEqual(rows[0].raw_fractional_last_idx, 27.5)
        self.assertEqual(len(load_manifest(manifest)[0]), 1)
        frame, _ = load_fragment(rows[0].out_csv, self.source)
        self.assertEqual(list(frame), ['time_us', 'ch1', 'ch2'])
        self.assertEqual(frame.time_us.iloc[-1], self.t[0] + 55000)

    def test_float_timestamp_plan_rejected(self):
        with self.assertRaisesRegex(ValueError, 'integer microseconds'):
            dataset_plan(self.t.astype(float), parameters())

    def test_same_basename_different_sources_do_not_collide(self):
        other = self.root / 'second' / self.source.name
        other.parent.mkdir()
        write_source(other, self.t, self.raw * 2)
        params = parameters()
        first = process_source(self.source, self.root / 'out', params)
        second = process_source(other, self.root / 'out', params)
        self.assertNotEqual(Path(first[0].out_csv).parent, Path(second[0].out_csv).parent)
        manifest = write_manifest(self.root / 'out', first + second, [self.source, other], params)
        self.assertEqual(len(load_manifest(manifest)[0]), 10)

    def test_incomplete_manifest_refused_before_publication(self):
        p = parameters()
        rows = process_source(self.source, self.root / 'out', p)
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            write_manifest(self.root / 'out', rows[:-1], [self.source], p)
        self.assertFalse((self.root / 'out' / 'manifest.csv').exists())

    def test_evidence_uses_shared_audit_and_rejects_kind_and_signal_tamper(self):
        from tools.refactor_check import evidence
        rows, _ = self.generate()
        path = Path(rows[0].out_csv)
        proof = self.root / 'evidence.json'
        proof.write_text(json.dumps(dict(schema_version=1, tables=[dict(
            path=str(path), raw_csv=str(self.source), kind='jenqwei_dataset')])))
        checks, inputs = evidence(proof)
        self.assertEqual(checks[0]['rows'], 400)
        audit_path = path.parent / 'dataset.json'
        self.assertIn(audit_path, inputs)
        original = audit_path.read_text()
        audit = json.loads(original)
        audit['kind'] = 'unknown'
        audit_path.write_text(json.dumps(audit))
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            evidence(proof)
        audit_path.write_text(original)
        frame = pd.read_csv(path)
        frame.loc[0, 'ch1'] += 1
        frame.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            evidence(proof)

    def test_cli_failed_audit_and_no_empty_success(self):
        out = self.root / 'cli'
        result = subprocess.run([sys.executable, str(ROOT / 'build_jenqwei_tflite_dataset.py'),
                                 '--input-glob', str(self.source), '--outdir', str(out),
                                 '--n-splits', '100', '--allow-short-drop'], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        audit = json.loads((out / 'run_audit.json').read_text())
        self.assertEqual(audit['status'], 'failed')
        self.assertTrue(audit['failed']['plan']['segments'])
        self.assertFalse((out / 'manifest.csv').exists())


if __name__ == '__main__':
    unittest.main()
