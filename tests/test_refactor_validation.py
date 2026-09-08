"""Validation tooling must reject misleading passes and preserve exact coordinates."""
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools import refactor_check as checker


class RefactorValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def compare(self, a, b, **kwargs):
        actual, expected = self.root / 'new.npz', self.root / 'old.npz'
        np.savez(actual, **a)
        np.savez(expected, **b)
        return checker.compare_arrays(actual, expected, rtol=0, atol=1e-12, **kwargs)

    def test_numeric_difference_and_nan_policy(self):
        with self.assertRaises(AssertionError):
            self.compare({'x': [1.0]}, {'x': [1.01]})
        with self.assertRaises(AssertionError):
            self.compare({'x': [np.nan]}, {'x': [np.nan]})
        self.assertIsNone(self.compare({'x': [np.nan]}, {'x': [np.nan]}, equal_nan=True)['x'])
        with self.assertRaises(AssertionError):
            self.compare({'x': [np.nan, 1.]}, {'x': [1., np.nan]}, equal_nan=True)

    def test_large_integer_timestamps_are_exact(self):
        with self.assertRaises(ValueError):
            self.compare({'time_us': [2**60]}, {'time_us': [2**60 + 1]})
        self.assertEqual(self.compare({'time_us': [2**60]}, {'time_us': [2**60]}), {'time_us': 0.0})

    def test_keys_shape_empty_and_tolerance_rejected(self):
        for a, b in (({}, {}), ({'a': [1.]}, {'b': [1.]}),
                     ({'a': [1.]}, {'a': [[1.]]}), ({'a': []}, {'a': []})):
            with self.subTest(a=a, b=b), self.assertRaises(ValueError):
                self.compare(a, b)
        with self.assertRaises(ValueError):
            checker.compare_arrays('unused', 'unused', rtol=float('nan'), atol=0)

    def test_hash_manifest_rejects_tamper_empty_and_unknown_schema(self):
        artifact = self.root / 'plot.txt'
        artifact.write_text('original')
        manifest = self.root / 'manifest.json'
        value = {'schema_version': 1, 'files': [{'path': artifact.name, 'sha256': checker.sha256(artifact)}]}
        manifest.write_text(json.dumps(value))
        rows, inputs = checker.evidence(manifest)
        self.assertEqual(len(rows), 1)
        self.assertIn(artifact, inputs)
        artifact.write_text('changed')
        with self.assertRaisesRegex(ValueError, 'Hash mismatch'):
            checker.evidence(manifest)
        for value in ({'schema_version': 1}, {'schema_version': 1, 'filez': []}):
            manifest.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                checker.evidence(manifest)

    def test_unknown_table_kind_cannot_silently_downgrade(self):
        table = self.root / 'data.csv'
        table.write_text('x\n1\n')
        Path(str(table) + '.meta.json').write_text('{"kind": "unknown"}')
        manifest = self.root / 'manifest.json'
        manifest.write_text(json.dumps({'schema_version': 1, 'tables': [
            {'path': table.name, 'raw_csv': table.name, 'kind': 'unknown'}]}))
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            checker.evidence(manifest)

    def test_zero_test_run_is_not_a_pass(self):
        result = checker.run_command('zero', [sys.executable, '-c',
                                     'print("Ran 0 tests in 0.001s\\n\\nOK")'], self.root, tests=True)
        self.assertFalse(result['passed'])
        self.assertEqual(result['tests'], 0)

    def test_nonzero_exit_is_not_a_pass(self):
        result = checker.run_command('failure', [sys.executable, '-c',
                                     'print("failure detail"); raise SystemExit(2)'], self.root)
        self.assertFalse(result['passed'])
        self.assertIn('failure detail', Path(result['log']).read_text())

    def test_snapshot_detects_fixture_model_and_external_input_changes(self):
        fixture = self.root / 'tests' / 'fixtures' / 'reference.json'
        fixture.parent.mkdir(parents=True)
        fixture.write_text('{}')
        model = self.root / 'model.tflite'
        model.write_bytes(b'model')
        source = self.root / 'raw.csv'
        source.write_text('raw')
        with patch.object(checker, 'ROOT', self.root):
            previous = checker.snapshot([source])
            for file in (fixture, model, source):
                file.write_bytes(file.read_bytes() + b' ')
                current = checker.snapshot([source])
                self.assertNotEqual(current, previous)
                previous = current

    def test_status_missing_input_returns_unusable_result(self):
        report = self.root / 'summary.json'
        report.write_text(json.dumps({'passed': True, 'snapshot': {
            'inputs': {str(self.root / 'missing.csv'): 'old-hash'}}}))
        result = subprocess.run([sys.executable, str(checker.ROOT / 'tools/refactor_check.py'),
                                 'status', str(report)], capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 1)
        self.assertFalse(json.loads(result.stdout)['matching_passed_check'])
        self.assertNotIn('Traceback', result.stderr)


if __name__ == '__main__':
    unittest.main()
