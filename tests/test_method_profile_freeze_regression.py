"""Failure detection for the calibration baseline capture, not method replicas."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.freeze_method_profiles import compare_arrays, sha, verify_files


class MethodFreezeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.old, self.new = self.root/'old.npz', self.root/'new.npz'

    def test_detects_changed_values_and_nan_positions(self):
        np.savez(self.old, power=np.array([1., np.nan]))
        for values in ([1.000000001, np.nan], [np.nan, 1.]):
            np.savez(self.new, power=np.array(values))
            with self.assertRaises(AssertionError):
                compare_arrays(self.old, self.new)

    def test_detects_missing_keys_empty_archive_and_index_dtype_changes(self):
        np.savez(self.old, indices=np.array([1], dtype=np.int64))
        for data in ({}, {'other': np.array([1])}, {'indices': np.array([1.])}):
            np.savez(self.new, **data)
            with self.assertRaises(ValueError):
                compare_arrays(self.old, self.new)
        np.savez(self.old)
        np.savez(self.new)
        with self.assertRaises(ValueError):
            compare_arrays(self.old, self.new)

    def test_records_empty_arrays_without_erasing_shape_contract(self):
        np.savez(self.old, empty=np.empty((0, 2)), score=np.array([np.nan]))
        np.savez(self.new, empty=np.empty((0, 2)), score=np.array([np.nan]))
        self.assertEqual(compare_arrays(self.old, self.new), (2, ['empty']))
        np.savez(self.new, empty=np.empty((0, 3)), score=np.array([np.nan]))
        with self.assertRaises(ValueError):
            compare_arrays(self.old, self.new)

    def test_hash_is_bound_to_saved_input_and_missing_sources_fail(self):
        source = self.root/'source.txt'
        source.write_text('original')
        records = [dict(path='source.txt', sha256=sha(source))]
        verify_files(records, self.root)
        source.write_text('changed')
        with self.assertRaises(ValueError):
            verify_files(records, self.root)
        source.unlink()
        with self.assertRaises(FileNotFoundError):
            verify_files(records, self.root)
