import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

from lilia.io import load_merged_csv, read_lilia_frame
from merge_subject_csvs import collect_csvs, merge_subject


def write_recording(path, rows, offset=1000000000000000, channels='value,value', rate=500):
    path.write_text(f'File Name,raw\nAmp Gain,500,Abs Time Offset[us],{offset},Recording Start time[us],{offset}\n'
                    f'Channels,1,2\nSample Rate (per channel),{rate},{rate}\nTime[us],{channels}\n' + rows, encoding='utf-8')


class CSVContractTests(unittest.TestCase):
    def test_extra_columns_never_shift_time_and_channels(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'raw.csv'
            write_recording(p, '0,1,2,99,98\n2000,3,4,97,96\n')
            with self.assertWarnsRegex(UserWarning, 'trailing columns'):
                t, x = load_merged_csv(p)
            np.testing.assert_array_equal(t, [0, 2000])
            np.testing.assert_array_equal(x, [[1, 2], [3, 4]])

    def test_invalid_and_empty_time_are_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'raw.csv'
            write_recording(p, 'oops,1,2\n')
            with self.assertRaisesRegex(ValueError, 'Invalid microsecond'):
                read_lilia_frame(p)
            write_recording(p, '')
            with self.assertRaisesRegex(ValueError, 'no valid samples'):
                read_lilia_frame(p)

    def test_custom_output_rerun_is_idempotent_and_preserves_collisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a, b, out = root / 'a.csv', root / 'b.csv', root / 'custom.csv'
            write_recording(a, '0,1,2\n2000,3,4\n')
            write_recording(b, '0,1,2\n2000,9,8\n')
            (root / 'metrics.csv').write_text('time_s,quality\n1,1\n')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                merge_subject(collect_csvs(root, out.name), out)
                before = out.read_bytes()
                self.assertEqual(collect_csvs(root, out.name), [str(a), str(b)])
                merge_subject(collect_csvs(root, out.name), out)
            self.assertEqual(before, out.read_bytes())
            df = read_lilia_frame(out)
            self.assertEqual(len(df), 3)  # Only identical full rows were removed.
            self.assertEqual(df.iloc[0, 0], 1000000000000000)
            self.assertEqual(int(df.iloc[:, 0].duplicated().sum()), 1)
            self.assertNotIn(str(out), collect_csvs(root, 'another.csv'))

    def test_schema_failure_does_not_overwrite_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a, b, out = root / 'a.csv', root / 'b.csv', root / 'merged.csv'
            write_recording(a, '0,1,2\n')
            write_recording(b, '0,3,4\n', rate=200)
            out.write_text('existing result')
            with self.assertRaisesRegex(ValueError, 'schema differs'):
                merge_subject([a, b], out)
            self.assertEqual(out.read_text(), 'existing result')
