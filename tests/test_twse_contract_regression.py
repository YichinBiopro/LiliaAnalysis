import unittest

import numpy as np
import pandas as pd

import twse_index_lstm_rnn as twse


class TWSEContractTests(unittest.TestCase):
    def test_entire_label_horizon_stays_in_its_split(self):
        x, y, starts = twse.build_sequences(np.arange(100)[:, None], np.arange(100), 10, 5)
        train, val, test = twse.sequence_split_masks(starts, 5, 70, 85)
        self.assertLess(y[train].max(), 70)
        self.assertGreaterEqual(y[val].min(), 70)
        self.assertLess(y[val].max(), 85)
        self.assertGreaterEqual(y[test].min(), 85)
        self.assertEqual(int((~(train | val | test)).sum()), 8)
        np.testing.assert_array_equal(x[0, :, 0], np.arange(10))

    def test_missing_volume_does_not_drop_every_feature_row(self):
        close = np.arange(220, dtype=float) + 100
        frame = pd.DataFrame({'Date': pd.date_range('2020-01-01', periods=220),
                              'Open': close - 1, 'High': close + 1, 'Low': close - 2,
                              'Close': close, 'Volume': np.zeros(220)})
        result = twse.add_features(frame)
        self.assertEqual(len(result), 201)
        self.assertTrue(np.isfinite(result.select_dtypes('number')).all().all())
        self.assertTrue((result['VolChg'] == 0).all())

    def test_zero_price_produces_no_infinite_features(self):
        close = np.arange(220, dtype=float) + 100
        close[100] = 0
        frame = pd.DataFrame({'Open': close, 'High': close + 1, 'Low': close - 1,
                              'Close': close, 'Volume': np.ones(220)})
        result = twse.add_features(frame)
        self.assertTrue(np.isfinite(result).all().all())
