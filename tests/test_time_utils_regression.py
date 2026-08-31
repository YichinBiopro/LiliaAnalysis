import datetime
import unittest

from lilia.time_utils import (
    hhmm_to_local_dt,
    hhmm_to_utc_us,
    local_dt_to_utc_us,
    utc_us_to_local_dt,
)


class TimeUtilsRegressionTests(unittest.TestCase):
    def test_hhmm_to_local_dt(self):
        day = datetime.date(2026, 5, 12)
        dt = hhmm_to_local_dt("14:13", day)
        self.assertEqual(dt, datetime.datetime(2026, 5, 12, 14, 13))

    def test_hhmm_to_utc_us_matches_expected(self):
        day = datetime.date(2026, 5, 12)
        # 2026-05-12 14:13 (UTC+8) == 2026-05-12 06:13 UTC
        got = hhmm_to_utc_us("14:13", day, tz_offset_h=8)
        expected_dt = datetime.datetime(2026, 5, 12, 6, 13)
        epoch = datetime.datetime(1970, 1, 1)
        expected_us = int((expected_dt - epoch).total_seconds() * 1_000_000)
        self.assertEqual(got, expected_us)

    def test_round_trip_local_dt_and_utc_us(self):
        dt_local = datetime.datetime(2026, 5, 12, 15, 6)
        us = local_dt_to_utc_us(dt_local, tz_offset_h=8)
        dt_back = utc_us_to_local_dt(us, tz_offset_h=8)
        self.assertEqual(dt_back, dt_local)


if __name__ == "__main__":
    unittest.main()
