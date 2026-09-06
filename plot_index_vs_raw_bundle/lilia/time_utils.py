"""Shared time-conversion helpers for session/event alignment."""

from __future__ import annotations

import datetime


UTC_EPOCH = datetime.datetime(1970, 1, 1)


def hhmm_to_local_dt(hhmm: str, session_date: datetime.date) -> datetime.datetime:
    """Convert ``HH:MM`` to naive local datetime on ``session_date``."""
    h, m = map(int, hhmm.split(":"))
    return datetime.datetime(session_date.year, session_date.month, session_date.day, h, m)


def local_dt_to_utc_us(dt_local: datetime.datetime, tz_offset_h: int) -> int:
    """Convert local datetime (naive or aware) to UTC Unix microseconds."""
    if dt_local.tzinfo is None:
        tz = datetime.timezone(datetime.timedelta(hours=tz_offset_h))
        dt_local = dt_local.replace(tzinfo=tz)
    dt_utc = dt_local.astimezone(datetime.timezone.utc).replace(tzinfo=None)
    return int((dt_utc - UTC_EPOCH).total_seconds() * 1_000_000)


def hhmm_to_utc_us(hhmm: str, session_date: datetime.date, tz_offset_h: int) -> int:
    """Convert session-local ``HH:MM`` to UTC Unix microseconds."""
    dt_local = hhmm_to_local_dt(hhmm, session_date)
    return local_dt_to_utc_us(dt_local, tz_offset_h)


def utc_us_to_local_dt(us: int, tz_offset_h: int) -> datetime.datetime:
    """Convert UTC Unix microseconds to naive local datetime."""
    return UTC_EPOCH + datetime.timedelta(microseconds=int(us)) + datetime.timedelta(hours=tz_offset_h)
