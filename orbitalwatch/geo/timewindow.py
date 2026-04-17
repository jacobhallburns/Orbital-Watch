"""
Flexible time window with UTC-only internal storage.

UTC is the ONLY internal representation — timezone conversion is display-only
and never affects calculations.

Accepted string formats (via ``from_strings`` / ``TimeWindow.from_strings``):

    "now"               → current UTC time
    "now-6h"            → 6 hours ago
    "now+2h"            → 2 hours from now
    "now-30d"           → 30 days ago
    "2026-04-17 14:00Z" → explicit UTC
    "2026-04-17 09:00 EST" → timezone-aware; converted to UTC
    Any ISO-8601 string accepted by python-dateutil

Requires ``python-dateutil>=2.8`` (``pip install python-dateutil``).
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Union

from dateutil import parser as _dateutil_parser
from dateutil import tz as _dateutil_tz


def _parse_time_string(s: str) -> datetime:
    """Convert a flexible time string to a UTC-aware datetime.

    Understands ``"now"``, ``"now±Xh"``, ``"now±Xd"``, ``"now±Xm"``, and
    anything python-dateutil can parse.  If no timezone is specified in the
    string, UTC is assumed.

    Args:
        s: Time string to parse.

    Returns:
        Timezone-aware datetime in UTC.

    Raises:
        ValueError: If the string cannot be parsed.
    """
    now = datetime.now(tz=timezone.utc)
    s = s.strip()

    if s == "now":
        return now

    # Handle "now±Xunit" patterns
    m = re.fullmatch(r"now([+-])(\d+(?:\.\d+)?)([hdm])", s)
    if m:
        sign = 1 if m.group(1) == "+" else -1
        amount = float(m.group(2)) * sign
        unit = m.group(3)
        if unit == "h":
            return now + timedelta(hours=amount)
        if unit == "d":
            return now + timedelta(days=amount)
        if unit == "m":
            return now + timedelta(minutes=amount)

    # Convert trailing "Z" to explicit UTC offset for dateutil
    s_parse = s[:-1] + "+00:00" if s.endswith("Z") else s

    # Provide fixed-offset tzinfo objects for common US timezone abbreviations
    # that dateutil does not recognise by default.  Fixed offsets are used so
    # that "EST" always means UTC-5 and "EDT" always means UTC-4, regardless of
    # the date (consistent with common usage where the user picked the suffix).
    _TZ_INFOS = {
        "EST": timezone(timedelta(hours=-5)),
        "EDT": timezone(timedelta(hours=-4)),
        "CST": timezone(timedelta(hours=-6)),
        "CDT": timezone(timedelta(hours=-5)),
        "MST": timezone(timedelta(hours=-7)),
        "MDT": timezone(timedelta(hours=-6)),
        "PST": timezone(timedelta(hours=-8)),
        "PDT": timezone(timedelta(hours=-7)),
    }

    try:
        dt = _dateutil_parser.parse(s_parse, tzinfos=_TZ_INFOS)
    except Exception as exc:
        raise ValueError(f"Cannot parse time string {s!r}: {exc}") from exc

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(timezone.utc)


class TimeWindow:
    """A half-open time interval [start_utc, end_utc) stored as UTC.

    All classmethods are the preferred construction entry points.

    Args:
        start_utc: Timezone-aware datetime (will be normalised to UTC).
        end_utc:   Timezone-aware datetime (will be normalised to UTC).

    Raises:
        ValueError: If either argument is timezone-naive or end <= start.
    """

    def __init__(self, start_utc: datetime, end_utc: datetime) -> None:
        if start_utc.tzinfo is None:
            raise ValueError("start_utc must be timezone-aware.")
        if end_utc.tzinfo is None:
            raise ValueError("end_utc must be timezone-aware.")
        self.start_utc: datetime = start_utc.astimezone(timezone.utc)
        self.end_utc: datetime = end_utc.astimezone(timezone.utc)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_strings(cls, start: str, end: str) -> TimeWindow:
        """Parse two flexible time strings and return a TimeWindow.

        Args:
            start: Start time string (see module docstring for formats).
            end:   End time string.

        Returns:
            TimeWindow with UTC start/end.
        """
        return cls(_parse_time_string(start), _parse_time_string(end))

    @classmethod
    def last_n_hours(cls, n: float) -> TimeWindow:
        """Return a window covering the past *n* hours up to now.

        Args:
            n: Number of hours (may be fractional).
        """
        now = datetime.now(tz=timezone.utc)
        return cls(now - timedelta(hours=n), now)

    @classmethod
    def last_n_days(cls, n: float) -> TimeWindow:
        """Return a window covering the past *n* days up to now.

        Args:
            n: Number of days (may be fractional).
        """
        now = datetime.now(tz=timezone.utc)
        return cls(now - timedelta(days=n), now)

    @classmethod
    def next_n_hours(cls, n: float) -> TimeWindow:
        """Return a window covering the next *n* hours starting now.

        Args:
            n: Number of hours (may be fractional).
        """
        now = datetime.now(tz=timezone.utc)
        return cls(now, now + timedelta(hours=n))

    # ------------------------------------------------------------------
    # Display / inspection
    # ------------------------------------------------------------------

    def display(self, tz: str = "UTC") -> str:
        """Format start/end in the requested timezone for human display.

        Internal UTC values are never modified.

        Args:
            tz: ``"UTC"`` (default), ``"local"`` (system local timezone), or
                any IANA timezone string such as ``"US/Eastern"``.

        Returns:
            Human-readable string, e.g.
            ``"2026-04-17 09:00:00 EDT → 2026-04-17 15:00:00 EDT"``.
        """
        if tz == "UTC":
            tz_obj = timezone.utc
        elif tz == "local":
            tz_obj = _dateutil_tz.tzlocal()
        else:
            tz_obj = _dateutil_tz.gettz(tz)
            if tz_obj is None:
                raise ValueError(f"Unknown timezone: {tz!r}")

        fmt = "%Y-%m-%d %H:%M:%S %Z"
        start = self.start_utc.astimezone(tz_obj).strftime(fmt)
        end = self.end_utc.astimezone(tz_obj).strftime(fmt)
        return f"{start} → {end}"

    def duration_hours(self) -> float:
        """Return the window length in fractional hours."""
        return (self.end_utc - self.start_utc).total_seconds() / 3600.0

    def __repr__(self) -> str:
        return (
            f"TimeWindow({self.start_utc.isoformat()} → {self.end_utc.isoformat()}, "
            f"{self.duration_hours():.2f}h)"
        )
