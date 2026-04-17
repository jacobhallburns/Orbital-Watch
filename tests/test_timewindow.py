"""Tests for orbitalwatch.geo.timewindow — TimeWindow and string parsing."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from orbitalwatch.geo.timewindow import TimeWindow, _parse_time_string


class TestParseTimeString:
    def test_now_returns_recent_utc(self):
        before = datetime.now(tz=timezone.utc)
        result = _parse_time_string("now")
        after = datetime.now(tz=timezone.utc)
        assert before <= result <= after

    def test_now_minus_6h(self):
        before = datetime.now(tz=timezone.utc)
        result = _parse_time_string("now-6h")
        expected = before - timedelta(hours=6)
        assert abs((result - expected).total_seconds()) < 2

    def test_now_minus_24h(self):
        result = _parse_time_string("now-24h")
        expected = datetime.now(tz=timezone.utc) - timedelta(hours=24)
        assert abs((result - expected).total_seconds()) < 2

    def test_now_minus_30d(self):
        result = _parse_time_string("now-30d")
        expected = datetime.now(tz=timezone.utc) - timedelta(days=30)
        assert abs((result - expected).total_seconds()) < 2

    def test_explicit_utc_z(self):
        result = _parse_time_string("2026-04-17 14:00Z")
        assert result == datetime(2026, 4, 17, 14, 0, 0, tzinfo=timezone.utc)

    def test_explicit_utc_offset(self):
        result = _parse_time_string("2026-04-17T14:00:00+00:00")
        assert result == datetime(2026, 4, 17, 14, 0, 0, tzinfo=timezone.utc)

    def test_est_converts_to_utc(self):
        # 09:00 EST = UTC-5 = 14:00 UTC
        result = _parse_time_string("2026-04-17 09:00 EST")
        assert result == datetime(2026, 4, 17, 14, 0, 0, tzinfo=timezone.utc)

    def test_result_is_utc_aware(self):
        result = _parse_time_string("2026-04-17 14:00Z")
        assert result.tzinfo is not None
        assert result.utcoffset().total_seconds() == 0

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError):
            _parse_time_string("not a date")


class TestTimeWindowConstructor:
    def test_requires_aware_start(self):
        naive = datetime(2026, 1, 1)
        aware = datetime(2026, 1, 2, tzinfo=timezone.utc)
        with pytest.raises(ValueError):
            TimeWindow(naive, aware)

    def test_requires_aware_end(self):
        aware = datetime(2026, 1, 1, tzinfo=timezone.utc)
        naive = datetime(2026, 1, 2)
        with pytest.raises(ValueError):
            TimeWindow(aware, naive)

    def test_start_end_stored_as_utc(self):
        # Use a fixed UTC-5 offset (EST) so the test is unambiguous regardless of DST
        est = timezone(timedelta(hours=-5))
        start = datetime(2026, 4, 17, 9, 0, 0, tzinfo=est)    # 14:00 UTC
        end = datetime(2026, 4, 17, 15, 0, 0, tzinfo=est)      # 20:00 UTC
        w = TimeWindow(start, end)
        assert w.start_utc == datetime(2026, 4, 17, 14, 0, 0, tzinfo=timezone.utc)
        assert w.end_utc == datetime(2026, 4, 17, 20, 0, 0, tzinfo=timezone.utc)


class TestFromStrings:
    def test_explicit_utc(self):
        w = TimeWindow.from_strings("2026-04-17 00:00Z", "2026-04-18 00:00Z")
        assert w.duration_hours() == pytest.approx(24.0, abs=0.01)

    def test_est_to_utc(self):
        # EST is fixed UTC-5; 09:00 EST = 14:00 UTC, 15:00 EST = 20:00 UTC
        w = TimeWindow.from_strings("2026-04-17 09:00 EST", "2026-04-17 15:00 EST")
        assert w.start_utc == datetime(2026, 4, 17, 14, 0, 0, tzinfo=timezone.utc)
        assert w.end_utc == datetime(2026, 4, 17, 20, 0, 0, tzinfo=timezone.utc)


class TestClassMethods:
    def test_last_n_hours_duration(self):
        w = TimeWindow.last_n_hours(24.0)
        assert w.duration_hours() == pytest.approx(24.0, abs=0.01)

    def test_last_n_hours_ends_near_now(self):
        before = datetime.now(tz=timezone.utc)
        w = TimeWindow.last_n_hours(6.0)
        after = datetime.now(tz=timezone.utc)
        assert before <= w.end_utc <= after

    def test_last_n_days_duration(self):
        w = TimeWindow.last_n_days(7.0)
        assert w.duration_hours() == pytest.approx(168.0, abs=0.01)

    def test_next_n_hours_starts_near_now(self):
        before = datetime.now(tz=timezone.utc)
        w = TimeWindow.next_n_hours(6.0)
        after = datetime.now(tz=timezone.utc)
        assert before <= w.start_utc <= after

    def test_next_n_hours_duration(self):
        w = TimeWindow.next_n_hours(12.0)
        assert w.duration_hours() == pytest.approx(12.0, abs=0.01)


class TestDurationHours:
    def test_exact_24h(self):
        start = datetime(2026, 4, 17, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(2026, 4, 18, 0, 0, 0, tzinfo=timezone.utc)
        w = TimeWindow(start, end)
        assert w.duration_hours() == pytest.approx(24.0)

    def test_fractional_hours(self):
        start = datetime(2026, 4, 17, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(2026, 4, 17, 0, 30, 0, tzinfo=timezone.utc)
        w = TimeWindow(start, end)
        assert w.duration_hours() == pytest.approx(0.5)


class TestDisplay:
    def test_display_utc(self):
        start = datetime(2026, 4, 17, 14, 0, 0, tzinfo=timezone.utc)
        end = datetime(2026, 4, 17, 20, 0, 0, tzinfo=timezone.utc)
        w = TimeWindow(start, end)
        result = w.display("UTC")
        assert "2026-04-17 14:00:00" in result
        assert "2026-04-17 20:00:00" in result

    def test_display_does_not_change_internal_utc(self):
        start = datetime(2026, 4, 17, 14, 0, 0, tzinfo=timezone.utc)
        end = datetime(2026, 4, 17, 20, 0, 0, tzinfo=timezone.utc)
        w = TimeWindow(start, end)
        _ = w.display("US/Eastern")
        # Internal values must remain UTC-aware
        assert w.start_utc.utcoffset().total_seconds() == 0
        assert w.end_utc.utcoffset().total_seconds() == 0

    def test_display_eastern_shifts_hours(self):
        start = datetime(2026, 4, 17, 14, 0, 0, tzinfo=timezone.utc)
        end = datetime(2026, 4, 17, 20, 0, 0, tzinfo=timezone.utc)
        w = TimeWindow(start, end)
        result = w.display("US/Eastern")
        # EST is UTC-5; EDT is UTC-4.  Either way the hour should be < 14.
        assert "14:00" not in result  # should have shifted

    def test_display_invalid_tz_raises(self):
        w = TimeWindow.next_n_hours(1)
        with pytest.raises(ValueError):
            w.display("Not/ATimezone")
