"""Tests for orbitalwatch.sources.registry — SourceRegistry with health tracking."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from orbitalwatch.sources.base import BaseSource, SourceFetchError
from orbitalwatch.sources.cache import TLECache
from orbitalwatch.sources.registry import AllSourcesFailedError, SourceRegistry, SourceStatus
from orbitalwatch.tle.model import TLERecord
from orbitalwatch.tle.parser import parse


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _fix_checksum(line: str) -> str:
    total = sum(int(c) if c.isdigit() else (1 if c == "-" else 0) for c in line[:68])
    return line[:68] + str(total % 10)


def _make_record(norad_id: int, name: str = "TEST SAT") -> TLERecord:
    l1 = f"1 {norad_id:05d}U 98067A   24001.50000000  .00001234  00000-0  27416-4 0  999"
    l2 = f"2 {norad_id:05d}  51.6400 100.0000 0001234  90.0000 270.0000 15.50000000 1234"
    return parse(name, _fix_checksum(l1), _fix_checksum(l2), source="test")


def _make_record_with_source(norad_id: int, name: str, source: str) -> TLERecord:
    l1 = f"1 {norad_id:05d}U 98067A   24001.50000000  .00001234  00000-0  27416-4 0  999"
    l2 = f"2 {norad_id:05d}  51.6400 100.0000 0001234  90.0000 270.0000 15.50000000 1234"
    return parse(name, _fix_checksum(l1), _fix_checksum(l2), source=source)


class _OkSource(BaseSource):
    """Always returns data successfully."""

    def __init__(self, norad_id: int = 25544, name_prefix: str = "good") -> None:
        self._norad_id = norad_id
        self._name_prefix = name_prefix
        self._record = _make_record_with_source(norad_id, f"{name_prefix.upper()}-SAT", name_prefix)

    @property
    def name(self) -> str:
        return self._name_prefix

    def fetch_by_norad_id(self, norad_id: int) -> TLERecord:
        if norad_id != self._norad_id:
            raise SourceFetchError(f"Unknown NORAD {norad_id}")
        return self._record

    def fetch_category(self, category: str) -> list[TLERecord]:
        return [self._record]

    def fetch_many(self, norad_ids: list[int]) -> list[TLERecord]:
        return [self._record] if self._norad_id in norad_ids else []


class _FailSource(BaseSource):
    """Always raises SourceFetchError."""

    def __init__(self, label: str = "broken") -> None:
        self._label = label

    @property
    def name(self) -> str:
        return self._label

    def fetch_by_norad_id(self, norad_id: int) -> TLERecord:
        raise SourceFetchError(f"{self._label}: always fails")

    def fetch_category(self, category: str) -> list[TLERecord]:
        raise SourceFetchError(f"{self._label}: always fails")

    def fetch_many(self, norad_ids: list[int]) -> list[TLERecord]:
        raise SourceFetchError(f"{self._label}: always fails")


@pytest.fixture
def cache(tmp_path: Path) -> TLECache:
    return TLECache(db_path=tmp_path / "reg_test.db", ttl_seconds=3600)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHealthySource:
    def test_fetch_by_norad_id_returns_data(self, cache):
        reg = SourceRegistry(cache=cache)
        reg.register(_OkSource(25544))
        record = reg.fetch_by_norad_id(25544)
        assert record.norad_id == 25544

    def test_fetch_category_returns_list(self, cache):
        reg = SourceRegistry(cache=cache)
        reg.register(_OkSource(25544))
        records = reg.fetch_category("stations")
        assert len(records) >= 1

    def test_source_health_shows_healthy(self, cache):
        reg = SourceRegistry(cache=cache)
        reg.register(_OkSource(25544))
        reg.fetch_by_norad_id(25544)
        statuses = reg.source_health()
        assert len(statuses) == 1
        assert statuses[0].healthy is True
        assert statuses[0].last_success is not None


class TestFailedSourceFallsBackToCache:
    def test_falls_back_to_cache_and_logs_warning(self, cache, caplog):
        import logging
        record = _make_record(25544)
        cache.put(record)

        reg = SourceRegistry(cache=cache)
        reg.register(_FailSource("broken"))

        with caplog.at_level(logging.WARNING):
            result = reg.fetch_by_norad_id(25544)

        assert result.norad_id == 25544
        assert any("broken" in msg for msg in caplog.messages)

    def test_failed_source_marked_unhealthy(self, cache):
        cache.put(_make_record(25544))
        reg = SourceRegistry(cache=cache)
        reg.register(_FailSource())
        reg.fetch_by_norad_id(25544)
        statuses = reg.source_health()
        assert statuses[0].healthy is False
        assert statuses[0].error_message is not None


class TestAllSourcesFailedWithNoCache:
    def test_raises_all_sources_failed(self, cache):
        reg = SourceRegistry(cache=cache)
        reg.register(_FailSource("src1"))
        reg.register(_FailSource("src2"))
        with pytest.raises(AllSourcesFailedError):
            reg.fetch_by_norad_id(99999)

    def test_error_contains_status_report(self, cache):
        reg = SourceRegistry(cache=cache)
        reg.register(_FailSource("primary"))
        reg.register(_FailSource("backup"))
        try:
            reg.fetch_by_norad_id(99999)
        except AllSourcesFailedError as exc:
            assert len(exc.statuses) == 2
            names = {s.name for s in exc.statuses}
            assert "primary" in names
            assert "backup" in names

    def test_category_raises_all_sources_failed(self, cache):
        reg = SourceRegistry(cache=cache)
        reg.register(_FailSource())
        with pytest.raises(AllSourcesFailedError):
            reg.fetch_category("stations")


class TestFallbackToNextSource:
    def test_failed_primary_uses_secondary(self, cache):
        reg = SourceRegistry(cache=cache)
        reg.register(_FailSource("primary"), priority=1)
        reg.register(_OkSource(25544, "backup"), priority=2)
        record = reg.fetch_by_norad_id(25544)
        assert record.norad_id == 25544

    def test_priority_ordering_respected(self, cache):
        reg = SourceRegistry(cache=cache)
        # priority=10 tried first; if it succeeds, priority=1 is never tried
        reg.register(_OkSource(25544, "secondary"), priority=10)
        reg.register(_OkSource(25544, "primary"), priority=1)
        record = reg.fetch_by_norad_id(25544)
        assert record.source == "primary"  # lower priority number wins


class TestSourceHealth:
    def test_health_returns_one_per_source(self, cache):
        reg = SourceRegistry(cache=cache)
        reg.register(_OkSource(1, "s1"), priority=1)
        reg.register(_OkSource(2, "s2"), priority=2)
        assert len(reg.source_health()) == 2

    def test_health_names_match_registered(self, cache):
        reg = SourceRegistry(cache=cache)
        reg.register(_OkSource(1, "alpha"))
        reg.register(_OkSource(2, "beta"))
        names = {s.name for s in reg.source_health()}
        assert names == {"alpha", "beta"}


class TestStrictMode:
    def test_strict_raises_on_first_failure(self, cache):
        reg = SourceRegistry(cache=cache, strict=True)
        reg.register(_FailSource("strict-src"))
        with pytest.raises(AllSourcesFailedError):
            reg.fetch_by_norad_id(25544)

    def test_strict_does_not_try_cache(self, cache):
        cache.put(_make_record(25544))
        reg = SourceRegistry(cache=cache, strict=True)
        reg.register(_FailSource("strict-src"))
        with pytest.raises(AllSourcesFailedError):
            reg.fetch_by_norad_id(25544)


class TestCustomSourceRegistration:
    def test_priority_1_tried_first(self, cache):
        reg = SourceRegistry(cache=cache)
        reg.register(_FailSource("fallback"), priority=99)
        reg.register(_OkSource(25544, "custom"), priority=1)
        record = reg.fetch_by_norad_id(25544)
        assert record.source == "custom"
