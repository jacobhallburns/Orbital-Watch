"""Tests for orbitalwatch.sources.cache (SQLite-backed TLE cache)."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from orbitalwatch.sources.cache import TLECache
from orbitalwatch.tle.model import TLERecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fix_checksum(line: str) -> str:
    total = sum(int(c) if c.isdigit() else (1 if c == "-" else 0) for c in line[:68])
    return line[:68] + str(total % 10)


def _make_record(norad_id: int, name: str = "TEST SAT") -> TLERecord:
    """Build a minimal but checksum-valid TLERecord for testing."""
    l1 = f"1 {norad_id:05d}U 98067A   24001.50000000  .00001234  00000-0  27416-4 0  999"
    l2 = f"2 {norad_id:05d}  51.6400 100.0000 0001234  90.0000 270.0000 15.50000000 1234"
    l1 = _fix_checksum(l1)
    l2 = _fix_checksum(l2)
    from orbitalwatch.tle.parser import parse
    return parse(name, l1, l2, source="test")


@pytest.fixture
def cache(tmp_path: Path) -> TLECache:
    """A fresh in-memory-equivalent cache backed by a temp file."""
    db = tmp_path / "test_cache.db"
    return TLECache(db_path=db, ttl_seconds=3600)


@pytest.fixture
def sample_record() -> TLERecord:
    return _make_record(25544, "ISS (ZARYA)")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPutAndGet:
    def test_get_returns_none_when_empty(self, cache):
        assert cache.get(25544) is None

    def test_put_then_get_returns_record(self, cache, sample_record):
        cache.put(sample_record)
        result = cache.get(25544)
        assert result is not None
        assert result.norad_id == 25544

    def test_get_returns_tle_record_type(self, cache, sample_record):
        cache.put(sample_record)
        result = cache.get(25544)
        assert isinstance(result, TLERecord)

    def test_name_preserved(self, cache, sample_record):
        cache.put(sample_record)
        result = cache.get(25544)
        assert result.name == "ISS (ZARYA)"

    def test_line1_line2_preserved(self, cache, sample_record):
        cache.put(sample_record)
        result = cache.get(25544)
        assert result.line1 == sample_record.line1
        assert result.line2 == sample_record.line2

    def test_source_preserved(self, cache, sample_record):
        cache.put(sample_record)
        result = cache.get(25544)
        assert result.source == "test"

    def test_put_overwrites_existing(self, cache):
        rec1 = _make_record(25544, "OLD NAME")
        rec2 = _make_record(25544, "NEW NAME")
        cache.put(rec1)
        cache.put(rec2)
        result = cache.get(25544)
        assert result.name == "NEW NAME"


class TestTTL:
    def test_fresh_entry_returned(self, tmp_path):
        cache = TLECache(db_path=tmp_path / "c.db", ttl_seconds=60)
        cache.put(_make_record(25544))
        assert cache.get(25544) is not None

    def test_expired_entry_returns_none(self, tmp_path):
        # TTL of 0 seconds means entries expire immediately
        cache = TLECache(db_path=tmp_path / "c.db", ttl_seconds=0)
        # ttl=0 means disabled (always fresh), so use ttl=1 and sleep
        cache2 = TLECache(db_path=tmp_path / "c2.db", ttl_seconds=1)
        cache2.put(_make_record(25544))
        time.sleep(1.1)
        assert cache2.get(25544) is None

    def test_zero_ttl_always_returns_cached(self, tmp_path):
        cache = TLECache(db_path=tmp_path / "c.db", ttl_seconds=0)
        cache.put(_make_record(25544))
        # Even after a tiny delay, ttl=0 means always fresh
        assert cache.get(25544) is not None


class TestPutMany:
    def test_put_many_stores_all(self, cache):
        records = [_make_record(i, f"SAT-{i}") for i in range(10000, 10005)]
        cache.put_many(records)
        assert cache.count() == 5
        for i in range(10000, 10005):
            assert cache.get(i) is not None

    def test_put_many_empty_list(self, cache):
        cache.put_many([])
        assert cache.count() == 0


class TestInvalidateAndClear:
    def test_invalidate_removes_entry(self, cache, sample_record):
        cache.put(sample_record)
        cache.invalidate(25544)
        assert cache.get(25544) is None

    def test_invalidate_nonexistent_is_noop(self, cache):
        cache.invalidate(99999)  # should not raise

    def test_clear_removes_all(self, cache):
        for i in range(10000, 10004):
            cache.put(_make_record(i))
        cache.clear()
        assert cache.count() == 0


class TestCount:
    def test_count_zero_on_empty(self, cache):
        assert cache.count() == 0

    def test_count_increments(self, cache):
        cache.put(_make_record(10001))
        assert cache.count() == 1
        cache.put(_make_record(10002))
        assert cache.count() == 2


class TestContextManager:
    def test_context_manager_closes_cleanly(self, tmp_path):
        db = tmp_path / "cm_test.db"
        with TLECache(db_path=db) as cache:
            cache.put(_make_record(25544))
        # After __exit__, the DB file should still exist
        assert db.exists()
