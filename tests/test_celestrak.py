"""Tests for orbitalwatch.sources.celestrak using mocked HTTP responses."""

from __future__ import annotations

import pytest
import responses as resp_mock

from orbitalwatch.sources.celestrak import CelesTrakSource, CelesTrakError, _GP_URL
from orbitalwatch.tle.model import TLERecord


# ---------------------------------------------------------------------------
# Helpers — valid 3-line TLE text blocks
# ---------------------------------------------------------------------------

def _fix_checksum(line: str) -> str:
    total = sum(int(c) if c.isdigit() else (1 if c == "-" else 0) for c in line[:68])
    return line[:68] + str(total % 10)


def _make_tle_block(name: str, norad_id: int) -> str:
    """Build a minimal but checksum-valid 3-line TLE block."""
    l1 = f"1 {norad_id:05d}U 98067A   24001.50000000  .00001234  00000-0  27416-4 0  999"
    l2 = f"2 {norad_id:05d}  51.6400 100.0000 0001234  90.0000 270.0000 15.50000000 1234"
    return f"{name}\n{_fix_checksum(l1)}\n{_fix_checksum(l2)}"


_BLOCK_ISS = _make_tle_block("ISS (ZARYA)", 25544)
_BLOCK_HST = _make_tle_block("HST", 20580)
_TWO_SAT_BODY = _BLOCK_ISS + "\n" + _BLOCK_HST


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFetchCategory:
    @resp_mock.activate
    def test_returns_tle_records(self):
        resp_mock.add(resp_mock.GET, _GP_URL, body=_TWO_SAT_BODY, status=200)
        records = CelesTrakSource().fetch_category("stations")
        assert len(records) == 2
        assert all(isinstance(r, TLERecord) for r in records)

    @resp_mock.activate
    def test_norad_ids_correct(self):
        resp_mock.add(resp_mock.GET, _GP_URL, body=_TWO_SAT_BODY, status=200)
        records = CelesTrakSource().fetch_category("stations")
        assert {r.norad_id for r in records} == {25544, 20580}

    @resp_mock.activate
    def test_source_label_is_celestrak(self):
        resp_mock.add(resp_mock.GET, _GP_URL, body=_BLOCK_ISS, status=200)
        records = CelesTrakSource().fetch_category("stations")
        assert records[0].source == "celestrak"

    @resp_mock.activate
    def test_http_error_raises_celestrak_error(self):
        resp_mock.add(resp_mock.GET, _GP_URL, status=503)
        with pytest.raises(CelesTrakError):
            CelesTrakSource().fetch_category("stations")

    @resp_mock.activate
    def test_empty_body_returns_empty(self):
        resp_mock.add(resp_mock.GET, _GP_URL, body="", status=200)
        records = CelesTrakSource().fetch_category("stations")
        assert records == []

    @resp_mock.activate
    def test_group_param_sent(self):
        resp_mock.add(resp_mock.GET, _GP_URL, body=_TWO_SAT_BODY, status=200)
        CelesTrakSource().fetch_category("stations")
        assert "GROUP=stations" in resp_mock.calls[0].request.url

    @resp_mock.activate
    def test_format_tle_param_sent(self):
        resp_mock.add(resp_mock.GET, _GP_URL, body=_BLOCK_ISS, status=200)
        CelesTrakSource().fetch_category("stations")
        assert "FORMAT=TLE" in resp_mock.calls[0].request.url


class TestFetchActive:
    @resp_mock.activate
    def test_delegates_to_active_category(self):
        resp_mock.add(resp_mock.GET, _GP_URL, body=_TWO_SAT_BODY, status=200)
        records = CelesTrakSource().fetch_active()
        assert len(records) == 2
        assert "GROUP=active" in resp_mock.calls[0].request.url


class TestFetchByNoradId:
    @resp_mock.activate
    def test_returns_single_record(self):
        resp_mock.add(resp_mock.GET, _GP_URL, body=_BLOCK_ISS, status=200)
        rec = CelesTrakSource().fetch_by_norad_id(25544)
        assert isinstance(rec, TLERecord)
        assert rec.norad_id == 25544

    @resp_mock.activate
    def test_empty_response_raises(self):
        resp_mock.add(resp_mock.GET, _GP_URL, body="", status=200)
        with pytest.raises(CelesTrakError, match="No data"):
            CelesTrakSource().fetch_by_norad_id(99999)

    @resp_mock.activate
    def test_catnr_param_sent(self):
        resp_mock.add(resp_mock.GET, _GP_URL, body=_BLOCK_ISS, status=200)
        CelesTrakSource().fetch_by_norad_id(25544)
        assert "CATNR=25544" in resp_mock.calls[0].request.url

    @resp_mock.activate
    def test_http_error_raises_celestrak_error(self):
        resp_mock.add(resp_mock.GET, _GP_URL, status=404)
        with pytest.raises(CelesTrakError):
            CelesTrakSource().fetch_by_norad_id(25544)
