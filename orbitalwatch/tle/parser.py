"""
Parse raw two-line element strings into TLERecord instances.

Validates line checksums, extracts every standard TLE field with the
correct Python type, and raises descriptive errors on malformed input.
"""

from __future__ import annotations

from datetime import datetime, timezone

from .model import TLERecord


class TLEParseError(ValueError):
    """Raised when a TLE cannot be parsed or fails checksum validation."""


# ---------------------------------------------------------------------------
# Checksum
# ---------------------------------------------------------------------------

def _checksum(line: str) -> int:
    """Compute the modulo-10 checksum of a TLE line (digits + '-'=1)."""
    total = 0
    for ch in line[:68]:
        if ch.isdigit():
            total += int(ch)
        elif ch == "-":
            total += 1
    return total % 10


def _validate_checksum(line: str, line_no: int) -> None:
    if len(line) < 69:
        raise TLEParseError(f"Line {line_no} too short ({len(line)} chars, need ≥69)")
    expected = int(line[68])
    computed = _checksum(line)
    if computed != expected:
        raise TLEParseError(
            f"Line {line_no} checksum mismatch: computed {computed}, stored {expected}"
        )


# ---------------------------------------------------------------------------
# Epoch conversion
# ---------------------------------------------------------------------------

def _parse_epoch(epoch_str: str) -> datetime:
    """Convert TLE epoch 'YYDDD.DDDDDDDD' to a UTC datetime."""
    year2 = int(epoch_str[:2])
    # Per NORAD convention: 57-99 → 1957-1999, 00-56 → 2000-2056
    year = 2000 + year2 if year2 < 57 else 1900 + year2
    day_of_year = float(epoch_str[2:])
    # day_of_year is 1-based with fractional day
    from datetime import timedelta
    base = datetime(year, 1, 1, tzinfo=timezone.utc)
    return base + timedelta(days=day_of_year - 1)


# ---------------------------------------------------------------------------
# Decimal point expansion helpers
# ---------------------------------------------------------------------------

def _parse_decimal_assumed(s: str) -> float:
    """Parse a field like '0013746' that has an implied leading '0.'."""
    return float("0." + s.strip())


def _parse_exponential(s: str) -> float:
    """Parse a packed exponential like ' 00000-0' or '-11606-4' → float."""
    s = s.strip()
    # Format: [sign]mantissa[sign]exponent  e.g. '-11606-4' or ' 00000-0'
    # Find the exponent sign (last + or -)
    sign = 1.0
    if s[0] in "+-":
        sign = -1.0 if s[0] == "-" else 1.0
        s = s[1:]
    # The exponent marker is the last + or - in s
    for i in range(len(s) - 1, 0, -1):
        if s[i] in "+-":
            mantissa = float("0." + s[:i])
            exp = int(s[i:])
            return sign * mantissa * (10 ** exp)
    # Fallback: plain integer mantissa, no exponent
    return sign * float("0." + s)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse(name: str, line1: str, line2: str, source: str = "raw") -> TLERecord:
    """Parse a three-element TLE (name + two lines) into a TLERecord.

    Args:
        name:   Satellite name (line 0), any leading/trailing whitespace stripped.
        line1:  TLE line 1 (69 characters).
        line2:  TLE line 2 (69 characters).
        source: Provenance label stored on the record.

    Returns:
        A fully populated, immutable TLERecord.

    Raises:
        TLEParseError: On checksum failure or malformed fields.
    """
    name = name.strip()
    line1 = line1.strip()
    line2 = line2.strip()

    if not line1.startswith("1 "):
        raise TLEParseError(f"Line 1 does not start with '1 ': {line1!r}")
    if not line2.startswith("2 "):
        raise TLEParseError(f"Line 2 does not start with '2 ': {line2!r}")

    _validate_checksum(line1, 1)
    _validate_checksum(line2, 2)

    # --- Line 1 fields ---
    # Col indices (0-based, inclusive): per TLE standard
    norad_id        = int(line1[2:7])
    classification  = line1[7]
    int_designator  = line1[9:17].strip()
    epoch_str       = line1[18:32].strip()
    epoch           = _parse_epoch(epoch_str)

    mm_dot_raw = line1[33:43].strip()
    mean_motion_dot = float(mm_dot_raw)

    mm_ddot_raw = line1[44:52].strip()
    mean_motion_ddot = _parse_exponential(mm_ddot_raw)

    bstar_raw = line1[53:61].strip()
    bstar = _parse_exponential(bstar_raw)

    element_set_no = int(line1[64:68].strip())

    # --- Line 2 fields ---
    norad_id_l2 = int(line2[2:7])
    if norad_id_l2 != norad_id:
        raise TLEParseError(
            f"NORAD ID mismatch between lines: line1={norad_id}, line2={norad_id_l2}"
        )

    inclination_deg         = float(line2[8:16].strip())
    raan_deg                = float(line2[17:25].strip())
    eccentricity            = _parse_decimal_assumed(line2[26:33].strip())
    arg_perigee_deg         = float(line2[34:42].strip())
    mean_anomaly_deg        = float(line2[43:51].strip())
    mean_motion_rev_per_day = float(line2[52:63].strip())
    rev_at_epoch            = int(line2[63:68].strip())

    return TLERecord(
        name=name,
        norad_id=norad_id,
        classification=classification,
        int_designator=int_designator,
        epoch=epoch,
        mean_motion_dot=mean_motion_dot,
        mean_motion_ddot=mean_motion_ddot,
        bstar=bstar,
        element_set_no=element_set_no,
        rev_at_epoch=rev_at_epoch,
        inclination_deg=inclination_deg,
        raan_deg=raan_deg,
        eccentricity=eccentricity,
        arg_perigee_deg=arg_perigee_deg,
        mean_anomaly_deg=mean_anomaly_deg,
        mean_motion_rev_per_day=mean_motion_rev_per_day,
        line1=line1,
        line2=line2,
        source=source,
        fetched_at=datetime.now(tz=timezone.utc),
    )


def parse_many(text: str, source: str = "raw") -> list[TLERecord]:
    """Parse a block of TLE text containing one or more three-line sets.

    Blank lines between sets are ignored. Lines are grouped in triplets
    (name, line1, line2). Raises TLEParseError on the first bad set.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) % 3 != 0:
        raise TLEParseError(
            f"TLE text has {len(lines)} non-blank lines; expected a multiple of 3"
        )
    records = []
    for i in range(0, len(lines), 3):
        records.append(parse(lines[i], lines[i + 1], lines[i + 2], source=source))
    return records
