"""
Canonical TLE dataclass shared across all Orbital Watch modules.

Every data source, parser, cache, and predictor speaks TLERecord.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True)
class TLERecord:
    """A parsed two-line element set with metadata.

    All numeric fields carry the types defined in the TLE standard.
    String fields are stripped of surrounding whitespace.
    """

    # --- identification ---
    name: str
    norad_id: int
    classification: str          # 'U' unclassified, 'C' classified, 'S' secret
    int_designator: str          # COSPAR international designator

    # --- epoch ---
    epoch: datetime              # UTC epoch derived from TLE epoch field

    # --- first-derivative of mean motion (rev/day²) ---
    mean_motion_dot: float

    # --- second-derivative of mean motion (rev/day³), scaled ×10^n ---
    mean_motion_ddot: float

    # --- BSTAR drag term (1/earth-radii) ---
    bstar: float

    # --- element set metadata ---
    element_set_no: int
    rev_at_epoch: int

    # --- Keplerian / mean elements ---
    inclination_deg: float
    raan_deg: float              # right ascension of ascending node
    eccentricity: float          # dimensionless (no leading decimal point in TLE)
    arg_perigee_deg: float
    mean_anomaly_deg: float
    mean_motion_rev_per_day: float

    # --- raw strings for round-trip fidelity ---
    line1: str
    line2: str

    # --- provenance ---
    source: str = "unknown"
    fetched_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))

    def tle_lines(self) -> tuple[str, str]:
        """Return the raw (line1, line2) tuple for downstream SGP4 use."""
        return self.line1, self.line2
