# Orbital Watch

Orbital Watch is a Python library for satellite pass prediction, TLE management, and orbital visualization. It fetches live two-line element sets from public sources like CelesTrak, caches them locally, propagates orbits using SGP4/SDP4, and computes accurate overhead pass windows for any ground location — all with a clean, analyst-friendly API and zero required authentication.

## Installation

```bash
pip install orbitalwatch
# or, for development:
pip install -e ".[dev]"
```

## Quick Start — passover query

```python
from orbitalwatch.sources.celestrak import CelesTrakSource
from orbitalwatch.predict.passover import PassoverPredictor
from orbitalwatch.geo.location import GroundStation

# Define your observing site
site = GroundStation(name="Austin TX", lat=30.2672, lon=-97.7431, elev_m=150)

# Grab the ISS TLE (NORAD ID 25544)
source = CelesTrakSource()
tle = source.fetch_by_norad_id(25544)

# Predict the next 5 passes above 10° elevation
predictor = PassoverPredictor(min_elevation_deg=10.0)
passes = predictor.next_passes(tle, site, count=5)

for p in passes:
    print(f"Rise {p.rise_utc}  Max El {p.max_elevation_deg:.1f}°  Set {p.set_utc}")
```

## Project Structure

```
orbitalwatch/
├── sources/       # TLE data fetchers (CelesTrak, Space-Track, …)
├── tle/           # TLE parsing, validation, dataclass
├── predict/       # Pass prediction, SGP4 propagation
├── geo/           # Ground stations, coordinate transforms
└── viz/           # Map and plot rendering (folium, matplotlib/cartopy)
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
