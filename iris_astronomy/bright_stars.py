#!/usr/bin/env python3
"""The naked-eye star catalogue, fetched once and cached on disk.

One copy, shared by everything that needs to know where the stars are: the live
skymap draws it, and the plate solver matches against it. Two copies would mean
two answers to "is that star up", and the whole point of the solver is that its
answer can be trusted.

Fetched from Vizier V/50 (the Bright Star Catalogue) down to V=6.0, which is
roughly the naked-eye limit and about 5000 stars. Cached to local/, never
re-fetched on a timer -- a scheduled job must not depend on a remote catalogue
service being up.
"""
import os
import sys
from pathlib import Path

import numpy as np

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

CACHE = "local/bright_stars.npy"
VMAG_LIMIT = 6.0


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def catalogue(root=None) -> np.ndarray:
    """(N, 3) array of RA deg, Dec deg, Vmag. Fetches on first call only."""
    root = Path(root) if root else _root()
    cache = root / CACHE
    if cache.exists():
        return np.load(cache)

    from astroquery.vizier import Vizier
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    v = Vizier(columns=["RAJ2000", "DEJ2000", "Vmag"], row_limit=-1)
    v.ROW_LIMIT = -1
    cat = v.query_constraints(catalog="V/50/catalog",
                              Vmag="<%.1f" % VMAG_LIMIT)[0]
    c = SkyCoord(ra=cat["RAJ2000"], dec=cat["DEJ2000"],
                 unit=(u.hourangle, u.deg))
    arr = np.column_stack([c.ra.deg.astype(float), c.dec.deg.astype(float),
                           np.asarray(cat["Vmag"], dtype=float)])
    arr = arr[np.isfinite(arr).all(axis=1)]
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, arr)
    return arr


def above_horizon(when, min_alt_deg=3.0, root=None):
    """Catalogue stars up at `when`, as (alt, az, vmag, ra, dec) arrays."""
    import astropy.units as u
    from astropy.coordinates import AltAz, EarthLocation, SkyCoord
    from astropy.time import Time
    from configs import config

    loc_cfg = config.data()["location"]
    site = EarthLocation.from_geodetic(float(loc_cfg["longitude"]) * u.deg,
                                       float(loc_cfg["latitude"]) * u.deg,
                                       float(loc_cfg.get("elevation", 0)) * u.m)
    cat = catalogue(root)
    sc = SkyCoord(ra=cat[:, 0] * u.deg, dec=cat[:, 1] * u.deg).transform_to(
        AltAz(obstime=Time(when), location=site))
    alt, az = sc.alt.deg, sc.az.deg
    up = alt > min_alt_deg
    return alt[up], az[up], cat[up, 2], cat[up, 0], cat[up, 1]


if __name__ == "__main__":
    arr = catalogue()
    print("catalogue: %d stars, V %.2f .. %.2f"
          % (len(arr), arr[:, 2].min(), arr[:, 2].max()))
