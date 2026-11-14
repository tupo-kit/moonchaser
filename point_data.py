"""Shared, memory-only store of per-point sky data. No VTK, no Tk.

The search window and the comparison page both need, for one point, the terrain horizon and the classified
Sun / Earth sky (sky.classify_sky) over a time interval. This module keeps those results so that no point is
computed twice, whichever window asked for it first:

  search  -> stores every point that matched   -> the comparison reuses it (no ephemeris, horizon or
                                                  classification work)
  compare -> stores every point it calculated  -> a later search or search column reuses it

Keys contain the point (rounded to 1e-6 deg), the interval (first epoch, last epoch, number of samples), the
ephemeris source and the eye height, so a different interval, step, source or eye height is simply a miss.
Nothing is written to disk. The sky store is limited by a memory budget (oldest entries are dropped first),
so a miss only means "calculate it again".

Dependencies: terrain <- sky <- compare <- point_data.
"""
import threading
from collections import OrderedDict

import numpy as np

import sky
import terrain
from compare import point_stats

DEFAULT_SOURCE = "Skyfield"   # the search always uses Skyfield
DEFAULT_BUDGET_MB = 768       # sky store size limit (about 35 KB per point for a month at 2 h steps)
MAX_HORIZONS = 20000          # horizon store size limit (about 12 KB each)


def _wrap_lon(lon):
    return (float(lon) + 180.0) % 360.0 - 180.0


def _nbytes(classified):
    return sum(value.nbytes for body in classified.values() for value in body.values()
               if isinstance(value, np.ndarray))


class PointDataCache:
    """Thread-safe store: classified sky per (point, interval, source) and terrain horizon per point."""

    def __init__(self, budget_mb=DEFAULT_BUDGET_MB, max_horizons=MAX_HORIZONS):
        self._lock = threading.Lock()
        self._sky = OrderedDict()       # key -> (classified, nbytes)
        self._horizons = OrderedDict()  # key -> horizon tuple of terrain.compute_horizon_360
        self._bytes = 0
        self.budget_bytes = int(budget_mb * 1024 * 1024)
        self.max_horizons = int(max_horizons)

    # -- keys
    @staticmethod
    def _point_key(lat, lon):
        return (round(float(lat), 6), round(_wrap_lon(lon), 6), float(terrain.EYE_HEIGHT_M),
                terrain.H, terrain.W)

    def _sky_key(self, lat, lon, epochs, source):
        return (self._point_key(lat, lon), epochs[0], epochs[-1], len(epochs), source)

    # -- sky
    def get_sky(self, lat, lon, epochs, source=DEFAULT_SOURCE):
        """The classified sky {"Sun": {...}, "Earth": {...}} of a point, or None. Treat the arrays as read-only."""
        if len(epochs) < 2:
            return None
        key = self._sky_key(lat, lon, epochs, source)
        with self._lock:
            item = self._sky.get(key)
            if item is None:
                return None
            self._sky.move_to_end(key)
            return item[0]

    def put_sky(self, lat, lon, epochs, source, classified):
        if len(epochs) < 2:
            return
        size = _nbytes(classified)
        if size > self.budget_bytes:
            return
        key = self._sky_key(lat, lon, epochs, source)
        with self._lock:
            old = self._sky.pop(key, None)
            if old is not None:
                self._bytes -= old[1]
            self._sky[key] = (classified, size)
            self._bytes += size
            while self._bytes > self.budget_bytes and len(self._sky) > 1:
                _, (_, freed) = self._sky.popitem(last=False)
                self._bytes -= freed

    # -- horizon
    def get_horizon(self, lat, lon):
        key = self._point_key(lat, lon)
        with self._lock:
            horizon = self._horizons.get(key)
            if horizon is not None:
                self._horizons.move_to_end(key)
            return horizon

    def put_horizon(self, lat, lon, horizon):
        key = self._point_key(lat, lon)
        with self._lock:
            self._horizons[key] = horizon
            self._horizons.move_to_end(key)
            while len(self._horizons) > self.max_horizons:
                self._horizons.popitem(last=False)

    # -- housekeeping
    def clear(self):
        with self._lock:
            self._sky.clear()
            self._horizons.clear()
            self._bytes = 0

    def info(self):
        """(points with stored sky data, megabytes used by them)."""
        with self._lock:
            return len(self._sky), self._bytes / (1024.0 * 1024.0)


CACHE = PointDataCache()   # the one store shared by the search window and the comparison


def sky_for(lat, lon, epochs, fetch, cache=None, source=DEFAULT_SOURCE):
    """Classified sky of one point: from the cache, otherwise ephemeris + terrain horizon + classification."""
    if cache is not None:
        hit = cache.get_sky(lat, lon, epochs, source)
        if hit is not None:
            return hit
    elevation_m = float(terrain.sample_dem_bilinear(lat, lon))
    positions = fetch(lon, lat, (elevation_m + terrain.EYE_HEIGHT_M) / 1000.0, epochs)
    horizon = cache.get_horizon(lat, lon) if cache is not None else None
    if horizon is None:
        horizon = terrain.compute_horizon_360(lat, lon)
        if cache is not None:
            cache.put_horizon(lat, lon, horizon)
    classified = sky.classify_sky(positions, horizon[0])
    if cache is not None:
        cache.put_sky(lat, lon, epochs, source, classified)
    return classified


def flatten_stats(stats):
    """compare.point_stats() result as flat search-row values (all keys always present)."""
    if stats is None:
        return {}
    out = {}
    for name in ("sun", "earth"):
        body = stats[name.capitalize()]
        out.update({
            f"{name}_center_pct": 100.0 * body["fraction"],
            f"{name}_per_lunar_day_s": body["per_lunar_day_s"],
            f"{name}_mean_period_s": body["mean_period_s"],
            f"{name}_period_partial": body["period_partial"],
        })
        for axis in ("az", "el"):
            pair = body[axis]
            out[f"{name}_{axis}_mean"] = pair[0] if pair else None
            out[f"{name}_{axis}_sd"] = pair[1] if pair and np.isfinite(pair[1]) else None
    lunar = stats.get("lunar")
    out["sun_lunar_mean"] = lunar[0] if lunar else None
    out["sun_lunar_std"] = lunar[1] if lunar else None
    out["lunar_days"] = lunar[2] if lunar else None
    return out


def metrics_from_sky(epochs, classified, source=DEFAULT_SOURCE, only_above=True):
    """Everything the comparison tables show for one point, as flat values for a search row."""
    data = {f"Sun {source}": classified["Sun"], f"Earth {source}": classified["Earth"]}
    return flatten_stats(point_stats(epochs, data, source, only_above))
