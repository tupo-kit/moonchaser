"""Coarse, cancellable lunar surface search for terrain-aware visibility.

Hosted by the Search panel of desktop_ui.py. Skyfield is used locally. Percentages are estimated by
trapezoidal integration of sampled visibility (partly visible counts; Earth
occulting the Sun is included). An optional sampled maximum slope filters sites.
Minimum and maximum Sun / Earth visibility limit the results; for searches longer than 2 days the
average Sun visibility per lunar day (and its standard deviation) can be added to each point.

Shared data: every matching point is stored in point_data.CACHE (horizon and classified sky), so the comparison
page opens without recomputing it. The optional columns of the results table (above-horizon time, azimuth /
elevation statistics, lunar-day statistics, maximum slope) use exactly the statistics of compare.py; they never
influence the search itself and are calculated afterwards, in the background, from the stored data.
"""
import atexit
import csv
import colorsys
import multiprocessing
import os
import queue
import sys
import threading
import time
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from concurrent.futures.process import BrokenProcessPool
from datetime import timedelta
from functools import partial

import numpy as np
import point_data
import sky
import terrain
from compare import (LUNAR_DAY_DAYS, MAX_POINTS, format_duration, lunar_day_stats, max_slope_cached, next_sort,
                     normalize_point, same_point, visibility_percent)  # visibility_percent, lunar_day_stats: re-exported
MAX_GRID_POINTS = 100_000
MAX_DISPLAY_RESULTS = 500
LONG_INTERVAL_DAYS = 2.0  # lunar-day statistics are offered above this interval length
_elevation_stats_lock = threading.Lock()
_elevation_stats_source = None
_elevation_stats_result = None


def global_elevation_stats(cancel=None):
    """Return exact DEM minimum and median in metres using chunked passes.

    Histogram bins identify the middle value(s), then only those bins are
    collected and partitioned. The result is reused while the DEM is loaded.
    """
    global _elevation_stats_source, _elevation_stats_result
    dem = terrain.dem_km
    if dem is None or not dem.size:
        raise RuntimeError('The Moon elevation map has not been loaded.')
    with _elevation_stats_lock:
        if _elevation_stats_source is dem:
            return _elevation_stats_result
        minimum, maximum = float(np.min(dem)), float(np.max(dem))
        if not np.isfinite([minimum, maximum]).all():
            raise ValueError('The elevation map contains invalid values.')
        if minimum == maximum:
            result = (minimum * 1000, minimum * 1000)
        else:
            edges = np.linspace(minimum, maximum, 4097, dtype=np.float64)
            counts = np.zeros(4096, dtype=np.int64)
            for first in range(0, dem.shape[0], 64):
                if cancel is not None and cancel.is_set():
                    return None
                block = np.asarray(dem[first:first + 64], dtype=np.float64)
                counts += np.histogram(block, bins=edges)[0]
            cumulative = np.cumsum(counts)
            ranks = ((dem.size - 1) // 2, dem.size // 2)
            bins = [int(np.searchsorted(cumulative, rank, side='right')) for rank in ranks]
            buckets = {index: [] for index in set(bins)}
            for first in range(0, dem.shape[0], 64):
                if cancel is not None and cancel.is_set():
                    return None
                block = np.asarray(dem[first:first + 64], dtype=np.float64)
                for index, values in buckets.items():
                    mask = (block >= edges[index]) & ((block <= edges[index + 1]) if index == 4095 else (block < edges[index + 1]))
                    if np.any(mask):
                        values.append(block[mask])
            middle = []
            for index in set(bins):
                values = np.concatenate(buckets[index])
                before = int(cumulative[index - 1]) if index else 0
                local_ranks = [rank - before for rank, b in zip(ranks, bins) if b == index]
                values.partition(local_ranks)
                middle.extend(float(values[k]) for k in local_ranks)
            result = (minimum * 1000.0, sum(middle) * 500.0)
        _elevation_stats_source, _elevation_stats_result = dem, result
        return result


def cluster_color(index):
    """Stable, well-spaced hues for cluster markers and outlines."""
    hue = (0.02 + int(index) * 0.6180339887498949) % 1.0
    return colorsys.hsv_to_rgb(hue, 0.78, 0.96)


def cluster_text_color(index):
    """A darker version of the cluster hue for light list backgrounds."""
    hue = (0.02 + int(index) * 0.6180339887498949) % 1.0
    return '#%02x%02x%02x' % tuple(round(x * 255) for x in colorsys.hsv_to_rgb(hue, 0.85, 0.60))


def grid_points(lat_min, lat_max, lon_min, lon_max, spacing):
    values = [float(lat_min), float(lat_max), float(lon_min), float(lon_max), float(spacing)]
    if not np.isfinite(values).all():
        raise ValueError('Grid values must be finite.')
    lat_min, lat_max, lon_min, lon_max, spacing = values
    if not (-90 <= lat_min <= lat_max <= 90):
        raise ValueError('Latitude range must be ordered and within -90 to 90 degrees.')
    if lon_min > lon_max:
        raise ValueError('The first longitude must not exceed the second.')
    # Spans over 360 degrees describe the whole Moon; do not duplicate it.
    lon_max = min(lon_max, lon_min + 360.0)
    if not 0.1 <= spacing <= 180:
        raise ValueError('Grid spacing must be between 0.1 and 180 degrees.')
    count_lat = int(np.floor((lat_max - lat_min) / spacing + 1e-9)) + 1
    count_lon = int(np.floor((lon_max - lon_min) / spacing + 1e-9)) + 1
    if count_lat * count_lon > MAX_GRID_POINTS + count_lon * 2:
        raise ValueError(f'Grid is too large; increase spacing or narrow the region (maximum {MAX_GRID_POINTS:,} points).')
    lats = lat_min + np.arange(count_lat) * spacing
    lons = lon_min + np.arange(count_lon) * spacing
    if len(lons) > 1 and np.isclose(lons[-1] - lons[0], 360.0, atol=1e-9, rtol=0):
        lons = lons[:-1]
    points = []
    for lat in lats:
        if np.isclose(abs(lat), 90.0, atol=1e-9, rtol=0):
            points.append((float(lat), float((lons[0] + 180.0) % 360.0 - 180.0)))
        else:
            points.extend((float(lat), float((lon + 180.0) % 360.0 - 180.0)) for lon in lons)
    if len(points) > MAX_GRID_POINTS:
        raise ValueError(f'Grid is too large; maximum {MAX_GRID_POINTS:,} points.')
    return points


# visibility_percent() and lunar_day_stats() live in compare.py (shared with the comparison tables).


def lunar_passes(lunar, limits):
    """True if (mean, SD, days) lies within (min avg, max avg, min SD, max SD); None means no limit."""
    low_avg, high_avg, low_sd, high_sd = limits
    mean, std = lunar[0], lunar[1]
    return ((low_avg is None or mean >= low_avg) and (high_avg is None or mean <= high_avg)
            and (low_sd is None or std >= low_sd) and (high_sd is None or std <= high_sd))


def optimistic_horizon_floor(observer_elevation_m, min_terrain_m, max_dist_km=250.0):
    """A conservative lower bound on every computed terrain-horizon angle.

    The real horizon includes a terrain sample at max_dist_km in every
    direction. Replacing its height with the minimum of the entire DEM (plus
    a small margin) cannot raise that sample's angle. This accounts for lunar
    curvature and an observer above the terrain; zero degrees would not be a
    safe bound for high locations or nearby valleys.
    """
    theta = max_dist_km * 1000.0 / terrain.MOON_RADIUS_M
    ground_radius = terrain.MOON_RADIUS_M + min_terrain_m - 1.0
    observer_radius = terrain.MOON_RADIUS_M + observer_elevation_m + terrain.EYE_HEIGHT_M
    angle = np.degrees(np.arctan2(ground_radius * np.cos(theta) - observer_radius,
                                  ground_radius * np.sin(theta)))
    return float(angle) - 1e-6


def can_meet_thresholds_without_horizon(epochs, positions, floor_angle, min_sun, min_earth):
    """Reject only when even an optimistic disk/horizon cannot meet a limit.

    The whole disk is allowed above the global lower-bound horizon. Sun/Earth
    mutual eclipses are ignored here, so this can only overestimate visibility
    compared with the detailed sampled terrain and eclipse check.
    """
    possible_to_pass = True
    for body, threshold in (("Sun", min_sun), ("Earth", min_earth)):
        elevation = np.asarray(positions[body]["el"], dtype=float)
        radius = np.asarray(positions[body]["radius"], dtype=float)
        if elevation.shape != (len(epochs),) or radius.shape != (len(epochs),):
            raise ValueError(f"{body} ephemeris length does not match the time samples.")
        if not np.isfinite(elevation).all() or not np.isfinite(radius).all() or np.any(radius < 0):
            raise ValueError(f"{body} ephemeris contains invalid values.")
        possible = elevation + radius >= floor_angle
        if visibility_percent(epochs, possible) + 1e-9 < threshold:
            possible_to_pass = False
    return possible_to_pass


def search(points, epochs, min_sun, min_earth, fetch, cancel=None, progress=None,
           cache=None, cache_source=point_data.DEFAULT_SOURCE):
    """Return passing sites; stop between sites if cancel is set.

    `fetch(lon, lat, altitude_km, epochs)` returns Sun and Earth Skyfield
    arrays. Errors abort rather than silently classifying failed sites as dark.
    """
    thresholds = np.asarray([min_sun, min_earth], dtype=float)
    if not np.isfinite(thresholds).all() or np.any(thresholds < 0) or np.any(thresholds > 100):
        raise ValueError('Visibility thresholds must be between 0 and 100 percent.')
    if len(epochs) < 2 or epochs[-1] <= epochs[0]:
        raise ValueError('End time must be later than start time.')
    if terrain.dem_km is None:
        raise RuntimeError('The Moon elevation map has not been loaded.')
    min_terrain_m = float(np.min(terrain.dem_km)) * 1000.0
    if not np.isfinite(min_terrain_m):
        raise ValueError('The elevation map contains invalid values.')
    results = []
    start = time.monotonic()
    completed = 0
    filtered = 0
    for index, (lat, lon) in enumerate(points):
        if cancel is not None and cancel.is_set():
            break
        if progress:
            elapsed = time.monotonic() - start
            progress(index, len(points), len(results), elapsed, lat, lon, filtered)
        try:
            elevation_m = float(terrain.sample_dem_bilinear(lat, lon))
            altitude = (elevation_m + terrain.EYE_HEIGHT_M) / 1000.0
            positions = fetch(lon, lat, altitude, epochs)
            if cancel is not None and cancel.is_set():
                break
            floor_angle = optimistic_horizon_floor(elevation_m, min_terrain_m)
            possible = can_meet_thresholds_without_horizon(
                epochs, positions, floor_angle, min_sun, min_earth)
            if not possible:
                completed += 1
                filtered += 1
                continue
            horizon_full = terrain.compute_horizon_360(lat, lon)
            if cancel is not None and cancel.is_set():
                break
            classified = sky.classify_sky(positions, horizon_full[0])
            sun = visibility_percent(epochs, classified['Sun']['visible'])
            earth = visibility_percent(epochs, classified['Earth']['visible'])
        except Exception as error:
            raise RuntimeError(f'Search failed at latitude {lat:.3f}, longitude {lon:.3f}: {error}') from error
        completed += 1
        if sun >= min_sun and earth >= min_earth:
            results.append({'lat': lat, 'lon': lon, 'sun_percent': sun, 'earth_percent': earth})
            if cache is not None:
                cache.put_horizon(lat, lon, horizon_full)
                cache.put_sky(lat, lon, epochs, cache_source, classified)
    if progress:
        progress(completed,
                 len(points), len(results), time.monotonic() - start, None, None, filtered)
    return sorted(results, key=lambda p: (min(p['sun_percent'] - min_sun, p['earth_percent'] - min_earth),
                                           p['sun_percent'] + p['earth_percent']), reverse=True)


def grid_cells(lat_min, lat_max, lon_min, lon_max, spacing):
    """Cover the chosen lat/lon rectangle with non-overlapping angular cells."""
    vals = np.asarray([lat_min, lat_max, lon_min, lon_max, spacing], dtype=float)
    if not np.isfinite(vals).all():
        raise ValueError("Cell bounds and spacing must be finite.")
    lat_min, lat_max, lon_min, lon_max, spacing = vals
    if not (-90 <= lat_min < lat_max <= 90):
        raise ValueError("Latitude bounds must be increasing and between -90 and 90.")
    if lon_min >= lon_max:
        raise ValueError("The first longitude must be less than the second.")
    # Keep the user's continuous convention across the seam, at most one turn.
    lon_max = min(lon_max, lon_min + 360.0)
    if not 0.1 <= spacing <= 180:
        raise ValueError("Cell spacing must be between 0.1 and 180 degrees.")
    nlat = int(np.ceil((lat_max - lat_min) / spacing - 1e-12))
    nlon = int(np.ceil((lon_max - lon_min) / spacing - 1e-12))
    if nlat * nlon > MAX_GRID_POINTS:
        raise ValueError(f"Too many cells; choose at most {MAX_GRID_POINTS:,}.")
    return [(float(lat_min + i * spacing), float(min(lat_min + (i + 1) * spacing, lat_max)),
             float(lon_min + j * spacing), float(min(lon_min + (j + 1) * spacing, lon_max)))
            for i in range(nlat) for j in range(nlon)]


def subdivide_cluster(sectors, bounds, spacing):
    """Refine only member sectors, clipped to the user's chosen bounds."""
    lat0, lat1, lon0, lon1 = bounds
    cells = []
    for a, b, c, d in sectors:
        clipped = (max(a, lat0), min(b, lat1), max(c, lon0), min(d, lon1))
        if clipped[0] >= clipped[1] or clipped[2] >= clipped[3]:
            continue
        subcells = grid_cells(*clipped, spacing)
        if len(cells) + len(subcells) > MAX_GRID_POINTS:
            raise ValueError(f'Too many cells in this cluster; choose at most {MAX_GRID_POINTS:,}.')
        cells.extend(subcells)
    if not cells:
        raise ValueError('The chosen range does not intersect the selected cluster.')
    return cells


def cell_sample_points(cell, side):
    """Center first, then an evenly spaced grid from near center outward."""
    lat0, lat1, lon0, lon1 = cell
    centre = ((lat0 + lat1) / 2, (lon0 + lon1) / 2)
    candidates = [centre] + [(float(lat), float(lon))
                             for lat in np.linspace(lat0, lat1, side)
                             for lon in np.linspace(lon0, lon1, side)]
    def normalize(point):
        lat, lon = point
        if np.isclose(abs(lat), 90.0, atol=1e-9, rtol=0):
            lon = 0.0
        else:
            lon = (lon + 180.0) % 360.0 - 180.0
        return (round(lat, 10), round(lon, 10))
    # Keep center first, removing duplicate center, pole and seam points.
    unique = list(dict.fromkeys(map(normalize, candidates)))
    center = normalize(centre)
    rest = sorted(unique[1:], key=lambda p: (p[0] - center[0]) ** 2 +
                  (((p[1] - center[1] + 180) % 360) - 180) ** 2)
    return [unique[0], *rest]


def interval_union(intervals):
    """Merge touching coordinate intervals for compact range reporting."""
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1] + 1e-8:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def optimistic_cell_ranges(cells, epochs, min_sun, min_earth, fetch, cancel=None, progress=None, horizon_result=None):
    """Screen broad 4-by-4 cell blocks using optimistic visibility estimates.

    The center is allowed the full angular width of the block, a generous
    parallax margin and a median-height observer looking toward the lowest
    terrain. This is approximate and can exclude a higher valid observer.
    """
    heights = global_elevation_stats(cancel)
    if heights is None:
        return [], {}, [], []
    min_height, median_height = heights
    floor = optimistic_horizon_floor(median_height, min_height)
    if horizon_result:
        horizon_result(min_height, median_height, floor)
    lat_starts = sorted({cell[0] for cell in cells})
    lon_starts = sorted({cell[2] for cell in cells})
    lat_index = {x: i for i, x in enumerate(lat_starts)}
    lon_index = {x: i for i, x in enumerate(lon_starts)}
    blocks = {}
    for cell in cells:
        key = (lat_index[cell[0]] // 4, lon_index[cell[2]] // 4)
        blocks.setdefault(key, []).append(cell)
    eligible, cached = [], {}
    cache_limit = max(1, 500_000 // len(epochs))
    total = len(cells)
    completed = 0
    for block in blocks.values():
        if cancel is not None and cancel.is_set():
            break
        bounds = (min(c[0] for c in block), max(c[1] for c in block),
                  min(c[2] for c in block), max(c[3] for c in block))
        center = cell_sample_points(bounds, 3)[0]
        if progress:
            progress(completed, total, len(eligible), center)
        height = float(terrain.sample_dem_bilinear(*center))
        try:
            positions = fetch(center[1], center[0], (height + terrain.EYE_HEIGHT_M) / 1000.0, epochs)
        except Exception as error:
            raise RuntimeError(f"Range estimate failed at latitude {center[0]:.3f}, longitude {center[1]:.3f}: {error}") from error
        if cancel is not None and cancel.is_set():
            break
        margin = min(180.0, (bounds[1] - bounds[0] + bounds[3] - bounds[2]) / 2.0 + 2.0)
        possible = True
        for body, threshold in (("Sun", min_sun), ("Earth", min_earth)):
            el = np.asarray(positions[body]["el"], dtype=float)
            radius = np.asarray(positions[body]["radius"], dtype=float)
            if el.shape != (len(epochs),) or radius.shape != (len(epochs),) or not np.isfinite(el).all() or not np.isfinite(radius).all() or np.any(radius < 0):
                raise ValueError(f"Invalid {body} ephemeris in optimistic range screen.")
            if visibility_percent(epochs, el + radius + margin >= floor) + 1e-9 < threshold:
                possible = False
        completed += len(block)
        if possible:
            eligible.extend(block)
            if len(block) == 1 and len(cached) < cache_limit:
                cached[center] = positions
    lat_ranges = interval_union((cell[0], cell[1]) for cell in eligible)
    lon_ranges = interval_union((cell[2], cell[3]) for cell in eligible)
    if progress:
        progress(completed, total, len(eligible), None)
    return eligible, cached, lat_ranges, lon_ranges


def cluster_results(results, wrap_longitude=False, longitude_bounds=None):
    """Group edge-adjacent passing cells; this does not fill in untested area."""
    def r(x):
        return round(float(x), 8)
    # Older point-only results remain individually selectable.
    cells, singles = [], []
    for row in results:
        (cells if all(k in row for k in ('cell_lat_min', 'cell_lat_max', 'cell_lon_min', 'cell_lon_max'))
         else singles).append(row)
    south = {(r(x['cell_lat_min']), r(x['cell_lon_min']), r(x['cell_lon_max'])): i for i, x in enumerate(cells)}
    west = {(r(x['cell_lat_min']), r(x['cell_lat_max']), r(x['cell_lon_min'])): i for i, x in enumerate(cells)}
    seam_start, seam_end = longitude_bounds if longitude_bounds is not None else (-180.0, 180.0)
    parent = list(range(len(cells)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def join(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a
    for i, row in enumerate(cells):
        la, lb = r(row['cell_lat_min']), r(row['cell_lat_max'])
        lo, hi = r(row['cell_lon_min']), r(row['cell_lon_max'])
        neighbour = south.get((lb, lo, hi))
        if neighbour is not None:
            join(i, neighbour)
        neighbour = west.get((la, lb, hi))
        if neighbour is None and wrap_longitude and np.isclose(hi, seam_end, atol=1e-8, rtol=0):
            neighbour = west.get((la, lb, r(seam_start)))
        if neighbour is not None:
            join(i, neighbour)
    groups = {}
    for i, row in enumerate(cells):
        groups.setdefault(find(i), []).append(row)
    return list(groups.values()) + [[x] for x in singles]


def search_cells(cells, epochs, min_sun, min_earth, fetch, samples_per_side=3,
                 screen_horizon=2.0, screen_mode="Pessimistic center", cancel=None, progress=None,
                 use_ranges=True, range_progress=None, range_result=None, max_slope=None, slope_radius_km=1.0,
                 horizon_result=None, max_sun=100.0, max_earth=100.0, lunar_stats=False, lunar_filter=None,
                 cache=None, cache_source=point_data.DEFAULT_SOURCE, workers=None):
    """Return at most one terrain-checked match per cell.

    workers: None or 1 = this thread only; 0 = one worker process per core (minus two); n > 1 = n processes.
    The processes always use Skyfield directly (the `fetch` argument is not used by them).

    With a cache (point_data.PointDataCache) a point stored by an earlier search or by the comparison is not
    fetched, horizon-scanned or classified again, and every matching point is stored for later use.

    The center is checked against real terrain first. If it fails, the optional
    pessimistic horizon screen decides whether to try the remaining samples.
    This heuristic can miss valid locations elsewhere in the cell.
    """
    if terrain.dem_km is None:
        raise RuntimeError("The Moon elevation map has not been loaded.")
    if not (2 <= samples_per_side <= 9):
        raise ValueError("Samples per side must be an integer from 2 to 9.")
    if not np.isfinite(screen_horizon) or not -90 <= screen_horizon <= 90:
        raise ValueError("Screen horizon must be between -90 and 90 degrees.")
    if screen_mode not in ("Pessimistic center", "All cells"):
        raise ValueError("Choose a valid cell screening mode.")
    if not np.isfinite([min_sun, min_earth]).all() or not (0 <= min_sun <= 100 and 0 <= min_earth <= 100):
        raise ValueError("Visibility thresholds must be between 0 and 100 percent.")
    if not np.isfinite([max_sun, max_earth]).all() or not (min_sun <= max_sun <= 100 and min_earth <= max_earth <= 100):
        raise ValueError("Maximum visibility must be between the minimum and 100 percent.")
    if lunar_filter is not None:
        low_avg, high_avg, low_sd, high_sd = lunar_filter
        for low, high, name in ((low_avg, high_avg, 'average Sun %'), (low_sd, high_sd, 'standard deviation')):
            if any(v is not None and (not np.isfinite(v) or not 0 <= v <= 100) for v in (low, high)):
                raise ValueError(f'The {name} limits must be between 0 and 100.')
            if low is not None and high is not None and low > high:
                raise ValueError(f'The minimum {name} must not exceed the maximum.')
    if max_slope is not None and (not np.isfinite(max_slope) or not 0 <= max_slope <= 90):
        raise ValueError('Maximum slope must be between 0 and 90 degrees.')
    if not np.isfinite(slope_radius_km) or not 0 <= slope_radius_km <= 20:
        raise ValueError('Slope radius must be between 0 and 20 km.')
    if len(epochs) < 2 or epochs[-1] <= epochs[0]:
        raise ValueError("End time must be later than start time.")
    min_terrain = float(np.min(terrain.dem_km)) * 1000.0
    if not np.isfinite(min_terrain):
        raise ValueError("The elevation map contains invalid values.")
    if workers is not None and workers != 1:
        return search_cells_parallel(cells, epochs, min_sun, min_earth, samples_per_side, screen_horizon, screen_mode,
                                     cancel, progress, use_ranges, range_progress, range_result, max_slope,
                                     slope_radius_km, horizon_result, max_sun, max_earth, lunar_stats, lunar_filter,
                                     cache, cache_source, workers)
    start = time.monotonic()
    total_cells = len(cells)
    precomputed = {}
    if use_ranges:
        eligible, precomputed, lat_ranges, lon_ranges = optimistic_cell_ranges(
            cells, epochs, min_sun, min_earth, fetch, cancel, range_progress, horizon_result)
        if cancel is not None and cancel.is_set():
            return []
        if range_result:
            range_result(lat_ranges, lon_ranges, len(eligible), total_cells)
        cells = eligible
    initial_filtered = total_cells - len(cells)
    results, checked = [], {}
    completed = filtered = initial_filtered
    for cell in cells:
        if cancel is not None and cancel.is_set():
            break
        points = cell_sample_points(cell, samples_per_side)
        center = points[0]
        if progress:
            progress(completed, total_cells, len(results), time.monotonic() - start, *center, filtered)
        try:
            center_positions = precomputed.pop(center, None)
            for point in points:
                if cancel is not None and cancel.is_set():
                    break
                if point not in checked:
                    elevation = float(terrain.sample_dem_bilinear(*point))
                    stored = cache.get_sky(point[0], point[1], epochs, cache_source) if cache is not None else None
                    # A stored classified sky also holds the Sun / Earth az, el and radius of a plain ephemeris result.
                    pos = (center_positions if point == center and center_positions is not None else
                           stored if stored is not None else
                           fetch(point[1], point[0], (elevation + terrain.EYE_HEIGHT_M) / 1000.0, epochs))
                    if point == center:
                        center_positions = pos
                    if cancel is not None and cancel.is_set():
                        break
                    floor = optimistic_horizon_floor(elevation, min_terrain)
                    if ((point == center and screen_mode == 'Pessimistic center') or
                            can_meet_thresholds_without_horizon(epochs, pos, floor, min_sun, min_earth)):
                        if stored is not None:
                            classified, horizon_full = stored, None
                        else:
                            horizon_full = terrain.compute_horizon_360(*point)
                            classified = sky.classify_sky(pos, horizon_full[0])
                        sun = visibility_percent(epochs, classified["Sun"]["visible"])
                        earth = visibility_percent(epochs, classified["Earth"]["visible"])
                        if min_sun <= sun <= max_sun and min_earth <= earth <= max_earth:
                            slope = terrain.compute_slope_deg(*point)
                            area_slope = (terrain.max_slope_in_radius(*point, slope_radius_km)
                                          if max_slope is not None else None)
                            lunar = (lunar_day_stats(epochs, classified['Sun']['visible'])
                                     if lunar_stats or lunar_filter is not None else None)
                            if lunar_filter is not None and not lunar_passes(lunar, lunar_filter):
                                checked[point] = None
                            else:
                                checked[point] = ((sun, earth, slope, area_slope, lunar)
                                                  if area_slope is None or area_slope <= max_slope else None)
                                if checked[point] is not None and cache is not None and stored is None:
                                    cache.put_horizon(point[0], point[1], horizon_full)
                                    cache.put_sky(point[0], point[1], epochs, cache_source, classified)
                        else:
                            checked[point] = None
                    else:
                        checked[point] = None
                value = checked[point]
                if (value is not None and min_sun <= value[0] <= max_sun
                        and min_earth <= value[1] <= max_earth):
                    lunar = value[4]
                    results.append({"lat": point[0], "lon": point[1], "sun_percent": value[0],
                                    "earth_percent": value[1], "slope_deg": value[2], "max_slope_deg": value[3],
                                    "sun_lunar_mean": lunar[0] if lunar else None,
                                    "sun_lunar_std": lunar[1] if lunar else None,
                                    "lunar_days": lunar[2] if lunar else None,
                                    "cell_lat_min": cell[0], "cell_lat_max": cell[1],
                                    "cell_lon_min": cell[2], "cell_lon_max": cell[3]})
                    break
                if point == center and screen_mode == 'Pessimistic center':
                    if center_positions is None:
                        elevation = float(terrain.sample_dem_bilinear(*center))
                        center_positions = fetch(center[1], center[0], (elevation + terrain.EYE_HEIGHT_M) / 1000.0, epochs)
                    if cancel is not None and cancel.is_set():
                        break
                    can_meet_thresholds_without_horizon(epochs, center_positions, -90.0, 0.0, 0.0)
                    approx = sky.classify_sky(center_positions, np.full(360, screen_horizon))
                    if (visibility_percent(epochs, approx['Sun']['visible']) < min_sun or
                            visibility_percent(epochs, approx['Earth']['visible']) < min_earth):
                        filtered += 1
                        break
            if cancel is not None and cancel.is_set():
                break
        except Exception as error:
            raise RuntimeError(f"Search failed in cell {cell}: {error}") from error
        completed += 1
    if progress:
        progress(completed, total_cells, len(results), time.monotonic() - start, None, None, filtered)
    return sorted(results, key=lambda p: (min(p['sun_percent'] - min_sun, p['earth_percent'] - min_earth),
                                           p['sun_percent'] + p['earth_percent']), reverse=True)


def search_optimistic_sectors(cells, epochs, min_sun, min_earth, fetch, cancel=None, progress=None, horizon_result=None,
                              workers=None):
    """Keep cells with optimistic visibility estimates, without terrain horizon rays.

    Each body is allowed the cell's full angular margin and full disk visibility;
    mutual eclipses are ignored. The horizon assumes the median observer height,
    so it is not a guaranteed bound for sites higher than that median.
    """
    if terrain.dem_km is None:
        raise RuntimeError('The Moon elevation map has not been loaded.')
    if len(epochs) < 2 or epochs[-1] <= epochs[0]:
        raise ValueError('End time must be later than start time.')
    if not np.isfinite([min_sun, min_earth]).all() or not (0 <= min_sun <= 100 and 0 <= min_earth <= 100):
        raise ValueError('Visibility thresholds must be between 0 and 100 percent.')
    heights = global_elevation_stats(cancel)
    if heights is None:
        return []
    min_height, median_height = heights
    floor = optimistic_horizon_floor(median_height, min_height)
    if horizon_result:
        horizon_result(min_height, median_height, floor)
    if workers is not None and workers != 1:   # 0 = automatic, n > 1 = n processes (Skyfield is used directly)
        return _sectors_parallel(cells, epochs, min_sun, min_earth, floor, cancel, progress, workers)
    results, filtered, start, completed = [], 0, time.monotonic(), 0
    total = len(cells)
    for index, cell in enumerate(cells):
        if cancel is not None and cancel.is_set():
            break
        center = cell_sample_points(cell, 2)[0]
        if progress:
            progress(index, total, len(results), time.monotonic() - start, *center, filtered)
        try:
            elevation = float(terrain.sample_dem_bilinear(*center))
            positions = fetch(center[1], center[0], (elevation + terrain.EYE_HEIGHT_M) / 1000.0, epochs)
            if cancel is not None and cancel.is_set():
                break
            margin = min(180.0, (cell[1] - cell[0] + cell[3] - cell[2]) / 2.0 + 2.0)
            bounds = []
            for body in ('Sun', 'Earth'):
                el = np.asarray(positions[body]['el'], dtype=float)
                radius = np.asarray(positions[body]['radius'], dtype=float)
                if el.shape != (len(epochs),) or radius.shape != (len(epochs),) or not np.isfinite(el).all() or not np.isfinite(radius).all() or np.any(radius < 0):
                    raise ValueError(f'Invalid {body} ephemeris.')
                bounds.append(visibility_percent(epochs, el + radius + margin >= floor))
            if bounds[0] + 1e-9 >= min_sun and bounds[1] + 1e-9 >= min_earth:
                results.append({'kind': 'sector', 'lat': center[0], 'lon': center[1],
                                'sun_percent': bounds[0], 'earth_percent': bounds[1],
                                'cell_lat_min': cell[0], 'cell_lat_max': cell[1],
                                'cell_lon_min': cell[2], 'cell_lon_max': cell[3]})
            else:
                filtered += 1
            completed += 1
        except Exception as error:
            raise RuntimeError(f'Optimistic search failed in sector {cell}: {error}') from error
    if progress:
        progress(completed, total, len(results), time.monotonic() - start, None, None, filtered)
    return sorted(results, key=lambda p: (min(p['sun_percent'] - min_sun, p['earth_percent'] - min_earth),
                                          p['sun_percent'] + p['earth_percent']), reverse=True)


# ======================================================================================================
# Multi-process evaluation. The serial search runs one point after another in one thread (one core, and the GIL
# stops threads from helping). Here worker processes each evaluate whole cells; they read the DEM from one shared
# memory block (terrain.share_dem), so RAM grows by the per-process Python / Skyfield data only, not 1 GB each.
# ======================================================================================================
_POOL = None
_POOL_KEY = None
_POOL_LOCK = threading.Lock()


@contextmanager
def _main_hidden():
    """A spawned worker normally re-imports the program's main script (here moon.py: VTK, Qt, PIL ...). Workers only
    need terrain / sky / this module, so hide the main script from multiprocessing while they are being started."""
    main = sys.modules.get('__main__')
    saved = {name: getattr(main, name) for name in ('__file__', '__spec__') if main is not None and hasattr(main, name)}
    try:
        for name in saved:
            if name == '__spec__':
                main.__spec__ = None
            else:
                delattr(main, name)
        yield
    finally:
        for name, value in saved.items():
            setattr(main, name, value)


def default_workers():
    return max(1, (os.cpu_count() or 2) - 2)   # two logical cores stay free for the GUI and the system


def _init_worker(shm_name, shape, r_max_m):
    terrain.attach_dem(shm_name, shape, r_max_m)
    terrain.slope_deg_at = terrain._compute_slope_deg_at   # no map_modes (VTK side) in a worker; same formula
    sky.preload()


def _get_pool(workers):
    """One persistent pool: workers start (and attach the DEM / ephemeris) once, not once per search."""
    global _POOL, _POOL_KEY
    sky.preload()   # validate / download the kernels once here, never in all workers at the same time
    name, shape, r_max_m = terrain.share_dem()
    with _POOL_LOCK:
        if _POOL is not None and _POOL_KEY == (workers, name):
            return _POOL
        if _POOL is not None:
            _POOL.shutdown(wait=False, cancel_futures=True)
        # 'spawn': never fork a process that owns Qt / VTK / Tk threads.
        _POOL = ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn'),
                                    initializer=_init_worker, initargs=(name, shape, r_max_m))
        _POOL_KEY = (workers, name)
        # Start every worker now, while the main script is hidden (they are busy initialising, so each submit
        # starts a new process). A worker started later would still work, only slower to start.
        with _main_hidden():
            for index in range(workers):
                _POOL.submit(_ping, index)
        return _POOL


def shutdown_pool():
    global _POOL, _POOL_KEY
    with _POOL_LOCK:
        pool, _POOL, _POOL_KEY = _POOL, None, None
    if pool is not None:
        pool.shutdown(wait=False, cancel_futures=True)


atexit.register(shutdown_pool)


def _ping(_):
    return os.getpid()


def warm_up(workers=0):
    """Optional: start the workers in the background (for example when the search window opens)."""
    _get_pool(workers or default_workers())


def _evaluate_cell(cell, p):
    """One terrain-checked cell: (row or None, screened out, (point, horizon, classified) or None)."""
    epochs, min_sun, min_earth = p['epochs'], p['min_sun'], p['min_earth']
    max_sun, max_earth = p['max_sun'], p['max_earth']
    points = cell_sample_points(cell, p['side'])
    center = points[0]
    for point in points:
        elevation = float(terrain.sample_dem_bilinear(*point))
        pos = sky.get_positions_nocache(point[1], point[0], (elevation + terrain.EYE_HEIGHT_M) / 1000.0, epochs)
        floor = optimistic_horizon_floor(elevation, p['min_terrain'])
        # The cheap optimistic bound also guards the centre (the serial code skipped it there). That is safe: if even
        # the optimistic estimate fails, the real terrain cannot pass, and the horizon scan is saved.
        if can_meet_thresholds_without_horizon(epochs, pos, floor, min_sun, min_earth):
            horizon_full = terrain.compute_horizon_360(*point)
            classified = sky.classify_sky(pos, horizon_full[0])
            sun = visibility_percent(epochs, classified['Sun']['visible'])
            earth = visibility_percent(epochs, classified['Earth']['visible'])
            if min_sun <= sun <= max_sun and min_earth <= earth <= max_earth:
                area_slope = (terrain.max_slope_in_radius(*point, p['slope_radius_km'])
                              if p['max_slope'] is not None else None)
                lunar = (lunar_day_stats(epochs, classified['Sun']['visible'])
                         if p['lunar_stats'] or p['lunar_filter'] is not None else None)
                if ((lunar is None or p['lunar_filter'] is None or lunar_passes(lunar, p['lunar_filter']))
                        and (area_slope is None or area_slope <= p['max_slope'])):
                    row = {'lat': point[0], 'lon': point[1], 'sun_percent': sun, 'earth_percent': earth,
                           'slope_deg': None, 'max_slope_deg': area_slope,   # slope_deg is filled in by the caller
                           'sun_lunar_mean': lunar[0] if lunar else None,
                           'sun_lunar_std': lunar[1] if lunar else None,
                           'lunar_days': lunar[2] if lunar else None,
                           'cell_lat_min': cell[0], 'cell_lat_max': cell[1],
                           'cell_lon_min': cell[2], 'cell_lon_max': cell[3]}
                    return row, False, (point, horizon_full, classified)
        if point == center and p['mode'] == 'Pessimistic center':
            approx = sky.classify_sky(pos, np.full(360, p['screen_horizon']))
            if (visibility_percent(epochs, approx['Sun']['visible']) < min_sun or
                    visibility_percent(epochs, approx['Earth']['visible']) < min_earth):
                return None, True, None   # the other samples are not tried
    return None, False, None


def _run_cells(cells, p):
    terrain.EYE_HEIGHT_M = p['eye']
    return [_evaluate_cell(cell, p) for cell in cells]


def _run_blocks(items, p):
    """Optimistic range screen of 4x4 blocks; items are (bounds, cells). One bool per block."""
    terrain.EYE_HEIGHT_M = p['eye']
    epochs, out = p['epochs'], []
    for bounds, _cells in items:
        center = cell_sample_points(bounds, 3)[0]
        height = float(terrain.sample_dem_bilinear(*center))
        pos = sky.get_positions_nocache(center[1], center[0], (height + terrain.EYE_HEIGHT_M) / 1000.0, epochs)
        margin = min(180.0, (bounds[1] - bounds[0] + bounds[3] - bounds[2]) / 2.0 + 2.0)
        ok = True
        for body, threshold in (('Sun', p['min_sun']), ('Earth', p['min_earth'])):
            el = np.asarray(pos[body]['el'], dtype=float)
            radius = np.asarray(pos[body]['radius'], dtype=float)
            if (el.shape != (len(epochs),) or radius.shape != (len(epochs),) or not np.isfinite(el).all()
                    or not np.isfinite(radius).all() or np.any(radius < 0)):
                raise ValueError(f'Invalid {body} ephemeris in optimistic range screen.')
            if visibility_percent(epochs, el + radius + margin >= p['floor']) + 1e-9 < threshold:
                ok = False
        out.append(ok)
    return out


def _run_sky(points, p):
    """Terrain horizon + classified Sun / Earth sky of several points (for the optional columns)."""
    terrain.EYE_HEIGHT_M = p['eye']
    out = []
    for lat, lon in points:
        elevation = float(terrain.sample_dem_bilinear(lat, lon))
        pos = sky.get_positions_nocache(lon, lat, (elevation + terrain.EYE_HEIGHT_M) / 1000.0, p['epochs'])
        horizon = terrain.compute_horizon_360(lat, lon)
        out.append((horizon, sky.classify_sky(pos, horizon[0])))
    return out


def _stream(pool, function, items, chunk, window, cancel, handler):
    """Submit chunks with a bounded number in flight (so Cancel reacts quickly); handler(part, result) per chunk."""
    starts = iter(range(0, len(items), chunk))
    pending = {}

    def fill():
        while len(pending) < window:
            first = next(starts, None)
            if first is None:
                return
            part = items[first:first + chunk]
            pending[pool.submit(function, part)] = part
    fill()
    try:
        while pending:
            if cancel is not None and cancel.is_set():
                break
            done, _ = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)
            for future in done:
                handler(pending.pop(future), future.result())   # a worker's exception is raised here
            fill()
    finally:
        for future in pending:
            future.cancel()


def prefetch_sky(points, epochs, cache, source, cancel, workers=0):
    """Calculate (in worker processes) and store the sky data of points that the cache does not have yet."""
    todo = [pt for pt in points if cache.get_sky(pt[0], pt[1], epochs, source) is None]
    if not todo:
        return
    workers = workers or default_workers()
    p = {'epochs': epochs, 'eye': terrain.EYE_HEIGHT_M}

    def store(part, outputs):
        for (lat, lon), (horizon, classified) in zip(part, outputs):
            cache.put_horizon(lat, lon, horizon)
            cache.put_sky(lat, lon, epochs, source, classified)
    try:
        _stream(_get_pool(workers), partial(_run_sky, p=p), todo, max(1, min(16, len(todo) // (workers * 4) or 1)),
                workers * 3, cancel, store)
    except BrokenProcessPool:
        shutdown_pool()   # the caller falls back to the serial calculation, point by point


def search_cells_parallel(cells, epochs, min_sun, min_earth, side, screen_horizon, screen_mode, cancel, progress,
                          use_ranges, range_progress, range_result, max_slope, slope_radius_km, horizon_result,
                          max_sun, max_earth, lunar_stats, lunar_filter, cache, cache_source, workers=0):
    """The same result as the serial search_cells, evaluated by a pool of processes."""
    workers = workers or default_workers()
    heights = global_elevation_stats(cancel)
    if heights is None:
        return []
    min_height, median_height = heights
    floor = optimistic_horizon_floor(median_height, min_height)
    if horizon_result:
        horizon_result(min_height, median_height, floor)
    p = dict(epochs=epochs, min_sun=min_sun, min_earth=min_earth, max_sun=max_sun, max_earth=max_earth,
             side=side, screen_horizon=screen_horizon, mode=screen_mode, max_slope=max_slope,
             slope_radius_km=slope_radius_km, lunar_stats=lunar_stats, lunar_filter=lunar_filter,
             min_terrain=float(np.min(terrain.dem_km)) * 1000.0, floor=floor, eye=terrain.EYE_HEIGHT_M)
    pool = _get_pool(workers)
    window = workers * 3
    start = time.monotonic()
    total_cells = len(cells)
    try:
        if use_ranges:
            lat_index = {x: i for i, x in enumerate(sorted({c[0] for c in cells}))}
            lon_index = {x: i for i, x in enumerate(sorted({c[2] for c in cells}))}
            blocks = {}
            for cell in cells:
                blocks.setdefault((lat_index[cell[0]] // 4, lon_index[cell[2]] // 4), []).append(cell)
            items = [((min(c[0] for c in b), max(c[1] for c in b), min(c[2] for c in b), max(c[3] for c in b)), b)
                     for b in blocks.values()]
            eligible, state = [], {'done': 0}

            def on_blocks(part, flags):
                for (_bounds, block), ok in zip(part, flags):
                    state['done'] += len(block)
                    if ok:
                        eligible.extend(block)
                if range_progress:
                    range_progress(state['done'], total_cells, len(eligible), None)
            _stream(pool, partial(_run_blocks, p=p), items, max(1, min(16, len(items) // (workers * 4) or 1)),
                    window, cancel, on_blocks)
            if cancel is not None and cancel.is_set():
                return []
            if range_result:
                range_result(interval_union((c[0], c[1]) for c in eligible),
                             interval_union((c[2], c[3]) for c in eligible), len(eligible), total_cells)
            cells = eligible
        filtered = completed = total_cells - len(cells)
        results = []

        def on_cells(part, outputs):
            nonlocal filtered, completed
            for cell, (row, screened, item) in zip(part, outputs):
                completed += 1
                filtered += bool(screened)
                if row is not None:
                    row['slope_deg'] = terrain.compute_slope_deg(row['lat'], row['lon'])   # same source as the table
                    results.append(row)
                    if cache is not None and item is not None:
                        point, horizon_full, classified = item
                        cache.put_horizon(point[0], point[1], horizon_full)
                        cache.put_sky(point[0], point[1], epochs, cache_source, classified)
            if progress:
                progress(completed, total_cells, len(results), time.monotonic() - start,
                         part[-1][0], part[-1][2], filtered)
        _stream(pool, partial(_run_cells, p=p), cells, max(1, min(32, len(cells) // (workers * 8) or 1)),
                window, cancel, on_cells)
    except BrokenProcessPool:
        shutdown_pool()   # the next search starts a fresh pool
        raise RuntimeError('A search worker process died (out of memory?). Use fewer worker processes.')
    if progress:
        progress(completed, total_cells, len(results), time.monotonic() - start, None, None, filtered)
    return sorted(results, key=lambda r: (min(r['sun_percent'] - min_sun, r['earth_percent'] - min_earth),
                                          r['sun_percent'] + r['earth_percent']), reverse=True)


def _run_sectors(cells, p):
    """Optimistic Sun / Earth bounds of several sectors: per cell None or (sun %, earth %)."""
    terrain.EYE_HEIGHT_M = p['eye']
    epochs, out = p['epochs'], []
    for cell in cells:
        center = cell_sample_points(cell, 2)[0]
        elevation = float(terrain.sample_dem_bilinear(*center))
        pos = sky.get_positions_nocache(center[1], center[0], (elevation + terrain.EYE_HEIGHT_M) / 1000.0, epochs)
        margin = min(180.0, (cell[1] - cell[0] + cell[3] - cell[2]) / 2.0 + 2.0)
        bounds = []
        for body in ('Sun', 'Earth'):
            el = np.asarray(pos[body]['el'], dtype=float)
            radius = np.asarray(pos[body]['radius'], dtype=float)
            if (el.shape != (len(epochs),) or radius.shape != (len(epochs),) or not np.isfinite(el).all()
                    or not np.isfinite(radius).all() or np.any(radius < 0)):
                raise ValueError(f'Invalid {body} ephemeris.')
            bounds.append(visibility_percent(epochs, el + radius + margin >= p['floor']))
        ok = bounds[0] + 1e-9 >= p['min_sun'] and bounds[1] + 1e-9 >= p['min_earth']
        out.append((bounds[0], bounds[1], center) if ok else None)
    return out


def _sectors_parallel(cells, epochs, min_sun, min_earth, floor, cancel, progress, workers):
    workers = workers or default_workers()
    p = dict(epochs=epochs, min_sun=min_sun, min_earth=min_earth, floor=floor, eye=terrain.EYE_HEIGHT_M)
    pool = _get_pool(workers)
    start = time.monotonic()
    total = len(cells)
    results, state = [], {'done': 0, 'filtered': 0}

    def on_cells(part, outputs):
        for cell, value in zip(part, outputs):
            state['done'] += 1
            if value is None:
                state['filtered'] += 1
                continue
            sun, earth, center = value
            results.append({'kind': 'sector', 'lat': center[0], 'lon': center[1], 'sun_percent': sun,
                            'earth_percent': earth, 'cell_lat_min': cell[0], 'cell_lat_max': cell[1],
                            'cell_lon_min': cell[2], 'cell_lon_max': cell[3]})
        if progress:
            progress(state['done'], total, len(results), time.monotonic() - start,
                     part[-1][0], part[-1][2], state['filtered'])
    try:
        _stream(pool, partial(_run_sectors, p=p), cells, max(1, min(64, len(cells) // (workers * 8) or 1)),
                workers * 3, cancel, on_cells)
    except BrokenProcessPool:
        shutdown_pool()
        raise RuntimeError('A search worker process died (out of memory?). Use fewer worker processes.')
    if progress:
        progress(state['done'], total, len(results), time.monotonic() - start, None, None, state['filtered'])
    return sorted(results, key=lambda r: (min(r['sun_percent'] - min_sun, r['earth_percent'] - min_earth),
                                          r['sun_percent'] + r['earth_percent']), reverse=True)


def _search_worker(messages, cancel, cells, epochs, sun, earth, fetch, side, horizon, mode, use_ranges,
                   max_slope=None, slope_radius_km=1.0, use_terrain=True, max_sun=100.0, max_earth=100.0,
                   lunar_stats=False, lunar_filter=None, cache=None, workers=None):
    def progress(done, total, matches, elapsed, lat, lon, filtered):
        messages.put(('progress', (done, total, matches, elapsed, lat, lon, filtered)))
    try:
        def range_progress(done, total, eligible, center):
            messages.put(('range_progress', (done, total, eligible)))
        def range_result(lat_ranges, lon_ranges, eligible, total):
            messages.put(('ranges', (lat_ranges, lon_ranges, eligible, total)))
        def horizon_result(low, median, floor):
            messages.put(('optimistic_horizon', (low, median, floor)))
        if use_ranges or not use_terrain:
            messages.put(('phase', 'Calculating the exact median and lowest elevation...'))
        if use_terrain:
            results = search_cells(cells, epochs, sun, earth, fetch, side, horizon, mode, cancel, progress,
                                   use_ranges, range_progress, range_result, max_slope, slope_radius_km, horizon_result,
                                   max_sun, max_earth, lunar_stats, lunar_filter, cache,
                                   point_data.DEFAULT_SOURCE, workers)
        else:
            results = search_optimistic_sectors(cells, epochs, sun, earth, fetch, cancel, progress, horizon_result, workers)
        messages.put(('done', (results, cancel.is_set())))
    except Exception as error:
        messages.put(('error', str(error)))


def _num(form):
    return lambda value, row: '-' if value is None else format(value, form)


def _duration(value, row):
    return format_duration(value)


def _period(partial_key):
    return lambda value, row: format_duration(value) + ('*' if value is not None and row.get(partial_key) else '')


# key, heading, group, width, row key (None: built by populate_results), formatter, CSV name
COLUMN_SPECS = (
    ('lat', 'Latitude', 'Position', 90, 'lat', _num('+.3f'), 'latitude_deg'),
    ('lon', 'Longitude', 'Position', 90, 'lon', _num('+.3f'), 'longitude_deg'),
    ('sun', 'Sun visible %', 'Visible % of the interval', 110, 'sun_percent', _num('.1f'), 'sun_visible_percent'),
    ('earth', 'Earth visible %', 'Visible % of the interval', 115, 'earth_percent', _num('.1f'), 'earth_visible_percent'),
    ('lunar_mean', 'Sun avg %/lunar day', 'Sun per lunar day', 135, 'sun_lunar_mean', _num('.2f'),
     'sun_avg_percent_per_lunar_day'),
    ('lunar_sd', 'Sun SD %', 'Sun per lunar day', 80, 'sun_lunar_std', _num('.2f'), 'sun_percent_per_lunar_day_sd'),
    ('elevation', 'Elevation m', 'Terrain', 95, 'elevation_m', _num('+,.0f'), 'elevation_m'),
    ('slope', 'Slope deg', 'Terrain', 80, 'slope_deg', _num('.2f'), 'slope_deg'),
    ('max_slope', 'Max slope deg', 'Terrain', 100, 'max_slope_deg', _num('.2f'), 'max_sampled_slope_deg'),
    ('sun_center', 'Sun above horizon %', 'Centre above the horizon', 140, 'sun_center_pct', _num('.2f'),
     'sun_center_above_horizon_percent'),
    ('earth_center', 'Earth above horizon %', 'Centre above the horizon', 150, 'earth_center_pct', _num('.2f'),
     'earth_center_above_horizon_percent'),
    ('sun_per_day', 'Sun above / lunar day', 'Centre above the horizon', 145, 'sun_per_lunar_day_s', _duration,
     'sun_above_per_lunar_day_s'),
    ('earth_per_day', 'Earth above / lunar day', 'Centre above the horizon', 155, 'earth_per_lunar_day_s', _duration,
     'earth_above_per_lunar_day_s'),
    ('sun_period', 'Avg sunlit period', 'Centre above the horizon', 125, 'sun_mean_period_s',
     _period('sun_period_partial'), 'sun_mean_sunlit_period_s'),
    ('earth_period', 'Avg Earth period', 'Centre above the horizon', 125, 'earth_mean_period_s',
     _period('earth_period_partial'), 'earth_mean_visible_period_s'),
    ('sun_az', 'Sun AZ avg deg', 'Directions (samples above the horizon)', 110, 'sun_az_mean', _num('.2f'), 'sun_az_avg_deg'),
    ('sun_az_sd', 'Sun AZ SD deg', 'Directions (samples above the horizon)', 110, 'sun_az_sd', _num('.2f'), 'sun_az_sd_deg'),
    ('sun_el', 'Sun EL avg deg', 'Directions (samples above the horizon)', 110, 'sun_el_mean', _num('+.2f'), 'sun_el_avg_deg'),
    ('sun_el_sd', 'Sun EL SD deg', 'Directions (samples above the horizon)', 110, 'sun_el_sd', _num('.2f'), 'sun_el_sd_deg'),
    ('earth_az', 'Earth AZ avg deg', 'Directions (samples above the horizon)', 115, 'earth_az_mean', _num('.2f'),
     'earth_az_avg_deg'),
    ('earth_az_sd', 'Earth AZ SD deg', 'Directions (samples above the horizon)', 115, 'earth_az_sd', _num('.2f'),
     'earth_az_sd_deg'),
    ('earth_el', 'Earth EL avg deg', 'Directions (samples above the horizon)', 115, 'earth_el_mean', _num('+.2f'),
     'earth_el_avg_deg'),
    ('earth_el_sd', 'Earth EL SD deg', 'Directions (samples above the horizon)', 115, 'earth_el_sd', _num('.2f'),
     'earth_el_sd_deg'),
    ('cells', 'Cells', 'Position', 60, None, None, None),
    ('cell', 'Cell bounds (lat / lon)', 'Position', 250, None, None, None),
)
SPEC = {spec[0]: spec for spec in COLUMN_SPECS}
PERCENT_COLUMNS = frozenset(('lat', 'lon', 'sun', 'earth', 'lunar_mean', 'lunar_sd',
                             'sun_center', 'earth_center', 'cells', 'cell'))
DEFAULT_COLUMNS = ('lat', 'lon', 'sun', 'earth', 'cells', 'cell')
# Columns that need the sky statistics of compare.py (calculated in the background from the shared data).
STATS_KEYS = frozenset(spec[0] for spec in COLUMN_SPECS
                       if spec[2] in ('Sun per lunar day', 'Centre above the horizon')
                       or spec[2].startswith('Directions'))
LUNAR_KEYS = frozenset(('lunar_mean', 'lunar_sd'))
# Not available for candidate sectors (they are regions, not points).
SECTOR_HIDDEN = STATS_KEYS | {'elevation', 'max_slope', 'slope'}
FILL_REFRESH_SECONDS = 1.5


def _fill_worker(messages, cancel, generation, rows, epochs, fetch, cache, need_stats, need_slope, radius, workers=1):
    """Optional columns: statistics from the shared sky data (or calculated and stored), and maximum slopes."""
    total = len(rows)
    try:
        if need_stats and cache is not None and workers != 1:
            # Missing sky data is calculated by the worker processes first; the loop below then only reads it.
            messages.put(('fill_progress', (generation, 0, total)))
            prefetch_sky([(r['lat'], r['lon']) for r in rows if not r.get('_stats')], epochs, cache,
                          point_data.DEFAULT_SOURCE, cancel, workers)
        for done, row in enumerate(rows, start=1):
            if cancel.is_set():
                break
            update = {}
            if need_stats and not row.get('_stats'):
                classified = point_data.sky_for(row['lat'], row['lon'], epochs, fetch, cache)
                update.update(point_data.metrics_from_sky(epochs, classified))
                update['_stats'] = True
            if need_slope and row.get('max_slope_deg') is None:
                update['max_slope_deg'] = max_slope_cached(row['lat'], row['lon'], radius)
            messages.put(('fill_row', (generation, row, update)))
            if done % 25 == 0 or done == total:
                messages.put(('fill_progress', (generation, done, total)))
        messages.put(('fill_done', (generation, cancel.is_set())))
    except Exception as error:
        messages.put(('fill_error', (generation, str(error))))


class SearchDialog:
    """Tk-only UI; all horizon and ephemeris work stays in a worker thread."""
    workers_setting = 0   # worker processes: 0 = automatic, 1 = this thread only (class default: SearchPanel in
                          # desktop_ui does not call __init__)
    def __init__(self, parent, fetch, previous=None, on_close=None, existing_points=None, on_apply=None,
                 cache=None):
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.ttk, self.fetch = tk, ttk, fetch
        self.cache = cache  # point_data.PointDataCache shared with the comparison (None: no sharing)
        self.workers_setting = 0       # worker processes: 0 = automatic, 1 = this thread only
        self._fill_cancel = threading.Event()
        self._fill_generation = 0
        self._fill_running = False
        self._fill_dirty = False
        self._fill_refreshed = 0.0
        self._tick = 0
        self.on_close = on_close
        # Called when a button wants the viewer to act (show on globe, add points, analyze, clear).
        # The window stays open; only Close (or the window's own close button) closes it.
        self.on_apply = on_apply
        self.existing_points = existing_points or (lambda: [])
        self._closed = False
        self.window = tk.Toplevel(parent)
        self.window.title('Search lunar visibility')
        self.window.protocol('WM_DELETE_WINDOW', self.close)
        width = min(1320, max(640, self.window.winfo_screenwidth() - 40))
        height = min(860, max(520, self.window.winfo_screenheight() - 80))
        self.window.geometry(f'{width}x{height}')
        self.messages = queue.Queue()
        self.cancel = threading.Event()
        self.worker = None
        self.timer = None
        self.results = []
        self.selected = None
        self.selected_points = []
        self.add_to_compare = False
        self.show_matches = False
        self.clear_markers = False
        self.analyze = False
        self.marker_results = None
        self.marker_mode = 'Points'
        self._display_items = []
        self.last_parameters = None
        self._refinement_sectors = None
        self._sort = None              # (column key, descending) of the results table, or None
        self._cluster_cache = None
        self.finished = False
        self.was_cancelled = False
        shell = ttk.Frame(self.window, padding=(16, 12))
        shell.pack(fill='both', expand=True)
        heading = ttk.Frame(shell)
        heading.pack(fill='x', pady=(0, 10))
        ttk.Label(heading, text='Visibility search', font=('TkDefaultFont', 16, 'bold')).pack(side='left')
        ttk.Label(heading, text='  Set criteria, run a search, then inspect the results.').pack(side='left')
        self.workspace_tabs = None
        if width < 1000:
            panes = ttk.Notebook(shell)
            settings_pane = ttk.Frame(panes, padding=(0, 8))
            results_pane = ttk.Frame(panes, padding=(0, 8))
            panes.add(settings_pane, text='Search settings')
            panes.add(results_pane, text='Results')
            self.workspace_tabs = panes
            self.workspace_settings, self.workspace_results = settings_pane, results_pane
        else:
            panes = ttk.Panedwindow(shell, orient='horizontal')
            settings_pane = ttk.Frame(panes, padding=(0, 0, 12, 0), width=420)
            results_pane = ttk.Frame(panes, padding=(12, 0, 0, 0), width=720)
            panes.add(settings_pane, weight=0)
            panes.add(results_pane, weight=1)
        panes.pack(fill='both', expand=True)
        ttk.Label(settings_pane, text='SEARCH SETTINGS', font=('TkDefaultFont', 10, 'bold')).pack(anchor='w', pady=(0, 6))
        now = sky.utc_now()
        self.f = {}  # name -> Entry
        self.use_terrain = tk.BooleanVar(master=self.window, value=True)
        self.lunar_avg = tk.BooleanVar(master=self.window, value=False)
        self.lunar_sd = tk.BooleanVar(master=self.window, value=False)
        sections = ttk.Notebook(settings_pane)
        sections.pack(fill='both', expand=True)
        pages, page_tabs = [], []
        for title in ('Basics', 'Area & slope', 'Advanced'):
            tab = ttk.Frame(sections)
            page_tabs.append(tab)
            sections.add(tab, text=title)
            canvas = tk.Canvas(tab, height=300, highlightthickness=0, borderwidth=0)
            scroll = ttk.Scrollbar(tab, orient='vertical', command=canvas.yview)
            canvas.configure(yscrollcommand=scroll.set)
            scroll.pack(side='right', fill='y')
            canvas.pack(side='left', fill='both', expand=True)
            page = ttk.Frame(canvas, padding=(8, 8))
            window_id = canvas.create_window((0, 0), window=page, anchor='nw')
            page.bind('<Configure>', lambda event, c=canvas: c.configure(scrollregion=c.bbox('all')))
            canvas.bind('<Configure>', lambda event, c=canvas, item=window_id: c.itemconfigure(item, width=event.width))
            pages.append(page)
        self.settings_tabs, self.area_page = sections, pages[1]
        self.area_tab = page_tabs[1]
        for page in pages:
            page.columnconfigure(0, weight=1)

        def section(title, row, column):
            box = ttk.LabelFrame(pages[row], text=title, padding=(8, 6))
            box.grid(row=column, column=0, sticky='ew', pady=(0, 8))
            box.columnconfigure(1, weight=1)
            box.columnconfigure(3, weight=1)
            return box

        def field(box, name, label, value, row, column=0):
            ttk.Label(box, text=label).grid(row=row, column=column * 2, sticky='w', padx=(0, 5), pady=2)
            entry = ttk.Entry(box, width=11 if column else 14)
            entry.insert(0, value)
            entry.grid(row=row, column=column * 2 + 1, sticky='ew', padx=(0, 10), pady=2)
            self.f[name] = entry

        time_box = section('Time', 0, 0)
        field(time_box, 'start', 'Start (UTC)', now.strftime('%Y-%m-%d %H:%M:%S'), 0)
        field(time_box, 'end', 'End (UTC)', (now + timedelta(days=LUNAR_DAY_DAYS)).strftime('%Y-%m-%d %H:%M:%S'), 1)
        field(time_box, 'step', 'Step (hours)', '2', 2)
        visible_box = section('Visible % of the time', 0, 1)
        field(visible_box, 'min_earth', 'Min Earth', '50', 0, 0)
        field(visible_box, 'max_earth', 'Max Earth', '100', 0, 1)
        field(visible_box, 'min_sun', 'Min Sun', '30', 1, 0)
        field(visible_box, 'max_sun', 'Max Sun', '100', 1, 1)
        ttk.Label(visible_box, text='Maximums need the terrain search (not used by optimistic sectors).',
                  wraplength=420).grid(row=2, column=0, columnspan=4, sticky='w')
        area_box = section('Area', 1, 0)
        field(area_box, 'spacing', 'Grid spacing (deg)', '10', 0)
        field(area_box, 'lat', 'Latitude min, max', '-90, 90', 1)
        field(area_box, 'lon', 'Longitude start, end', '-180, 180', 2)
        slope_box = section('Slope', 1, 1)
        field(slope_box, 'max_slope', 'Max slope (deg; blank = off)', '', 0)
        field(slope_box, 'slope_radius', 'Slope radius (km; 0-20)', '1', 1)
        lunar_box = section('Lunar-day statistics', 2, 0)
        self.lunar_avg_check = ttk.Checkbutton(lunar_box, text='Calculate and show average Sun % per lunar day for each point',
                                               variable=self.lunar_avg, command=self._update_lunar_state)
        self.lunar_avg_check.grid(row=0, column=0, columnspan=4, sticky='w')
        self.lunar_sd_check = ttk.Checkbutton(lunar_box, text='Calculate and show its standard deviation',
                                              variable=self.lunar_sd, command=self._update_lunar_state)
        self.lunar_sd_check.grid(row=1, column=0, columnspan=4, sticky='w')
        field(lunar_box, 'lunar_min_avg', 'Min avg Sun %', '', 2, 0)
        field(lunar_box, 'lunar_max_avg', 'Max avg Sun %', '', 2, 1)
        field(lunar_box, 'lunar_min_sd', 'Min SD %', '', 3, 0)
        field(lunar_box, 'lunar_max_sd', 'Max SD %', '', 3, 1)
        self.lunar_note = ttk.Label(lunar_box, text='Blank limit = no limit. Limits also apply when the columns are hidden.',
                                    wraplength=470)
        self.lunar_note.grid(row=4, column=0, columnspan=4, sticky='w')
        other_box = section('Screening', 2, 1)
        ttk.Checkbutton(settings_pane, text='Use terrain horizon  (off: candidate clusters)', variable=self.use_terrain, command=self._update_lunar_state
                        ).pack(anchor='w', before=sections, pady=(0, 6))
        field(other_box, 'side', 'Samples per side (2-9)', '3', 0)
        field(other_box, 'horizon', 'Pessimistic horizon (deg)', '2', 1)
        ttk.Label(other_box, text='Cell screening').grid(row=2, column=0, sticky='w', pady=2)
        self.screen_mode = ttk.Combobox(other_box, values=('Pessimistic center', 'All cells'), state='readonly', width=14)
        self.screen_mode.set('Pessimistic center')
        self.screen_mode.grid(row=2, column=1, sticky='ew', padx=(0, 10), pady=2)
        ttk.Label(other_box, text='Optimistic range estimate').grid(row=3, column=0, sticky='w', pady=2)
        self.range_mode = ttk.Combobox(other_box, values=('On', 'Off'), state='readonly', width=14)
        self.range_mode.set('On')
        self.range_mode.grid(row=3, column=1, sticky='ew', pady=2)
        field(other_box, 'workers', 'Worker processes (0 = auto, 1 = off)', '0', 4)
        for name in ('start', 'end'):
            self.f[name].bind('<KeyRelease>', lambda event: self._update_lunar_state())
            self.f[name].bind('<FocusOut>', lambda event: self._update_lunar_state())
        self._update_lunar_state()
        ttk.Label(settings_pane, text='Screening is approximate and can miss suitable cells.',
                  wraplength=390).pack(anchor='w', pady=(8, 3))
        controls = ttk.Frame(settings_pane)
        controls.pack(fill='x', before=sections, pady=(3, 0))
        self.start_button = ttk.Button(controls, text='Start search', command=self.start)
        self.start_button.pack(side='left')
        self.cancel_button = ttk.Button(controls, text='Cancel search', command=self.cancel.set, state='disabled')
        self.cancel_button.pack(side='left', padx=6)
        self.progress = ttk.Progressbar(settings_pane, mode='determinate')
        self.progress.pack(fill='x', before=sections, pady=(8, 3))
        self.status = ttk.Label(settings_pane, text='Choose criteria and start the search.', wraplength=380)
        self.status.pack(anchor='w', before=sections)
        self._details_open = False
        self.details_toggle = ttk.Button(settings_pane, text='Show calculation details', command=self._toggle_status_details)
        self.details_toggle.pack(anchor='w', before=sections, pady=(6, 3))
        self.details_panel = ttk.Frame(settings_pane)
        self.fill_label = ttk.Label(self.details_panel, text='', wraplength=380)
        self.fill_label.pack(anchor='w')
        self.cache_label = ttk.Label(self.details_panel, text='', wraplength=380)
        self.cache_label.pack(anchor='w')
        self.range_label = ttk.Label(self.details_panel, text='', wraplength=380)
        self.range_label.pack(anchor='w')
        self.horizon_label = ttk.Label(self.details_panel, text='', wraplength=380)
        self.horizon_label.pack(anchor='w')
        frame = results_pane
        ttk.Label(frame, text='RESULTS', font=('TkDefaultFont', 10, 'bold')).pack(anchor='w')
        ttk.Label(frame, text='Select with Shift or Ctrl. Click a column heading to sort.',
                  wraplength=600).pack(anchor='w', pady=(2, 6))
        mode_frame = ttk.Frame(frame)
        mode_frame.pack(fill='x', pady=(4, 0))
        ttk.Label(mode_frame, text='Results:').pack(side='left')
        self.view_mode = ttk.Combobox(mode_frame, values=('Points', 'Clusters'), state='readonly', width=14)
        self.view_mode.set('Points')
        self.view_mode.pack(side='left', padx=5)
        self.view_mode.bind('<<ComboboxSelected>>', lambda event: self.populate_results('Showing'))
        columns = tuple(spec[0] for spec in COLUMN_SPECS)
        self._columns = columns
        self._heading_labels = {spec[0]: spec[1] for spec in COLUMN_SPECS}
        self.column_vars = {key: tk.BooleanVar(master=self.window, value=key in DEFAULT_COLUMNS) for key in columns}
        table_frame = ttk.Frame(frame)
        table_frame.pack(fill='both', expand=True, pady=8)
        self.table_frame = table_frame
        self.table = ttk.Treeview(table_frame, columns=columns, show='headings', height=8, selectmode='extended')
        scrollbar = ttk.Scrollbar(table_frame, orient='vertical', command=self.table.yview)
        h_scrollbar = ttk.Scrollbar(table_frame, orient='horizontal', command=self.table.xview)
        self.table.configure(yscrollcommand=scrollbar.set, xscrollcommand=h_scrollbar.set)
        scrollbar.pack(side='right', fill='y')
        h_scrollbar.pack(side='bottom', fill='x')
        for spec in COLUMN_SPECS:
            self.table.column(spec[0], width=spec[3], minwidth=spec[3], stretch=False, anchor='center')
        self._refresh_headings()
        self.table.pack(side='left', fill='both', expand=True)
        self.table.bind('<Double-1>', self.view_selected)
        self.table.bind('<Control-a>', self.select_all)
        self.table.bind('<Control-A>', self.select_all)
        buttons = ttk.Frame(frame)
        # Pack actions before the expanding table so small windows do not clip them.
        buttons.pack(fill='x', before=table_frame, pady=(3, 0))
        actions = ttk.Frame(buttons)
        actions.pack(fill='x')
        more_actions = ttk.Frame(buttons)
        more_actions.pack(fill='x', pady=(3, 0))
        self.view_button = ttk.Button(actions, text='View selected', command=self.view_selected, state='disabled')
        self.view_button.pack(side='left')
        self.select_all_button = ttk.Button(actions, text='Select all', command=self.select_all, state='disabled')
        self.select_all_button.pack(side='left', padx=5)
        self.add_button = ttk.Button(actions, text='Add selected', command=self.add_selected, state='disabled')
        self.add_button.pack(side='left')
        self.analyze_button = ttk.Button(actions, text='Compare selected', command=self.analyze_selected, state='disabled')
        self.analyze_button.pack(side='left', padx=5)
        self.marks_button = ttk.Button(more_actions, text='Show all on globe', command=self.show_all, state='disabled')
        self.marks_button.pack(side='left', padx=5)
        self.sector_button = ttk.Button(more_actions, text='Search selected cluster', command=self.refine_selected_sector, state='disabled')
        self.sector_button.pack(side='left', padx=5)
        self.export_button = ttk.Button(more_actions, text='Save CSV', command=self.export, state='disabled')
        self.export_button.pack(side='left', padx=5)
        ttk.Button(more_actions, text='Close', command=self.close).pack(side='right')
        ttk.Button(more_actions, text='Clear markers', command=self.clear).pack(side='right', padx=5)
        # Column choice: a collapsible set of check boxes, in sections, right above the table heading.
        columns_bar = ttk.Frame(frame)
        columns_bar.pack(fill='x', before=table_frame)
        self.columns_toggle = ttk.Button(columns_bar, text='Choose columns', command=self._toggle_column_panel)
        self.columns_toggle.pack(side='left')
        ttk.Label(columns_bar, text='  Optional columns are calculated after the search.').pack(side='left')
        self.columns_panel = tk.Toplevel(self.window)
        self.columns_panel.title('Result columns')
        self.columns_panel.withdraw()
        self.columns_panel.protocol('WM_DELETE_WINDOW', self._toggle_column_panel)
        column_content = ttk.Frame(self.columns_panel, padding=14)
        column_content.pack(fill='both', expand=True)
        ttk.Label(column_content, text='Result columns', font=('TkDefaultFont', 14, 'bold')).pack(anchor='w')
        ttk.Label(column_content, text='Choose the information to show in the table.').pack(anchor='w', pady=(2, 8))
        column_tabs = ttk.Notebook(column_content)
        column_tabs.pack(fill='both', expand=True)
        self._columns_panel_open = False
        groups = []
        for spec in COLUMN_SPECS:
            if spec[0] not in PERCENT_COLUMNS:
                continue
            if spec[2] not in groups:
                groups.append(spec[2])
        for number, group in enumerate(groups):
            box = ttk.Frame(column_tabs, padding=10)
            tab_name = {'Visible % of the interval': 'Visibility', 'Sun per lunar day': 'Lunar day',
                        'Centre above the horizon': 'Above horizon',
                        'Directions (samples above the horizon)': 'Directions'}.get(group, group)
            column_tabs.add(box, text=tab_name)
            members = [spec for spec in COLUMN_SPECS if spec[2] == group and spec[0] in PERCENT_COLUMNS]
            for index, spec in enumerate(members):
                ttk.Checkbutton(box, text=spec[1], variable=self.column_vars[spec[0]],
                                command=self._columns_changed).grid(row=index // 2, column=index % 2, sticky='w', padx=6, pady=3)
        ttk.Button(column_content, text='Done', command=self._toggle_column_panel).pack(anchor='e', pady=(10, 0))
        self.window.bind('<Escape>', lambda event: self.close())
        if previous is not None:
            self.restore(previous)
        # The Tk root is withdrawn and is not the VTK window. Making this
        # dialog transient or grabbing it can leave it hidden behind VTK.
        self.window.deiconify()
        self.window.update_idletasks()
        self.window.lift()
        self.window.focus_force()
        self.poll()

    @staticmethod
    def _pair(text):
        values = text.replace(';', ',').split(',')
        if len(values) != 2:
            raise ValueError('Enter each coordinate range as min, max.')
        return tuple(map(float, values))

    def _set(self, name, value):
        entry = self.f[name]
        entry.configure(state='normal')  # a disabled entry ignores insert(); _update_lunar_state sets it again
        entry.delete(0, 'end')
        entry.insert(0, str(value))

    def _interval_days(self):
        try:
            return (sky.parse_utc(self.f['end'].get()) - sky.parse_utc(self.f['start'].get())).total_seconds() / 86400.0
        except Exception:
            return None

    def _update_lunar_state(self):
        """The lunar-day statistics are only offered when the interval is longer than 2 days."""
        days = self._interval_days()
        long_interval = days is not None and days > LONG_INTERVAL_DAYS
        use_terrain = bool(self.use_terrain.get())
        self.lunar_avg_check.configure(state='normal' if long_interval and use_terrain else 'disabled')
        self.lunar_sd_check.configure(state='normal' if long_interval and use_terrain and self.lunar_avg.get() else 'disabled')
        for name in ('lunar_min_avg', 'lunar_max_avg', 'lunar_min_sd', 'lunar_max_sd'):
            enabled = long_interval and use_terrain and (not name.endswith('_sd') or self.lunar_avg.get())
            self.f[name].configure(state='normal' if enabled else 'disabled')
        if days is None:
            note = 'Enter a valid start and end to see whether these options are available.'
        elif not use_terrain:
            note = 'Lunar-day statistics require a terrain-horizon search.'
        elif long_interval:
            note = (f'Interval: {days:,.1f} days (about {days / LUNAR_DAY_DAYS:,.2f} lunar days). The Sun visibility '
                    'is computed for each complete lunar day when selected or when a limit is entered. '
                    'Standard deviation requires the average. Blank limit = no limit.')
            if days < LUNAR_DAY_DAYS:
                note += ' Shorter than one lunar day: the whole interval counts as one day (SD 0).'
        else:
            note = f'Available when the interval is longer than {LONG_INTERVAL_DAYS:.0f} days (now {days:,.1f} days).'
        self.lunar_note.configure(text=note)

    def _refresh_headings(self):
        """Heading texts (with the sort arrow) and the columns that are ticked and apply to the current results."""
        sector = self.is_sector_results()
        labels = dict(self._heading_labels)
        if sector:
            labels['sun'], labels['earth'] = 'Sun optimistic %', 'Earth optimistic %'
        shown = [key for key in self._columns
                 if self.column_vars[key].get() and not (sector and key in SECTOR_HIDDEN)] or ['lat']
        self.table.configure(displaycolumns=shown)
        for key in self._columns:
            arrow = (' \u25bc' if self._sort[1] else ' \u25b2') if self._sort and self._sort[0] == key else ''
            self.table.heading(key, text=labels[key] + arrow, command=lambda k=key: self._sort_by(k))

    def _columns_changed(self):
        self._refresh_headings()
        if self.results:
            self.populate_results('Showing')
            self._maybe_fill()

    def _visible_keys(self):
        return [key for key in self._columns if self.column_vars[key].get()]

    def _fill_plan(self):
        """(rows that need work, need statistics, need max slope) for the ticked optional columns."""
        if not self.results or not self.finished or self.is_sector_results() or self.last_parameters is None:
            return [], False, False
        keys = set(self._visible_keys())
        stats_keys = keys & STATS_KEYS
        need_stats = bool(stats_keys)
        if need_stats and stats_keys <= LUNAR_KEYS:
            # Only the lunar-day columns: rows that already have them from the search are complete.
            rows_stats = [r for r in self.results if not r.get('_stats') and r.get('sun_lunar_mean') is None]
        else:
            rows_stats = [r for r in self.results if not r.get('_stats')] if need_stats else []
        need_slope = 'max_slope' in keys
        rows_slope = [r for r in self.results if r.get('max_slope_deg') is None] if need_slope else []
        wanted = {id(r): r for r in rows_stats}
        wanted.update({id(r): r for r in rows_slope})
        shown = {id(row) for members in self._display_items for row in members}
        rows = sorted(wanted.values(), key=lambda r: id(r) not in shown)  # rows on screen first
        return rows, bool(rows_stats), bool(rows_slope)

    def _maybe_fill(self):
        """Start the background calculation of the ticked optional columns, if something is missing."""
        if self._fill_running or self._closed:
            return
        rows, need_stats, need_slope = self._fill_plan()
        if not rows:
            return
        start, stop, step_hours = self.last_parameters[:3]
        try:
            epochs = sky.time_samples(start, stop, float(step_hours) * 60)
        except Exception as error:
            self.fill_label.configure(text=f'Optional columns unavailable: {error}')
            return
        radius = self.last_parameters[15] if len(self.last_parameters) > 15 else 1.0
        self._fill_generation += 1
        self._fill_cancel = threading.Event()
        self._fill_running = True
        self._fill_dirty = False
        self._fill_refreshed = time.monotonic()
        self.fill_label.configure(text=f'Calculating optional columns for {len(rows):,} points (from the shared data where stored)...')
        threading.Thread(target=_fill_worker, daemon=True,
                         args=(self.messages, self._fill_cancel, self._fill_generation, rows, epochs, self.fetch,
                               self.cache, need_stats, need_slope, radius, self.workers_setting)).start()

    def _stop_fill(self):
        self._fill_cancel.set()
        self._fill_generation += 1  # messages of the stopped calculation are ignored
        self._fill_running = False
        self.fill_label.configure(text='')

    def _sort_by(self, key):
        self._sort = next_sort(self._sort, key)
        self._refresh_headings()
        if self.results:
            self.populate_results('Sorted' if self._sort else 'Unsorted')

    def start(self):
        if self.worker is not None and self.worker.is_alive():
            return
        try:
            get = lambda name: self.f[name].get()
            start, stop = sky.parse_utc(get('start')), sky.parse_utc(get('end'))
            step_hours, spacing = float(get('step')), float(get('spacing'))
            min_sun, max_sun = float(get('min_sun')), float(get('max_sun'))
            min_earth, max_earth = float(get('min_earth')), float(get('max_earth'))
            lat_min, lat_max = self._pair(get('lat'))
            lon_min, lon_max = self._pair(get('lon'))
            use_terrain = bool(self.use_terrain.get())
            if use_terrain and self._refinement_sectors is not None:
                cells = subdivide_cluster(self._refinement_sectors, (lat_min, lat_max, lon_min, lon_max), spacing)
            else:
                cells = grid_cells(lat_min, lat_max, lon_min, lon_max, spacing)
            side_value = float(get('side'))
            if not np.isfinite(side_value) or not side_value.is_integer() or not 2 <= side_value <= 9:
                raise ValueError('Samples per side must be an integer from 2 to 9.')
            side = int(side_value)
            horizon = float(get('horizon'))
            if not np.isfinite(horizon) or not -90 <= horizon <= 90:
                raise ValueError('Screen horizon must be between -90 and 90 degrees.')
            slope_text = get('max_slope').strip()
            max_slope = float(slope_text) if slope_text else None
            radius = float(get('slope_radius'))
            if max_slope is not None and (not np.isfinite(max_slope) or not 0 <= max_slope <= 90):
                raise ValueError('Maximum slope must be between 0 and 90 degrees.')
            if not np.isfinite(radius) or not 0 <= radius <= 20:
                raise ValueError('Slope radius must be between 0 and 20 km.')
            mode = self.screen_mode.get()
            range_mode = self.range_mode.get()
            if range_mode not in ('On', 'Off'):
                raise ValueError('Choose a valid optimistic range option.')
            if mode not in ('Pessimistic center', 'All cells'):
                raise ValueError('Choose a valid cell screening mode.')
            epochs = sky.time_samples(start, stop, step_hours * 60)
            if len(epochs) < 2:
                raise ValueError('End time must be later than start time.')
            workers_value = float((get('workers') if 'workers' in self.f else '0') or 0)
            if not np.isfinite(workers_value) or not workers_value.is_integer() or not 0 <= workers_value <= 128:
                raise ValueError('Worker processes must be an integer from 0 (automatic) to 128.')
            workers = int(workers_value)
            limits = np.array([min_sun, max_sun, min_earth, max_earth])
            if not np.isfinite(limits).all() or np.any(limits < 0) or np.any(limits > 100):
                raise ValueError('Visibility limits must be between 0 and 100 percent.')
            if min_sun > max_sun:
                raise ValueError('Minimum Sun visibility must not exceed the maximum.')
            if min_earth > max_earth:
                raise ValueError('Minimum Earth visibility must not exceed the maximum.')
            long_interval = (stop - start).total_seconds() / 86400.0 > LONG_INTERVAL_DAYS
            lunar_avg = bool(self.lunar_avg.get()) and long_interval and use_terrain
            lunar_sd = bool(self.lunar_sd.get()) and lunar_avg
            lunar_filter = None
            if long_interval and use_terrain:
                def limit(name):
                    text = get(name).strip()
                    return float(text) if text else None
                limits = (limit('lunar_min_avg'), limit('lunar_max_avg'),
                          limit('lunar_min_sd') if self.lunar_avg.get() else None,
                          limit('lunar_max_sd') if self.lunar_avg.get() else None)
                if any(v is not None for v in limits):
                    lunar_filter = limits
                    # A filtered metric is always shown, so the table explains why a point passed.
                    lunar_avg = True
                    lunar_sd = lunar_sd or limits[2] is not None or limits[3] is not None
                    for low, high, name in ((limits[0], limits[1], 'average Sun %'), (limits[2], limits[3], 'standard deviation')):
                        if any(v is not None and (not np.isfinite(v) or not 0 <= v <= 100) for v in (low, high)):
                            raise ValueError(f'The {name} limits must be between 0 and 100.')
                        if low is not None and high is not None and low > high:
                            raise ValueError(f'The minimum {name} must not exceed the maximum.')
        except (ValueError, OverflowError) as error:
            self.status.configure(text=str(error))
            return
        self._stop_fill()
        self.workers_setting = workers
        self.cancel.clear()
        self._refinement_sectors = None
        self.results = []
        self.selected = None
        self.show_matches = False
        self.marker_results = None
        self.finished = False
        self.last_parameters = (start, stop, step_hours, min_sun, min_earth, spacing, lat_min, lat_max, lon_min, lon_max,
                                side, horizon, mode, range_mode, max_slope, radius, use_terrain,
                                max_sun, max_earth, lunar_avg, lunar_sd, lunar_filter)
        self.view_mode.configure(values=('Points', 'Clusters') if use_terrain else ('Clusters',))
        self.view_mode.set('Points' if use_terrain else 'Clusters')
        if lunar_avg:  # the lunar-day values are being calculated: show them (they can be unticked again)
            self.column_vars['lunar_mean'].set(True)
        if lunar_sd:
            self.column_vars['lunar_sd'].set(True)
        self._refresh_headings()
        self.table.delete(*self.table.get_children())
        for button in (self.view_button, self.select_all_button, self.add_button, self.analyze_button, self.marks_button, self.export_button, self.sector_button):
            button.configure(state='disabled')
        self.start_button.configure(state='disabled')
        self.cancel_button.configure(state='normal')
        self.progress.configure(maximum=len(cells), value=0)
        self.range_label.configure(text='')
        self.horizon_label.configure(text='')
        if use_terrain:
            text = f'Preparing {len(cells):,} terrain cells, up to {side * side + (side % 2 == 0)} samples each...'
        else:
            text = f'Preparing {len(cells):,} optimistic sectors; terrain horizon and slope are not checked...'
            if max_sun < 100 or max_earth < 100:
                text += ' (maximum Sun / Earth limits are ignored here)'
        self.status.configure(text=text)
        self.worker = threading.Thread(target=_search_worker,
            args=(self.messages, self.cancel, cells, epochs, min_sun, min_earth, self.fetch, side, horizon, mode,
                  range_mode == 'On', max_slope, radius, use_terrain, max_sun, max_earth, lunar_avg, lunar_filter,
                  self.cache, workers),
            daemon=True)
        self.worker.start()

    def poll(self):
        try:
            while True:
                kind, value = self.messages.get_nowait()
                if kind.startswith('fill_'):
                    self._handle_fill_message(kind, value)
                    continue
                if kind == 'phase':
                    self.status.configure(text=value)
                elif kind == 'optimistic_horizon':
                    low, median, floor = value
                    self.horizon_label.configure(text=f'Optimistic horizon {floor:+.2f} deg | median elevation {median / 1000:+.3f} km | lowest {low / 1000:+.3f} km')
                elif kind == 'range_progress':
                    done, total, eligible = value
                    self.progress.configure(value=done)
                    self.status.configure(text=f'Estimating optimistic ranges: {done:,}/{total:,} cells | {eligible:,} possible')
                elif kind == 'ranges':
                    lat_ranges, lon_ranges, eligible, total = value
                    def fmt(ranges):
                        text = ', '.join(f'{a:g} to {b:g}' for a, b in ranges[:5])
                        return text + (' ...' if len(ranges) > 5 else '') if text else 'none'
                    self.range_label.configure(text=f'Approximate latitude / longitude unions: lat {fmt(lat_ranges)}; lon {fmt(lon_ranges)} ({eligible:,}/{total:,} cells)')
                elif kind == 'progress':
                    done, total, matches, elapsed, lat, lon, filtered = value
                    self.progress.configure(value=done)
                    eta = f' | estimated {elapsed / done * (total - done) / 60:.1f} min left' if done else ''
                    where = f' | {lat:+.2f}, {lon:+.2f}' if lat is not None else ''
                    self.status.configure(text=f'{done:,}/{total:,} cells | {matches:,} matches | {filtered:,} filtered | {elapsed / 60:.1f} min elapsed{eta}{where}')
                elif kind == 'done':
                    self.results, cancelled = value
                    self.finished = True
                    self.was_cancelled = cancelled
                    self.populate_results('Cancelled' if cancelled else 'Complete')
                    self.start_button.configure(state='normal')
                    self.cancel_button.configure(state='disabled')
                    if self.workspace_tabs is not None:
                        self.workspace_tabs.select(self.workspace_results)
                    self._maybe_fill()
                else:
                    self.status.configure(text=value)
                    self.start_button.configure(state='normal')
                    self.cancel_button.configure(state='disabled')
        except queue.Empty:
            pass
        self._tick += 1
        if self.cache is not None and self._tick % 10 == 1:
            count, megabytes = self.cache.info()
            self.cache_label.configure(
                text=f'Shared data: {count:,} points stored in memory ({megabytes:,.0f} MB); the comparison reuses them.')
        self.timer = self.window.after(100, self.poll)

    def _handle_fill_message(self, kind, value):
        generation = value[0]
        if generation != self._fill_generation:
            return  # belongs to a stopped calculation or an older search
        if kind == 'fill_row':
            row, update = value[1], value[2]
            row.update(update)
            self._fill_dirty = True
        elif kind == 'fill_progress':
            self.fill_label.configure(text=f'Calculating optional columns: {value[1]:,}/{value[2]:,} points')
            if self._fill_dirty and time.monotonic() - self._fill_refreshed >= FILL_REFRESH_SECONDS:
                self._fill_dirty, self._fill_refreshed = False, time.monotonic()
                self.populate_results('Showing')
        else:  # fill_done / fill_error
            self._fill_running = False
            if kind == 'fill_error':
                self.fill_label.configure(text=f'Optional columns stopped: {value[1]}')
            else:
                self.fill_label.configure(text='Optional columns ready.' if not value[1] else '')
            if self._fill_dirty:
                self._fill_dirty = False
                self.populate_results('Showing')
            if kind == 'fill_done' and not value[1]:
                self._maybe_fill()  # columns ticked while it was running

    def _clusters(self):
        if self.last_parameters:
            start, end = self.last_parameters[8:10]
            wrap = end - start >= 360.0 - 1e-8
            bounds = (start, start + 360.0)
        else:
            wrap, bounds = False, None
        key = (id(self.results), len(self.results), wrap, bounds)
        if self._cluster_cache is None or self._cluster_cache[0] != key:
            self._cluster_cache = (key, cluster_results(self.results, wrap_longitude=wrap, longitude_bounds=bounds))
        return self._cluster_cache[1]

    def _marker_rows(self, members=None):
        clusters = self._clusters()
        cluster_for_row = {id(row): index for index, cluster in enumerate(clusters) for row in cluster}
        source = self.results if members is None else members
        return [dict(row, cluster_index=cluster_for_row[id(row)]) for row in source]

    def _group_values(self, members, clustered, sector):
        """Numbers shown (and sorted) for one table row: a point, or a cluster of cells."""
        row = members[0]
        values = {
            'lat': None if sector else row['lat'], 'lon': None if sector else row['lon'],
            'sun': max(m['sun_percent'] for m in members) if sector else row['sun_percent'],
            'earth': max(m['earth_percent'] for m in members) if sector else row['earth_percent'],
            'cells': len(members) if clustered else 1,
        }
        for spec in COLUMN_SPECS:  # every other column is a plain value of the row (None when not calculated)
            if spec[0] not in values and spec[4] is not None:
                values[spec[0]] = None if sector else row.get(spec[4])
        if not sector and values['elevation'] is None and self.column_vars['elevation'].get():
            row['elevation_m'] = values['elevation'] = float(terrain.sample_dem_bilinear(row['lat'], row['lon']))
        lat_min = min(x.get('cell_lat_min', x['lat']) for x in members)
        lon_min = min(x.get('cell_lon_min', x['lon']) for x in members)
        sort = dict(values, cell=(lat_min, lon_min))
        if sector:  # candidate sectors have no point: sort by where they are
            sort['lat'], sort['lon'] = lat_min, lon_min
        return values, sort

    def _sorted(self, entries):
        """Order (members, display values, sort values) entries by the clicked column; empty values go last."""
        if self._sort is None:
            return entries
        key, descending = self._sort
        present = [e for e in entries if e[2].get(key) is not None]
        missing = [e for e in entries if e[2].get(key) is None]
        present.sort(key=lambda e: e[2][key], reverse=descending)
        return present + missing

    def populate_results(self, label):
        previous = {id(self._display_items[int(i)][0]) for i in self.table.selection()
                    if int(i) < len(self._display_items)}
        self.table.delete(*self.table.get_children())
        clusters = self._clusters()
        cluster_for_row = {id(row): index for index, cluster in enumerate(clusters) for row in cluster}
        clustered = self.view_mode.get() == 'Clusters'
        sector_cluster = self.is_sector_results()
        groups = clusters if clustered else [[row] for row in self.results]
        entries = []
        for members in groups:
            values, sort_values = self._group_values(members, clustered, sector_cluster)
            entries.append((members, values, sort_values))
        entries = self._sorted(entries)
        self._display_items = [entry[0] for entry in entries[:MAX_DISPLAY_RESULTS]]
        number = lambda value, form: '-' if value is None else format(value, form)
        reselect = []
        for index, (members, values, _) in enumerate(entries[:MAX_DISPLAY_RESULTS]):
            row = members[0]
            cluster_index = cluster_for_row[id(row)]
            if clustered:
                lat_min = min(x.get('cell_lat_min', x['lat']) for x in members)
                lat_max = max(x.get('cell_lat_max', x['lat']) for x in members)
                lon_intervals = interval_union((x.get('cell_lon_min', x['lon']), x.get('cell_lon_max', x['lon'])) for x in members)
                lon_text = ','.join(f'{a:g}..{b:g}' for a, b in lon_intervals[:2])
                if len(lon_intervals) > 2:
                    lon_text += ',...'
                area = f"Cluster {cluster_index + 1}: {len(members)} cells; lat {lat_min:g}..{lat_max:g}, lon {lon_text}"
            else:
                area = (f"{row.get('cell_lat_min', row['lat']):g}..{row.get('cell_lat_max', row['lat']):g} / "
                        f"{row.get('cell_lon_min', row['lon']):g}..{row.get('cell_lon_max', row['lon']):g}")
            tag = f'cluster_{cluster_index}'
            self.table.tag_configure(tag, foreground=cluster_text_color(cluster_index))
            texts = {'cells': values['cells'], 'cell': area, 'sun': f"{values['sun']:.1f}",
                     'earth': f"{values['earth']:.1f}"}
            self.table.insert('', 'end', iid=str(index), tags=(tag,), values=tuple(
                texts[key] if key in texts else SPEC[key][5](values[key], row) for key in self._columns))
            if id(row) in previous:
                reselect.append(str(index))
        if reselect:
            self.table.selection_set(reselect)
        suffix = (f' Globe shows at most {MAX_DISPLAY_RESULTS} points; CSV contains all points.'
                  if not sector_cluster and len(self.results) > MAX_DISPLAY_RESULTS else '')
        if len(groups) > MAX_DISPLAY_RESULTS:
            suffix += f' Table shows the first {MAX_DISPLAY_RESULTS}' + (' in the chosen order.' if self._sort else '.')
        noun = 'clusters' if clustered else ('candidate sectors' if sector_cluster else 'points')
        if len(groups) == 1:
            noun = noun[:-1]
        explanation = ' Candidate clusters; optimistic estimates may miss high sites and differ by sector. Select a cluster to search.' if sector_cluster else ''
        self.status.configure(text=f"{label}: {len(groups):,} {noun}, {len(self.results):,} matching cells.{suffix}{explanation}")
        if self.results:
            for button in (self.select_all_button, self.marks_button, self.export_button):
                button.configure(state='normal')
            for button in (self.view_button, self.add_button, self.analyze_button):
                button.configure(state='disabled' if sector_cluster else 'normal')
            self.sector_button.configure(state='normal' if sector_cluster else 'disabled')

    def is_sector_results(self):
        return bool(self.last_parameters and len(self.last_parameters) > 16 and not self.last_parameters[16])

    def restore(self, previous):
        self.results = previous['results']
        self.last_parameters = previous['parameters']
        self.was_cancelled = previous['cancelled']
        self.finished = True
        parameters = self.last_parameters
        start, stop, step, sun, earth, spacing, lat_min, lat_max, lon_min, lon_max = parameters[:10]
        side, horizon, mode = parameters[10:13] if len(parameters) >= 13 else (3, 2.0, "Pessimistic center")
        range_mode = parameters[13] if len(parameters) > 13 else 'On'
        self.screen_mode.set(mode)
        self.range_mode.set(range_mode)
        sector = self.is_sector_results()
        self.use_terrain.set(not sector)
        self.view_mode.configure(values=('Clusters',) if sector else ('Points', 'Clusters'))
        self.view_mode.set('Clusters' if sector else 'Points')
        values = {'start': start.strftime('%Y-%m-%d %H:%M:%S'), 'end': stop.strftime('%Y-%m-%d %H:%M:%S'),
                  'step': step, 'min_sun': sun, 'min_earth': earth,
                  'max_sun': parameters[17] if len(parameters) > 17 else 100,
                  'max_earth': parameters[18] if len(parameters) > 18 else 100,
                  'spacing': spacing, 'lat': f'{lat_min:g}, {lat_max:g}', 'lon': f'{lon_min:g}, {lon_max:g}',
                  'side': side, 'horizon': horizon,
                  'max_slope': parameters[14] if len(parameters) > 14 and parameters[14] is not None else '',
                  'slope_radius': parameters[15] if len(parameters) > 15 else 1}
        for name, value in values.items():
            self._set(name, value)
        saved = parameters[21] if len(parameters) > 21 and parameters[21] else (None,) * 4
        for name, value in zip(('lunar_min_avg', 'lunar_max_avg', 'lunar_min_sd', 'lunar_max_sd'), saved):
            self._set(name, '' if value is None else value)
        self.lunar_avg.set(len(parameters) > 19 and bool(parameters[19]))
        self.lunar_sd.set(len(parameters) > 20 and bool(parameters[20]))
        if self.lunar_avg.get():
            self.column_vars['lunar_mean'].set(True)
        if self.lunar_sd.get():
            self.column_vars['lunar_sd'].set(True)
        self._update_lunar_state()
        self._refresh_headings()
        self.populate_results('Previous partial search' if self.was_cancelled else 'Previous search')
        self._maybe_fill()

    def view_selected(self, event=None):
        if self.is_sector_results():
            self.refine_selected_sector()
            return
        selected = self.table.selection()
        if selected:
            members = self._display_items[int(selected[0])]
            self.selected = members[0]
            self.marker_mode = self.view_mode.get()
            self.marker_results = self._marker_rows(members if self.marker_mode == 'Clusters' else None)
            self.show_matches = True
            self.apply_to_viewer('Shown on the globe.')

    def select_all(self, event=None):
        rows = self.table.get_children()
        self.table.selection_set(rows)
        self.status.configure(text=(f'Selected {len(rows):,} clusters; select one to search.' if self.is_sector_results()
                                    else f'Selected {len(rows):,} displayed rows. Comparisons hold at most {MAX_POINTS} points.'))
        return 'break'

    def add_selected(self, analyze=False):
        if self.is_sector_results():
            self.status.configure(text='Candidate clusters are not verified points. Search a selected cluster first.')
            return
        selected = self.table.selection()
        if not selected:
            self.status.configure(text='Select one or more rows first.')
            return
        points = list(self.existing_points())
        rows = []
        for item in selected:
            for row in self._display_items[int(item)]:
                point = normalize_point(row['lat'], row['lon'])
                if not any(same_point(point, other) for other in points):
                    points.append(point)
                    if len(points) > MAX_POINTS:
                        self.status.configure(text=f'At most {MAX_POINTS} points can be selected, including existing points. Select fewer rows.')
                        return
                if not any(same_point(point, normalize_point(r['lat'], r['lon'])) for r in rows):
                    rows.append(row)
        self.selected_points = rows
        self.selected = rows[0]
        self.add_to_compare = True
        self.analyze = analyze
        self.marker_mode = self.view_mode.get()
        self.marker_results = self._marker_rows()
        self.show_matches = True
        self.apply_to_viewer(f'Added {len(rows):,} point(s) to the comparison set; analysis opened.' if analyze
                             else f'Added {len(rows):,} point(s) to the comparison set.')

    def analyze_selected(self):
        self.add_selected(analyze=True)

    def refine_selected_sector(self):
        if not self.is_sector_results():
            self.status.configure(text='Run an optimistic sector search first.')
            return
        if self.worker is not None and self.worker.is_alive():
            self.status.configure(text='Wait for the search to finish or cancel it first.')
            return
        selected = self.table.selection()
        if len(selected) != 1:
            self.status.configure(text='Select exactly one cluster.')
            return
        members = self._display_items[int(selected[0])]
        self._refinement_sectors = [(row['cell_lat_min'], row['cell_lat_max'], row['cell_lon_min'], row['cell_lon_max'])
                                    for row in members]
        lat0 = min(cell[0] for cell in self._refinement_sectors)
        lat1 = max(cell[1] for cell in self._refinement_sectors)
        lon0 = min(cell[2] for cell in self._refinement_sectors)
        lon1 = max(cell[3] for cell in self._refinement_sectors)
        spacing = max(0.1, min(min(cell[1] - cell[0], cell[3] - cell[2]) for cell in self._refinement_sectors) / 5.0)
        for name, value in (('spacing', f'{spacing:.6g}'), ('lat', f'{lat0:g}, {lat1:g}'), ('lon', f'{lon0:g}, {lon1:g}')):
            self._set(name, value)
        self.use_terrain.set(True)
        if self.workspace_tabs is not None:
            self.workspace_tabs.select(self.workspace_settings)
        self.settings_tabs.select(self.area_tab)
        self.status.configure(text=f'{len(members):,} cluster sectors selected. Adjust parameters, then start the terrain search; only these sectors will be checked.')

    def select_sector_on_globe(self, row):
        """Select a displayed sector, then prepare the terrain search form."""
        if not self.is_sector_results():
            self.status.configure(text='The previous sector results are no longer available.')
            return
        self.view_mode.set('Clusters')
        self.populate_results('Candidate clusters')
        keys = ('cell_lat_min', 'cell_lat_max', 'cell_lon_min', 'cell_lon_max')
        for index, members in enumerate(self._display_items):
            if any(all(member.get(key) == row.get(key) for key in keys) for member in members):
                item = str(index)
                self.table.selection_set(item)
                self.table.see(item)
                self.refine_selected_sector()
                return
        self.status.configure(text='Cluster is outside the displayed rows; use the table or narrow the search.')

    def show_all(self):
        self.marker_mode = 'Clusters' if self.is_sector_results() else self.view_mode.get()
        self.marker_results = self._marker_rows()
        self.show_matches = True
        self.apply_to_viewer('All results shown on the globe.')

    def clear(self):
        self.clear_markers = True
        self.apply_to_viewer('Markers cleared.')

    def apply_to_viewer(self, message=None):
        """Let the viewer act on the one-shot request flags, then keep this window open."""
        try:
            if self.on_apply is not None:
                self.on_apply(self)
        finally:
            # The request has been handed over; do not repeat it when the window is closed later.
            self.show_matches = self.clear_markers = self.add_to_compare = self.analyze = False
        if message and not (self.worker is not None and self.worker.is_alive()):
            self.status.configure(text=message)

    def export_to(self, path):
        """Write results to a path selected by the active desktop interface."""
        try:
            with open(path, 'w', newline='', encoding='utf-8') as output:
                writer = csv.writer(output)
                if self.is_sector_results():
                    writer.writerow(('cluster', 'sector_count', 'lat_min', 'lat_max', 'longitude_intervals',
                                     'max_sun_optimistic_percent', 'max_earth_optimistic_percent', 'partial_search'))
                    for i, members in enumerate(self._clusters(), 1):
                        intervals = interval_union((m['cell_lon_min'], m['cell_lon_max']) for m in members)
                        writer.writerow((i, len(members), min(m['cell_lat_min'] for m in members),
                                         max(m['cell_lat_max'] for m in members),
                                         '; '.join(f'{a:g}..{b:g}' for a, b in intervals),
                                         max(m['sun_percent'] for m in members), max(m['earth_percent'] for m in members),
                                         self.was_cancelled))
                    self.status.configure(text=f'Saved {len(self._clusters()):,} candidate clusters to {path}')
                    return
                # The visible table columns first (raw numbers), then the settings of this search.
                exported = [SPEC[key] for key in self._visible_keys() if key not in ('cells', 'cell')]
                with_cell = self.column_vars['cell'].get()
                parameters = self.last_parameters
                start, stop, step, sun, earth, spacing, lat_min, lat_max, lon_min, lon_max = parameters[:10]
                side, horizon, mode = parameters[10:13] if len(parameters) >= 13 else (3, 2.0, "Pessimistic center")
                max_sun = parameters[17] if len(parameters) > 17 else 100
                max_earth = parameters[18] if len(parameters) > 18 else 100
                lunar_limits = parameters[21] if len(parameters) > 21 and parameters[21] else (None,) * 4
                writer.writerow(('result_kind', *[spec[6] for spec in exported],
                                 *(('cell_latitude_min', 'cell_latitude_max', 'cell_longitude_min',
                                    'cell_longitude_max') if with_cell else ()),
                                 'slope_limit_deg', 'slope_radius_km', 'start_utc', 'end_utc', 'step_hours',
                                 'min_sun_percent', 'min_earth_percent', 'max_sun_percent', 'max_earth_percent',
                                 'grid_spacing_deg', 'latitude_min', 'latitude_max', 'longitude_min', 'longitude_max',
                                 'partial_search', 'samples_per_side', 'screen_horizon_deg', 'screen_mode',
                                 'optimistic_range_filter', 'complete_lunar_days',
                                 'min_avg_sun_per_lunar_day', 'max_avg_sun_per_lunar_day',
                                 'min_sd_sun_per_lunar_day', 'max_sd_sun_per_lunar_day'))
                pending = 0
                for row in self.results:
                    values = []
                    for spec in exported:
                        value = row.get(spec[4])
                        if value is None and spec[0] == 'elevation':
                            value = float(terrain.sample_dem_bilinear(row['lat'], row['lon']))
                        if value is None and spec[0] in STATS_KEYS:
                            pending += 1
                        values.append(value)
                    writer.writerow((row.get('kind', 'point'), *values,
                                     *((row.get('cell_lat_min'), row.get('cell_lat_max'), row.get('cell_lon_min'),
                                        row.get('cell_lon_max')) if with_cell else ()),
                                     parameters[14] if len(parameters) > 14 else None,
                                     parameters[15] if len(parameters) > 15 else None,
                                     start.isoformat(), stop.isoformat(), step, sun, earth, max_sun, max_earth,
                                     spacing, lat_min, lat_max, lon_min, lon_max, self.was_cancelled,
                                     side, horizon, mode, parameters[13] if len(parameters) > 13 else 'Off',
                                     row.get('lunar_days'), *lunar_limits))
            if pending and self._fill_running:
                self.status.configure(text=f'Saved {len(self.results):,} results to {path}; optional columns that '
                                           'were still being calculated are empty.')
                return
            self.status.configure(text=f'Saved {len(self.results):,} results to {path}')
        except OSError as error:
            self.status.configure(text=f'Could not save results: {error}')
