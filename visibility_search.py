"""Coarse, cancellable lunar surface search for terrain-aware visibility.

Run via S in moon.py. Skyfield is used locally. Percentages are estimated by
trapezoidal integration of sampled visibility (partly visible counts; Earth
occulting the Sun is included). An optional sampled maximum slope filters sites.
"""
import csv
import colorsys
import queue
import threading
import time
from datetime import timedelta

import numpy as np
import sky
import terrain
from compare import MAX_POINTS, normalize_point, same_point

LUNAR_DAY_DAYS = 29.530588861
MAX_GRID_POINTS = 100_000
MAX_DISPLAY_RESULTS = 500


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


def visibility_percent(epochs, mask):
    """Time-weight samples with midpoint crossings; handles a short last step."""
    if len(epochs) < 2:
        raise ValueError('Start and end must be different.')
    values = np.asarray(mask, dtype=float)
    if values.shape != (len(epochs),):
        raise ValueError('Visibility result does not match the time samples.')
    durations = np.array([(b - a).total_seconds() for a, b in zip(epochs[:-1], epochs[1:])])
    if np.any(durations <= 0):
        raise ValueError('Times must be in strictly increasing order.')
    return float(100 * np.dot(durations, (values[:-1] + values[1:]) * 0.5) / durations.sum())


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
    observer_radius = terrain.MOON_RADIUS_M + observer_elevation_m + 2.0
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


def search(points, epochs, min_sun, min_earth, fetch, cancel=None, progress=None):
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
            altitude = (elevation_m + 2.0) / 1000.0
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
            horizon = terrain.compute_horizon_360(lat, lon)[0]
            if cancel is not None and cancel.is_set():
                break
            classified = sky.classify_sky(positions, horizon)
            sun = visibility_percent(epochs, classified['Sun']['visible'])
            earth = visibility_percent(epochs, classified['Earth']['visible'])
        except Exception as error:
            raise RuntimeError(f'Search failed at latitude {lat:.3f}, longitude {lon:.3f}: {error}') from error
        completed += 1
        if sun >= min_sun and earth >= min_earth:
            results.append({'lat': lat, 'lon': lon, 'sun_percent': sun, 'earth_percent': earth})
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


def optimistic_cell_ranges(cells, epochs, min_sun, min_earth, fetch, cancel=None, progress=None):
    """Screen broad 4-by-4 cell blocks using optimistic visibility estimates.

    The center is allowed the full angular width of the block, a generous
    parallax margin and a pessimistically low terrain bound. This is an
    approximate range screen, not a guarantee for every point in the block.
    """
    min_height = float(np.min(terrain.dem_km)) * 1000.0
    max_height = float(np.max(terrain.dem_km)) * 1000.0
    floor = optimistic_horizon_floor(max_height, min_height)
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
            positions = fetch(center[1], center[0], (height + 2.0) / 1000.0, epochs)
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
                 use_ranges=True, range_progress=None, range_result=None, max_slope=None, slope_radius_km=1.0):
    """Return at most one terrain-checked match per cell.

    A pessimistic center estimate determines the order in which samples are
    terrain-checked; it never rejects a cell. Refinement tests a finite grid,
    so it is not an exhaustive or guaranteed search of any cell.
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
    if max_slope is not None and (not np.isfinite(max_slope) or not 0 <= max_slope <= 90):
        raise ValueError('Maximum slope must be between 0 and 90 degrees.')
    if not np.isfinite(slope_radius_km) or not 0 <= slope_radius_km <= 20:
        raise ValueError('Slope radius must be between 0 and 20 km.')
    if len(epochs) < 2 or epochs[-1] <= epochs[0]:
        raise ValueError("End time must be later than start time.")
    min_terrain = float(np.min(terrain.dem_km)) * 1000.0
    if not np.isfinite(min_terrain):
        raise ValueError("The elevation map contains invalid values.")
    start = time.monotonic()
    total_cells = len(cells)
    precomputed = {}
    if use_ranges:
        eligible, precomputed, lat_ranges, lon_ranges = optimistic_cell_ranges(
            cells, epochs, min_sun, min_earth, fetch, cancel, range_progress)
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
            if screen_mode == "Pessimistic center":
                if center_positions is None:
                    elevation = float(terrain.sample_dem_bilinear(*center))
                    center_positions = fetch(center[1], center[0], (elevation + 2.0) / 1000.0, epochs)
                if cancel is not None and cancel.is_set():
                    break
                # Validate even when thresholds are zero.
                can_meet_thresholds_without_horizon(epochs, center_positions, -90.0, 0.0, 0.0)
                approx = sky.classify_sky(center_positions, np.full(360, screen_horizon))
                if (visibility_percent(epochs, approx["Sun"]["visible"]) < min_sun or
                        visibility_percent(epochs, approx["Earth"]["visible"]) < min_earth):
                    # A conservative horizon can fail even where the real
                    # terrain passes. Try the other samples first, then the
                    # center itself before giving up on this cell.
                    points = points[1:] + points[:1]
            for point in points:
                if cancel is not None and cancel.is_set():
                    break
                if point not in checked:
                    elevation = float(terrain.sample_dem_bilinear(*point))
                    pos = (center_positions if point == center and center_positions is not None else
                           fetch(point[1], point[0], (elevation + 2.0) / 1000.0, epochs))
                    if cancel is not None and cancel.is_set():
                        break
                    floor = optimistic_horizon_floor(elevation, min_terrain)
                    if can_meet_thresholds_without_horizon(epochs, pos, floor, min_sun, min_earth):
                        horizon = terrain.compute_horizon_360(*point)[0]
                        classified = sky.classify_sky(pos, horizon)
                        sun = visibility_percent(epochs, classified["Sun"]["visible"])
                        earth = visibility_percent(epochs, classified["Earth"]["visible"])
                        if sun >= min_sun and earth >= min_earth:
                            slope = terrain.compute_slope_deg(*point)
                            area_slope = (terrain.max_slope_in_radius(*point, slope_radius_km)
                                          if max_slope is not None else None)
                            checked[point] = ((sun, earth, slope, area_slope)
                                              if area_slope is None or area_slope <= max_slope else None)
                        else:
                            checked[point] = None
                    else:
                        checked[point] = None
                value = checked[point]
                if value is not None and value[0] >= min_sun and value[1] >= min_earth:
                    results.append({"lat": point[0], "lon": point[1], "sun_percent": value[0],
                                    "earth_percent": value[1], "slope_deg": value[2], "max_slope_deg": value[3],
                                    "cell_lat_min": cell[0], "cell_lat_max": cell[1],
                                    "cell_lon_min": cell[2], "cell_lon_max": cell[3]})
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


def _search_worker(messages, cancel, cells, epochs, sun, earth, fetch, side, horizon, mode, use_ranges,
                   max_slope=None, slope_radius_km=1.0):
    def progress(done, total, matches, elapsed, lat, lon, filtered):
        messages.put(('progress', (done, total, matches, elapsed, lat, lon, filtered)))
    try:
        def range_progress(done, total, eligible, center):
            messages.put(('range_progress', (done, total, eligible)))
        def range_result(lat_ranges, lon_ranges, eligible, total):
            messages.put(('ranges', (lat_ranges, lon_ranges, eligible, total)))
        results = search_cells(cells, epochs, sun, earth, fetch, side, horizon, mode, cancel, progress,
                               use_ranges, range_progress, range_result, max_slope, slope_radius_km)
        messages.put(('done', (results, cancel.is_set())))
    except Exception as error:
        messages.put(('error', str(error)))


class SearchDialog:
    """Tk-only UI; all horizon and ephemeris work stays in a worker thread."""
    def __init__(self, parent, fetch, previous=None, on_close=None, existing_points=None):
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.ttk, self.fetch = tk, ttk, fetch
        self.on_close = on_close
        self.existing_points = existing_points or (lambda: [])
        self._closed = False
        self.window = tk.Toplevel(parent)
        self.window.title('Search lunar visibility')
        self.window.protocol('WM_DELETE_WINDOW', self.close)
        width = min(1120, max(640, self.window.winfo_screenwidth() - 40))
        height = min(820, max(480, self.window.winfo_screenheight() - 80))
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
        self.finished = False
        self.was_cancelled = False
        frame = ttk.Frame(self.window, padding=12)
        frame.pack(fill='both', expand=True)
        settings = ttk.Frame(frame)
        settings.pack(fill='x')
        now = sky.utc_now()
        defaults = [
            ('Start UTC', now.strftime('%Y-%m-%d %H:%M:%S')),
            ('End UTC', (now + timedelta(days=LUNAR_DAY_DAYS)).strftime('%Y-%m-%d %H:%M:%S')),
            ('Time step (hours)', '2'), ('Minimum Sun visible (%)', '30'),
            ('Minimum Earth visible (%)', '50'), ('Grid spacing (degrees)', '10'),
            ('Latitude min / max', '-90, 90'), ('Longitude start / end', '-180, 180'),
            ('Samples per side (2-9)', '3'), ('Screen horizon (degrees)', '2'),
            ('Maximum slope (degrees; blank = off)', ''), ('Slope radius (km; 0-20)', '1'),
        ]
        self.fields = []
        for index, (label, value) in enumerate(defaults):
            row, col = divmod(index, 2)
            ttk.Label(settings, text=label).grid(row=row, column=col * 2, sticky='w', padx=(0, 5), pady=3)
            entry = ttk.Entry(settings, width=23)
            entry.insert(0, value)
            entry.grid(row=row, column=col * 2 + 1, sticky='ew', padx=(0, 12), pady=3)
            self.fields.append(entry)
        ttk.Label(settings, text='Cell screening').grid(row=6, column=0, sticky='w', pady=3)
        self.screen_mode = ttk.Combobox(settings, values=('Pessimistic center', 'All cells'), state='readonly', width=23)
        self.screen_mode.set('Pessimistic center')
        self.screen_mode.grid(row=6, column=1, sticky='ew', pady=3)
        ttk.Label(settings, text='Optimistic range estimate').grid(row=6, column=2, sticky='w', pady=3)
        self.range_mode = ttk.Combobox(settings, values=('On', 'Off'), state='readonly', width=23)
        self.range_mode.set('On')
        self.range_mode.grid(row=6, column=3, sticky='ew', pady=3)
        ttk.Label(frame, text='Shift-click or Ctrl-click to select results (maximum 12 points).\n'
                              'Slope is sampled at DEM scale; optimistic range screening can miss cells.',
                  wraplength=840).pack(anchor='w', pady=(4, 2))
        controls = ttk.Frame(frame)
        controls.pack(fill='x')
        self.start_button = ttk.Button(controls, text='Start search', command=self.start)
        self.start_button.pack(side='left')
        self.cancel_button = ttk.Button(controls, text='Cancel search', command=self.cancel.set, state='disabled')
        self.cancel_button.pack(side='left', padx=6)
        self.progress = ttk.Progressbar(frame, mode='determinate')
        self.progress.pack(fill='x', pady=(8, 3))
        self.status = ttk.Label(frame, text='Choose criteria and start the search.', wraplength=720)
        self.status.pack(anchor='w')
        self.range_label = ttk.Label(frame, text='', wraplength=840)
        self.range_label.pack(anchor='w')
        mode_frame = ttk.Frame(frame)
        mode_frame.pack(fill='x', pady=(4, 0))
        ttk.Label(mode_frame, text='Results:').pack(side='left')
        self.view_mode = ttk.Combobox(mode_frame, values=('Points', 'Clusters'), state='readonly', width=14)
        self.view_mode.set('Points')
        self.view_mode.pack(side='left', padx=5)
        self.view_mode.bind('<<ComboboxSelected>>', lambda event: self.populate_results('Showing'))
        columns = ('lat', 'lon', 'sun', 'earth', 'slope', 'max_slope', 'cell')
        table_frame = ttk.Frame(frame)
        table_frame.pack(fill='both', expand=True, pady=8)
        self.table = ttk.Treeview(table_frame, columns=columns, show='headings', height=14, selectmode='extended')
        scrollbar = ttk.Scrollbar(table_frame, orient='vertical', command=self.table.yview)
        self.table.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right', fill='y')
        for key, label in zip(columns, ('Latitude', 'Longitude', 'Sun visible %', 'Earth visible %', 'Slope deg', 'Max slope deg', 'Cell bounds (lat / lon)')):
            self.table.heading(key, text=label)
            self.table.column(key, width=260 if key == 'cell' else 125, anchor='center')
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
        self.analyze_button = ttk.Button(actions, text='Analyze selected', command=self.analyze_selected, state='disabled')
        self.analyze_button.pack(side='left', padx=5)
        self.marks_button = ttk.Button(more_actions, text='Show all on globe', command=self.show_all, state='disabled')
        self.marks_button.pack(side='left', padx=5)
        self.export_button = ttk.Button(more_actions, text='Save CSV', command=self.export, state='disabled')
        self.export_button.pack(side='left', padx=5)
        ttk.Button(more_actions, text='Close', command=self.close).pack(side='right')
        ttk.Button(more_actions, text='Clear markers', command=self.clear).pack(side='right', padx=5)
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

    def start(self):
        if self.worker is not None and self.worker.is_alive():
            return
        try:
            start, stop = sky.parse_utc(self.fields[0].get()), sky.parse_utc(self.fields[1].get())
            step_hours, sun, earth, spacing = (float(self.fields[i].get()) for i in range(2, 6))
            lat_min, lat_max = self._pair(self.fields[6].get())
            lon_min, lon_max = self._pair(self.fields[7].get())
            cells = grid_cells(lat_min, lat_max, lon_min, lon_max, spacing)
            side_value = float(self.fields[8].get())
            if not np.isfinite(side_value) or not side_value.is_integer() or not 2 <= side_value <= 9:
                raise ValueError('Samples per side must be an integer from 2 to 9.')
            side = int(side_value)
            horizon = float(self.fields[9].get())
            if not np.isfinite(horizon) or not -90 <= horizon <= 90:
                raise ValueError('Screen horizon must be between -90 and 90 degrees.')
            slope_text = self.fields[10].get().strip()
            max_slope = float(slope_text) if slope_text else None
            radius = float(self.fields[11].get())
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
            if not np.isfinite([sun, earth]).all() or not (0 <= sun <= 100 and 0 <= earth <= 100):
                raise ValueError('Visibility thresholds must be between 0 and 100 percent.')
        except (ValueError, OverflowError) as error:
            self.status.configure(text=str(error))
            return
        self.cancel.clear()
        self.results = []
        self.selected = None
        self.show_matches = False
        self.marker_results = None
        self.finished = False
        self.last_parameters = (start, stop, step_hours, sun, earth, spacing, lat_min, lat_max, lon_min, lon_max, side, horizon, mode, range_mode, max_slope, radius)
        self.table.delete(*self.table.get_children())
        for button in (self.view_button, self.select_all_button, self.add_button, self.analyze_button, self.marks_button, self.export_button):
            button.configure(state='disabled')
        self.start_button.configure(state='disabled')
        self.cancel_button.configure(state='normal')
        self.progress.configure(maximum=len(cells), value=0)
        self.range_label.configure(text='')
        self.status.configure(text=f'Preparing {len(cells):,} cells, up to {side * side + (side % 2 == 0)} sampled points per cell...')
        self.worker = threading.Thread(target=_search_worker,
            args=(self.messages, self.cancel, cells, epochs, sun, earth, self.fetch, side, horizon, mode, range_mode == 'On', max_slope, radius), daemon=True)
        self.worker.start()

    def poll(self):
        try:
            while True:
                kind, value = self.messages.get_nowait()
                if kind == 'range_progress':
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
                    if self.results:
                        for button in (self.view_button, self.select_all_button, self.add_button, self.analyze_button, self.marks_button, self.export_button):
                            button.configure(state='normal')
                else:
                    self.status.configure(text=value)
                    self.start_button.configure(state='normal')
                    self.cancel_button.configure(state='disabled')
        except queue.Empty:
            pass
        self.timer = self.window.after(100, self.poll)

    def _clusters(self):
        if self.last_parameters:
            start, end = self.last_parameters[8:10]
            wrap = end - start >= 360.0 - 1e-8
            bounds = (start, start + 360.0)
        else:
            wrap, bounds = False, None
        return cluster_results(self.results, wrap_longitude=wrap, longitude_bounds=bounds)

    def _marker_rows(self, members=None):
        clusters = self._clusters()
        cluster_for_row = {id(row): index for index, cluster in enumerate(clusters) for row in cluster}
        source = self.results if members is None else members
        return [dict(row, cluster_index=cluster_for_row[id(row)]) for row in source]

    def populate_results(self, label):
        self.table.delete(*self.table.get_children())
        clusters = self._clusters()
        cluster_for_row = {id(row): index for index, cluster in enumerate(clusters) for row in cluster}
        if self.view_mode.get() == 'Clusters':
            groups = clusters
        else:
            groups = [[row] for row in self.results]
        self._display_items = groups[:MAX_DISPLAY_RESULTS]
        for index, members in enumerate(self._display_items):
            row = members[0]
            if self.view_mode.get() == 'Clusters':
                lat_min = min(x.get('cell_lat_min', x['lat']) for x in members)
                lat_max = max(x.get('cell_lat_max', x['lat']) for x in members)
                lon_intervals = interval_union((x.get('cell_lon_min', x['lon']), x.get('cell_lon_max', x['lon'])) for x in members)
                lon_text = ','.join(f'{a:g}..{b:g}' for a, b in lon_intervals[:2])
                if len(lon_intervals) > 2:
                    lon_text += ',...'
                area = f"Cluster {index + 1}: {len(members)} cells; lat {lat_min:g}..{lat_max:g}, lon {lon_text}"
            else:
                area = (f"{row.get('cell_lat_min', row['lat']):g}..{row.get('cell_lat_max', row['lat']):g} / "
                        f"{row.get('cell_lon_min', row['lon']):g}..{row.get('cell_lon_max', row['lon']):g}")
            cluster_index = cluster_for_row[id(row)]
            tag = f'cluster_{cluster_index}'
            self.table.tag_configure(tag, foreground=cluster_text_color(cluster_index))
            self.table.insert('', 'end', iid=str(index), tags=(tag,), values=(f"{row['lat']:+.3f}", f"{row['lon']:+.3f}",
                              f"{row['sun_percent']:.1f}", f"{row['earth_percent']:.1f}",
                              f"{row['slope_deg']:.2f}" if row.get('slope_deg') is not None else '-',
                              f"{row['max_slope_deg']:.2f}" if row.get('max_slope_deg') is not None else '-', area))
        suffix = (f' Globe shows at most {MAX_DISPLAY_RESULTS} cells; CSV contains all points.'
                  if len(self.results) > MAX_DISPLAY_RESULTS else '')
        if len(groups) > MAX_DISPLAY_RESULTS:
            suffix += f' Table shows first {MAX_DISPLAY_RESULTS}.'
        noun = 'clusters' if self.view_mode.get() == 'Clusters' else 'points'
        self.status.configure(text=f"{label}: {len(groups):,} {noun}, {len(self.results):,} matching cells.{suffix}")
        if self.results:
            for button in (self.view_button, self.select_all_button, self.add_button, self.analyze_button, self.marks_button, self.export_button):
                button.configure(state='normal')

    def restore(self, previous):
        self.results = previous['results']
        self.last_parameters = previous['parameters']
        self.was_cancelled = previous['cancelled']
        self.finished = True
        start, stop, step, sun, earth, spacing, lat_min, lat_max, lon_min, lon_max = self.last_parameters[:10]
        side, horizon, mode = self.last_parameters[10:13] if len(self.last_parameters) >= 13 else (3, 2.0, "Pessimistic center")
        range_mode = self.last_parameters[13] if len(self.last_parameters) > 13 else 'On'
        self.screen_mode.set(mode)
        self.range_mode.set(range_mode)
        values = (start.strftime('%Y-%m-%d %H:%M:%S'), stop.strftime('%Y-%m-%d %H:%M:%S'),
                  step, sun, earth, spacing, f'{lat_min:g}, {lat_max:g}', f'{lon_min:g}, {lon_max:g}', side, horizon,
                  self.last_parameters[14] if len(self.last_parameters) > 14 and self.last_parameters[14] is not None else '',
                  self.last_parameters[15] if len(self.last_parameters) > 15 else 1)
        for field, value in zip(self.fields, values):
            field.delete(0, 'end')
            field.insert(0, str(value))
        self.populate_results('Previous partial search' if self.was_cancelled else 'Previous search')

    def view_selected(self, event=None):
        selected = self.table.selection()
        if selected:
            members = self._display_items[int(selected[0])]
            self.selected = members[0]
            self.marker_mode = self.view_mode.get()
            self.marker_results = self._marker_rows(members if self.marker_mode == 'Clusters' else None)
            self.show_matches = True
            self.close()

    def select_all(self, event=None):
        rows = self.table.get_children()
        self.table.selection_set(rows)
        self.status.configure(text=f'Selected {len(rows):,} displayed rows. Comparisons hold at most {MAX_POINTS} points.')
        return 'break'

    def add_selected(self, analyze=False):
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
        self.close()

    def analyze_selected(self):
        self.add_selected(analyze=True)

    def show_all(self):
        self.marker_mode = self.view_mode.get()
        self.marker_results = self._marker_rows()
        self.show_matches = True
        self.close()

    def clear(self):
        self.clear_markers = True
        self.close()

    def export(self):
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(parent=self.window, defaultextension='.csv',
                                            filetypes=[('CSV files', '*.csv')])
        if not path:
            return
        try:
            with open(path, 'w', newline='', encoding='utf-8') as output:
                writer = csv.writer(output)
                writer.writerow(('latitude_deg', 'longitude_deg', 'sun_visible_percent', 'earth_visible_percent',
                                 'slope_deg', 'max_sampled_slope_deg', 'slope_limit_deg', 'slope_radius_km',
                                 'start_utc', 'end_utc', 'step_hours', 'min_sun_percent', 'min_earth_percent',
                                 'grid_spacing_deg', 'latitude_min', 'latitude_max', 'longitude_min', 'longitude_max',
                                 'partial_search', 'cell_latitude_min', 'cell_latitude_max', 'cell_longitude_min', 'cell_longitude_max',
                                 'samples_per_side', 'screen_horizon_deg', 'screen_mode', 'optimistic_range_filter'))
                start, stop, step, sun, earth, spacing, lat_min, lat_max, lon_min, lon_max = self.last_parameters[:10]
                side, horizon, mode = self.last_parameters[10:13] if len(self.last_parameters) >= 13 else (3, 2.0, "Pessimistic center")
                for row in self.results:
                    writer.writerow((row['lat'], row['lon'], row['sun_percent'], row['earth_percent'],
                                     row.get('slope_deg'), row.get('max_slope_deg'),
                                     self.last_parameters[14] if len(self.last_parameters) > 14 else None,
                                     self.last_parameters[15] if len(self.last_parameters) > 15 else None,
                                     start.isoformat(), stop.isoformat(), step, sun, earth, spacing,
                                     lat_min, lat_max, lon_min, lon_max, self.was_cancelled,
                                     row.get('cell_lat_min'), row.get('cell_lat_max'), row.get('cell_lon_min'), row.get('cell_lon_max'),
                                     side, horizon, mode, self.last_parameters[13] if len(self.last_parameters) > 13 else 'Off'))
            self.status.configure(text=f'Saved {len(self.results):,} results to {path}')
        except OSError as error:
            self.status.configure(text=f'Could not save results: {error}')

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.cancel.set()
        if self.timer is not None:
            try:
                self.window.after_cancel(self.timer)
            except self.tk.TclError:
                pass
        self.window.destroy()
        if self.on_close is not None:
            self.on_close(self)
