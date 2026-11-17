"""Point comparison helpers: palette, point-set files, per-point statistics, table text and sorting. No VTK, no UI.

Statistics are computed from the classified sky data of sky.classify_sky() ("az", "el", "terrain", "visible").
Two different definitions are available everywhere (comparison tables and search columns):
  "Visible"       sky.classify_sky()["visible"]: centre or part of the disk is clear of terrain, and for the Sun
                  not eclipsed by Earth. This is what the search filters on.
  "Above horizon" the body centre is above the local terrain horizon (eclipses are not counted).
All table text is plain ASCII so it renders with the built-in monospace font.
"""
import json
import math
from functools import lru_cache

import numpy as np
import terrain

MAX_POINTS = 12
POINT_COLORS = (
    (1.00, 0.35, 0.35), (0.35, 0.80, 1.00), (0.45, 1.00, 0.45), (1.00, 0.80, 0.20),
    (0.85, 0.50, 1.00), (1.00, 0.60, 0.20), (0.30, 1.00, 0.85), (1.00, 0.50, 0.80),
    (0.75, 0.85, 0.30), (0.60, 0.60, 1.00), (0.90, 0.90, 0.90), (0.75, 0.55, 0.35),
)
SYNODIC_DAYS = 29.530588861  # one lunar day (new moon to new moon), in Earth days
SYNODIC_SECONDS = SYNODIC_DAYS * 86400.0
LUNAR_DAY_DAYS = SYNODIC_DAYS  # name used by the search window
MIN_RELIABLE_DAYS = 2.0
TABLE_COUNT = 4                # comparison tables built by build_tables()
TABLE_SLOPE_RADIUS_KM = 1.0    # "max slope" columns: largest sampled slope within this radius


# --------------------------------------------------------------------------- points and files

def normalize_point(lat, lon):
    """Validate a point; longitude is wrapped to [-180, 180)."""
    try:
        lat, lon = float(lat), float(lon)
    except (TypeError, ValueError):
        raise ValueError("Coordinates must be numbers.") from None
    if not (math.isfinite(lat) and math.isfinite(lon)):
        raise ValueError("Coordinates must be finite numbers.")
    if not -90.0 <= lat <= 90.0:
        raise ValueError("Latitude must be between -90 and 90.")
    if not -180.0 <= lon < 180.0:
        lon = (lon + 180.0) % 360.0 - 180.0
    return lat, lon


def same_point(a, b, tolerance=1e-6):
    dlon = abs((a[1] - b[1] + 180.0) % 360.0 - 180.0)
    return abs(a[0] - b[0]) <= tolerance and dlon <= tolerance


def save_points(path, points, names=None):
    """Write {"points": [...]}, one point per line: [lat, lon], or {"lat", "lon", "name"} for a named point."""
    names = names or {}
    rows = []
    for point in points:
        lat, lon = point
        name = names.get(point)
        if name:
            rows.append(f"  {json.dumps({'lat': float(lat), 'lon': float(lon), 'name': name}, ensure_ascii=False)}")
        else:
            rows.append(f"  [{float(lat)!r}, {float(lon)!r}]")
    with open(path, "w", encoding="utf-8") as file:
        file.write('{\n "points": [\n' + ",\n".join(rows) + '\n ]\n}\n')


def load_points(path):
    """Read a point set. Accepts the JSON written by save_points, or text with one 'lat, lon' per line."""
    return load_named_points(path)[0]


def load_named_points(path):
    """Like load_points, but returns (points, names) where names maps a point to its optional name."""
    try:
        with open(path, encoding="utf-8-sig") as file:
            text = file.read()
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"Could not read the file: {error}") from None
    raw = []
    try:
        data = json.loads(text)
    except ValueError:
        for number, line in enumerate(text.splitlines(), start=1):
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.replace(",", " ").replace(";", " ").split()
            if len(parts) != 2:
                raise ValueError(f"Line {number}: expected 'latitude, longitude'.")
            raw.append((parts, None))
    else:
        items = data.get("points") if isinstance(data, dict) else data
        if not isinstance(items, list):
            raise ValueError('Expected {"points": [[lat, lon], ...]}.')
        for item in items:
            name = None
            if isinstance(item, dict):
                name = item.get("name")
                item = (item.get("lat"), item.get("lon"))
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError("Each point must be [latitude, longitude].")
            raw.append((item, name.strip() if isinstance(name, str) else None))
    points = []
    names = {}
    for item, name in raw:
        point = normalize_point(*item)
        if not any(same_point(point, other) for other in points):
            points.append(point)
            if name:
                names[point] = name
    if len(points) > MAX_POINTS:
        raise ValueError(f"A set holds at most {MAX_POINTS} points; the file has {len(points)}.")
    return points, names


# --------------------------------------------------------------------------- visibility maths

def _trapezoid_percent(durations, values):
    """Percent of the time a 0/1 series is on; every step counts the mean of its two end samples."""
    return float(100 * np.dot(durations, (values[:-1] + values[1:]) * 0.5) / durations.sum())


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
    return _trapezoid_percent(durations, values)


def lunar_day_series(seconds, visible):
    """[(start_s, end_s, percent)] for every complete lunar day counted from the first sample.

    A day without two samples is left out. An interval shorter than one lunar day gives a single entry
    that covers the whole interval.
    """
    values = np.asarray(visible, dtype=float)
    t = np.asarray(seconds, dtype=float)
    if values.shape != t.shape or len(t) < 2:
        raise ValueError('Visibility result does not match the time samples.')

    def percent(lo, hi):
        index = np.nonzero((t >= lo) & (t <= hi))[0]
        if len(index) < 2:
            return None
        ts, vs = t[index], values[index]
        span = ts[-1] - ts[0]
        if span <= 0:
            return None
        return float(100.0 * np.sum((vs[:-1] + vs[1:]) * 0.5 * np.diff(ts)) / span)

    length = LUNAR_DAY_DAYS * 86400.0
    series = []
    for k in range(int(t[-1] // length)):
        value = percent(k * length, (k + 1) * length)
        if value is not None:
            series.append((k * length, (k + 1) * length, value))
    if not series:
        series = [(float(t[0]), float(t[-1]), percent(0.0, float(t[-1])) or 0.0)]
    return series


def lunar_day_stats(epochs, visible):
    """(mean %, standard deviation %, number of lunar days) of the Sun visibility per lunar day.

    The interval is cut into consecutive lunar days (LUNAR_DAY_DAYS) from its start and the visibility
    percentage of each complete lunar day is computed; the mean and the (population) standard deviation
    are taken over these days. Shorter intervals count as a single day (SD 0).
    """
    seconds = np.array([(epoch - epochs[0]).total_seconds() for epoch in epochs])
    days = [value for _, _, value in lunar_day_series(seconds, visible)]
    return float(np.mean(days)), float(np.std(days)), len(days)


@lru_cache(maxsize=4096)
def max_slope_cached(lat, lon, radius_km=TABLE_SLOPE_RADIUS_KM):
    """terrain.max_slope_in_radius, remembered (the DEM never changes while the program runs)."""
    return float(terrain.max_slope_in_radius(lat, lon, radius_km))


@lru_cache(maxsize=4096)
def location_values(lat, lon):
    """(elevation m, slope deg, max slope deg within TABLE_SLOPE_RADIUS_KM or None) of a point."""
    elevation = float(terrain.sample_dem_bilinear(lat, lon))
    slope = float(terrain.compute_slope_deg(lat, lon))
    try:
        steepest = max_slope_cached(lat, lon, TABLE_SLOPE_RADIUS_KM)
    except ValueError:
        steepest = None
    return elevation, slope, steepest


# --------------------------------------------------------------------------- statistics

def above_periods(seconds, clearance):
    """Stretches with clearance > 0 as (start_s, end_s, complete); edges are linearly interpolated.

    complete is False when the stretch is cut off by the start or end of the interval.
    """
    t = np.asarray(seconds, dtype=float)
    c = np.asarray(clearance, dtype=float)
    up = c > 0.0
    n = len(t)
    periods, i = [], 0
    while i < n:
        if not up[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and up[j + 1]:
            j += 1
        if i > 0:
            start, left = t[i - 1] + (0.0 - c[i - 1]) / (c[i] - c[i - 1]) * (t[i] - t[i - 1]), True
        else:
            start, left = t[0], False
        if j < n - 1:
            end, right = t[j] + c[j] / (c[j] - c[j + 1]) * (t[j + 1] - t[j]), True
        else:
            end, right = t[-1], False
        periods.append((float(start), float(end), left and right))
        i = j + 1
    return periods


def circular_stats(degrees):
    """(mean, standard deviation) of angles in degrees; mean in [0, 360). None if empty."""
    a = np.radians(np.asarray(degrees, dtype=float))
    if not len(a):
        return None
    c, s = float(np.mean(np.cos(a))), float(np.mean(np.sin(a)))
    r = min(math.hypot(c, s), 1.0)
    mean = math.degrees(math.atan2(s, c)) % 360.0
    if mean >= 360.0:
        mean = 0.0
    std = math.degrees(math.sqrt(max(-2.0 * math.log(r), 0.0))) if r > 1e-12 else float("nan")
    return mean, abs(std)


def linear_stats(values):
    v = np.asarray(values, dtype=float)
    return (float(np.mean(v)), float(np.std(v))) if len(v) else None


def body_stats(seconds, body, only_above=True):
    """Statistics of one body (a classify_sky() entry) over the whole interval."""
    seconds = np.asarray(seconds, dtype=float)
    span = float(seconds[-1] - seconds[0])
    clearance = np.asarray(body["el"], dtype=float) - np.asarray(body["terrain"], dtype=float)
    periods = above_periods(seconds, clearance)
    fraction = min(sum(end - start for start, end, _ in periods) / span, 1.0)
    complete = [end - start for start, end, whole in periods if whole]
    partial = not complete and bool(periods)
    lengths = complete or [end - start for start, end, _ in periods]
    mask = clearance > 0.0 if only_above else np.ones(len(clearance), dtype=bool)
    visible = body.get("visible")
    durations = np.diff(seconds)
    return {
        "visible_pct": (_trapezoid_percent(durations, np.asarray(visible, dtype=float))
                        if visible is not None and np.all(durations > 0) else None),
        "fraction": fraction,
        "per_lunar_day_s": fraction * SYNODIC_SECONDS,
        "mean_period_s": float(np.mean(lengths)) if lengths else None,
        "period_partial": partial,
        "az": circular_stats(np.asarray(body["az"], dtype=float)[mask]),
        "el": linear_stats(np.asarray(body["el"], dtype=float)[mask]),
    }


def point_stats(epochs, data, source, only_above=True):
    """{"Sun": ..., "Earth": ..., "span_s": ...} for one point, or None without usable data."""
    if data is None or len(epochs) < 2:
        return None
    names = {key: f"{key} {source}" for key in ("Sun", "Earth")}
    if any(name not in data for name in names.values()):
        return None
    seconds = np.array([(epoch - epochs[0]).total_seconds() for epoch in epochs])
    if seconds[-1] <= 0:
        return None
    result = {key: body_stats(seconds, data[name], only_above) for key, name in names.items()}
    result["span_s"] = float(seconds[-1])
    try:
        result["lunar"] = lunar_day_stats(epochs, data[names["Sun"]]["visible"])
    except (KeyError, ValueError):
        result["lunar"] = None
    return result


# --------------------------------------------------------------------------- table text

def format_duration(seconds):
    if seconds is None:
        return "-"
    if seconds < 3600.0:
        return f"{seconds / 60.0:.1f} min"
    if seconds < 86400.0:
        return f"{seconds / 3600.0:.2f} h"
    return f"{seconds / 86400.0:.2f} d"


def _number(pair, index, form):
    return "-" if pair is None or not math.isfinite(pair[index]) else format(pair[index], form)


def _value(pair, index):
    """Sort key of one entry of a (mean, SD) pair; None sorts last."""
    return None if pair is None or not math.isfinite(pair[index]) else float(pair[index])


def _finite(value):
    return float(value) if value is not None and math.isfinite(value) else None


def sort_order(keys, sort):
    """Row order for a sort setting (column, descending) or None. Rows without a value always come last."""
    count = len(keys)
    if sort is None:
        return list(range(count))
    column, descending = sort
    present = [i for i in range(count) if keys[i][column] is not None]
    missing = [i for i in range(count) if keys[i][column] is None]
    present.sort(key=lambda i: keys[i][column], reverse=descending)  # stable: ties stay in point order
    return present + missing


def _render(columns, rows):
    """columns: [(line1, line2)], rows: [[cell, ...]].

    Returns (two header lines, row strings, spans) where spans[i] = (first, last+1) character of column i.
    """
    widths = [max([len(h1), len(h2)] + [len(row[i]) for row in rows]) for i, (h1, h2) in enumerate(columns)]

    def join(cells):
        # The first column (point name) is left aligned, the numbers are right aligned.
        return "  ".join(cell.ljust(w) if i == 0 else cell.rjust(w) for i, (cell, w) in enumerate(zip(cells, widths)))

    spans, position = [], 0
    for width in widths:
        spans.append((position, position + width))
        position += width + 2
    header = (join([h[0] for h in columns]), join([h[1] for h in columns]))
    return header, [join(row) for row in rows], spans


def _mark(columns, sort):
    """Put an arrow in the second header line of the sorted column."""
    marked = list(columns)
    if sort is not None:
        line1, line2 = marked[sort[0]]
        marked[sort[0]] = (line1, (line2 + " " if line2 else "") + ("v" if sort[1] else "^"))
    return marked


def next_sort(current, column):
    """Click on a header: ascending, then descending, then back to the point order."""
    if current is None or current[0] != column:
        return (column, False)
    return (column, True) if not current[1] else None


def build_tables(points, stats, sorts=None):
    """TABLE_COUNT tables, one row per point (location, visibility, time above the horizon, directions).

    sorts: one (column, descending) or None per table.
    Returns [(title, (header1, header2), [row, ...], order, spans), ...]; order[i] is the point index
    shown in row i (the rows are sorted, the point names P1, P2, ... stay with their points).
    """
    sorts = list(sorts or []) + [None] * TABLE_COUNT
    na = "n/a"

    def num(value, form):
        return "-" if value is None or not math.isfinite(value) else format(value, form)

    def span(mean, partial):
        return format_duration(mean) + ("*" if partial and mean is not None else "")

    # Each table: columns [(line1, line2)], then per point (text cells, sort keys) after the point name.
    columns = [
        [("Point", ""), ("Lat", "deg"), ("Lon", "deg"), ("Elevation", "m"), ("Slope", "deg"),
         ("Max slope", f"deg {TABLE_SLOPE_RADIUS_KM:g} km")],
        [("Point", ""), ("Sun visible", "%"), ("Earth visible", "%"), ("Sun/lunar day", "avg %"),
         ("Sun/lunar day", "SD %"), ("Lunar", "days")],
        [("Point", ""), ("Sun above", "horizon %"), ("Sun above/", "lunar day"), ("Avg sunlit", "period"),
         ("Earth above", "horizon %"), ("Earth above/", "lunar day"), ("Avg Earth", "period")],
        [("Point", ""), ("Sun AZ", "avg deg"), ("Sun AZ", "SD deg"), ("Sun EL", "avg deg"),
         ("Sun EL", "SD deg"), ("Earth AZ", "avg deg"), ("Earth AZ", "SD deg"),
         ("Earth EL", "avg deg"), ("Earth EL", "SD deg")],
    ]
    titles = ("Location", "Sun / Earth visible (disk, partly visible counts; Sun eclipses)",
              "Time with the body centre above the horizon", "Direction statistics")
    rows = [[] for _ in range(TABLE_COUNT)]
    keys = [[] for _ in range(TABLE_COUNT)]
    for index, ((lat, lon), s) in enumerate(zip(points, stats)):
        name = f"P{index + 1}"
        elevation, slope, steepest = location_values(lat, lon)
        rows[0].append([name, f"{lat:+.3f}", f"{lon:+.3f}", f"{elevation:+,.0f}", f"{slope:.2f}", num(steepest, ".2f")])
        keys[0].append([index, lat, lon, _finite(elevation), _finite(slope), _finite(steepest)])
        if s is None:
            for t, width in ((1, 5), (2, 6), (3, 8)):
                rows[t].append([name] + [na] * width)
                keys[t].append([index] + [None] * width)
            continue
        sun, earth, lunar = s["Sun"], s["Earth"], s.get("lunar")
        rows[1].append([name, num(sun.get("visible_pct"), ".2f"), num(earth.get("visible_pct"), ".2f"),
                        num(lunar[0] if lunar else None, ".2f"), num(lunar[1] if lunar else None, ".2f"),
                        str(lunar[2]) if lunar else "-"])
        keys[1].append([index, _finite(sun.get("visible_pct")), _finite(earth.get("visible_pct")),
                        _finite(lunar[0]) if lunar else None, _finite(lunar[1]) if lunar else None,
                        lunar[2] if lunar else None])
        rows[2].append([name, f"{100.0 * sun['fraction']:.2f}", format_duration(sun["per_lunar_day_s"]),
                        span(sun["mean_period_s"], sun["period_partial"]), f"{100.0 * earth['fraction']:.2f}",
                        format_duration(earth["per_lunar_day_s"]),
                        span(earth["mean_period_s"], earth["period_partial"])])
        keys[2].append([index, sun["fraction"], sun["per_lunar_day_s"], sun["mean_period_s"],
                        earth["fraction"], earth["per_lunar_day_s"], earth["mean_period_s"]])
        rows[3].append([name, _number(sun["az"], 0, ".2f"), _number(sun["az"], 1, ".2f"),
                        _number(sun["el"], 0, "+.2f"), _number(sun["el"], 1, ".2f"),
                        _number(earth["az"], 0, ".2f"), _number(earth["az"], 1, ".2f"),
                        _number(earth["el"], 0, "+.2f"), _number(earth["el"], 1, ".2f")])
        keys[3].append([index, _value(sun["az"], 0), _value(sun["az"], 1), _value(sun["el"], 0), _value(sun["el"], 1),
                        _value(earth["az"], 0), _value(earth["az"], 1), _value(earth["el"], 0), _value(earth["el"], 1)])
    result = []
    for t in range(TABLE_COUNT):
        sort = sorts[t]
        if sort is not None and not 0 <= sort[0] < len(columns[t]):
            sort = None
        order = sort_order(keys[t], sort)
        header, lines, spans = _render(_mark(columns[t], sort), [rows[t][i] for i in order])
        result.append((titles[t], header, lines, order, spans))
    return result


def table_notes(span_s, only_above):
    lines = []
    if span_s is not None:
        days = span_s / 86400.0
        if days < MIN_RELIABLE_DAYS:
            lines.append(f"! Interval is {days:.2f} days; use 2 or more days for representative results.")
        elif days < SYNODIC_DAYS:
            lines.append(f"! Interval is shorter than one lunar day ({SYNODIC_DAYS:.2f} d): 'above/lunar day' is extrapolated.")
    lines += [
        "Visible = centre or part of the disk clear of terrain; the Sun also must not be eclipsed by Earth.",
        "Above horizon = body centre above the local terrain horizon (eclipses not counted).",
        f"Sun/lunar day = Sun visible % of each complete lunar day ({SYNODIC_DAYS:.2f} d) from the interval start;"
        " mean and SD over those days (a shorter interval counts as one day, SD 0).",
        f"Max slope = largest sampled slope within {TABLE_SLOPE_RADIUS_KM:g} km of the point.",
        f"Above/lunar day = share of time above horizon x {SYNODIC_DAYS:.2f} d (one lunar day).",
        "Avg sunlit period = mean length of one continuous Sun-above-horizon stretch; * = cut by the interval edge.",
        "AZ: north = 0, clockwise; avg and SD are circular. EL: elevation above the local horizontal plane.",
        "AZ / EL statistics use " + ("only the samples where the body is above the horizon." if only_above
                                     else "all samples."),
        "Click a column heading to sort (ascending, descending, original order). Rows without a value go last.",
    ]
    return "\n".join(lines)
