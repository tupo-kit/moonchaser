"""Point comparison helpers: palette, point-set files, per-point statistics, table text and sorting. No VTK, no UI.

Statistics are computed from the classified sky data of sky.classify_sky() ("az", "el", "terrain" arrays).
"Above horizon" means the body centre is above the local terrain horizon (eclipses are not counted).
All table text is plain ASCII so it renders with the built-in monospace font.
"""
import json
import math

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
MIN_RELIABLE_DAYS = 2.0


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


def save_points(path, points):
    """Write only the coordinates: {"points": [[lat, lon], ...]}, one point per line."""
    rows = ",\n".join(f"  [{float(lat)!r}, {float(lon)!r}]" for lat, lon in points)
    with open(path, "w", encoding="utf-8") as file:
        file.write('{\n "points": [\n' + rows + '\n ]\n}\n')


def load_points(path):
    """Read a point set. Accepts the JSON written by save_points, or text with one 'lat, lon' per line."""
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
            raw.append(parts)
    else:
        items = data.get("points") if isinstance(data, dict) else data
        if not isinstance(items, list):
            raise ValueError('Expected {"points": [[lat, lon], ...]}.')
        for item in items:
            if isinstance(item, dict):
                item = (item.get("lat"), item.get("lon"))
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError("Each point must be [latitude, longitude].")
            raw.append(item)
    points = []
    for item in raw:
        point = normalize_point(*item)
        if not any(same_point(point, other) for other in points):
            points.append(point)
    if len(points) > MAX_POINTS:
        raise ValueError(f"A set holds at most {MAX_POINTS} points; the file has {len(points)}.")
    return points


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
    return {
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
    """Two tables, one row per point.

    sorts: [sort_of_table_1, sort_of_table_2], each (column, descending) or None.
    Returns [(title, (header1, header2), [row, ...], order, spans), ...]; order[i] is the point index
    shown in row i (the rows are sorted, the point names P1, P2, ... stay with their points).
    """
    sorts = list(sorts or [None, None]) + [None, None]
    na = "n/a"
    rows_a, rows_b, keys_a, keys_b = [], [], [], []
    for index, ((lat, lon), s) in enumerate(zip(points, stats)):
        name = f"P{index + 1}"
        place = [f"{lat:+.3f}", f"{lon:+.3f}"]
        slope_value = terrain.compute_slope_deg(lat, lon)
        slope = f"{slope_value:.2f}"
        if s is None:
            rows_a.append([name, *place, slope, na, na, na, na])
            rows_b.append([name, *[na] * 8])
            keys_a.append([index, lat, lon, _finite(slope_value), None, None, None, None])
            keys_b.append([index] + [None] * 8)
            continue
        sun, earth = s["Sun"], s["Earth"]
        period = format_duration(sun["mean_period_s"]) + ("*" if sun["period_partial"] else "")
        rows_a.append([name, *place, slope, f"{100.0 * sun['fraction']:.2f}",
                       format_duration(sun["per_lunar_day_s"]), period, f"{100.0 * earth['fraction']:.2f}"])
        keys_a.append([index, lat, lon, _finite(slope_value), sun["fraction"], sun["per_lunar_day_s"],
                       sun["mean_period_s"], earth["fraction"]])
        rows_b.append([name, _number(sun["az"], 0, ".2f"), _number(sun["az"], 1, ".2f"),
                       _number(sun["el"], 0, "+.2f"), _number(sun["el"], 1, ".2f"),
                       _number(earth["az"], 0, ".2f"), _number(earth["az"], 1, ".2f"),
                       _number(earth["el"], 0, "+.2f"), _number(earth["el"], 1, ".2f")])
        keys_b.append([index, _value(sun["az"], 0), _value(sun["az"], 1), _value(sun["el"], 0), _value(sun["el"], 1),
                       _value(earth["az"], 0), _value(earth["az"], 1), _value(earth["el"], 0), _value(earth["el"], 1)])
    columns_a = [("Point", ""), ("Lat", "deg"), ("Lon", "deg"), ("Slope", "deg"), ("Sun above", "horizon %"),
                 ("Sun above/", "lunar day"), ("Avg sunlit", "period"), ("Earth above", "horizon %")]
    columns_b = [("Point", ""), ("Sun AZ", "avg deg"), ("Sun AZ", "SD deg"), ("Sun EL", "avg deg"),
                 ("Sun EL", "SD deg"), ("Earth AZ", "avg deg"), ("Earth AZ", "SD deg"),
                 ("Earth EL", "avg deg"), ("Earth EL", "SD deg")]
    result = []
    for title, columns, rows, keys, sort in (("Time above the horizon", columns_a, rows_a, keys_a, sorts[0]),
                                             ("Direction statistics", columns_b, rows_b, keys_b, sorts[1])):
        if sort is not None and not 0 <= sort[0] < len(columns):
            sort = None
        order = sort_order(keys, sort)
        header, lines, spans = _render(_mark(columns, sort), [rows[i] for i in order])
        result.append((title, header, lines, order, spans))
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
        "Above horizon = body centre above the local terrain horizon (eclipses not counted).",
        f"Above/lunar day = share of time above horizon x {SYNODIC_DAYS:.2f} d (one lunar day).",
        "Avg sunlit period = mean length of one continuous Sun-above-horizon stretch; * = cut by the interval edge.",
        "AZ: north = 0, clockwise; avg and SD are circular. EL: elevation above the local horizontal plane.",
        "AZ / EL statistics use " + ("only the samples where the body is above the horizon." if only_above
                                     else "all samples."),
        "Click a column heading to sort (ascending, descending, original order). Rows without a value go last.",
    ]
    return "\n".join(lines)
