"""Sun / Earth positions seen from a lunar observer, plus time helpers. No VTK, no UI.

Sections: time helpers and saved interval, visibility classification (terrain + Earth eclipse),
Skyfield (offline, DE440 kernels, ~1550-2650), JPL Horizons (online, wide date range).
Both position sources return {"Sun"|"Earth": {"az": deg, "el": deg, "radius": deg}} arrays
(azimuth north=0 clockwise; altitude is passed in km; longitude/latitude in degrees).
Skyfield kernels are cached in a skyfield_data/ folder beside this file.
"""
import hashlib
import json
import os
import sys
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from threading import Lock

import numpy as np

MAX_TIME_SAMPLES = 10_000
INTERVAL_PATH = os.path.join(os.path.expanduser("~"), ".moon_horizon_interval.json")
TIME_UNITS = {"seconds": 1/60, "minutes": 1, "hours": 60, "days": 1440}
SKY_METHODS = {"Skyfield": ("Skyfield",), "JPL Horizons": ("JPL",)}
SUN_RADIUS_DEG = float(np.degrees(np.arcsin(695700.0 / 149597870.7)))  # mean, 1 AU
EARTH_RADIUS_DEG = float(np.degrees(np.arcsin(6378.1366 / 384400.0)))  # mean Earth-Moon distance


def load_interval():
    try:
        with open(INTERVAL_PATH, encoding="utf-8") as file:
            data = json.load(file)
        start, stop = parse_utc(data["start"]), parse_utc(data["stop"])
        step, unit = float(data["step"]), data["unit"]
        method = data.get("method", "Skyfield")
        if method not in SKY_METHODS:
            method = "Skyfield"
        time_samples(start, stop, step * TIME_UNITS[unit])
        return start, stop, step, unit, method
    except (OSError, ValueError, TypeError, KeyError, OverflowError):
        return None


def save_interval(start, stop, step, unit, method="Skyfield"):
    data = {"start": parse_utc(start).isoformat(), "stop": parse_utc(stop).isoformat(), "step": float(step), "unit": unit,
            "method": method}
    try:
        with open(INTERVAL_PATH, "w", encoding="utf-8") as file:
            json.dump(data, file)
    except OSError as error:
        print(f"Could not save the time interval: {error}", file=sys.stderr)


def utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0)


def parse_utc(value):
    try:
        result = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except ValueError:
        raise ValueError("Use a UTC date/time such as 2026-11-14 12:30:00.") from None
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    return result.astimezone(timezone.utc)


def time_samples(start, stop, step_minutes):
    start, stop = parse_utc(start), parse_utc(stop)
    try:
        step = float(step_minutes) * 60.0
    except (ValueError, TypeError):
        raise ValueError("Sample step must be a positive number of minutes.") from None
    if not np.isfinite(step) or step <= 0:
        raise ValueError("Sample step must be a positive number of minutes.")
    duration = (stop - start).total_seconds()
    if duration < 0:
        raise ValueError("The interval end must not be before its start.")
    count = int(np.floor(duration / step)) + 1
    if count > MAX_TIME_SAMPLES:
        raise ValueError(f"Use a larger step: at most {MAX_TIME_SAMPLES} samples are allowed.")
    epochs = [start + timedelta(seconds=index * step) for index in range(count)]
    if epochs[-1] < stop:
        epochs.append(stop)
    if len(epochs) > MAX_TIME_SAMPLES:
        raise ValueError(f"Use a larger step: at most {MAX_TIME_SAMPLES} samples are allowed.")
    return epochs


def sky_unit(azimuth, elevation):
    azimuth, elevation = np.radians(azimuth), np.radians(elevation)
    return np.stack((np.cos(elevation) * np.sin(azimuth),
                     np.cos(elevation) * np.cos(azimuth), np.sin(elevation)), axis=-1)


def classify_sky(positions, horizon):
    """Test each centre, then four limb points only when the centre is blocked.

    A clear centre counts as visible. The four extra points are an approximate
    test for partial visibility and may miss thin crescents or terrain gaps.
    Earth's illuminated phase is not modelled.
    """
    result = {}
    bearings = np.arange(360)
    earth_center = sky_unit(positions["Earth"]["az"], positions["Earth"]["el"])
    earth_radius = np.radians(np.asarray(positions["Earth"]["radius"]))
    earth_limit = np.cos(earth_radius)
    for name in ("Sun", "Earth"):
        body = {key: np.asarray(value).copy() for key, value in positions[name].items()}
        az, el = np.radians(body["az"]), np.radians(body["el"])
        centre = sky_unit(body["az"], body["el"])
        terrain = np.interp(body["az"], bearings, horizon, period=360)
        terrain_clear = body["el"] > terrain
        earth_clear = np.ones(len(el), dtype=bool)
        if name == "Sun":
            earth_clear = np.sum(centre * earth_center, axis=-1) < earth_limit
        center_clear = terrain_clear & earth_clear
        visible = center_clear.copy()
        state = np.full(len(el), "Hidden by terrain", dtype=object)
        if name == "Sun":
            state[terrain_clear & ~earth_clear] = "Eclipsed by Earth"
        state[center_clear] = "Visible"

        # Chunk the exceptional samples to keep memory bounded for long intervals.
        blocked = np.flatnonzero(~center_clear)
        for offset in range(0, len(blocked), 8192):
            idx = blocked[offset:offset + 8192]
            a, e = az[idx], el[idx]
            horizontal = np.stack((np.cos(a), -np.sin(a), np.zeros_like(a)), axis=-1)
            vertical = np.stack((-np.sin(e) * np.sin(a), -np.sin(e) * np.cos(a), np.cos(e)), axis=-1)
            rho = np.radians(body["radius"][idx])[:, None, None]
            tangents = np.stack((vertical, -vertical, horizontal, -horizontal), axis=1)
            rays = np.cos(rho) * centre[idx, None, :] + np.sin(rho) * tangents
            ray_az = np.degrees(np.arctan2(rays[..., 0], rays[..., 1])) % 360.0
            ray_el = np.degrees(np.arctan2(rays[..., 2], np.hypot(rays[..., 0], rays[..., 1])))
            ray_terrain_clear = ray_el > np.interp(ray_az, bearings, horizon, period=360)
            ray_clear = ray_terrain_clear.copy()
            if name == "Sun":
                ray_clear &= np.sum(rays * earth_center[idx, None, :], axis=-1) < earth_limit[idx, None]
                terrain_any = terrain_clear[idx] | ray_terrain_clear.any(axis=1)
                state[idx[terrain_any]] = "Eclipsed by Earth"
            partial = ray_clear.any(axis=1)
            visible[idx[partial]] = True
            state[idx[partial]] = "Partly visible"
        body.update(visible=visible, state=state, terrain=terrain)
        result[name] = body
    return result


# ==============================================================================
# Skyfield: local lunar observer calculations using cached JPL kernels.
# Install: python -m pip install skyfield
# DE440 kernels are downloaded into skyfield_data on first use.
# The position and orientation files cover approximately 1550 to January 2650.
# ==============================================================================

# NASA NAIF publishes this checksum for the complete DE440 SPK file.
_DE440_MD5 = "c9d581bfd84209dbeee8b1583939b148"


def _kernel_valid(path):
    """Detect Git LFS pointers, truncated kernels and a damaged DE440 SPK."""
    try:
        with path.open("rb") as file:
            head = file.read(128)
            if head.startswith(b"version https://git-lfs.github.com/spec/v1"):
                return False
            if path.name == "de440.bsp":
                if not head.startswith(b"DAF/SPK"):
                    return False
                digest = hashlib.md5(usedforsecurity=False)
                digest.update(head)
                for block in iter(lambda: file.read(1024 * 1024), b""):
                    digest.update(block)
                return digest.hexdigest() == _DE440_MD5
            if path.name.endswith(".bpc"):
                return head.startswith(b"DAF/PCK") and path.stat().st_size >= 10 * 1024 * 1024
            return head.lstrip().startswith((b"KPL/FK", b"KPL/PCK")) and path.stat().st_size > 1024
    except FileNotFoundError:
        return False


def _read_kernel(load, folder, url, consume=None):
    """Replace a broken local kernel and retry a failed cached parse once."""
    path = folder / url.rsplit("/", 1)[-1]
    cached = path.is_file()
    if cached and not _kernel_valid(path):
        print(f"Invalid Skyfield data: {path.name}; downloading a replacement.")
        path.unlink()
        cached = False
    try:
        value = load(url)
        result = consume(value) if consume else value
    except (ValueError, EOFError, OSError, UnicodeError) as error:
        if not cached:
            raise RuntimeError(f"Could not load {path.name}: {error}") from error
        print(f"Could not read cached {path.name}; downloading a replacement.")
        path.unlink(missing_ok=True)
        try:
            value = load(url)
            result = consume(value) if consume else value
        except (ValueError, EOFError, OSError, UnicodeError) as retry_error:
            raise RuntimeError(f"Could not download or load {path.name}: {retry_error}") from retry_error
    if not _kernel_valid(path):
        path.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded {path.name} is invalid; check your connection and try again.")
    return result


@lru_cache(maxsize=2)
def _load_resources(folder_path):
    """Keep ephemerides and both frame segments open for later requests."""
    from skyfield.api import Loader, PlanetaryConstants
    folder = Path(folder_path)
    folder.mkdir(parents=True, exist_ok=True)
    load = Loader(str(folder))
    timescale = load.timescale()
    eph = _read_kernel(load, folder, 'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440.bsp')
    constants = PlanetaryConstants()
    _read_kernel(load, folder, 'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/fk/satellites/moon_de440_250416.tf', constants.read_text)
    _read_kernel(load, folder, 'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00011.tpc', constants.read_text)
    _read_kernel(load, folder, 'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/moon_pa_de440_200625.bpc', constants.read_binary)
    # Skyfield maps only one segment per frame, so build a frame for each.
    segments = getattr(constants, "_segment_list", None)
    if segments is None or len(segments) != 2 or 31008 not in constants._segment_map:
        raise RuntimeError("Unsupported Skyfield lunar kernel layout; install Skyfield 1.49 or newer.")
    frames = []
    for segment in segments:
        constants._segment_map[31008] = segment
        frames.append(constants.build_frame_named('MOON_ME_DE440_ME421'))
    return timescale, eph, constants, tuple(frames)


def preload(data_dir=None):
    """Download, validate, and keep the Skyfield kernels ready for use."""
    folder = Path(data_dir) if data_dir else Path(__file__).resolve().parent / "skyfield_data"
    _load_resources(str(folder.resolve()))


def _calculate_positions(lon, lat, altitude_km, epochs, data_dir=None, progress=None):
    folder = Path(data_dir) if data_dir else Path(__file__).resolve().parent / 'skyfield_data'
    if progress is not None:
        progress("Loading Skyfield data (first request may download files)")
    ts, eph, constants, frames = _load_resources(str(folder.resolve()))
    times = ts.from_datetimes(epochs)
    dates = np.atleast_1d(times.tdb)
    if np.any((dates < 2287184.5) | (dates > 2688976.5)):
        raise ValueError("DE440 lunar orientation covers 1549-12-31 to 2650-01-25 TDB.")
    result = {name: {key: np.empty(len(dates)) for key in ("az", "el", "radius")}
              for name in ("Sun", "Earth")}
    for segment_index, mask in enumerate((dates < 2607184.5, dates >= 2607184.5)):
        indices = np.flatnonzero(mask)
        if not len(indices):
            continue
        frame = frames[segment_index]
        observer = eph['moon'] + constants.build_latlon_degrees(
            frame, float(lat), float(lon), elevation_m=float(altitude_km) * 1000)
        selected_times = times[indices]
        if progress is not None:
            progress("Calculating lunar observer position")
        observer_position = observer.at(selected_times)
        for name, radius_km in (('Sun', 695700.0), ('Earth', 6378.1366)):
            if progress is not None:
                progress(f"Calculating {name} positions")
            apparent = observer_position.observe(eph[name.lower()]).apparent()
            elevation, azimuth, distance = apparent.altaz()
            result[name]['az'][indices] = np.atleast_1d(azimuth.degrees) % 360
            result[name]['el'][indices] = np.atleast_1d(elevation.degrees)
            result[name]['radius'][indices] = np.degrees(
                np.arcsin(radius_km / np.atleast_1d(distance.km)))
    return result


# A bounded in-memory cache of previously calculated observations. The terrain
# visibility is computed separately by the viewer, so horizon changes are safe.
_POSITION_CACHE = OrderedDict()
_POSITION_CACHE_LOCK = Lock()
_POSITION_CACHE_LIMIT = 100_000


def get_positions(lon, lat, altitude_km, epochs, data_dir=None, progress=None):
    """Reuse exact previously calculated site/time samples; calculate misses."""
    epochs = list(epochs)
    folder = Path(data_dir) if data_dir else Path(__file__).resolve().parent / "skyfield_data"
    site = (str(folder.resolve()), float(lon), float(lat), float(altitude_km))
    result = {name: {key: np.empty(len(epochs), dtype=float) for key in ("az", "el", "radius")}
              for name in ("Sun", "Earth")}
    names = ("Sun", "Earth")
    keys = ("az", "el", "radius")
    missing = {}
    with _POSITION_CACHE_LOCK:
        for index, epoch in enumerate(epochs):
            key = (site, epoch)
            value = _POSITION_CACHE.get(key)
            if value is None:
                missing.setdefault(key, []).append(index)
            else:
                _POSITION_CACHE.move_to_end(key)
                for j, (name, field) in enumerate((n, k) for n in names for k in keys):
                    result[name][field][index] = value[j]
    if missing:
        if progress is not None:
            progress(f"Calculating {len(missing)} new samples; reusing {len(epochs) - sum(map(len, missing.values()))}")
        missing_keys = list(missing)
        calculated = _calculate_positions(lon, lat, altitude_km,
                                          [key[1] for key in missing_keys], data_dir, progress)
        with _POSITION_CACHE_LOCK:
            for i, key in enumerate(missing_keys):
                value = tuple(float(calculated[name][field][i]) for name in names for field in keys)
                for index in missing[key]:
                    for j, (name, field) in enumerate((n, k) for n in names for k in keys):
                        result[name][field][index] = value[j]
                _POSITION_CACHE[key] = value
                _POSITION_CACHE.move_to_end(key)
                if len(_POSITION_CACHE) > _POSITION_CACHE_LIMIT:
                    _POSITION_CACHE.popitem(last=False)
    elif progress is not None:
        progress(f"Reusing {len(epochs)} cached samples")
    return result


# ==============================================================================
# JPL Horizons (internet): Moon-surface observer, azimuth/elevation only.
# Install: python -m pip install astroquery
# ==============================================================================

def _horizons_class():
    """Import astroquery lazily so Skyfield-only use never needs it."""
    try:
        from astroquery.jplhorizons import Horizons
    except ImportError as error:
        raise RuntimeError(f"Install astroquery: {error}") from error
    return Horizons


def get_angle(lon, lat, alt, t_start, t_stop, t_step, body):
    """Alt in km!!!
    lon and lat in deg
    start, stop format: YYYY-MM-DD [HH:MM:SS]
    step in n[y|d|m|s|h]
    body: Earth(399), Sun(10)
    """
    Horizons = _horizons_class()
    if body == "Earth":
        body = 399 # 399 is code for Earth in horizon, didnt work with name (ValueError: Ambiguous target name; provide unique id)
    obj = Horizons(
        id=body, # sun id
        id_type=None,
        location={"lon": lon, "lat": lat, "elevation": alt, "body": 301},  # lon/lat in deg, altitude in km, 301 - moon id
        epochs={"start": t_start, "stop": t_stop, "step": f"{t_step}"},
    )
    eph = obj.ephemerides(quantities="4", refraction=False) #quantitis = "4" outputs azimuth and elevation in deg.
    return eph["datetime_str", "AZ", "EL"].to_pandas()


def fetch_jpl(lon, lat, altitude, epochs, progress=None):
    """JPL Horizons through get_angle (azimuth and elevation only)."""
    _horizons_class()  # fail early, with a clear message, if astroquery is missing
    text = lambda moment: f"{moment:%Y-%m-%d %H:%M:%S}"
    # Split into evenly spaced runs: get_angle takes start, stop and step.
    runs, run = [], [epochs[0]]
    for epoch in epochs[1:]:
        if len(run) >= 2 and epoch - run[-1] != run[1] - run[0]:
            runs.append(run)
            run = [epoch]
        else:
            run.append(epoch)
    runs.append(run)
    result = {}
    for name, body, radius in (("Sun", 10, SUN_RADIUS_DEG), ("Earth", "Earth", EARTH_RADIUS_DEG)):
        if progress is not None:
            progress(f"Requesting {name} from JPL Horizons")
        az, el = [], []
        for run in runs:
            if len(run) == 1:
                table = get_angle(lon, lat, altitude, text(run[0]), text(run[0] + timedelta(minutes=1)), "1m", body)
            else:
                seconds = int(round((run[1] - run[0]).total_seconds()))
                step = (f"{seconds // 86400}d" if seconds % 86400 == 0 else f"{seconds // 3600}h" if seconds % 3600 == 0
                        else f"{seconds // 60}m" if seconds % 60 == 0 else f"{seconds}s")
                table = get_angle(lon, lat, altitude, text(run[0]), text(run[-1]), step, body)
            values = (np.asarray(table["AZ"], dtype=float)[:len(run)], np.asarray(table["EL"], dtype=float)[:len(run)])
            if any(len(v) != len(run) for v in values):
                raise RuntimeError(f"JPL returned an incomplete {name} time series.")
            az.extend(values[0].tolist())
            el.extend(values[1].tolist())
        az, el = np.asarray(az), np.asarray(el)
        if not (np.isfinite(az).all() and np.isfinite(el).all()):
            raise RuntimeError(f"JPL returned unavailable {name} coordinates.")
        result[name] = {"az": az % 360.0, "el": el, "radius": np.full(len(az), radius)}
    return result
