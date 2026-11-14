"""Local lunar observer calculations using Skyfield and cached JPL kernels.

Install: python -m pip install skyfield
DE440 kernels are downloaded into skyfield_data on first use.
The position and orientation files cover approximately 1550 to January 2650.
"""
from pathlib import Path
from functools import lru_cache
from collections import OrderedDict
from threading import Lock
import hashlib
import numpy as np


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
