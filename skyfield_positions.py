"""Local lunar observer calculations using Skyfield and cached JPL kernels.

Install: python -m pip install skyfield
DE440 kernels are downloaded into skyfield_data on first use.
The position and orientation files cover approximately 1550 to January 2650.
"""
from pathlib import Path
from functools import lru_cache
import numpy as np


@lru_cache(maxsize=2)
def _load_resources(folder_path):
    """Keep ephemerides and both frame segments open for later requests."""
    from skyfield.api import Loader, PlanetaryConstants
    folder = Path(folder_path)
    folder.mkdir(parents=True, exist_ok=True)
    load = Loader(str(folder))
    timescale = load.timescale()
    eph = load('https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440.bsp')
    constants = PlanetaryConstants()
    constants.read_text(load('https://naif.jpl.nasa.gov/pub/naif/generic_kernels/fk/satellites/moon_de440_250416.tf'))
    constants.read_text(load('https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00011.tpc'))
    constants.read_binary(load('https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/moon_pa_de440_200625.bpc'))
    # Skyfield maps only one segment per frame, so build a frame for each.
    segments = getattr(constants, "_segment_list", None)
    if segments is None or len(segments) != 2 or 31008 not in constants._segment_map:
        raise RuntimeError("Unsupported Skyfield lunar kernel layout; install Skyfield 1.49 or newer.")
    frames = []
    for segment in segments:
        constants._segment_map[31008] = segment
        frames.append(constants.build_frame_named('MOON_ME_DE440_ME421'))
    return timescale, eph, constants, tuple(frames)


def get_positions(lon, lat, altitude_km, epochs, data_dir=None, progress=None):
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
