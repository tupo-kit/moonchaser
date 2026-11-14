"""Local lunar observer calculations using Skyfield and cached JPL kernels.

Install: python -m pip install skyfield
DE440 kernels are downloaded into skyfield_data on first use.
The position and orientation files cover approximately 1550 to January 2650.
"""
from pathlib import Path
import numpy as np


def get_positions(lon, lat, altitude_km, epochs, data_dir=None, progress=None):
    from skyfield.api import Loader, PlanetaryConstants
    folder = Path(data_dir) if data_dir else Path(__file__).resolve().parent / 'skyfield_data'
    folder.mkdir(parents=True, exist_ok=True)
    load = Loader(str(folder))
    ts = load.timescale()
    times = ts.from_datetimes(epochs)
    if progress is not None:
        progress("Loading Skyfield data (download may be needed)")
    eph = load('https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440.bsp')
    constants = PlanetaryConstants()
    constants.read_text(load('https://naif.jpl.nasa.gov/pub/naif/generic_kernels/fk/satellites/moon_de440_250416.tf'))
    constants.read_text(load('https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00011.tpc'))
    constants.read_binary(load('https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/moon_pa_de440_200625.bpc'))
    # Skyfield currently keeps only the last PCK segment per frame. DE440's
    # lunar orientation kernel has two segments for frame 31008. Build an
    # observer for each segment and split intervals that cross the boundary.
    segments = getattr(constants, "_segment_list", None)
    if segments is None or len(segments) != 2 or 31008 not in constants._segment_map:
        raise RuntimeError("Unsupported Skyfield lunar kernel layout; install Skyfield 1.49 or newer.")
    dates = np.atleast_1d(times.tdb)
    if np.any((dates < 2287184.5) | (dates > 2688976.5)):
        raise ValueError("DE440 lunar orientation covers 1549-12-31 to 2650-01-25 TDB.")
    result = {name: {key: np.empty(len(dates)) for key in ("az", "el", "radius")}
              for name in ("Sun", "Earth")}
    for segment_index, mask in enumerate((dates < 2607184.5, dates >= 2607184.5)):
        indices = np.flatnonzero(mask)
        if not len(indices):
            continue
        constants._segment_map[31008] = segments[segment_index]
        frame = constants.build_frame_named('MOON_ME_DE440_ME421')
        observer = eph['moon'] + constants.build_latlon_degrees(
            frame, float(lat), float(lon), elevation_m=float(altitude_km) * 1000)
        selected_times = times[indices]
        for name, radius_km in (('Sun', 695700.0), ('Earth', 6378.1366)):
            if progress is not None:
                progress(f"Calculating {name} positions")
            apparent = observer.at(selected_times).observe(eph[name.lower()]).apparent()
            elevation, azimuth, distance = apparent.altaz()
            result[name]['az'][indices] = np.atleast_1d(azimuth.degrees) % 360
            result[name]['el'][indices] = np.atleast_1d(elevation.degrees)
            result[name]['radius'][indices] = np.degrees(
                np.arcsin(radius_km / np.atleast_1d(distance.km)))
    return result
