"""Local lunar observer calculations using Skyfield and cached JPL kernels.

Install: python -m pip install skyfield
The four kernels are downloaded into skyfield_data on first use. Later calls
work offline while the cached kernels remain available. DE421 and the lunar
orientation kernel cover 1900 through 2050.
"""
from pathlib import Path
import numpy as np


def get_positions(lon, lat, altitude_km, epochs, data_dir=None):
    from skyfield.api import Loader, PlanetaryConstants
    folder = Path(data_dir) if data_dir else Path(__file__).resolve().parent / 'skyfield_data'
    folder.mkdir(parents=True, exist_ok=True)
    load = Loader(str(folder))
    ts = load.timescale()
    times = ts.from_datetimes(epochs)
    eph = load('de421.bsp')
    constants = PlanetaryConstants()
    constants.read_text(load('moon_080317.tf'))
    constants.read_text(load('pck00008.tpc'))
    constants.read_binary(load('moon_pa_de421_1900-2050.bpc'))
    frame = constants.build_frame_named('MOON_ME_DE421')
    observer = eph['moon'] + constants.build_latlon_degrees(
        frame, float(lat), float(lon), elevation_m=float(altitude_km) * 1000)
    result = {}
    for name, radius_km in (('Sun', 695700.0), ('Earth', 6378.1366)):
        apparent = observer.at(times).observe(eph[name.lower()]).apparent()
        elevation, azimuth, distance = apparent.altaz()
        result[name] = {'az': np.atleast_1d(azimuth.degrees) % 360,
                        'el': np.atleast_1d(elevation.degrees),
                        'radius': np.degrees(np.arcsin(radius_km / np.atleast_1d(distance.km)))}
    return result
