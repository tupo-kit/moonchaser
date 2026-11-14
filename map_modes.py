"""Generate and cache original-resolution elevation and slope map textures."""
import hashlib
import os
import re
import threading

import numpy as np
import terrain

VERSION = 'v1-geodesic-pixel-slope'
CHUNK_ROWS = 8
_slope_cache_signature = None
_slope_cache_array = None
_cache_prepared_for = None
_cache_dir_lock = threading.Lock()


def _cache_dir():
    """Put generated files together and move caches made by earlier versions."""
    global _cache_prepared_for
    directory = os.path.join(terrain.BASE_DIR, 'terrain_cache')
    if directory == _cache_prepared_for and os.path.isdir(directory):
        return directory
    with _cache_dir_lock:
        if directory == _cache_prepared_for and os.path.isdir(directory):
            return directory
        os.makedirs(directory, exist_ok=True)
        pattern = re.compile(r'moon_(?:slope|slope_rgb|elevation_rgb)_[0-9a-f]{20}\.npy\Z')
        with os.scandir(terrain.BASE_DIR) as entries:
            for entry in entries:
                if entry.is_file() and pattern.fullmatch(entry.name):
                    target = os.path.join(directory, entry.name)
                    if not os.path.exists(target):
                        os.replace(entry.path, target)
        _cache_prepared_for = directory
    return directory


def _cache_key():
    stat = os.stat(terrain.DEM_PATH)
    desc = f'{VERSION}:{stat.st_size}:{stat.st_mtime_ns}:{terrain.H}:{terrain.W}'
    return hashlib.sha256(desc.encode()).hexdigest()[:20]


def _read_cache(path, shape, dtype):
    try:
        result = np.load(path, mmap_mode='r', allow_pickle=False)
        if result.shape == shape and result.dtype == dtype:
            return result
    except (OSError, ValueError, EOFError):
        pass
    return None


def get_cached_slope_grid():
    """Open a complete matching slope grid if available, without generating it."""
    global _slope_cache_signature, _slope_cache_array
    if not terrain.H or not terrain.W:
        return None
    try:
        path = os.path.join(_cache_dir(), f'moon_slope_{_cache_key()}.npy')
        stat = os.stat(path)
        signature = (path, stat.st_size, stat.st_mtime_ns, terrain.H, terrain.W)
    except OSError:
        _slope_cache_signature = _slope_cache_array = None
        return None
    if signature != _slope_cache_signature:
        _slope_cache_array = _read_cache(path, (terrain.H, terrain.W), np.dtype('float32'))
        _slope_cache_signature = signature
    return _slope_cache_array


def _make_cache(path, shape, dtype, fill):
    temporary = f'{path}.{os.getpid()}.{threading.get_ident()}.part'
    array = None
    try:
        array = np.lib.format.open_memmap(temporary, mode='w+', dtype=dtype, shape=shape)
        fill(array)
        array.flush()
        del array
        array = None
        os.replace(temporary, path)
    finally:
        if array is not None:
            del array
        if os.path.exists(temporary):
            os.remove(temporary)
    result = _read_cache(path, shape, np.dtype(dtype))
    if result is None:
        raise RuntimeError(f'Could not read the generated map cache: {path}')
    return result


def slope_grid(progress=lambda value: None):
    """Calculate geodesic slopes at every DEM pixel center, once per dataset."""
    h, w = terrain.H, terrain.W
    path = os.path.join(_cache_dir(), f'moon_slope_{_cache_key()}.npy')
    existing = _read_cache(path, (h, w), np.dtype('float32'))
    if existing is not None:
        return existing
    def fill(out):
        lon = -180.0 + (np.arange(w, dtype=np.float64) + 0.5) * 360.0 / w
        for first in range(0, h, CHUNK_ROWS):
            last = min(h, first + CHUNK_ROWS)
            lat = 90.0 - (np.arange(first, last, dtype=np.float64) + 0.5) * 180.0 / h
            out[first:last] = terrain._compute_slope_deg_at(lat[:, None], lon[None, :])
            progress(f'Computing slopes: {100 * last / h:.0f}%')
    return _make_cache(path, (h, w), 'float32', fill)


def _lookup(stops):
    stops = np.asarray(stops, dtype=float)
    return np.round(np.column_stack([np.interp(np.arange(256), stops[:, 0] * 255, stops[:, i])
                                     for i in range(1, 4)])).astype(np.uint8)


ELEVATION_COLORS = _lookup([(0, 20, 30, 120), (.25, 10, 130, 220), (.5, 45, 190, 70),
                            (.75, 245, 190, 30), (1, 240, 235, 230)])
SLOPE_COLORS = _lookup([(0, 0, 150, 60), (.22, 235, 235, 0), (.5, 255, 115, 0), (1, 190, 0, 30)])


def generate(mode, progress=lambda value: None):
    """Return a memory-mapped RGB texture and human-readable scale."""
    if mode not in ('Elevation', 'Slope'):
        raise ValueError('Unknown terrain map mode.')
    h, w = terrain.H, terrain.W
    # The slope palette now uses 0-90 degrees, so never reuse the old 0-180 texture.
    texture_name = 'slope_rgb90' if mode == 'Slope' else 'elevation_rgb'
    path = os.path.join(_cache_dir(), f'moon_{texture_name}_{_cache_key()}.npy')
    low, high = ((float(np.min(terrain.dem_km)), float(np.max(terrain.dem_km)))
                 if mode == 'Elevation' else (0.0, 90.0))
    if high <= low:
        high = low + 1.0
    colors = ELEVATION_COLORS if mode == 'Elevation' else SLOPE_COLORS
    legend = {'title': 'Elevation (km)' if mode == 'Elevation' else 'Slope (deg)',
              'low': low, 'high': high, 'colors': colors}
    cached = _read_cache(path, (h, w, 3), np.dtype('uint8'))
    if cached is not None:
        return cached, legend
    source = terrain.dem_km if mode == 'Elevation' else slope_grid(progress)
    def fill(out):
        for first in range(0, h, CHUNK_ROWS):
            last = min(h, first + CHUNK_ROWS)
            block = np.asarray(source[first:last])
            indices = np.clip((block - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)
            out[first:last] = colors[indices]
            progress(f'Preparing {mode.lower()} colors: {100 * last / h:.0f}%')
    return _make_cache(path, (h, w, 3), 'uint8', fill), legend
