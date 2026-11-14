"""Moon terrain data and geometry: no VTK, no UI, no sky positions.

Prepares and loads the NASA colour/elevation maps, samples the DEM, and computes the
terrain horizon, the panorama and the latitude/longitude graticule.

IMPORTANT: load_data() reassigns the module globals dem_km, H, W, R_MAX_M and COLOR_PATH.
Other files must `import terrain` and read `terrain.COLOR_PATH`; writing
`from terrain import COLOR_PATH` would keep a stale copy. Functions and constants that are
never reassigned (sample_dem_bilinear, MOON_RADIUS_M, ...) are safe to import by name.
"""
from contextlib import contextmanager
import os
import urllib.request

import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COLOR_PATH = os.path.join(BASE_DIR, "moon_color_13k.jpg")
DEM_PATH = os.path.join(BASE_DIR, "moon_dem_13k.tif")
MAP_SIZE = (12960, 6480)
COLOR_SOURCE = os.path.join(BASE_DIR, "lroc_color_poles_16k.tif")
DEM_SOURCE = os.path.join(BASE_DIR, "ldem_64_uint.tif")
COLOR_SOURCE_URL = "https://svs.gsfc.nasa.gov/vis/a000000/a004700/a004720/lroc_color_poles_16k.tif"
DEM_SOURCE_URL = "https://svs.gsfc.nasa.gov/vis/a000000/a004700/a004720/ldem_64_uint.tif"
MOON_RADIUS_M = 1_737_400.0
RELIEF_EXAGGERATION = 3.0

# Initialized by load_data(), so importing this module never downloads files.
dem_km = None
H = W = 0
R_MAX_M = MOON_RADIUS_M


def download(url, path):
    """Download to a temp file first so a failed download never leaves a corrupt file."""
    tmp = path + ".part"
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "MoonHorizon/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response, open(tmp, "wb") as output:
            import shutil
            shutil.copyfileobj(response, output)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


@contextmanager
def allow_large_nasa_image():
    # These two known NASA source images exceed Pillow's default pixel limit.
    # Limit the exception to startup while reading those sources.
    old_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None
    try:
        yield
    finally:
        Image.MAX_IMAGE_PIXELS = old_limit


def prepare_nasa_map(source, url, output, expected_size, kind):
    """Download an identified NASA source and atomically prepare a 13K map."""
    if os.path.isfile(output):
        return
    if not os.path.isfile(source):
        print(f"Downloading NASA {kind} map; this may take a while...", flush=True)
        download(url, source)
    temporary = output + ".part"
    try:
        print(f"Preparing 13K {kind} map...", flush=True)
        with allow_large_nasa_image():
            with Image.open(source) as image:
                if image.size != expected_size:
                    raise ValueError(f"Unexpected {kind} map size {image.size}; expected {expected_size}: {source}")
                if kind == "elevation":
                    if image.mode not in ("I;16", "I;16L", "I;16B", "I"):
                        raise ValueError(f"Expected NASA's 16-bit elevation map, got mode {image.mode}.")
                    resized = image.resize(MAP_SIZE, Image.Resampling.BILINEAR)
                    # NASA uint16 is in half-metres, offset +20,000 from the
                    # signed lunar elevations. Convert to kilometres.
                    elevations = np.asarray(resized, dtype=np.float32)
                    elevations -= 20000.0
                    elevations /= 2000.0
                    Image.fromarray(elevations).save(temporary, format="TIFF")
                else:
                    image = image.convert("RGB")
                    image = image.resize(MAP_SIZE, Image.Resampling.LANCZOS)
                    image.save(temporary, format="JPEG", quality=95, subsampling=0)
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def load_dem(path):
    """Load the prepared 13K float elevation map in kilometres."""
    try:
        with Image.open(path) as image:
            if image.size != MAP_SIZE or image.mode != "F":
                raise ValueError(f"Expected a {MAP_SIZE[0]}x{MAP_SIZE[1]} floating-point elevation TIFF: {path}")
            return np.asarray(image, dtype=np.float32).copy()
    except (OSError, Image.DecompressionBombError) as error:
        raise RuntimeError(f"Cannot read elevation map {path}: {error}") from error


def get_dem():
    prepare_nasa_map(DEM_SOURCE, DEM_SOURCE_URL, DEM_PATH, (23040, 11520), "elevation")
    return load_dem(DEM_PATH)


def choose_texture(texture_path=None):
    if texture_path:
        path = os.path.abspath(os.path.expanduser(texture_path))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Texture file does not exist: {path}")
        return path
    prepare_nasa_map(COLOR_SOURCE, COLOR_SOURCE_URL, COLOR_PATH, (16384, 8192), "color")
    return COLOR_PATH


def load_data(texture_path=None):
    global dem_km, H, W, R_MAX_M, COLOR_PATH
    COLOR_PATH = choose_texture(texture_path)
    with Image.open(COLOR_PATH) as image:
        print(f"Texture: {os.path.basename(COLOR_PATH)} ({image.width} x {image.height})", flush=True)
        image.verify()
    print("Loading 13K Moon elevation map...", flush=True)
    dem_km = get_dem()
    if not np.all(np.isfinite(dem_km)):
        raise RuntimeError("The elevation map contains invalid values.")
    H, W = dem_km.shape
    R_MAX_M = MOON_RADIUS_M + float(np.max(dem_km)) * 1000.0


def sample_dem_bilinear(lat_deg, lon_deg):
    lat_arr = np.asarray(lat_deg, dtype=np.float64)
    lon_arr = np.asarray(lon_deg, dtype=np.float64)
    r_f = np.clip((90.0 - lat_arr) / 180.0 * H - 0.5, 0.0, H - 1.00001)
    c_f = np.mod((lon_arr + 180.0) / 360.0 * W - 0.5, W)
    r0, c0 = np.floor(r_f).astype(int), np.floor(c_f).astype(int)
    r1, c1 = np.minimum(r0 + 1, H - 1), np.mod(c0 + 1, W)
    dr, dc = r_f - r0, c_f - c0
    return (
        dem_km[r0, c0] * (1 - dr) * (1 - dc)
        + dem_km[r0, c1] * (1 - dr) * dc
        + dem_km[r1, c0] * dr * (1 - dc)
        + dem_km[r1, c1] * dr * dc
    ) * 1000.0


def lat_lon_to_xyz(lat_deg, lon_deg, offset=0.0):
    r = 1.0 + (sample_dem_bilinear(lat_deg, lon_deg) / MOON_RADIUS_M) * RELIEF_EXAGGERATION + offset
    lat_r, lon_r = np.radians(lat_deg), np.radians(lon_deg)
    return r * np.cos(lat_r) * np.cos(lon_r), r * np.cos(lat_r) * np.sin(lon_r), r * np.sin(lat_r)


GRATICULE_LADDER = (30.0, 10.0, 5.0, 2.0, 1.0, 0.5)  # grid spacings in degrees, coarse to fine
GRATICULE_SAMPLE_DEG = 0.5      # render-mesh spacing: line points sit exactly on mesh vertices
GRID_MIN_SPACING_PX = 130.0     # grid lines are never drawn closer together than this on screen


def choose_graticule_step(pixels_per_degree, current=None, min_px=GRID_MIN_SPACING_PX):
    """Finest ladder spacing whose lines are still >= min_px apart on screen.

    Zooming in switches to a finer grid at once; zooming out only coarsens once the
    current lines are clearly too dense (70 % of min_px), so the grid does not flicker.
    """
    wanted = GRATICULE_LADDER[0]
    for step in GRATICULE_LADDER:
        if step * pixels_per_degree >= min_px:
            wanted = step
        else:
            break
    if current in GRATICULE_LADDER and wanted > current and current * pixels_per_degree >= 0.7 * min_px:
        return current
    return wanted


def pole_heights():
    """Height of each pole point of the render mesh (it collapses a pole to one point)."""
    lons = np.linspace(-180.0, 180.0, 721)
    return {sign: float(np.mean(lat_lon_to_xyz(np.full_like(lons, 90.0 * sign), lons)[2])) for sign in (-1, 1)}


def graticule_lines(step_deg, center_lat=0.0, center_lon=0.0, radius_deg=180.0):
    """Latitude/longitude lines that follow the terrain, limited to a spherical cap.

    Returns (minor, major): lists of (N, 3) polylines; major = equator and 0 deg meridian.
    Points come from lat_lon_to_xyz, the function that builds the rendered mesh, and are
    spaced like its vertices, so the lines lie on the surface.
    """
    step = float(step_deg)
    ratio = step / GRATICULE_SAMPLE_DEG
    if not 0 < step <= 90 or abs(ratio - round(ratio)) > 1e-9:
        raise ValueError("Grid spacing must be a positive multiple of 0.5 degrees, at most 90.")
    sample = GRATICULE_SAMPLE_DEG
    radius = min(float(radius_deg), 180.0)
    lat_lo, lat_hi = max(-90.0, center_lat - radius), min(90.0, center_lat + radius)
    full_lon = radius >= 90.0 or abs(center_lat) + radius >= 90.0 - 1e-9
    if full_lon:
        lon_lo, lon_hi = -180.0, 180.0
    else:  # longitude half-width of the cap (it does not contain a pole here)
        half = np.degrees(np.arcsin(min(1.0, np.sin(np.radians(radius)) / np.cos(np.radians(center_lat)))))
        lon_lo, lon_hi = center_lon - half, center_lon + half
    lat_samples = np.clip(np.arange(np.floor(lat_lo / sample), np.ceil(lat_hi / sample) + 1) * sample, -90.0, 90.0)
    lon_samples = np.arange(np.floor(lon_lo / sample), np.ceil(lon_hi / sample) + 1) * sample

    meridians = np.arange(int(np.ceil(lon_lo / step - 1e-9)), int(np.floor(lon_hi / step + 1e-9)) + 1) * step
    if full_lon:
        meridians = meridians[meridians < 180.0 - 1e-9]  # +180 duplicates -180
    parallels = np.arange(int(np.ceil(lat_lo / step - 1e-9)), int(np.floor(lat_hi / step + 1e-9)) + 1) * step
    parallels = parallels[np.abs(parallels) < 90.0 - 1e-9]  # the poles are points

    mx, my, mz = lat_lon_to_xyz(lat_samples[None, :], meridians[:, None])
    meridian_points = np.stack((mx, my, mz), axis=-1)
    for sign, height in pole_heights().items():
        meridian_points[:, lat_samples * sign >= 90.0 - 1e-9, :] = (0.0, 0.0, height)
    px, py, pz = lat_lon_to_xyz(parallels[:, None], lon_samples[None, :])
    parallel_points = np.stack((px, py, pz), axis=-1)

    minor, major = [], []
    for lon, line in zip(meridians, meridian_points):
        turn = lon % 360.0
        (major if min(turn, 360.0 - turn) < 1e-9 else minor).append(line)
    for lat, line in zip(parallels, parallel_points):
        (major if abs(lat) < 1e-9 else minor).append(line)
    return minor, major


def compute_horizon_360(lat_deg, lon_deg, eye_height_m=2.0, max_dist_km=250.0, step_km=0.5):
    """Terrain horizon for one observer, at every integer azimuth (north=0).

    Search every bearing independently. Cartesian great-circle propagation avoids
    the longitude singularity of the usual destination formula at either pole.
    Heights/angles use physical terrain; 3x relief is a display setting only.
    """
    if not (-90 <= lat_deg <= 90) or not np.isfinite(lon_deg):
        raise ValueError("Observer coordinates must be finite, with latitude in [-90, 90].")
    if eye_height_m <= 0 or step_km <= 0 or max_dist_km <= 0:
        raise ValueError("Eye height, search distance and sample step must be positive.")
    r_obs = MOON_RADIUS_M + float(sample_dem_bilinear(lat_deg, lon_deg)) + eye_height_m
    azimuths = np.radians(np.arange(360))
    lat0, lon0 = np.radians([lat_deg, lon_deg])
    up = np.array([np.cos(lat0) * np.cos(lon0),
                   np.cos(lat0) * np.sin(lon0), np.sin(lat0)])
    north = np.array([-np.sin(lat0) * np.cos(lon0),
                      -np.sin(lat0) * np.sin(lon0), np.cos(lat0)])
    east = np.array([-np.sin(lon0), np.cos(lon0), 0.0])
    directions = np.cos(azimuths)[:, None] * north + np.sin(azimuths)[:, None] * east
    max_angles = np.full(360, -np.inf)
    h_dists = np.zeros(360)
    h_lats = np.full(360, float(lat_deg))
    h_lons = np.full(360, float(lon_deg))
    # Fine samples near the observer, followed by the requested regular spacing.
    distances = np.unique(np.concatenate((
        np.arange(0.05, min(step_km, max_dist_km), 0.05),
        np.arange(step_km, max_dist_km, step_km), [max_dist_km])))
    for distance in distances:
        theta = distance * 1000.0 / MOON_RADIUS_M
        targets = np.cos(theta) * up + np.sin(theta) * directions
        r_lats = np.degrees(np.arctan2(targets[:, 2], np.hypot(targets[:, 0], targets[:, 1])))
        r_lons = np.degrees(np.arctan2(targets[:, 1], targets[:, 0]))
        r_tgt = MOON_RADIUS_M + sample_dem_bilinear(r_lats, r_lons)
        angles = np.arctan2(r_tgt * np.cos(theta) - r_obs, r_tgt * np.sin(theta))
        improved = angles > max_angles
        max_angles[improved] = angles[improved]
        h_dists[improved] = distance
        h_lats[improved] = r_lats[improved]
        h_lons[improved] = r_lons[improved]
    return np.degrees(max_angles), h_dists, h_lats, h_lons


def terrain_panorama_samples(lat_deg, lon_deg, texture_rgb, max_dist_km=250.0, step_km=0.5, bearings=360):
    """Project terrain samples into azimuth/elevation and sample map colors.

    Distances match compute_horizon_360. The DEM and map are coarse; this is
    an approximate, daytime surface view, not a photograph or lighting model.
    """
    az = np.linspace(0.0, 2 * np.pi, bearings, endpoint=False)
    lat0, lon0 = np.radians([lat_deg, lon_deg])
    up = np.array([np.cos(lat0) * np.cos(lon0), np.cos(lat0) * np.sin(lon0), np.sin(lat0)])
    north = np.array([-np.sin(lat0) * np.cos(lon0), -np.sin(lat0) * np.sin(lon0), np.cos(lat0)])
    east = np.array([-np.sin(lon0), np.cos(lon0), 0.0])
    direction = np.cos(az)[:, None] * north + np.sin(az)[:, None] * east
    distances = np.unique(np.concatenate((np.arange(0.05, min(step_km, max_dist_km), 0.05),
                                         np.arange(step_km, max_dist_km, step_km), [max_dist_km])))
    theta = distances * 1000.0 / MOON_RADIUS_M
    points = np.cos(theta)[None, :, None] * up + np.sin(theta)[None, :, None] * direction[:, None, :]
    lat = np.degrees(np.arctan2(points[..., 2], np.hypot(points[..., 0], points[..., 1])))
    lon = np.degrees(np.arctan2(points[..., 1], points[..., 0]))
    ground_r = MOON_RADIUS_M + sample_dem_bilinear(lat, lon)
    observer_r = MOON_RADIUS_M + float(sample_dem_bilinear(lat_deg, lon_deg)) + 2.0
    angles = np.degrees(np.arctan2(ground_r * np.cos(theta)[None, :] - observer_r,
                                   ground_r * np.sin(theta)[None, :]))
    h, w = texture_rgb.shape[:2]
    row = np.clip(((90.0 - lat) / 180.0 * h).astype(int), 0, h - 1)
    col = np.mod(((lon + 180.0) / 360.0 * w).astype(int), w)
    return np.maximum.accumulate(angles, axis=1), texture_rgb[row, col]


def terrain_panorama_rgba(samples, low, high, rows=256):
    """Fill each pixel with its nearest terrain sample; sky remains transparent."""
    angles, colors = samples
    levels = np.linspace(low, high, rows)
    bearing_count = angles.shape[0]
    pixels = np.zeros((rows, bearing_count + 1, 4), dtype=np.uint8)
    for bearing in range(bearing_count):
        indices = np.searchsorted(angles[bearing], levels, side="left")
        visible = indices < angles.shape[1]
        pixels[visible, bearing, :3] = colors[bearing, indices[visible]]
        pixels[visible, bearing, 3] = 255
    pixels[:, bearing_count] = pixels[:, 0]
    return pixels


def compute_slope_deg(lat, lon):
    step = 180.0 / H  # one DEM pixel in degrees
    lat_hi, lat_lo = min(lat + step, 90.0), max(lat - step, -90.0)
    dy_m = np.radians(lat_hi - lat_lo) * MOON_RADIUS_M
    dz_dy = (sample_dem_bilinear(lat_hi, lon) - sample_dem_bilinear(lat_lo, lon)) / dy_m
    dx_m = max(np.radians(2 * 360.0 / W) / 2 * MOON_RADIUS_M * np.cos(np.radians(lat)) * 2, 1.0)
    dz_dx = (sample_dem_bilinear(lat, lon + 360.0 / W) - sample_dem_bilinear(lat, lon - 360.0 / W)) / dx_m
    return float(np.degrees(np.arctan(np.hypot(dz_dx, dz_dy))))


def parse_coordinates(text):
    text = text.replace("\N{MINUS SIGN}", "-").replace("\N{EN DASH}", "-")
    parts = text.replace(",", " ").replace(";", " ").split()
    if len(parts) != 2:
        raise ValueError("Enter latitude, longitude; for example: -43.31, -11.36")
    try:
        lat, lon = map(float, parts)
    except ValueError:
        raise ValueError("Coordinates must be two decimal numbers.") from None
    if not np.isfinite([lat, lon]).all():
        raise ValueError("Coordinates must be finite numbers.")
    if not -90 <= lat <= 90:
        raise ValueError("Latitude must be between -90 and 90.")
    # Accept east longitudes such as 312 as well as signed longitudes.
    return lat, (lon + 180.0) % 360.0 - 180.0
