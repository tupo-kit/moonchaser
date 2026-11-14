"""Moon Horizon: a native desktop viewer, with no browser or web server.

Install: python -m pip install numpy Pillow vtk skyfield astroquery
Coordinate editor: requires Tkinter (Linux package: python3-tk).
Run:     python moon.py
Texture: python moon.py --texture /path/to/color.jpg

NASA color and elevation maps download on first run and are resized to 13K.
Left-drag to orbit; middle-drag or Shift-left-drag to pan; wheel to zoom.
Ctrl-left-drag (or Ctrl-middle-drag) orbits the terrain point under the cursor.
Click the Moon to select an observer.
Zoom stops above the terrain. The inset shows all 360 horizon bearings.
Prepared maps are saved beside this script for later runs.
Press G to enter latitude, longitude; R to reset; F for fullscreen; Escape to close.
Sun/Earth positions use Skyfield (offline, limited date range) or JPL Horizons via get_angle.py (internet, wide range);
you pick the model in the time interval window, and only that model is queried. Time defaults to UTC now.
Click Edit time interval (or T) to choose a range; this opens a fullscreen horizon + Sun/Earth graph.
That window scrolls (mouse wheel, Up/Down, PageUp/PageDown, Home/End) to more graphs: Sun elevation vs
time and Earth elevation vs time.
Use Previous/Next to step samples and Close to return to the horizon-only view.
"""

import argparse
from contextlib import contextmanager
import json
import os
import sys
import urllib.request
import queue
import threading
import time
from datetime import datetime, timedelta, timezone

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
MAX_TIME_SAMPLES = 10_000
INTERVAL_PATH = os.path.join(os.path.expanduser("~"), ".moon_horizon_interval.json")
TIME_UNITS = {"seconds": 1/60, "minutes": 1, "hours": 60, "days": 1440}
SKY_METHODS = {"Skyfield": ("Skyfield",), "JPL Horizons": ("JPL",)}
SUN_RADIUS_DEG = float(np.degrees(np.arcsin(695700.0 / 149597870.7)))  # mean, 1 AU
EARTH_RADIUS_DEG = float(np.degrees(np.arcsin(6378.1366 / 384400.0)))  # mean Earth-Moon distance
SKY_LEGEND = {"JPL": "JPL Sun yellow / Earth blue", "Skyfield": "Skyfield Sun orange / Earth green"}
SKY_COLORS = {"Sun JPL": (1.0, 0.8, 0.12), "Earth JPL": (0.35, 0.65, 1.0), "Sun Skyfield": (1.0, 0.35, 0.1), "Earth Skyfield": (0.3, 1.0, 0.5)}
PANEL_HEIGHT = 0.5  # each extra graph is half a window tall; the page scrolls under the window
EXTRA_PANELS = ("Sun", "Earth")
PAGE_HEIGHT = 1.0 + PANEL_HEIGHT * len(EXTRA_PANELS)
SCROLL_STEP = 0.08


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


def terrain_panorama_samples(lat_deg, lon_deg, texture_rgb, max_dist_km=250.0, step_km=0.5):
    """Project terrain samples into azimuth/elevation and sample map colors.

    Distances match compute_horizon_360. The DEM and map are coarse; this is
    an approximate, daytime surface view, not a photograph or lighting model.
    """
    az = np.radians(np.arange(360))
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
    pixels = np.zeros((rows, 361, 4), dtype=np.uint8)
    for bearing in range(360):
        indices = np.searchsorted(angles[bearing], levels, side="left")
        visible = indices < angles.shape[1]
        pixels[visible, bearing, :3] = colors[bearing, indices[visible]]
        pixels[visible, bearing, 3] = 255
    pixels[:, 360] = pixels[:, 0]
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


class CoordinateDialog:
    """Native text editing and clipboard support, opened only by G."""

    def __init__(self, parent, initial_text, on_submit):
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.on_submit = on_submit
        self.window = tk.Toplevel(parent)
        self.window.title("Go to coordinates")
        self.window.resizable(False, False)
        frame = ttk.Frame(self.window, padding=14)
        frame.grid(sticky="nsew")
        ttk.Label(frame, text="Latitude, longitude (signed or 0 to 360)").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))
        self.text = tk.Text(frame, width=44, height=2, wrap="word", font="TkFixedFont",
                            undo=True, autoseparators=True, maxundo=100,
                            exportselection=False, padx=7, pady=7)
        self.text.grid(row=1, column=0, columnspan=2, sticky="ew")
        self.text.insert("1.0", initial_text)
        self.text.edit_reset()
        self.error = ttk.Label(frame, foreground="#c02020", wraplength=430)
        self.error.grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 4))
        ttk.Button(frame, text="Cancel", command=self.window.destroy).grid(
            row=3, column=0, sticky="e", padx=(0, 8))
        ttk.Button(frame, text="Go", command=self.submit).grid(row=3, column=1, sticky="e")
        self.text.bind("<Return>", self.submit)
        self.text.bind("<KP_Enter>", self.submit)
        self.window.bind("<Escape>", lambda event: self.window.destroy())

        # Handle selection replacement consistently on X11, Windows, and macOS.
        self.text.bind("<<Paste>>", self.paste)
        self.text.bind("<<Copy>>", self.copy)
        self.text.bind("<<Cut>>", self.cut)
        self.text.bind("<<Undo>>", self.undo)
        self.text.bind("<<Redo>>", self.redo)
        shortcuts = {"a": self.select_all, "c": self.copy, "x": self.cut,
                     "v": self.paste, "z": self.undo, "y": self.redo}
        for modifier in (("Control", "Command") if sys.platform == "darwin" else ("Control",)):
            for key, handler in shortcuts.items():
                self.text.bind(f"<{modifier}-Key-{key}>", handler)
            self.text.bind(f"<{modifier}-Shift-Key-Z>", self.redo)
        self.text.bind("<Shift-Insert>", self.paste)
        self.menu = tk.Menu(self.window, tearoff=False)
        for label, handler in (("Undo", self.undo), ("Redo", self.redo),
                               ("Cut", self.cut), ("Copy", self.copy),
                               ("Paste", self.paste), ("Select all", self.select_all)):
            self.menu.add_command(label=label, command=handler)
        self.text.bind("<Button-3>", self.show_menu)
        if sys.platform == "darwin":
            self.text.bind("<Control-Button-1>", self.show_menu)
        self.window.protocol("WM_DELETE_WINDOW", self.window.destroy)
        self.window.update_idletasks()
        self.window.lift()
        self.window.grab_set()
        self.text.focus_force()
        self.select_all()

    def select_all(self, event=None):
        self.text.tag_add("sel", "1.0", "end-1c")
        self.text.mark_set("insert", "end-1c")
        return "break"

    def replace_selection(self, value):
        self.text.edit_separator()
        automatic = self.text.cget("autoseparators")
        self.text.configure(autoseparators=False)
        try:
            if self.text.tag_ranges("sel"):
                self.text.delete("sel.first", "sel.last")
            self.text.insert("insert", value)
            self.text.see("insert")
        finally:
            self.text.edit_separator()
            self.text.configure(autoseparators=automatic)

    def copy(self, event=None):
        if self.text.tag_ranges("sel"):
            value = self.text.get("sel.first", "sel.last")
            self.window.clipboard_clear()
            self.window.clipboard_append(value)
        return "break"

    def cut(self, event=None):
        if self.text.tag_ranges("sel"):
            self.copy()
            self.replace_selection("")
        return "break"

    def paste(self, event=None):
        try:
            value = self.window.clipboard_get()
        except self.tk.TclError:
            return "break"  # Empty clipboard or clipboard has no text.
        self.replace_selection(value)
        return "break"

    def undo(self, event=None):
        try:
            self.text.edit_undo()
        except self.tk.TclError:
            pass
        return "break"

    def redo(self, event=None):
        try:
            self.text.edit_redo()
        except self.tk.TclError:
            pass
        return "break"

    def show_menu(self, event):
        try:
            self.menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.menu.grab_release()
        return "break"

    def submit(self, event=None):
        try:
            self.on_submit(self.text.get("1.0", "end-1c"))
        except ValueError as error:
            self.error.configure(text=str(error))
            self.text.focus_set()
        else:
            self.window.destroy()
        return "break"


class TimeIntervalDialog:
    def __init__(self, parent, start, stop, step, unit, on_submit, method="Skyfield"):
        import tkinter as tk
        from tkinter import ttk

        self.window = tk.Toplevel(parent)
        self.window.title("Sun / Earth time interval (UTC)")
        self.window.resizable(False, False)
        frame = ttk.Frame(self.window, padding=14)
        frame.grid()
        self.fields = []
        values = (start.strftime("%Y-%m-%d %H:%M:%S"),
                  stop.strftime("%Y-%m-%d %H:%M:%S"), str(step))
        for row, (label, value) in enumerate(zip(
                ("Start UTC", "End UTC", "Step"), values)):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", padx=(0, 12), pady=4)
            entry = ttk.Entry(frame, width=28)
            entry.insert(0, value)
            entry.grid(row=row, column=1, sticky="ew", pady=4)
            entry.bind("<Control-a>", lambda event: (
                event.widget.selection_range(0, "end"), "break")[-1])
            self.fields.append(entry)
        self.unit = ttk.Combobox(frame, values=("seconds", "minutes", "hours", "days"), state="readonly", width=10)
        self.unit.set(unit)
        self.unit.grid(row=2, column=2, padx=(8, 0))
        ttk.Label(frame, text="Model").grid(row=3, column=0, sticky="w", padx=(0, 12), pady=4)
        self.method = ttk.Combobox(frame, values=tuple(SKY_METHODS), state="readonly", width=26)
        self.method.set(method if method in SKY_METHODS else "Skyfield")
        self.method.grid(row=3, column=1, sticky="ew", pady=4)
        ttk.Label(frame, text=f"UTC dates: YYYY-MM-DD HH:MM:SS. Up to {MAX_TIME_SAMPLES} samples.").grid(
            row=4, column=0, columnspan=2, sticky="w", pady=6)
        self.error = ttk.Label(frame, foreground="#c02020", wraplength=430)
        self.error.grid(row=5, column=0, columnspan=2, sticky="w")

        self.apply_button = ttk.Button(frame, text="Apply interval")

        def apply(event=None):
            try:
                on_submit(self.fields[0].get(), self.fields[1].get(),
                          float(self.fields[2].get()), self.unit.get(), self.method.get())
            except ValueError as error:
                self.error.configure(text=str(error))
            else:
                self.window.destroy()
            return "break"

        ttk.Button(frame, text="Cancel", command=self.window.destroy).grid(row=6, column=0, pady=(10, 0))
        self.apply_button.configure(command=apply)
        self.apply_button.grid(row=6, column=1, sticky="e", pady=(10, 0))
        self.window.bind("<Return>", apply)
        self.window.bind("<Escape>", lambda event: self.window.destroy())
        self.window.update_idletasks()
        self.window.lift()
        self.window.grab_set()
        self.fields[0].focus_force()
        self.fields[0].selection_range(0, "end")


class MoonWindow:
    """VTK owns the desktop window and mouse interaction; no HTML is involved."""

    def __init__(self, offscreen=False, enable_ephemeris=True, ephemeris_fetcher=None):
        import vtk
        from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

        self.vtk = vtk
        self.lat, self.lon = -43.31, -11.36
        self.orbit_pivot = np.zeros(3)
        self._enable_ephemeris = enable_ephemeris
        self._ephemeris_fetcher = ephemeris_fetcher  # optional override used for any model (testing)
        self._sky_jobs = queue.Queue(maxsize=1)
        self._sky_results = queue.Queue()
        self._sky_progress = queue.Queue()
        self._sky_stop = threading.Event()
        self._sky_thread = None
        self._preload_lock = threading.Lock()
        self._preload_state = ("loading", "") if enable_ephemeris and ephemeris_fetcher is None else ("ready", "")
        self._preload_running = False
        self._preload_done = threading.Event()
        if self._preload_state[0] == "ready":
            self._preload_done.set()
        self._sky_timer = self._sky_pump_observer = None
        self._sky_generation = 0
        self._sky_data = None
        self._sky_status = "Waiting for Sun / Earth positions"
        self._now_mode = True
        self._shown_sources = {"Skyfield": True, "JPL": True}
        self._legend_expanded = True
        self._saved_interval = load_interval()
        self._sky_method = self._saved_interval[4] if self._saved_interval else "Skyfield"
        self._graph_expanded = False  # False: horizon only. True: fullscreen horizon + Sun/Earth graph.
        self._scroll = 0.0  # window heights the expanded page is scrolled down
        self._panel_boxes = {}
        self._epochs = [utc_now()]
        self._time_index = 0
        self._interval_step_minutes = 60
        self._last_now_query = time.monotonic()
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.025, 0.035, 0.055)
        self.renderer.UseFXAAOn()
        self.window = vtk.vtkRenderWindow()
        self.window.SetWindowName("Moon Horizon - Desktop Viewer")
        self.window.SetSize(1280, 900)
        self.window.SetMultiSamples(8)
        self.window.AddRenderer(self.renderer)
        if offscreen:
            self.window.SetOffScreenRendering(True)
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.window)

        # Keep the full source texture instead of reducing it to 540 x 270.
        # Duplicate the longitude seam so texture coordinates wrap correctly.
        columns, rows = 721, 361
        lon, lat = np.meshgrid(np.linspace(-180, 180, columns),
                               np.linspace(90, -90, rows))
        x, y, z = lat_lon_to_xyz(lat, lon, offset=0.0)
        # Every longitude meets at one point at each pole. DEM pixel sampling
        # must not create overlapping tips with different heights there.
        for row in (0, -1):
            x[row, :] = y[row, :] = 0.0
            z[row, :] = float(np.mean(z[row, :]))
        points = vtk.vtkPoints()
        points.SetData(numpy_to_vtk(np.column_stack((x.ravel(), y.ravel(), z.ravel())), deep=True))
        indices = np.arange(rows * columns).reshape(rows, columns)
        a, b = indices[:-1, :-1].ravel(), indices[:-1, 1:].ravel()
        c, d = indices[1:, 1:].ravel(), indices[1:, :-1].ravel()
        # Outward facing quads, suitable for texture interpolation.
        cells = np.column_stack((a, d, c, b)).astype(np.int64)
        offsets = np.arange(0, cells.size + 1, 4, dtype=np.int64)
        polygons = vtk.vtkCellArray()
        polygons.SetData(numpy_to_vtkIdTypeArray(offsets, deep=True),
                         numpy_to_vtkIdTypeArray(cells.ravel(), deep=True))
        self.mesh = vtk.vtkPolyData()
        self.mesh.SetPoints(points)
        self.mesh.SetPolys(polygons)
        uv = np.column_stack(((lon.ravel() + 180) / 360,
                              (lat.ravel() + 90) / 180)).astype(np.float32)
        self.mesh.GetPointData().SetTCoords(numpy_to_vtk(uv, deep=True))
        self.surface_locator = vtk.vtkStaticCellLocator()
        self.surface_locator.SetDataSet(self.mesh)
        self.surface_locator.BuildLocator()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.mesh)
        reader = vtk.vtkImageReader2Factory.CreateImageReader2(COLOR_PATH)
        if reader is None:
            raise ValueError(f"Unsupported or unreadable texture: {COLOR_PATH}")
        reader.SetFileName(COLOR_PATH)
        reader.Update()
        self.texture_dimensions = reader.GetOutput().GetDimensions()[:2]
        self.texture = vtk.vtkTexture()
        self.texture.SetInputConnection(reader.GetOutputPort())
        self.texture.InterpolateOn()
        self.texture.MipmapOn()
        self.moon = vtk.vtkActor()
        self.moon.SetMapper(mapper)
        self.moon.SetTexture(self.texture)
        # Stable, camera-independent illumination, without a moving highlight.
        self.moon.GetProperty().LightingOff()
        self.renderer.AddActor(self.moon)
        self.surface_bound = float(np.max(np.sqrt(x*x + y*y + z*z)))
        # This bound sizes ray casts and limits zoom-out only. Zoom-in measures
        # the actual rendered terrain, not a sphere around its highest point.
        self.minimum_surface_clearance = 250.0 / MOON_RADIUS_M
        self.maximum_camera_radius = max(15.0, self.surface_bound * 2)

        self.horizon = vtk.vtkActor()
        self.horizon.GetProperty().SetColor(0.0, 0.9, 1.0)
        self.horizon.GetProperty().SetLineWidth(3)
        self.horizon.GetProperty().LightingOff()
        self.horizon.PickableOff()
        self.renderer.AddActor(self.horizon)
        # A flat screen-sized marker sits at the terrain hit without a raised
        # world-space sphere or any artificial radial height.
        self.observer_points = vtk.vtkPoints()
        self.observer_points.SetDataTypeToDouble()
        self.observer_points.InsertNextPoint(0, 0, 0)
        observer_data = vtk.vtkPolyData()
        observer_data.SetPoints(self.observer_points)
        observer_vertices = vtk.vtkCellArray()
        observer_vertices.InsertNextCell(1)
        observer_vertices.InsertCellPoint(0)
        observer_data.SetVerts(observer_vertices)
        marker_mapper = vtk.vtkPolyDataMapper()
        marker_mapper.SetInputData(observer_data)
        self.configure_surface_overlay(marker_mapper)
        self.observer = vtk.vtkActor()
        self.observer.SetMapper(marker_mapper)
        self.observer.GetProperty().SetColor(1.0, 0.3, 0.15)
        self.observer.GetProperty().LightingOff()
        self.observer.GetProperty().SetPointSize(12)
        self.observer.GetProperty().RenderPointsAsSpheresOff()
        self.observer.PickableOff()
        self.renderer.AddActor(self.observer)

        self.stats = self.make_text(0.025, 0.97, 19, top=True)
        self.coordinate_editing = False
        self._coordinate_root = None
        self._coordinate_timer = None
        self._coordinate_pump_observer = None
        self.help = self.make_text(0.025, 0.035, 15)
        self.help.SetInput("Left-drag: orbit   |   Middle/Shift-drag: pan\n"
                           "Ctrl-drag: orbit cursor point   |   Wheel: zoom   |   Click: select\n"
                           "G: coordinates   |   T: time interval   |   R: reset\n"
                           "F: fullscreen   |   Esc: close\n"
                           "Cyan: local horizon   |   Orange: observer   |   Relief: 3x")
        self.profile_background = self.make_profile_background()
        self._panorama_samples = None
        self._panorama_range = None
        self._panorama_texture_rgb = None
        self.panorama = self.make_panorama_actor()
        self.profile_title = self.make_text(0.61, 0.32, 14, top=True)
        self.profile = self.make_overlay_lines([], (0.0, 0.9, 1.0), 2)
        self.profile_axes = self.make_overlay_lines([
            [(0.61, 0.12), (0.97, 0.12), (0.97, 0.25), (0.61, 0.25), (0.61, 0.12)],
            *[[(x, 0.12), (x, 0.25)] for x in (0.70, 0.79, 0.88)]
        ], (0.4, 0.48, 0.58), 1)
        self.profile_labels = []
        for x, text in zip(np.linspace(0.61, 0.97, 5), ("N 0", "E 90", "S 180", "W 270", "N 360")):
            label = self.make_text(float(x), 0.09, 12)
            label.GetTextProperty().SetJustificationToCentered()
            label.SetInput(text)
            self.profile_labels.append(label)
        self.sky_trails = {name: self.make_overlay_lines([], color, 2) for name, color in SKY_COLORS.items()}
        self.sky_samples = {name: self.make_overlay_points([], color, 4) for name, color in SKY_COLORS.items()}
        self.sky_markers = {name: self.make_overlay_points([], color, 10) for name, color in SKY_COLORS.items()}
        self.sky_labels = {name: self.make_text(0.61, 0.26, 12) for name in SKY_COLORS}
        for name, actor in self.sky_labels.items():
            actor.GetTextProperty().SetColor(*SKY_COLORS[name])
            actor.SetVisibility(False)
        self.source_buttons = {source: self.make_text(0.64, 0.7, 15) for source in ("Skyfield", "JPL")}
        self.legend_button = self.make_text(0.025, 0.98, 16, top=True)
        self.progress_text = self.make_text(0.35, 0.025, 16)
        self.progress_text.SetVisibility(False)
        self.time_button = self.make_text(0.61, 0.46, 16)
        self.time_button.SetInput("[ Edit time interval / T ]")
        self.time_button.GetTextProperty().SetColor(0.25, 0.9, 1.0)
        self.now_button = self.make_text(0.90, 0.46, 16)
        self.now_button.SetInput("[ Now ]")
        self.time_caption = self.make_text(0.61, 0.43, 12, top=True)
        self.previous_button = self.make_text(0.61, 0.37, 14)
        self.previous_button.SetInput("[ Previous ]")
        self.next_button = self.make_text(0.77, 0.37, 14)
        self.next_button.SetInput("[ Next ]")
        self.close_button = self.make_text(0.91, 0.46, 16)
        self.close_button.SetInput("[ Close ]")
        self.panels = {}
        for key in EXTRA_PANELS:
            title = self.make_text(0.08, 0.0, 14, top=True)
            axes = self.make_overlay_lines([], (0.4, 0.48, 0.58), 1)
            xlabels, ylabels = [], []
            for _ in range(5):
                label = self.make_text(0.0, 0.0, 12)
                label.GetTextProperty().SetJustificationToCentered()
                xlabels.append(label)
            for _ in range(2):
                label = self.make_text(0.0, 0.0, 12)
                label.GetTextProperty().SetJustificationToRight()
                label.GetTextProperty().SetVerticalJustificationToCentered()
                ylabels.append(label)
            self.panels[key] = {"title": title, "axes": axes, "xlabels": xlabels, "ylabels": ylabels}
        self.panel_terrain = {name: self.make_overlay_lines([], (0.0, 0.9, 1.0), 1) for name in SKY_COLORS}
        self.panel_lines = {name: self.make_overlay_lines([], color, 2) for name, color in SKY_COLORS.items()}
        self.time_cursor = self.make_overlay_lines([], (0.9, 0.95, 1.0), 1)
        self.scrollbar = self.make_overlay_lines([], (0.45, 0.55, 0.7), 6)
        self.apply_layout()
        self.picker = vtk.vtkCellPicker()
        self.picker.SetTolerance(0.001)
        self.picker.PickFromListOn()
        self.picker.AddPickList(self.moon)

        owner = self

        class MoonInteraction(vtk.vtkInteractorStyleUser):
            def __init__(self):
                super().__init__()
                self._moon_drag_press = None
                self._moon_drag_last = None
                self._moon_drag_dragged = False
                self._moon_drag_mode = self._moon_drag_button = None
                self._moon_drag_pan_depth = None
                self._moon_drag_cursor_pivot = None
                self.AddObserver("LeftButtonPressEvent", self.left_down)
                self.AddObserver("LeftButtonReleaseEvent", self.left_up)
                self.AddObserver("MiddleButtonPressEvent", self.middle_down)
                self.AddObserver("MiddleButtonReleaseEvent", self.middle_up)
                self.AddObserver("MouseMoveEvent", self.mouse_move)
                self.AddObserver("MouseWheelForwardEvent", lambda obj, event: owner.wheel(-1))
                self.AddObserver("MouseWheelBackwardEvent", lambda obj, event: owner.wheel(1))
                self.AddObserver("KeyPressEvent", self.key_press)
                self.AddObserver("CharEvent", self.key)
                self.AddObserver("ConfigureEvent", lambda obj, event: owner.constrain_camera())

            def left_down(self, obj, event):
                interactor = self.GetInteractor()
                mode = "look" if interactor.GetControlKey() else ("pan" if interactor.GetShiftKey() else "orbit")
                self.start_drag(mode, "left")

            def middle_down(self, obj, event):
                mode = "look" if self.GetInteractor().GetControlKey() else "pan"
                self.start_drag(mode, "middle")

            def start_drag(self, mode, button):
                if owner.coordinate_editing or self._moon_drag_press is not None:
                    return
                self._moon_drag_mode, self._moon_drag_button = mode, button
                self._moon_drag_press = self.GetInteractor().GetEventPosition()
                if owner.over_sky_controls(*self._moon_drag_press):
                    self._moon_drag_mode = "ui"
                self._moon_drag_last = self._moon_drag_press
                self._moon_drag_dragged = False
                self._moon_drag_cursor_pivot = (owner.cursor_surface_target(*self._moon_drag_press)
                                                if self._moon_drag_mode == "look" else None)
                self._moon_drag_pan_depth = float(np.linalg.norm(
                    np.array(owner.renderer.GetActiveCamera().GetPosition()) - owner.view_surface_target()))

            def mouse_move(self, obj, event):
                if self._moon_drag_press is None or owner.coordinate_editing:
                    return
                current = self.GetInteractor().GetEventPosition()
                if self._moon_drag_mode == "ui":
                    if np.hypot(current[0] - self._moon_drag_press[0], current[1] - self._moon_drag_press[1]) > 5:
                        self._moon_drag_dragged = True
                    return
                if ((self._moon_drag_mode in ("pan", "look") and current != self._moon_drag_press)
                        or np.hypot(current[0] - self._moon_drag_press[0], current[1] - self._moon_drag_press[1]) > 5):
                    self._moon_drag_dragged = True
                if self._moon_drag_dragged:
                    dx, dy = current[0] - self._moon_drag_last[0], current[1] - self._moon_drag_last[1]
                    if self._moon_drag_mode == "pan":
                        owner.pan(dx, dy, depth=self._moon_drag_pan_depth)
                    elif self._moon_drag_mode == "look":
                        owner.orbit_cursor(dx, dy, self._moon_drag_cursor_pivot)
                    else:
                        owner.orbit(dx, dy)
                self._moon_drag_last = current

            def left_up(self, obj, event):
                self.end_drag("left")

            def middle_up(self, obj, event):
                self.end_drag("middle")

            def end_drag(self, button):
                if self._moon_drag_button != button:
                    return
                if self._moon_drag_press is not None and not self._moon_drag_dragged and self._moon_drag_mode in ("orbit", "ui") and button == "left":
                    owner.pick(*self.GetInteractor().GetEventPosition())
                self._moon_drag_press = self._moon_drag_last = None
                self._moon_drag_mode = self._moon_drag_button = self._moon_drag_pan_depth = None
                self._moon_drag_cursor_pivot = None

            def key_press(self, obj, event):
                interactor = self.GetInteractor()
                key = (interactor.GetKeySym() or "").lower()
                if owner.coordinate_editing:
                    return  # The native editor handles its own keyboard events.
                elif key == "escape":
                    owner.interactor.TerminateApp()
                elif owner._graph_expanded and key in ("up", "down", "prior", "next", "home", "end"):
                    owner.scroll_key(key)

            def key(self, obj, event):
                interactor = self.GetInteractor()
                key = (interactor.GetKeySym() or "").lower()
                if owner.coordinate_editing:
                    return
                if key in ("escape", "return", "kp_enter", "backspace", "delete") or interactor.GetControlKey():
                    return  # Special keys were handled once in KeyPressEvent.
                if key == "g":
                    owner.begin_coordinate_entry()
                elif key == "t":
                    owner.begin_time_entry()
                elif key == "bracketleft":
                    owner.step_time(-1)
                elif key == "bracketright":
                    owner.step_time(1)
                elif key == "r":
                    owner.reset_camera()
                    owner.window.Render()
                elif key == "f":
                    owner.toggle_fullscreen()
                    owner.window.Render()
                # Ignore VTK's default wireframe/representation shortcuts.

        self.style = MoonInteraction()
        self.interactor.SetInteractorStyle(self.style)
        self.select_location(self.lat, self.lon, render=False)
        self.reset_camera()
        if self._enable_ephemeris and self._ephemeris_fetcher is None:
            self.start_skyfield_preload()

    def start_skyfield_preload(self):
        with self._preload_lock:
            if self._preload_running or self._preload_state[0] == "ready":
                return
            self._preload_running = True
            self._preload_done.clear()
            self._preload_state = ("loading", "")
        def load_data():
            try:
                from skyfield_positions import preload
                preload()
            except Exception as error:
                message = str(error).strip().splitlines()
                result = ("error", (message[0] if message else type(error).__name__)[:200])
            else:
                result = ("ready", "")
            with self._preload_lock:
                self._preload_state = result
                self._preload_running = False
                self._preload_done.set()
        threading.Thread(target=load_data, daemon=True).start()

    def skyfield_preload_status(self):
        with self._preload_lock:
            return self._preload_state

    def make_text(self, x, y, size, top=False):
        actor = self.vtk.vtkTextActor()
        actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        actor.SetPosition(x, y)
        prop = actor.GetTextProperty()
        prop.SetFontFamilyToArial()
        prop.SetFontSize(size)
        prop.SetColor(0.92, 0.96, 1.0)
        prop.SetBackgroundColor(0.025, 0.035, 0.055)
        prop.SetBackgroundOpacity(0.85)
        if top:
            prop.SetVerticalJustificationToTop()
        self.renderer.AddViewProp(actor)
        return actor

    def overlay_mapper(self, data):
        mapper = self.vtk.vtkPolyDataMapper2D()
        mapper.SetInputData(data)
        coordinate = self.vtk.vtkCoordinate()
        coordinate.SetCoordinateSystemToNormalizedViewport()
        mapper.SetTransformCoordinate(coordinate)
        return mapper

    def make_profile_background(self):
        points = self.vtk.vtkPoints()
        self._bg_points = points
        for x, y in ((0.59, 0.055), (0.99, 0.055), (0.99, 0.35), (0.59, 0.35)):
            points.InsertNextPoint(x, y, 0)
        cells = self.vtk.vtkCellArray()
        cells.InsertNextCell(4)
        for index in range(4):
            cells.InsertCellPoint(index)
        data = self.vtk.vtkPolyData()
        data.SetPoints(points)
        data.SetPolys(cells)
        actor = self.vtk.vtkActor2D()
        actor.SetMapper(self.overlay_mapper(data))
        actor.GetProperty().SetColor(0.025, 0.035, 0.055)
        self.renderer.AddViewProp(actor)
        return actor

    def make_panorama_actor(self):
        """A textured quad between the graph background and its overlays."""
        points = self.vtk.vtkPoints()
        for _ in range(4):
            points.InsertNextPoint(0, 0, 0)
        self._panorama_quad_points = points
        cells = self.vtk.vtkCellArray()
        cells.InsertNextCell(4)
        for index in range(4):
            cells.InsertCellPoint(index)
        data = self.vtk.vtkPolyData()
        data.SetPoints(points)
        data.SetPolys(cells)
        coords = self.vtk.vtkFloatArray()
        coords.SetNumberOfComponents(2)
        for uv in ((0, 0), (1, 0), (1, 1), (0, 1)):
            coords.InsertNextTuple2(*uv)
        data.GetPointData().SetTCoords(coords)
        self._panorama_image = self.vtk.vtkImageData()
        texture = self.vtk.vtkTexture()
        texture.SetInputData(self._panorama_image)
        texture.InterpolateOn()
        actor = self.vtk.vtkTexturedActor2D()
        actor.SetMapper(self.overlay_mapper(data))
        actor.SetTexture(texture)
        actor.PickableOff()
        self._panorama_texture = texture
        self.renderer.AddViewProp(actor)
        return actor

    def update_panorama(self):
        if self._panorama_samples is None:
            self.panorama.SetVisibility(False)
            return
        self.panorama.SetVisibility(True)
        x0, x1, y0, y1 = self._plot_box
        for i, (x, y) in enumerate(((x0, y0), (x1, y0), (x1, y1), (x0, y1))):
            self._panorama_quad_points.SetPoint(i, x, y, 0)
        self._panorama_quad_points.Modified()
        self.panorama.GetMapper().GetInput().Modified()
        limits = (self._profile_low, self._profile_high)
        if limits != self._panorama_range:
            from vtk.util.numpy_support import numpy_to_vtk
            rgba = terrain_panorama_rgba(self._panorama_samples, *limits)
            self._panorama_image.SetDimensions(rgba.shape[1], rgba.shape[0], 1)
            colors = numpy_to_vtk(rgba.reshape(-1, 4), deep=True, array_type=self.vtk.VTK_UNSIGNED_CHAR)
            colors.SetNumberOfComponents(4)
            self._panorama_image.GetPointData().SetScalars(colors)
            self._panorama_image.Modified()
            self._panorama_range = limits

    def overlay_lines_data(self, segments):
        points = self.vtk.vtkPoints()
        cells = self.vtk.vtkCellArray()
        for segment in segments:
            cells.InsertNextCell(len(segment))
            for x, y in segment:
                cells.InsertCellPoint(points.InsertNextPoint(float(x), float(y), 0))
        data = self.vtk.vtkPolyData()
        data.SetPoints(points)
        data.SetLines(cells)
        return data

    def make_overlay_lines(self, segments, color, width):
        actor = self.vtk.vtkActor2D()
        actor.SetMapper(self.overlay_mapper(self.overlay_lines_data(segments)))
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetLineWidth(width)
        self.renderer.AddViewProp(actor)
        return actor

    def overlay_point_data(self, coordinates):
        points = self.vtk.vtkPoints()
        cells = self.vtk.vtkCellArray()
        for x, y in coordinates:
            index = points.InsertNextPoint(float(x), float(y), 0)
            cells.InsertNextCell(1)
            cells.InsertCellPoint(index)
        data = self.vtk.vtkPolyData()
        data.SetPoints(points)
        data.SetVerts(cells)
        return data

    def make_overlay_points(self, coordinates, color, size):
        actor = self.vtk.vtkActor2D()
        actor.SetMapper(self.overlay_mapper(self.overlay_point_data(coordinates)))
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetPointSize(size)
        self.renderer.AddViewProp(actor)
        return actor

    def apply_layout(self):
        """Small inset with only the horizon, or a fullscreen combined graph."""
        expanded = self._graph_expanded
        s = self._scroll if expanded else 0.0  # screen y = page y + s
        if expanded:
            self._plot_box = (0.06, 0.97, 0.10 + s, 0.62 + s)
            background = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
            bx = 0.64
            self._controls = {"time": (bx, 0.92 + s, 0.83, 0.96 + s), "now": (0.85, 0.92 + s, 0.905, 0.96 + s),
                              "close": (0.91, 0.92 + s, 0.985, 0.96 + s),
                              "previous": (bx, 0.79 + s, bx + 0.13, 0.825 + s),
                              "next": (bx + 0.16, 0.79 + s, bx + 0.28, 0.825 + s),
                              **{source: (bx, 0.625 + s, bx + 0.16, 0.66 + s) for source in SKY_METHODS[self._sky_method]},
                              "legend": (0.02, 0.95 + s, 0.30, 0.995 + s)}
            positions = {self.time_button: (bx, 0.93 + s), self.now_button: (0.85, 0.93 + s),
                         self.close_button: (0.91, 0.93 + s), self.time_caption: (bx, 0.89 + s),
                         self.previous_button: (bx, 0.795 + s), self.next_button: (bx + 0.16, 0.795 + s),
                         self.profile_title: (bx, 0.765 + s)}
        else:
            self._plot_box = (0.61, 0.97, 0.12, 0.25)
            background = ((0.59, 0.055), (0.99, 0.055), (0.99, 0.35), (0.59, 0.35))
            self._controls = {"time": (0.61, 0.45, 0.86, 0.49), "legend": (0.02, 0.95, 0.30, 0.995)}
            positions = {self.time_button: (0.61, 0.46), self.now_button: (0.90, 0.46),
                         self.close_button: (0.91, 0.46), self.time_caption: (0.61, 0.43),
                         self.previous_button: (0.61, 0.37), self.next_button: (0.77, 0.37),
                         self.profile_title: (0.61, 0.32)}
        for index, (x, y) in enumerate(background):
            self._bg_points.SetPoint(index, x, y, 0)
        self._bg_points.Modified()
        self.profile_background.GetMapper().GetInput().Modified()
        for actor in self.source_buttons.values():
            actor.SetPosition(0.64, 0.635 + s)
        self.legend_button.SetInput("[v] Point details" if self._legend_expanded else "[>] Point details")
        self.legend_button.SetVisibility(True)
        self.legend_button.SetPosition(0.025, 0.98 + s)
        self.stats.SetPosition(0.025, 0.93 + s)
        self.stats.SetVisibility(self._legend_expanded)
        for source, actor in self.source_buttons.items():
            actor.SetInput(("[x] " if self._shown_sources[source] else "[ ] ") + source)
            actor.SetVisibility(expanded and source in SKY_METHODS[self._sky_method])
        for actor, (x, y) in positions.items():
            actor.SetPosition(x, y)
        x0, x1, y0, y1 = self._plot_box
        verticals = [[(x0 + (x1 - x0) * f, y0), (x0 + (x1 - x0) * f, y1)] for f in (0.25, 0.5, 0.75)]
        self.profile_axes.GetMapper().SetInputData(self.overlay_lines_data(
            [[(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)], *verticals]))
        for x, label in zip(np.linspace(x0, x1, 5), self.profile_labels):
            label.SetPosition(float(x), y0 - 0.03)
        for actor in (self.now_button, self.close_button, self.time_caption,
                      self.previous_button, self.next_button):
            actor.SetVisibility(expanded)
        self.help.SetVisibility(not expanded)
        self.layout_panels(s)
        # Scroll bar: thumb length = visible fraction of the page.
        thumb = []
        if expanded:
            top = 0.98 - 0.96 * s / PAGE_HEIGHT
            thumb = [[(0.993, top), (0.993, top - 0.96 / PAGE_HEIGHT)]]
        self.scrollbar.GetMapper().SetInputData(self.overlay_lines_data(thumb))
        # Re-add the stats text last so it is drawn above the fullscreen background.
        self.renderer.RemoveViewProp(self.stats)
        self.renderer.AddViewProp(self.stats)
        self.renderer.RemoveViewProp(self.progress_text)
        self.renderer.AddViewProp(self.progress_text)

    def layout_panels(self, s):
        """Place the extra graphs below the main one; page y = -PANEL_HEIGHT * k, screen y = page y + s."""
        for k, key in enumerate(EXTRA_PANELS, start=1):
            base = -PANEL_HEIGHT * k + s
            x0, x1, y0, y1 = 0.08, 0.97, base + 0.10, base + 0.38
            self._panel_boxes[key] = (x0, x1, y0, y1)
            panel = self.panels[key]
            verticals = [[(x0 + (x1 - x0) * f, y0), (x0 + (x1 - x0) * f, y1)] for f in (0.25, 0.5, 0.75)]
            panel["axes"].GetMapper().SetInputData(self.overlay_lines_data(
                [[(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)], *verticals]))
            panel["title"].SetPosition(x0, base + 0.47)
            for x, label in zip(np.linspace(x0, x1, 5), panel["xlabels"]):
                label.SetPosition(float(x), y0 - 0.035)
            panel["ylabels"][0].SetPosition(x0 - 0.005, y0)
            panel["ylabels"][1].SetPosition(x0 - 0.005, y1)

    def scroll_to(self, value):
        if not self._graph_expanded:
            return
        value = float(np.clip(value, 0.0, PAGE_HEIGHT - 1.0))
        if value == self._scroll:
            return
        self._scroll = value
        self.apply_layout()
        self.update_profile(self.horizon_angles)
        self.window.Render()

    def scroll_key(self, key):
        if key == "home":
            self.scroll_to(0.0)
        elif key == "end":
            self.scroll_to(PAGE_HEIGHT - 1.0)
        else:
            self.scroll_to(self._scroll + {"up": -SCROLL_STEP, "down": SCROLL_STEP,
                                           "prior": -0.8, "next": 0.8}[key])

    def wheel(self, direction):
        """Wheel zooms the Moon, or scrolls the page while the graph window is open."""
        if self._graph_expanded:
            self.scroll_to(self._scroll + direction * SCROLL_STEP)
        else:
            self.zoom(1.08 if direction > 0 else 1 / 1.08)

    def update_panels(self):
        """Sun and Earth elevation graphs."""
        expanded = self._graph_expanded
        for panel in self.panels.values():
            for actor in (panel["title"], *panel["xlabels"], *panel["ylabels"]):
                actor.SetVisibility(expanded)
        for actor in (*self.panel_terrain.values(), *self.panel_lines.values(), self.time_cursor):
            actor.GetMapper().SetInputData(self.overlay_lines_data([]))
        if not expanded:
            return
        data, epochs = self._sky_data, self._epochs
        ready = data is not None and len(epochs) > 1
        if data is None:
            note = self._sky_status or "Waiting for Sun / Earth positions"
        else:
            note = "Needs an interval with 2 or more samples (Edit time interval)"

        def hide_labels(panel):
            for actor in (*panel["xlabels"], *panel["ylabels"]):
                actor.SetVisibility(False)

        def shown(name):
            return data is not None and name in data and self._shown_sources[name.rsplit(" ", 1)[-1]]

        seconds = np.array([(epoch - epochs[0]).total_seconds() for epoch in epochs])
        span = float(seconds[-1])
        cursor = []
        for key in ("Sun", "Earth"):
            panel, (x0, x1, y0, y1) = self.panels[key], self._panel_boxes[key]
            title = f"{key} elevation above the horizon vs time (UTC)"
            names = [name for name in SKY_COLORS if name.startswith(key + " ") and shown(name)]
            if not ready or not names:
                panel["title"].SetInput(title + "\n" + (note if not ready else "Source hidden (see the checkbox)"))
                hide_labels(panel)
                continue
            xs = x0 + (x1 - x0) * seconds / span
            # Plot the body centre relative to the terrain in its direction.
            low = 0.0
            high = max(0.01, max(float(np.max(np.maximum(
                data[n]["el"] - data[n]["terrain"], 0.0))) for n in names) * 1.08)
            for name in names:
                clearance = np.maximum(data[name]["el"] - data[name]["terrain"], 0.0)
                ys = y0 + (y1 - y0) * clearance / high
                self.panel_lines[name].GetMapper().SetInputData(
                    self.overlay_lines_data([np.column_stack((xs, ys))]))
            panel["title"].SetInput(f"{key} centre above terrain horizon vs time (UTC)\n"
                                    "0 deg = terrain horizon; below horizon shown as 0\n"
                                    f"Scale 0 to {high:.1f} deg")
            for fraction, label in zip(np.linspace(0, 1, 5), panel["xlabels"]):
                label.SetInput(f"{epochs[0] + timedelta(seconds=float(fraction) * span):%m-%d %H:%M}")
                label.SetVisibility(True)
            for label, value in zip(panel["ylabels"], (low, high)):
                label.SetInput(f"{value:+.1f}")
                label.SetVisibility(True)
            x = float(xs[self._time_index])
            cursor.append([(x, y0), (x, y1)])
        self.time_cursor.GetMapper().SetInputData(self.overlay_lines_data(cursor))

    def close_graph(self):
        self._graph_expanded = False
        self._scroll = 0.0
        self._now_mode = True
        self._sky_generation += 1  # discard any response still in flight
        self._sky_data = None
        self._sky_status = ""
        self.progress_text.SetVisibility(False)
        self._epochs = [utc_now()]
        self._time_index = 0
        self.apply_layout()
        self.update_profile(self.horizon_angles)
        self.update_stats()
        self.window.Render()

    def update_profile(self, angles):
        low, high = float(np.min(angles)), float(np.max(angles))
        if self._sky_data is not None:
            for name, body in self._sky_data.items():
                if not self._shown_sources[name.rsplit(" ", 1)[-1]]:
                    continue
                low = min(low, float(np.min(body["el"] - body["radius"])))
                high = max(high, float(np.max(body["el"] + body["radius"])))
        low = min(low, float(np.min(angles)) - 5.0)
        padding = max((high - low) * 0.08, 0.01)
        self._profile_low, self._profile_high = low - padding, high + padding
        self.update_panorama()
        values = np.append(angles, angles[0])
        x0, x1, y0, y1 = self._plot_box
        x = np.linspace(x0, x1, len(values))
        y = self.profile_y(values)
        self.profile.GetMapper().SetInputData(self.overlay_lines_data([np.column_stack((x, y))]))
        self.profile_title.SetInput(
            f"360-degree horizon: selected observer\n"
            f"Elevation scale {self._profile_low:+.2f} to {self._profile_high:+.2f} deg"
            + ("".join("\n" + SKY_LEGEND[source] for source in SKY_METHODS[self._sky_method] if self._shown_sources[source])
               if self._graph_expanded else ""))
        for name in SKY_COLORS:
            body = self._sky_data.get(name) if self._sky_data is not None and self._graph_expanded and self._shown_sources[name.rsplit(" ", 1)[-1]] else None
            segments, segment, coordinates = [], [], []
            previous_az = None
            if body is not None:
                for az, el, visible in zip(body["az"], body["el"], body["visible"]):
                    if previous_az is not None and abs(az - previous_az) > 180:
                        if len(segment) > 1:
                            segments.append(segment)
                        segment = []
                    point = (x0 + (x1 - x0) * az / 360.0, float(self.profile_y(el)))
                    coordinates.append(point)
                    segment.append(point)
                    previous_az = az
                if len(segment) > 1:
                    segments.append(segment)
            self.sky_trails[name].GetMapper().SetInputData(self.overlay_lines_data(segments))
            self.sky_samples[name].GetMapper().SetInputData(self.overlay_point_data(coordinates))
            marker = []
            if body is not None:
                az, el = body["az"][self._time_index], body["el"][self._time_index]
                point = (x0 + (x1 - x0) * az / 360.0, float(self.profile_y(el)))
                marker = [point]
                label = name + (" (partial)" if body["state"][self._time_index] == "Partly visible" else "")
                self.sky_labels[name].SetInput(label)
                self.sky_labels[name].SetPosition(min(point[0] + 0.008, x1 - 0.09),
                                                   float(np.clip(point[1] + (0.01 if name == "Sun" else -0.018), y0 + 0.01, y1 + 0.005)))
            self.sky_markers[name].GetMapper().SetInputData(self.overlay_point_data(marker))
            self.sky_labels[name].SetVisibility(bool(marker))
        self.update_time_caption()
        self.update_panels()

    def profile_y(self, elevation):
        _, _, y0, y1 = self._plot_box
        return y0 + (y1 - y0) * (np.asarray(elevation) - self._profile_low) / (self._profile_high - self._profile_low)

    def update_time_caption(self):
        epoch = self._epochs[self._time_index]
        mode = ("Now (refreshes every minute)" if self._now_mode else
                f"Sample {self._time_index + 1}/{len(self._epochs)} | trails = all samples")
        self.time_caption.SetInput(f"UTC {epoch:%Y-%m-%d %H:%M:%S}\n{mode}")
        for actor, enabled in ((self.previous_button, self._time_index > 0),
                               (self.next_button, self._time_index < len(self._epochs) - 1)):
            actor.GetTextProperty().SetColor(*((0.9, 0.95, 1.0) if enabled else (0.4, 0.45, 0.5)))

    def update_stats(self):
        if not self._graph_expanded:
            self.stats.SetInput(self._stats_base)
            return
        lines = [self._stats_base, f"UTC {self._epochs[self._time_index]:%Y-%m-%d %H:%M:%S}"]
        if self._sky_data is None:
            lines.append(self._sky_status)
        else:
            for name in SKY_COLORS:
                if name not in self._sky_data or not self._shown_sources[name.rsplit(" ", 1)[-1]]:
                    continue
                body, index = self._sky_data[name], self._time_index
                lines.append(f"{name}: {body['state'][index]}  |  AZ {body['az'][index]:.2f}, EL {body['el'][index]:+.2f} deg")
            if len(self._epochs) > 1:
                counts = [f"{name} {np.count_nonzero(body['visible'])}/{len(self._epochs)}" for name, body in self._sky_data.items() if self._shown_sources[name.rsplit(" ", 1)[-1]]]
                lines.append("Visible samples: " + "  |  ".join(counts))
            if self._sky_status:
                lines.append(self._sky_status)
            lines.append("Center visible = visible; partial is approximate (4 limb points).")
        self.stats.SetInput("\n".join(lines))

    def request_sky_update(self):
        if self._now_mode:
            self._epochs = [utc_now()]
            self._time_index = 0
            self._last_now_query = time.monotonic()
        self._sky_generation += 1
        self._sky_data = None  # Never show a previous location's bodies at a new point.
        self._sky_status = "Loading Sun / Earth positions..." if self._enable_ephemeris else "Sun / Earth lookup disabled"
        active = self._enable_ephemeris and self._graph_expanded
        self.progress_text.SetInput("Working: starting calculation...")
        self.progress_text.SetVisibility(active)
        self.update_profile(self.horizon_angles)
        self.update_stats()
        if not active:
            return
        self.window.Render()
        request = (self._sky_generation, self.lon, self.lat,
                   (float(sample_dem_bilinear(self.lat, self.lon)) + 2.0) / 1000.0,
                   list(self._epochs), self.horizon_angles.copy(), SKY_METHODS[self._sky_method])
        # Coalesce fast clicks: keep only the newest waiting location/interval.
        try:
            self._sky_jobs.get_nowait()
        except queue.Empty:
            pass
        self._sky_jobs.put_nowait(request)
        if self._sky_thread is None:
            self._sky_thread = threading.Thread(target=self.sky_worker, daemon=True)
            self._sky_thread.start()

    @staticmethod
    def fetch_jpl(lon, lat, altitude, epochs, progress=None):
        """JPL Horizons through get_angle.py (azimuth and elevation only)."""
        try:
            from get_angle import get_angle
        except ImportError as error:
            raise RuntimeError(f"Install astroquery and keep get_angle.py beside moon.py: {error}") from error
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

    def fetch_skyfield(self, lon, lat, altitude, epochs, progress=None):
        if self.skyfield_preload_status()[0] != "ready":
            self.start_skyfield_preload()
            if progress is not None:
                progress("Preparing Skyfield data")
            self._preload_done.wait()
            status, detail = self.skyfield_preload_status()
            if status != "ready":
                raise RuntimeError(f"Skyfield data unavailable: {detail}")
        try:
            from skyfield_positions import get_positions
        except ImportError as error:
            raise RuntimeError(f"Install skyfield and keep skyfield_positions.py beside moon.py: {error}") from error
        return get_positions(lon, lat, altitude, epochs, progress=progress)

    def toggle_fullscreen(self):
        # Avoid SetFullScreen, which recreates the native VTK window on some drivers.
        if getattr(self, "_windowed_geometry", None) is None:
            self._windowed_geometry = (tuple(self.window.GetPosition()), tuple(self.window.GetSize()))
            width, height = self.window.GetScreenSize()
            self.window.SetPosition(0, 0)
            self.window.SetSize(width, height)
        else:
            position, size = self._windowed_geometry
            self._windowed_geometry = None
            self.window.SetPosition(*position)
            self.window.SetSize(*size)
        self.constrain_camera()
        self.window.Render()

    def sky_worker(self):
        while not self._sky_stop.is_set():
            try:
                request = self._sky_jobs.get(timeout=0.2)
            except queue.Empty:
                continue
            generation, lon, lat, altitude, epochs, horizon, methods = request
            data, errors = {}, []
            fetchers = {"JPL": self.fetch_jpl, "Skyfield": self.fetch_skyfield}
            for source in methods:
                fetch = self._ephemeris_fetcher or fetchers[source]
                try:
                    progress = lambda message: self._sky_progress.put((generation, message))
                    progress(f"Starting {source} lookup")
                    if self._ephemeris_fetcher is not None:
                        positions = fetch(lon, lat, altitude, epochs)
                    else:
                        positions = fetch(lon, lat, altitude, epochs, progress=progress)
                    progress("Checking terrain visibility and eclipses")
                    classified = classify_sky(positions, horizon)
                    data.update({f"{body} {source}": values for body, values in classified.items()})
                except Exception as error:
                    message = str(error).strip().splitlines()
                    errors.append(f"{source}: " + (message[0] if message else type(error).__name__)[:100])
            self._sky_results.put((generation, data or None, "; ".join(errors) or None))

    def poll_sky(self, obj=None, event=None):
        changed = False
        finished = False
        progress_changed = False
        while True:
            try:
                generation, message = self._sky_progress.get_nowait()
            except queue.Empty:
                break
            if generation == self._sky_generation and self._graph_expanded:
                self.progress_text.SetInput("Working: " + message + "...")
                self.progress_text.SetVisibility(True)
                progress_changed = True
        while True:
            try:
                generation, data, error = self._sky_results.get_nowait()
            except queue.Empty:
                break
            if generation != self._sky_generation:
                continue  # A newer click or time edit made this response obsolete.
            self.progress_text.SetInput("Working: building graphs...")
            self.progress_text.SetVisibility(True)
            self.window.Render()
            self._sky_data = data
            self._sky_status = f"Unavailable: {error}" if error else ""
            changed = True
            finished = True
        if self._now_mode and self._graph_expanded and time.monotonic() - self._last_now_query >= 60:
            self.request_sky_update()
            changed = True
        if changed:
            self.update_profile(self.horizon_angles)
            self.update_stats()
            if finished:
                self.progress_text.SetVisibility(False)
            self.window.Render()
        elif progress_changed:
            self.window.Render()

    def set_time_interval(self, start, stop, step, unit="minutes", method=None):
        if unit not in TIME_UNITS:
            raise ValueError("Choose a valid time unit.")
        method = self._sky_method if method is None else method
        if method not in SKY_METHODS:
            raise ValueError("Choose Skyfield or JPL Horizons.")
        epochs = time_samples(start, stop, float(step) * TIME_UNITS[unit])
        self._sky_method = method
        self._saved_interval = (parse_utc(start), parse_utc(stop), float(step), unit, method)
        save_interval(start, stop, step, unit, method)
        self._graph_expanded = True
        self.apply_layout()
        self._now_mode = False
        self._epochs = epochs
        self._time_index = 0
        self._interval_step_minutes = float(step) * TIME_UNITS[unit]
        self.request_sky_update()

    def toggle_source(self, source):
        self._shown_sources[source] = not self._shown_sources[source]
        self.apply_layout()
        self.update_profile(self.horizon_angles)
        self.update_stats()
        self.window.Render()

    def toggle_legend(self):
        self._legend_expanded = not self._legend_expanded
        self.apply_layout()
        self.window.Render()

    def set_now(self):
        self._now_mode = True
        self._time_index = 0
        self.request_sky_update()
        self.window.Render()

    def step_time(self, delta):
        self._time_index = int(np.clip(self._time_index + delta, 0, len(self._epochs) - 1))
        self.update_profile(self.horizon_angles)
        self.update_stats()
        self.window.Render()

    def over_sky_controls(self, x, y):
        if self._graph_expanded:
            return True  # the graph covers the whole window
        width, height = self.window.GetSize()
        return (x >= width * 0.59 and y <= height * 0.49) or (0.02 <= x / width <= 0.30 and 0.95 <= y / height <= 0.995)

    def begin_coordinate_entry(self, render=True):
        if self.coordinate_editing:
            return
        try:
            import tkinter as tk
        except ImportError:
            print("Coordinate text input needs Tkinter. Install your Python Tk package "
                  "(Debian/Ubuntu: sudo apt install python3-tk).", file=sys.stderr)
            return
        dialog = None
        try:
            self.get_native_root()
            self.coordinate_editing = True
            dialog = CoordinateDialog(
                self._coordinate_root, f"{self.lat:.6f}, {self.lon:.6f}",
                lambda text: self.lookup_coordinates(text, render=False))
            self._coordinate_root.wait_window(dialog.window)
        except tk.TclError as error:
            print(f"Could not open coordinate editor: {error}", file=sys.stderr)
        finally:
            if dialog is not None:
                try:
                    dialog.window.destroy()
                except tk.TclError:
                    pass
            self.coordinate_editing = False
            if render:
                self.window.Render()

    def get_native_root(self):
        if self._coordinate_root is None:
            import tkinter as tk
            self._coordinate_root = tk.Tk(className="MoonHorizon")
            self._coordinate_root.withdraw()
            # Keep the native clipboard owner alive after a dialog is closed.
            self._coordinate_pump_observer = self.interactor.AddObserver(
                "TimerEvent", self.pump_coordinate_clipboard)
            self._coordinate_timer = self.interactor.CreateRepeatingTimer(50)
        return self._coordinate_root

    def begin_time_entry(self):
        if self.coordinate_editing:
            return
        try:
            import tkinter as tk
        except ImportError:
            print("Time interval editing needs Tkinter (Debian/Ubuntu: python3-tk).", file=sys.stderr)
            return
        dialog = None
        try:
            root = self.get_native_root()
            self.coordinate_editing = True
            if self._saved_interval is not None:
                start, stop, step, unit, _ = self._saved_interval
            else:
                start = utc_now()
                stop = start + timedelta(days=1)
                step, unit = 60, "minutes"
            dialog = TimeIntervalDialog(root, start, stop, step, unit, self.set_time_interval, self._sky_method)
            root.wait_window(dialog.window)
        except tk.TclError as error:
            print(f"Could not open time interval editor: {error}", file=sys.stderr)
        finally:
            if dialog is not None:
                try:
                    dialog.window.destroy()
                except tk.TclError:
                    pass
            self.coordinate_editing = False
            self.window.Render()

    def pump_coordinate_clipboard(self, obj=None, event=None):
        if self._coordinate_root is not None and not self.coordinate_editing:
            self._coordinate_root.update()

    def lookup_coordinates(self, text, render=True):
        lat, lon = parse_coordinates(text)
        distance = np.linalg.norm(self.renderer.GetActiveCamera().GetPosition())
        self.select_location(lat, lon, render=False)
        self.reset_camera(distance=distance)
        if render:
            self.window.Render()
        return lat, lon

    def configure_surface_overlay(self, mapper):
        # Depth bias separates coincident fragments without lifting geometry.
        # The ordinary depth test still hides marks on the Moon's far side.
        mapper.SetResolveCoincidentTopologyToPolygonOffset()
        mapper.SetRelativeCoincidentTopologyLineOffsetParameters(0.0, -4.0)
        mapper.SetRelativeCoincidentTopologyPointOffsetParameter(-4.0)

    def terrain_overlay_points(self, directions):
        """Project marks onto the same mesh used to render the Moon."""
        result = []
        for direction in np.atleast_2d(directions):
            direction = np.asarray(direction, dtype=float)
            direction = direction / np.linalg.norm(direction)
            hit = self.surface_hit(direction * (self.surface_bound + 1.0), -direction)
            if hit is None:
                hit, _ = self.closest_surface_point(direction)
            result.append(hit)
        return np.asarray(result)

    def select_location(self, lat, lon, render=True):
        from vtk.util.numpy_support import numpy_to_vtk

        self.lat, self.lon = float(lat), float(lon)
        angles, distances, hlat, hlon = compute_horizon_360(lat, lon)
        self.horizon_angles = angles
        self.horizon_distances = distances
        if self._panorama_texture_rgb is None:
            with Image.open(COLOR_PATH) as texture_image:
                texture_image = texture_image.convert("RGB")
                texture_image.thumbnail((2048, 1024), Image.Resampling.LANCZOS)
                self._panorama_texture_rgb = np.asarray(texture_image)
        self._panorama_samples = terrain_panorama_samples(lat, lon, self._panorama_texture_rgb)
        self._panorama_range = None
        self.update_profile(angles)
        # Follow the terrain between bearings rather than drawing straight chords
        # that can pass below the surface when adjacent horizon distances differ.
        lat_r, lon_r = np.radians(hlat), np.radians(hlon)
        unit = np.column_stack((np.cos(lat_r) * np.cos(lon_r),
                                np.cos(lat_r) * np.sin(lon_r), np.sin(lat_r)))
        fraction = np.arange(8)[None, :, None] / 8.0
        curve = ((1 - fraction) * unit[:, None, :]
                 + fraction * np.roll(unit, -1, axis=0)[:, None, :]).reshape(-1, 3)
        curve /= np.linalg.norm(curve, axis=1)[:, None]
        curve = np.vstack((curve, curve[0]))
        surface_curve = self.terrain_overlay_points(curve)
        points = self.vtk.vtkPoints()
        points.SetData(numpy_to_vtk(surface_curve, deep=True))
        line = self.vtk.vtkPolyLine()
        line.GetPointIds().SetNumberOfIds(len(surface_curve))
        for index in range(len(surface_curve)):
            line.GetPointIds().SetId(index, index)
        cells = self.vtk.vtkCellArray()
        cells.InsertNextCell(line)
        data = self.vtk.vtkPolyData()
        data.SetPoints(points)
        data.SetLines(cells)
        mapper = self.vtk.vtkPolyDataMapper()
        mapper.SetInputData(data)
        self.configure_surface_overlay(mapper)
        self.horizon.SetMapper(mapper)
        direction = np.array(lat_lon_to_xyz(lat, lon, offset=0.0))
        surface_point = self.terrain_overlay_points(direction)[0]
        self.observer_points.SetPoint(0, *surface_point)
        self.observer_points.Modified()
        elevation = float(sample_dem_bilinear(lat, lon))
        self._stats_base = (
            f"MOON HORIZON\n"
            f"Latitude {lat:+.3f} deg    Longitude {lon:+.3f} deg\n"
            f"Elevation {elevation:+,.0f} m    Slope {compute_slope_deg(lat, lon):.1f} deg\n"
            f"Horizon distance {np.min(distances):.1f} to {np.max(distances):.1f} km\n"
            f"Horizon angle {np.min(angles):+.2f} to {np.max(angles):+.2f} deg\n"
            "360 bearings; 2 m eye height; terrain search to 250 km\n"
            f"Texture: {self.texture_dimensions[0]} x {self.texture_dimensions[1]}")
        self.request_sky_update()
        if render:
            self.window.Render()

    def pick(self, x, y):
        width, height = self.window.GetSize()
        if self.coordinate_editing:
            return
        if self.over_sky_controls(x, y):
            nx, ny = x / width, y / height
            actions = {"time": self.begin_time_entry, "now": self.set_now, "close": self.close_graph,
                       "previous": lambda: self.step_time(-1), "next": lambda: self.step_time(1),
                       "Skyfield": lambda: self.toggle_source("Skyfield"),
                       "JPL": lambda: self.toggle_source("JPL"),
                       "legend": self.toggle_legend}
            for name, (left, bottom, right, top) in self._controls.items():
                if left <= nx <= right and bottom <= ny <= top:
                    actions[name]()
                    break
            return
        if self.picker.Pick(x, y, 0, self.renderer):
            px, py, pz = self.picker.GetPickPosition()
            radius = np.sqrt(px * px + py * py + pz * pz)
            if radius > 0:
                lat = np.degrees(np.arcsin(np.clip(pz / radius, -1, 1)))
                lon = np.degrees(np.arctan2(py, px))
                self.select_location(float(lat), float(lon))

    def reset_camera(self, distance=4.2):
        self.orbit_pivot = np.zeros(3)
        camera = self.renderer.GetActiveCamera()
        x, y, z = lat_lon_to_xyz(self.lat, self.lon, offset=0)
        direction = np.array([x, y, z], dtype=float)
        direction /= np.linalg.norm(direction)
        camera.SetFocalPoint(0, 0, 0)
        camera.SetPosition(*(direction * distance))
        camera.SetViewUp(0, 0, 1)
        # A stable up direction when the selected observer is at a pole.
        if abs(direction[2]) > 0.99:
            camera.SetViewUp(0, 1, 0)
        camera.SetViewAngle(35)
        self.constrain_camera()
        camera.SetFocalPoint(*self.view_surface_target())

    def surface_hit(self, origin, direction):
        """First intersection with the real terrain mesh along a forward ray."""
        origin = np.asarray(origin, dtype=float)
        direction = np.asarray(direction, dtype=float)
        length = float(np.linalg.norm(direction))
        if length < 1e-12:
            return None
        direction /= length
        end = origin + direction * (np.linalg.norm(origin) + 3 * self.surface_bound)
        t = self.vtk.reference(0.0)
        point, pcoords = [0.0] * 3, [0.0] * 3
        sub_id, cell_id = self.vtk.reference(0), self.vtk.reference(0)
        found = self.surface_locator.IntersectWithLine(
            origin, end, 1e-10, t, point, pcoords, sub_id, cell_id, self.vtk.vtkGenericCell())
        return np.array(point) if found else None

    def closest_surface_point(self, position):
        point = [0.0] * 3
        cell_id, sub_id = self.vtk.reference(0), self.vtk.reference(0)
        distance_squared = self.vtk.reference(0.0)
        self.surface_locator.FindClosestPoint(
            position, point, self.vtk.vtkGenericCell(), cell_id, sub_id, distance_squared)
        return np.array(point), np.sqrt(max(float(distance_squared), 0.0))

    def view_surface_target(self):
        camera = self.renderer.GetActiveCamera()
        position = np.array(camera.GetPosition())
        target = self.surface_hit(position, np.array(camera.GetFocalPoint()) - position)
        if target is None:
            target, _ = self.closest_surface_point(position)
        return target

    def constrain_camera(self):
        camera = self.renderer.GetActiveCamera()
        position = np.array(camera.GetPosition())
        radius = float(np.linalg.norm(position))
        radial = position / radius if radius > 1e-12 else np.array([1.0, 0.0, 0.0])
        # Recover from an orbit that crosses a mountain or moves below terrain.
        # The radial intersection is with this location's mesh, not a global bound.
        exterior = radial * (self.surface_bound + 1.0)
        surface = self.surface_hit(exterior, -radial)
        if surface is not None:
            local_radius = float(np.dot(surface, radial))
            radius = max(radius, local_radius + self.minimum_surface_clearance)
        radius = min(radius, self.maximum_camera_radius)
        position = radial * radius
        # On sloping terrain, radial height can overstate the nearest clearance.
        for _ in range(3):
            closest, clearance = self.closest_surface_point(position)
            if clearance >= self.minimum_surface_clearance * (1 - 1e-8):
                break
            normal = (position - closest) / clearance if clearance > 1e-12 else radial
            position = closest + normal * self.minimum_surface_clearance * 1.001
        camera.SetPosition(*position)
        camera.OrthogonalizeViewUp()
        _, clearance = self.closest_surface_point(position)
        radius = float(np.linalg.norm(position))
        camera.SetClippingRange(max(1e-8, clearance * 0.05), radius + self.surface_bound * 1.5)

    def zoom(self, factor, render=True):
        if not np.isfinite(factor) or factor <= 0:
            raise ValueError("Zoom factor must be positive and finite.")
        camera = self.renderer.GetActiveCamera()
        self.constrain_camera()
        position = np.array(camera.GetPosition())
        target = self.surface_hit(position, camera.GetDirectionOfProjection())
        if target is None:
            return  # Looking into the sky must not snap the view back to the Moon.
        offset = position - target
        surface_distance = float(np.linalg.norm(offset))
        # Scale the camera-to-hit distance. The actual surface point is the
        # fixed zoom anchor, including for off-centre and oblique views.
        new_distance = max(self.minimum_surface_clearance, surface_distance * factor)
        if surface_distance > 1e-12:
            camera.SetPosition(*(target + offset / surface_distance * new_distance))
            camera.SetFocalPoint(*target)
        self.constrain_camera()
        if render:
            self.window.Render()

    def orbit(self, dx, dy, render=True):
        camera = self.renderer.GetActiveCamera()
        width, height = self.window.GetSize()
        position = np.array(camera.GetPosition())
        target = self.view_surface_target()
        surface_distance = max(float(np.linalg.norm(position - target)), 1e-8)
        tangent = np.tan(np.radians(camera.GetViewAngle()) / 2)
        speed = min(180.0 / max(width, height, 1),
                    np.degrees(2 * tangent * surface_distance / (max(height, 1) * self.surface_bound)))
        # Keep the navigation pivot independent of the surface zoom anchor.
        # Replacing the orbit pivot with a new ray hit on every move makes the
        # Moon tumble around whichever small surface patch is currently visible.
        focal = np.array(camera.GetFocalPoint())
        up = np.array(camera.GetViewUp())
        forward = np.array(camera.GetDirectionOfProjection())
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)

        def rotate(vector, axis, degrees):
            axis = axis / np.linalg.norm(axis)
            angle = np.radians(degrees)
            return (vector * np.cos(angle) + np.cross(axis, vector) * np.sin(angle)
                    + axis * np.dot(axis, vector) * (1 - np.cos(angle)))

        yaw_axis = up.copy()
        pitch_axis = rotate(right, yaw_axis, -dx * speed)

        def turn(vector):
            return rotate(rotate(vector, yaw_axis, -dx * speed), pitch_axis, dy * speed)

        camera.SetPosition(*(self.orbit_pivot + turn(position - self.orbit_pivot)))
        camera.SetFocalPoint(*(self.orbit_pivot + turn(focal - self.orbit_pivot)))
        camera.SetViewUp(*turn(up))
        self.constrain_camera()
        if render:
            self.window.Render()

    def cursor_surface_target(self, x, y):
        """Pick a terrain pivot without changing the selected observer."""
        if self.picker.Pick(x, y, 0, self.renderer):
            return np.asarray(self.picker.GetPickPosition(), dtype=float)
        return np.asarray(self.view_surface_target(), dtype=float)

    def orbit_cursor(self, dx, dy, pivot, render=True):
        """Orbit around the pressed terrain point; drag follows the mouse."""
        camera = self.renderer.GetActiveCamera()
        pivot = np.asarray(pivot, dtype=float)
        height = max(self.window.GetSize()[1], 1)
        speed = np.degrees(2 * np.tan(np.radians(camera.GetViewAngle()) / 2) / height)
        position = np.asarray(camera.GetPosition(), dtype=float)
        focus = np.asarray(camera.GetFocalPoint(), dtype=float)
        up = np.asarray(camera.GetViewUp(), dtype=float)
        forward = np.asarray(camera.GetDirectionOfProjection(), dtype=float)
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)

        def rotate(vector, axis, degrees):
            axis = axis / np.linalg.norm(axis)
            angle = np.radians(degrees)
            return (vector * np.cos(angle) + np.cross(axis, vector) * np.sin(angle)
                    + axis * np.dot(axis, vector) * (1 - np.cos(angle)))

        # Drag right moves the surrounding surface right; drag up moves it up.
        yaw = -dx * speed
        pitch = dy * speed
        yawed_right = rotate(right, up, yaw)
        def turn(vector):
            return rotate(rotate(vector, up, yaw), yawed_right, pitch)
        camera.SetPosition(*(pivot + turn(position - pivot)))
        camera.SetFocalPoint(*(pivot + turn(focus - pivot)))
        camera.SetViewUp(*turn(up))
        self.constrain_camera()
        if render:
            self.window.Render()

    def pan(self, dx, dy, depth=None, render=True):
        """Translate camera and focus together in the screen plane."""
        camera = self.renderer.GetActiveCamera()
        position = np.array(camera.GetPosition())
        focal = np.array(camera.GetFocalPoint())
        forward = np.array(camera.GetDirectionOfProjection())
        up = np.array(camera.GetViewUp())
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        if depth is None:
            depth = float(np.linalg.norm(position - self.view_surface_target()))
        height = max(self.window.GetSize()[1], 1)
        units_per_pixel = 2 * max(depth, self.minimum_surface_clearance) * np.tan(
            np.radians(camera.GetViewAngle()) / 2) / height
        motion = -(right * dx + up * dy) * units_per_pixel
        camera.SetPosition(*(position + motion))
        camera.SetFocalPoint(*(focal + motion))
        self.orbit_pivot += motion
        self.constrain_camera()
        # If terrain collision corrects the position, apply the same correction
        # to the focus and pivot so a pan never introduces an unintended tilt.
        correction = np.array(camera.GetPosition()) - (position + motion)
        camera.SetFocalPoint(*(focal + motion + correction))
        self.orbit_pivot += correction
        if render:
            self.window.Render()

    def run(self):
        self.interactor.Initialize()
        self._sky_pump_observer = self.interactor.AddObserver("TimerEvent", self.poll_sky)
        self._sky_timer = self.interactor.CreateRepeatingTimer(100)
        self.poll_sky()
        self.window.Render()
        try:
            self.interactor.Start()
        finally:
            self._sky_stop.set()
            if self._sky_timer is not None:
                self.interactor.DestroyTimer(self._sky_timer)
            if self._sky_pump_observer is not None:
                self.interactor.RemoveObserver(self._sky_pump_observer)
            if self._coordinate_timer is not None:
                self.interactor.DestroyTimer(self._coordinate_timer)
            if self._coordinate_pump_observer is not None:
                self.interactor.RemoveObserver(self._coordinate_pump_observer)
            if self._coordinate_root is not None:
                self._coordinate_root.destroy()
            self.window.Finalize()


def main():
    parser = argparse.ArgumentParser(description="Native Moon terrain and horizon viewer")
    parser.add_argument("--texture", help="Optional path to a Moon color texture")
    arguments = parser.parse_args()
    try:
        load_data(arguments.texture)
        MoonWindow().run()
    except ImportError as error:
        print(f"Missing desktop dependency: {error}\n"
              "Install with: python -m pip install numpy Pillow vtk", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"Moon Horizon could not start: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
