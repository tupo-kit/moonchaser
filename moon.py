"""Moon Horizon: a native desktop viewer, with no browser or web server.

Install: python -m pip install numpy Pillow vtk astroquery skyfield
Coordinate editor: requires Tkinter (Linux package: python3-tk).
Run:     python moon.py
Texture: python moon.py --texture /path/to/moon_color_8k.jpg

Use your local 8K texture; the NASA elevation map downloads on first run.
Left-drag to orbit; middle-drag or Shift-left-drag to pan; wheel to zoom.
Ctrl-left-drag (or Ctrl-middle-drag) looks around from the current camera position.
Click the Moon to select an observer.
Zoom stops above the terrain. The inset shows all 360 horizon bearings.
Put moon_color_8k.jpg beside this script to use it automatically.
Press G to enter latitude, longitude; R to reset; F for fullscreen; Escape to close.
Sun/Earth positions use JPL Horizons over the internet. Time defaults to UTC now.
Click Edit time interval (or T) to choose a range and a method (JPL Horizons, Skyfield or both); this opens a fullscreen horizon + Sun/Earth graph.
Use Previous/Next to step samples and Close to return to the horizon-only view.
"""

import argparse
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
COLOR_PATH = os.path.join(BASE_DIR, "moon_color_8k.jpg")
DEM_16_PATH = os.path.join(BASE_DIR, "moon_dem_16.tif")
DEM_4_PATH = os.path.join(BASE_DIR, "moon_dem.tif")
DEM_4_URL = "https://svs.gsfc.nasa.gov/vis/a000000/a004700/a004720/ldem_4.tif"
MOON_RADIUS_M = 1_737_400.0
RELIEF_EXAGGERATION = 3.0

# Initialized by load_data(), so importing this module never downloads files.
dem_km = None
H = W = 0
R_MAX_M = MOON_RADIUS_M
MAX_TIME_SAMPLES = 1000
SKY_METHODS = {"JPL Horizons": ("JPL",), "Skyfield": ("Skyfield",), "Both": ("JPL", "Skyfield")}
SKY_LEGEND = {"JPL": "JPL Sun yellow / Earth blue", "Skyfield": "Skyfield Sun orange / Earth green"}
SKY_COLORS = {"Sun JPL": (1.0, 0.8, 0.12), "Earth JPL": (0.35, 0.65, 1.0), "Sun Skyfield": (1.0, 0.35, 0.1), "Earth Skyfield": (0.3, 1.0, 0.5)}


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


def query_sky_positions(lon, lat, altitude_km, epochs):
    """Apparent airless positions at the actual lunar observer, in UTC."""
    try:
        from astroquery.jplhorizons import Horizons
        from astropy.time import Time
    except ImportError:
        raise RuntimeError("Install astroquery: python -m pip install astroquery") from None
    julian_dates = np.atleast_1d(Time(epochs, scale="utc").jd).tolist()
    location = {"lon": float(lon), "lat": float(lat),
                "elevation": float(altitude_km), "body": 301}
    result = {}
    for name, body_id in (("Sun", 10), ("Earth", 399)):
        chunks = {"az": [], "el": [], "radius": []}
        # Keep individual request URLs small even for long intervals.
        for first in range(0, len(julian_dates), 100):
            obj = Horizons(id=body_id, id_type=None, location=location,
                           epochs=julian_dates[first:first + 100])
            obj.TIMEOUT = 30
            eph = obj.ephemerides(quantities="4,13", refraction=False)
            for key, column in (("az", "AZ"), ("el", "EL"), ("radius", "ang_width")):
                values = np.asarray(np.ma.filled(np.ma.asarray(eph[column], dtype=float), np.nan))
                if not np.isfinite(values).all():
                    raise RuntimeError(f"JPL returned unavailable {name} coordinates.")
                chunks[key].extend(values.tolist())
        result[name] = {key: np.asarray(values, dtype=float) for key, values in chunks.items()}
        result[name]["az"] %= 360.0
        result[name]["radius"] /= 7200.0  # angular diameter in arcsec -> radius in degrees
        if any(len(values) != len(epochs) for values in result[name].values()):
            raise RuntimeError(f"JPL returned an incomplete {name} time series.")
    return result


def sky_unit(azimuth, elevation):
    azimuth, elevation = np.radians(azimuth), np.radians(elevation)
    return np.stack((np.cos(elevation) * np.sin(azimuth),
                     np.cos(elevation) * np.cos(azimuth), np.sin(elevation)), axis=-1)


def classify_sky(positions, horizon):
    """Sample body disks against terrain; also test Earth occulting the Sun.

    Partial visibility is approximate at the DEM/profile sampling resolution.
    These geometric tests do not model the illuminated phase of Earth's disk.
    """
    result = {}
    earth_center = sky_unit(positions["Earth"]["az"], positions["Earth"]["el"])
    earth_radius = np.radians(positions["Earth"]["radius"])
    # Centre plus two rings: check interior and limb, including narrow crescents.
    theta = np.tile(np.linspace(0, 2 * np.pi, 48, endpoint=False), 2)
    fractions = np.repeat([0.5, 1.0], 48)
    theta, fractions = np.r_[0, theta], np.r_[0, fractions]
    for name in ("Sun", "Earth"):
        body = {key: np.asarray(value).copy() for key, value in positions[name].items()}
        az, el = np.radians(body["az"]), np.radians(body["el"])
        centre = sky_unit(body["az"], body["el"])
        horizontal = np.stack((np.cos(az), -np.sin(az), np.zeros_like(az)), axis=-1)
        vertical = np.stack((-np.sin(el) * np.sin(az), -np.sin(el) * np.cos(az), np.cos(el)), axis=-1)
        rho = np.radians(body["radius"])[:, None] * fractions[None, :]
        rays = (np.cos(rho)[..., None] * centre[:, None, :]
                + np.sin(rho)[..., None] * (
                    np.cos(theta)[None, :, None] * vertical[:, None, :]
                    + np.sin(theta)[None, :, None] * horizontal[:, None, :]))
        ray_az = np.degrees(np.arctan2(rays[..., 0], rays[..., 1])) % 360.0
        ray_el = np.degrees(np.arctan2(rays[..., 2], np.hypot(rays[..., 0], rays[..., 1])))
        terrain = np.interp(ray_az, np.arange(360), horizon, period=360)
        terrain_clear = ray_el > terrain
        clear = terrain_clear.copy()
        if name == "Sun":
            separation = np.arccos(np.clip(np.sum(rays * earth_center[:, None, :], axis=-1), -1, 1))
            clear &= separation > earth_radius[:, None]
        visible = clear.any(axis=1)
        full = clear.all(axis=1)
        state = np.full(len(visible), "Hidden by terrain", dtype=object)
        if name == "Sun":
            state[terrain_clear.any(axis=1) & ~visible] = "Eclipsed by Earth"
        state[visible & ~full] = "Partly visible"
        state[full] = "Visible"
        body.update(visible=visible, state=state,
                    terrain=np.interp(body["az"], np.arange(360), horizon, period=360))
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


def load_dem(path):
    """Return DEM as float32 array (H, 2H) in km, or None if the file is unusable.
    Reads a real TIFF through PIL; falls back to headerless raw float32."""
    try:
        with Image.open(path) as im:
            arr = np.array(im, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[1] == 2 * arr.shape[0]:
            return arr
    except Exception:
        pass
    size = os.path.getsize(path)
    for h, w in ((2880, 5760), (720, 1440)):
        n = h * w * 4
        if n <= size < n + 100_000:  # tolerate a small header
            return np.fromfile(path, dtype="<f4", count=h * w, offset=size - n).reshape((h, w))
    return None


def get_dem():
    for p in (DEM_16_PATH, DEM_4_PATH):
        if os.path.exists(p):
            d = load_dem(p)
            if d is not None:
                return d
            os.remove(p)  # corrupt/partial file -> re-download
    download(DEM_4_URL, DEM_4_PATH)
    d = load_dem(DEM_4_PATH)
    if d is None:
        raise RuntimeError("Could not read the downloaded DEM file.")
    return d


def choose_texture(texture_path=None):
    """Prefer the user's real 8K image; never upscale a 2K image."""
    if texture_path:
        path = os.path.abspath(os.path.expanduser(texture_path))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Texture file does not exist: {path}")
        return path
    for stem in ("moon_color_8k", "lroc_color_8k"):
        for extension in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
            path = os.path.join(BASE_DIR, stem + extension)
            if os.path.isfile(path):
                return path
    return os.path.join(BASE_DIR, "moon_color_8k.jpg")


def load_data(texture_path=None):
    global dem_km, H, W, R_MAX_M, COLOR_PATH
    COLOR_PATH = choose_texture(texture_path)
    if not os.path.exists(COLOR_PATH):
        raise FileNotFoundError(
            "Place moon_color_8k.jpg beside moon.py, or pass --texture /path/to/your/image.jpg")
    with Image.open(COLOR_PATH) as image:
        print(f"Texture: {os.path.basename(COLOR_PATH)} ({image.width} x {image.height})", flush=True)
        image.verify()
    print("Loading Moon elevation map...", flush=True)
    dem_km = get_dem()
    if not np.all(np.isfinite(dem_km)):
        raise RuntimeError("The elevation map contains invalid values.")
    if float(np.max(np.abs(dem_km))) > 100.0:
        dem_km = dem_km / 1000.0
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
        self.error = tk.StringVar(master=self.window)
        ttk.Label(frame, textvariable=self.error, foreground="#c02020", wraplength=430).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(8, 4))
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
            self.error.set(str(error))
            self.text.focus_set()
        else:
            self.window.destroy()
        return "break"


class TimeIntervalDialog:
    def __init__(self, parent, start, stop, step_minutes, on_submit, method="Both"):
        import tkinter as tk
        from tkinter import ttk

        self.window = tk.Toplevel(parent)
        self.window.title("Sun / Earth time interval (UTC)")
        self.window.resizable(False, False)
        frame = ttk.Frame(self.window, padding=14)
        frame.grid()
        self.fields = []
        values = (start.strftime("%Y-%m-%d %H:%M:%S"),
                  stop.strftime("%Y-%m-%d %H:%M:%S"), str(step_minutes))
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
        self.unit.set("minutes")
        self.unit.grid(row=2, column=2, padx=(8, 0))
        ttk.Label(frame, text="Method").grid(row=3, column=0, sticky="w", padx=(0, 12), pady=4)
        self.method = ttk.Combobox(frame, values=tuple(SKY_METHODS), state="readonly", width=26)
        self.method.set(method if method in SKY_METHODS else "Both")
        self.method.grid(row=3, column=1, sticky="ew", pady=4)
        ttk.Label(frame, text=f"UTC dates: YYYY-MM-DD HH:MM:SS. Up to {MAX_TIME_SAMPLES} samples.").grid(
            row=4, column=0, columnspan=2, sticky="w", pady=6)
        self.error = tk.StringVar(master=self.window)
        ttk.Label(frame, textvariable=self.error, foreground="#c02020", wraplength=430).grid(
            row=5, column=0, columnspan=2, sticky="w")

        def apply(event=None):
            try:
                on_submit(self.fields[0].get(), self.fields[1].get(),
                          float(self.fields[2].get()) * {"seconds": 1/60, "minutes": 1, "hours": 60, "days": 1440}[self.unit.get()],
                          self.method.get())
            except ValueError as error:
                self.error.set(str(error))
            else:
                self.window.destroy()
            return "break"

        ttk.Button(frame, text="Cancel", command=self.window.destroy).grid(row=6, column=0, pady=(10, 0))
        ttk.Button(frame, text="Apply interval", command=apply).grid(row=6, column=1, sticky="e", pady=(10, 0))
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
        self._ephemeris_fetcher = ephemeris_fetcher or query_sky_positions
        self._sky_jobs = queue.Queue(maxsize=1)
        self._sky_results = queue.Queue()
        self._sky_stop = threading.Event()
        self._sky_thread = None
        self._sky_timer = self._sky_pump_observer = None
        self._sky_generation = 0
        self._sky_data = None
        self._sky_status = "Waiting for Sun / Earth positions"
        self._now_mode = True
        self._sky_method = "Both"
        self._graph_expanded = False  # False: horizon only. True: fullscreen horizon + Sun/Earth graph.
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
                           "Ctrl-drag: look around   |   Wheel: zoom   |   Click: select\n"
                           "G: coordinates   |   T: time interval   |   R: reset\n"
                           "F: fullscreen   |   Esc: close\n"
                           "Cyan: local horizon   |   Orange: observer   |   Relief: 3x")
        self.profile_background = self.make_profile_background()
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
                self.AddObserver("LeftButtonPressEvent", self.left_down)
                self.AddObserver("LeftButtonReleaseEvent", self.left_up)
                self.AddObserver("MiddleButtonPressEvent", self.middle_down)
                self.AddObserver("MiddleButtonReleaseEvent", self.middle_up)
                self.AddObserver("MouseMoveEvent", self.mouse_move)
                self.AddObserver("MouseWheelForwardEvent", lambda obj, event: owner.zoom(1 / 1.08) if not owner._graph_expanded else None)
                self.AddObserver("MouseWheelBackwardEvent", lambda obj, event: owner.zoom(1.08) if not owner._graph_expanded else None)
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
                        owner.look_around(dx, dy)
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

            def key_press(self, obj, event):
                interactor = self.GetInteractor()
                key = (interactor.GetKeySym() or "").lower()
                if owner.coordinate_editing:
                    return  # The native editor handles its own keyboard events.
                elif key == "escape":
                    owner.interactor.TerminateApp()

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
        if expanded:
            self._plot_box = (0.06, 0.97, 0.10, 0.62)
            background = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
            bx = 0.64
            self._controls = {"time": (bx, 0.92, 0.83, 0.96), "now": (0.85, 0.92, 0.905, 0.96),
                              "close": (0.91, 0.92, 0.985, 0.96),
                              "previous": (bx, 0.79, bx + 0.13, 0.825), "next": (bx + 0.16, 0.79, bx + 0.28, 0.825)}
            positions = {self.time_button: (bx, 0.93), self.now_button: (0.85, 0.93),
                         self.close_button: (0.91, 0.93), self.time_caption: (bx, 0.89),
                         self.previous_button: (bx, 0.795), self.next_button: (bx + 0.16, 0.795),
                         self.profile_title: (bx, 0.765)}
        else:
            self._plot_box = (0.61, 0.97, 0.12, 0.25)
            background = ((0.59, 0.055), (0.99, 0.055), (0.99, 0.35), (0.59, 0.35))
            self._controls = {"time": (0.61, 0.45, 0.86, 0.49)}
            positions = {self.time_button: (0.61, 0.46), self.now_button: (0.90, 0.46),
                         self.close_button: (0.91, 0.46), self.time_caption: (0.61, 0.43),
                         self.previous_button: (0.61, 0.37), self.next_button: (0.77, 0.37),
                         self.profile_title: (0.61, 0.32)}
        for index, (x, y) in enumerate(background):
            self._bg_points.SetPoint(index, x, y, 0)
        self._bg_points.Modified()
        self.profile_background.GetMapper().GetInput().Modified()
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
        # Re-add the stats text last so it is drawn above the fullscreen background.
        self.renderer.RemoveViewProp(self.stats)
        self.renderer.AddViewProp(self.stats)

    def close_graph(self):
        self._graph_expanded = False
        self._now_mode = True
        self._sky_generation += 1  # discard any response still in flight
        self._sky_data = None
        self._sky_status = ""
        self._epochs = [utc_now()]
        self._time_index = 0
        self.apply_layout()
        self.update_profile(self.horizon_angles)
        self.update_stats()
        self.window.Render()

    def update_profile(self, angles):
        low, high = float(np.min(angles)), float(np.max(angles))
        if self._sky_data is not None:
            for body in self._sky_data.values():
                low = min(low, float(np.min(body["el"] - body["radius"])))
                high = max(high, float(np.max(body["el"] + body["radius"])))
        padding = max((high - low) * 0.08, 0.01)
        self._profile_low, self._profile_high = low - padding, high + padding
        values = np.append(angles, angles[0])
        x0, x1, y0, y1 = self._plot_box
        x = np.linspace(x0, x1, len(values))
        y = self.profile_y(values)
        self.profile.GetMapper().SetInputData(self.overlay_lines_data([np.column_stack((x, y))]))
        self.profile_title.SetInput(
            f"360-degree horizon: selected observer\n"
            f"Elevation scale {self._profile_low:+.2f} to {self._profile_high:+.2f} deg"
            + ("".join("\n" + SKY_LEGEND[source] for source in SKY_METHODS[self._sky_method])
               if self._graph_expanded else ""))
        for name in SKY_COLORS:
            body = self._sky_data.get(name) if self._sky_data is not None else None
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
                if name not in self._sky_data:
                    continue
                body, index = self._sky_data[name], self._time_index
                lines.append(f"{name}: {body['state'][index]}  |  AZ {body['az'][index]:.2f}, EL {body['el'][index]:+.2f} deg")
            if len(self._epochs) > 1:
                counts = [f"{name} {np.count_nonzero(body['visible'])}/{len(self._epochs)}" for name, body in self._sky_data.items()]
                lines.append("Visible samples: " + "  |  ".join(counts))
            if self._sky_status:
                lines.append(self._sky_status)
            lines.append("Partial visibility is approximate; terrain horizon sampled at 1 deg.")
        self.stats.SetInput("\n".join(lines))

    def request_sky_update(self):
        if self._now_mode:
            self._epochs = [utc_now()]
            self._time_index = 0
            self._last_now_query = time.monotonic()
        self._sky_generation += 1
        self._sky_data = None  # Never show a previous location's bodies at a new point.
        self._sky_status = "Loading Sun / Earth positions..." if self._enable_ephemeris else "Sun / Earth lookup disabled"
        self.update_profile(self.horizon_angles)
        self.update_stats()
        if not self._enable_ephemeris or not self._graph_expanded:
            return
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
    def fetch_skyfield(lon, lat, altitude, epochs):
        try:
            from skyfield_positions import get_positions
        except ImportError as error:
            raise RuntimeError(f"Install skyfield and keep skyfield_positions.py beside moon.py: {error}") from error
        return get_positions(lon, lat, altitude, epochs)

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
            for source, fetch in (("JPL", self._ephemeris_fetcher), ("Skyfield", self.fetch_skyfield)):
                if source not in methods:
                    continue
                try:
                    classified = classify_sky(fetch(lon, lat, altitude, epochs), horizon)
                    data.update({f"{body} {source}": values for body, values in classified.items()})
                except Exception as error:
                    message = str(error).strip().splitlines()
                    errors.append(f"{source}: " + (message[0] if message else type(error).__name__)[:100])
            self._sky_results.put((generation, data or None, "; ".join(errors) or None))

    def poll_sky(self, obj=None, event=None):
        changed = False
        while True:
            try:
                generation, data, error = self._sky_results.get_nowait()
            except queue.Empty:
                break
            if generation != self._sky_generation:
                continue  # A newer click or time edit made this response obsolete.
            self._sky_data = data
            self._sky_status = f"Unavailable: {error}" if error else ""
            changed = True
        if self._now_mode and self._graph_expanded and time.monotonic() - self._last_now_query >= 60:
            self.request_sky_update()
            changed = True
        if changed:
            self.update_profile(self.horizon_angles)
            self.update_stats()
            self.window.Render()

    def set_time_interval(self, start, stop, step_minutes, method=None):
        epochs = time_samples(start, stop, step_minutes)
        if method is not None:
            if method not in SKY_METHODS:
                raise ValueError("Choose JPL Horizons, Skyfield or Both.")
            self._sky_method = method
        self._graph_expanded = True
        self.apply_layout()
        self._now_mode = False
        self._epochs = epochs
        self._time_index = 0
        self._interval_step_minutes = float(step_minutes)
        self.request_sky_update()

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
        return x >= width * 0.59 and y <= height * 0.49

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
            start = self._epochs[0] if self._graph_expanded else utc_now()
            stop = start + timedelta(days=1) if (self._now_mode or not self._graph_expanded) else self._epochs[-1]
            dialog = TimeIntervalDialog(root, start, stop, self._interval_step_minutes, self.set_time_interval,
                                        self._sky_method)
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
                       "previous": lambda: self.step_time(-1), "next": lambda: self.step_time(1)}
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

    def look_around(self, dx, dy, render=True):
        """Turn the view in place: the camera position is the rotation centre."""
        camera = self.renderer.GetActiveCamera()
        height = max(self.window.GetSize()[1], 1)
        speed = np.degrees(2 * np.tan(np.radians(camera.GetViewAngle()) / 2) / height)
        camera.Yaw(dx * speed)
        camera.Pitch(dy * speed)
        camera.OrthogonalizeViewUp()
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
    parser.add_argument("--texture", help="Path to your 8K (or other resolution) Moon texture")
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
