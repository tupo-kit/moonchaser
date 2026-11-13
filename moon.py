"""Moon Horizon: a native desktop viewer, with no browser or web server.

Install: python -m pip install numpy Pillow vtk
Run:     python moon.py
Texture: python moon.py --texture /path/to/moon_color_8k.jpg

Use your local 8K texture; the NASA elevation map downloads on first run.
Left-drag to orbit; middle-drag or Shift-left-drag to pan; wheel to zoom.
Ctrl-left-drag (or Ctrl-middle-drag) looks around from the current camera position.
Click the Moon to select an observer.
Zoom stops above the terrain. The inset shows all 360 horizon bearings.
Put moon_color_8k.jpg beside this script to use it automatically.
Press G to enter latitude, longitude; R to reset; F for fullscreen; Escape to close.
"""

import argparse
import os
import sys
import urllib.request

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


class MoonWindow:
    """VTK owns the desktop window and mouse interaction; no HTML is involved."""

    def __init__(self, offscreen=False):
        import vtk
        from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

        self.vtk = vtk
        self.lat, self.lon = -43.31, -11.36
        self.orbit_pivot = np.zeros(3)
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
        self.coordinate_buffer = ""
        self.coordinate_prompt = self.make_text(0.025, 0.75, 16, top=True)
        self.coordinate_prompt.GetTextProperty().SetColor(0.25, 0.9, 1.0)
        self.update_coordinate_prompt()
        self.help = self.make_text(0.025, 0.035, 15)
        self.help.SetInput("Left-drag: orbit   |   Middle/Shift-drag: pan\n"
                           "Ctrl-drag: look around   |   Wheel: zoom   |   Click: select\n"
                           "G: coordinates   |   R: reset   |   F: fullscreen   |   Esc: close\n"
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
                self.AddObserver("MouseWheelForwardEvent", lambda obj, event: owner.zoom(1 / 1.08))
                self.AddObserver("MouseWheelBackwardEvent", lambda obj, event: owner.zoom(1.08))
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
                self._moon_drag_last = self._moon_drag_press
                self._moon_drag_dragged = False
                self._moon_drag_pan_depth = float(np.linalg.norm(
                    np.array(owner.renderer.GetActiveCamera().GetPosition()) - owner.view_surface_target()))

            def mouse_move(self, obj, event):
                if self._moon_drag_press is None or owner.coordinate_editing:
                    return
                current = self.GetInteractor().GetEventPosition()
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
                if self._moon_drag_press is not None and not self._moon_drag_dragged and self._moon_drag_mode == "orbit" and button == "left":
                    owner.pick(*self.GetInteractor().GetEventPosition())
                self._moon_drag_press = self._moon_drag_last = None
                self._moon_drag_mode = self._moon_drag_button = self._moon_drag_pan_depth = None

            def key_press(self, obj, event):
                interactor = self.GetInteractor()
                key = (interactor.GetKeySym() or "").lower()
                if owner.coordinate_editing:
                    if key in ("escape", "return", "kp_enter", "backspace", "delete") or interactor.GetControlKey():
                        owner.edit_coordinates(key, control=bool(interactor.GetControlKey()))
                elif key == "escape":
                    owner.interactor.TerminateApp()

            def key(self, obj, event):
                interactor = self.GetInteractor()
                key = (interactor.GetKeySym() or "").lower()
                if key in ("escape", "return", "kp_enter", "backspace", "delete") or interactor.GetControlKey():
                    return  # Special keys were handled once in KeyPressEvent.
                if owner.coordinate_editing:
                    owner.edit_coordinates(key, interactor.GetKeyCode())
                elif key == "g":
                    owner.begin_coordinate_entry()
                elif key == "r":
                    owner.reset_camera()
                    owner.window.Render()
                elif key == "f":
                    owner.window.SetFullScreen(not owner.window.GetFullScreen())
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

    def update_profile(self, angles):
        low, high = float(np.min(angles)), float(np.max(angles))
        padding = max((high - low) * 0.1, 0.01)
        values = np.append(angles, angles[0])
        x = np.linspace(0.61, 0.97, len(values))
        y = 0.12 + 0.13 * (values - low + padding) / (high - low + 2 * padding)
        self.profile.GetMapper().SetInputData(self.overlay_lines_data([np.column_stack((x, y))]))
        self.profile_title.SetInput(
            f"360-degree horizon: selected observer\n"
            f"Elevation angle {low:+.2f} to {high:+.2f} deg")

    def update_coordinate_prompt(self, error=""):
        self.coordinate_prompt.SetVisibility(self.coordinate_editing)
        if self.coordinate_editing:
            self.coordinate_prompt.SetInput(
                "Latitude, longitude (signed or 0 to 360)\n"
                f"> {self.coordinate_buffer}_\n"
                + (error or "Enter: go   |   Esc: cancel   |   Ctrl+A: clear"))
        else:
            self.coordinate_prompt.SetInput("")

    def begin_coordinate_entry(self, render=True):
        self.coordinate_editing = True
        self.coordinate_buffer = ""
        self.update_coordinate_prompt()
        if render:
            self.window.Render()

    def edit_coordinates(self, key, character="", control=False, render=True):
        if not self.coordinate_editing:
            return
        error = ""
        if key == "escape":
            self.coordinate_editing = False
        elif key in ("return", "kp_enter"):
            try:
                self.lookup_coordinates(self.coordinate_buffer, render=False)
                self.coordinate_editing = False
            except ValueError as exc:
                error = str(exc)
        elif control and key == "a":
            self.coordinate_buffer = ""
        elif key in ("backspace", "delete"):
            self.coordinate_buffer = self.coordinate_buffer[:-1]
        elif not control and character and character in "0123456789+-. ,;":
            if len(self.coordinate_buffer) < 64:
                self.coordinate_buffer += character
        self.update_coordinate_prompt(error)
        if render:
            self.window.Render()

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
        self.stats.SetInput(
            f"MOON HORIZON\n"
            f"Latitude {lat:+.3f} deg    Longitude {lon:+.3f} deg\n"
            f"Elevation {elevation:+,.0f} m    Slope {compute_slope_deg(lat, lon):.1f} deg\n"
            f"Horizon distance {np.min(distances):.1f} to {np.max(distances):.1f} km\n"
            f"Horizon angle {np.min(angles):+.2f} to {np.max(angles):+.2f} deg\n"
            "360 bearings; 2 m eye height; terrain search to 250 km\n"
            f"Texture: {self.texture_dimensions[0]} x {self.texture_dimensions[1]}")
        if render:
            self.window.Render()

    def pick(self, x, y):
        width, height = self.window.GetSize()
        if self.coordinate_editing:
            return
        if x >= width * 0.59 and y <= height * 0.35:
            return  # The profile overlay is not a selectable location on the Moon.
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
        self.window.Render()
        try:
            self.interactor.Start()
        finally:
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


