"""Moon Horizon: a native desktop viewer, with no browser or web server.

Install: python -m pip install numpy Pillow vtk skyfield astroquery
Coordinate editor: requires Tkinter (Linux package: python3-tk).
Run:     python moon.py
Texture: python moon.py --texture /path/to/color.jpg

NASA color and elevation maps download on first run and retain their original resolution.
Left-drag to orbit; middle-drag or Shift-left-drag to pan; wheel to zoom.
Ctrl-left-drag (or Ctrl-middle-drag) orbits the terrain point under the cursor.
Click the Moon to select an observer.
Zoom stops above the terrain. The inset shows all 360 horizon bearings.
Prepared maps are saved beside this script for later runs.
Press G to enter latitude, longitude; L to hide/show the lat/lon grid (its spacing follows the zoom); V for terrain view; R to reset; F for fullscreen; Escape to close.
Sun/Earth positions use Skyfield (offline, limited date range) or JPL Horizons (internet, wide range);
you pick the model in the time interval window, and only that model is queried. Time defaults to UTC now.
Press T to choose a time range; this opens a fullscreen horizon + Sun/Earth graph.
That window scrolls (mouse wheel, Up/Down, PageUp/PageDown, Home/End) to more graphs: Sun elevation vs
time and Earth elevation vs time.
Point comparison: Shift+click (or A for the current point) adds numbered, coloured points; P opens the point
manager (add, remove, save to / load from a file; only the coordinates are stored). T opens the analysis page:
the horizon graph shows one point (use the < > buttons, or the , and .
keys, to change it), the Sun/Earth graphs combine all points (a button per point switches its graph on/off)
and, further down the page, tables compare sunlight, Earth visibility and the azimuth / elevation statistics.
Use Previous/Next, the Left/Right arrow keys (Shift: 10 samples) or the time slider under the horizon graph to step
samples; drag the scroll bar at the right edge to scroll; Close returns to the horizon-only view.

Files (keep them in one folder; run moon.py):
  terrain.py  DEM/texture download + loading, terrain sampling, horizon, panorama, lat/lon grid (no VTK).
  sky.py      time helpers, Sun/Earth visibility classification, Skyfield (offline) and JPL Horizons (online).
  hud.py      HudMixin: 2D overlay HUD, horizon inset, Sun/Earth graphs, slider, scroll bar.
  globe.py    GlobeMixin: 3D globe, camera/navigation, picking, grid, pole marks, terrain view.
  compare.py  point comparison without VTK/UI: palette, point-set files, statistics, table text.
  visibility_search.py  coarse, terrain-aware search for visibility thresholds.
  moon.py     Tk dialogs, MoonWindow (glue: state, input handling, sky worker thread) and main().
Dependencies: terrain <- sky <- compare <- hud, globe <- moon. Other files must use `terrain.COLOR_PATH`, not a copy.
"""

import argparse
import queue
import sys
import threading
import time
from datetime import timedelta

import numpy as np
from PIL import Image

import sky
import terrain
from compare import (MAX_POINTS, load_points, normalize_point, point_stats, same_point, save_points)
from visibility_search import SearchDialog, MAX_DISPLAY_RESULTS, cluster_color
from globe import GlobeMixin
import map_modes
from hud import SCROLL_STEP, HudMixin
from sky import (MAX_TIME_SAMPLES, SKY_METHODS, TIME_UNITS, classify_sky, get_positions, load_interval, parse_utc,
                 preload, save_interval, time_samples, utc_now)
from terrain import compute_horizon_360, compute_slope_deg, load_data, parse_coordinates, sample_dem_bilinear, lat_lon_to_xyz, terrain_panorama_samples


def run_responsive_event_loop(interactor, delay=0.01):
    """Keep VTK on the main thread while giving Python workers time to run."""
    interactor.SetDone(False)
    while not interactor.GetDone():
        interactor.ProcessEvents()
        if not interactor.GetDone():
            # Sleep releases Python's GIL. The interactor's blocking Start()
            # can otherwise starve the calculation threads while idle.
            time.sleep(delay)


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
    def __init__(self, parent, start, stop, step, unit, on_submit, method="Skyfield", title=None):
        import tkinter as tk
        from tkinter import ttk

        self.window = tk.Toplevel(parent)
        self.window.title(title or "Sun / Earth time interval (UTC)")
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


class PointsDialog:
    """Edit the comparison set: add the current location or typed coordinates, remove, save and load."""

    def __init__(self, parent, points, current, on_submit):
        import tkinter as tk
        from tkinter import ttk

        self.points = list(points)
        self.current = current
        self.on_submit = on_submit
        self.window = tk.Toplevel(parent)
        self.window.title("Comparison points")
        self.window.resizable(False, False)
        frame = ttk.Frame(self.window, padding=14)
        frame.grid(sticky="nsew")
        ttk.Label(frame, text=f"Points to compare (2 to {MAX_POINTS}):  latitude, longitude").grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))
        self.listbox = tk.Listbox(frame, width=34, height=12, selectmode="extended",
                                  exportselection=False, font="TkFixedFont")
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=scrollbar.set)
        self.listbox.grid(row=1, column=0, rowspan=6, sticky="nsew")
        scrollbar.grid(row=1, column=1, rowspan=6, sticky="ns")
        buttons = (("Add current location", self.add_current), ("Remove selected", self.remove_selected),
                   ("Clear all", self.clear), ("Load (replace)...", lambda: self.load(False)),
                   ("Load (add)...", lambda: self.load(True)), ("Save...", self.save))
        for row, (label, command) in enumerate(buttons, start=1):
            ttk.Button(frame, text=label, command=lambda c=command: self.guard(c)).grid(
                row=row, column=2, sticky="ew", padx=(10, 0), pady=2)
        ttk.Label(frame, text="Add by coordinates, e.g. -43.31, -11.36").grid(
            row=7, column=0, columnspan=3, sticky="w", pady=(10, 2))
        self.entry = ttk.Entry(frame, width=30)
        self.entry.grid(row=8, column=0, columnspan=2, sticky="ew")
        ttk.Button(frame, text="Add", command=lambda: self.guard(self.add_typed)).grid(
            row=8, column=2, sticky="ew", padx=(10, 0))
        self.error = ttk.Label(frame, foreground="#c02020", wraplength=430)
        self.error.grid(row=9, column=0, columnspan=3, sticky="w", pady=(8, 4))
        bottom = ttk.Frame(frame)
        bottom.grid(row=10, column=0, columnspan=3, sticky="e", pady=(6, 0))
        ttk.Button(bottom, text="Cancel", command=self.window.destroy).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(bottom, text="Apply", command=self.apply).grid(row=0, column=1)
        self.entry.bind("<Return>", lambda event: (self.guard(self.add_typed), "break")[-1])
        self.entry.bind("<Control-a>", lambda event: (event.widget.selection_range(0, "end"), "break")[-1])
        self.window.bind("<Escape>", lambda event: self.window.destroy())
        self.refresh()
        self.window.update_idletasks()
        self.window.lift()
        self.window.grab_set()
        self.entry.focus_force()

    def refresh(self):
        self.listbox.delete(0, "end")
        for number, (lat, lon) in enumerate(self.points, start=1):
            self.listbox.insert("end", f"P{number:<2} {lat:+10.5f}, {lon:+11.5f}")

    def guard(self, action):
        try:
            action()
        except ValueError as error:
            self.error.configure(text=str(error))
        else:
            self.error.configure(text="")
        self.refresh()

    def add(self, lat, lon):
        point = normalize_point(lat, lon)
        if any(same_point(point, other) for other in self.points):
            raise ValueError("That point is already in the set.")
        if len(self.points) >= MAX_POINTS:
            raise ValueError(f"A set holds at most {MAX_POINTS} points.")
        self.points.append(point)

    def add_current(self):
        self.add(*self.current)

    def add_typed(self):
        lat, lon = parse_coordinates(self.entry.get())
        self.add(lat, lon)
        self.entry.delete(0, "end")

    def remove_selected(self):
        chosen = set(self.listbox.curselection())
        if not chosen:
            raise ValueError("Select the points to remove first.")
        self.points = [point for index, point in enumerate(self.points) if index not in chosen]

    def clear(self):
        self.points = []

    def save(self):
        from tkinter import filedialog
        if not self.points:
            raise ValueError("There are no points to save.")
        path = filedialog.asksaveasfilename(
            parent=self.window, title="Save points", defaultextension=".json", initialfile="moon_points.json",
            filetypes=(("Point set (JSON)", "*.json"), ("All files", "*.*")))
        if path:
            try:
                save_points(path, self.points)
            except OSError as error:
                raise ValueError(f"Could not save the file: {error}") from None

    def load(self, append):
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            parent=self.window, title="Load points",
            filetypes=(("Point set (JSON or text)", "*.json *.txt *.csv"), ("All files", "*.*")))
        if not path:
            return
        loaded = load_points(path)
        if not append:
            self.points = loaded
            return
        merged = list(self.points)
        for point in loaded:
            if not any(same_point(point, other) for other in merged):
                merged.append(point)
        if len(merged) > MAX_POINTS:
            raise ValueError(f"A set holds at most {MAX_POINTS} points; loading would give {len(merged)}.")
        self.points = merged

    def apply(self):
        self.on_submit(list(self.points))
        self.window.destroy()


class MoonWindow(GlobeMixin, HudMixin):
    """VTK owns the desktop window and mouse interaction; no HTML is involved."""

    def on_render_start(self, obj=None, event=None):
        # The globe updates caption text and visibility first. Stack the HUD
        # using measured pixel heights, including a possible two-line warning.
        super().on_render_start(obj, event)
        if not hasattr(self, '_map_label'):
            return
        settings = self._overlay_settings
        expanded = self._graph_expanded
        self._overlay_button.SetVisibility(not expanded)
        self.help.SetVisibility(not expanded and settings['help'])
        self.grid_caption.SetVisibility(self.grid_caption.GetVisibility() and settings['grid_caption'])
        self.points_caption.SetVisibility(self.points_caption.GetVisibility() and settings['points_caption'])
        self._map_label.SetVisibility(not expanded and settings['map_status'])
        self.horizon.SetVisibility(not self._terrain_view and settings['horizon_outline'])
        self.legend_button.SetVisibility(expanded or settings['point_details'])
        self.stats.SetVisibility(self._legend_expanded and (expanded or settings['point_details']))
        inset = expanded or settings['horizon_inset']
        x0, x1, y0, y1 = self._plot_box
        self._horizon_map_button.SetPosition(x1, y1 + (0.07 if expanded else 0.018))
        self._horizon_map_button.SetVisibility(inset)
        for actor in (self.profile_background, self.profile_title, self.profile, self.profile_axes, *self.profile_labels):
            actor.SetVisibility(inset)
        self.panorama.SetVisibility(inset and self._panorama_samples is not None)
        for group in (self.sky_hlines, self.sky_trails_below, self.sky_trails,
                      self.sky_samples_below, self.sky_samples):
            for actor in group.values():
                actor.SetVisibility(inset)
        for rings in self.sky_markers.values():
            for actor in rings:
                actor.SetVisibility(inset)
        if not inset:
            for group in (self.sky_labels, self.sky_values):
                for actor in group.values():
                    actor.SetVisibility(False)
        self.update_map_legend()
        if self._graph_expanded:
            return
        height = max(self.window.GetSize()[1], 1)
        bottom = self.help.GetPosition()[1] * height
        if self.help.GetVisibility():
            size = [0.0, 0.0]
            self.help.GetSize(self.renderer, size)
            bottom += size[1] + 6
        for actor in (self.grid_caption, self.points_caption, self._map_label):
            if actor.GetVisibility():
                actor.SetPosition(0.025, bottom / height)
                size = [0.0, 0.0]
                actor.GetSize(self.renderer, size)
                bottom += size[1] + 6

    def __init__(self, offscreen=False, enable_ephemeris=True, ephemeris_fetcher=None):
        import vtk

        self.vtk = vtk
        self.lat, self.lon = -43.31, -11.36
        self.orbit_pivot = np.zeros(3)
        self._terrain_view = False
        self._terrain_actor = None
        self._terrain_saved_camera = None
        self._zoom_anchor = None
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
        # Point comparison (compare.py): the set, and the state of the comparison window.
        self._search_actor = None
        self._search_outline_actor = None
        self._search_point_picker = None
        self._search_rows = []
        self._search_selected = None
        self._search_parameters = None
        self._search_session = None
        self._visibility_dialog = None
        self._tk_pump_active = False
        self.GLOBE_HELP = self.GLOBE_HELP + "\nS: search visibility   |   I: analyze selected search point\nM: change map   |   K: hide/show map legend"
        self._points = []              # [(lat, lon), ...] chosen with Shift+click / A / P
        self._compare_active = False   # True: the graph window compares _compare_points
        self._compare_points = []      # snapshot of _points used by the running comparison
        self._compare_index = 0        # point shown in the horizon graph
        self._compare_visible = []     # per point: shown in the combined Sun/Earth graphs
        self._compare = None           # results: points, horizons, data, errors, epochs, source, stats
        self._stats_above_only = True  # azimuth / elevation statistics over samples above the horizon only
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
        self.coordinate_editing = False
        self._coordinate_root = None
        self._coordinate_timer = None
        self._coordinate_pump_observer = None
        self._overlay_settings = {name: True for name in (
            'help', 'horizon_inset', 'horizon_outline', 'point_details',
            'grid_caption', 'points_caption', 'map_status', 'map_legend')}
        self._overlay_dialog = None
        self._overlay_vars = {}

        # Build the 3D globe first, then the 2D HUD on top (draw order matches the original single-file build).
        self._init_globe()
        self._init_hud()
        self._panorama_texture_rgb = self._color_rgb  # share the original-resolution color pixels
        self._map_results = queue.Queue()
        self._map_mode = self._map_wanted = 'Color'
        self._horizon_mode = self._horizon_wanted = 'Color'
        self._loaded_maps = {'Color': (self._color_rgb, self._color_texture_data, None)}
        self._map_busy = None
        self._map_rgb = self._map_image = None
        self._map_label = self.make_text(0.025, 0.25, 14)
        self._map_label.SetInput('Map: Color  |  M: change map')
        self._horizon_map_button = self.make_text(0.97, 0.27, 14)
        self._horizon_map_button.GetTextProperty().SetJustificationToRight()
        self._horizon_map_button.SetInput('[ Horizon: Color > ]')
        self._overlay_button = self.make_text(0.73, 0.975, 14, top=True)
        self._overlay_button.SetInput('[ Display / H ]')
        self._map_legend_open = True
        self._map_legend_button = self.make_text(0.87, 0.82, 14)
        self._map_legend_button.SetVisibility(False)
        self._map_colorbar = vtk.vtkScalarBarActor()
        self._map_colorbar.SetPosition(0.87, 0.37)
        self._map_colorbar.SetWidth(0.12)
        self._map_colorbar.SetHeight(0.42)
        self._map_colorbar.SetNumberOfLabels(5)
        self._map_colorbar.SetLabelFormat('%.1f')
        self._map_colorbar.GetLabelTextProperty().SetColor(1, 1, 1)
        self._map_colorbar.GetTitleTextProperty().SetColor(1, 1, 1)
        self._map_colorbar.SetVisibility(False)
        self.renderer.AddViewProp(self._map_colorbar)

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
                mode = "surface_look" if owner._terrain_view and not owner._graph_expanded else ("look" if interactor.GetControlKey() else ("pan" if interactor.GetShiftKey() else "orbit"))
                self.start_drag(mode, "left")

            def middle_down(self, obj, event):
                mode = "surface_look" if owner._terrain_view and not owner._graph_expanded else ("look" if self.GetInteractor().GetControlKey() else "pan")
                self.start_drag(mode, "middle")

            def start_drag(self, mode, button):
                if owner.coordinate_editing or self._moon_drag_press is not None:
                    return
                self._moon_drag_mode, self._moon_drag_button = mode, button
                self._moon_drag_press = self.GetInteractor().GetEventPosition()
                if owner.over_sky_controls(*self._moon_drag_press):
                    self._moon_drag_mode = "ui"
                if owner.scrollbar_hit(*self._moon_drag_press):
                    self._moon_drag_mode = "scrollbar"
                    owner.scrollbar_press(self._moon_drag_press[1])
                elif owner.slider_hit(*self._moon_drag_press):
                    self._moon_drag_mode = "slider"
                    owner.slider_drag(self._moon_drag_press[0])
                self._moon_drag_last = self._moon_drag_press
                self._moon_drag_dragged = False
                self._moon_drag_cursor_pivot = (owner.cursor_surface_target(*self._moon_drag_press)
                                                if self._moon_drag_mode == "look" else None)
                self._moon_drag_pan_depth = (float(np.linalg.norm(
                    np.array(owner.renderer.GetActiveCamera().GetPosition()) - owner.view_surface_target()))
                    if self._moon_drag_mode == "pan" else None)

            def mouse_move(self, obj, event):
                if self._moon_drag_press is None or owner.coordinate_editing:
                    return
                current = self.GetInteractor().GetEventPosition()
                if self._moon_drag_mode in ("scrollbar", "slider"):
                    if self._moon_drag_mode == "scrollbar":
                        owner.scrollbar_drag(current[1])
                    else:
                        owner.slider_drag(current[0])
                    self._moon_drag_last = current
                    return
                if self._moon_drag_mode == "ui":
                    if np.hypot(current[0] - self._moon_drag_press[0], current[1] - self._moon_drag_press[1]) > 5:
                        self._moon_drag_dragged = True
                    return
                if ((self._moon_drag_mode in ("pan", "look", "surface_look") and current != self._moon_drag_press)
                        or np.hypot(current[0] - self._moon_drag_press[0], current[1] - self._moon_drag_press[1]) > 5):
                    self._moon_drag_dragged = True
                if self._moon_drag_dragged:
                    dx, dy = current[0] - self._moon_drag_last[0], current[1] - self._moon_drag_last[1]
                    if self._moon_drag_mode == "pan":
                        owner.pan(dx, dy, depth=self._moon_drag_pan_depth)
                    elif self._moon_drag_mode == "surface_look":
                        owner.terrain_look(dx, dy)
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
                if self._moon_drag_press is not None and button == "left":
                    position = self.GetInteractor().GetEventPosition()
                    if not self._moon_drag_dragged and self._moon_drag_mode in ("orbit", "ui"):
                        owner.pick(*position)
                    elif (self._moon_drag_mode == "pan" and np.hypot(position[0] - self._moon_drag_press[0],
                                                                      position[1] - self._moon_drag_press[1]) <= 3):
                        owner.add_point_at(*position)  # Shift+click without moving: add a comparison point
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
                elif owner._graph_expanded and key in ("left", "right"):
                    count = 10 if interactor.GetShiftKey() else 1
                    owner.step_time(count if key == "right" else -count)

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
                elif key == "s":
                    owner.begin_visibility_search()
                elif key == "i":
                    owner.analyze_selected_search_point()
                elif key == "a":
                    owner.add_current_point()
                elif key == "p":
                    owner.begin_points_entry()
                elif key == "m":
                    owner.cycle_map_mode()
                elif key == "k":
                    owner.toggle_map_legend()
                elif key == "h":
                    owner.begin_overlay_settings()
                elif key in ("comma", "less"):
                    owner.step_point(-1)
                elif key in ("period", "greater"):
                    owner.step_point(1)
                elif key == "bracketleft":
                    owner.step_time(-1)
                elif key == "bracketright":
                    owner.step_time(1)
                elif key == "l":
                    owner._grid_enabled = not owner._grid_enabled
                    owner.window.Render()
                elif key == "v":
                    owner.toggle_terrain_view()
                elif key == "r":
                    if owner._terrain_view:
                        owner.toggle_terrain_view(render=False)
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

    def cycle_map_mode(self):
        if self.coordinate_editing or self._graph_expanded:
            return
        modes = ('Color', 'Elevation', 'Slope')
        self._map_wanted = modes[(modes.index(self._map_wanted) + 1) % len(modes)]
        self._horizon_wanted = self._map_wanted
        self._apply_loaded_maps()
        self._schedule_map_generation()
        self.window.Render()

    def cycle_horizon_map_mode(self):
        """Cycle the horizon independently, without changing the globe."""
        if self.coordinate_editing:
            return
        modes = ('Color', 'Elevation', 'Slope')
        self._horizon_wanted = modes[(modes.index(self._horizon_wanted) + 1) % len(modes)]
        self._apply_loaded_maps()
        self._schedule_map_generation()
        self.window.Render()

    def _apply_loaded_maps(self):
        if self._map_wanted in self._loaded_maps and self._map_wanted != self._map_mode:
            rgb, image, legend = self._loaded_maps[self._map_wanted]
            self._map_rgb, self._map_image = rgb, image
            self.texture.SetInputData(image)
            self._map_mode = self._map_wanted
            self._map_label.SetInput(f'Map: {self._map_mode}  |  M: change map  |  K: legend')
            self.update_map_legend(legend)
        if self._horizon_wanted in self._loaded_maps and self._horizon_wanted != self._horizon_mode:
            rgb, _, _ = self._loaded_maps[self._horizon_wanted]
            self._panorama_texture_rgb = rgb
            self._horizon_mode = self._horizon_wanted
            self.update_panorama_samples(self.lat, self.lon)
            self.update_profile(self.horizon_angles)
        if self._horizon_wanted == self._horizon_mode:
            self._horizon_map_button.SetInput(f'[ Horizon: {self._horizon_mode} > ]')
        else:
            self._horizon_map_button.SetInput(f'[ Horizon: {self._horizon_wanted}... ]')

    def _schedule_map_generation(self):
        if self._map_busy is not None:
            return
        for mode in (self._map_wanted, self._horizon_wanted):
            if mode not in self._loaded_maps:
                self._start_map_generation(mode)
                return

    def update_map_legend(self, legend=None):
        if legend is not None:
            lookup = self.vtk.vtkLookupTable()
            lookup.SetNumberOfTableValues(len(legend['colors']))
            lookup.SetTableRange(legend['low'], legend['high'])
            lookup.Build()
            for i, rgb in enumerate(legend['colors']):
                lookup.SetTableValue(i, *(float(value) / 255 for value in rgb), 1.0)
            self._map_colorbar.SetLookupTable(lookup)
            self._map_colorbar.SetTitle(legend['title'])
            self._map_lookup = lookup
        visible = (self._map_mode in ('Elevation', 'Slope') and not self._graph_expanded
                   and self._overlay_settings['map_legend'])
        self._map_colorbar.SetVisibility(visible and self._map_legend_open)
        self._map_legend_button.SetVisibility(visible)
        self._map_legend_button.SetInput('[-] Legend' if self._map_legend_open else '[+] Legend')

    def toggle_map_legend(self):
        if self._map_mode not in ('Elevation', 'Slope') or self._graph_expanded:
            return
        self._map_legend_open = not self._map_legend_open
        self.update_map_legend()
        self.window.Render()

    def begin_overlay_settings(self):
        """Non-modal, individually switchable globe overlays."""
        if self.coordinate_editing:
            return
        import tkinter as tk
        from tkinter import ttk
        if self._overlay_dialog is not None:
            self._overlay_dialog.deiconify()
            self._overlay_dialog.lift()
            return
        try:
            root = self.get_native_root()
            dialog = tk.Toplevel(root)
            self._overlay_dialog = dialog
            dialog.title('Display options')
            frame = ttk.Frame(dialog, padding=14)
            frame.pack(fill='both', expand=True)
            ttk.Label(frame, text='Show on globe:').pack(anchor='w', pady=(0, 6))
            labels = (('help', 'Keyboard and mouse help'), ('horizon_inset', 'Horizon graph'),
                      ('horizon_outline', 'Horizon outline on globe'), ('point_details', 'Point details'),
                      ('grid_caption', 'Grid information'), ('points_caption', 'Comparison points information'),
                      ('map_status', 'Map mode information'), ('map_legend', 'Map color legend'))
            for key, title in labels:
                variable = tk.BooleanVar(master=dialog, value=self._overlay_settings[key])
                self._overlay_vars[key] = variable
                def changed(name=key, var=variable):
                    self._overlay_settings[name] = bool(var.get())
                    self.window.Render()
                ttk.Checkbutton(frame, text=title, variable=variable, command=changed).pack(anchor='w', pady=2)
            ttk.Label(frame, text='Use H or Display to reopen these options.').pack(anchor='w', pady=(8, 4))
            ttk.Button(frame, text='Close', command=self.close_overlay_settings).pack(anchor='e')
            dialog.protocol('WM_DELETE_WINDOW', self.close_overlay_settings)
            dialog.bind('<Escape>', lambda event: self.close_overlay_settings())
            dialog.deiconify()
            dialog.lift()
            dialog.focus_force()
        except tk.TclError as error:
            self._overlay_dialog = None
            self._overlay_vars.clear()
            print(f'Could not open display options: {error}', file=sys.stderr)

    def close_overlay_settings(self):
        dialog, self._overlay_dialog = self._overlay_dialog, None
        self._overlay_vars.clear()  # Release Tk variables on the main thread.
        if dialog is not None:
            dialog.destroy()

    def over_sky_controls(self, x, y):
        if self._graph_expanded:
            return True
        width, height = self.window.GetSize()
        nx, ny = x / max(width, 1), y / max(height, 1)
        return ((self._overlay_settings['horizon_inset'] and 0.59 <= nx <= 0.99 and 0.055 <= ny <= 0.35)
                or (self._overlay_settings['point_details'] and 0.02 <= nx <= 0.30 and 0.95 <= ny <= 0.995))

    def _start_map_generation(self, mode):
        self._map_busy = mode
        self._map_label.SetInput(f'Preparing {mode} map...  |  M: change map')
        self.window.Render()
        def worker():
            last = [None]
            def progress(message):
                if message != last[0]:
                    self._map_results.put(('progress', mode, message))
                    last[0] = message
            try:
                rgb, label = map_modes.generate(mode, progress)
                self._map_results.put(('ready', mode, (rgb, label)))
            except Exception as error:
                self._map_results.put(('error', mode, str(error)))
        threading.Thread(target=worker, daemon=True).start()

    def poll_maps(self):
        from vtk.util.numpy_support import numpy_to_vtk
        changed = False
        while True:
            try:
                kind, mode, value = self._map_results.get_nowait()
            except queue.Empty:
                break
            if kind == 'progress':
                if mode in (self._map_wanted, self._horizon_wanted):
                    self._map_label.SetInput(value + '  |  M: change map')
                    if self._horizon_wanted == mode:
                        self._horizon_map_button.SetInput(f'[ Horizon: {value} ]')
                    changed = True
                continue
            self._map_busy = None
            if kind == 'error':
                if mode in (self._map_wanted, self._horizon_wanted):
                    if self._horizon_wanted == mode:
                        self._horizon_wanted = self._horizon_mode
                    if self._map_wanted == mode:
                        self._map_wanted = self._map_mode
                    self._map_label.SetInput(f'{mode} map failed: {value}')
                    self._horizon_map_button.SetInput(f'[ Horizon: {self._horizon_mode} > ]')
                    changed = True
            else:
                rgb, legend = value
                height, width = rgb.shape[:2]
                pixels = numpy_to_vtk(rgb.reshape(-1, 3), deep=False)
                image = self.vtk.vtkImageData()
                image.SetDimensions(width, height, 1)
                image.GetPointData().SetScalars(pixels)
                self._loaded_maps[mode] = (rgb, image, legend)
                self._apply_loaded_maps()
                changed = True
            self._schedule_map_generation()
        if changed:
            self.window.Render()

    def start_skyfield_preload(self):
        with self._preload_lock:
            if self._preload_running or self._preload_state[0] == "ready":
                return
            self._preload_running = True
            self._preload_done.clear()
            self._preload_state = ("loading", "")
        def load_data():
            try:
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

    def wheel(self, direction):
        """Wheel zooms the Moon, or scrolls the page while the graph window is open."""
        if self._graph_expanded:
            self.scroll_to(self._scroll + direction * SCROLL_STEP)
        elif self._terrain_view:
            camera = self.renderer.GetActiveCamera()
            camera.SetViewAngle(float(np.clip(camera.GetViewAngle() * (1.08 if direction > 0 else 1 / 1.08), 15, 110)))
            self.window.Render()
        else:
            self.zoom(1.08 if direction > 0 else 1 / 1.08)

    def close_graph(self):
        was_compare = self._compare_active
        self._graph_expanded = False
        self.update_map_legend()
        self._scroll = 0.0
        self._now_mode = True
        self._sky_generation += 1  # discard any response still in flight
        self._sky_data = None
        self._sky_status = ""
        self._compare_active = False
        self._compare = None
        self._compare_points = []
        self.progress_text.SetVisibility(False)
        self._epochs = [utc_now()]
        self._time_index = 0
        if was_compare:  # the globe follows the point that was shown in the horizon graph
            self.select_location(self.lat, self.lon, render=False)
        self.apply_layout()
        self.update_profile(self.horizon_angles)
        self.update_stats()
        self.window.Render()

    def request_sky_update(self):
        if self._now_mode:
            self._epochs = [utc_now()]
            self._time_index = 0
            self._last_now_query = time.monotonic()
        self._sky_generation += 1
        self._sky_data = None  # Never show a previous location's bodies at a new point.
        self._compare = None
        self._sky_status = "Loading Sun / Earth positions..." if self._enable_ephemeris else "Sun / Earth lookup disabled"
        active = self._enable_ephemeris and self._graph_expanded
        self.progress_text.SetInput("Working: starting calculation...")
        self.progress_text.SetVisibility(active)
        if self._compare_active:
            self.sync_compare_points()
        self.update_profile(self.horizon_angles)
        self.update_stats()
        if not active:
            return
        self.window.Render()
        # One job per point: (lon, lat, altitude km, terrain horizon or None to compute it in the worker).
        if self._compare_active:
            jobs = [(lon, lat, self.site_altitude_km(lat, lon), None) for lat, lon in self._compare_points]
        else:
            jobs = [(self.lon, self.lat, self.site_altitude_km(self.lat, self.lon), self.horizon_angles.copy())]
        request = (self._sky_generation, jobs, list(self._epochs), SKY_METHODS[self._sky_method])
        # Coalesce fast clicks: keep only the newest waiting location/interval.
        try:
            self._sky_jobs.get_nowait()
        except queue.Empty:
            pass
        self._sky_jobs.put_nowait(request)
        if self._sky_thread is None:
            self._sky_thread = threading.Thread(target=self.sky_worker, daemon=True)
            self._sky_thread.start()

    # JPL fetch lives in sky.py; kept as a staticmethod so sky_worker can call self.fetch_jpl.
    fetch_jpl = staticmethod(sky.fetch_jpl)

    def fetch_skyfield(self, lon, lat, altitude, epochs, progress=None):
        if self.skyfield_preload_status()[0] != "ready":
            self.start_skyfield_preload()
            if progress is not None:
                progress("Preparing Skyfield data")
            self._preload_done.wait()
            status, detail = self.skyfield_preload_status()
            if status != "ready":
                raise RuntimeError(f"Skyfield data unavailable: {detail}")
        return get_positions(lon, lat, altitude, epochs, progress=progress)

    @staticmethod
    def site_altitude_km(lat, lon):
        return (float(sample_dem_bilinear(lat, lon)) + 2.0) / 1000.0

    def sky_for_point(self, lon, lat, altitude, epochs, horizon, methods, progress):
        """Sun / Earth data of one point for every requested model: (data or None, error text or None)."""
        data, errors = {}, []
        fetchers = {"JPL": self.fetch_jpl, "Skyfield": self.fetch_skyfield}
        for source in methods:
            fetch = self._ephemeris_fetcher or fetchers[source]
            try:
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
        return data or None, "; ".join(errors) or None

    def sky_worker(self):
        while not self._sky_stop.is_set():
            try:
                request = self._sky_jobs.get(timeout=0.2)
            except queue.Empty:
                continue
            generation, jobs, epochs, methods = request
            results, errors = [], []
            for number, (lon, lat, altitude, horizon) in enumerate(jobs, start=1):
                prefix = f"Point {number}/{len(jobs)}: " if len(jobs) > 1 else ""
                progress = lambda message, prefix=prefix: self._sky_progress.put((generation, prefix + message))
                entry = {"data": None, "horizon": None, "error": None}
                try:
                    if horizon is None:  # comparison: the terrain horizon of every point is computed here
                        progress("Computing terrain horizon")
                        entry["horizon"] = compute_horizon_360(lat, lon)
                        horizon = entry["horizon"][0]
                    entry["data"], entry["error"] = self.sky_for_point(
                        lon, lat, altitude, epochs, horizon, methods, progress)
                except Exception as error:
                    message = str(error).strip().splitlines()
                    entry["error"] = (message[0] if message else type(error).__name__)[:100]
                if entry["error"]:
                    errors.append((f"P{number}: " if len(jobs) > 1 else "") + entry["error"])
                results.append(entry)
            self._sky_results.put((generation, results, "; ".join(errors) or None))

    def poll_sky(self, obj=None, event=None):
        self.poll_maps()
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
                generation, results, error = self._sky_results.get_nowait()
            except queue.Empty:
                break
            if generation != self._sky_generation:
                continue  # A newer click or time edit made this response obsolete.
            self.progress_text.SetInput("Working: building graphs...")
            self.progress_text.SetVisibility(True)
            self.window.Render()
            if self._compare_active:
                self.apply_compare_results(results, error)
            else:
                self._sky_data = results[0]["data"] if results else None
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
        self.update_map_legend()
        self._compare_active = True
        self._analysis_single_point = None
        self._compare = None
        current = (self.lat, self.lon)
        self._compare_points = []
        self._compare_index = next((i for i, point in enumerate(self._points) if same_point(point, current)), 0)
        self.apply_layout()
        self._now_mode = False
        self._epochs = epochs
        self._time_index = 0
        self._interval_step_minutes = float(step) * TIME_UNITS[unit]
        self.request_sky_update()

    def set_now(self):
        self._now_mode = True
        self._time_index = 0
        self.request_sky_update()
        self.window.Render()

    def begin_visibility_search(self):
        if self.coordinate_editing:
            return
        try:
            import tkinter as tk
        except ImportError:
            print("Visibility search needs Tkinter.", file=sys.stderr)
            return
        if self._visibility_dialog is not None:
            self._visibility_dialog.window.deiconify()
            self._visibility_dialog.window.lift()
            self._visibility_dialog.window.focus_force()
            return
        try:
            root = self.get_native_root()
            # Non-modal: the existing VTK timer pumps Tk events, so both the
            # viewer and the search window remain responsive.
            self._visibility_dialog = SearchDialog(root, self.fetch_skyfield,
                                                    self._search_session, self.finish_visibility_search,
                                                    existing_points=lambda: list(self._points))
        except tk.TclError as error:
            print(f"Could not open visibility search: {error}", file=sys.stderr)

    def finish_visibility_search(self, dialog):
        self._visibility_dialog = None
        if dialog.finished:
            self._search_session = {"results": dialog.results, "parameters": dialog.last_parameters,
                                    "cancelled": dialog.was_cancelled}
        if dialog.clear_markers:
            self.show_visibility_markers([])
            self._search_selected = None
        elif dialog.show_matches:
            if self._graph_expanded:
                self.close_graph()
            if self._terrain_view:
                self.toggle_terrain_view(render=False)
            self._search_parameters = dialog.last_parameters
            if dialog.add_to_compare:
                points = list(self._points)
                for row in dialog.selected_points:
                    point = normalize_point(row['lat'], row['lon'])
                    if not any(same_point(point, other) for other in points):
                        points.append(point)
                if len(points) > MAX_POINTS:
                    self._points_note = f'A set holds at most {MAX_POINTS} points.'
                    self.refresh_point_marks()
                    self.window.Render()
                    return
                self.set_points(points)
            rows = dialog.marker_results if dialog.marker_results is not None else dialog.results
            self.show_visibility_markers(rows[:MAX_DISPLAY_RESULTS], mode=dialog.marker_mode)
            if dialog.selected is not None:
                point = dialog.selected
                self.lookup_coordinates(f"{point['lat']}, {point['lon']}", render=False)
                self._search_selected = point
                if dialog.analyze:
                    self.analyze_selected_search_point(render=False, use_points=dialog.add_to_compare)
        self.window.Render()

    def analyze_selected_search_point(self, render=True, use_points=None):
        """Analyze the searched point or selected point set over the search interval."""
        if self._search_selected is None or self._search_parameters is None:
            return
        if self._terrain_view:
            self.toggle_terrain_view(render=False)
        start, stop, step_hours = self._search_parameters[:3]
        epochs = time_samples(start, stop, float(step_hours) * 60)
        self._sky_method = "Skyfield"
        self._graph_expanded = True
        self.update_map_legend()
        self._scroll = 0.0
        self._compare_active = True
        if use_points is None:
            use_points = len(self._points) > 1
        self._analysis_single_point = None if use_points else normalize_point(
            self._search_selected['lat'], self._search_selected['lon'])
        self._compare_index = 0
        self._compare = None
        self._compare_points = []
        self._now_mode = False
        self._epochs = epochs
        self._time_index = 0
        self._interval_step_minutes = float(step_hours) * 60.0
        self.apply_layout()
        self.request_sky_update()
        if render:
            self.window.Render()

    def show_visibility_markers(self, results, mode='Points'):
        """Draw selectable points and, in cluster mode, colored cell outlines."""
        from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
        for actor in (self._search_actor, self._search_outline_actor):
            if actor is not None:
                self.renderer.RemoveActor(actor)
        self._search_actor = self._search_outline_actor = self._search_point_picker = None
        self._search_rows = list(results)
        if not results:
            return
        lat = np.radians([item['lat'] for item in results])
        lon = np.radians([item['lon'] for item in results])
        directions = np.column_stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)))
        locations = self.terrain_overlay_points(directions)
        points = self.vtk.vtkPoints()
        points.SetData(numpy_to_vtk(np.ascontiguousarray(locations, dtype=np.float64), deep=True))
        n = len(locations)
        cells = self.vtk.vtkCellArray()
        cells.SetData(numpy_to_vtkIdTypeArray(np.arange(n + 1, dtype=np.int64), deep=True),
                      numpy_to_vtkIdTypeArray(np.arange(n, dtype=np.int64), deep=True))
        data = self.vtk.vtkPolyData()
        data.SetPoints(points)
        data.SetVerts(cells)
        colors = np.asarray([np.rint(np.asarray(cluster_color(row.get('cluster_index', 0))) * 255)
                             for row in results], dtype=np.uint8)
        data.GetPointData().SetScalars(numpy_to_vtk(colors, deep=True, array_type=self.vtk.VTK_UNSIGNED_CHAR))
        mapper = self.vtk.vtkPolyDataMapper()
        mapper.SetInputData(data)
        mapper.SetScalarModeToUsePointData()
        mapper.SetColorModeToDirectScalars()
        self.configure_surface_overlay(mapper)
        actor = self.vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetPointSize(9)
        actor.GetProperty().LightingOff()
        actor.PickableOn()
        self.renderer.AddActor(actor)
        self._search_actor = actor
        picker = self.vtk.vtkPointPicker()
        picker.SetTolerance(0.008)
        picker.PickFromListOn()
        picker.AddPickList(actor)
        self._search_point_picker = picker
        if mode == 'Clusters':
            self.show_search_cell_outlines(results)

    def show_search_cell_outlines(self, results):
        """Outline matched cells in their cluster colors on the terrain mesh."""
        from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
        units, offsets, colors = [], [0], []
        for row in results:
            keys = ('cell_lat_min', 'cell_lat_max', 'cell_lon_min', 'cell_lon_max')
            if not all(key in row for key in keys):
                continue
            lo_lat, hi_lat, lo_lon, hi_lon = (row[key] for key in keys)
            count = max(2, min(12, int(np.ceil(max(hi_lat-lo_lat, hi_lon-lo_lon) / 2.5))))
            lats = np.concatenate((np.full(count, lo_lat), np.linspace(lo_lat, hi_lat, count, endpoint=False),
                                   np.full(count, hi_lat), np.linspace(hi_lat, lo_lat, count, endpoint=False), [lo_lat]))
            lons = np.concatenate((np.linspace(lo_lon, hi_lon, count, endpoint=False), np.full(count, hi_lon),
                                   np.linspace(hi_lon, lo_lon, count, endpoint=False), np.full(count, lo_lon), [lo_lon]))
            lat, lon = np.radians(lats), np.radians(lons)
            units.append(np.column_stack((np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat))))
            offsets.append(offsets[-1] + len(lats))
            colors.append(np.rint(np.asarray(cluster_color(row.get('cluster_index', 0))) * 255))
        if not units:
            return
        coords = self.terrain_overlay_points(np.concatenate(units))
        points = self.vtk.vtkPoints()
        points.SetData(numpy_to_vtk(np.ascontiguousarray(coords, dtype=np.float64), deep=True))
        cells = self.vtk.vtkCellArray()
        cells.SetData(numpy_to_vtkIdTypeArray(np.asarray(offsets, dtype=np.int64), deep=True),
                      numpy_to_vtkIdTypeArray(np.arange(len(coords), dtype=np.int64), deep=True))
        data = self.vtk.vtkPolyData()
        data.SetPoints(points)
        data.SetLines(cells)
        data.GetCellData().SetScalars(numpy_to_vtk(np.asarray(colors, dtype=np.uint8), deep=True,
                                                    array_type=self.vtk.VTK_UNSIGNED_CHAR))
        mapper = self.vtk.vtkPolyDataMapper()
        mapper.SetInputData(data)
        mapper.SetScalarModeToUseCellData()
        mapper.SetColorModeToDirectScalars()
        self.configure_surface_overlay(mapper)
        actor = self.vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetLineWidth(2)
        actor.GetProperty().LightingOff()
        actor.PickableOff()
        self.renderer.AddActor(actor)
        self._search_outline_actor = actor

    def pick_search_marker(self, x, y):
        if self._search_point_picker is None or self._terrain_view:
            return False
        if not self._search_point_picker.Pick(x, y, 0, self.renderer):
            return False
        index = self._search_point_picker.GetPointId()
        if index < 0 or index >= len(self._search_rows):
            return False
        point = np.asarray(self._search_actor.GetMapper().GetInput().GetPoint(index))
        camera_position = np.asarray(self.renderer.GetActiveCamera().GetPosition())
        hit = self.surface_hit(camera_position, point - camera_position)
        if hit is None or np.linalg.norm(hit - point) > 0.002:
            return False  # the globe hides the marker
        row = self._search_rows[index]
        self.select_location(row['lat'], row['lon'], render=False)
        self._search_selected = row
        self._stats_base += "\nSearch point selected; press I to analyze its interval"
        self.update_stats()
        self.window.Render()
        return True

    def toggle_terrain_view(self, render=True):
        GlobeMixin.toggle_terrain_view(self, render=False)
        for actor in (self._search_actor, self._search_outline_actor):
            if actor is not None:
                actor.SetVisibility(not self._terrain_view)
        if render:
            self.window.Render()

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
            title = (f"Compare {len(self._points)} points: time interval (UTC)" if len(self._points) >= 2 else None)
            dialog = TimeIntervalDialog(root, start, stop, step, unit, self.set_time_interval, self._sky_method, title)
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
        if self._coordinate_root is not None and not self.coordinate_editing and not self._tk_pump_active:
            self._tk_pump_active = True
            try:
                self._coordinate_root.update()
            finally:
                self._tk_pump_active = False

    def lookup_coordinates(self, text, render=True):
        lat, lon = parse_coordinates(text)
        if self._terrain_view:
            self.toggle_terrain_view(render=False)
        distance = np.linalg.norm(self.renderer.GetActiveCamera().GetPosition())
        self.select_location(lat, lon, render=False)
        self.reset_camera(distance=distance)
        if render:
            self.window.Render()
        return lat, lon

    def select_location(self, lat, lon, render=True):
        from vtk.util.numpy_support import numpy_to_vtk

        self._search_selected = None
        self.lat, self.lon = float(lat), float(lon)
        self._points_note = ""
        angles, distances, hlat, hlon = compute_horizon_360(lat, lon)
        self.horizon_angles = angles
        self.horizon_distances = distances
        self.update_panorama_samples(lat, lon)
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
        self._stats_base = self.build_stats_base(lat, lon, angles, distances)
        self.request_sky_update()
        if render:
            self.window.Render()

    def update_panorama_samples(self, lat, lon):
        """Terrain colours / angles for the panorama behind the horizon graph."""
        if self._panorama_texture_rgb is None:
            with terrain.allow_large_nasa_image():
                with Image.open(terrain.COLOR_PATH) as texture_image:
                    texture_image = texture_image.convert("RGB")
                    self._panorama_texture_rgb = np.asarray(texture_image)
        self._panorama_samples = terrain_panorama_samples(
            lat, lon, self._panorama_texture_rgb, bearings=max(360, self.window.GetSize()[0]))
        self._panorama_range = None

    def build_stats_base(self, lat, lon, angles, distances, heading="MOON HORIZON"):
        elevation = float(sample_dem_bilinear(lat, lon))
        return (
            f"{heading}\n"
            f"Latitude {lat:+.3f} deg    Longitude {lon:+.3f} deg\n"
            f"Elevation {elevation:+,.0f} m    Slope {compute_slope_deg(lat, lon):.1f} deg\n"
            f"Horizon distance {np.min(distances):.1f} to {np.max(distances):.1f} km\n"
            f"Horizon angle {np.min(angles):+.2f} to {np.max(angles):+.2f} deg\n"
            "360 bearings; 2 m eye height; terrain search to 250 km\n"
            f"Texture: {self.texture_dimensions[0]} x {self.texture_dimensions[1]}")

    # ------------------------------------------------------------------ point comparison

    def sync_compare_points(self):
        """Take over the current point set for a comparison (all points shown, index kept in range)."""
        single = getattr(self, '_analysis_single_point', None)
        points = [single] if single is not None else list(self._points) or [(self.lat, self.lon)]
        if points != self._compare_points:
            self._compare_points = points
            self._compare_visible = [True] * len(points)
            self._compare_index = min(self._compare_index, len(points) - 1)
            self.apply_layout()

    def add_point(self, lat, lon, render=True):
        try:
            point = normalize_point(lat, lon)
        except ValueError as error:
            self._points_note = str(error)
        else:
            if any(same_point(point, other) for other in self._points):
                self._points_note = "That point is already in the set."
            elif len(self._points) >= MAX_POINTS:
                self._points_note = f"A set holds at most {MAX_POINTS} points."
            else:
                self._points.append(point)
                self._points_note = ""
                self.points_changed(render)
                return True
        if render:
            self.window.Render()
        return False

    def add_current_point(self):
        if not self.coordinate_editing:
            self.add_point(self.lat, self.lon)

    def add_point_at(self, x, y):
        """Shift+click on the Moon: select that location and add it to the comparison set."""
        if self.coordinate_editing or self._graph_expanded or self._terrain_view:
            return
        if self.picker.Pick(x, y, 0, self.renderer):
            px, py, pz = self.picker.GetPickPosition()
            radius = np.sqrt(px * px + py * py + pz * pz)
            if radius > 0:
                lat = float(np.degrees(np.arcsin(np.clip(pz / radius, -1, 1))))
                lon = float(np.degrees(np.arctan2(py, px)))
                points = list(self._points)
                for point in (normalize_point(self.lat, self.lon), normalize_point(lat, lon)):
                    if not any(same_point(point, other) for other in points):
                        points.append(point)
                if len(points) > MAX_POINTS:
                    self._points_note = f'A set holds at most {MAX_POINTS} points.'
                    self.window.Render()
                    return
                self.select_location(lat, lon, render=False)
                self.set_points(points)
                self.window.Render()

    def set_points(self, points):
        points = list(points)
        if points == self._points:
            return
        self._points = points
        self._points_note = ""
        self.points_changed()

    def points_changed(self, render=True):
        self.refresh_point_marks()
        if self._compare_active:
            self._analysis_single_point = None
            self.request_sky_update()  # recalculate for the new set, including a single point
        if render:
            self.window.Render()

    def exit_compare(self):
        """Fewer than two points left: carry on with the single point that is shown."""
        self._compare_active = False
        self._compare = None
        self._compare_points = []
        self.apply_layout()
        self.request_sky_update()

    def begin_points_entry(self):
        if self.coordinate_editing or self._terrain_view:
            return
        try:
            import tkinter as tk
        except ImportError:
            print("The point manager needs Tkinter (Debian/Ubuntu: python3-tk).", file=sys.stderr)
            return
        dialog = None
        try:
            root = self.get_native_root()
            self.coordinate_editing = True
            dialog = PointsDialog(root, self._points, (self.lat, self.lon), self.set_points)
            root.wait_window(dialog.window)
        except tk.TclError as error:
            print(f"Could not open the point manager: {error}", file=sys.stderr)
        finally:
            if dialog is not None:
                try:
                    dialog.window.destroy()
                except tk.TclError:
                    pass
            self.coordinate_editing = False
            self.window.Render()

    def compute_compare_stats(self):
        compare = self._compare
        if compare is None:
            return
        compare["stats"] = [point_stats(compare["epochs"], data, compare["source"], self._stats_above_only)
                            for data in compare["data"]]
        epochs = compare["epochs"]
        compare["stats_span"] = (epochs[-1] - epochs[0]).total_seconds() if len(epochs) > 1 else None

    def apply_compare_results(self, results, error):
        points = list(self._compare_points)
        self._compare = {"points": points, "horizons": [r["horizon"] for r in results],
                         "data": [r["data"] for r in results], "errors": [r["error"] for r in results],
                         "epochs": list(self._epochs), "source": SKY_METHODS[self._sky_method][0], "stats": None}
        self.compute_compare_stats()
        self.compare_activate(min(self._compare_index, len(points) - 1), refresh=False)

    def compare_activate(self, index, refresh=True):
        """Show point `index` in the horizon graph (its terrain, panorama, details and Sun/Earth trails)."""
        compare = self._compare
        if compare is None or not 0 <= index < len(compare["points"]):
            return
        self._compare_index = index
        lat, lon = compare["points"][index]
        horizon = compare["horizons"][index] or (np.zeros(360), np.zeros(360), None, None)
        angles, distances = horizon[0], horizon[1]
        self.lat, self.lon = lat, lon
        self.horizon_angles, self.horizon_distances = angles, distances
        self.update_panorama_samples(lat, lon)
        self._stats_base = self.build_stats_base(
            lat, lon, angles, distances, f"MOON HORIZON - POINT {index + 1}/{len(compare['points'])}")
        self._sky_data = compare["data"][index]
        error = compare["errors"][index]
        self._sky_status = f"Unavailable: {error}" if error else ""
        if refresh:
            self.update_profile(angles)
            self.update_stats()
            self.window.Render()

    def step_point(self, delta):
        """Next / previous point of a running comparison (wraps around)."""
        count = len(self._compare_points)
        if self._compare_active and self._graph_expanded and self._compare is not None and count > 1:
            self.compare_activate((self._compare_index + delta) % count)

    def pick(self, x, y):
        width, height = self.window.GetSize()
        if self.coordinate_editing:
            return
        if self._horizon_map_button.GetVisibility():
            px, py = self._horizon_map_button.GetPosition()
            size = [0.0, 0.0]
            self._horizon_map_button.GetSize(self.renderer, size)
            if px * width - size[0] - 5 <= x <= px * width + 5 and py * height - 5 <= y <= py * height + size[1] + 5:
                self.cycle_horizon_map_mode()
                return
        if not self._graph_expanded and 0.73 <= x / width <= 0.87 and 0.94 <= y / height <= 1.0:
            self.begin_overlay_settings()
            return
        if (not self._graph_expanded and self._map_mode in ('Elevation', 'Slope')
                and self._overlay_settings['map_legend']
                and 0.87 <= x / width <= 1.0 and 0.81 <= y / height <= 0.87):
            self.toggle_map_legend()
            return
        if self.over_sky_controls(x, y):
            nx, ny = x / width, y / height
            actions = {"now": self.set_now, "close": self.close_graph,
                       "previous": lambda: self.step_time(-1), "next": lambda: self.step_time(1),
                       "Skyfield": lambda: self.toggle_source("Skyfield"),
                       "JPL": lambda: self.toggle_source("JPL"),
                       "legend": self.toggle_legend,
                       "point_prev": lambda: self.step_point(-1), "point_next": lambda: self.step_point(1),
                       "stats_mode": self.toggle_stats_mode}
            for name, (left, bottom, right, top) in self._controls.items():
                if left <= nx <= right and bottom <= ny <= top:
                    if name.startswith("vis:"):  # "vis:<panel>:<point index>": point on/off in the combined graphs
                        self.toggle_point_visibility(int(name.rsplit(":", 1)[1]))
                    else:
                        actions[name]()
                    break
            return
        if self.pick_search_marker(x, y):
            return
        if self.picker.Pick(x, y, 0, self.renderer):
            self._search_selected = None
            px, py, pz = self.picker.GetPickPosition()
            radius = np.sqrt(px * px + py * py + pz * pz)
            if radius > 0:
                lat = np.degrees(np.arcsin(np.clip(pz / radius, -1, 1)))
                lon = np.degrees(np.arctan2(py, px))
                self.select_location(float(lat), float(lon))

    def run(self):
        self.interactor.Initialize()
        self._sky_pump_observer = self.interactor.AddObserver("TimerEvent", self.poll_sky)
        self._sky_timer = self.interactor.CreateRepeatingTimer(100)
        self.poll_sky()
        self.window.Render()
        # Explicitly handle window-manager close as well as Escape.
        exit_observer = self.interactor.AddObserver(
            "ExitEvent", lambda obj, event: obj.SetDone(True))
        try:
            run_responsive_event_loop(self.interactor)
        finally:
            self.interactor.RemoveObserver(exit_observer)
            self._sky_stop.set()
            if self._sky_timer is not None:
                self.interactor.DestroyTimer(self._sky_timer)
            if self._sky_pump_observer is not None:
                self.interactor.RemoveObserver(self._sky_pump_observer)
            if self._coordinate_timer is not None:
                self.interactor.DestroyTimer(self._coordinate_timer)
            if self._coordinate_pump_observer is not None:
                self.interactor.RemoveObserver(self._coordinate_pump_observer)
            if self._visibility_dialog is not None:
                # Stop its worker and timer before destroying the Tk root.
                self._visibility_dialog.on_close = None
                self._visibility_dialog.close()
                self._visibility_dialog = None
            if self._overlay_dialog is not None:
                self.close_overlay_settings()
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
