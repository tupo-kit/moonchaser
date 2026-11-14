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
Press G to enter latitude, longitude; L to hide/show the lat/lon grid (its spacing follows the zoom); V for terrain view; R to reset; F for fullscreen; Escape to close.
Sun/Earth positions use Skyfield (offline, limited date range) or JPL Horizons (internet, wide range);
you pick the model in the time interval window, and only that model is queried. Time defaults to UTC now.
Press T to choose a time range; this opens a fullscreen horizon + Sun/Earth graph.
That window scrolls (mouse wheel, Up/Down, PageUp/PageDown, Home/End) to more graphs: Sun elevation vs
time and Earth elevation vs time.
Use Previous/Next, the Left/Right arrow keys (Shift: 10 samples) or the time slider under the horizon graph to step
samples; drag the scroll bar at the right edge to scroll; Close returns to the horizon-only view.

Files (keep them in one folder; run moon.py):
  terrain.py  DEM/texture download + loading, terrain sampling, horizon, panorama, lat/lon grid (no VTK).
  sky.py      time helpers, Sun/Earth visibility classification, Skyfield (offline) and JPL Horizons (online).
  hud.py      HudMixin: 2D overlay HUD, horizon inset, Sun/Earth graphs, slider, scroll bar.
  globe.py    GlobeMixin: 3D globe, camera/navigation, picking, grid, pole marks, terrain view.
  moon.py     Tk dialogs, MoonWindow (glue: state, input handling, sky worker thread) and main().
Dependencies: terrain <- sky <- hud, globe <- moon. Other files must use `terrain.COLOR_PATH`, not a copy.
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
from globe import GlobeMixin
from hud import SCROLL_STEP, HudMixin
from sky import (MAX_TIME_SAMPLES, SKY_METHODS, TIME_UNITS, classify_sky, get_positions, load_interval, parse_utc,
                 preload, save_interval, time_samples, utc_now)
from terrain import compute_horizon_360, compute_slope_deg, load_data, parse_coordinates, sample_dem_bilinear, lat_lon_to_xyz, terrain_panorama_samples


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


class MoonWindow(GlobeMixin, HudMixin):
    """VTK owns the desktop window and mouse interaction; no HTML is involved."""

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

        # Build the 3D globe first, then the 2D HUD on top (draw order matches the original single-file build).
        self._init_globe()
        self._init_hud()

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

    def set_now(self):
        self._now_mode = True
        self._time_index = 0
        self.request_sky_update()
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

        self.lat, self.lon = float(lat), float(lon)
        angles, distances, hlat, hlon = compute_horizon_360(lat, lon)
        self.horizon_angles = angles
        self.horizon_distances = distances
        if self._panorama_texture_rgb is None:
            with Image.open(terrain.COLOR_PATH) as texture_image:
                texture_image = texture_image.convert("RGB")
                self._panorama_texture_rgb = np.asarray(texture_image).copy()
        self._panorama_samples = terrain_panorama_samples(
            lat, lon, self._panorama_texture_rgb, bearings=max(360, self.window.GetSize()[0]))
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
            actions = {"now": self.set_now, "close": self.close_graph,
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
