"""Moon Horizon: a native desktop viewer, with no browser or web server.

Install: python -m pip install -r requirements.txt
Run:     python moon.py
Texture: python moon.py --texture /path/to/color.jpg

NASA color and elevation maps download on first run and retain their original resolution.
Use the Navigate tool to orbit; middle-drag or Shift-left-drag to pan; wheel to zoom.
Ctrl-left-drag (or Ctrl-middle-drag) orbits the terrain point under the cursor.
Use Add to pick observers, or the search panel's area tool to set search bounds.
Zoom stops above the terrain. The inset shows all 360 horizon bearings.
Prepared maps are saved beside this script for later runs.
Press G to focus the coordinate field; L to hide/show the lat/lon grid (its spacing follows the zoom); V for terrain view; E to set the eye height; R to reset; F for fullscreen. Escape does nothing; close the window to quit.
Sun/Earth positions use Skyfield (offline, limited date range) or JPL Horizons (internet, wide range);
you pick the model in the Analysis tab, and only that model is queried. Time defaults to UTC now.
Press T to open the Analysis tab; use the sidebar to switch between the map and analysis.
That page scrolls (mouse wheel, Up/Down, PageUp/PageDown, Home/End) to more graphs: Sun elevation vs
time and Earth elevation vs time.
Point comparison: Shift-click with Add adds a temporary selection; Add to points saves it. P opens the Points tab
(rename, tick the points to analyze, remove, save to / load from a file; names are stored with the coordinates). T opens the analysis page:
the horizon graph shows one point (choose it in the Analysis sidebar), the Sun/Earth graphs combine all points
and, in a Qt panel under the graphs, tables compare sunlight, Earth visibility and the azimuth / elevation statistics (Export CSV in the top bar).
Use Previous/Next, the Left/Right arrow keys (Shift: 10 samples) or the fixed time slider to step
samples; drag the scroll bar at the right edge to scroll; use the sidebar to return to the map.

Files (keep them in one folder; run moon.py):
  terrain.py  DEM/texture download + loading, terrain sampling, horizon, panorama, lat/lon grid (no VTK).
  sky.py      time helpers, Sun/Earth visibility classification, Skyfield (offline) and JPL Horizons (online).
  hud.py      HudMixin: 2D overlay HUD, horizon inset, Sun/Earth graphs, slider, scroll bar.
  globe.py    GlobeMixin: 3D globe, camera/navigation, picking, grid, pole marks, terrain view.
  compare.py  point comparison without VTK/UI: palette, point-set files, statistics, table text.
  point_data.py  memory-only store shared by the search window and the comparison (horizon + classified sky per
              point and interval), so no point is calculated twice.
  visibility_search.py  coarse, terrain-aware search for visibility thresholds.
  desktop_ui.py  Qt shell: header, top bar (Globe | Analysis), sidebar tabs (Search, Points, Analysis), tables panel, location card.
  moon.py     MoonWindow (glue: state, input handling, sky worker thread) and main().
Dependencies: terrain <- sky <- compare <- point_data <- visibility_search, hud, globe <- moon. Other files must use `terrain.COLOR_PATH`, not a copy.
"""

import argparse
import queue
import signal
import sys
import threading
import time

import numpy as np
from PIL import Image

import sky
import terrain
from compare import MAX_POINTS, normalize_point, point_stats, same_point
from visibility_search import cluster_color
from globe import GlobeMixin
import map_modes
import point_data
from hud import SCROLL_STEP, HudMixin
from sky import (SKY_METHODS, TIME_UNITS, classify_sky, get_positions, load_interval, parse_utc,
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


class MoonWindow(GlobeMixin, HudMixin):
    """VTK owns the desktop window and mouse interaction; no HTML is involved."""

    def on_render_start(self, obj=None, event=None):
        # The globe updates caption text and visibility first. Stack the HUD
        # using measured pixel heights, including a possible two-line warning.
        super().on_render_start(obj, event)
        if not hasattr(self, '_map_label'):
            return
        size = tuple(self.window.GetSize())
        if self._panorama_samples is not None and min(size) > 1 and size != self._panorama_window_size:
            self._panorama_window_size = size
            self.update_panorama()
        self.rescale_text()
        if self._desktop_shell is not None:
            self._desktop_shell.sync_state()
        settings = self._overlay_settings
        expanded = self._graph_expanded
        self._overlay_button.SetVisibility(not expanded and self._desktop_shell is None)
        self.help.SetVisibility(not expanded and settings['help'] and self._desktop_shell is None)
        self.grid_caption.SetVisibility(False)
        self.points_caption.SetVisibility(False)
        self._map_label.SetVisibility(not expanded and settings['map_status'] and self._desktop_shell is None)
        self.horizon.SetVisibility(not self._terrain_view and settings['horizon_outline'] and not getattr(self, '_observer_suppressed', False))
        if getattr(self, '_observer_suppressed', False):
            self.observer.SetVisibility(False)
        if self._selection_actor is not None:
            self._selection_actor.SetVisibility(not self._terrain_view and not expanded)
        self.legend_button.SetVisibility(False)
        self.stats.SetVisibility(expanded and self._desktop_shell is None)
        inset = expanded or (settings['horizon_inset'] and not getattr(self, '_observer_suppressed', False))
        x0, x1, y0, y1 = self._plot_box
        self._horizon_map_button.SetPosition(x1, y1 + (0.07 if expanded else 0.018))
        self._horizon_map_button.SetVisibility(inset and not (expanded and self._desktop_shell is not None))
        self._horizon_close.SetPosition(self._horizon_preview_rect[2] - 0.025, self._horizon_preview_rect[3] - 0.035)
        self._horizon_close.SetVisibility(False)
        for actor in (self.profile_background, self.profile_title, self.profile, self.profile_axes, *self.profile_labels):
            actor.SetVisibility(inset)
        if expanded and self._desktop_shell is not None:
            self.profile_title.SetVisibility(False)
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
            self._hide_zen_overlays()
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
        self._hide_zen_overlays()

    def _hide_zen_overlays(self):
        if not getattr(self, '_zen', False):
            return
        props = self.renderer.GetViewProps(); props.InitTraversal()
        for _ in range(props.GetNumberOfItems()):
            actor = props.GetNextProp()
            if actor is not None and actor.IsA('vtkActor2D'):
                self._zen_actors.setdefault(id(actor), (actor, actor.GetVisibility()))
                actor.SetVisibility(False)

    def set_zen(self, enabled):
        self._zen = bool(enabled)
        if not enabled:
            for actor, visible in self._zen_actors.values():
                actor.SetVisibility(visible)
            self._zen_actors.clear()
        self.window.Render()

    def __init__(self, offscreen=False, enable_ephemeris=True, ephemeris_fetcher=None,
                 render_window=None, interactor=None, desktop_shell=None):
        import vtk

        self.vtk = vtk
        self._desktop_shell = desktop_shell
        self._zen = False
        self._zen_actors = {}
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
        self._horizon_bodies = {"Sun": True, "Earth": True}
        self._legend_expanded = True
        self._saved_interval = load_interval()
        self._sky_method = self._saved_interval[4] if self._saved_interval else "Skyfield"
        # Point comparison (compare.py): the set, and the state of the comparison window.
        self._search_actor = None
        self._search_outline_actor = None
        self._search_fill_actor = None
        self._search_fill_picker = None
        self._search_fill_rows = []
        self._cluster_occupancy = None
        self._observer_suppressed = False
        self._search_point_picker = None
        self._search_rows = []
        self._search_selected = None
        self._search_parameters = None
        self._search_session = None
        self._point_cache = point_data.CACHE  # shared with the search window: stored sky data is never recomputed
        self._visibility_dialog = None
        self._points = []              # saved analysis points, independent of temporary selections
        self._selected_locations = []  # temporary map selections, independent of saved analysis points
        self._selection_actor = None
        self._rectangle_actor = None
        self._area_fill_actor = None
        self._area_bounds = None
        self._area_drawing = False
        self._previous_map_tool = 'Navigate'
        self._panorama_window_size = None
        self._horizon_preview_rect = (0.59, 0.055, 0.99, 0.35)
        self._map_tool = 'Navigate'
        self._point_names = {}         # {point: name}; a point without a name is shown as 'Point N'
        self._point_off = set()        # points kept in the list and on the globe but left out of the analysis
        self._compare_active = False   # True: the graph window compares _compare_points
        self._analysis_points_override = None
        self._compare_points = []      # snapshot of _points used by the running comparison
        self._compare_index = 0        # point shown in the horizon graph
        self._compare_visible = []     # per point: shown in the combined Sun/Earth graphs
        self._compare = None           # results: points, horizons, data, errors, epochs, source, stats
        self._stats_above_only = True  # azimuth / elevation statistics over samples above the horizon only
        self._graph_expanded = False  # False: horizon only. True: fullscreen horizon + Sun/Earth graph.
        self._analysis_snapshot = None
        self._analysis_can_reopen = False
        self._analysis_inputs = None
        self._hovered_close = None
        self._scroll = 0.0  # window heights the expanded page is scrolled down
        self._panel_boxes = {}
        self._epochs = [utc_now()]
        self._time_index = 0
        self._interval_step_minutes = 60
        self._last_now_query = time.monotonic()
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.025, 0.035, 0.055)
        self.renderer.UseFXAAOn()
        self.window = render_window if render_window is not None else vtk.vtkRenderWindow()
        self.window.SetWindowName("Moonchaser")
        if render_window is None:
            self.window.SetSize(1280, 900)
        # Some X11/GLX drivers cannot create a multisampled context. FXAA on
        # the renderer still smooths edges without requiring an MSAA visual.
        self.window.SetMultiSamples(0)
        self.window.AddRenderer(self.renderer)
        if offscreen:
            self.window.SetOffScreenRendering(True)
        self.interactor = interactor if interactor is not None else vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.window)
        self._overlay_settings = {name: True for name in (
            'help', 'horizon_inset', 'horizon_outline', 'point_details',
            'grid_caption', 'points_caption', 'map_status', 'map_legend')}

        # Build the 3D globe first, then the 2D HUD on top (draw order matches the original single-file build).
        self._init_globe()
        self._init_hud()
        self._horizon_close = self.make_text(0.965, 0.315, 17)
        self._horizon_close.SetInput('×')
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
        self._map_colorbar.SetOrientationToHorizontal()
        self._map_colorbar.SetPosition(0.025, 0.025)
        self._map_colorbar.SetWidth(0.30)
        self._map_colorbar.SetHeight(0.075)
        self._map_colorbar.SetBarRatio(0.2)
        self._map_colorbar.SetNumberOfLabels(3)
        self._map_colorbar.UnconstrainedFontSizeOn()
        self._map_colorbar.GetLabelTextProperty().SetFontSize(12)
        self._map_colorbar.GetTitleTextProperty().SetFontSize(12)
        self._map_colorbar.SetLabelFormat('%.1f')
        self._map_colorbar.GetLabelTextProperty().SetColor(1, 1, 1)
        self._map_colorbar.GetTitleTextProperty().SetColor(1, 1, 1)
        self._map_colorbar.GetLabelTextProperty().SetFontFamilyToArial()
        self._map_colorbar.GetTitleTextProperty().SetFontFamilyToArial()
        self._map_colorbar.GetLabelTextProperty().ShadowOff()
        self._map_colorbar.GetTitleTextProperty().ShadowOff()
        self._map_colorbar.SetVisibility(False)
        self.renderer.AddViewProp(self._map_colorbar)

        owner = self

        class MoonInteraction(vtk.vtkInteractorStyleUser):
            def __init__(self):
                super().__init__()
                self._area_last_update = 0.0
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
                self.AddObserver("RightButtonPressEvent", self.right_down)
                self.AddObserver("MouseMoveEvent", self.mouse_move)
                self.AddObserver("MouseWheelForwardEvent", lambda obj, event: owner.wheel(-1))
                self.AddObserver("MouseWheelBackwardEvent", lambda obj, event: owner.wheel(1))
                self.AddObserver("KeyPressEvent", self.key_press)
                self.AddObserver("CharEvent", self.key)
                self.AddObserver("ConfigureEvent", lambda obj, event: owner.constrain_camera())

            def left_down(self, obj, event):
                interactor = self.GetInteractor()
                self._selection_extend = bool(interactor.GetShiftKey())
                edge = owner.horizon_resize_edge(*interactor.GetEventPosition())
                if edge:
                    mode = 'resize'
                    self._resize_edge, self._resize_rect = edge, owner._horizon_preview_rect
                elif not owner._graph_expanded and not owner._terrain_view and owner._map_tool == 'Add':
                    mode = 'look' if interactor.GetControlKey() else ('pan' if interactor.GetShiftKey() else 'select')
                elif not owner._graph_expanded and not owner._terrain_view and owner._map_tool == 'Area':
                    mode = 'rectangle'
                else:
                    mode = "surface_look" if owner._terrain_view and not owner._graph_expanded else ("look" if interactor.GetControlKey() else ("pan" if interactor.GetShiftKey() else "orbit"))
                self.start_drag(mode, "left")

            def middle_down(self, obj, event):
                mode = "surface_look" if owner._terrain_view and not owner._graph_expanded else ("look" if self.GetInteractor().GetControlKey() else "pan")
                self.start_drag(mode, "middle")

            def right_down(self, obj, event):
                if owner._area_drawing:
                    self._moon_drag_press = self._moon_drag_last = None
                    self._moon_drag_mode = self._moon_drag_button = self._moon_drag_pan_depth = None
                    if owner._desktop_shell is not None:
                        owner._desktop_shell.cancel_area_tool()

            def start_drag(self, mode, button):
                if self._moon_drag_press is not None:
                    return
                self._moon_drag_mode, self._moon_drag_button = mode, button
                self._moon_drag_press = self.GetInteractor().GetEventPosition()
                if mode != "resize" and owner.over_sky_controls(*self._moon_drag_press):
                    self._moon_drag_mode = "ui"
                if mode != "resize" and owner.scrollbar_hit(*self._moon_drag_press):
                    self._moon_drag_mode = "scrollbar"
                    owner.scrollbar_press(self._moon_drag_press[1])
                elif mode != "resize" and owner.slider_hit(*self._moon_drag_press):
                    self._moon_drag_mode = "slider"
                    owner.slider_drag(self._moon_drag_press[0])
                self._moon_drag_last = self._moon_drag_press
                self._moon_drag_dragged = False
                self._moon_drag_cursor_pivot = (owner.cursor_surface_target(*self._moon_drag_press)
                                                if self._moon_drag_mode == "look" else None)
                self._moon_drag_pan_depth = (float(np.linalg.norm(
                    np.array(owner.renderer.GetActiveCamera().GetPosition()) - owner.view_surface_target()))
                    if self._moon_drag_mode in ("pan", "select") else None)

            def mouse_move(self, obj, event):
                if self._moon_drag_press is None:
                    pos = self.GetInteractor().GetEventPosition()
                    owner.update_close_hover(*pos)
                    if owner._desktop_shell is not None:
                        owner._desktop_shell.set_preview_cursor(owner.horizon_resize_edge(*pos))
                    return
                current = self.GetInteractor().GetEventPosition()
                if self._moon_drag_mode == 'resize':
                    owner.resize_horizon_preview(self._resize_edge, self._resize_rect, self._moon_drag_press, current)
                    self._moon_drag_dragged = True
                    return
                if self._moon_drag_mode == 'rectangle':
                    if np.hypot(current[0] - self._moon_drag_press[0], current[1] - self._moon_drag_press[1]) > 5:
                        self._moon_drag_dragged = True
                        now = time.monotonic()
                        if now - self._area_last_update >= 0.08:
                            self._area_last_update = now
                            owner.select_search_rectangle(self._moon_drag_press, current, commit=False)
                    return
                if self._moon_drag_mode == 'select':
                    if np.hypot(current[0] - self._moon_drag_press[0], current[1] - self._moon_drag_press[1]) > 5:
                        self._moon_drag_dragged = True
                    if self._moon_drag_dragged:
                        dx, dy = current[0] - self._moon_drag_last[0], current[1] - self._moon_drag_last[1]
                        owner.orbit(dx, dy)
                    self._moon_drag_last = current
                    return
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
                if ((self._moon_drag_mode in ("look", "surface_look") and current != self._moon_drag_press)
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
                    if self._moon_drag_mode == 'rectangle' and self._moon_drag_dragged:
                        owner.select_search_rectangle(self._moon_drag_press, position, commit=True)
                    elif not self._moon_drag_dragged and self._moon_drag_mode == 'select':
                        owner.pick(*position, select_new=True, extend=self._selection_extend)
                    elif not self._moon_drag_dragged and self._moon_drag_mode == 'pan' and self._selection_extend:
                        owner.pick(*position, select_new=(owner._map_tool == 'Add'), extend=True)
                    elif not self._moon_drag_dragged and self._moon_drag_mode in ("orbit", "ui"):
                        owner.pick(*position)
                self._moon_drag_press = self._moon_drag_last = None
                self._moon_drag_mode = self._moon_drag_button = self._moon_drag_pan_depth = None
                self._moon_drag_cursor_pivot = None

            def key_press(self, obj, event):
                interactor = self.GetInteractor()
                key = (interactor.GetKeySym() or "").lower()
                # Escape is deliberately ignored (it used to quit); close the window to leave the program.
                if owner._graph_expanded and key in ("up", "down", "prior", "next", "home", "end"):
                    owner.scroll_key(key)
                elif owner._graph_expanded and key in ("left", "right"):
                    count = 10 if interactor.GetShiftKey() else 1
                    owner.step_time(count if key == "right" else -count)

            def key(self, obj, event):
                interactor = self.GetInteractor()
                key = (interactor.GetKeySym() or "").lower()
                if key in ("escape", "return", "kp_enter", "backspace", "delete") or interactor.GetControlKey():
                    return  # Special keys were handled once in KeyPressEvent.
                if key == "g":
                    owner.begin_coordinate_entry()
                elif key == "t":
                    owner.begin_time_entry()
                elif key == "e":
                    owner.begin_eye_height_entry()
                elif key == "s":
                    owner.begin_visibility_search()
                elif key == "i":
                    owner.analyze_selected_search_point()
                elif key == "p":
                    owner.begin_points_entry()
                elif key == "m":
                    owner.cycle_map_mode()
                elif key == "k":
                    owner.toggle_map_legend()
                elif key == "h":
                    owner.begin_overlay_settings()
                elif key in ("comma", "less") and owner._desktop_shell is None:
                    owner.step_point(-1)
                elif key in ("period", "greater") and owner._desktop_shell is None:
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
        if self._graph_expanded:
            return
        modes = ('Color', 'Elevation', 'Slope')
        self._map_wanted = modes[(modes.index(self._map_wanted) + 1) % len(modes)]
        self._horizon_wanted = self._map_wanted
        self._apply_loaded_maps()
        self._schedule_map_generation()
        self.window.Render()

    def cycle_horizon_map_mode(self):
        """Cycle the horizon independently, without changing the globe."""
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
        self._map_legend_button.SetVisibility(False)

    def toggle_map_legend(self):
        if self._map_mode not in ('Elevation', 'Slope') or self._graph_expanded:
            return
        self._map_legend_open = not self._map_legend_open
        self.update_map_legend()
        self.window.Render()

    def begin_overlay_settings(self):
        if self._desktop_shell is not None:
            self._desktop_shell.show_view_menu()

    def over_sky_controls(self, x, y):
        if self._graph_expanded:
            return True
        width, height = self.window.GetSize()
        nx, ny = x / max(width, 1), y / max(height, 1)
        left, bottom, right, top = self._horizon_preview_rect
        return bool(self._overlay_settings['horizon_inset'] and left <= nx <= right and bottom <= ny <= top)

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
        self._analysis_can_reopen = True
        if self._compare_active:
            self._analysis_snapshot = {
                'compare': self._compare, 'points': list(self._compare_points),
                'visible': list(self._compare_visible), 'index': self._compare_index,
                'epochs': list(self._epochs), 'time_index': self._time_index,
                'method': self._sky_method, 'step': self._interval_step_minutes,
            }
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
            self.select_location(self.lat, self.lon, render=False, extend_selection=True)
        self.apply_layout()
        self.update_profile(self.horizon_angles)
        self.update_stats()
        self.window.Render()

    def reopen_analysis(self):
        snapshot = self._analysis_snapshot
        if snapshot is None or self._graph_expanded:
            return
        self._graph_expanded = True
        self._compare_active = True
        self._compare = snapshot['compare']
        self._compare_points = list(snapshot['points'])
        self._compare_visible = list(snapshot['visible'])
        self._compare_index = snapshot['index']
        self._epochs = list(snapshot['epochs'])
        self._time_index = min(snapshot['time_index'], len(self._epochs)-1)
        self._sky_method = snapshot['method']
        self._interval_step_minutes = snapshot['step']
        self._now_mode = False
        self._scroll = 0.0
        if self._compare is None:
            self.apply_layout()
            self.request_sky_update()
            self.window.Render()
            return
        self.compare_activate(self._compare_index, refresh=False)
        self.apply_layout(); self.update_profile(self.horizon_angles)
        self.update_stats(); self.update_table()
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
        return (float(sample_dem_bilinear(lat, lon)) + terrain.EYE_HEIGHT_M) / 1000.0

    def sky_for_point(self, lon, lat, altitude, epochs, horizon, methods, progress):
        """Sun / Earth data of one point for every requested model: (data or None, error text or None)."""
        data, errors = {}, []
        fetchers = {"JPL": self.fetch_jpl, "Skyfield": self.fetch_skyfield}
        shared = self._ephemeris_fetcher is None  # a test fetcher is never mixed with real stored data
        for source in methods:
            fetch = self._ephemeris_fetcher or fetchers[source]
            try:
                classified = self._point_cache.get_sky(lat, lon, epochs, source) if shared else None
                if classified is not None:
                    progress(f"Using stored {source} data")
                else:
                    progress(f"Starting {source} lookup")
                    if self._ephemeris_fetcher is not None:
                        positions = fetch(lon, lat, altitude, epochs)
                    else:
                        positions = fetch(lon, lat, altitude, epochs, progress=progress)
                    progress("Checking terrain visibility and eclipses")
                    classified = classify_sky(positions, horizon)
                    if shared:
                        self._point_cache.put_sky(lat, lon, epochs, source, classified)
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
                        shared = self._ephemeris_fetcher is None
                        stored = self._point_cache.get_horizon(lat, lon) if shared else None
                        if stored is None:
                            progress("Computing terrain horizon")
                            stored = compute_horizon_360(lat, lon)
                            if shared:
                                self._point_cache.put_horizon(lat, lon, stored)
                        entry["horizon"] = stored
                        horizon = stored[0]
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
        self._analysis_snapshot = None
        self._analysis_can_reopen = False
        self._sky_method = method
        self._saved_interval = (parse_utc(start), parse_utc(stop), float(step), unit, method)
        save_interval(start, stop, step, unit, method)
        self._graph_expanded = True
        self.update_map_legend()
        self._compare_active = True
        self._analysis_single_point = None
        self._analysis_points_override = None
        self._compare = None
        self._compare_visible = [visible and point not in self._point_off
                                 for point, visible in zip(self._compare_points, self._compare_visible)]
        current = (self.lat, self.lon)
        self._compare_index = next((i for i, point in enumerate(self._points) if same_point(point, current)), 0)
        self.apply_layout()
        self._now_mode = False
        self._epochs = epochs
        self._time_index = 0
        self._interval_step_minutes = float(step) * TIME_UNITS[unit]
        self.record_analysis(list(self._points), start, stop, self._interval_step_minutes, method, len(epochs))
        self.request_sky_update()

    def record_analysis(self, points, start, stop, step_minutes, method, samples):
        self._analysis_inputs = {
            'points': list(points), 'start': parse_utc(start).replace(microsecond=0),
            'stop': parse_utc(stop).replace(microsecond=0), 'step': step_minutes, 'method': method,
            'eye': terrain.EYE_HEIGHT_M, 'samples': samples}

    def set_now(self):
        self._now_mode = True
        self._time_index = 0
        self.request_sky_update()
        self.window.Render()

    def begin_visibility_search(self):
        if self._desktop_shell is not None:
            self._desktop_shell.show_panel('search')

    def apply_visibility_search(self, dialog):
        """A button of the (still open) search window asked for markers / points / analysis."""
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
            is_sector = bool(rows and rows[0].get('kind') == 'sector')
            self.show_visibility_markers(rows, mode=dialog.marker_mode)
            if dialog.selected is not None:
                point = dialog.selected
                self.lookup_coordinates(f"{point['lat']}, {point['lon']}", render=False)
                self._search_selected = point
                if dialog.analyze:
                    selected = [normalize_point(row['lat'], row['lon']) for row in dialog.selected_points]
                    self.analyze_selected_search_point(render=False, points=selected or None)
                    if self._desktop_shell is not None:
                        self._desktop_shell.show_panel('time')
                elif dialog.add_to_compare and self._desktop_shell is not None:
                    self._desktop_shell.show_panel('points')
        self.window.Render()

    def analyze_selected_search_point(self, render=True, use_points=None, points=None):
        """Analyze the searched point or selected point set over the search interval."""
        if self._search_selected is None or self._search_parameters is None:
            return
        self._analysis_snapshot = None
        self._analysis_can_reopen = False
        if self._terrain_view:
            self.toggle_terrain_view(render=False)
        start, stop, step_hours = self._search_parameters[:3]
        epochs = time_samples(start, stop, float(step_hours) * 60)
        self._sky_method = "Skyfield"
        self._graph_expanded = True
        self.update_map_legend()
        self._scroll = 0.0
        self._compare_active = True
        self._analysis_points_override = list(points) if points is not None else None
        if use_points is None:
            use_points = len(self.analysis_points()) > 1
        self._analysis_single_point = None if use_points or points is not None else normalize_point(
            self._search_selected['lat'], self._search_selected['lon'])
        self._compare_index = 0
        self._compare = None
        self._compare_points = []
        self._now_mode = False
        self._epochs = epochs
        self._time_index = 0
        self._interval_step_minutes = float(step_hours) * 60.0
        analyzed = self._analysis_points_override or ([self._analysis_single_point] if self._analysis_single_point else self.analysis_points())
        self.record_analysis(analyzed,
                             start, stop, self._interval_step_minutes, self._sky_method, len(epochs))
        self.apply_layout()
        self.request_sky_update()
        if render:
            self.window.Render()

    def show_visibility_markers(self, results, mode='Points'):
        """Draw selectable point markers or ungridded filled clusters."""
        from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
        for actor in (self._search_actor, self._search_outline_actor, self._search_fill_actor):
            if actor is not None:
                self.renderer.RemoveActor(actor)
        self._search_actor = self._search_outline_actor = self._search_point_picker = None
        self._search_fill_actor = self._search_fill_picker = None
        self._search_fill_rows = []
        self._cluster_occupancy = None
        self._search_rows = list(results)
        if not results:
            self.refresh_search_area_mask()
            return
        if mode == 'Clusters':
            self.show_search_cluster_fills(results)
            self.refresh_search_area_mask()
            return
        self.refresh_search_area_mask()
        if results[0].get('kind') == 'sector':
            return  # Candidate clusters are areas, not verified point markers.
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

    def show_search_cluster_fills(self, results):
        """Color globe mesh cells under candidate sectors with translucent cluster hues."""
        from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
        # The globe uses 0.5-degree quads. Select intersecting quads so even a
        # sector smaller than one mesh cell has a visible approximation.
        rows, cols = 360, 720
        lat_centers = 90.0 - (np.arange(rows) + 0.5) * 180.0 / rows
        lon_centers = -180.0 + (np.arange(cols) + 0.5) * 360.0 / cols
        owners = np.full((rows, cols), -1, dtype=np.int32)
        for index, row in enumerate(results):
            if not all(k in row for k in ('cell_lat_min', 'cell_lat_max', 'cell_lon_min', 'cell_lon_max')):
                continue
            lat0, lat1 = row['cell_lat_min'], row['cell_lat_max']
            lon0, lon1 = row['cell_lon_min'], row['cell_lon_max']
            half_lat = (lat1 - lat0) / 2.0
            half_lon = (lon1 - lon0) / 2.0
            rs = np.flatnonzero(np.abs(lat_centers - (lat0 + lat1) / 2) < half_lat + 0.25 - 1e-10)
            delta = (lon_centers - (lon0 + lon1) / 2 + 180) % 360 - 180
            cs = np.flatnonzero(np.abs(delta) < half_lon + 0.25 - 1e-10)
            owners[np.ix_(rs, cs)] = index
        self._cluster_occupancy = owners >= 0
        rr, cc = np.nonzero(self._cluster_occupancy)
        if not len(rr):
            return
        a = rr * (cols + 1) + cc
        quads = np.column_stack((a, a + cols + 1, a + cols + 2, a + 1)).astype(np.int64)
        cells = self.vtk.vtkCellArray()
        cells.SetData(numpy_to_vtkIdTypeArray(np.arange(0, quads.size + 1, 4, dtype=np.int64), deep=True),
                      numpy_to_vtkIdTypeArray(quads.ravel(), deep=True))
        data = self.vtk.vtkPolyData()
        data.SetPoints(self.mesh.GetPoints())
        data.SetPolys(cells)
        indices = owners[rr, cc]
        palette = np.asarray([np.rint(np.asarray(cluster_color(row.get('cluster_index', 0))) * 255)
                              for row in results], dtype=np.uint8)
        data.GetCellData().SetScalars(numpy_to_vtk(np.ascontiguousarray(palette[indices]), deep=True,
                                                   array_type=self.vtk.VTK_UNSIGNED_CHAR))
        mapper = self.vtk.vtkPolyDataMapper()
        mapper.SetInputData(data)
        mapper.SetScalarModeToUseCellData()
        mapper.SetColorModeToDirectScalars()
        self.configure_surface_overlay(mapper)
        mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(0.0, -3.0)
        actor = self.vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().LightingOff()
        actor.GetProperty().EdgeVisibilityOff()
        actor.GetProperty().SetOpacity(0.35)
        actor.SetPickable(True)
        self.renderer.AddActor(actor)
        self._search_fill_actor = actor
        self._search_fill_rows = [results[int(i)] for i in indices]
        picker = self.vtk.vtkCellPicker()
        picker.SetTolerance(0.0001)
        picker.PickFromListOn()
        picker.AddPickList(actor)
        self._search_fill_picker = picker

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

    def pick_search_marker(self, x, y, extend=False):
        if self._terrain_view:
            return False
        if self._search_fill_picker is not None and self._search_fill_picker.Pick(x, y, 0, self.renderer):
            index = self._search_fill_picker.GetCellId()
            if 0 <= index < len(self._search_fill_rows):
                point = np.asarray(self._search_fill_picker.GetPickPosition())
                camera = np.asarray(self.renderer.GetActiveCamera().GetPosition())
                hit = self.surface_hit(camera, point - camera)
                if hit is not None and np.linalg.norm(hit - point) <= 0.002:
                    self.begin_visibility_search()
                    if self._visibility_dialog is not None:
                        self._visibility_dialog.select_sector_on_globe(self._search_fill_rows[index])
                    return True
        if self._search_point_picker is None:
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
        if row.get('kind') == 'sector':
            self.begin_visibility_search()
            if self._visibility_dialog is not None:
                self._visibility_dialog.select_sector_on_globe(row)
            return True
        self.select_location(row['lat'], row['lon'], render=False, extend_selection=extend)
        self._search_selected = row
        self._stats_base += "\nSearch point selected; press I to analyze its interval"
        self.update_stats()
        self.window.Render()
        return True

    def toggle_terrain_view(self, render=True):
        GlobeMixin.toggle_terrain_view(self, render=False)
        for actor in (self._search_actor, self._search_outline_actor, self._search_fill_actor):
            if actor is not None:
                actor.SetVisibility(not self._terrain_view)
        if render:
            self.window.Render()

    def begin_coordinate_entry(self, render=True):
        if self._desktop_shell is not None:
            self._desktop_shell.focus_coordinates()

    def begin_time_entry(self):
        if self._desktop_shell is not None:
            self._desktop_shell.show_panel('time')

    def begin_eye_height_entry(self):
        if self._desktop_shell is not None:
            self._desktop_shell.edit_eye_height()

    def set_eye_height(self, value):
        """Change the observer eye height (m) and recalculate everything that depends on it."""
        old = terrain.EYE_HEIGHT_M
        new = terrain.set_eye_height(value)  # ValueError: shown by the dialog
        if abs(new - old) < 1e-12:
            return
        self._analysis_snapshot = None
        self._analysis_can_reopen = False
        if self._compare_active and self._graph_expanded:
            if self._desktop_shell is None:
                self.request_sky_update()  # the worker recomputes every point's horizon and Sun / Earth data
        else:
            selected = self._search_selected  # select_location forgets it; the eye height does not change which point it is
            self.select_location(self.lat, self.lon, render=False)
            self._search_selected = selected
        if self._terrain_view:
            self.place_terrain_camera()

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

    def select_location(self, lat, lon, render=True, extend_selection=False):
        self._observer_suppressed = False
        if not self._terrain_view:
            self.observer.SetVisibility(True)
        from vtk.util.numpy_support import numpy_to_vtk

        self._search_selected = None
        self.lat, self.lon = float(lat), float(lon)
        point = normalize_point(lat, lon)
        if not extend_selection:
            self._selected_locations = [point]
        elif not any(same_point(point, other) for other in self._selected_locations):
            self._selected_locations.append(point)
        self.refresh_selection_marks()
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
        self._selected_stats = (float(sample_dem_bilinear(lat, lon)), compute_slope_deg(lat, lon),
                                float(np.min(angles)), float(np.max(angles)))
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

    def horizon_resize_edge(self, x, y):
        if self._graph_expanded or self._terrain_view or not self._overlay_settings['horizon_inset']:
            return ''
        width, height = self.window.GetSize()
        left, bottom, right, top = self._horizon_preview_rect
        nx, ny = x / max(1, width), y / max(1, height)
        margin_x, margin_y = 7 / max(1, width), 7 / max(1, height)
        if not (left-margin_x <= nx <= right+margin_x and bottom-margin_y <= ny <= top+margin_y):
            return ''
        horizontal = 'l' if abs(nx-left) <= margin_x else ('r' if abs(nx-right) <= margin_x else '')
        vertical = 'b' if abs(ny-bottom) <= margin_y else ('t' if abs(ny-top) <= margin_y else '')
        return horizontal + vertical

    def resize_horizon_preview(self, edge, original, press, current):
        width, height = self.window.GetSize()
        dx = (current[0] - press[0]) / max(1, width)
        dy = (current[1] - press[1]) / max(1, height)
        left, bottom, right, top = original
        min_width, min_height = min(0.8, 240 / max(1, width)), min(0.8, 150 / max(1, height))
        if 'l' in edge: left = max(0.0, min(right - min_width, left + dx))
        if 'r' in edge: right = min(1.0, max(left + min_width, right + dx))
        if 'b' in edge: bottom = max(0.0, min(top - min_height, bottom + dy))
        if 't' in edge: top = min(1.0, max(bottom + min_height, top + dy))
        self._horizon_preview_rect = left, bottom, right, top
        self.apply_layout()
        self.update_profile(self.horizon_angles)
        self.window.Render()

    def build_stats_base(self, lat, lon, angles, distances, heading="SELECTED LOCATION"):
        elevation = float(sample_dem_bilinear(lat, lon))
        return (
            f"{heading}\n"
            f"Latitude {lat:+.3f} deg    Longitude {lon:+.3f} deg\n"
            f"Elevation {elevation:+,.0f} m    Slope {compute_slope_deg(lat, lon):.1f} deg\n"
            f"Horizon {np.min(angles):+.2f} to {np.max(angles):+.2f} deg\n"
            f"Horizon distance {np.min(distances):.1f} to {np.max(distances):.1f} km")

    # ------------------------------------------------------------------ point comparison

    def sync_compare_points(self):
        """Analyze the full saved list while retaining visibility by point."""
        single = getattr(self, '_analysis_single_point', None)
        override = self._analysis_points_override
        points = list(override) if override is not None else ([single] if single is not None else list(self._points) or [(self.lat, self.lon)])
        if points != self._compare_points:
            previous = list(zip(self._compare_points, self._compare_visible))
            self._compare_visible = [next((visible for old, visible in previous if same_point(old, point)),
                                          point not in self._point_off) for point in points]
            self._compare_points = points
            self._compare_index = min(self._compare_index, len(points) - 1)
            self.apply_layout()

    def analysis_points(self):
        return [point for point in self._points if point not in self._point_off]

    def point_index(self, lat, lon):
        return next((i for i, point in enumerate(self._points) if same_point(point, (lat, lon))), None)

    def compare_color(self, index):
        from compare import POINT_COLORS
        if 0 <= index < len(self._compare_points):
            saved = self.point_index(*self._compare_points[index])
            if saved is not None:
                index = saved
        return POINT_COLORS[index % len(POINT_COLORS)]

    def set_point_name(self, point, name):
        name = name.strip()
        if name:
            self._point_names[point] = name
        else:
            self._point_names.pop(point, None)

    def set_point_included(self, point, included):
        if included:
            self._point_off.discard(point)
        else:
            self._point_off.add(point)
        self.points_changed(render=False)
        self.window.Render()

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
        points = list(self._points)
        for point in self._selected_locations or [(self.lat, self.lon)]:
            if not any(same_point(point, other) for other in points):
                points.append(point)
        if len(points) > MAX_POINTS:
            self._points_note = f'A set holds at most {MAX_POINTS} points.'
            if self._desktop_shell is not None:
                self._desktop_shell.card.setToolTip(self._points_note)
            self.window.Render()
            return
        self.set_points(points)

    def refresh_selection_marks(self):
        """Show additional unsaved selections without modifying the saved point set."""
        if self._selection_actor is not None:
            self.renderer.RemoveActor(self._selection_actor)
            self._selection_actor = None
        extra = [p for p in self._selected_locations if not same_point(p, (self.lat, self.lon))]
        if not extra:
            return
        points = self.vtk.vtkPoints()
        vertices = self.vtk.vtkCellArray()
        for lat, lon in extra:
            position = self.terrain_overlay_points(np.asarray(lat_lon_to_xyz(lat, lon, offset=0.0)))[0]
            index = points.InsertNextPoint(*position)
            vertices.InsertNextCell(1); vertices.InsertCellPoint(index)
        data = self.vtk.vtkPolyData(); data.SetPoints(points); data.SetVerts(vertices)
        mapper = self.vtk.vtkPolyDataMapper(); mapper.SetInputData(data)
        self.configure_surface_overlay(mapper)
        actor = self.vtk.vtkActor(); actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1.0, 0.5, 0.18); actor.GetProperty().SetPointSize(10)
        actor.GetProperty().LightingOff(); actor.PickableOff()
        self.renderer.AddActor(actor); self._selection_actor = actor

    def add_point_at(self, x, y):
        """Add a temporary selection without changing the saved point set."""
        if self._graph_expanded or self._terrain_view:
            return
        location = self.surface_location_at(x, y)
        if location is not None:
            self.select_location(*location, extend_selection=True)

    def set_points(self, points, names=None):
        points = list(points)
        if names is not None:
            self._point_names = dict(names)
            self._point_off.clear()
        self._point_names = {point: name for point, name in self._point_names.items() if point in points}
        self._point_off &= set(points)
        if points == self._points:
            if self._desktop_shell is not None:
                self._desktop_shell.refresh_points()
            return
        removed = [old for old in self._points if not any(same_point(old, current) for current in points)]
        self._points = points
        if removed:
            def was_removed(point):
                return any(same_point(point, old) for old in removed)
            self._selected_locations = [point for point in self._selected_locations if not was_removed(point)]
            self.refresh_selection_marks()
            if was_removed((self.lat, self.lon)):
                self._observer_suppressed = True
                self.observer.SetVisibility(False)
            if self._search_actor is not None:
                remaining = [row for row in self._search_rows if not was_removed((row['lat'], row['lon']))]
                self.show_visibility_markers(remaining, mode='Points')
            if self._search_selected is not None and was_removed((self._search_selected['lat'], self._search_selected['lon'])):
                self._search_selected = None
        self._points_note = ""
        self.points_changed()

    def points_changed(self, render=True):
        self.refresh_point_marks()
        if self._desktop_shell is not None:
            self._desktop_shell.refresh_points()
        if self._compare_active and self._desktop_shell is None:
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
        if self._desktop_shell is not None and not self._terrain_view:
            self._desktop_shell.show_panel('points')

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

    def update_close_hover(self, x, y):
        width, height = self.window.GetSize()
        hit = None
        for actor in (self._horizon_close, self.close_button):
            if not actor.GetVisibility():
                continue
            position = actor.GetPosition(); size = [0.0, 0.0]
            actor.GetSize(self.renderer, size)
            if position[0]*width-6 <= x <= position[0]*width+size[0]+6 and position[1]*height-6 <= y <= position[1]*height+size[1]+6:
                hit = actor
                break
        if hit is self._hovered_close:
            return
        if self._hovered_close is not None:
            prop = self._hovered_close.GetTextProperty()
            prop.SetColor(0.92, 0.96, 1.0); prop.SetBackgroundOpacity(0.08)
        self._hovered_close = hit
        if hit is not None:
            prop = hit.GetTextProperty()
            prop.SetColor(1.0, 1.0, 1.0); prop.SetBackgroundColor(0.25, 0.38, 0.52)
            prop.SetBackgroundOpacity(0.95)
        self.window.Render()

    def surface_location_at(self, x, y):
        if not self.picker.Pick(x, y, 0, self.renderer):
            return None
        px, py, pz = self.picker.GetPickPosition()
        radius = np.sqrt(px * px + py * py + pz * pz)
        if radius <= 0:
            return None
        return (float(np.degrees(np.arcsin(np.clip(pz / radius, -1, 1)))),
                float(np.degrees(np.arctan2(py, px))))

    def pick_saved_point(self, x, y, extend=False):
        width, height = self.window.GetSize()
        camera = np.asarray(self.renderer.GetActiveCamera().GetPosition())
        closest = None
        for index, mark in enumerate(self.point_marks):
            if not mark['point'].GetVisibility():
                continue
            position = mark['position']
            self.renderer.SetWorldPoint(*position, 1.0); self.renderer.WorldToDisplay()
            px, py, _ = self.renderer.GetDisplayPoint()
            distance = np.hypot(px - x, py - y)
            if distance > 15 or (closest is not None and distance >= closest[0]):
                continue
            hit = self.surface_hit(camera, position - camera)
            if hit is not None and np.linalg.norm(hit - position) <= 0.003:
                closest = (distance, index)
        if closest is None:
            return False
        self.select_location(*self._points[closest[1]], extend_selection=extend)
        return True

    def select_search_rectangle(self, start, end, commit=False):
        first = self.surface_location_at(*start)
        last = self.surface_location_at(*end)
        if first is None or last is None:
            return
        lat0, lat1 = sorted((first[0], last[0]))
        lon_delta = (last[1] - first[1] + 180.0) % 360.0 - 180.0
        lon0, lon1 = sorted((first[1], first[1] + lon_delta))
        if lat1 - lat0 < 1e-5 or lon1 - lon0 < 1e-5:
            return
        self.draw_search_area((lat0, lat1, lon0, lon1))
        if commit and self._desktop_shell is not None:
            panel = self._desktop_shell.search
            for field, value in ((panel.lat_min, lat0), (panel.lat_max, lat1),
                                 (panel.lon_min, lon0), (panel.lon_max, lon1)):
                field.setText(f'{value:.6f}'.rstrip('0').rstrip('.'))
            self._desktop_shell.finish_area_tool()
            self._desktop_shell.show_panel('search')
            panel.criteria_scroll.ensureWidgetVisible(panel.area_box)

    def refresh_search_area_mask(self):
        actor = getattr(self, '_area_fill_actor', None)
        bounds = getattr(self, '_area_bounds', None)
        if actor is not None and bounds is not None and actor.GetVisibility():
            self._area_bounds = None
            self.draw_search_area(bounds)

    def draw_search_area(self, bounds):
        """Draw the selected lat/lon area on the globe using the existing surface mesh."""
        from vtk.util.numpy_support import numpy_to_vtkIdTypeArray
        lat0, lat1, lon0, lon1 = map(float, bounds)
        if not (np.isfinite([lat0, lat1, lon0, lon1]).all() and -90 <= lat0 < lat1 <= 90 and lon0 < lon1):
            self.hide_search_area()
            return
        lon1 = min(lon1, lon0 + 360.0)
        bounds = (lat0, lat1, lon0, lon1)
        if bounds == self._area_bounds and self._area_fill_actor is not None:
            self._area_fill_actor.VisibilityOn()
            if self._rectangle_actor is not None: self._rectangle_actor.VisibilityOn()
            self.window.Render()
            return
        n = max(48, min(720, int(np.ceil(max(lat1 - lat0, lon1 - lon0) * 2))))
        lats = np.concatenate((np.full(n, lat0), np.linspace(lat0, lat1, n),
                               np.full(n, lat1), np.linspace(lat1, lat0, n), [lat0]))
        lons = np.concatenate((np.linspace(lon0, lon1, n), np.full(n, lon1),
                               np.linspace(lon1, lon0, n), np.full(n, lon0), [lon0]))
        latr, lonr = np.radians(lats), np.radians(lons)
        directions = np.column_stack((np.cos(latr)*np.cos(lonr), np.cos(latr)*np.sin(lonr), np.sin(latr)))
        coords = self.terrain_overlay_points(directions)
        if self._rectangle_actor is not None:
            self.renderer.RemoveActor(self._rectangle_actor)
        self._rectangle_actor = self.make_graticule_actor((0.16, 0.9, 1.0), 3.0, 1.0, -4.0)
        self._rectangle_actor.GetMapper().SetInputData(self.polylines_to_polydata([coords]))
        self._rectangle_actor.VisibilityOn()
        if self._area_fill_actor is not None:
            self.renderer.RemoveActor(self._area_fill_actor)
        rows, cols = 360, 720
        lat_centers = 90.0 - (np.arange(rows) + 0.5) * 0.5
        lon_centers = -180.0 + (np.arange(cols) + 0.5) * 0.5
        rr = np.flatnonzero((lat_centers > lat0 - .25 + 1e-10) & (lat_centers < lat1 + .25 - 1e-10))
        delta = (lon_centers - (lon0 + lon1) / 2 + 180) % 360 - 180
        cc = np.flatnonzero(np.abs(delta) < (lon1 - lon0) / 2 + .25 - 1e-10)
        grid_r, grid_c = np.meshgrid(rr, cc, indexing='ij')
        occupancy = getattr(self, '_cluster_occupancy', None)
        if occupancy is not None:
            keep = ~occupancy[grid_r, grid_c]
            grid_r, grid_c = grid_r[keep], grid_c[keep]
        base = grid_r.ravel() * (cols + 1) + grid_c.ravel()
        quads = np.column_stack((base, base + cols + 1, base + cols + 2, base + 1)).astype(np.int64)
        cells = self.vtk.vtkCellArray()
        cells.SetData(numpy_to_vtkIdTypeArray(np.arange(0, quads.size + 1, 4, dtype=np.int64), deep=True),
                      numpy_to_vtkIdTypeArray(quads.ravel(), deep=True))
        data = self.vtk.vtkPolyData(); data.SetPoints(self.mesh.GetPoints()); data.SetPolys(cells)
        mapper = self.vtk.vtkPolyDataMapper(); mapper.SetInputData(data)
        self.configure_surface_overlay(mapper)
        mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(0.0, -2.0)
        fill = self.vtk.vtkActor(); fill.SetMapper(mapper); fill.GetProperty().SetColor(0.16, 0.70, 0.85)
        fill.GetProperty().SetOpacity(0.25); fill.GetProperty().LightingOff(); fill.PickableOff()
        self.renderer.AddActor(fill); self._area_fill_actor = fill; self._area_bounds = bounds
        self.window.Render()

    def hide_search_area(self):
        changed = False
        for actor in (self._rectangle_actor, self._area_fill_actor):
            if actor is not None and actor.GetVisibility():
                actor.VisibilityOff(); changed = True
        if changed: self.window.Render()

    def pick(self, x, y, select_new=False, extend=False):
        width, height = self.window.GetSize()
        if self._horizon_map_button.GetVisibility():
            px, py = self._horizon_map_button.GetPosition()
            size = [0.0, 0.0]
            self._horizon_map_button.GetSize(self.renderer, size)
            if px * width - size[0] - 5 <= x <= px * width + 5 and py * height - 5 <= y <= py * height + size[1] + 5:
                self.cycle_horizon_map_mode()
                return
        if self._desktop_shell is None and not self._graph_expanded and 0.73 <= x / width <= 0.87 and 0.94 <= y / height <= 1.0:
            self.begin_overlay_settings()
            return
        if self.over_sky_controls(x, y):
            if self._compare_active and self._compare is not None and self._table_pool:
                slots, _ = self.table_layout(self.compare_n())
                if len(self._table_pool) >= len(slots):
                    actor = self._table_pool[len(slots) - 1]
                    pos = actor.GetPosition(); size = [0.0, 0.0]
                    actor.GetSize(self.renderer, size)
                    if actor.GetVisibility() and pos[0]*width <= x <= pos[0]*width+size[0] and pos[1]*height-size[1] <= y <= pos[1]*height:
                        self._table_notes_expanded = not self._table_notes_expanded
                        self.update_table(); self.apply_layout(); self.window.Render()
                        return
            if self.table_header_click(x, y):  # sort a comparison table by the clicked column
                return
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
                    if name.startswith('notes:'):
                        key = name.split(':', 1)[1]
                        self._graph_notes_open[key] = not self._graph_notes_open[key]
                        self.apply_layout()
                        self.window.Render()
                    elif name.startswith("vis:"):
                        self.toggle_point_visibility(int(name.rsplit(":", 1)[1]))
                    else:
                        actions[name]()
                    break
            return
        if not select_new:
            if self.pick_saved_point(x, y, extend=extend) or self.pick_search_marker(x, y, extend=extend):
                return
            return
        location = self.surface_location_at(x, y)
        if location is not None:
            self.select_location(*location, extend_selection=extend)

    def run(self, qt_app=None):
        if self._desktop_shell is not None:
            self._desktop_shell.show()
        self.interactor.Initialize()
        self._sky_pump_observer = self.interactor.AddObserver("TimerEvent", self.poll_sky)
        self._sky_timer = self.interactor.CreateRepeatingTimer(100)
        self.poll_sky()
        self.window.Render()
        # Handle the window-manager close button (Escape no longer quits).
        exit_observer = self.interactor.AddObserver(
            "ExitEvent", lambda obj, event: obj.SetDone(True))
        previous_sigint = signal.signal(signal.SIGINT, signal.SIG_DFL)
        try:
            if qt_app is not None:
                qt_app.exec() if hasattr(qt_app, 'exec') else qt_app.exec_()
            else:
                run_responsive_event_loop(self.interactor)
        finally:
            signal.signal(signal.SIGINT, previous_sigint)
            self.interactor.RemoveObserver(exit_observer)
            self._sky_stop.set()
            if self._sky_timer is not None:
                self.interactor.DestroyTimer(self._sky_timer)
            if self._sky_pump_observer is not None:
                self.interactor.RemoveObserver(self._sky_pump_observer)
            if self._visibility_dialog is not None:
                self._visibility_dialog.on_close = None
                self._visibility_dialog.close()
                self._visibility_dialog = None
            self.window.Finalize()


def main():
    parser = argparse.ArgumentParser(description="Native Moon terrain and horizon viewer")
    parser.add_argument("--texture", help="Optional path to a Moon color texture")
    arguments = parser.parse_args()
    try:
        from desktop_ui import MainShell, create_application
        app = create_application()
        load_data(arguments.texture)
        shell = MainShell()
        render_window = shell.vtk_widget.GetRenderWindow()
        window = MoonWindow(render_window=render_window, interactor=render_window.GetInteractor(), desktop_shell=shell)
        shell.bind_owner(window)
        window.run(qt_app=app)
    except ImportError as error:
        print(f"Missing desktop dependency: {error}\n"
              "Install with: python -m pip install numpy Pillow vtk skyfield PySide6", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"Moon Horizon could not start: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
