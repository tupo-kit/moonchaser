"""Screen-space HUD for the Moon viewer: horizon inset, fullscreen Sun/Earth graphs, slider, scroll bar.

HudMixin is mixed into moon.MoonWindow and expects the host to provide: self.vtk, self.renderer,
self.window, self.lat/lon, self.horizon_angles, the time/sky state (_epochs, _time_index, _sky_data,
_sky_status, _graph_expanded, _scroll, _now_mode, _sky_method, _shown_sources, _legend_expanded,
_interval_step_minutes, _stats_base) and _panorama_*. _init_hud() builds every actor and must run
after the renderer exists. All overlay coordinates are normalized viewport (x right, y up).
Pure helpers (split_at_zero, dash_polyline, dim_color) have no VTK dependency.

Point comparison (see compare.py): while self._compare_active, the page shows the horizon graph of one
point (arrow buttons switch it), Sun/Earth graphs that combine every point (one colour per point, one
on/off button per point) and, below them, the comparison tables. Extra host state: _compare_active,
_compare_points, _compare_index, _compare_visible, _compare (results), _stats_above_only.
"""
from datetime import timedelta

import numpy as np

from compare import POINT_COLORS, build_tables, table_notes
from sky import SKY_METHODS
from terrain import terrain_panorama_rgba, terrain_panorama_samples

SKY_LEGEND = {"JPL": "JPL Sun yellow / Earth blue", "Skyfield": "Skyfield Sun orange / Earth green"}
SKY_COLORS = {"Sun JPL": (1.0, 0.8, 0.12), "Earth JPL": (0.35, 0.65, 1.0), "Sun Skyfield": (1.0, 0.35, 0.1), "Earth Skyfield": (0.3, 1.0, 0.5)}
PANEL_HEIGHT = 0.5  # each extra graph is half a window tall; the page scrolls under the window
EXTRA_PANELS = ("Sun", "Earth")
PAGE_HEIGHT = 1.0 + PANEL_HEIGHT * len(EXTRA_PANELS)
SCROLL_STEP = 0.08
TABLE_ROW = 0.034      # comparison table: one text row, in window heights
TOGGLES_PER_ROW = 6    # comparison graphs: point on/off buttons per row

BELOW_DASH_PX, BELOW_GAP_PX = 7.0, 5.0
BACKGROUND_RGB = (0.025, 0.035, 0.055)


def dim_color(color, factor=0.6):
    """Blend a colour towards the dark background (used for 'below the horizon')."""
    return tuple(factor * c + (1.0 - factor) * b for c, b in zip(color, BACKGROUND_RGB))


def split_at_zero(points, values):
    """Split a polyline where `values` changes sign, adding the exact crossing point.

    Returns [(points, below)] where below is True for the parts with values < 0.
    """
    points, values = np.asarray(points, dtype=float), np.asarray(values, dtype=float)
    if len(points) < 2:
        return []
    below = values < 0
    pieces, start, start_point = [], 0, None
    for i in np.flatnonzero(below[1:] != below[:-1]):
        v0, v1 = values[i], values[i + 1]
        crossing = points[i] + (v0 / (v0 - v1)) * (points[i + 1] - points[i])
        piece = points[start:i + 1]
        if start_point is not None:
            piece = np.vstack((start_point, piece))
        pieces.append((np.vstack((piece, crossing)), bool(below[i])))
        start, start_point = i + 1, crossing
    piece = points[start:]
    if start_point is not None:
        piece = np.vstack((start_point, piece))
    pieces.append((piece, bool(below[-1])))
    return [(piece, flag) for piece, flag in pieces if len(piece) > 1]


def dash_polyline(points, dash, gap, scale=(1.0, 1.0)):
    """Cut a polyline into dashes. dash/gap are in pixels when scale = (width, height)."""
    scale = np.asarray(scale, dtype=float)
    pts = np.asarray(points, dtype=float) * scale
    if len(pts) < 2:
        return []
    arc = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))))
    total = float(arc[-1])
    if total <= 0.0:
        return []
    starts = np.arange(0.0, total, float(dash + gap))
    ends = np.minimum(starts + dash, total)

    def at(distance):
        return np.array([[np.interp(distance, arc, pts[:, 0]), np.interp(distance, arc, pts[:, 1])]])

    pieces = []
    for a, b in zip(starts, ends):
        first, last = np.searchsorted(arc, [a, b], side="right")[0], np.searchsorted(arc, b, side="left")
        pieces.append(np.vstack((at(a), pts[first:last], at(b))) / scale)
    return [piece for piece in pieces if len(piece) > 1]


class HudMixin:
    GLOBE_HELP = ("Left-drag: orbit   |   Middle/Shift-drag: pan\n"
                  "Ctrl-drag: orbit cursor point   |   Wheel: zoom   |   Click: select\n"
                  "G: coordinates   |   T: time interval (2+ points: compare)   |   V: terrain view\n"
                  "R: reset   |   F: fullscreen   |   E: eye height\n"
                  "Cyan: local horizon   |   Orange: observer   |   Relief: 3x\n"
                  "L: lat/lon grid (auto detail)   |   Red N / blue S: poles\n"
                  "Shift+click / A: add comparison point   |   P: manage, save, load points")

    def _init_hud(self):
        """Create every HUD actor (stats, help, horizon inset, graphs, slider, scroll bar) and lay them out."""
        self.stats = self.make_text(0.025, 0.97, 19, top=True)
        self.help = self.make_text(0.025, 0.035, 15)
        self.help.SetInput(self.GLOBE_HELP)
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
        # Drawn bottom to top: level line, below-horizon parts (dim, dashed), above-horizon parts, selected sample.
        self.sky_hlines = {name: self.make_overlay_lines([], dim_color(color, 0.8), 1) for name, color in SKY_COLORS.items()}
        self.sky_trails_below = {name: self.make_overlay_lines([], dim_color(color), 1) for name, color in SKY_COLORS.items()}
        self.sky_trails = {name: self.make_overlay_lines([], color, 2) for name, color in SKY_COLORS.items()}
        self.sky_samples_below = {name: self.make_overlay_points([], dim_color(color), 3) for name, color in SKY_COLORS.items()}
        self.sky_samples = {name: self.make_overlay_points([], color, 4) for name, color in SKY_COLORS.items()}
        self.sky_markers = {name: self.make_bullseye(color) for name, color in SKY_COLORS.items()}
        self.sky_values = {name: self.make_value_label(color) for name, color in SKY_COLORS.items()}
        self.sky_labels = {name: self.make_text(0.61, 0.26, 14) for name in SKY_COLORS}
        for name, actor in self.sky_labels.items():
            actor.GetTextProperty().SetColor(*SKY_COLORS[name])
            actor.GetTextProperty().SetBold(True)
            actor.SetVisibility(False)
        self.source_buttons = {source: self.make_text(0.64, 0.7, 15) for source in ("Skyfield", "JPL")}
        self.legend_button = self.make_text(0.025, 0.98, 16, top=True)
        self.progress_text = self.make_text(0.35, 0.025, 16)
        self.progress_text.SetVisibility(False)
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
        self.panel_hlines = {name: self.make_overlay_lines([], dim_color(color, 0.8), 1) for name, color in SKY_COLORS.items()}
        self.panel_markers = {name: self.make_bullseye(color) for name, color in SKY_COLORS.items()}
        self.panel_values = {name: self.make_value_label(color) for name, color in SKY_COLORS.items()}
        self._panel_state = {}
        # Time slider under the horizon graph (mouse: click or drag it).
        self._slider_box = None
        self.slider_track = self.make_overlay_lines([], (0.4, 0.48, 0.58), 4)
        self.slider_fill = self.make_overlay_lines([], (0.0, 0.9, 1.0), 4)
        self.slider_handle = self.make_bullseye((0.0, 0.9, 1.0))
        self.slider_texts = {key: self.make_text(0.0, 0.0, 12) for key in ("start", "current", "end")}
        self.slider_texts["current"].GetTextProperty().SetJustificationToCentered()
        self.slider_texts["end"].GetTextProperty().SetJustificationToRight()
        for actor in self.slider_texts.values():
            actor.SetVisibility(False)
        # Point comparison: point selector, per-point graph actors (created on demand), table text pool.
        self.point_prev_button = self.make_text(0.64, 0.93, 16)
        self.point_prev_button.SetInput("[ < ]")
        self.point_label = self.make_text(0.69, 0.93, 16)
        self.point_label.GetTextProperty().SetBold(True)
        self.point_next_button = self.make_text(0.79, 0.93, 16)
        self.point_next_button.SetInput("[ > ]")
        for actor in (self.point_prev_button, self.point_label, self.point_next_button):
            actor.SetVisibility(False)
        self._cmp_lines, self._cmp_markers, self._cmp_toggles, self._cmp_state = {}, {}, {}, {}
        self._table_pool = []
        self._scrollbar_grab = 0.0
        self.scrollbar_track = self.make_overlay_lines([], (0.16, 0.2, 0.28), 8)
        self.scrollbar = self.make_overlay_lines([], (0.55, 0.65, 0.8), 8)
        self.apply_layout()

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
        width, height = self.window.GetSize()
        bearings = max(360, int(np.ceil((x1 - x0) * width)))
        if bearings > self._panorama_samples[0].shape[0]:
            self._panorama_samples = terrain_panorama_samples(
                self.lat, self.lon, self._panorama_texture_rgb, bearings=bearings)
            self._panorama_range = None
        rows = max(2, int(np.ceil((y1 - y0) * height)))
        limits = (self._profile_low, self._profile_high, rows)
        if limits != self._panorama_range:
            from vtk.util.numpy_support import numpy_to_vtk
            rgba = terrain_panorama_rgba(self._panorama_samples, *limits)
            self._panorama_image.SetDimensions(rgba.shape[1], rgba.shape[0], 1)
            colors = numpy_to_vtk(rgba.reshape(-1, 4), deep=True, array_type=self.vtk.VTK_UNSIGNED_CHAR)
            colors.SetNumberOfComponents(4)
            self._panorama_image.GetPointData().SetScalars(colors)
            self._panorama_image.Modified()
            self._panorama_range = limits

    def make_bullseye(self, color):
        """Selected-sample marker: white ring, dark ring, then the body colour in the middle."""
        return tuple(self.make_overlay_points([], ring_color, size)
                     for ring_color, size in (((1.0, 1.0, 1.0), 20), ((0.02, 0.03, 0.05), 16), (color, 11)))

    def set_bullseye(self, bullseye, coordinates):
        data = self.overlay_point_data(coordinates)
        for actor in bullseye:
            actor.GetMapper().SetInputData(data)

    def dashed_pieces(self, pieces):
        width, height = self.window.GetSize()
        scale = (max(width, 1), max(height, 1))
        return [dash for piece in pieces for dash in dash_polyline(piece, BELOW_DASH_PX, BELOW_GAP_PX, scale)]

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
        compare = expanded and self._compare_active
        if expanded:
            self._scroll = min(self._scroll, self.page_height() - 1.0)
        s = self._scroll if expanded else 0.0  # screen y = page y + s
        if expanded:
            self._plot_box = (0.06, 0.97, 0.10 + s, 0.62 + s)
            background = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
            bx = 0.64
            self._controls = {"now": (0.85, 0.92 + s, 0.905, 0.96 + s),
                              "close": (0.91, 0.92 + s, 0.985, 0.96 + s),
                              "previous": (bx, 0.79 + s, bx + 0.13, 0.825 + s),
                              "next": (bx + 0.16, 0.79 + s, bx + 0.28, 0.825 + s),
                              **{source: (bx, 0.625 + s, bx + 0.16, 0.66 + s) for source in SKY_METHODS[self._sky_method]},
                              "legend": (0.02, 0.95 + s, 0.30, 0.995 + s)}
            positions = {self.now_button: (0.85, 0.93 + s),
                         self.close_button: (0.91, 0.93 + s), self.time_caption: (bx, 0.89 + s),
                         self.previous_button: (bx, 0.795 + s), self.next_button: (bx + 0.16, 0.795 + s),
                         self.profile_title: (bx, 0.765 + s)}
            if compare:  # no "Now" in a comparison; the point selector takes the top row
                del self._controls["now"]
                self._controls["point_prev"] = (bx, 0.925 + s, bx + 0.045, 0.96 + s)
                self._controls["point_next"] = (0.79, 0.925 + s, 0.835, 0.96 + s)
                positions[self.point_prev_button] = (bx, 0.93 + s)
                positions[self.point_label] = (0.69, 0.93 + s)
                positions[self.point_next_button] = (0.79, 0.93 + s)
        else:
            self._plot_box = (0.61, 0.97, 0.12, 0.25)
            background = ((0.59, 0.055), (0.99, 0.055), (0.99, 0.35), (0.59, 0.35))
            self._controls = {"legend": (0.02, 0.95, 0.30, 0.995)}
            positions = {self.now_button: (0.90, 0.46),
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
        self.now_button.SetVisibility(expanded and not compare)
        for actor in (self.point_prev_button, self.point_label, self.point_next_button):
            actor.SetVisibility(compare)
        self.help.SetVisibility(not expanded)
        self.layout_panels(s)
        self._slider_box = (x0, x1, 0.04 + s) if expanded else None
        # Scroll bar (drag the thumb, or click the track): thumb length = visible fraction of the page.
        thumb, track = [], []
        if expanded:
            page = self.page_height()
            top = 0.98 - 0.96 * s / page
            thumb = [[(0.993, top), (0.993, top - 0.96 / page)]]
            track = [[(0.993, 0.98), (0.993, 0.02)]]
        self.scrollbar.GetMapper().SetInputData(self.overlay_lines_data(thumb))
        self.scrollbar_track.GetMapper().SetInputData(self.overlay_lines_data(track))
        self.update_slider()
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
        self.layout_compare(s)

    def compare_n(self):
        return len(self._compare_points) if self._compare_active and self._graph_expanded else 0

    def page_height(self):
        """Pages are 2 window heights (horizon + Sun + Earth); a comparison adds the tables."""
        if self._compare_active and self._graph_expanded:
            return PAGE_HEIGHT + self.table_layout(len(self._compare_points))[1]
        return PAGE_HEIGHT

    def ensure_compare_actors(self, n):
        for key in EXTRA_PANELS:
            for i in range(n):
                if (key, i) in self._cmp_lines:
                    continue
                color = POINT_COLORS[i % len(POINT_COLORS)]
                self._cmp_lines[(key, i)] = self.make_overlay_lines([], color, 2)
                self._cmp_markers[(key, i)] = self.make_bullseye(color)
                toggle = self.make_text(0.4, 0.0, 13)
                toggle.GetTextProperty().SetBold(True)
                toggle.SetVisibility(False)
                self._cmp_toggles[(key, i)] = toggle

    def layout_compare(self, s):
        """Position the point on/off buttons of the combined graphs and the comparison tables."""
        n = self.compare_n()
        self.ensure_compare_actors(n)
        for (key, i), toggle in self._cmp_toggles.items():
            active = i < n
            toggle.SetVisibility(active)
            if not active:
                continue
            base = -PANEL_HEIGHT * (EXTRA_PANELS.index(key) + 1) + s
            x = 0.40 + (i % TOGGLES_PER_ROW) * 0.095
            y = base + 0.45 - (i // TOGGLES_PER_ROW) * 0.03
            toggle.SetPosition(x, y)
            self._controls[f"vis:{key}:{i}"] = (x, y - 0.004, x + 0.092, y + 0.026)
        self.layout_table(s)

    def table_layout(self, n):
        """Text slots of the table section: y offsets down from its top, and the section height."""
        y, slots = 0.03, [0.03]  # slot 0: the "stats over" button
        y += 0.05
        for _ in range(2):  # slot per table: title, two header lines, then one slot per point
            slots.append(y)
            y += 0.045
            slots.append(y)
            y += 2 * TABLE_ROW + 0.005
            for _ in range(n):
                slots.append(y)
                y += TABLE_ROW
            y += 0.03
        slots.append(y)  # notes
        return slots, y + 0.18

    def table_actor(self, slot):
        while len(self._table_pool) <= slot:
            actor = self.make_text(0.03, 0.0, 12, top=True)
            actor.SetVisibility(False)
            self._table_pool.append(actor)
        return self._table_pool[slot]

    def layout_table(self, s):
        n = self.compare_n()
        if not n:
            for actor in self._table_pool:
                actor.SetVisibility(False)
            return
        slots, _ = self.table_layout(n)
        for slot, offset in enumerate(slots):
            self.table_actor(slot).SetPosition(0.03, s - 1.0 - offset)
        self._controls["stats_mode"] = (0.03, s - 1.0 - slots[0] - 0.035, 0.52, s - 1.0 - slots[0] + 0.005)

    def table_font(self, chars):
        width, height = self.window.GetSize()
        by_width = int(0.94 * width / (max(chars, 1) * 0.62))  # monospace advance is about 0.6 em
        by_height = int(TABLE_ROW * height / 1.5)
        return int(np.clip(min(by_width, by_height), 8, 16))

    def update_table(self):
        n = self.compare_n()
        if not n or self._compare is None or self._compare.get("stats") is None:
            for actor in self._table_pool:
                actor.SetVisibility(False)
            return
        slots, _ = self.table_layout(n)
        for slot in range(len(slots)):
            self.table_actor(slot)
        white = (0.92, 0.96, 1.0)
        compare = self._compare
        tables = build_tables(compare["points"], compare["stats"])
        notes = table_notes(compare["stats_span"], self._stats_above_only)
        chars = max(len(line) for _, header, rows in tables for line in (*header, *rows))
        size = self.table_font(chars)

        def put(slot, text, color, font_size, mono=True):
            actor = self.table_actor(slot)
            actor.SetInput(text)
            prop = actor.GetTextProperty()
            prop.SetColor(*color)
            prop.SetFontSize(font_size)
            (prop.SetFontFamilyToCourier if mono else prop.SetFontFamilyToArial)()
            actor.SetVisibility(True)

        put(0, "[ Stats over: " + ("samples above the horizon" if self._stats_above_only else "all samples")
            + " ]   (click to change)", white, 14, mono=False)
        for t, (title, header, rows) in enumerate(tables):
            base = 1 + t * (2 + n)
            put(base, f"{title}  ({compare['source']})", white, size)
            put(base + 1, "\n".join(header), white, size)
            for i, row in enumerate(rows):
                put(base + 2 + i, row, POINT_COLORS[i % len(POINT_COLORS)], size)
        put(len(slots) - 1, notes, (0.62, 0.7, 0.8), max(8, size - 1), mono=False)
        for actor in self._table_pool[len(slots):]:
            actor.SetVisibility(False)

    def toggle_stats_mode(self):
        self._stats_above_only = not self._stats_above_only
        self.compute_compare_stats()
        self.update_table()
        self.window.Render()

    def toggle_point_visibility(self, index):
        if 0 <= index < len(self._compare_visible):
            self._compare_visible[index] = not self._compare_visible[index]
            self.update_panels()
            self.window.Render()

    def clear_compare_view(self):
        """Hide everything that belongs to a comparison (used when it is not active)."""
        for actor in self._cmp_lines.values():
            actor.GetMapper().SetInputData(self.overlay_lines_data([]))
        for marker in self._cmp_markers.values():
            self.set_bullseye(marker, [])
        for toggle in self._cmp_toggles.values():
            toggle.SetVisibility(False)
        for actor in self._table_pool:
            actor.SetVisibility(False)
        self._cmp_state = {}

    def scroll_to(self, value):
        if not self._graph_expanded:
            return
        value = float(np.clip(value, 0.0, self.page_height() - 1.0))
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
            self.scroll_to(self.page_height() - 1.0)
        else:
            self.scroll_to(self._scroll + {"up": -SCROLL_STEP, "down": SCROLL_STEP,
                                           "prior": -0.8, "next": 0.8}[key])

    def update_panels(self):
        """Sun and Earth elevation graphs."""
        expanded = self._graph_expanded
        for panel in self.panels.values():
            for actor in (panel["title"], *panel["xlabels"], *panel["ylabels"]):
                actor.SetVisibility(expanded)
        for actor in (*self.panel_terrain.values(), *self.panel_lines.values()):
            actor.GetMapper().SetInputData(self.overlay_lines_data([]))
        self._panel_state = {}
        if expanded and self._compare_active:
            self.update_compare_panels()
            return
        self.clear_compare_view()
        if not expanded:
            self.update_panel_markers()
            return
        data, epochs = self._sky_data, self._epochs
        ready = data is not None and len(epochs) > 1
        if data is None:
            note = self._sky_status or "Waiting for Sun / Earth positions"
        else:
            note = "Needs an interval with 2 or more samples (press T)"

        def hide_labels(panel):
            for actor in (*panel["xlabels"], *panel["ylabels"]):
                actor.SetVisibility(False)

        def shown(name):
            return data is not None and name in data and self._shown_sources[name.rsplit(" ", 1)[-1]]

        seconds = np.array([(epoch - epochs[0]).total_seconds() for epoch in epochs])
        span = float(seconds[-1])
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
            values = {}
            for name in names:
                clearance = np.maximum(data[name]["el"] - data[name]["terrain"], 0.0)
                values[name] = clearance
                ys = y0 + (y1 - y0) * clearance / high
                self.panel_lines[name].GetMapper().SetInputData(
                    self.overlay_lines_data([np.column_stack((xs, ys))]))
            self._panel_state[key] = {"xs": xs, "high": high, "names": names, "values": values}
            panel["title"].SetInput(f"{key} centre above terrain horizon vs time (UTC)\n"
                                    "0 deg = terrain horizon; below horizon shown as 0\n"
                                    f"Scale 0 to {high:.1f} deg")
            for fraction, label in zip(np.linspace(0, 1, 5), panel["xlabels"]):
                label.SetInput(f"{epochs[0] + timedelta(seconds=float(fraction) * span):%m-%d %H:%M}")
                label.SetVisibility(True)
            for label, value in zip(panel["ylabels"], (low, high)):
                label.SetInput(f"{value:+.1f}")
                label.SetVisibility(True)
        self.update_panel_markers()

    def update_compare_panels(self):
        """Sun and Earth graphs that combine every point of the comparison (one colour per point)."""
        n = self.compare_n()
        self.ensure_compare_actors(n)
        for actor in self._cmp_lines.values():
            actor.GetMapper().SetInputData(self.overlay_lines_data([]))
        self._cmp_state = {}
        k = min(self._compare_index, max(n - 1, 0))
        self.point_label.SetInput(f"Point {k + 1}/{n}")
        self.point_label.GetTextProperty().SetColor(*POINT_COLORS[k % len(POINT_COLORS)])
        compare, epochs = self._compare, self._epochs
        ready = compare is not None and len(epochs) > 1
        if compare is None:
            note = self._sky_status or "Waiting for Sun / Earth positions"
        else:
            note = "Needs an interval with 2 or more samples (press T)"
        source = SKY_METHODS[self._sky_method][0]
        seconds = np.array([(epoch - epochs[0]).total_seconds() for epoch in epochs])
        span = float(seconds[-1]) if len(seconds) > 1 else 0.0
        for key in EXTRA_PANELS:
            panel, (x0, x1, y0, y1) = self.panels[key], self._panel_boxes[key]
            title = f"{key} centre above terrain horizon vs time (UTC), {n} points"
            name = f"{key} {source}"
            shown = []
            if ready and span > 0 and self._shown_sources[source]:
                shown = [i for i in range(n) if self._compare_visible[i] and compare["data"][i] is not None
                         and name in compare["data"][i]]
            if not shown:
                if not ready or span <= 0:
                    detail = note
                elif not self._shown_sources[source]:
                    detail = "Source hidden (see the checkbox)"
                else:
                    detail = "No point selected: use the buttons on the right"
                panel["title"].SetInput(title + "\n" + detail)
                for actor in (*panel["xlabels"], *panel["ylabels"]):
                    actor.SetVisibility(False)
                continue
            values = {i: np.maximum(compare["data"][i][name]["el"] - compare["data"][i][name]["terrain"], 0.0)
                      for i in shown}
            high = max(0.01, max(float(np.max(v)) for v in values.values()) * 1.08)
            xs = x0 + (x1 - x0) * seconds / span
            for i, clearance in values.items():
                ys = y0 + (y1 - y0) * clearance / high
                self._cmp_lines[(key, i)].GetMapper().SetInputData(
                    self.overlay_lines_data([np.column_stack((xs, ys))]))
            self._cmp_state[key] = {"xs": xs, "high": high, "values": values}
            panel["title"].SetInput(f"{title}\n0 deg = terrain horizon; below horizon shown as 0\n"
                                    f"Scale 0 to {high:.1f} deg")
            for fraction, label in zip(np.linspace(0, 1, 5), panel["xlabels"]):
                label.SetInput(f"{epochs[0] + timedelta(seconds=float(fraction) * span):%m-%d %H:%M}")
                label.SetVisibility(True)
            for label, value in zip(panel["ylabels"], (0.0, high)):
                label.SetInput(f"{value:+.1f}")
                label.SetVisibility(True)
        self.update_panel_markers()
        self.update_table()

    def update_compare_markers(self):
        """Selected sample: a marker per point on each combined graph, and the on/off buttons with values."""
        n = self.compare_n()
        for marker in self._cmp_markers.values():
            self.set_bullseye(marker, [])
        for key in EXTRA_PANELS:
            state = self._cmp_state.get(key)
            x0, x1, y0, y1 = self._panel_boxes[key]
            for i in range(n):
                color = POINT_COLORS[i % len(POINT_COLORS)]
                on = self._compare_visible[i]
                text = ("[x] " if on else "[ ] ") + f"P{i + 1}"
                if state is not None and i in state["values"]:
                    index = min(self._time_index, len(state["xs"]) - 1)
                    value = float(state["values"][i][index])
                    text += f" {value:5.1f}"
                    self.set_bullseye(self._cmp_markers[(key, i)], [
                        (float(state["xs"][index]), y0 + (y1 - y0) * value / state["high"])])
                toggle = self._cmp_toggles[(key, i)]
                toggle.SetInput(text)
                toggle.GetTextProperty().SetColor(*(color if on else dim_color(color)))
                toggle.SetVisibility(True)

    def make_value_label(self, color):
        """Degree readout drawn at the right edge of a graph, level with the selected sample."""
        label = self.make_text(0.9, 0.5, 13)
        prop = label.GetTextProperty()
        prop.SetColor(*color)
        prop.SetBold(True)
        prop.SetJustificationToRight()
        label.SetVisibility(False)
        return label

    def shown_body(self, name):
        if self._sky_data is None or not self._graph_expanded or not self._shown_sources[name.rsplit(" ", 1)[-1]]:
            return None
        return self._sky_data.get(name)

    def update_panel_markers(self):
        """Selected sample on the Sun/Earth graphs: a marker, a level line and its degrees at the edge."""
        for name in SKY_COLORS:
            self.set_bullseye(self.panel_markers[name], [])
            self.panel_hlines[name].GetMapper().SetInputData(self.overlay_lines_data([]))
            self.panel_values[name].SetVisibility(False)
        if self._compare_active and self._graph_expanded:
            self.update_compare_markers()
        for key, state in self._panel_state.items():
            x0, x1, y0, y1 = self._panel_boxes[key]
            index = min(self._time_index, len(state["xs"]) - 1)
            for name in state["names"]:
                value = float(state["values"][name][index])
                y = y0 + (y1 - y0) * value / state["high"]
                self.set_bullseye(self.panel_markers[name], [(float(state["xs"][index]), y)])
                self.panel_hlines[name].GetMapper().SetInputData(self.overlay_lines_data([[(x0, y), (x1, y)]]))
                label = self.panel_values[name]
                label.SetInput(f"{value:.1f} deg")
                label.SetPosition(x1 - 0.004, float(np.clip(y + 0.004, y0, y1 - 0.025)))
                label.SetVisibility(True)

    def update_profile_marker(self, name):
        """Selected sample on the horizon graph: a marker, a level line and its elevation at the edge."""
        x0, x1, y0, y1 = self._plot_box
        body = self.shown_body(name)
        marker, level = [], []
        label, value = self.sky_labels[name], self.sky_values[name]
        if body is not None:
            index = min(self._time_index, len(body["az"]) - 1)
            az, el = float(body["az"][index]), float(body["el"][index])
            px, py = x0 + (x1 - x0) * az / 360.0, float(self.profile_y(el))
            marker, level = [(px, py)], [[(x0, py), (x1, py)]]
            text = name + (" (partial)" if body["state"][index] == "Partly visible" else "")
            if el < float(body["terrain"][index]):
                text += " (below horizon)"
            label.SetInput(text)
            label.SetPosition(min(px + 0.012, x1 - (0.17 if "(" in text else 0.09)),
                              float(np.clip(py + (0.012 if name == "Sun" else -0.024), y0 + 0.01, y1 + 0.005)))
            value.SetInput(f"{el:+.1f} deg")
            value.SetPosition(x1 - 0.004, float(np.clip(py + 0.004, y0, y1 - 0.025)))
        self.sky_hlines[name].GetMapper().SetInputData(self.overlay_lines_data(level))
        self.set_bullseye(self.sky_markers[name], marker)
        label.SetVisibility(bool(marker))
        value.SetVisibility(bool(marker))

    def update_selection(self):
        """Cheap refresh after the selected sample changes (trails and scales stay as they are)."""
        for name in SKY_COLORS:
            self.update_profile_marker(name)
        self.update_panel_markers()
        self.update_time_caption()
        self.update_slider()

    def set_time_index(self, index):
        index = int(np.clip(index, 0, len(self._epochs) - 1))
        if index == self._time_index:
            return
        self._time_index = index
        self.update_selection()
        self.update_stats()
        self.window.Render()

    def update_slider(self):
        epochs = self._epochs
        if not (self._graph_expanded and self._slider_box is not None and len(epochs) > 1):
            for actor in (self.slider_track, self.slider_fill):
                actor.GetMapper().SetInputData(self.overlay_lines_data([]))
            self.set_bullseye(self.slider_handle, [])
            for actor in self.slider_texts.values():
                actor.SetVisibility(False)
            return
        x0, x1, y = self._slider_box
        index = min(self._time_index, len(epochs) - 1)
        handle = x0 + (x1 - x0) * index / (len(epochs) - 1)
        self.slider_track.GetMapper().SetInputData(self.overlay_lines_data([[(x0, y), (x1, y)]]))
        self.slider_fill.GetMapper().SetInputData(self.overlay_lines_data([[(x0, y), (handle, y)]] if index else []))
        self.set_bullseye(self.slider_handle, [(handle, y)])
        form = "%m-%d %H:%M:%S" if self._interval_step_minutes < 1 else "%m-%d %H:%M"
        placements = {"start": (x0, epochs[0]), "end": (x1, epochs[-1]),
                      "current": (float(np.clip(handle, x0 + 0.12, x1 - 0.12)), epochs[index])}
        for key, (x, epoch) in placements.items():
            self.slider_texts[key].SetInput(f"{epoch:{form}}")
            self.slider_texts[key].SetPosition(x, y - 0.032)
            self.slider_texts[key].SetVisibility(True)

    def slider_hit(self, x, y):
        if not self._graph_expanded or self._slider_box is None or len(self._epochs) < 2:
            return False
        width, height = max(self.window.GetSize()[0], 1), max(self.window.GetSize()[1], 1)
        x0, x1, ys = self._slider_box
        return x0 - 0.02 <= x / width <= x1 + 0.02 and abs(y / height - ys) <= 0.03

    def slider_drag(self, x):
        width = max(self.window.GetSize()[0], 1)
        x0, x1, _ = self._slider_box
        fraction = float(np.clip((x / width - x0) / (x1 - x0), 0.0, 1.0))
        self.set_time_index(round(fraction * (len(self._epochs) - 1)))

    def scrollbar_hit(self, x, y):
        return self._graph_expanded and x / max(self.window.GetSize()[0], 1) >= 0.985

    def scrollbar_press(self, y):
        """Grab the thumb where it was pressed; a click on the bare track centres the thumb there."""
        length = 0.96 / self.page_height()
        top = 0.98 - 0.96 * self._scroll / self.page_height()
        ny = y / max(self.window.GetSize()[1], 1)
        self._scrollbar_grab = top - ny if top - length <= ny <= top else length / 2
        self.scrollbar_drag(y)

    def scrollbar_drag(self, y):
        top = y / max(self.window.GetSize()[1], 1) + self._scrollbar_grab
        self.scroll_to((0.98 - top) * self.page_height() / 0.96)

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
            body = self.shown_body(name)
            above, below = [], []
            dots_above = dots_below = np.empty((0, 2))
            if body is not None:
                az, el = np.asarray(body["az"], dtype=float), np.asarray(body["el"], dtype=float)
                clearance = el - np.asarray(body["terrain"], dtype=float)  # < 0: under the terrain horizon
                points = np.column_stack((x0 + (x1 - x0) * az / 360.0, self.profile_y(el)))
                # Break the trail where the azimuth wraps from 360 to 0, then where it crosses the horizon.
                breaks = np.flatnonzero(np.abs(np.diff(az)) > 180.0) + 1
                for chunk, values in zip(np.split(points, breaks), np.split(clearance, breaks)):
                    for piece, is_below in split_at_zero(chunk, values):
                        (below if is_below else above).append(piece)
                dots_above, dots_below = points[clearance >= 0], points[clearance < 0]
            self.sky_trails[name].GetMapper().SetInputData(self.overlay_lines_data(above))
            self.sky_trails_below[name].GetMapper().SetInputData(self.overlay_lines_data(self.dashed_pieces(below)))
            self.sky_samples[name].GetMapper().SetInputData(self.overlay_point_data(dots_above))
            self.sky_samples_below[name].GetMapper().SetInputData(self.overlay_point_data(dots_below))
            self.update_profile_marker(name)
        self.update_time_caption()
        self.update_panels()
        self.update_slider()

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

    def step_time(self, delta):
        self.set_time_index(self._time_index + delta)

    def over_sky_controls(self, x, y):
        if self._graph_expanded:
            return True  # the graph covers the whole window
        width, height = self.window.GetSize()
        return (0.59 <= x / width <= 0.99 and 0.055 <= y / height <= 0.35) or (0.02 <= x / width <= 0.30 and 0.95 <= y / height <= 0.995)
