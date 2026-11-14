"""3D globe for the Moon viewer: terrain mesh, camera/navigation, picking, lat/lon grid, pole marks, terrain view.

GlobeMixin is mixed into moon.MoonWindow and expects the host to provide: self.vtk, self.renderer,
self.window, self.interactor, self.lat/lon, the flags _terrain_view/_graph_expanded/_grid_*,
orbit_pivot, _zoom_anchor, and (from HudMixin) make_text(). _init_globe() builds the mesh, texture,
horizon/observer/grid actors and the picker, and must run after the renderer exists.
Texture path is read as terrain.COLOR_PATH (set by terrain.load_data()).
"""
import numpy as np
from PIL import Image

import terrain
from compare import MAX_POINTS, POINT_COLORS
from terrain import MOON_RADIUS_M, graticule_lines, lat_lon_to_xyz, choose_graticule_step, pole_heights, sample_dem_bilinear


def load_color_texture(path):
    """Decode with Pillow, then share its north-first RGB pixels with VTK."""
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk

    try:
        with terrain.allow_large_nasa_image():
            with Image.open(path) as image:
                rgb = np.asarray(image.convert('RGB'))
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Cannot decode color map {path}. The TIFF may be incomplete or damaged; "
                           f"check or download the file again. Details: {error}") from error
    height, width, _ = rgb.shape
    pixels = numpy_to_vtk(rgb.reshape(-1, 3), deep=False)
    texture_data = vtk.vtkImageData()
    texture_data.SetDimensions(width, height, 1)
    texture_data.GetPointData().SetScalars(pixels)
    return rgb, texture_data

TERRAIN_MAX_PITCH = 89.0       # degrees up / down in the terrain view (never past the zenith or nadir)
TERRAIN_LOOK_INVERT_Y = False  # True: dragging up looks up (flight-simulator style) instead of down


def terrain_grid_axis(half_width_km, side):
    """Grid lines from -half_width to +half_width (side must be odd): dense near 0, coarser far away.

    Spacing grows with distance (about 4 m next to the observer, 2 km at the rim), and a line sits
    exactly at 0 so one vertex is the ground point below the observer.
    """
    if side % 2 == 0:
        raise ValueError("The terrain grid needs an odd number of lines.")
    t = np.arange(-(side // 2), side // 2 + 1) / (side // 2)
    return half_width_km * t * np.abs(t)


class GlobeMixin:
    def _init_globe(self):
        """Create the Moon mesh, texture, horizon/observer/grid actors, pole marks and the picker."""
        import vtk
        from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

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
        # Pillow's first row is north; VTK's TIFF reader used south-first rows.
        uv = np.column_stack(((lon.ravel() + 180) / 360,
                              (90 - lat.ravel()) / 180)).astype(np.float32)
        self.mesh.GetPointData().SetTCoords(numpy_to_vtk(uv, deep=True))
        self.surface_locator = vtk.vtkStaticCellLocator()
        self.surface_locator.SetDataSet(self.mesh)
        self.surface_locator.BuildLocator()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.mesh)
        self._color_rgb, self._color_texture_data = load_color_texture(terrain.COLOR_PATH)
        self.texture_dimensions = self._color_texture_data.GetDimensions()[:2]
        self.texture = vtk.vtkTexture()
        self.texture.SetInputData(self._color_texture_data)
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

        # Latitude / longitude grid. Its spacing follows the zoom (update_graticule), and only
        # the part of the Moon the camera can see is built. L hides/shows it.
        self._grid_enabled = True
        self._grid_step = None
        self._grid_built = None  # (step, unit vector of the cap centre, cap radius in degrees)
        self.grid_minor = self.make_graticule_actor((0.85, 0.9, 1.0), 1.5, 0.6, -2.0)
        self.grid_major = self.make_graticule_actor((1.0, 0.85, 0.2), 3.0, 0.95, -3.0)
        self.grid_caption = self.make_text(0.025, 0.23, 14)
        # Comparison points: a coloured surface mark and a screen-space number each (refresh_point_marks).
        self.point_marks = []
        self._terrain_heading = self._terrain_pitch = 0.0  # terrain view look direction, degrees
        self._points_note = ""
        self.points_caption = self.make_text(0.025, 0.26, 14)
        # North and south pole marks: a point on the surface and a screen-space letter.
        self.pole_marks = {1: self.make_pole_mark(1), -1: self.make_pole_mark(-1)}
        self.renderer.AddObserver("StartEvent", self.on_render_start)

        self.picker = vtk.vtkCellPicker()
        self.picker.SetTolerance(0.001)
        self.picker.PickFromListOn()
        self.picker.AddPickList(self.moon)

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

    def configure_surface_overlay(self, mapper):
        # Depth bias separates coincident fragments without lifting geometry.
        # The ordinary depth test still hides marks on the Moon's far side.
        mapper.SetResolveCoincidentTopologyToPolygonOffset()
        mapper.SetRelativeCoincidentTopologyLineOffsetParameters(0.0, -4.0)
        mapper.SetRelativeCoincidentTopologyPointOffsetParameter(-4.0)

    def make_graticule_actor(self, color, width, opacity, line_offset):
        actor = self.vtk.vtkActor()
        mapper = self.vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.vtk.vtkPolyData())
        self.configure_surface_overlay(mapper)
        mapper.SetRelativeCoincidentTopologyLineOffsetParameters(0.0, line_offset)  # horizon line (-4) stays on top
        actor.SetMapper(mapper)
        prop = actor.GetProperty()
        prop.SetColor(*color)
        prop.SetLineWidth(width)
        prop.SetOpacity(opacity)
        prop.LightingOff()
        actor.PickableOff()
        actor.VisibilityOff()
        self.renderer.AddActor(actor)
        return actor

    def polylines_to_polydata(self, polylines):
        from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

        data = self.vtk.vtkPolyData()
        if not len(polylines):
            return data
        points = np.ascontiguousarray(np.vstack(polylines), dtype=np.float64)
        lengths = np.array([len(line) for line in polylines], dtype=np.int64)
        offsets = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
        cells = self.vtk.vtkCellArray()
        cells.SetData(numpy_to_vtkIdTypeArray(offsets, deep=True),
                      numpy_to_vtkIdTypeArray(np.arange(len(points), dtype=np.int64), deep=True))
        vtk_points = self.vtk.vtkPoints()
        vtk_points.SetData(numpy_to_vtk(points, deep=True))
        data.SetPoints(vtk_points)
        data.SetLines(cells)
        return data

    def make_pole_mark(self, sign):
        height = pole_heights()[sign]
        color = (1.0, 0.45, 0.4) if sign > 0 else (0.45, 0.75, 1.0)
        points = self.vtk.vtkPoints()
        points.SetDataTypeToDouble()
        points.InsertNextPoint(0.0, 0.0, height)
        vertices = self.vtk.vtkCellArray()
        vertices.InsertNextCell(1)
        vertices.InsertCellPoint(0)
        data = self.vtk.vtkPolyData()
        data.SetPoints(points)
        data.SetVerts(vertices)
        mapper = self.vtk.vtkPolyDataMapper()
        mapper.SetInputData(data)
        self.configure_surface_overlay(mapper)
        actor = self.vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().LightingOff()
        actor.GetProperty().SetPointSize(11)
        actor.GetProperty().RenderPointsAsSpheresOn()
        actor.PickableOff()
        self.renderer.AddActor(actor)
        label = self.make_text(0.5, 0.5, 22)
        label.SetInput("N" if sign > 0 else "S")
        prop = label.GetTextProperty()
        prop.SetColor(*color)
        prop.SetBold(True)
        prop.SetJustificationToCentered()
        label.SetVisibility(False)
        return {"point": actor, "label": label, "position": np.array([0.0, 0.0, height])}

    def update_pole_marks(self):
        """Show N / S next to a pole while it is on the camera's side of the Moon."""
        show = not self._terrain_view and not self._graph_expanded
        width, height = self.window.GetSize()
        camera_position = np.array(self.renderer.GetActiveCamera().GetPosition())
        for mark in self.pole_marks.values():
            position = mark["position"]
            mark["point"].SetVisibility(show)
            placed = False
            # A pole is visible when the camera is above the surface's tangent plane there.
            if show and width > 0 and height > 0 and float(np.dot(position, camera_position - position)) > 0.0:
                self.renderer.SetWorldPoint(position[0], position[1], position[2], 1.0)
                self.renderer.WorldToDisplay()
                x, y, _ = self.renderer.GetDisplayPoint()
                if 0.0 <= x <= width and 0.0 <= y <= height:
                    mark["label"].SetPosition(x / width, (y + 14.0) / height)
                    placed = True
            mark["label"].SetVisibility(placed)

    def grid_pixels_per_degree(self):
        """Screen pixels per degree of lunar surface at the point the camera looks at."""
        camera = self.renderer.GetActiveCamera()
        distance = float(np.linalg.norm(np.array(camera.GetPosition()) - self.view_surface_target()))
        height = max(self.window.GetSize()[1], 1)
        pixels_per_radian = height / (2.0 * np.tan(np.radians(camera.GetViewAngle()) / 2.0) * max(distance, 1e-9))
        return float(pixels_per_radian * np.pi / 180.0)

    def update_graticule(self):
        """Pick the grid spacing for the current zoom and rebuild the visible part when needed."""
        show = self._grid_enabled and not self._terrain_view and not self._graph_expanded
        self.grid_caption.SetVisibility(show)
        if not show:
            self.grid_minor.SetVisibility(False)
            self.grid_major.SetVisibility(False)
            return
        position = np.array(self.renderer.GetActiveCamera().GetPosition())
        distance = float(np.linalg.norm(position))
        if distance < 1e-9:
            return
        step = choose_graticule_step(self.grid_pixels_per_degree(), self._grid_step)
        unit = position / distance
        # Everything the camera can see lies inside the cap below it that reaches the horizon.
        ratio = 0.97 / distance  # 0.97: slightly below the lowest terrain
        cap = 90.0 if ratio >= 1.0 else float(np.degrees(np.arccos(ratio)))
        rebuild = self._grid_built is None or self._grid_built[0] != step
        if not rebuild:
            moved = float(np.degrees(np.arccos(np.clip(np.dot(unit, self._grid_built[1]), -1.0, 1.0))))
            rebuild = moved + cap > self._grid_built[2]
        if rebuild:
            radius = min(180.0, cap * 1.3 + 2.0)
            lat = float(np.degrees(np.arcsin(np.clip(unit[2], -1.0, 1.0))))
            lon = float(np.degrees(np.arctan2(unit[1], unit[0])))
            minor, major = graticule_lines(step, lat, lon, radius)
            self.grid_minor.GetMapper().SetInputData(self.polylines_to_polydata(minor))
            self.grid_major.GetMapper().SetInputData(self.polylines_to_polydata(major))
            self._grid_built = (step, unit, radius)
            self._grid_step = step
        self.grid_minor.SetVisibility(True)
        self.grid_major.SetVisibility(True)
        self.grid_caption.SetInput(f"Grid: every {step:g} deg   (L: hide)")

    def refresh_point_marks(self):
        """Rebuild the marks of the comparison set (self._points) on the Moon."""
        for mark in self.point_marks:
            self.renderer.RemoveActor(mark["point"])
            self.renderer.RemoveViewProp(mark["label"])
        self.point_marks = []
        for index, (lat, lon) in enumerate(self._points):
            color = POINT_COLORS[index % len(POINT_COLORS)]
            position = self.terrain_overlay_points(np.array(lat_lon_to_xyz(lat, lon, offset=0.0)))[0]
            points = self.vtk.vtkPoints()
            points.SetDataTypeToDouble()
            points.InsertNextPoint(*position)
            vertices = self.vtk.vtkCellArray()
            vertices.InsertNextCell(1)
            vertices.InsertCellPoint(0)
            data = self.vtk.vtkPolyData()
            data.SetPoints(points)
            data.SetVerts(vertices)
            mapper = self.vtk.vtkPolyDataMapper()
            mapper.SetInputData(data)
            self.configure_surface_overlay(mapper)
            actor = self.vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(*color)
            actor.GetProperty().LightingOff()
            actor.GetProperty().SetPointSize(12)
            actor.GetProperty().RenderPointsAsSpheresOn()
            actor.PickableOff()
            self.renderer.AddActor(actor)
            label = self.make_text(0.5, 0.5, 18)
            label.SetInput(str(index + 1))
            prop = label.GetTextProperty()
            prop.SetColor(*color)
            prop.SetBold(True)
            prop.SetJustificationToCentered()
            label.SetVisibility(False)
            self.point_marks.append({"point": actor, "label": label, "position": np.asarray(position, dtype=float)})

    def update_point_marks(self):
        """Show each comparison number next to its mark while the mark is on the camera's side of the Moon."""
        show = not self._terrain_view and not self._graph_expanded
        width, height = self.window.GetSize()
        camera_position = np.array(self.renderer.GetActiveCamera().GetPosition())
        for mark in self.point_marks:
            position = mark["position"]
            mark["point"].SetVisibility(show)
            placed = False
            if show and width > 0 and height > 0 and float(np.dot(position, camera_position - position)) > 0.0:
                self.renderer.SetWorldPoint(position[0], position[1], position[2], 1.0)
                self.renderer.WorldToDisplay()
                x, y, _ = self.renderer.GetDisplayPoint()
                if 0.0 <= x <= width and 0.0 <= y <= height:
                    mark["label"].SetPosition(x / width, (y + 14.0) / height)
                    placed = True
            mark["label"].SetVisibility(placed)
        count = len(self._points)
        text = (f"Comparison points: {count}/{MAX_POINTS}   (Shift+click / A: add, P: manage"
                + (", T: compare)" if count >= 2 else ")")) if count else \
            "Comparison points: none   (Shift+click / A: add, P: manage / load)"
        if self._points_note:
            text += "\n" + self._points_note
        self.points_caption.SetInput(text)
        self.points_caption.SetVisibility(show)

    def on_render_start(self, obj=None, event=None):
        self.update_graticule()
        self.update_pole_marks()
        self.update_point_marks()

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
        if self._terrain_view:
            camera.SetClippingRange(*self.terrain_clipping())
            return
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
        focus = np.array(camera.GetFocalPoint())
        forward = np.asarray(camera.GetDirectionOfProjection())
        target = self.surface_hit(position, forward)
        if target is None and self._zoom_anchor is not None:
            direction = np.asarray(self._zoom_anchor) - position
            if np.dot(direction, forward) > 0:
                target = self.surface_hit(position, direction)
        if target is None:
            # If there is no prior pivot, try the surface currently under the mouse.
            x, y = self.interactor.GetEventPosition()
            if self.picker.Pick(x, y, 0, self.renderer):
                target = np.asarray(self.picker.GetPickPosition(), dtype=float)
        if target is None:
            return
        offset = position - target
        surface_distance = float(np.linalg.norm(offset))
        new_distance = max(self.minimum_surface_clearance, surface_distance * factor)
        if surface_distance > 1e-12:
            new_position = target + offset / surface_distance * new_distance
            delta = new_position - position
            camera.SetPosition(*new_position)
            camera.SetFocalPoint(*(focus + delta))
            self.constrain_camera()
            # Keep the viewing direction unchanged if terrain clearance pushes
            # the camera away from the requested position.
            correction = np.asarray(camera.GetPosition()) - new_position
            camera.SetFocalPoint(*(np.asarray(camera.GetFocalPoint()) + correction))
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

    def make_terrain_actor(self, half_width_km=250.0, side=501):
        """Local terrain mesh around the observer, in metres: x east, y north, z up.

        The origin is the ground point below the observer, so the eye is at (0, 0, eye height). Working
        in metres around the camera (instead of in unit-sphere coordinates) keeps the 32-bit vertex
        precision at centimetres, and the grid is dense near the observer so the ground under and
        around the camera follows the elevation map smoothly. Returns (actor, ground elevation in m).
        """
        from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
        axis = terrain_grid_axis(half_width_km, side)
        east_km, north_km = np.meshgrid(axis, axis)
        distance = np.hypot(east_km, north_km)
        lat0, lon0 = np.radians([self.lat, self.lon])
        up = np.array([np.cos(lat0) * np.cos(lon0), np.cos(lat0) * np.sin(lon0), np.sin(lat0)])
        north = np.array([-np.sin(lat0) * np.cos(lon0), -np.sin(lat0) * np.sin(lon0), np.cos(lat0)])
        east = np.array([-np.sin(lon0), np.cos(lon0), 0.0])
        theta = distance * 1000.0 / MOON_RADIUS_M
        # sinc(theta / pi) / radius avoids division by zero at the observer.
        scale = np.sinc(theta / np.pi) * 1000.0 / MOON_RADIUS_M
        direction = (np.cos(theta)[..., None] * up + scale[..., None] *
                     (north_km[..., None] * north + east_km[..., None] * east))
        lat = np.degrees(np.arctan2(direction[..., 2], np.hypot(direction[..., 0], direction[..., 1])))
        lon = np.degrees(np.arctan2(direction[..., 1], direction[..., 0]))
        ground_m = float(sample_dem_bilinear(self.lat, self.lon))
        radius_m = MOON_RADIUS_M + sample_dem_bilinear(lat, lon)
        relative = direction * radius_m[..., None] - up * (MOON_RADIUS_M + ground_m)
        local = np.stack((relative @ east, relative @ north, relative @ up), axis=-1)
        points = self.vtk.vtkPoints()
        points.SetData(numpy_to_vtk(local.reshape(-1, 3).astype(np.float32), deep=True))
        indices = np.arange(side * side).reshape(side, side)
        cells_np = np.column_stack((indices[:-1, :-1].ravel(), indices[:-1, 1:].ravel(),
                                    indices[1:, 1:].ravel(), indices[1:, :-1].ravel()))
        offsets = np.arange(0, cells_np.size + 1, 4, dtype=np.int64)
        cells = self.vtk.vtkCellArray()
        cells.SetData(numpy_to_vtkIdTypeArray(offsets, deep=True),
                      numpy_to_vtkIdTypeArray(cells_np.astype(np.int64).ravel(), deep=True))
        # Unwrap longitude around the observer to avoid a visible texture seam.
        lon_unwrapped = self.lon + (lon - self.lon + 180.0) % 360.0 - 180.0
        # Same convention as the globe mesh: the first image row is north, so v = (90 - lat) / 180.
        uv = np.column_stack(((lon_unwrapped.ravel() + 180.0) / 360.0,
                              (90.0 - lat.ravel()) / 180.0)).astype(np.float32)
        data = self.vtk.vtkPolyData()
        data.SetPoints(points)
        data.SetPolys(cells)
        data.GetPointData().SetTCoords(numpy_to_vtk(uv, deep=True))
        mapper = self.vtk.vtkPolyDataMapper()
        mapper.SetInputData(data)
        actor = self.vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.SetTexture(self.texture)
        actor.GetProperty().LightingOff()
        actor.PickableOff()
        return actor, ground_m

    def terrain_clipping(self):
        """Clipping range in metres. Nothing is nearer than the ground below the eye, so near can be large."""
        return min(max(0.05, 0.4 * float(terrain.EYE_HEIGHT_M)), 5.0), 4.0e5

    def place_terrain_camera(self):
        """Eye at the current eye height, looking along (heading, pitch); the view-up stays vertical.

        Looking around only changes two angles (heading from north, pitch above the horizon), so the
        camera can never roll or tilt sideways and the ground is always below the eye.
        """
        camera = self.renderer.GetActiveCamera()
        heading, pitch = np.radians(self._terrain_heading), np.radians(self._terrain_pitch)
        forward = np.array([np.cos(pitch) * np.sin(heading), np.cos(pitch) * np.cos(heading), np.sin(pitch)])
        eye = np.array([0.0, 0.0, float(terrain.EYE_HEIGHT_M)])
        camera.SetPosition(*eye)
        camera.SetFocalPoint(*(eye + forward * 10.0))
        camera.SetViewUp(0.0, 0.0, 1.0)
        camera.SetClippingRange(*self.terrain_clipping())

    def toggle_terrain_view(self, render=True):
        """Switch between the globe and a look-around view at the observer."""
        if self._graph_expanded and not self._terrain_view:
            return
        camera = self.renderer.GetActiveCamera()
        if not self._terrain_view:
            actor, _ = self.make_terrain_actor()
            self._terrain_saved_camera = (camera.GetPosition(), camera.GetFocalPoint(),
                                          camera.GetViewUp(), camera.GetViewAngle())
            # Never RemoveActor() the terrain actor: VTK then releases the GPU copy of the shared
            # texture, and re-uploading the full-size texture (with mipmaps) takes about 2 seconds
            # when returning to the globe. Keep one hidden actor and only swap its mesh.
            if self._terrain_actor is None:
                self._terrain_actor = actor
                self.renderer.AddActor(actor)
            else:
                self._terrain_actor.GetMapper().SetInputData(actor.GetMapper().GetInput())
                self._terrain_actor.SetVisibility(True)
            self.moon.SetVisibility(False)
            self.horizon.SetVisibility(False)
            self.observer.SetVisibility(False)
            self._terrain_view = True
            self._terrain_heading, self._terrain_pitch = 0.0, 0.0  # start looking north at the horizon
            camera.SetViewAngle(75)
            self.place_terrain_camera()
            self.help.SetInput("Terrain view: left-drag to look   |   Wheel: field of view\n"
                               "V: return to globe   |   G: coordinates   |   T: time interval\n"
                               "E: eye height")
        else:
            self._terrain_actor.SetVisibility(False)
            self._terrain_view = False
            self.moon.SetVisibility(True)
            self.horizon.SetVisibility(True)
            self.observer.SetVisibility(True)
            position, focus, up, angle = self._terrain_saved_camera
            camera.SetPosition(*position)
            camera.SetFocalPoint(*focus)
            camera.SetViewUp(*up)
            camera.SetViewAngle(angle)
            self.constrain_camera()
            self.help.SetInput(self.GLOBE_HELP)
        if render:
            self.window.Render()

    def terrain_look(self, dx, dy, render=True):
        """Drag to look around: the scene follows the mouse. Heading wraps, pitch stops at +/-89 degrees."""
        camera = self.renderer.GetActiveCamera()
        height = max(self.window.GetSize()[1], 1)
        speed = np.degrees(2 * np.tan(np.radians(camera.GetViewAngle()) / 2) / height)
        vertical = -1.0 if TERRAIN_LOOK_INVERT_Y else 1.0
        self._terrain_heading = (self._terrain_heading - dx * speed) % 360.0
        self._terrain_pitch = float(np.clip(self._terrain_pitch - vertical * dy * speed,
                                            -TERRAIN_MAX_PITCH, TERRAIN_MAX_PITCH))
        self.place_terrain_camera()
        if render:
            self.window.Render()

    def cursor_surface_target(self, x, y):
        """Pick a terrain pivot without changing the selected observer."""
        if self.picker.Pick(x, y, 0, self.renderer):
            pivot = np.asarray(self.picker.GetPickPosition(), dtype=float)
        else:
            pivot = np.asarray(self.view_surface_target(), dtype=float)
        self._zoom_anchor = pivot.copy()
        return pivot

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
