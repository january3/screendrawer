"""Class for different brushes."""
import logging                                                   # <remove>
import gi                                                        # <remove>
import numpy as np                                               # <remove>
gi.require_version('Gtk', '3.0')                                 # <remove> pylint: disable=wrong-import-position

import cairo                                                     # <remove>
from .utils import path_bbox, smooth_coords                      # <remove>
from .utils import coords_rotate, transform_coords               # <remove>
from .utils import coords_rotate_np, transform_coords_np               # <remove>
from .utils import calculate_length                              # <remove>
from .utils import first_point_after_length                      # <remove>
from .brushutils import calc_normal_outline, calc_normal_outline_tapered # <remove>
from .brushutils import calc_pencil_outline, smooth_pressure, bin_values # <remove>
from .brushutils import calculate_slant_outlines                 # <remove>
from .brushutils import find_intervals, min_pr                   # <remove>
from .brushutils import get_current_color_and_alpha              # <remove>
from .brushutils import calc_pencil_segments              # <remove>
log = logging.getLogger(__name__)                                # <remove>
#log.setLevel(logging.INFO)                                      # <remove>

# ----------- Brush factory ----------------

class BrushFactory:
    """
    Factory class for creating brushes.
    """
    @classmethod
    def all_brushes(cls, **kwargs):
        """
        Create dict of all brushes.
        """

        brushes = {
                "rounded": BrushRound(**kwargs),
                "slanted": BrushSlanted(**kwargs),
                "marker":  BrushMarker(**kwargs),
                "pencil":  BrushPencil(**kwargs),
                "tapered": BrushTapered(**kwargs),
                }

        return brushes

    @classmethod
    def create_brush(cls, brush_type = "marker", **kwargs):
        """
        Create a brush of the specified type.
        """

        if not brush_type:
            brush_type = "rounded"

        log.debug("Selecting BRUSH %s, kwargs %s", brush_type, kwargs)

        brushes = {
                "rounded": BrushRound,
                "slanted": BrushSlanted,
                "marker":  BrushMarker,
                "pencil":  BrushPencil,
                "tapered": BrushTapered,
                "simple":  BrushSimple,
                }

        if brush_type in brushes:
            return brushes[brush_type](**kwargs)

        raise NotImplementedError("Brush type", brush_type, "not implemented")

# ----------- Brushes ----------------

class Brush:
    """Base class for brushes."""
    def __init__(self, rounded = False, brush_type = "marker", outline = None, smooth_path = True):
        self.__outline = outline or [ ]
        self.__coords = [ ]
        self.__pressure = [ ]
        self.__rounded = rounded
        self.__outline = [ ]
        self.__coords = [ ]
        self.__brush_type = brush_type
        self.__smooth_path = smooth_path
        self.__bbox = None

    def to_dict(self):
        """Return a dictionary representation of the brush."""
        return {
                "brush_type": self.__brush_type,
                "smooth_path": self.smooth_path(),
               }

    def bbox_move(self, dx, dy):
        """Move the bbox by dx, xy"""
        if self.__bbox is None:
            self.bbox(force = True)
            return

        x, y, w, h = self.__bbox
        self.__bbox = (x + dx, y + dy, w, h)

    def bbox(self, force = False):
        """Get bounding box of the brush."""

        if self.__outline is None or len(self.__outline) < 4:
            return None
        if not self.__bbox or force:
            xy_max = np.max(self.__outline, axis = 0)
            xy_min = np.min(self.__outline, axis = 0)
            self.__bbox = [xy_min[0], xy_min[1], 
                           xy_max[0] - xy_min[0], 
                           xy_max[1] - xy_min[1]]
        return self.__bbox

    def brush_type(self):
        """Get brush type."""
        return self.__brush_type

    def smooth_path(self, smooth_path = None):
        """Set or get smooth path."""
        if smooth_path is not None:
            self.__smooth_path = smooth_path
        return self.__smooth_path

    def coords(self, coords = None):
        """Set or get brush coordinates."""
        if coords is not None:
            self.__coords = coords
        return self.__coords

    def outline(self, new_outline = None):
        """Get brush outline."""
        if new_outline is not None:
            self.__outline = new_outline
        if self.__outline is None or len(self.__outline) < 4:
            return None
        return self.__outline

    def pressure(self, pressure = None):
        """Set or get brush pressure."""
        if pressure is not None:
            self.__pressure = pressure
        return self.__pressure

    def draw(self, cr, outline = False):
        """Draw the brush on the Cairo context."""
        if self.__outline is None or len(self.__outline) < 4:
            return

        cr.move_to(self.__outline[0][0], self.__outline[0][1])

        for point in self.__outline[1:]:
            cr.line_to(point[0], point[1])
        cr.close_path()

        if outline:
            cr.set_source_rgb(0, 1, 1)
            cr.set_line_width(0.1)
            coords = self.coords()

            for i in range(len(coords) - 1):
                cr.move_to(coords[i][0], coords[i][1])
                cr.line_to(coords[i + 1][0], coords[i + 1][1])
                cr.stroke()
                # make a dot at each coord
                cr.arc(coords[i][0], coords[i][1], .8, 0, 2 * 3.14159)
                cr.fill()

    def move(self, dx, dy):
        """Move the outline."""
        self.__outline = [ (x + dx, y + dy) for x, y in self.__outline ]
        self.bbox_move(dx, dy)
        #print("bbox now 1:", self.bbox())
        #self.bbox(force = True)
        #print("bbox now 2:", self.bbox())

    def rotate(self, angle, rot_origin):
        """Rotate the outline."""
        self.__outline = coords_rotate(self.__outline, angle, rot_origin)
        self.bbox(force = True)

    def scale(self, old_bbox, new_bbox):
        """Scale the outline."""
        self.__outline = transform_coords(self.__outline, old_bbox, new_bbox)
        self.bbox(force = True)

    def calc_width(self, pressure, lwd):
        """Calculate the width of the brush."""

        widths = [ lwd * (0.25 + p) for p in pressure ]
        return widths

    def coords_add(self, point, pressure):
        """Add a point to the brush coords."""

    def calculate(self, line_width, coords, pressure = None):
        """Recalculate the outline of the brush."""

        pressure = pressure if pressure is not None else [1] * len(coords)

        if coords is not None and pressure is not None:
            if len(coords) != len(pressure):
                log.warning("Pressure and coords don't match (%d <> %d)",
                    len(coords), len(pressure))
                pressure = None

        nc = len(coords)

        if nc < 2:
            return None

        # we are smoothing only shorter lines to avoid generating too many
        # points
        if self.smooth_path() and nc < 100:
            coords, pressure = smooth_coords(coords, pressure)

        if len(coords) != len(pressure):
            log.warning("Pressure and coords don't match (%d <> %d)",
                    len(coords), len(pressure))

        self.coords(coords)

        widths = self.calc_width(pressure, line_width)
        outline_l, outline_r = calc_normal_outline(coords, widths, self.__rounded)
        outline  = np.vstack((outline_l, outline_r[::-1]))

        self.__outline = outline
        self.bbox(force = True)
        return outline

class BrushMarker(Brush):
    """Marker brush."""
    def __init__(self, outline = None, smooth_path = True):
        super().__init__(rounded = False, brush_type = "marker",
                         outline = outline, smooth_path = smooth_path)

class BrushRound(Brush):
    """Round brush."""
    def __init__(self, outline = None, smooth_path = True):
        super().__init__(rounded = True, brush_type = "rounded",
                         outline = outline, smooth_path = smooth_path)

class BrushTapered(Brush):
    """Tapered brush."""
    def __init__(self, outline = None, smooth_path = True):
        super().__init__(rounded = False, brush_type = "tapered",
                         outline = outline, smooth_path = smooth_path)

    def calculate(self, line_width, coords, pressure = None):
        """Recalculate the outline of the brush."""

        if coords is not None and pressure is not None:
            if len(coords) != len(pressure):
                raise ValueError("Pressure and coords don't match")

        pressure = pressure or [1] * len(coords)

        lwd = line_width

        if len(coords) < 2:
            return None

        n_taper = 5
        if len(coords) < n_taper:
            n_taper = len(coords) - 2

        length_to_taper = calculate_length(coords[:(len(coords) - n_taper)])

        if self.smooth_path():
            coords, pressure = smooth_coords(coords, pressure, 20)

        taper_pos = first_point_after_length(coords, length_to_taper)
        taper_length = calculate_length(coords[taper_pos:])

        outline_l, outline_r = calc_normal_outline_tapered(coords, pressure, lwd,
                                                    taper_pos, taper_length)

        outline  = outline_l + outline_r[::-1]

        if len(coords) != len(pressure):
            #raise ValueError("Pressure and coords don't match")
            log.warning("Pressure and coords don't match (%d <> %d)",
                    len(coords), len(pressure))
        self.outline(outline)
        self.bbox(force = True)
        return outline

class BrushPencil(Brush):
    """
    Pencil brush, v3.

    This is more or less an experimental pencil.

    This version attempts to draw with the same stroke, but with
    transparency varying depending on pressure. The idea is to create short
    segments with different transparency. Each segment is drawn using a
    gradient, from starting transparency to ending transparency.
    """
    def __init__(self, outline = None, bins = None, smooth_path = True): # pylint: disable=unused-argument
        super().__init__(rounded = True, brush_type = "pencil",
                         outline = outline, smooth_path = smooth_path)
        self.__pressure  = [ ]
        self.__coords = [ ]
        self.__gradients = None
        self.__midpoints = None
        self.__seg_info = None

    def draw(self, cr, outline = False):
        """Draw the brush on the Cairo context."""
        if self.__coords is None or len(self.__coords) < 2:
            return

        if len(self.__pressure) != len(self.__coords):
            log.warning("Pressure and coords don't match (%d <> %d)",
                    len(self.__pressure), len(self.__coords))
            return

        if outline:
            cr.set_line_width(0.4)
            cr.set_source_rgba(0, 1, 1, 1)

        mp    = self.__midpoints
        segs  = self.outline()
        sinfo = self.__seg_info

        rgba = get_current_color_and_alpha(cr)

        for seg_i in range(len(sinfo)):

            seg_pos = sinfo[seg_i, 0]
            seg_len = sinfo[seg_i, 1]
            cr.move_to(segs[seg_pos][0], segs[seg_pos][1])

            for i in range(seg_pos + 1, seg_pos + seg_len):
                cr.line_to(segs[i][0], segs[i][1])
            cr.close_path()

            if outline:
                cr.stroke()
            else:
                gr = self.get_gradient(rgba, mp[seg_i])
                cr.set_source(gr)
                cr.fill()

        if outline:
            self.draw_outline(cr)

    def get_gradient(self, c, info):
        """Get a gradient for the brush segment."""

        gr = cairo.LinearGradient(info[0], info[1],
                                  info[2], info[3])
        gr.add_color_stop_rgba(0, c[0], c[1], c[2], c[3] * info[4])
        gr.add_color_stop_rgba(1, c[0], c[1], c[2], c[3] * info[5])
        return gr

    def draw_outline(self, cr):
        """Draw the outline of the brush."""

        mp    = self.__midpoints
        segs  = self.outline()
        sinfo = self.__seg_info

        cr.set_source_rgba(0, 1, 1, 1)
        cr.set_line_width(0.04)
        #cr.stroke()

        # segment points
        for seg_i in range(len(sinfo)):
            seg_pos = sinfo[seg_i, 0]
            seg_len = sinfo[seg_i, 1]
            cr.move_to(segs[seg_pos][0], segs[seg_pos][1])

            for i in range(seg_pos, seg_pos + seg_len):
                #cr.line_to(self.__coords[i, 0], self.__coords[i, 1])
                cr.arc(segs[i][0], segs[i][1], .7, 0, 2 * 3.14159)
                cr.fill()

        # segment midpoints, start and end
        for seg_i in range(len(sinfo)):
            cr.arc(mp[seg_i, 0], mp[seg_i, 1], .7, 0, 2 * 3.14159)
            cr.fill()
            cr.arc(mp[seg_i, 2], mp[seg_i, 3], .7, 0, 2 * 3.14159)
            cr.fill()

    def calculate(self, line_width, coords, pressure = None):
        """Recalculate the outline of the brush."""

        if coords is not None and pressure is not None:
            if len(coords) != len(pressure):
                raise ValueError("Pressure and coords don't match")

        pressure = pressure or [1] * len(coords)

        lwd = line_width
        nc = len(coords)

        if nc < 4:
            return None

        if self.smooth_path() and nc < 25:
            coords, pressure = smooth_coords(coords, pressure)

        self.__pressure  = pressure
        self.__coords    = coords

        widths = np.full(len(coords), lwd * .67)
        segments, self.__seg_info, self.__midpoints = calc_pencil_segments(coords, widths, pressure)

        self.outline(segments)

        self.bbox(force = True)
        return self.outline()

class BrushSlanted(Brush):
    """Slanted brush."""
    def __init__(self, slant = None, smooth_path = True):
        super().__init__(brush_type = "slanted", outline = None, smooth_path = smooth_path)

        self.__slant = slant or [(-0.4, 0.6), (0.3, - 0.6)]

    def to_dict(self):
        """Return a dictionary representation of the brush."""
        return {
                "brush_type": self.brush_type(),
                "smooth_path": self.smooth_path(),
                "slant": self.__slant
                }

    def rotate(self, angle, rot_origin):
        """Rotate the outline."""
        #self.outline(coords_rotate(self.outline(), angle, rot_origin))
        self.__slant = coords_rotate(self.__slant, angle, (0, 0))
        self.bbox(force = True)

    def calculate(self, line_width = 1, coords = None, pressure = None):
        """Recalculate the outline of the brush."""

        if coords is not None and pressure is not None:
            if len(coords) != len(pressure):
                raise ValueError("Pressure and coords don't match")

        if self.smooth_path():
            coords, pressure = smooth_coords(coords, pressure)

        # we need to multiply everything by line_width
        dx0, dy0, dx1, dy1 = [ line_width * x for x in self.__slant[0] + self.__slant[1] ]

        outline_l, outline_r = calculate_slant_outlines(coords, dx0, dy0, dx1, dy1)
        outline = outline_l + outline_r[::-1]
        self.outline(outline)

        #print("done calculating slanted brush")
        self.bbox(force = True)
        return outline

class BrushSimple(Brush):
    """Simplistic brush for testing purposes."""
    def __init__(self, smooth_path = True):
        super().__init__(brush_type = "simple", outline = None, smooth_path = True)

        self.__line_width = 1
        self.__bbox = None

    def catmull_rom_spline(self, p0, p1, p2, p3, t):
        """Calculate a point on the Catmull-Rom spline."""

        t2 = t * t
        t3 = t2 * t

        part1 = 2 * p1
        part2 = -p0 + p2
        part3 = 2 * p0 - 5 * p1 + 4 * p2 - p3
        part4 = -p0 + 3 * p1 - 3 * p2 + p3

        return 0.5 * (part1 + part2 * t + part3 * t2 + part4 * t3)

    def calculate_control_points(self, points):
        """Calculate the control points for a smooth curve using Catmull-Rom splines."""
        n = len(points)
        p0 = points[:-3]
        p1 = points[1:-2]
        p2 = points[2:-1]
        p3 = points[3:]

        t = 0.5  # Tension parameter for Catmull-Rom spline

        c1 = p1 + (p2 - p0) * t / 3
        c2 = p2 - (p3 - p1) * t / 3

        return np.squeeze(c1), np.squeeze(c2)

    def draw (self, cr, outline = False):
        """Draw a smooth spline through the given points using Catmull-Rom splines."""
        points = self.outline()

        if len(points) < 4 or outline:
            self.draw_simple(cr, outline)

        points = np.array(points)

        # Move to the first point
        cr.move_to(points[0][0], points[0][1])

        for i in range(1, len(points) - 2):
            p1 = points[i]
            p2 = points[i + 1]
            c1, c2 = self.calculate_control_points(points[i-1:i+3])

            # Draw the cubic Bezier curve using the calculated control points
            cr.curve_to(c1[0], c1[1], c2[0], c2[1], p2[0], p2[1])

        # Draw the path
        cr.set_line_width(self.__line_width / 3)
        cr.stroke()


    def draw_simple(self, cr, outline = False):
        """Draw the brush on the Cairo context."""

        coords = self.outline()
        cr.set_line_width(self.__line_width / 3)

        cr.move_to(coords[0][0], coords[0][1])

        for point in coords[1:]:
            cr.line_to(point[0], point[1])

        cr.stroke()

    def bbox(self, force = False):
        """Get bounding box of the brush."""

        outline = self.outline()

        log.debug("bbox brush simple")
        if not outline:
            log.debug("no outline, returning None")
            return None
        if not self.__bbox or force:
            log.debug("recalculating bbox")
            self.__bbox = path_bbox(outline,
                                    lw = self.__line_width)
        return self.__bbox

    def calculate(self, line_width = 1, coords = None, pressure = None):
        """Recalculate the outline of the brush."""
        log.debug("calculate calling")
        coords, _ = smooth_coords(coords)
        self.__line_width = line_width

        self.outline(coords)
        self.bbox(force = True)
        return coords
