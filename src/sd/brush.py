"""Class for different brushes."""
import logging                                                   # <remove>
import gi                                                        # <remove>
import numpy as np                                               # <remove>
gi.require_version('Gtk', '3.0')                                 # <remove> pylint: disable=wrong-import-position

import cairo                                                     # <remove>
from .utils import path_bbox, smooth_coords                      # <remove>
from .utils import coords_rotate, transform_coords               # <remove>
from .utils import calculate_length                              # <remove>
from .utils import first_point_after_length                      # <remove>
from .brushutils import calc_normal_outline, calc_normal_outline_tapered # <remove>
from .brushutils import calc_normal_outline_bck
from .brushutils import calc_pencil_outline, smooth_pressure, bin_values # <remove>
from .brushutils import calculate_slant_outlines                 # <remove>
from .brushutils import find_intervals, min_pr                   # <remove>
from .brushutils import get_current_color_and_alpha              # <remove>
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
        self.bbox(force = True)

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

        lwd = line_width

        if len(coords) < 2:
            return None

        if self.smooth_path():
            coords, pressure = smooth_coords(coords, pressure)

        self.coords(coords)

        widths = self.calc_width(pressure, lwd)
        #outline_l, outline_r = calc_normal_outline_bck(coords, widths, self.__rounded)
        outline_l, outline_r = calc_normal_outline(coords, widths, self.__rounded)
        outline  = np.vstack((outline_l, outline_r[::-1]))
        #outline  = list(map(tuple, outline))

        if len(coords) != len(pressure):
            log.warning("Pressure and coords don't match (%d <> %d)",
                    len(coords), len(pressure))

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
        self.__fused = None
        self.__outline_segments = None

    def move(self, dx, dy):
        """Move the outline."""
        self.outline([ (x + dx, y + dy) for x, y in self.outline() ])
        self.__coords = [ (x + dx, y + dy) for x, y in self.__coords ]
        for s in self.__outline_segments:
            for i, p in enumerate(s):
                s[i] = (p[0] + dx, p[1] + dy)
        self.bbox(force = True)

    def rotate(self, angle, rot_origin):
        """Rotate the outline."""
        self.outline(coords_rotate(self.outline(),   angle, rot_origin))
        self.__coords = coords_rotate(self.__coords, angle, rot_origin)
        new_segm = [ ]
        for s in self.__outline_segments:
            new_segm.append(coords_rotate(s, angle, rot_origin))
        self.__outline_segments = new_segm
        self.bbox(force = True)

    def scale(self, old_bbox, new_bbox):
        """Scale the outline."""
        self.outline(transform_coords(self.outline(),   old_bbox, new_bbox))
        self.__coords = transform_coords(self.__coords, old_bbox, new_bbox)
        new_segm = [ ]
        for s in self.__outline_segments:
            new_segm.append(transform_coords(s, old_bbox, new_bbox))
        self.__outline_segments = new_segm
        self.bbox(force = True)

    def draw(self, cr, outline = False):
        """Draw the brush on the Cairo context."""
        r, g, b, a = get_current_color_and_alpha(cr)
        if not self.__coords or len(self.__coords) < 2:
            return

        if len(self.__pressure) != len(self.__coords):
            log.warning("Pressure and coords don't match (%d <> %d)",
                    len(self.__pressure), len(self.__coords))
            return

        #log.debug(f"Drawing pencil brush with coords: {len(self.__coords)} {len(self.__pressure)}")

        fbins = self.__outline_segments
        grads = self.__gradients
        midp  = self.__midpoints

        if outline:
            cr.set_line_width(0.4)
            cr.stroke()

        for i, segm in enumerate(fbins):
            gr = cairo.LinearGradient(midp[i][0][0], midp[i][0][1],
                                      midp[i][1][0], midp[i][1][1])
            gr.add_color_stop_rgba(0, r, g, b, a * grads[i][0])
            gr.add_color_stop_rgba(1, r, g, b, a * grads[i][1])

            cr.move_to(segm[0][0], segm[0][1])
            for p in segm:
                cr.line_to(p[0], p[1])
            cr.close_path()
            if outline:
                cr.stroke_preserve()
            else:
                cr.set_source(gr)
                cr.fill()

    def segment_midpoints(self, segments):
        """Calculate the midpoints of the segments start and end edge."""
        ret = [ ]

        for segm in segments:
            p0 = ((segm[0][0] + segm[-1][0])/2,
                  (segm[0][1] + segm[-1][1])/2)
            nn = int(len(segm)/2)
            p1 = ((segm[nn - 1][0] + segm[nn][0])/2,
                  (segm[nn - 1][1] + segm[nn][1])/2)
            ret.append((p0, p1))
        return ret

    def fuse_segments(self, pp):
        """Fuse segments which are next to each other"""

        ret = [ ]
        n_segments = len(pp)

        prev_pp = pp[0]
        ret_pp  = [ ]
        cur = [ ]

        # find groups of consecutive segments
        #log.debug(f"n_segments: {n_segments}")
        for j in range(n_segments):

            if pp[j] == prev_pp:
                #log.debug(f"pp[j]={pp[j]} prev_pp={prev_pp} appending {j} to the current segment")
                cur.append(j)
            else:
                #log.debug(f"new segment at: {j}")
                cur.append(j)
                ret_pp.append((min_pr(prev_pp), min_pr(pp[j])))
                ret.append(cur)
                cur = [j]

            prev_pp = pp[j]
        ret_pp.append((min_pr(prev_pp), min_pr(pp[-1])))
        ret.append(cur)

        #log.debug(f"length of ret: {len(ret)} length of ret_pp: {len(ret_pp)}")
        #for i, b in enumerate(ret):
        #    log.debug(f"bin {i} contents: {len(b)} pp: {[int(100 * ppp) for ppp in ret_pp[i]]}")

        return ret, ret_pp

    def construct_segments(self, fused_bins, outline_l, outline_r):
        """From bins, construct the segments of the outline."""
        ret = [ ]

        # construct the segments
        for segm in fused_bins:
            cur_segm = [ ]
            ret.append(cur_segm)
            for j in segm:
                cur_segm.append(outline_l[j])
            for j in segm[::-1]:
                cur_segm.append(outline_r[j])

        return ret

    def calculate(self, line_width, coords, pressure = None):
        """Recalculate the outline of the brush."""

        if coords is not None and pressure is not None:
            if len(coords) != len(pressure):
                raise ValueError("Pressure and coords don't match")

        pressure = pressure or [1] * len(coords)

        lwd = line_width

        if len(coords) < 2:
            return None

        if self.smooth_path():
            coords, pressure = smooth_coords(coords, pressure, 10)

        outline_l, outline_r, pp = calc_pencil_outline(coords, pressure, lwd)
        log.debug("outline lengths: %d %d %d",
            len(outline_l), len(outline_r), len(pp))

        self.__pressure  = pressure
        self.__coords    = coords
        self.outline(outline_l + outline_r[::-1])

        nbins = 5
        pp = smooth_pressure(pp, 5)
        pp = bin_values(pp, nbins)

        self.__fused, self.__gradients = self.fuse_segments(pp)
        self.__outline_segments = self.construct_segments(self.__fused, outline_l, outline_r)
        self.__midpoints = self.segment_midpoints(self.__outline_segments)

        #log.debug("outline bbox: %s", path_bbox(self.outline()))
        self.bbox(force = True)
        return self.outline()

class BrushPencilV2(Brush):
    """
    Pencil brush, v2.

    This is more or less an experimental pencil.

    This version attempts to draw with the same stroke, but to draw
    """
    def __init__(self, outline = None, bins = None, smooth_path = True): # pylint: disable=unused-argument
        super().__init__(rounded = True, brush_type = "pencil",
                         outline = outline, smooth_path = smooth_path)
        self.__pressure  = [ ]
        self.__bins = [ ]
        self.__bin_lw = [ ]
        self.__outline_l = [ ]
        self.__outline_r = [ ]
        self.__coords = [ ]

    def move(self, dx, dy):
        """Move the outline."""
        self.outline([ (x + dx, y + dy) for x, y in self.outline() ])
        self.__coords = [ (x + dx, y + dy) for x, y in self.__coords ]
        self.__outline_l = [ (x + dx, y + dy) for x, y in self.__outline_l ]
        self.__outline_r = [ (x + dx, y + dy) for x, y in self.__outline_r ]

    def rotate(self, angle, rot_origin):
        """Rotate the outline."""
        self.outline(coords_rotate(self.outline(),   angle, rot_origin))
        self.__coords = coords_rotate(self.__coords, angle, rot_origin)
        self.__outline_l = coords_rotate(self.__outline_l, angle, rot_origin)
        self.__outline_r = coords_rotate(self.__outline_r, angle, rot_origin)

    def scale(self, old_bbox, new_bbox):
        """Scale the outline."""
        self.outline(transform_coords(self.outline(),   old_bbox, new_bbox))
        self.__coords = transform_coords(self.__coords, old_bbox, new_bbox)
        self.__outline_l = transform_coords(self.__outline_l, old_bbox, new_bbox)
        self.__outline_r = transform_coords(self.__outline_r, old_bbox, new_bbox)

    def draw(self, cr, outline = False):
        """Draw the brush on the Cairo context."""
        r, g, b, a = get_current_color_and_alpha(cr)
        if not self.__coords or len(self.__coords) < 2:
            return

        if len(self.__pressure) != len(self.__coords):
            log.warning("Pressure and coords don't match (%d <> %d)",
                    len(self.__pressure), len(self.__coords))
            return

        #print("drawing pencil brush with coords:", len(coords), len(self.__pressure))

        bins   = self.__bins
        outline_l, outline_r = self.__outline_l, self.__outline_r
        n = len(bins)
        #print("n bins:", n, "n outline_l: ", len(outline_l), "n outline_r:", len(outline_r))
        if outline:
            cr.set_line_width(0.4)
            cr.stroke()

        for i in range(n):
            cr.set_source_rgba(r, g, b, a)# * self.__bin_transp[i])
            if not outline:
                cr.set_line_width(self.__bin_lw[i])
            for j in bins[i]:
                #print("i = ", i, "j = ", j)
                if j < len(outline_l) - 1:
                    #print(outline_l[j], outline_r[j])
                    #print(outline_l[j + 1], outline_r[j + 1])
                    cr.move_to(outline_l[j][0], outline_l[j][1])
                    cr.line_to(outline_l[j + 1][0], outline_l[j + 1][1])
                    cr.line_to(outline_r[j + 1][0], outline_r[j + 1][1])
                    cr.line_to(outline_r[j][0], outline_r[j][1])
                    cr.close_path()
                else:
                    log.warning("warning: j out of bounds: %d, %d",
                                j, len(outline_l))
            if outline:
                cr.stroke_preserve()
            else:
                cr.fill()

    def calculate(self, line_width, coords, pressure = None):
        """Recalculate the outline of the brush."""

        if coords is not None and pressure is not None:
            if len(coords) != len(pressure):
                raise ValueError("Pressure and coords don't match")

        pressure = pressure or [1] * len(coords)

        lwd = line_width

        if len(coords) < 2:
            return None

        #print("1.length of coords and pressure:", len(coords), len(pressure))
        if self.smooth_path():
            coords, pressure = smooth_coords(coords, pressure, 20)
        #print("2.length of coords and pressure:", len(coords), len(pressure))

        outline_l, outline_r, pp = calc_pencil_outline(coords, pressure, lwd)
       #print("outline lengths:", len(outline_l), len(outline_r))

        self.__outline_l = outline_l
        self.__outline_r = outline_r
        self.__pressure  = pressure
        self.__coords    = coords
        self.outline(outline_l + outline_r[::-1])

        nbins = 32
        #pp = [pressure[0]] * 5 + pressure + [pressure[-1]] * 5
        plength = len(pp)
        self.__bins, binsize = find_intervals(pp[:(plength - 1)], nbins)

        self.__bin_lw = [ lwd * (0.75 + 0.25 * i * binsize) for i in range(1, nbins + 1) ]

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
