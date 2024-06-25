"""Class for different brushes."""
import gi                                                          # <remove>
gi.require_version('Gtk', '3.0')                                   # <remove> pylint: disable=wrong-import-position

import cairo                                                       # <remove>
from .utils import path_bbox, smooth_path                          # <remove>
from .utils import calculate_angle2                                # <remove>
from .utils import coords_rotate, transform_coords                 # <remove>
from .utils import calc_arc_coords, calc_arc_coords2, normal_vec   # <remove>
from .utils import calculate_length, distance                      # <remove>
from .utils import first_point_after_length                        # <remove>
from .utils import midpoint, calc_intersect, calc_intersect_2      # <remove>

def get_current_color_and_alpha(ctx):
    """Get the current color and alpha from the Cairo context."""
    pattern = ctx.get_source()
    if isinstance(pattern, cairo.SolidPattern):
        r, g, b, a = pattern.get_rgba()
        return r, g, b, a
    # For non-solid patterns, return a default or indicate it's not solid
    return None, None, None, None  # Or raise an exception or another appropriate response

def find_intervals(values, n_bins):
    """
    Divide the range [0, 1] into n_bins and return a list of bins with 
    indices of values falling into each bin.
    """

    # Calculate the bin size
    bin_size = 1 / n_bins

    # Initialize bins as a list of empty lists
    bins = [[] for _ in range(n_bins)]

    # Assign each value to a bin
    for i, value in enumerate(values):
        if value == 1:  # Handle the edge case where value is exactly 1
            bins[-1].append(i)
        else:
            bin_index = int(value / bin_size)
            bins[bin_index].append(i)

    return bins, bin_size

def calculate_slant_outlines(coords, dx0, dy0, dx1, dy1):
    """Calculate the left and right outlines."""
    outline_l, outline_r = [ ], [ ]
    slant_vec   = (dx0 - dx1, dy0 - dy1)
    p_prev = coords[0]
    x, y = coords[0]
    outline_l.append((x + dx0, y + dy0))
    outline_r.append((x + dx1, y + dy1))
    prev_cs_angle = None

    for p in coords[1:]:
        x, y = p
        coord_slant_angle = calculate_angle2((x - p_prev[0], y - p_prev[1]), slant_vec)

        # avoid crossing of outlines
        if prev_cs_angle is not None:
            if prev_cs_angle * coord_slant_angle < 0:
                outline_l, outline_r = outline_r, outline_l

        prev_cs_angle = coord_slant_angle

        outline_l.append((x + dx0, y + dy0))
        outline_r.append((x + dx1, y + dy1))
        p_prev = p

    return outline_l, outline_r


def normal_vec_scaled(p0, p1, width):
    """Calculate the normal vector of a line segment."""
    nx, ny = normal_vec(p0, p1)
    nx, ny = nx * width, ny * width
    return nx, ny

def calc_segments_2(p0, p1, width):
    """Calculate the segments of an outline."""
    nx, ny = normal_vec_scaled(p0, p1, width)

    l_seg_s = (p0[0] + nx, p0[1] + ny)
    r_seg_s = (p0[0] - nx, p0[1] - ny)
    l_seg_e = (p1[0] + nx, p1[1] + ny)
    r_seg_e = (p1[0] - nx, p1[1] - ny)

    return (l_seg_s, l_seg_e), (r_seg_s, r_seg_e)

def calc_segments_3(p0, p1, w1, w2):
    """Calculate the segments of an outline."""
    nx1, ny1 = normal_vec_scaled(p0, p1, w1)
    nx2, ny2 = normal_vec_scaled(p0, p1, w2)

    l_seg_s = (p0[0] + nx1, p0[1] + ny1)
    r_seg_s = (p0[0] - nx1, p0[1] - ny1)
    l_seg_e = (p1[0] + nx2, p1[1] + ny2)
    r_seg_e = (p1[0] - nx2, p1[1] - ny2)

    return (l_seg_s, l_seg_e), (r_seg_s, r_seg_e)


def calc_outline_short_generic(coords, pressure, line_width, rounded = False):
    """Calculate the normal outline for a 2-coordinate path"""
    n = len(coords)

    outline_l = []
    outline_r = []
    line_width = line_width or 1

    p0, p1 = coords[0], coords[1]
    width  = line_width * pressure[0] / 2

    l_seg1, r_seg1 = calc_segments_2(p0, p1, width)

    if rounded:
        arc_coords = calc_arc_coords(l_seg1[0], r_seg1[0], p1, 10)
        outline_r.extend(arc_coords)

    outline_l.append(l_seg1[0])
    outline_l.append(l_seg1[1])
    outline_r.append(r_seg1[0])
    outline_r.append(r_seg1[1])

    if rounded:
        arc_coords = calc_arc_coords(l_seg1[1], r_seg1[1], p1, 10)
        outline_l.extend(arc_coords)

    return outline_l, outline_r

def calc_normal_outline_short(coords, widths, rounded = False):
    """Calculate the normal outline for a 2-coordinate path"""
    n = len(coords)

    outline_l = []
    outline_r = []

    p0, p1 = coords[0], coords[1]
    width  = widths[0] / 2

    l_seg1, r_seg1 = calc_segments_2(p0, p1, width)

    if rounded:
        arc_coords = calc_arc_coords(l_seg1[0], r_seg1[0], p1, 10)
        outline_r.extend(arc_coords)

    outline_l.append(l_seg1[0])
    outline_l.append(l_seg1[1])
    outline_r.append(r_seg1[0])
    outline_r.append(r_seg1[1])

    if rounded:
        arc_coords = calc_arc_coords(l_seg1[1], r_seg1[1], p1, 10)
        outline_l.extend(arc_coords)

    return outline_l, outline_r


def calc_normal_outline(coords, widths, rounded = False):
    """Calculate the normal outline of a path."""
    n = len(coords)
    print("CALCULATING NORMAL OUTLINE")

    if n < 2:
        return [], []

    if n == 2:
        return calc_normal_outline_short(coords, widths, rounded)

    outline_l = []
    outline_r = []

    p0, p1 = coords[0], coords[1]
    w1 = widths[0] / 2
    w2 = widths[1] / 2
    l_seg1, r_seg1 = calc_segments_3(p0, p1, w1, w2)

    ## append the points for the first coord
    if rounded:
        arc_coords = calc_arc_coords(l_seg1[0], r_seg1[0], p1, 10)
        outline_r.extend(arc_coords)
    outline_l.append(l_seg1[0])
    outline_r.append(r_seg1[0])

    for i in range(n - 2):
        p2 = coords[i + 2]
        w1 = widths[i + 1] / 2
        w2 = widths[i + 2] / 2

        l_seg2, r_seg2 = calc_segments_3(p1, p2, w1, w2)

        intersect_l = calc_intersect(l_seg1, l_seg2)
        intersect_r = calc_intersect(r_seg1, r_seg2)

        if intersect_l is not None:
            outline_l.append(intersect_l)
        else:
            outline_l.append(l_seg1[1])
            if rounded:
                arc_coords = calc_arc_coords2(l_seg1[1], l_seg2[0], p1, 10)
                outline_l.extend(arc_coords)
            outline_l.append(l_seg2[0])

        if intersect_r is not None:
            outline_r.append(intersect_r)
        else:
            outline_r.append(r_seg1[1])
            if rounded:
                arc_coords = calc_arc_coords2(r_seg1[1], r_seg2[0], p1, 10)
                outline_r.extend(arc_coords)
            outline_r.append(r_seg2[0])

        p0, p1 = p1, p2
        l_seg1, r_seg1 = l_seg2, r_seg2

    outline_l.append(l_seg2[1])
    outline_r.append(r_seg2[1])

    if rounded:
        arc_coords = calc_arc_coords(l_seg1[1], r_seg1[1], p0, 10)
        outline_l.extend(arc_coords)

    return outline_l, outline_r

def point_mean(p1, p2):
    """Calculate the mean of two points."""
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

def calc_normal_outline_pencil(coords, pressure, line_width, rounded = True):
    """
    Calculate the normal outline of a path.

    This one is used in the pencil brush v2.
    """
    #rounded = False
    n = len(coords)

    print("calc_normal_outline_pencil")

    if n < 2:
        return [], [], []

    if n == 2:
        outline_l, outline_r = calc_outline_short_generic(coords, pressure, line_width, rounded)
        return outline_l, outline_r, pressure

    pressure_ret = [ ]
    line_width = line_width or 1
    outline_l = []
    outline_r = []

    p0, p1 = coords[0], coords[1]
    width  = line_width #* pressure[0] / 2
    l_seg1, r_seg1 = calc_segments_2(p0, p1, width)

    ## append the points for the first coord
    if rounded:
        arc_coords = calc_arc_coords(l_seg1[0], r_seg1[0], p1, 10)
        outline_r.extend(arc_coords[:5][::-1])
        outline_l.extend(arc_coords[5:])
        pressure_ret = pressure_ret + [pressure[0]] * 5
    outline_l.append(l_seg1[0])
    outline_r.append(r_seg1[0])
    pressure_ret.append(pressure[0])

    for i in range(n - 2):
        p2 = coords[i + 2]
        width  = line_width #* pressure[i] / 2

        l_seg2, r_seg2 = calc_segments_2(p1, p2, width)

        intersect_l = calc_intersect_2(l_seg1, l_seg2)
        intersect_r = calc_intersect_2(r_seg1, r_seg2)
        outline_l.append(intersect_l)
        outline_r.append(intersect_r)
        pressure_ret.append(pressure[i])

        l_seg1, r_seg1 = l_seg2, r_seg2
        p0, p1 = p1, p2

    outline_l.append(l_seg1[1])
    outline_r.append(r_seg1[1])
    pressure_ret.append(pressure[-1])

    if rounded:
        print("rounding")
        arc_coords = calc_arc_coords(l_seg1[1], r_seg1[1], p0, 10)
        outline_r.extend(arc_coords[:5])
        outline_l.extend(arc_coords[5:][::-1])
        pressure_ret = pressure_ret + [pressure[-1]] * 5

    print("calc_normal_outline_pencil, outline lengths:", len(outline_l), len(outline_r))
    return outline_l, outline_r, pressure_ret

def calc_normal_outline_tapered(coords, pressure, line_width, taper_pos, taper_length):
    """Calculate the normal outline of a path for tapered brush."""
    n = len(coords)

    print("CALCULATING TAPERED NORMAL OUTLINE")

    if n < 2:
        return [], []

    if n == 2:
        return outline_short_generic(coords, pressure, line_width, False)

    line_width = line_width or 1

    outline_l = []
    outline_r = []

    taper_l_cur = 0
    p0, p1 = coords[0], coords[1]

    for i in range(n - 2):
        p2 = coords[i + 2]

        if i >= taper_pos:
            taper_l_cur += distance(p0, p1)
            if taper_l_cur > taper_length:
                taper_l_cur = taper_length
            width  = line_width * pressure[i] / 2 * (1 - taper_l_cur / taper_length)
        else:
            width  = line_width * pressure[i] / 2

        l_seg1, r_seg1 = calc_segments_2(p0, p1, width)
        l_seg2, r_seg2 = calc_segments_2(p1, p2, width)

        if i == 0:
        ## append the points for the first coord
            arc_coords = calc_arc_coords(l_seg1[0], r_seg1[0], p1, 10)
            outline_r.extend(arc_coords)
            outline_l.append(l_seg1[0])
            outline_r.append(r_seg1[0])

        outline_l.append(l_seg1[1])
        outline_l.append(l_seg2[0])
        outline_r.append(r_seg1[1])
        outline_r.append(r_seg2[0])

        if i == n - 3:
            outline_l.append(point_mean(l_seg2[1], r_seg2[1]))
        p0, p1 = p1, p2

    return outline_l, outline_r


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

        print("Selecting BRUSH " + brush_type)
        print("BRUSH KWARGS", kwargs)

        if brush_type == "rounded":
            return BrushRound(**kwargs)

        if brush_type == "slanted":
            return BrushSlanted(**kwargs)

        if brush_type == "marker":
            #print("returning marker brush")
            return BrushMarker(**kwargs)

        if brush_type == "pencil":
            #print("returning pencil brush")
            return BrushPencil(**kwargs)

        if brush_type == "tapered":
            #print("returning tapered brush")
            return BrushTapered(**kwargs)

        raise NotImplementedError("Brush type", brush_type, "not implemented")

class Brush:
    """Base class for brushes."""
    def __init__(self, rounded = False, brush_type = "marker", outline = None, smooth_path = True):
        self.__outline = outline or [ ]
        self.__coords = [ ]
        self.__pressure = [ ]
        self.__rounded = rounded
        self.__outline = [ ]
        self.__brush_type = brush_type
        self.__smooth_path = smooth_path

    def to_dict(self):
        """Return a dictionary representation of the brush."""
        return {
                "brush_type": self.__brush_type,
                "smooth_path": self.smooth_path(),
               }

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
        return self.__outline

    def pressure(self, pressure = None):
        """Set or get brush pressure."""
        if pressure is not None:
            self.__pressure = pressure
        return self.__pressure

    def bbox(self):
        """Get bounding box of the brush."""
        return path_bbox(self.__outline)

    def draw(self, cr, outline = False):
        """Draw the brush on the Cairo context."""
        if not self.__outline or len(self.__outline) < 4:
            return
        cr.move_to(self.__outline[0][0], self.__outline[0][1])
        for point in self.__outline[1:]:
            cr.line_to(point[0], point[1])
        cr.close_path()

    def move(self, dx, dy):
        """Move the outline."""
        self.__outline = [ (x + dx, y + dy) for x, y in self.__outline ]

    def rotate(self, angle, rot_origin):
        """Rotate the outline."""
        self.__outline = coords_rotate(self.__outline, angle, rot_origin)

    def scale(self, old_bbox, new_bbox):
        """Scale the outline."""
        self.__outline = transform_coords(self.__outline, old_bbox, new_bbox)

    def calc_width(self, pressure, lwd):
        """Calculate the width of the brush."""

        widths = [ lwd * (0.25 + p) for p in pressure ]
        return widths

    def calculate(self, line_width, coords, pressure = None):
        """Recalculate the outline of the brush."""

        if coords is not None and pressure is not None:
            if len(coords) != len(pressure):
                print("Brush: Warning: Pressure and coords don't match (",
                                 "coords=", len(coords), "pressure=", len(pressure), ")")
                pressure = None

        pressure = pressure or [1] * len(coords)

        lwd = line_width

        if len(coords) < 2:
            return None

        #print("1.length of coords and pressure:", len(coords), len(pressure))
        if self.smooth_path():
            coords, pressure = smooth_path(coords, pressure, 20)
        #print("2.length of coords and pressure:", len(coords), len(pressure))

        widths = self.calc_width(pressure, lwd)

        outline_l, outline_r = calc_normal_outline(coords, widths, self.__rounded)

        outline  = outline_l + outline_r[::-1]

        if len(coords) != len(pressure):
            #raise ValueError("Pressure and coords don't match")
            print("Pressure and coords don't match:", len(coords), len(pressure))
        self.__outline = outline
        return outline

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
        tot_length = calculate_length(coords)

        #print("1.length of coords and pressure:", len(coords), len(pressure))
        if self.smooth_path():
            coords, pressure = smooth_path(coords, pressure, 20)
        #print("2.length of coords and pressure:", len(coords), len(pressure))

        taper_pos = first_point_after_length(coords, length_to_taper)
        #print("length to taper:", int(length_to_taper), int(tot_length), "taper pos:", taper_pos, "/", len(coords))
        taper_length = calculate_length(coords[taper_pos:])


        outline_l, outline_r = calc_normal_outline_tapered(coords, pressure, lwd, 
                                                    taper_pos, taper_length)

        outline  = outline_l + outline_r[::-1]

        if len(coords) != len(pressure):
            #raise ValueError("Pressure and coords don't match")
            print("Pressure and coords don't match:", len(coords), len(pressure))
        self.outline(outline)
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

class BrushPencil(Brush):
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
        self.__bin_transp = [ ]
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
            print("Pressure and outline don't match:", len(self.__pressure), len(self.__coords))
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
                    print("warning: j out of bounds:", j, len(outline_l))
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
            coords, pressure = smooth_path(coords, pressure, 20)
        #print("2.length of coords and pressure:", len(coords), len(pressure))

        outline_l, outline_r, pp = calc_normal_outline_pencil(coords, pressure, lwd, True)
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
        self.__bin_transp = [ 0.25 + 0.75 * binsize * i for i in range(1, nbins + 1) ]

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

    def calculate(self, line_width = 1, coords = None, pressure = None):
        """Recalculate the outline of the brush."""

        if coords is not None and pressure is not None:
            if len(coords) != len(pressure):
                raise ValueError("Pressure and coords don't match")

        if self.smooth_path():
            coords, pressure = smooth_path(coords, pressure, 20)

        # we need to multiply everything by line_width
        dx0, dy0, dx1, dy1 = [ line_width * x for x in self.__slant[0] + self.__slant[1] ]

        outline_l, outline_r = calculate_slant_outlines(coords, dx0, dy0, dx1, dy1)
        outline = outline_l + outline_r[::-1]
        self.outline(outline)

        #print("done calculating slanted brush")
        return outline
