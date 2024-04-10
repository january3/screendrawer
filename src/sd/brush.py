"""Class for different brushes."""
import gi                                                          # <remove>
gi.require_version('Gtk', '3.0')                                   # <remove> pylint: disable=wrong-import-position

import cairo                                                       # <remove>
from .utils import path_bbox, smooth_path                          # <remove>
from .utils import calculate_angle2                                # <remove>
from .utils import coords_rotate, transform_coords                 # <remove>
from .utils import calc_arc_coords, normal_vec                     # <remove>
from .utils import calculate_length, distance                      # <remove>
from .utils import first_point_after_length                        # <remove>

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

def calc_segments(p0, p1, width):
    """Calculate the segments of an outline."""
    nx, ny = normal_vec_scaled(p0, p1, width)

    l_seg_s = (p0[0] + nx, p0[1] + ny)
    r_seg_s = (p0[0] - nx, p0[1] - ny)
    l_seg_e = (p1[0] + nx, p1[1] + ny)
    r_seg_e = (p1[0] - nx, p1[1] - ny)

    return l_seg_s, r_seg_s, l_seg_e, r_seg_e


def calc_normal_outline(coords, pressure, line_width, rounded = False):
    """Calculate the normal outline of a path."""
    n = len(coords)

    outline_l = []
    outline_r = []

    for i in range(n - 2):
        p0, p1, p2 = coords[i], coords[i + 1], coords[i + 2]
        width  = line_width * pressure[i] / 2

        l_seg1_s, r_seg1_s, l_seg1_e, r_seg1_e = calc_segments(p0, p1, width)
        l_seg2_s, r_seg2_s, l_seg2_e, r_seg2_e = calc_segments(p1, p2, width)

        if i == 0:
        ## append the points for the first coord
            if rounded:
                arc_coords = calc_arc_coords(l_seg1_s, r_seg1_s, p1, 10)
                outline_r.extend(arc_coords)
            outline_l.append(l_seg1_s)
            outline_r.append(r_seg1_s)

        outline_l.append(l_seg1_e)
        outline_l.append(l_seg2_s)
        outline_r.append(r_seg1_e)
        outline_r.append(r_seg2_s)

        if i == n - 3:
            outline_l.append(l_seg2_e)
            outline_r.append(r_seg2_e)
            if rounded:
                arc_coords = calc_arc_coords(l_seg2_e, r_seg2_e, p1, 10)
                outline_l.extend(arc_coords)
    return outline_l, outline_r

def point_mean(p1, p2):
    """Calculate the mean of two points."""
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

def calc_normal_outline2(coords, pressure, line_width, rounded = True):
    """
    Calculate the normal outline of a path.

    This one is used in the pencil brush.
    """
    n = len(coords)

    outline_l = []
    outline_r = []

    print("calculating for coords:", len(coords), "pressure:", len(pressure))
    for i in range(n - 2):
        p0, p1, p2 = coords[i], coords[i + 1], coords[i + 2]
        width  = line_width * pressure[i] / 2

        l_seg1_s, r_seg1_s, l_seg1_e, r_seg1_e = calc_segments(p0, p1, width)
        l_seg2_s, r_seg2_s, l_seg2_e, r_seg2_e = calc_segments(p1, p2, width)

        if i == 0:
        ## append the points for the first coord
            if rounded:
                arc_coords = calc_arc_coords(l_seg1_s, r_seg1_s, p1, 10)
                outline_r.extend(arc_coords[:5][::-1])
                outline_l.extend(arc_coords[5:])
            outline_l.append(l_seg1_s)
            outline_r.append(r_seg1_s)

        outline_l.append(point_mean(l_seg1_e, l_seg2_s))
        outline_r.append(point_mean(r_seg1_e, r_seg2_s))

        if i == n - 3:
            outline_l.append(l_seg2_e)
            outline_r.append(r_seg2_e)
            if rounded:
                print("rounding")
                arc_coords = calc_arc_coords(l_seg2_e, r_seg2_e, p1, 10)
                outline_r.extend(arc_coords[:5])
                outline_l.extend(arc_coords[5:][::-1])
    return outline_l, outline_r

def calc_normal_outline3(coords, pressure, line_width, taper_pos, taper_length):
    """Calculate the normal outline of a path."""
    n = len(coords)

    outline_l = []
    outline_r = []

    taper_l_cur = 0

    for i in range(n - 2):
        p0, p1, p2 = coords[i], coords[i + 1], coords[i + 2]
        if i >= taper_pos:
            taper_l_cur += distance(p0, p1)
            if taper_l_cur > taper_length:
                taper_l_cur = taper_length
            width  = line_width * pressure[i] / 2 * (1 - taper_l_cur / taper_length)
        else:
            width  = line_width * pressure[i] / 2

        l_seg1_s, r_seg1_s, l_seg1_e, r_seg1_e = calc_segments(p0, p1, width)
        l_seg2_s, r_seg2_s, l_seg2_e, r_seg2_e = calc_segments(p1, p2, width)

        if i == 0:
        ## append the points for the first coord
            arc_coords = calc_arc_coords(l_seg1_s, r_seg1_s, p1, 10)
            outline_r.extend(arc_coords)
            outline_l.append(l_seg1_s)
            outline_r.append(r_seg1_s)

        outline_l.append(l_seg1_e)
        outline_l.append(l_seg2_s)
        outline_r.append(r_seg1_e)
        outline_r.append(r_seg2_s)

        if i == n - 3:
            outline_l.append(point_mean(l_seg2_e, r_seg2_e))
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

        print("BrushFactory brush type:", brush_type)

        if brush_type == "rounded":
            return BrushRound(**kwargs)

        if brush_type == "slanted":
            return BrushSlanted(**kwargs)

        if brush_type == "marker":
            print("returning marker brush")
            return BrushMarker(**kwargs)

        if brush_type == "pencil":
            print("returning pencil brush")
            return BrushPencil(**kwargs)

        if brush_type == "tapered":
            print("returning tapered brush")
            return BrushTapered(**kwargs)

        raise NotImplementedError("Brush type not implemented")

class Brush:
    """Base class for brushes."""
    def __init__(self, rounded = False, brush_type = "marker", outline = None):
        self.__outline = outline or [ ]
        self.__coords = [ ]
        self.__pressure = [ ]
        self.__rounded = rounded
        self.__outline = [ ]
        self.__brush_type = brush_type

    def to_dict(self):
        """Return a dictionary representation of the brush."""
        return {
                "brush_type": self.__brush_type,
               # "outline": self.__outline,
               }

    def brush_type(self):
        """Get brush type."""
        return self.__brush_type

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

    def draw(self, cr):
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

    def calculate(self, line_width, coords, pressure = None):
        """Recalculate the outline of the brush."""

        if coords is not None and pressure is not None:
            if len(coords) != len(pressure):
                raise ValueError("Pressure and coords don't match")

        pressure = pressure or [1] * len(coords)

        lwd = line_width

        if len(coords) < 3:
            return None

        #print("1.length of coords and pressure:", len(coords), len(pressure))
        coords, pressure = smooth_path(coords, pressure, 20)
        #print("2.length of coords and pressure:", len(coords), len(pressure))

        outline_l, outline_r = calc_normal_outline(coords, pressure, lwd, self.__rounded)

        #outline_l, _ = smooth_path(outline_l, None, 20)
        #outline_r, _ = smooth_path(outline_r, None, 20)
        outline  = outline_l + outline_r[::-1]

        if len(coords) != len(pressure):
            #raise ValueError("Pressure and coords don't match")
            print("Pressure and coords don't match:", len(coords), len(pressure))
        self.__outline = outline
        return outline

class BrushTapered(Brush):
    """Tapered brush."""
    def __init__(self, outline = None):
        super().__init__(rounded = False, brush_type = "tapered",
                         outline = outline)

    def calculate(self, line_width, coords, pressure = None):
        """Recalculate the outline of the brush."""

        if coords is not None and pressure is not None:
            if len(coords) != len(pressure):
                raise ValueError("Pressure and coords don't match")

        pressure = pressure or [1] * len(coords)

        lwd = line_width

        if len(coords) < 4:
            return None

        n_taper = 5
        if len(coords) < n_taper:
            n_taper = len(coords) - 2

        length_to_taper = calculate_length(coords[:(len(coords) - n_taper)])
        tot_length = calculate_length(coords)

        #print("1.length of coords and pressure:", len(coords), len(pressure))
        coords, pressure = smooth_path(coords, pressure, 20)
        #print("2.length of coords and pressure:", len(coords), len(pressure))

        taper_pos = first_point_after_length(coords, length_to_taper)
        print("length to taper:", int(length_to_taper), int(tot_length), "taper pos:", taper_pos, "/", len(coords))
        taper_length = calculate_length(coords[taper_pos:])


        #outline_l, outline_r = calc_normal_outline(coords, pressure, lwd, True)
        outline_l, outline_r = calc_normal_outline3(coords, pressure, lwd, 
                                                    taper_pos, taper_length)

        #outline_l, _ = smooth_path(outline_l, None, 20)
        #outline_r, _ = smooth_path(outline_r, None, 20)
        outline  = outline_l + outline_r[::-1]

        if len(coords) != len(pressure):
            #raise ValueError("Pressure and coords don't match")
            print("Pressure and coords don't match:", len(coords), len(pressure))
        self.outline(outline)
        return outline

class BrushMarker(Brush):
    """Marker brush."""
    def __init__(self, outline = None):
        super().__init__(rounded = False, brush_type = "marker",
                         outline = outline)

class BrushRound(Brush):
    """Round brush."""
    def __init__(self, outline = None):
        super().__init__(rounded = True, brush_type = "rounded",
                         outline = outline)

class BrushPencil(Brush):
    """
    Pencil brush, v2.

    This is more or less an experimental pencil.

    This version attempts to draw with the same stroke, but to draw
    """
    def __init__(self, outline = None, bins = None): # pylint: disable=unused-argument
        super().__init__(rounded = True, brush_type = "pencil",
                         outline = outline)
        self.__pressure  = [ ]
        self.__bins = [ ]
        self.__bin_lw = [ ]
        self.__bin_transp = [ ]
        self.__outline_l = [ ]
        self.__outline_r = [ ]
        self.__coords = [ ]

    def to_dict(self):
        """Return a dictionary representation of the brush."""
        return {
                "brush_type": self.brush_type(),
               }

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

    def draw(self, cr):
        """Draw the brush on the Cairo context."""
        r, g, b, a = get_current_color_and_alpha(cr)
        if not self.__coords or len(self.__coords) < 4:
            return

        if len(self.__pressure) != len(self.__coords):
            print("Pressure and outline don't match:", len(self.__pressure), len(self.__coords))
            return

        #print("drawing pencil brush with coords:", len(coords), len(self.__pressure))

        bins   = self.__bins
        outline_l, outline_r = self.__outline_l, self.__outline_r
        n = len(bins)

        for i in range(n):
            cr.set_source_rgba(r, g, b, a)# * self.__bin_transp[i])
            cr.set_line_width(self.__bin_lw[i])
            for j in bins[i]:
                cr.move_to(outline_l[j][0], outline_l[j][1])
                cr.line_to(outline_l[j + 1][0], outline_l[j + 1][1])
                cr.line_to(outline_r[j + 1][0], outline_r[j + 1][1])
                cr.line_to(outline_r[j][0], outline_r[j][1])
                cr.close_path()
            cr.fill()

    def calculate(self, line_width, coords, pressure = None):
        """Recalculate the outline of the brush."""

        if coords is not None and pressure is not None:
            if len(coords) != len(pressure):
                raise ValueError("Pressure and coords don't match")

        pressure = pressure or [1] * len(coords)

        lwd = line_width

        if len(coords) < 3:
            return None

        #print("1.length of coords and pressure:", len(coords), len(pressure))
        coords, pressure = smooth_path(coords, pressure, 20)
        #print("2.length of coords and pressure:", len(coords), len(pressure))

        outline_l, outline_r = calc_normal_outline2(coords, pressure, lwd, True)
       #print("outline lengths:", len(outline_l), len(outline_r))

        self.__outline_l = outline_l
        self.__outline_r = outline_r
        self.__pressure  = pressure
        self.__coords    = coords
        self.outline(outline_l + outline_r[::-1])

        nbins = 32
        pp = [pressure[0]] * 5 + pressure + [pressure[-1]] * 5
        plength = len(pp)
        self.__bins, binsize = find_intervals(pp[:(plength - 1)], nbins)

        self.__bin_lw = [ lwd * (0.75 + 0.25 * i * binsize) for i in range(1, nbins + 1) ]
        self.__bin_transp = [ 0.25 + 0.75 * binsize * i for i in range(1, nbins + 1) ]

        return self.outline()


class BrushSlanted(Brush):
    """Slanted brush."""
    def __init__(self, slant = None):
        super().__init__(brush_type = "slanted")

        self.__slant = slant or [(-0.4, 0.6), (0.3, - 0.6)]

    def to_dict(self):
        """Return a dictionary representation of the brush."""
        return { "brush_type": self.brush_type(), "slant": self.__slant }

    def rotate(self, angle, rot_origin):
        """Rotate the outline."""
        #self.outline(coords_rotate(self.outline(), angle, rot_origin))
        self.__slant = coords_rotate(self.__slant, angle, (0, 0))

    def calculate(self, line_width = 1, coords = None, pressure = None):
        """Recalculate the outline of the brush."""

        if coords is not None and pressure is not None:
            if len(coords) != len(pressure):
                raise ValueError("Pressure and coords don't match")

        coords, pressure = smooth_path(coords, pressure, 20)

        # we need to multiply everything by line_width
        dx0, dy0, dx1, dy1 = [ line_width * x for x in self.__slant[0] + self.__slant[1] ]

        outline_l, outline_r = calculate_slant_outlines(coords, dx0, dy0, dx1, dy1)
        outline = outline_l + outline_r[::-1]
        self.outline(outline)

        #print("done calculating slanted brush")
        return outline
