"""Class for different brushes."""
import cairo                                                       # <remove>
from .utils import path_bbox, smooth_path     # <remove>
from .utils import calculate_angle2                                # <remove>
from .utils import coords_rotate, transform_coords                 # <remove>
from .utils import calc_arc_coords, normal_vec                     # <remove>

def get_current_color_and_alpha(ctx):
    pattern = ctx.get_source()
    if isinstance(pattern, cairo.SolidPattern):
        r, g, b, a = pattern.get_rgba()
        return r, g, b, a
    else:
        # For non-solid patterns, return a default or indicate it's not solid
        return None, None, None, None  # Or raise an exception or another appropriate response

def find_intervals(values, n_bins):
    """Divide the range [0, 1] into n_bins and return a list of bins with indices of values falling into each bin."""
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

def calc_normal_outline(coords, pressure, line_width, rounded = False):
    """Calculate the normal outline of a path."""
    n = len(coords)

    outline_l = []
    outline_r = []

    for i in range(n - 2):
        p0, p1, p2 = coords[i], coords[i + 1], coords[i + 2]
        nx, ny = normal_vec(p0, p1)
        mx, my = normal_vec(p1, p2)

        width  = line_width * pressure[i] / 2

        left_segment1_start = (p0[0] + nx * width, p0[1] + ny * width)
        left_segment1_end   = (p1[0] + nx * width, p1[1] + ny * width)
        left_segment2_start = (p1[0] + mx * width, p1[1] + my * width)
        left_segment2_end   = (p2[0] + mx * width, p2[1] + my * width)

        right_segment1_start = (p0[0] - nx * width, p0[1] - ny * width)
        right_segment1_end   = (p1[0] - nx * width, p1[1] - ny * width)
        right_segment2_start = (p1[0] - mx * width, p1[1] - my * width)
        right_segment2_end   = (p2[0] - mx * width, p2[1] - my * width)

        if i == 0:
        ## append the points for the first coord
            if rounded:
                arc_coords = calc_arc_coords( left_segment1_start,
                                              right_segment1_start,
                                             p1, 10)
                outline_r.extend(arc_coords)
            outline_l.append(left_segment1_start)
            outline_r.append(right_segment1_start)

        outline_l.append(left_segment1_end)
        outline_l.append(left_segment2_start)
        outline_r.append(right_segment1_end)
        outline_r.append(right_segment2_start)

        if i == n - 3:
            outline_l.append(left_segment2_end)
            outline_r.append(right_segment2_end)
            if rounded:
                arc_coords = calc_arc_coords( left_segment2_end,
                                              right_segment2_end,
                                             p1, 10)
                outline_l.extend(arc_coords)
    return outline_l, outline_r

def point_mean(p1, p2):
    """Calculate the mean of two points."""
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

def calc_normal_outline2(coords, pressure, line_width, rounded = False):
    """Calculate the normal outline of a path."""
    n = len(coords)

    outline_l = []
    outline_r = []

    for i in range(n - 2):
        p0, p1, p2 = coords[i], coords[i + 1], coords[i + 2]
        nx, ny = normal_vec(p0, p1)
        mx, my = normal_vec(p1, p2)

        width  = line_width * pressure[i] / 2

        left_segment1_start = (p0[0] + nx * width, p0[1] + ny * width)
        left_segment1_end   = (p1[0] + nx * width, p1[1] + ny * width)
        left_segment2_start = (p1[0] + mx * width, p1[1] + my * width)
        left_segment2_end   = (p2[0] + mx * width, p2[1] + my * width)

        right_segment1_start = (p0[0] - nx * width, p0[1] - ny * width)
        right_segment1_end   = (p1[0] - nx * width, p1[1] - ny * width)
        right_segment2_start = (p1[0] - mx * width, p1[1] - my * width)
        right_segment2_end   = (p2[0] - mx * width, p2[1] - my * width)

        if i == 0:
        ## append the points for the first coord
            if rounded:
                arc_coords = calc_arc_coords( left_segment1_start,
                                              right_segment1_start,
                                             p1, 10)
                outline_r.extend(arc_coords)
            outline_l.append(left_segment1_start)
            outline_r.append(right_segment1_start)

        outline_l.append(point_mean(left_segment1_end, left_segment2_start))
        outline_r.append(point_mean(right_segment1_end, right_segment2_start))

        if i == n - 3:
            outline_l.append(left_segment2_end)
            outline_r.append(right_segment2_end)
            if rounded:
                arc_coords = calc_arc_coords( left_segment2_end,
                                              right_segment2_end,
                                             p1, 10)
                outline_l.extend(arc_coords)
    return outline_l, outline_r

class BrushFactory:
    """
    Factory class for creating brushes.
    """
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

        return Brush()

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

class BrushPencil_v1(Brush):
    """
    Pencil brush.

    This is more or less an experimental pencil. 
    """
    def __init__(self, outline = None, bins = None):
        super().__init__(rounded = True, brush_type = "pencil", 
                         outline = outline)
        self.__pressure  = [ ]
        self.__line_width = 1
        self.__bins = [ ]
        self.__bin_lw = [ ]
        self.__bin_transp = [ ]

    def draw(self, cr):
        """Draw the brush on the Cairo context."""
        r, g, b, a = get_current_color_and_alpha(cr)
        if not self.outline() or len(self.outline()) < 4:
            return

        if len(self.__pressure) != len(self.outline()):
            print("Pressure and outline don't match:", len(self.__pressure), len(self.outline()))
            return

        bins   = self.__bins
        coords = self.outline()
        n = len(bins)

        for i in range(n):
            cr.set_source_rgba(r, g, b, a * self.__bin_transp[i])
            cr.set_line_width(self.__bin_lw[i])
            for j in bins[i]:
                cr.move_to(coords[j][0], coords[j][1])
                cr.line_to(coords[j + 1][0], coords[j + 1][1])
            cr.stroke()

    def calculate(self, line_width, coords, pressure = None):
        """Recalculate the outline of the brush."""
        pressure = pressure or [1] * len(coords)

        lwd = line_width
        self.__line_width = lwd

        if len(coords) < 3:
            return None

        coords, pressure = smooth_path(coords, pressure, 20)
        self.__pressure  = pressure

        bin1, bin2, bin3 = [], [], []

        for i in range(len(self.__pressure) - 1):
            p = self.__pressure[i]
            if p < 0.33:
                bin1.append(i)
            elif p < 0.66:
                bin2.append(i)
            else:
                bin3.append(i)

        self.outline(coords)

        self.__bins = [bin1, bin2, bin3]
        self.__bin_lw = [ lwd * 0.5, lwd * 0.75, lwd * 1.0 ]
        self.__bin_transp = [ 0.15, 0.5, 1.0 ]

        return self.outline()


class BrushPencil(Brush):
    """
    Pencil brush, v2.

    This is more or less an experimental pencil. 

    This version attempts to draw with the same stroke, but to draw 
    """
    def __init__(self, outline = None, bins = None):
        super().__init__(rounded = True, brush_type = "pencil", 
                         outline = outline)
        self.__pressure  = [ ]
        self.__line_width = 1
        self.__bins = [ ]
        self.__bin_lw = [ ]
        self.__bin_transp = [ ]

    def to_dict(self):
        """Return a dictionary representation of the brush."""
        return { 
                "brush_type": self.brush_type(),
               }

    def move(self, dx, dy):
        """Move the outline."""
        self.outline([ (x + dx, y + dy) for x, y in self.outline() ])
        self.__outline_l = [ (x + dx, y + dy) for x, y in self.__outline_l ]
        self.__outline_r = [ (x + dx, y + dy) for x, y in self.__outline_r ]

    def rotate(self, angle, rot_origin):
        """Rotate the outline."""
        self.outline(coords_rotate(self.outline(),   angle, rot_origin))
        self.__outline_l = coords_rotate(self.__outline_l, angle, rot_origin)
        self.__outline_r = coords_rotate(self.__outline_r, angle, rot_origin)

    def scale(self, old_bbox, new_bbox):
        """Scale the outline."""
        self.outline(transform_coords(self.outline(),   old_bbox, new_bbox))
        self.__outline_l = transform_coords(self.__outline_l, old_bbox, new_bbox)
        self.__outline_r = transform_coords(self.__outline_r, old_bbox, new_bbox)

    def draw(self, cr):
        """Draw the brush on the Cairo context."""
        r, g, b, a = get_current_color_and_alpha(cr)
        if not self.outline() or len(self.outline()) < 4:
            return

        if len(self.__pressure) != len(self.outline()):
            print("Pressure and outline don't match:", len(self.__pressure), len(self.outline()))
            return

        #print("drawing pencil brush with coords:", len(coords), len(self.__pressure))

        bins   = self.__bins
        coords = self.outline()
        outline_l, outline_r = self.__outline_l, self.__outline_r
        n = len(bins)

        for i in range(n):
            cr.set_source_rgba(r, g, b, a * self.__bin_transp[i])
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
        pressure = pressure or [1] * len(coords)

        lwd = line_width
        self.__line_width = lwd

        if len(coords) < 3:
            return None

        #print("1.length of coords and pressure:", len(coords), len(pressure))
        coords, pressure = smooth_path(coords, pressure, 20)
        #print("2.length of coords and pressure:", len(coords), len(pressure))

        outline_l, outline_r = calc_normal_outline2(coords, pressure, lwd, False)
       #print("outline lengths:", len(outline_l), len(outline_r))

        self.__outline_l = outline_l
        self.__outline_r = outline_r
        self.__pressure  = pressure
        self.outline(coords)

        nbins = 32
        plength = len(self.__pressure)
        self.__bins, binsize = find_intervals(self.__pressure[:(plength - 1)], nbins)

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

        coords, pressure = smooth_path(coords, pressure, 20)

        # we need to multiply everything by line_width
        dx0, dy0, dx1, dy1 = [ line_width * x for x in self.__slant[0] + self.__slant[1] ]

        outline_l, outline_r = calculate_slant_outlines(coords, dx0, dy0, dx1, dy1)
        outline = outline_l + outline_r[::-1]
        self.outline(outline)

        #print("done calculating slanted brush")
        return outline


