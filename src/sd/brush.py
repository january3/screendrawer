"""Class for different brushes."""
from .utils import path_bbox, calc_normal_outline, smooth_path     # <remove>
from .utils import calculate_angle2                                # <remove>
from .utils import coords_rotate, transform_coords                 # <remove>

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

        return Brush()

class Brush:
    """Base class for brushes."""
    def __init__(self, rounded = False, brush_type = "marker"):
        self.__outline = [ ]
        self.__coords = [ ]
        self.__pressure = [ ]
        self.__rounded = rounded
        self.__outline = [ ]
        self.__brush_type = brush_type

    def to_dict(self):
        """Return a dictionary representation of the brush."""
        return { "brush_type": self.__brush_type }

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
    def __init__(self):
        super().__init__(rounded = False, brush_type = "marker")

class BrushRound(Brush):
    """Round brush."""
    def __init__(self):
        super().__init__(rounded = True, brush_type = "rounded")

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
