"""Class for different brushes."""
from .utils import path_bbox, calc_normal_outline, smooth_path                      # <remove>

class BrushFactory:
    """
    Factory class for creating brushes.
    """
    @classmethod
    def create_brush(cls, brush_type):
        """
        Create a brush of the specified type.
        """

        print("BrushFactory brush type:", brush_type)

        if brush_type == "rounded":
            return BrushRound()

        if brush_type == "slanted":
            return BrushSlanted()

        if brush_type == "marker":
            print("returning marker brush")
            return Brush(rounded = False)

        

        return Brush()

class Brush:
    """Base class for brushes."""
    def __init__(self, rounded = False):
        self.__outline = [ ]
        self.__coords = [ ]
        self.__pressure = [ ]
        self.__rounded = rounded

    def coords(self, coords = None):
        """Set or get brush coordinates."""
        if coords is not None:
            self.__coords = coords
        return self.__coords

    def outline(self):
        """Get brush outline."""
        return self.__outline

    def pressure(self, pressure = None):
        """Set or get brush pressure."""
        if pressure is not None:
            self.__pressure = pressure
        return self.__pressure

    def bbox(self):
        """Get bounding box of the brush."""
        return path_bbox(self.__outline)

    def calculate(self, line_width, coords, pressure = None):
        """Recalculate the outline of the brush."""
        pressure = pressure or [1] * len(coords)

        lwd = line_width

        if len(coords) < 3:
            return

        #print("1.length of coords and pressure:", len(coords), len(pressure))
        coords, pressure = smooth_path(coords, pressure, 20)
        #print("2.length of coords and pressure:", len(coords), len(pressure))

        outline_l, outline_r = calc_normal_outline(coords, pressure, lwd, self.__rounded)

        #outline_l, _ = smooth_path(outline_l, None, 20)
        #outline_r, _ = smooth_path(outline_r, None, 20)
        outline  = outline_l + outline_r[::-1]
        coords   = coords
        pressure = pressure

        if len(coords) != len(pressure):
            #raise ValueError("Pressure and coords don't match")
            print("Pressure and coords don't match:", len(coords), len(pressure))
        return outline

class BrushRound(Brush):
    """Round brush."""
    def __init__(self):
        super().__init__(rounded = True)

class BrushSlanted(Brush):
    """Slanted brush."""
    def __init__(self):
        super().__init__()

        self.__slant = (-0.3, 0.6, 0.3, - 0.6)

    def calculate(self, line_width, coords = None, pressure = None):
        """Recalculate the outline of the brush."""

        outline_l, outline_r = [ ], [ ]
        coords, pressure = smooth_path(coords, pressure, 20)
        n = len(coords)

        dx0, dy0, dx1, dy1 = [ x * line_width for x in self.__slant ]

        for p in coords:
            x, y = p
            outline_l.append((x + dx0, y + dy0))
            outline_r.append((x + dx1, y + dy1))

        return outline_l + outline_r[::-1]
