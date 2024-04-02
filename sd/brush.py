"""Class for different brushes."""
from .utils import calc_normal_outline, smooth_path                      # <remove>

class BrushFactory:
    """
    Factory class for creating brushes.
    """
    @classmethod
    def create_brush(cls, brush_type):
        """
        Create a brush of the specified type.
        """

        if brush_type == "round":
            return BrushRound()
        else:
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

    def recalculate(self, line_width, coords = None, pressure = None):
        """Recalculate the outline of the brush."""
        if not coords:
            coords = self.__coords
        if not pressure:
            pressure = self.__pressure or [1] * len(coords)

        lwd = line_width

        if len(coords) < 3:
            return

        #print("1.length of coords and pressure:", len(coords), len(pressure))
        coords, pressure = smooth_path(coords, pressure, 20)
        #print("2.length of coords and pressure:", len(coords), len(pressure))

        outline_l, outline_r = calc_normal_outline(coords, pressure, lwd, self.__rounded)

        #outline_l, _ = smooth_path(outline_l, None, 20)
        #outline_r, _ = smooth_path(outline_r, None, 20)
        self.__outline  = outline_l + outline_r[::-1]
        self.__coords   = coords
        self.__pressure = pressure

        if len(self.__coords) != len(self.__pressure):
            #raise ValueError("Pressure and coords don't match")
            print("Pressure and coords don't match:", len(self.__coords), len(self.__pressure))

class BrushRound(Brush):
    """Round brush."""
    def __init__(self):
        super().__init__(rounded = True)

