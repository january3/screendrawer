"""Path is a drawable that is like a shape, but not closed and has an outline"""

import gi                        # <remove>
gi.require_version('Gtk', '3.0') # <remove> pylint: disable=wrong-import-position

import cairo                   # <remove>
from .drawable import Drawable # <remove>
from .brush import BrushFactory                   #<remove>
from .utils import transform_coords, smooth_path, coords_rotate           # <remove>
from .utils import is_click_close_to_path, path_bbox, move_coords # <remove>

class PathRoot(Drawable):
    """ Path is like shape, but not closed and has an outline that depends on
        line width and pressure."""
    def __init__(self, mytype, coords, pen, outline = None, pressure = None, brush = None):
        super().__init__(mytype, coords, pen = pen)
        self.__pressure  = pressure or [1] * len(coords)
        self.__bb        = []
        self.__brush     = None
        self.__n_appended = 0

        if outline:
            print("Warning: outline is not used in Path")

    def brush(self, brush = None):
        """Set the brush for the path."""
        if not brush:
            return self.__brush
        self.__brush = brush

    def outline_recalculate(self):
        """Recalculate the outline of the path."""
        self.__brush.calculate(self.pen.line_width,
                                 coords = self.coords,
                                 pressure = self.__pressure)
        self.mod += 1

    def finish(self):
        """Finish the path."""
        self.outline_recalculate()

    def update(self, x, y, pressure):
        """Update the path with a new point."""
        self.path_append(x, y, pressure)
        self.mod += 1

    def move(self, dx, dy):
        """Move the path by dx, dy."""
        move_coords(self.coords, dx, dy)
        #move_coords(self.__outline, dx, dy)
        #self.outline_recalculate()
        self.__brush.move(dx, dy)
        self.__bb = None
        self.mod += 1

    def rotate_end(self):
        """Finish the rotation operation."""
        # rotate all coords and outline
        self.coords  = coords_rotate(self.coords,  self.rotation, self.rot_origin)
        #self.__outline = coords_rotate(self.__outline, self.rotation, self.rot_origin)
        self.__brush.rotate(self.rotation, self.rot_origin)
        self.outline_recalculate()
        self.rotation   = 0
        self.rot_origin = None
        # recalculate bbox
        self.__bb = path_bbox(self.coords)

    def is_close_to_click(self, click_x, click_y, threshold):
        """Check if a click is close to the path."""
        return is_click_close_to_path(click_x, click_y, self.coords, threshold)

    def to_dict(self):
        """Convert the path to a dictionary."""
        return {
            "type": self.type,
            "coords": self.coords,
            #"outline": self.__brush.outline(),
            "pressure": self.__pressure,
            "pen": self.pen.to_dict(),
            "brush": self.__brush.to_dict()
        }

    def stroke_change(self, direction):
        """Change the stroke size."""
        self.pen.stroke_change(direction)
        self.outline_recalculate()

    def smoothen(self, threshold=20):
        """Smoothen the path."""
        if len(self.coords) < 3:
            return
        self.coords, self.__pressure = smooth_path(self.coords, self.__pressure, 1)
        self.outline_recalculate()

    def pen_set(self, pen):
        """Set the pen for the path."""
        self.pen = pen.copy()
        self.outline_recalculate()

    def path_append(self, x, y, pressure = 1):
        """Append a point to the path, calculating the outline of the
           shape around the path. Only used when path is created to
           allow for a good preview. Later, the path is smoothed and recalculated."""
        #print("path_append. length of coords and pressure:", len(self.coords), len(self.__pressure))
        coords = self.coords
        width  = self.pen.line_width * pressure

        # record the number of append calls (not of the actually appended
        # points)
        self.__n_appended = self.__n_appended + 1
        #print("  appending. __n_appended now=", self.__n_appended)

        if len(coords) == 0:
            self.__pressure.append(pressure)
            coords.append((x, y))
            return

        lp = coords[-1]

        # ignore events too close to the last point
        if abs(x - lp[0]) < 1 and abs(y - lp[1]) < 1:
            #print("  [too close] coords length now:", len(coords))
            #print("  [too close] pressure length now:", len(self.__pressure))
            return

        self.__pressure.append(pressure)
        coords.append((x, y))
        #print("  coords length now:", len(coords))
        #print("  pressure length now:", len(self.__pressure))

        if len(coords) < 2:
            return
        self.outline_recalculate()
        self.__bb = None

    def path_pop(self):
        """Remove the last point from the path."""
        coords = self.coords

        #print("path_pop. __n_appended=", self.__n_appended)
        #print("  path_pop. 1. length of coords and pressure:", len(self.coords), len(self.__pressure))
        if len(coords) < 2:
            return

        self.__n_appended = self.__n_appended - 1

        if self.__n_appended >= len(self.coords):
            return

        self.__pressure.pop()
        coords.pop()
        #print("  path_pop. 2. length of coords and pressure:", len(self.coords), len(self.__pressure))

        if len(coords) < 2:
            return

        self.outline_recalculate()
        self.__bb = None

    def bbox(self, actual = False):
        """Return the bounding box"""
        if self.resizing:
            return self.resizing["bbox"]
        if not self.__bb:
            self.__bb = path_bbox(self.__brush.outline() or self.coords)
        return self.__bb

    def resize_end(self):
        """recalculate the outline after resizing"""
        #print("length of coords and pressure:", len(self.coords), len(self.__pressure))
        old_bbox = self.__bb or path_bbox(self.coords)
        self.coords = transform_coords(self.coords, old_bbox, self.resizing["bbox"])
        self.outline_recalculate()
        #self.pen.brush().scale(old_bbox, self.resizing["bbox"])
        self.resizing  = None
        self.__bb = path_bbox(self.__brush.outline() or self.coords)

    def draw_outline(self, cr):
        """draws each segment separately and makes a dot at each coord."""

        cr.set_source_rgb(1, 0, 0)
        cr.set_line_width(0.2)
        coords = self.coords
        for i in range(len(coords) - 1):
            cr.move_to(coords[i][0], coords[i][1])
            cr.line_to(coords[i + 1][0], coords[i + 1][1])
            cr.stroke()
            # make a dot at each coord
            cr.arc(coords[i][0], coords[i][1], 2, 0, 2 * 3.14159)  # Draw a circle at each point
            cr.fill()
        cr.arc(coords[-1][0], coords[-1][1], 2, 0, 2 * 3.14159)  # Draw a circle at each point
        cr.fill()

    def draw_simple(self, cr, bbox=None):
        """draws the path as a single line. Useful for resizing."""

        if len(self.coords) < 2:
            return

        if bbox:
            old_bbox = path_bbox(self.__brush.outline() or self.coords)
            coords = transform_coords(self.coords, old_bbox, bbox)
        else:
            coords = self.coords

        cr.set_source_rgb(*self.pen.color)
        cr.set_line_width(self.pen.line_width)

        cr.move_to(coords[0][0], coords[0][1])
        for point in coords[1:]:
            cr.line_to(point[0], point[1])
        cr.stroke()

    def draw_standard(self, cr, outline = False):
        """standard drawing of the path."""
        cr.set_fill_rule(cairo.FillRule.WINDING)
        #print("draw_standard")
        self.__brush.draw(cr, outline)

    def draw(self, cr, hover=False, selected=False, outline = False):
        """Draw the path."""
        #print("drawing path", self, "with brush", self.__brush, "of type",
        # self.__brush.brush_type())
        if not self.__brush.outline():
            print("Warning: no outline for brush", self.__brush.brush_type())
            return

        if len(self.__brush.outline()) < 2 or len(self.coords) < 2:
            print("Warning: not enough coords or outline for path")
            return

        if self.rotation != 0:
            cr.save()
            cr.translate(self.rot_origin[0], self.rot_origin[1])
            cr.rotate(self.rotation)
            cr.translate(-self.rot_origin[0], -self.rot_origin[1])

        cr.set_source_rgba(*self.pen.color, self.pen.transparency)
        if self.resizing:
            self.draw_simple(cr, bbox=self.resizing["bbox"])
        else:
            self.draw_standard(cr, outline)
            if outline:
                cr.set_line_width(0.4)
                cr.stroke()
                self.draw_outline(cr)
            else:
                cr.fill()

        if selected:
            cr.set_source_rgba(1, 0, 0)
            self.bbox_draw(cr, lw=.2)

        if hover:
            self.bbox_draw(cr, lw=.2)

        if self.rotation != 0:
            cr.restore()

    @classmethod
    def from_object(cls, obj):
        print("Path.from_object", obj)
        if obj.coords and len(obj.coords) > 2 and obj.pen:
            return cls(obj.coords, obj.pen)
        # issue a warning
        print("Path.from_object: invalid object")
        return obj

class Path(PathRoot):
    """ Path is like shape, but not closed and has an outline that depends on
        line width and pressure."""
    def __init__(self, coords, pen, outline = None, pressure = None, brush = None):
        super().__init__("path", coords, pen = pen, pressure = pressure)

        if brush:
            self.brush(BrushFactory.create_brush(**brush))
        else:
            brush_type = pen.brush_type()
            self.brush(BrushFactory.create_brush(brush_type))

        if len(self.coords) > 3 and not self.brush().outline():
            self.outline_recalculate()


class SegmentedPath(PathRoot):
    """Path with no smoothing at joints."""
    def __init__(self, coords, pen, outline = None, pressure = None, brush = None):
        super().__init__("segmented_path", coords, pen = pen, pressure = pressure)
        self.__pressure  = pressure or []
        self.__bb        = []

        if brush:
            print(brush)
            brush['smooth_path'] = False
            self.brush(BrushFactory.create_brush(**brush))
        else:
            brush_type = pen.brush_type()
            self.brush(BrushFactory.create_brush(brush_type, smooth_path = False))

        if len(self.coords) > 1 and not self.brush().outline():
            self.outline_recalculate()



Drawable.register_type("path", Path)
Drawable.register_type("segmented_path", SegmentedPath)
