"""
These are the objects that can be displayed. It includes groups, but
also primitives like boxes, paths and text.
"""

from .pen import Pen           # <remove>
from .utils import move_coords # <remove>


class DrawableRoot:
    """
    Dummy class for the root of the drawable object hierarchy.
    """
    def __init__(self, mytype, coords):
        self.type = mytype
        self.coords = coords
        self.mod  = 0
        self.origin       = None
        self.resizing     = None
        self.rotation     = 0
        self.rot_origin   = None

    def update(self, x, y, pressure): # pylint: disable=unused-argument
        """Called when the mouse moves during drawing."""
        self.mod += 1

    def finish(self):
        """Called when building (drawing, typing etc.) is concluded."""
        self.mod += 1

    def get_primitive(self):
        """This is for allowing to distinguish between primitives and groups."""
        return self

    # ------------ Drawable rotation methods ------------------
    def rotate_start(self, origin):
        """Start the rotation operation."""
        self.rot_origin = origin
        self.mod += 1

    def rotate(self, angle, set_angle = False):
        """Rotate the object by the specified angle."""
        # the self.rotation variable is for drawing while rotating
        self.mod += 1
        if set_angle:
            self.rotation = angle
        else:
            self.rotation += angle

    def rotate_end(self):
        """Finish the rotation operation."""
        raise NotImplementedError("rotate_end method not implemented")

    # ------------ Drawable resizing methods ------------------
    def resize_start(self, corner, origin):
        """Start the resizing operation."""
        self.resizing = {
            "corner": corner,
            "origin": origin,
            "bbox":   self.bbox()
            }
        self.mod += 1

    def resize_update(self, bbox):
        """Update during the resize of the object."""
        self.resizing["bbox"] = bbox
        self.mod += 1

    def resize_end(self):
        """Finish the resizing operation."""
        self.resizing = None
        # not implemented
        print("resize_end not implemented")
        self.mod += 1

    def bbox(self, actual = False): # pylint: disable=unused-argument
        """Return the bounding box of the object."""
        if self.resizing:
            return self.resizing["bbox"]
        left, top = min(p[0] for p in self.coords), min(p[1] for p in self.coords)
        width =    max(p[0] for p in self.coords) - left
        height =   max(p[1] for p in self.coords) - top
        return (left, top, width, height)

    def bbox_draw(self, cr, lw=0.2):
        """Draw the bounding box of the object."""
        bb = self.bbox(actual = True)
        x, y, w, h = bb
        cr.set_line_width(lw)
        cr.rectangle(x, y, w, h)
        cr.stroke()

    def move(self, dx, dy):
        """Move the object by dx, dy."""
        move_coords(self.coords, dx, dy)
        if self.rotation:
            self.rot_origin = (self.rot_origin[0] + dx, self.rot_origin[1] + dy)
        self.mod += 1

class Drawable(DrawableRoot):
    """
    Base class for drawable objects.

    This class represents a drawable object that can be displayed on a canvas.

    Attributes:
        type (str): The type of the drawable object.
        coords (list of tuples): The coordinates of the object's shape.
        origin (tuple): The original position of the object (when resizing etc).
        resizing (dict): The state of the object's resizing operation.
        rotation (float): The rotation angle of the object in radians.
        rot_origin (tuple): The origin of the rotation operation.
        pen (Pen): The pen used for drawing the object.
    """
    __registry = { }

    def __init__(self, mytype, coords, pen):
        super().__init__(mytype, coords)

        self.__filled     = False
        if pen:
            self.pen    = pen.copy()
        else:
            self.pen    = None
        
        self.__modified = None

    # ------------ Drawable attribute methods ------------------
    def modified(self, mod=False):
        """Was the object modified?"""
        if mod:
            self.mod += 1
        status = self.mod != self.__modified
        self.__modified = self.mod

        return status

    def pen_set(self, pen):
        """Set the pen of the object."""
        self.pen = pen.copy()
        self.mod += 1

    def stroke_change(self, direction):
        """Change the stroke size of the object."""
        self.pen.stroke_change(direction)
        self.mod += 1

    def smoothen(self, threshold=20):
        """Smoothen the object."""
        print(f"smoothening not implemented (threshold {threshold})")
        self.mod += 1

    def fill(self):
        """Return the fill status"""
        return self.__filled

    def fill_toggle(self):
        """Toggle the fill of the object."""
        self.mod += 1
        self.__filled = not self.__filled

    def fill_set(self, fill):
        """Fill the object with a color."""
        self.mod += 1
        self.__filled = fill

    def line_width_set(self, lw):
        """Set the color of the object."""
        self.mod += 1
        self.pen.line_width_set(lw)

    def color_set(self, color):
        """Set the color of the object."""
        self.mod += 1
        self.pen.color_set(color)

    def font_set(self, size, family, weight, style):
        """Set the font of the object."""
        self.pen.font_size    = size
        self.pen.font_family  = family
        self.pen.font_weight  = weight
        self.pen.font_style   = style
        self.mod += 1

    # ------------ Drawable modification methods ------------------
    def is_close_to_click(self, click_x, click_y, threshold):
        """Check if a click is close to the object."""
        if self.coords is None:
            return False
        if len(self.coords) == 1:
            x, y = self.coords[0]
            return (x - threshold <= click_x <= x + threshold and
                    y - threshold <= click_y <= y + threshold)

        x1, y1 = self.coords[0]
        x2, y2 = self.coords[1]

        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        # we return True if click is within the bbox
        return (x1 - threshold <= click_x <= x2 + threshold and
                y1 - threshold <= click_y <= y2 + threshold)

    def to_dict(self):
        """Convert the object to a dictionary."""
        return {
            "type": self.type,
            "coords": self.coords,
            "pen": self.pen.to_dict()
        }

    def draw(self, cr, hover=False, selected=False, outline=False):
        """Draw the object on the Cairo context."""
        raise NotImplementedError("draw method not implemented")

    # ------------ Drawable conversion methods ------------------
    @classmethod
    def register_type(cls, obj_type, obj_class):
        """Register a new drawable object class."""
        cls.__registry[obj_type] = obj_class

    @classmethod
    def from_dict(cls, d):
        """
        Create a drawable object from a dictionary.

        Objects must take all named arguments specified in their
        dictionary.
        """

        type_map = cls.__registry

        obj_type = d.pop("type")
        print("Generating object of type:", obj_type)
        if obj_type not in type_map:
            raise ValueError("Invalid type:", obj_type)

        if "pen" in d:
            d["pen"] = Pen.from_dict(d["pen"])

        return type_map.get(obj_type)(**d)
