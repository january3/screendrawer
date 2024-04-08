"""
These are the objects that can be displayed. It includes groups, but
also primitives like boxes, paths and text.
"""

import math                              # <remove>
import cairo                             # <remove>
import gi                                 # <remove>
gi.require_version('Gdk', '3.0')        # <remove>
from gi.repository import Gdk            # <remove>
from .pen import Pen                     # <remove>
from .utils import transform_coords, smooth_path, coords_rotate           # <remove>
from .utils import is_click_close_to_path, path_bbox, move_coords, flatten_and_unique # <remove>
from .utils import find_obj_in_bbox     # <remove>
from .texteditor import TextEditor      # <remove>
from .imageobj import ImageObj          # <remove>
from .brush import BrushFactory                   #<remove>


class DrawableFactory:
    """
    Factory class for creating drawable objects.
    """
    @classmethod
    def create_drawable(cls, mode, pen, ev):
        """
        Create a drawable object of the specified type.
        """
        pos = ev.pos()
        pressure = ev.pressure()
        ret_obj = None

        print("create object in mode", mode)
        #if mode == "text" or (mode == "draw" and shift_click and no_active_area):

        if mode == "text":
            print("creating text object")
            ret_obj = Text([ pos ], pen = pen, content = "")
            ret_obj.move_caret("Home")

        elif mode == "draw":
            print("creating path object")
            ret_obj = Path([ pos ], pen = pen, pressure = [ pressure ])

        elif mode == "rectangle":
            print("creating rectangle object")
            ret_obj = Rectangle([ pos ], pen = pen)

        elif mode == "shape":
            print("creating shape object")
            ret_obj = Shape([ pos ], pen = pen)

        elif mode == "circle":
            print("creating circle object")
            ret_obj = Circle([ pos, (pos[0], pos[1]) ], pen = pen)

        else:
            raise ValueError("Unknown mode:", mode)

        return ret_obj

    @classmethod
    def transmute(cls, obj, mode):
        """
        Transmute an object into another type.

        For groups, the behaviour is special: rather than converting the group
        into a single object, we convert all objects within the group into the
        new type by calling the transmute_to method of the group object.
        """
        print("transmuting object to", mode)

        if obj.type == "group":
            # for now, we do not pass transmutations to groups, because
            # we then cannot track the changes.
            return obj

        if mode == "text":
            obj = Text.from_object(obj)
        elif mode == "draw":
            obj = Path.from_object(obj)
        elif mode == "rectangle":
            obj = Rectangle.from_object(obj)
        elif mode == "shape":
            print("calling Shape.from_object")
            obj = Shape.from_object(obj)
        elif mode == "circle":
            obj = Circle.from_object(obj)
        else:
            raise ValueError("Unknown mode:", mode)

        return obj

class Drawable:
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
    def __init__(self, mytype, coords, pen):
        self.type         = mytype
        self.coords       = coords
        self.origin       = None
        self.resizing     = None
        self.rotation     = 0
        self.rot_origin   = None
        self.__filled     = False
        if pen:
            self.pen    = pen.copy()
        else:
            self.pen    = None

    def update(self, x, y, pressure):
        """Called when the mouse moves during drawing."""

    def finish(self):
        """Called when building (drawing, typing etc.) is concluded."""

    def get_primitive(self):
        """This is for allowing to distinguish between primitives and groups."""
        return self

    # ------------ Drawable rotation methods ------------------
    def rotate_start(self, origin):
        """Start the rotation operation."""
        self.rot_origin = origin

    def rotate(self, angle, set_angle = False):
        """Rotate the object by the specified angle."""
        # the self.rotation variable is for drawing while rotating
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

    def resize_update(self, bbox):
        """Update during the resize of the object."""
        self.resizing["bbox"] = bbox

    def resize_end(self):
        """Finish the resizing operation."""
        self.resizing = None
        # not implemented
        print("resize_end not implemented")

    # ------------ Drawable attribute methods ------------------
    def pen_set(self, pen):
        """Set the pen of the object."""
        self.pen = pen.copy()

    def stroke_change(self, direction):
        """Change the stroke size of the object."""
        self.pen.stroke_change(direction)

    def smoothen(self, threshold=20):
        """Smoothen the object."""
        print(f"smoothening not implemented (threshold {threshold})")

    def fill(self):
        """Return the fill status"""
        return self.__filled

    def fill_toggle(self):
        """Toggle the fill of the object."""
        self.__filled = not self.__filled

    def fill_set(self, fill):
        """Fill the object with a color."""
        self.__filled = fill

    def line_width_set(self, lw):
        """Set the color of the object."""
        self.pen.line_width_set(lw)

    def color_set(self, color):
        """Set the color of the object."""
        self.pen.color_set(color)

    def font_set(self, size, family, weight, style):
        """Set the font of the object."""
        self.pen.font_size    = size
        self.pen.font_family  = family
        self.pen.font_weight  = weight
        self.pen.font_style   = style

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

        ## by default, we just check whether the click is close to the bounding box
        # path = [ (x1, y1), (x1, y2), (x2, y2), (x2, y1), (x1, y1) ]
        ## return is_click_close_to_path(click_x, click_y, path, threshold)
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

    def move(self, dx, dy):
        """Move the object by dx, dy."""
        move_coords(self.coords, dx, dy)
        if self.rotation:
            self.rot_origin = (self.rot_origin[0] + dx, self.rot_origin[1] + dy)

    def bbox(self, actual = False):
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

    def draw(self, cr, hover=False, selected=False, outline=False):
        """Draw the object on the Cairo context."""
        raise NotImplementedError("draw method not implemented")

    # ------------ Drawable conversion methods ------------------
    @classmethod
    def from_dict(cls, d):
        """
        Create a drawable object from a dictionary.

        Objects must take all named arguments specified in their
        dictionary.
        """
        type_map = {
            "path": Path,
            "polygon": Shape, #back compatibility
            "shape": Shape,
            "circle": Circle,
            "rectangle": Rectangle,
            "box": Box,
            "image": Image,
            "group": DrawableGroup,
            "text": Text
        }

        obj_type = d.pop("type")
        if obj_type not in type_map:
            raise ValueError("Invalid type:", obj_type)

        if "pen" in d:
            d["pen"] = Pen.from_dict(d["pen"])
        #print("generating object of type", type, "with data", d)
        return type_map.get(obj_type)(**d)

    @classmethod
    def from_object(cls, obj):
        """
        Transmute Drawable object into another class.

        The default method doesn't do much, but subclasses can override it to
        allow conversions between different types of objects.
        """
        print("generic from_obj method called")
        return obj


class DrawableGroup(Drawable):
    """
    Class for creating groups of drawable objects or other groups.
    Most of the time it just passes events around.

    Attributes:
        objects (list): The list of objects in the group.
    """
    def __init__(self, objects = None, objects_dict = None, mytype = "group"):

        if objects is None:
            objects = [ ]
        if objects_dict:
            objects = [ Drawable.from_dict(d) for d in objects_dict ]

        print("Creating DrawableGroup with ", len(objects), "objects")
        super().__init__(mytype, [ (None, None) ], None)
        self.objects = objects

    def contains(self, obj):
        """Check if the group contains the object."""
        return obj in self.objects

    def is_close_to_click(self, click_x, click_y, threshold):
        """Check if a click is close to one of the objects."""
        for obj in self.objects:
            if obj.is_close_to_click(click_x, click_y, threshold):
                return True
        return False

    def fill_toggle(self):
        """Toggle the fill of the objects"""
        for obj in self.objects:
            obj.fill_toggle()

    def stroke_change(self, direction):
        """Change the stroke size of the objects in the group."""
        for obj in self.objects:
            obj.stroke_change(direction)

    def transmute_to(self, mode):
        """Transmute all objects within the group to a new type."""
        print("transmuting group to", mode)
        for i in range(len(self.objects)):
            self.objects[i] = DrawableFactory.transmute(self.objects[i], mode)

    def to_dict(self):
        """Convert the group to a dictionary."""
        return {
            "type": self.type,
            "objects_dict": [ obj.to_dict() for obj in self.objects ],
        }

    def resize_start(self, corner, origin):
        """Start the resizing operation."""
        self.resizing = {
            "corner": corner,
            "origin": origin,
            "bbox":   self.bbox(),
            "orig_bbox": self.bbox(),
            "objects": { obj: obj.bbox() for obj in self.objects }
            }

        for obj in self.objects:
            obj.resize_start(corner, origin)

    def get_primitive(self):
        """Return the primitives of the objects in the group."""
        primitives = [ obj.get_primitive() for obj in self.objects ]
        return flatten_and_unique(primitives)

    def rotate_start(self, origin):
        """Start the rotation operation."""
        self.rot_origin = origin
        for obj in self.objects:
            obj.rotate_start(origin)

    def rotate(self, angle, set_angle = False):
        """Rotate the objects in the group."""
        if set_angle:
            self.rotation = angle
        else:
            self.rotation += angle
        for obj in self.objects:
            obj.rotate(angle, set_angle)

    def rotate_end(self):
        """Finish the rotation operation."""
        for obj in self.objects:
            obj.rotate_end()
        self.rot_origin = None
        self.rotation = 0

    def resize_update(self, bbox):
        """Resize the group of objects. we need to calculate the new
           bounding box for each object within the group"""
        orig_bbox = self.resizing["orig_bbox"]

        scale_x, scale_y = bbox[2] / orig_bbox[2], bbox[3] / orig_bbox[3]

        for obj in self.objects:
            obj_bb = self.resizing["objects"][obj]

            x, y, w, h = obj_bb
            w2, h2 = w * scale_x, h * scale_y

            x2 = bbox[0] + (x - orig_bbox[0]) * scale_x
            y2 = bbox[1] + (y - orig_bbox[1]) * scale_y

            ## recalculate the new bbox of the object within our new bb
            obj.resize_update((x2, y2, w2, h2))

        self.resizing["bbox"] = bbox

    def resize_end(self):
        """Finish the resizing operation."""
        self.resizing = None
        for obj in self.objects:
            obj.resize_end()

    def length(self):
        """Return the number of objects in the group."""
        return len(self.objects)

    def bbox(self, actual = False):
        """Return the bounding box of the group."""
        if self.resizing:
            return self.resizing["bbox"]
        if not self.objects:
            return None

        left, top, width, height = self.objects[0].bbox(actual = actual)
        bottom, right = top + height, left + width

        for obj in self.objects[1:]:
            x, y, w, h = obj.bbox(actual = actual)
            left, top = min(left, x, x + w), min(top, y, y + h)
            bottom, right = max(bottom, y, y + h), max(right, x, x + w)

        width, height = right - left, bottom - top
        return (left, top, width, height)

    def add(self, obj):
        """Add an object to the group."""
        if obj not in self.objects:
            self.objects.append(obj)

    def remove(self, obj):
        """Remove an object from the group."""
        self.objects.remove(obj)

    def move(self, dx, dy):
        """Move the group by dx, dy."""
        for obj in self.objects:
            obj.move(dx, dy)

    def draw(self, cr, hover=False, selected=False, outline=False):
        """Draw the group of objects on the Cairo context."""
        for obj in self.objects:
            obj.draw(cr, hover=False, selected=selected)

        cr.set_source_rgb(0, 0, 0)

        if self.rotation:
            cr.save()
            x, y = self.rot_origin[0], self.rot_origin[1]
            cr.translate(x, y)
            cr.rotate(self.rotation)
            cr.translate(-x, -y)

        if selected:
            cr.set_source_rgb(1, 0, 0)
            self.bbox_draw(cr, lw=.5)
        if hover:
            self.bbox_draw(cr, lw=.5)

        if self.rotation:
            cr.restore()

class SelectionObject(DrawableGroup):
    """
    Class for handling the selection of objects.

    It is an extension of the DrawableGroup class, with additional methods for
    selecting and manipulating objects. Note that more often than not, the
    methods in this class need to have access to the global list of all
    object (e.g. to inverse a selection).

    Attributes:
        objects (list): The list of selected objects.
        _all_objects (list): The list of all objects in the canvas.
    """

    def __init__(self, all_objects):
        super().__init__([ ], None, mytype = "selection_object")

        print("Selection Object with ", len(all_objects), "objects")
        self._all_objects = all_objects

    def copy(self):
        """Return a copy of the selection object."""
        # the copy can be used for undo operations
        print("copying selection to a new selection object")
        return DrawableGroup(self.objects[:])

    def n(self):
        """Return the number of objects in the selection."""
        return len(self.objects)

    def is_empty(self):
        """Check if the selection is empty."""
        return not self.objects

    def clear(self):
        """Clear the selection."""
        self.objects = [ ]

    def set(self, objects):
        """Set the selection to a list of objects."""
        print("setting selection to", objects)
        self.objects = objects

    def add(self, obj):
        """Add an object to the selection."""
        print("adding object to selection:", obj, "selection is", self.objects)
        if not obj in self.objects:
            self.objects.append(obj)

    def all(self):
        """Select all objects."""
        print("selecting everything")
        self.objects = self._all_objects[:]
        print("selection has now", len(self.objects), "objects")
        print("all objects have", len(self._all_objects), "objects")

    def next(self):
        """
        Return a selection object with the next object in the list,
        relative to the current selection.
        """

        all_objects = self._all_objects

        if not all_objects:
            return

        if not self.objects:
            self.objects = [ all_objects[0] ]
            return

        idx = all_objects.index(self.objects[-1])
        idx += 1
        if idx >= len(all_objects):
            idx = 0

        self.objects = [ all_objects[idx] ]


    def prev(self):
        """
        Return a selection object with the previous object in the list,
        relative to the current selection.
        """

        all_objects = self._all_objects

        if not all_objects:
            return

        if not self.objects:
            self.objects = [ all_objects[-1] ]
            return

        idx = all_objects.index(self.objects[-1])
        idx -= 1
        if idx < 0:
            idx = len(all_objects) - 1
        self.objects = [ all_objects[idx] ]


    def reverse(self):
        """
        Return a selection object with the objects in reverse order.
        """
        if not self.objects:
            print("no selection yet, selecting everything")
            self.objects = self._all_objects[:]
            return

        new_sel = [ ]
        for obj in self._all_objects:
            if not self.contains(obj):
                new_sel.append(obj)

        self.objects = new_sel


class Image(Drawable):
    """Class for Images"""
    def __init__(self, coords, pen, image, image_base64 = None, transform = None, rotation = 0):

        self.__image = ImageObj(image, image_base64)

        self.transform = transform or (1, 1)

        width, height = self.__image.size()
        width, height = width * self.transform[0], height * self.transform[1]

        coords = [ (coords[0][0], coords[0][1]),
                   (coords[0][0] + width, coords[0][1] + height) ]
        super().__init__("image", coords, pen)
        self.__orig_bbox = None

        if rotation:
            self.rotation = rotation
            self.rotate_start((coords[0][0] + width / 2, coords[0][1] + height / 2))

    def _bbox_internal(self):
        """Return the bounding box of the object."""
        x, y = self.coords[0]
        w, h = self.coords[1]
        return (x, y, w - x, h - y)

    def draw(self, cr, hover=False, selected=False, outline=False):
        """Draw the object on the Cairo context."""
        cr.save()

        if self.rotation:
            cr.translate(self.rot_origin[0], self.rot_origin[1])
            cr.rotate(self.rotation)
            cr.translate(-self.rot_origin[0], -self.rot_origin[1])

        cr.translate(self.coords[0][0], self.coords[0][1])

        if self.transform:
            w_scale, h_scale = self.transform
            cr.scale(w_scale, h_scale)

        Gdk.cairo_set_source_pixbuf(cr, self.__image.pixbuf(), 0, 0)
        cr.paint()

        cr.restore()

        cr.set_source_rgb(*self.pen.color)
        if selected:
            self.bbox_draw(cr, lw=1.5)
        if hover:
            self.bbox_draw(cr, lw=.5)

    def bbox(self, actual = False):
        """Return the bounding box of the object."""
        bb = self._bbox_internal()
        if self.rotation:
            # first, calculate position bb after rotation relative to the
            # text origin
            x, y, w, h = bb
            x1, y1 = x + w, y + h
            bb = coords_rotate([(x, y), (x, y1), (x1, y), (x1, y1)], self.rotation, self.rot_origin)
            bb = path_bbox(bb)

        return bb

    def resize_start(self, corner, origin):
        """Start the resizing operation."""
        self.__orig_bbox = self.bbox()
        self.resizing = {
            "corner": corner,
            "origin": origin,
            "bbox":   self.bbox(),
            "transform": self.transform
            }

    def resize_update(self, bbox):
        """Update during the resize of the object."""
        self.resizing["bbox"] = bbox

        x1, y1, w1, h1 = bbox

        # calculate scale relative to the old bbox
        print("old bbox is", self.__orig_bbox)
        print("new bbox is", bbox)

        w_scale = w1 / self.__orig_bbox[2]
        h_scale = h1 / self.__orig_bbox[3]

        print("resizing image", w_scale, h_scale)
        print("old transform is", self.resizing["transform"])

        self.coords[0] = (x1, y1)
        self.coords[1] = (x1 + w1, y1 + h1)
        self.transform = (w_scale * self.resizing["transform"][0],
                          h_scale * self.resizing["transform"][1])

    def resize_end(self):
        """Finish the resizing operation."""
        width, height = self.__image.size()
        self.coords[1] = (self.coords[0][0] + width * self.transform[0],
                          self.coords[0][1] + height * self.transform[1])
        self.resizing = None

    def rotate_end(self):
        """Finish the rotation operation."""
        bb = self._bbox_internal()
        center_x, center_y = bb[0] + bb[2] / 2, bb[1] + bb[3] / 2
        new_center = coords_rotate([(center_x, center_y)], self.rotation, self.rot_origin)[0]
        self.move(new_center[0] - center_x, new_center[1] - center_y)
        self.rot_origin = new_center

    def is_close_to_click(self, click_x, click_y, threshold):
        """Check if a click is close to the object."""
        bb = self.bbox()
        if bb is None:
            return False
        x, y, width, height = bb
        if x <= click_x <= x + width and y <= click_y <= y + height:
            return True
        return False

    def to_dict(self):
        """Convert the object to a dictionary."""

        return {
            "type": self.type,
            "coords": self.coords,
            "pen": self.pen.to_dict(),
            "image": None,
            "rotation": self.rotation,
            "transform": self.transform,
            "image_base64": self.__image.base64(),
        }


class Text(Drawable):
    """Class for Text objects"""
    def __init__(self, coords, pen, content, rotation = None, rot_origin = None):
        super().__init__("text", coords, pen)

        # split content by newline
        # content = content.split("\n")
        self.__text = TextEditor(content)
        self.__bb   = None
        self.font_extents = None
        self.__show_caret = False

        if rotation:
            self.rotation = rotation
            self.rot_origin = rot_origin

    def move_caret(self, direction):
        """Move the caret."""
        self.__text.move_caret(direction)
        self.show_caret(True)

    def show_caret(self, show = None):
        """Show the caret."""
        if show is not None:
            self.__show_caret = show
        return self.__show_caret

    def is_close_to_click(self, click_x, click_y, threshold):
        """Check if a click is close to the object."""
        if self.__bb is None:
            return False
        x, y, width, height = self.__bb
        if x <= click_x <= x + width and y <= click_y <= y + height:
            return True
        return False

    def move(self, dx, dy):
        """Move the text object by dx, dy."""
        move_coords(self.coords, dx, dy)
        if self.rotation:
            self.rot_origin = (self.rot_origin[0] + dx, self.rot_origin[1] + dy)

    def rotate_end(self):
        """Finish the rotation operation."""
       ## hmm, not sure what this is supposed to do.
       #if self.bb:
       #    center_x, center_y = self.bb[0] + self.bb[2] / 2, self.bb[1] + self.bb[3] / 2
       #    new_center = coords_rotate([(center_x, center_y)], self.rotation, self.rot_origin)[0]
       #    self.move(new_center[0] - center_x, new_center[1] - center_y)
       #self.rot_origin = new_center

    def stroke_change(self, direction):
        """Change text size up or down."""
        self.pen.font_size += direction
        self.pen.font_size = max(8, min(128, self.pen.font_size))

    def resize_update(self, bbox):
        print("resizing text", bbox)
        if bbox[2] < 0:
            bbox = (bbox[0], bbox[1], 10, bbox[3])
        if bbox[3] < 0:
            print("flipping y")
            bbox = (bbox[0], bbox[1], bbox[2], 10)
        self.resizing["bbox"] = bbox

    def resize_end(self):
        """Finish the resizing operation."""
        new_bbox   = self.resizing["bbox"]
        old_bbox   = self.__bb

        if not self.font_extents:
            return

        # create a surface with the new size
        surface = cairo.ImageSurface(cairo.Format.ARGB32,
                                     2 * math.ceil(new_bbox[2]),
                                     2 * math.ceil(new_bbox[3]))
        cr = cairo.Context(surface)
        min_fs, max_fs = 8, 154

        if new_bbox[2] < old_bbox[2] or new_bbox[3] < old_bbox[3]:
            direction = -1
        else:
            direction = 1

        self.coords = [ (0, 0), (old_bbox[2], old_bbox[3]) ]
        # loop while font size not larger than max_fs and not smaller than
        # min_fs
        print("resizing text, direction=", direction, "font size is", self.pen.font_size)
        while True:
            self.pen.font_size += direction
            print("trying font size", self.pen.font_size)
            self.draw(cr, False, False)
            out_of_range_low = self.pen.font_size < min_fs and direction < 0
            out_of_range_up  = self.pen.font_size > max_fs and direction > 0
            if out_of_range_low or out_of_range_up:
                print("font size out of range")
                break
            current_bbox = self.__bb
            print("drawn, bbox is", self.__bb)
            if direction > 0 and (current_bbox[2] >= new_bbox[2] or
                                  current_bbox[3] >= new_bbox[3]):
                print("increased beyond the new bbox")
                break
            if direction < 0 and (current_bbox[2] <= new_bbox[2] and
                                  current_bbox[3] <= new_bbox[3]):
                break

        self.coords[0] = (new_bbox[0], new_bbox[1] + self.font_extents[0])
        print("final coords are", self.coords)
        print("font extents are", self.font_extents)

        # first
        self.resizing = None

    def to_dict(self):
        return {
            "type": self.type,
            "coords": self.coords,
            "pen": self.pen.to_dict(),
            "rotation": self.rotation,
            "rot_origin": self.rot_origin,
            "content": self.__text.to_string()
        }

    def bbox(self, actual = False):
        if self.resizing:
            return self.resizing["bbox"]
        if not self.__bb:
            bb = (self.coords[0][0], self.coords[0][1], 50, 50)
        else:
            bb = self.__bb
        if self.rotation:
            # first, calculate position bb after rotation relative to the
            # text origin
            x, y, w, h = bb
            x1, y1 = x + w, y + h
            bb = coords_rotate([(x, y), (x, y1), (x1, y), (x1, y1)], self.rotation, self.rot_origin)
            bb = path_bbox(bb)

        return bb

    def to_string(self):
        """Return the text as a single string."""
        return self.__text.to_string()

    def strlen(self):
        """Return the length of the text."""
        return self.__text.strlen()

    def add_text(self, text):
        """Add text to the object."""
        self.__text.add_text(text)

    def update_by_key(self, keyname, char):
        """Update the text object by keypress."""
        self.__text.update_by_key(keyname, char)

    def draw_caret(self, cr, xx0, yy0, height):
        """Draw the caret."""
        # draw the caret
        cr.set_line_width(1)
        cr.move_to(xx0, yy0)
        cr.line_to(xx0, yy0 + height)
        cr.stroke()
        cr.move_to(xx0 - 3, yy0)
        cr.line_to(xx0 + 3, yy0)
        cr.stroke()
        cr.move_to(xx0 - 3, yy0 + height)
        cr.line_to(xx0 + 3, yy0 + height)
        cr.stroke()

    def draw(self, cr, hover=False, selected=False, outline=False):
        """Draw the text object."""
        position = self.coords[0]
        content = self.__text.lines()
        caret_pos = self.__text.caret_pos()

        # get font info
        cr.select_font_face(self.pen.font_family,
                            self.pen.font_style == "italic" and
                                cairo.FONT_SLANT_ITALIC or cairo.FONT_SLANT_NORMAL,
                            self.pen.font_weight == "bold"  and
                                cairo.FONT_WEIGHT_BOLD  or cairo.FONT_WEIGHT_NORMAL)
        cr.set_font_size(self.pen.font_size)

        self.font_extents = cr.font_extents()

        dy   = 0

        # new bounding box
        bb = [position[0],
              position[1] - self.font_extents[0],
              0, 0]

        if self.rotation:
            cr.save()
            cr.translate(self.rot_origin[0], self.rot_origin[1])
            cr.rotate(self.rotation)
            cr.translate(-self.rot_origin[0], -self.rot_origin[1])

        for i in range(len(content)):
            fragment = content[i]

            #x_bearing, y_bearing, t_width, t_height, x_advance, y_advance
            x_bearing, _, t_width, _, _, _ = cr.text_extents(fragment)

            bb[2] = max(bb[2], t_width + x_bearing)
            bb[3] += self.font_extents[2]

            cr.set_font_size(self.pen.font_size)
            cr.move_to(position[0], position[1] + dy)
            cr.set_source_rgba(*self.pen.color, self.pen.transparency)
            cr.show_text(fragment)
            cr.stroke()

            # draw the caret
            if self.__show_caret and not caret_pos is None and i == self.__text.caret_line():
                x_bearing, _, t_width, _, _, _ = cr.text_extents("|" +
                                                        fragment[:caret_pos] + "|")
                _, _, t_width2, _, _, _ = cr.text_extents("|")
                cr.set_source_rgb(1, 0, 0)
                self.draw_caret(cr,
                                position[0] - x_bearing + t_width - 2 * t_width2,
                                position[1] + dy - self.font_extents[0],
                                self.font_extents[2])

            dy += self.font_extents[2]

        self.__bb = (bb[0], bb[1], bb[2], bb[3])

        if self.rotation:
            cr.restore()
        if selected:
            self.bbox_draw(cr, lw=1.5)
        if hover:
            self.bbox_draw(cr, lw=.5)

class Path(Drawable):
    """ Path is like shape, but not closed and has an outline that depends on
        line width and pressure."""
    def __init__(self, coords, pen, outline = None, pressure = None, brush = None):
        super().__init__("path", coords, pen = pen)
        self.__pressure  = pressure or []
        self.__bb        = []

        if brush:
            self.__brush = BrushFactory.create_brush(**brush)
        else:
            brush_type = pen.brush_type()
            self.__brush = BrushFactory.create_brush(brush_type)

       #if outline:
       #    self.__brush.outline(outline)

        if len(self.coords) > 3 and not self.__brush.outline():
            self.outline_recalculate()

    def outline_recalculate(self):
        """Recalculate the outline of the path."""
        self.__brush.calculate(self.pen.line_width,
                                 coords = self.coords,
                                 pressure = self.__pressure)

    def finish(self):
        """Finish the path."""
        self.outline_recalculate()

    def update(self, x, y, pressure):
        """Update the path with a new point."""
        self.path_append(x, y, pressure)

    def move(self, dx, dy):
        """Move the path by dx, dy."""
        move_coords(self.coords, dx, dy)
        #move_coords(self.__outline, dx, dy)
        #self.outline_recalculate()
        self.__brush.move(dx, dy)
        self.__bb = None

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
        print("smoothening path")
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
        coords = self.coords
        width  = self.pen.line_width * pressure

        if len(coords) == 0:
            self.__pressure.append(pressure)
            coords.append((x, y))
            return

        lp = coords[-1]
        if abs(x - lp[0]) < 1 and abs(y - lp[1]) < 1:
            return

        self.__pressure.append(pressure)
        coords.append((x, y))
        width = width / 2

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


    def draw_standard(self, cr):
        """standard drawing of the path."""
        cr.set_fill_rule(cairo.FillRule.WINDING)
        #print("draw_standard")
        self.__brush.draw(cr)

    def draw(self, cr, hover=False, selected=False, outline = False):
        """Draw the path."""
        #print("drawing path", self, "with brush", self.__brush, "of type", self.__brush.brush_type())
        if not self.__brush.outline():
            return

        if len(self.__brush.outline()) < 4 or len(self.coords) < 3:
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
            self.draw_standard(cr)
            if outline:
                print("drawing outline")
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

# ----------------------------
class Shape(Drawable):
    """Class for shapes (closed paths with no outline)."""
    def __init__(self, coords, pen, filled = True):
        super().__init__("shape", coords, pen)
        self.bb = None
        self.fill_set(filled)

    def finish(self):
        """Finish the shape."""
        print("finishing shape")
        self.coords, _ = smooth_path(self.coords)

    def update(self, x, y, pressure):
        """Update the shape with a new point."""
        self.path_append(x, y, pressure)

    def move(self, dx, dy):
        """Move the shape by dx, dy."""
        move_coords(self.coords, dx, dy)
        self.bb = None

    def rotate_end(self):
        """finish the rotation"""
        # rotate all coords and outline
        self.coords  = coords_rotate(self.coords,  self.rotation, self.rot_origin)
        self.rotation   = 0
        self.rot_origin = None
        # recalculate bbox
        self.bb = path_bbox(self.coords)

    def is_close_to_click(self, click_x, click_y, threshold):
        """Check if a click is close to the object."""
        bb = self.bbox()
        if bb is None:
            return False
        x, y, width, height = bb
        if x <= click_x <= x + width and y <= click_y <= y + height:
            return True
        return False

    def path_append(self, x, y, pressure = None):
        """Append a new point to the path."""
        self.coords.append((x, y))
        self.bb = None

    def fill_toggle(self):
        """Toggle the fill of the object."""
        old_bbox = self.bbox(actual = True)
        self.bb  = None
        self.fill_set(not self.fill())
        new_bbox = self.bbox(actual = True)
        self.coords = transform_coords(self.coords, new_bbox, old_bbox)
        self.bb = None

    def bbox(self, actual = False):
        """Calculate the bounding box of the shape."""
        if self.resizing:
            bb = self.resizing["bbox"]
        else:
            if not self.bb:
                self.bb = path_bbox(self.coords)
            bb = self.bb
        if actual and not self.fill():
            lw = self.pen.line_width
            bb = (bb[0] - lw / 2, bb[1] - lw / 2, bb[2] + lw, bb[3] + lw)
        return bb

    def resize_start(self, corner, origin):
        """Start the resizing operation."""
        bbox = path_bbox(self.coords)
        self.resizing = {
            "corner": corner,
            "origin": origin,
            "bbox":   bbox,
            "start_bbox": bbox
            }

    def resize_update(self, bbox):
        """Update during the resize of the object."""
        self.resizing["bbox"] = bbox

    def resize_end(self):
        """recalculate the coordinates after resizing"""
        old_bbox = self.resizing["start_bbox"]
        new_bbox = self.resizing["bbox"]
        self.coords = transform_coords(self.coords, old_bbox, new_bbox)
        self.resizing  = None
        if self.fill():
            self.bb = path_bbox(self.coords)
        else:
            self.bb = path_bbox(self.coords, lw = self.pen.line_width)
        self.bb = path_bbox(self.coords)

    def to_dict(self):
        """Convert the object to a dictionary."""
        return {
            "type": self.type,
            "coords": self.coords,
            "filled": self.fill(),
            "pen": self.pen.to_dict()
        }


    def draw_outline(self, cr):
        """draws each segment separately and makes a dot at each coord."""
        coords = self.coords

        for i in range(len(coords) - 1):
            cr.move_to(coords[i][0], coords[i][1])
            cr.line_to(coords[i + 1][0], coords[i + 1][1])
            cr.stroke()
            # make a dot at each coord
            cr.arc(coords[i][0], coords[i][1], 2, 0, 2 * 3.14159)  # Draw a circle at each point
            cr.fill()
        cr.move_to(coords[-1][0], coords[-1][1])
        cr.line_to(coords[0][0], coords[0][1])
        cr.stroke()

    def draw_simple(self, cr, bbox=None):
        """draws the path as a single line. Useful for resizing."""

        if len(self.coords) < 3:
            return

        if bbox:
            if self.fill():
                old_bbox = path_bbox(self.coords)
            else:
                old_bbox = path_bbox(self.coords)
                #old_bbox = path_bbox(self.coords, lw = self.pen.line_width)
            coords = transform_coords(self.coords, old_bbox, bbox)
        else:
            coords = self.coords

        cr.set_line_width(0.5)
        cr.move_to(coords[0][0], coords[0][1])

        for point in coords[1:]:
            cr.line_to(point[0], point[1])
        cr.close_path()


    def draw(self, cr, hover=False, selected=False, outline = False):
        """Draw the shape on the Cairo context."""
        if len(self.coords) < 3:
            return

        if self.rotation != 0:
            cr.save()
            cr.translate(self.rot_origin[0], self.rot_origin[1])
            cr.rotate(self.rotation)
            cr.translate(-self.rot_origin[0], -self.rot_origin[1])

        cr.set_source_rgba(*self.pen.color, self.pen.transparency)
        res_bb = self.resizing and self.resizing["bbox"] or None
        self.draw_simple(cr, res_bb)

        if outline:
            cr.stroke()
            self.draw_outline(cr)
        elif self.fill():
            cr.fill()
        else:
            cr.set_line_width(self.pen.line_width)
            cr.stroke()

        if selected:
            cr.set_source_rgba(1, 0, 0)
            self.bbox_draw(cr, lw=.2)

        if hover:
            self.bbox_draw(cr, lw=.2)

        if self.rotation != 0:
            cr.restore()

    @classmethod
    def from_object(cls, obj):
        """Create a shape from an object."""
        print("Shape.from_object", obj)
        if obj.coords and len(obj.coords) > 2 and obj.pen:
            return cls(obj.coords, obj.pen)

        # issue a warning
        print("Shape.from_object: invalid object")
        return obj

class Rectangle(Shape):
    """Class for creating rectangles."""
    def __init__(self, coords, pen, filled = False):
        super().__init__(coords, pen, filled)
        self.coords = coords
        self.type = "rectangle"
        # fill up coords to length 4
        n = 5 - len(coords)
        if n > 0:
            self.coords += [(coords[0][0], coords[0][1])] * n

    def finish(self):
        """Finish the rectangle."""
        print("finishing rectangle")
        #self.coords, _ = smooth_path(self.coords)

    def update(self, x, y, pressure):
        """
        Update the rectangle with a new point.

        Unlike the shape, we use four points only to define rectangle.

        We need more than two points, because subsequent transformations
        may change it to a parallelogram.
        """
        x0, y0 = self.coords[0]
        #if x < x0:
        #    x, x0 = x0, x

        #if y < y0:
        #    y, y0 = y0, y

        self.coords[0] = (x0, y0)
        self.coords[1] = (x, y0)
        self.coords[2] = (x, y)
        self.coords[3] = (x0, y)
        self.coords[4] = (x0, y0)


class Circle(Shape):
    """Class for creating circles."""
    def __init__(self, coords, pen, filled = False):
        super().__init__(coords, pen, filled)
        self.coords = coords
        self.type = "circle"
        self.__bb = [ (coords[0][0], coords[0][1]), (coords[0][0], coords[0][1]) ]
        # fill up coords to length 4

    def finish(self):
        """Finish the circle."""

    def update(self, x, y, pressure):
        """
        Update the circle with a new point.
        """
        x0, y0 = min(self.__bb[0][0], x), min(self.__bb[0][1], y)
        x1, y1 = max(self.__bb[1][0], x), max(self.__bb[1][1], y)

        n_points = 100

        # calculate coords for 100 points on an ellipse contained in the rectangle
        # given by x0, y0, x1, y1

        # calculate the center of the ellipse
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2

        # calculate the radius of the ellipse
        rx, ry = (x1 - x0) / 2, (y1 - y0) / 2

        # calculate the angle between two points
        angle = 2 * math.pi / n_points

        # calculate the points
        coords = []
        coords = [ (cx + rx * math.cos(i * angle),
                    cy + ry * math.sin(i * angle)) for i in range(n_points)
                  ]

       #for i in range(n_points):
       #    x = cx + rx * math.cos(i * angle)
       #    y = cy + ry * math.sin(i * angle)
       #    coords.append((x, y))

        self.coords = coords



class Box(Drawable):
    """Class for creating a box."""
    def __init__(self, coords, pen):
        super().__init__("box", coords, pen)

    def update(self, x, y, pressure):
        self.coords[1] = (x, y)

    def resize_end(self):
        bbox = self.bbox()
        self.coords = [ (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]) ]
        self.resizing = None

    def rotate_end(self):
        """Ignore rotation"""

    def rotate_start(self, origin):
        """Ignore rotation."""

    def rotate(self, angle, set_angle = False):
        """Ignore rotation."""

    def resize_update(self, bbox):
        self.resizing["bbox"] = bbox
        self.coords = [ (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]) ]

    def draw(self, cr, hover=False, selected=False, outline=False):

        if hover:
            cr.set_line_width(self.pen.line_width + 1)
        else:
            cr.set_line_width(self.pen.line_width)

        x1, y1 = self.coords[0]
        x2, y2 = self.coords[1]
        w, h = (abs(x1 - x2), abs(y1 - y2))
        x0, y0 = (min(x1, x2), min(y1, y2))

        if self.pen.fill_color:
            cr.set_source_rgba(*self.pen.fill_color, self.pen.transparency)
            cr.rectangle(x0, y0, w, h)
            cr.fill()
            cr.stroke()
        else:
            cr.set_source_rgba(*self.pen.color, self.pen.transparency)
            cr.rectangle(x0, y0, w, h)
            cr.stroke()

        if selected:
            cr.set_line_width(0.5)
            cr.arc(x0, y0, 10, 0, 2 * 3.14159)  # Draw a circle
            #cr.fill()  # Fill the circle to make a dot
            cr.stroke()

        if selected:
            cr.set_source_rgba(1, 0, 0)
            self.bbox_draw(cr, lw=.35)

        if hover:
            self.bbox_draw(cr, lw=.35)

class SelectionTool(Box):
    """Class for creating a box."""
    def __init__(self, coords, pen = None):
        if not pen:
            pen = Pen(line_width = 0.2, color = (1, 0, 0))
        super().__init__(coords, pen)

    def objects_in_selection(self, objects):
        """Return a list of objects that are in the selection."""
        bb  = self.bbox()
        obj = find_obj_in_bbox(bb, objects)
        return obj
