"""
These classes are the primitives for drawing: text, shapes, paths
"""

import math                             # <remove>
import gi                               # <remove>
gi.require_version('Gdk', '3.0')        # <remove> pylint: disable=wrong-import-position
from gi.repository import Gdk           # <remove>
import cairo                            # <remove>

from .drawable import Drawable # <remove>
from .pen import Pen           # <remove>
from .texteditor import TextEditor             # <remove>
from .imageobj import ImageObj                 # <remove>
from .utils import path_bbox, move_coords      # <remove>
from .utils import find_obj_in_bbox            # <remove>
from .utils import transform_coords            # <remove>
from .utils import smooth_path, coords_rotate  # <remove>

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
            bb = coords_rotate([(x, y), (x, y1), (x1, y), (x1, y1)],
                               self.rotation, self.rot_origin)
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
        self.mod += 1

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
        self.mod += 1

    def resize_end(self):
        """Finish the resizing operation."""
        width, height = self.__image.size()
        self.coords[1] = (self.coords[0][0] + width * self.transform[0],
                          self.coords[0][1] + height * self.transform[1])
        self.resizing = None
        self.mod += 1

    def rotate_end(self):
        """Finish the rotation operation."""
        bb = self._bbox_internal()
        center_x, center_y = bb[0] + bb[2] / 2, bb[1] + bb[3] / 2
        new_center = coords_rotate([(center_x, center_y)], self.rotation, self.rot_origin)[0]
        self.move(new_center[0] - center_x, new_center[1] - center_y)
        self.rot_origin = new_center
        self.mod += 1

    def is_close_to_click(self, click_x, click_y, threshold):
        """Check if a click is close to the object."""
        bb = self.bbox()
        if bb is None:
            return False
        x, y, width, height = bb
        if x <= click_x <= x + width and y <= click_y <= y + height:
            return True
        return False

    def image(self):
        """Return the image."""
        return self.__image.pixbuf()

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
        self.mod += 1

    def show_caret(self, show = None):
        """Show the caret."""
        if show is not None:
            self.__show_caret = show
            self.mod += 1
        return self.__show_caret

    def move(self, dx, dy):
        """Move the text object by dx, dy."""
        move_coords(self.coords, dx, dy)
        if self.rotation:
            self.rot_origin = (self.rot_origin[0] + dx, self.rot_origin[1] + dy)
        self.mod += 1

    def is_close_to_click(self, click_x, click_y, threshold):
        """Check if a click is close to the path."""
        if self.__bb is None:
            return False

        bb = self.__bb
        x0, y0 = bb[0], bb[1]
        x1, y1 = x0 + bb[2], y0 + bb[3]
        if self.rotation:
            click_x, click_y = coords_rotate([(click_x, click_y)],
                                             0-self.rotation,
                                             self.rot_origin)[0]

        return (x0 - threshold <= click_x <= x1 + threshold and
                y0 - threshold <= click_y <= y1 + threshold)


    def rotate_end(self):
        """Finish the rotation operation."""
        self.mod += 1
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
        self.mod += 1

    def resize_update(self, bbox):
        print("resizing text", [ int(x) for x in bbox])
        if bbox[2] < 0:
            bbox = (bbox[0], bbox[1], 10, bbox[3])
        if bbox[3] < 0:
            print("flipping y")
            bbox = (bbox[0], bbox[1], bbox[2], 10)
        self.resizing["bbox"] = bbox
        self.mod += 1

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
            #print("trying font size", self.pen.font_size)
            self.draw(cr, False, False)
            out_of_range_low = self.pen.font_size < min_fs and direction < 0
            out_of_range_up  = self.pen.font_size > max_fs and direction > 0
            if out_of_range_low or out_of_range_up:
                #print("font size out of range")
                break
            current_bbox = self.__bb
            #print("drawn, bbox is", self.__bb)
            if direction > 0 and (current_bbox[2] >= new_bbox[2] or
                                  current_bbox[3] >= new_bbox[3]):
                #print("increased beyond the new bbox")
                break
            if direction < 0 and (current_bbox[2] <= new_bbox[2] and
                                  current_bbox[3] <= new_bbox[3]):
                break

        self.coords[0] = (new_bbox[0], new_bbox[1] + self.font_extents[0])
        print("final coords are", self.coords)
        print("font extents are", self.font_extents)

        # first
        self.resizing = None
        self.mod += 1

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
        self.mod += 1

    def update_by_key(self, keyname, char):
        """Update the text object by keypress."""
        self.__text.update_by_key(keyname, char)
        self.mod += 1

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

        cr.save()
       #if self.resizing:
       #    cr.translate(self.resizing["bbox"][0], self.resizing["bbox"][1])
       #    scale_x = self.resizing["bbox"][2] / self.__bb[2]
       #    scale_y = self.resizing["bbox"][3] / self.__bb[3]
       #    cr.scale(scale_x, scale_y)
       #    cr.translate(-self.__bb[0], -self.__bb[1])

        if self.rotation:
            cr.translate(self.rot_origin[0], self.rot_origin[1])
            cr.rotate(self.rotation)
            cr.translate(-self.rot_origin[0], -self.rot_origin[1])

        for i, fragment in enumerate(content):

            cr.set_source_rgba(*self.pen.color, self.pen.transparency)
            #x_bearing, y_bearing, t_width, t_height, x_advance, y_advance
            x_bearing, _, t_width, _, _, _ = cr.text_extents(fragment)

            bb[2] = max(bb[2], t_width + x_bearing)
            bb[3] += self.font_extents[2]

            cr.set_font_size(self.pen.font_size)
            cr.move_to(position[0], position[1] + dy)
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

        cr.restore()

        if selected or self.resizing:
            cr.set_source_rgb(1, 0, 0)
            self.bbox_draw(cr, lw=.5)
        #self.bbox_draw(cr, lw=1.5)
        if hover:
            self.bbox_draw(cr, lw=.5)

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
        self.mod += 1

    def update(self, x, y, pressure):
        """Update the shape with a new point."""
        self.path_append(x, y, pressure)
        self.mod += 1

    def move(self, dx, dy):
        """Move the shape by dx, dy."""
        move_coords(self.coords, dx, dy)
        self.bb = None
        self.mod += 1

    def rotate_end(self):
        """finish the rotation"""
        # rotate all coords and outline
        self.coords  = coords_rotate(self.coords,  self.rotation, self.rot_origin)
        self.rotation   = 0
        self.rot_origin = None
        # recalculate bbox
        self.bb = path_bbox(self.coords)
        self.mod += 1

    def is_close_to_click(self, click_x, click_y, threshold):
        """Check if a click is close to the object."""
        bb = self.bbox(actual = True)
        if bb is None:
            return False
        x, y, width, height = bb
        if x <= click_x <= x + width and y <= click_y <= y + height:
            return True
        return False

    def path_append(self, x, y, pressure = None): # pylint: disable=unused-argument
        """Append a new point to the path."""
        self.coords.append((x, y))
        self.bb = None
        self.mod += 1

    def fill_toggle(self):
        """Toggle the fill of the object."""
        old_bbox = self.bbox(actual = True)
        self.bb  = None
        self.fill_set(not self.fill())
        new_bbox = self.bbox(actual = True)
        self.coords = transform_coords(self.coords, new_bbox, old_bbox)
        self.bb = None
        self.mod += 1

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
        self.mod += 1

    def resize_update(self, bbox):
        """Update during the resize of the object."""
        self.resizing["bbox"] = bbox
        self.mod += 1

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
        self.mod += 1

    def to_dict(self):
        """Convert the object to a dictionary."""
        return {
            "type": self.type,
            "coords": self.coords,
            "filled": self.fill(),
            "pen": self.pen.to_dict()
        }


    def draw_simple(self, cr, bbox=None):
        """draws the path as a single line. Useful for resizing."""

        if len(self.coords) < 3:
            return

        if bbox:
            old_bbox = path_bbox(self.coords)
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

        if outline:
            self.draw_simple(cr, res_bb)
            cr.set_line_width(0.5)
            cr.stroke()
        elif self.fill():
            self.draw_simple(cr, res_bb)
            cr.fill()
        else:
            self.draw_simple(cr, res_bb)
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
        self.mod += 1


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
        self.mod += 1

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

        self.mod += 1
        self.coords = coords

class Box(Drawable):
    """Class for creating a box."""
    def __init__(self, coords, pen):
        super().__init__("box", coords, pen)

    def update(self, x, y, pressure):
        self.coords[1] = (x, y)
        self.mod += 1

    def resize_end(self):
        bbox = self.bbox()
        self.coords = [ (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]) ]
        self.resizing = None
        self.mod += 1

    def rotate_end(self):
        """Ignore rotation"""

    def rotate_start(self, origin):
        """Ignore rotation."""

    def rotate(self, angle, set_angle = False):
        """Ignore rotation."""

    def resize_update(self, bbox):
        self.resizing["bbox"] = bbox
        self.coords = [ (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]) ]
        self.mod += 1

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

Drawable.register_type("text", Text)
Drawable.register_type("shape", Shape)
Drawable.register_type("rectangle", Rectangle)
Drawable.register_type("circle", Circle)
Drawable.register_type("box", Box)
Drawable.register_type("image", Image)
