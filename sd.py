#!/usr/bin/env python3

import gi
import copy
import yaml
import pickle
gi.require_version('Gtk', '3.0')

from gi.repository import Gtk, Gdk, GdkPixbuf
import cairo
import os
import time
import math
import base64
import tempfile
from io import BytesIO

savefile = os.path.expanduser("~/.screendrawer")
print(savefile)
# open file for appending if exists, or create if not

## ---------------------------------------------------------------------

def distance_point_to_segment(px, py, x1, y1, x2, y2):
    """Calculate the distance from a point (px, py) to a line segment (x1, y1) to (x2, y2)."""
    # Calculate the line segment's length squared
    length_squared = (x2 - x1) ** 2 + (y2 - y1) ** 2
    if length_squared == 0:
        # The segment is a point
        return math.sqrt((px - x1) ** 2 + (py - y1) ** 2)
    
    # Consider the line extending the segment, parameterized as x1 + t (x2 - x1), y1 + t (y2 - y1).
    # We find projection of point p onto the line. 
    # It falls where t = [(p-x1) . (x2-x1) + (p-y1) . (y2-y1)] / |x2-x1|^2
    t = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / length_squared
    t = max(0, min(1, t))
    
    # Projection falls on the segment
    projection_x = x1 + t * (x2 - x1)
    projection_y = y1 + t * (y2 - y1)
    
    return math.sqrt((px - projection_x) ** 2 + (py - projection_y) ** 2)

def find_obj_close_to_click(click_x, click_y, objects, threshold):
    for obj in objects[::-1]:
        if not obj is None and obj.is_close_to_click(click_x, click_y, threshold):
            return obj

    return None

def find_obj_in_bbox(bbox, objects):
    x, y, w, h = bbox
    ret = []
    for obj in objects:
        x_o, y_o, w_o, h_o = obj.bbox()
        if x_o >= x and y_o >= y and x_o + w_o <= x + w and y_o + h_o <= y + h:
            ret.append(obj)
    return ret

def normal_vec(x0, y0, x1, y1):
    dx, dy = x1 - x0, y1 - y0
    length = math.sqrt(dx**2 + dy**2)
    dx, dy = dx / length, dy / length
    return -dy, dx

def transform_coords(coords, bb1, bb2):
    """Transform coordinates from one bounding box to another."""
    x0, y0, w0, h0 = bb1
    x1, y1, w1, h1 = bb2
    ret = [ 
        (x1 + (x - x0) / w0 * w1, y1 + (y - y0) / h0 * h1) 
        for x, y in coords 
    ]
    return ret

def move_coords(coords, dx, dy):
    """Move a path by a given offset."""
    if not coords:
        ValueError("No coordinates to move")
    for i in range(len(coords)):
        coords[i] = (coords[i][0] + dx, coords[i][1] + dy)

def path_bbox(coords):
    """Calculate the bounding box of a path."""
    if not coords:
        return (0, 0, 0, 0)
    x0, y0 = coords[0]
    x1, y1 = coords[0]
    for x, y in coords[1:]:
        x0 = min(x0, x)
        y0 = min(y0, y)
        x1 = max(x1, x)
        y1 = max(y1, y)
    return (x0, y0, x1 - x0, y1 - y0)

def is_click_close_to_path(click_x, click_y, path, threshold):
    """Check if a click is close to any segment in the path."""

    for i in range(len(path) - 1):
        segment_start = path[i]
        segment_end = path[i + 1]
        distance = distance_point_to_segment(click_x, click_y, segment_start[0], segment_start[1], segment_end[0], segment_end[1])
        if distance <= threshold:
            return True
    return False

def is_click_in_bbox_corner(click_x, click_y, bbox, threshold):
    """Check if a click is in the corner of a bounding box."""
    x, y, w, h = bbox
    if (abs(click_x - x) < threshold) and (abs(click_y - y) < threshold):
        return "upper_left"
    if (abs(x + w - click_x) < threshold) and (abs(click_y - y) < threshold):
        return "upper_right"
    if (abs(click_x - x) < threshold) and (abs(y + h - click_y) < threshold):
        return "lower_left"
    if (abs(x + w - click_x) < threshold) and (abs(y + h - click_y) < threshold):
        return "lower_right"
    return None

def find_corners_next_to_click(click_x, click_y, objects, threshold):
    """Find the corner of a bounding box next to a click."""
    for obj in objects:
        bb = obj.bbox()
        if bb is None:
            continue
        corner = is_click_in_bbox_corner(click_x, click_y, bb, threshold)
        if corner:
            return obj, corner
    return None, None

def img_object_copy(obj):
    """Create a pixbuf copy of the given drawable object."""

    bb            = obj.bbox()
    width, height = math.ceil(bb[2]), math.ceil(bb[3])
    surface       = cairo.ImageSurface(cairo.Format.ARGB32, width, height)
    cr            = cairo.Context(surface)

    # move to top left corner
    obj.move(-bb[0], -bb[1])
    obj.draw(cr)
    # move back
    obj.move(bb[0], bb[1])

    pixbuf = Gdk.pixbuf_get_from_surface(surface, 0, 0, width, height)
    return pixbuf

## ---------------------------------------------------------------------

class MoveResizeEvent:
    """Simple class for handling move and resize events."""
    def __init__(self, type, obj, origin, corner=None):
        self.obj    = obj
        self.corner = corner
        self.origin = origin
        self.bbox   = obj.bbox()

    def origin_set(self, origin):
        self.origin = origin

    def origin_get(self):
        return self.origin

class MoveEvent(MoveResizeEvent):
    def __init__(self, obj, origin):
        super().__init__("move", obj, origin)

class ResizeEvent(MoveResizeEvent):
    def __init__(self, obj, origin, corner):
        super().__init__("resize", obj, origin, corner)
        obj.resize_start(corner, origin)

    def resize_event_update(self, x, y):
        dx = x - self.origin[0]
        dy = y - self.origin[1]
        bb = self.obj.bbox()

        corner = self.corner

        print("realizing resize by", dx, dy, "in corner", corner)
        if corner == "lower_left":
            newbb = (bb[0] + dx, bb[1], bb[2] - dx, bb[3] + dy)
        elif corner == "lower_right":
            newbb = (bb[0], bb[1], bb[2] + dx, bb[3] + dy)
        elif corner == "upper_left":
            newbb = (bb[0] + dx, bb[1] + dy, bb[2] - dx, bb[3] - dy)
        elif corner == "upper_right":
            newbb = (bb[0], bb[1] + dy, bb[2] + dx, bb[3] - dy)
        else:
            raise ValueError("Invalid corner:", corner)

        self.obj.resize_update(newbb)
        self.origin_set((x, y))

## ---------------------------------------------------------------------

class Drawable:
    """Base class for drawable objects."""
    def __init__(self, type, coords, color, line_width, fill_color = None):
        self.type       = type
        self.coords     = coords
        self.color      = color
        self.line_width = line_width
        self.origin     = None
        self.resizing   = None
        self.fill_color = fill_color


    def resize_start(self, corner, origin):
        self.resizing = {
            "corner": corner,
            "origin": origin,
            "bbox":   self.bbox()
            }

    def unfill(self):
        self.fill_color = None

    def fill(self, color = None):
        self.fill_color = color

    def resize_update(self, bbox):
        self.resizing["bbox"] = bbox

    def color_set(self, color):
        self.color = color

    def resize_end(self):
        self.resizing = None
        # not implemented
        print("resize_end not implemented")

    def origin_remove(self):
        self.origin = None

    def is_close_to_click(self, click_x, click_y, threshold):
        x1, y1 = self.coords[0]
        x2, y2 = self.coords[1]
     
        path = [ (x1, y1), (x1, y2), (x2, y2), (x2, y1), (x1, y1) ]
        return is_click_close_to_path(click_x, click_y, path, threshold)

    def to_dict(self):
        return {
            "type": self.type,
            "coords": self.coords,
            "color": self.color,
            "fill_color": self.fill_color,
            "line_width": self.line_width
        }

    def move(self, dx, dy):
        move_coords(self.coords, dx, dy)

    def bbox(self):
        if self.resizing:
            return self.resizing["bbox"]
        left, top = min(p[0] for p in self.coords), min(p[1] for p in self.coords)
        width =    max(p[0] for p in self.coords) - left
        height =   max(p[1] for p in self.coords) - top
        return (left, top, width, height)

    def bbox_draw(self, cr, bb=None, lw=0.2):
        if not bb:
            bb = self.bbox()
        x, y, w, h = bb
        cr.set_line_width(lw)
        cr.rectangle(x, y, w, h)
        cr.stroke()

    def draw(self, cr, hover=False, selected=False):
        raise NotImplementedError("draw method not implemented")

    @classmethod
    def from_dict(cls, d):
        type_map = {
            "path": Path,
            "circle": Circle,
            "box": Box,
            "image": Image,
            "group": DrawableGroup,
            "text": Text
        }
        type = d.pop("type")
        if type not in type_map:
            raise ValueError("Invalid type:", type)

        return type_map.get(type)(**d)


class DrawableGroup(Drawable):
    """Class for creating groups of drawable objects or other groups."""
    def __init__(self, objects = [ ], objects_dict = None):

        if objects_dict:
            objects = [ Drawable.from_dict(d) for d in objects_dict ]

        print("Creating DrawableGroup with objects", objects)
        super().__init__("drawable_group", [ (None, None) ], None, None)
        self.objects = objects
        self.type = "group"

    def contains(self, obj):
        return obj in self.objects

    def is_close_to_click(self, click_x, click_y, threshold):
        for obj in self.objects:
            if obj.is_close_to_click(click_x, click_y, threshold):
                return True
        return False

    def to_dict(self):
        return {
            "type": self.type,
            "objects_dict": [ obj.to_dict() for obj in self.objects ],
        }

    def color_set(self, color):
        for obj in self.objects:
            obj.color_set(color)

    def resize_start(self, corner, origin):
        self.resizing = {
            "corner": corner,
            "origin": origin,
            "bbox":   self.bbox()
            }
        for obj in self.objects:
            obj.resize_start(corner, origin)
 
 
    def resize_update(self, bbox):
        prev_bbox = self.resizing["bbox"]

        dx, dy           = bbox[0] - prev_bbox[0], bbox[1] - prev_bbox[1]
        scale_x, scale_y = bbox[2] / prev_bbox[2], bbox[3] / prev_bbox[3]

        for obj in self.objects:
            obj_bb = obj.bbox()

            x, y, w, h = obj_bb
            w2, h2 = w * scale_x, h * scale_y

            x2 = bbox[0] + (x - prev_bbox[0]) * scale_x
            y2 = bbox[1] + (y - prev_bbox[1]) * scale_y

            ## recalculate the new bbox of the object within our new bb
            obj.resize_update((x2, y2, w2, h2))

        self.resizing["bbox"] = bbox

    def resize_end(self):
        self.resizing = None
        for obj in self.objects:
            obj.resize_end()
 
    def length(self):
        return len(self.objects)

    def bbox(self):
        if self.resizing:
            return self.resizing["bbox"]
        if not self.objects:
            return None
        left, top, width, height = self.objects[0].bbox()
        bottom, right = top + height, left + width

        for obj in self.objects[1:]:
            x, y, w, h = obj.bbox()
            left, top = min(left, x), min(top, y)
            bottom, right = max(bottom, y + h), max(right, x + w)

        width, height = right - left, bottom - top
        return (left, top, width, height)

    def add(self, obj):
        if obj not in self.objects:
            self.objects.append(obj)

    def remove(self, obj):
        self.objects.remove(obj)

    def move(self, dx, dy):
        for obj in self.objects:
            obj.move(dx, dy)

    def draw(self, cr, hover=False, selected=False):
        for obj in self.objects:
            obj.draw(cr, hover=hover, selected=selected)
        cr.set_source_rgb(0, 0, 0)
        if selected:
            self.bbox_draw(cr, lw=1.5)
        if hover:
            self.bbox_draw(cr, lw=.5)



class Image(Drawable):
    def __init__(self, coords, color, line_width, image, image_base64 = None, transform = None):

        if image_base64:
            self.image_base64 = image_base64
            image = self.decode_base64(image_base64)
        else:
            self.image_base64 = None

        width, height = image.get_width(), image.get_height()
        coords = [ (coords[0][0], coords[0][1]), (coords[0][0] + width, coords[0][1] + height) ]
        super().__init__("image", coords, color, line_width)
        self.image = image
        self.transform = transform or None
        self.image_size = (width, height)

    def draw(self, cr, hover=False, selected=False):
        cr.save()
        cr.translate(self.coords[0][0], self.coords[0][1])

        if self.transform:
            w_scale, h_scale = self.transform
            cr.scale(w_scale, h_scale)

        Gdk.cairo_set_source_pixbuf(cr, self.image, 0, 0)
        cr.paint()

        cr.restore()

        cr.set_source_rgb(*self.color)
        if selected:
            self.bbox_draw(cr, lw=1.5)
        if hover:
            self.bbox_draw(cr, lw=.5)

    def coords_from_scale(self):
        x0, y0 = self.coords[0]
        w0, h0 = self.image_size
        w1, h1 = w0 * self.transform[0], h0 * self.transform[1]
        self.coords = [(x0, y0), (x0 + w1, y0 + h1)]

    def resize_update(self, bbox):
        self.resizing["bbox"] = bbox
        old_bbox = self.bbox()
        coords = self.coords

        x0, y0 = coords[0][0], coords[0][1]
        w0, h0 = self.image_size[0], self.image_size[1]
        x1, y1, w1, h1 = bbox

        w_scale = w1 / w0
        h_scale = h1 / h0

        self.coords[0] = (x1, y1)
        self.coords[1] = (x1 + w1, y1 + h1)
        self.transform = (w_scale, h_scale)

    def resize_end(self):
        self.resizing = None

    def is_close_to_click(self, click_x, click_y, threshold):
        bb = self.bbox()
        if bb is None:
            return False
        x, y, width, height = bb
        if click_x >= x and click_x <= x + width and click_y >= y and click_y <= y + height:
            return True

    def decode_base64(self, image_base64):
        image_binary = base64.b64decode(image_base64)

        # Step 2: Wrap the binary data in a BytesIO object
        image_io = BytesIO(image_binary)

        # Step 3: Load the image data into a GdkPixbuf
        loader = GdkPixbuf.PixbufLoader.new_with_type('png')  # Specify the image format if known
        loader.write(image_io.getvalue())
        loader.close()  # Finalize the loader
        image = loader.get_pixbuf()  # Get the loaded GdkPixbuf
        return image

    def encode_base64(self):
        buffer = BytesIO()
        with tempfile.NamedTemporaryFile(delete = True) as temp:
            self.image.savev(temp.name, "png", [], [])
            with open(temp.name, "rb") as f:
                image_base64 = base64.b64encode(f.read()).decode("utf-8")
        return image_base64

    def base64(self):
        if self.image_base64 is None:
            self.image_base64 = self.encode_base64()
        return self.image_base64

    def to_dict(self):

        return {
            "type": self.type,
            "coords": self.coords,
            "color": self.color,
            "image": None,
            "transform": self.transform,
            "image_base64": self.base64(),
            "line_width": self.line_width
        }


class Text(Drawable):
    def __init__(self, coords, color, line_width, content, size):
        super().__init__("text", coords, color, line_width)

        # split content by newline
        content = content.split("\n")
        self.content = content
        self.size    = size
        self.line    = 0
        self.cursor_pos = None
        self.bb         = None
        self.font_extents = None

    def is_close_to_click(self, click_x, click_y, threshold):
        if self.bb is None:
            return False
        x, y, width, height = self.bb
        if click_x >= x and click_x <= x + width and click_y >= y and click_y <= y + height:
            return True

    def resize_update(self, bbox):
        print("resizing text", bbox)
        if(bbox[2] < 0):
            bbox = (bbox[0], bbox[1], 10, bbox[3])
        if(bbox[3] < 0):
            print("flipping y")
            bbox = (bbox[0], bbox[1], bbox[2], 10)
        self.resizing["bbox"] = bbox

    def resize_end(self):
        new_bbox = self.resizing["bbox"]
        old_bbox = self.bb
        old_coords = self.coords
        # create a surface with the new size
        surface = cairo.ImageSurface(cairo.Format.ARGB32, 
                                     2 * math.ceil(new_bbox[2]), 
                                     2 * math.ceil(new_bbox[3]))
        cr = cairo.Context(surface)
        min_fs, max_fs = 8, 154
        print("new bbox is", new_bbox)
        print("old bbox is", old_bbox)
        if new_bbox[2] < old_bbox[2] or new_bbox[3] < old_bbox[3]:
            dir = -1
        else:
            dir = 1

        self.coords = [ (0, 0), (old_bbox[2], old_bbox[3]) ]
        # loop while font size not larger than max_fs and not smaller than
        # min_fs
        print("resizing text, dir=", dir, "font size is", self.size)
        while True:
            self.size += dir
            print("trying font size", self.size)
            self.draw(cr, False, False)
            if (self.size < min_fs and dir < 0) or (self.size > max_fs and dir > 0):
                print("font size out of range")
                break
            current_bbox = self.bb
            print("drawn, bbox is", self.bb)
            if dir > 0 and (current_bbox[2] >= new_bbox[2] or current_bbox[3] >= new_bbox[3]):
                print("increased beyond the new bbox")
                break
            if dir < 0 and (current_bbox[2] <= new_bbox[2] and current_bbox[3] <= new_bbox[3]):
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
            "color": self.color,
            "line_width": self.line_width,
            "content": self.as_string(),
            "size": self.size
        }

    def bbox(self):
        if self.resizing:
            return self.resizing["bbox"]
        if not self.bb:
            return (self.coords[0][0], self.coords[0][1], 50, 50)
        return self.bb

    def as_string(self):
        return "\n".join(self.content)

    def strlen(self):
        return len(self.as_string())

    def add_text(self, text):
        # split text by newline
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if i == 0:
                self.content[self.line] += line
                self.cursor_pos += len(text)
            else:
                self.content.insert(self.line + i, line)
                self.cursor_pos = len(line)

    def backspace(self):
        cnt = self.content
        if self.cursor_pos > 0:
            cnt[self.line] = cnt[self.line][:self.cursor_pos - 1] + cnt[self.line][self.cursor_pos:]
            self.cursor_pos -= 1
        elif self.line > 0:
            self.cursor_pos = len(cnt[self.line - 1])
            cnt[self.line - 1] += cnt[self.line]
            cnt.pop(self.line)
            self.line -= 1

    def newline(self):
        self.content.insert(self.line + 1, self.content[self.line][self.cursor_pos:])
        self.content[self.line] = self.content[self.line][:self.cursor_pos]
        self.line += 1
        self.cursor_pos = 0

    def add_char(self, char):
        self.content[self.line] = self.content[self.line][:self.cursor_pos] + char + self.content[self.line][self.cursor_pos:]
        self.cursor_pos += 1

    def move_cursor(self, direction):
        if direction == "End":
            self.line = len(self.content) - 1
            self.cursor_pos = len(self.content[self.line])
        elif direction == "Home":
            self.line = 0
            self.cursor_pos = 0
        elif direction == "Right":
            if self.cursor_pos < len(self.content[self.line]):
                self.cursor_pos += 1
            elif self.line < len(self.content) - 1:
                self.line += 1
                self.cursor_pos = 0
        elif direction == "Left":
            if self.cursor_pos > 0:
                self.cursor_pos -= 1
            elif self.line > 0:
                self.line -= 1
                self.cursor_pos = len(self.content[self.line])
        elif direction == "Down":
            if self.line < len(self.content) - 1:
                self.line += 1
                if self.cursor_pos > len(self.content[self.line]):
                    self.cursor_pos = len(self.content[self.line])
        elif direction == "Up":
            if self.line > 0:
                self.line -= 1
                if self.cursor_pos > len(self.content[self.line]):
                    self.cursor_pos = len(self.content[self.line])
        else:
            raise ValueError("Invalid direction:", direction)

    def draw_cursor(self, cr, xx0, yy0, height):
        cr.move_to(xx0, yy0)
        cr.line_to(xx0, yy0 + height)
        cr.stroke()
        cr.move_to(xx0 - 3, yy0)
        cr.line_to(xx0 + 3, yy0)
        cr.stroke()
        cr.move_to(xx0 - 3, yy0 + height)
        cr.line_to(xx0 + 3, yy0 + height)
        cr.stroke()

    def draw(self, cr, hover=False, selected=False):
        position, content, size, color, cursor_pos = self.coords[0], self.content, self.size, self.color, self.cursor_pos
        
        # get font info
        cr.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        cr.set_font_size(size)

        font_extents      = cr.font_extents()
        self.font_extents = font_extents
        ascent, height    = font_extents[0], font_extents[2]

        dy   = 0

        # new bounding box
        bb_x = position[0]
        bb_y = position[1] - ascent
        bb_w = 0
        bb_h = 0
        
        for i in range(len(content)):
            fragment = content[i]

            x_bearing, y_bearing, t_width, t_height, x_advance, y_advance = cr.text_extents(fragment)

            bb_w = max(bb_w, t_width + x_bearing)
            bb_h += height

            cr.set_font_size(size)
            cr.move_to(position[0], position[1] + dy)
            cr.set_source_rgb(*color)
            cr.show_text(fragment)
            cr.stroke()

            # draw the cursor
            if cursor_pos != None and i == self.line:
                x_bearing, y_bearing, t_width, t_height, x_advance, y_advance = cr.text_extents("|" + fragment[:cursor_pos] + "|")
                x_bearing2, y_bearing2, t_width2, t_height2, x_advance2, y_advance2 = cr.text_extents("|")
                cr.set_source_rgb(1, 0, 0)
                xx0, yy0 = position[0] - x_bearing + t_width - 2 * t_width2, position[1] + dy - ascent
                self.draw_cursor(cr, xx0, yy0, height)

            dy += height

        self.bb = (bb_x, bb_y, bb_w, bb_h)

        if selected: 
            self.bbox_draw(cr, lw=1.5)
        if hover:
            self.bbox_draw(cr, lw=.5)

class Path(Drawable):
    def __init__(self, coords, color, line_width, outline = None, pressure = None):
        super().__init__("path", coords, color, line_width)
        self.outline   = outline  or []
        self.pressure  = pressure or []
        self.outline_l = []
        self.outline_r = []

    def move(self, dx, dy):
        move_coords(self.coords, dx, dy)
        move_coords(self.outline, dx, dy)

    def is_close_to_click(self, click_x, click_y, threshold):
        return is_click_close_to_path(click_x, click_y, self.coords, threshold)

    def to_dict(self):
        return {
            "type": self.type,
            "coords": self.coords,
            "outline": self.outline,
            "pressure": self.pressure,
            "color": self.color,
            "line_width": self.line_width
        }

    def path_append(self, x, y, pressure = 1):
        """Append a point to the path, calculating the outline of the
           polygon around the path."""
        coords = self.coords
        width  = self.line_width * pressure

        if len(coords) == 0:
            self.pressure.append(pressure)
            coords.append((x, y))
            return

        lp = coords[-1]
        if abs(x - lp[0]) < 1 and abs(y - lp[1]) < 1:
            return

        self.pressure.append(pressure)
        coords.append((x, y))
        width = width / 2

        if len(coords) == 2:
            p1, p2 = coords[0], coords[1]
            nx, ny = normal_vec(p1[0], p1[1], p2[0], p2[1])
            self.outline_l.append((p1[0] + nx * width, p1[1] + ny * width))
            self.outline_l.append((p2[0] + nx * width, p2[1] + ny * width))
            self.outline_r.append((p1[0] - nx * width, p1[1] - ny * width))
            self.outline_r.append((p2[0] - nx * width, p2[1] - ny * width))
        if len(coords) > 2:
            p1, p2 = coords[-2], coords[-1]
            nx, ny = normal_vec(p1[0], p1[1], p2[0], p2[1])
            self.outline_l.append((p1[0] + nx * width, p1[1] + ny * width))
            self.outline_r.append((p1[0] - nx * width, p1[1] - ny * width))
        if len(coords) >= 2:
            self.outline = self.outline_l + self.outline_r[::-1]

    # XXX not efficient, this should be done in path_append and modified
    # upon move.
    def bbox(self):
        if self.resizing:
            return self.resizing["bbox"]
        return path_bbox(self.coords)

    def outline_recalculate(self, coords, pressure):
        """Takes new coords and pressure and recalculates the outline."""
        self.outline_l = []
        self.outline_r = []
        self.outline   = []
        self.pressure  = []
        self.coords    = []

        print(len(pressure), len(coords))   
        for x, y in coords:
            self.path_append(x, y, pressure.pop(0))

    def resize_end(self):
        """recalculate the outline after resizing"""
        print("length of coords and pressure:", len(self.coords), len(self.pressure))
        old_bbox = path_bbox(self.coords)
        new_coords = transform_coords(self.coords, old_bbox, self.resizing["bbox"])
        pressure   = self.pressure
        self.outline_recalculate(new_coords, pressure)
        self.resizing  = None

    def draw_simple(self, cr, hover=False, selected=False, bbox=None):

        if len(self.coords) < 2:
            return

        if bbox:
            old_bbox = path_bbox(self.coords)
            coords = transform_coords(self.coords, old_bbox, bbox)
        else:
            coords = self.coords

        cr.set_source_rgb(*self.color)
        cr.move_to(coords[0][0], coords[0][1])
        for point in coords[1:]:
            cr.line_to(point[0], point[1])
        cr.stroke()

        if selected:
            self.bbox_draw(cr, lw=1.5)

    def draw(self, cr, hover=False, selected=False):
        if len(self.outline) < 4 or len(self.coords) < 3:
            return

        dd = 1 if hover else 0

        if self.resizing:
            self.draw_simple(cr, hover=hover, selected=selected, bbox=self.resizing["bbox"])
            return
        
        #cr.set_source_rgb(*self.color)
        cr.set_source_rgba(*self.color, .75)

        if selected:
            self.bbox_draw(cr, lw=.5)

        cr.move_to(self.outline[0][0] + dd, self.outline[0][1] + dd)
        for point in self.outline[1:]:
            cr.line_to(point[0] + dd, point[1] + dd)
        cr.close_path()
        cr.fill()

class Circle(Drawable):
    def __init__(self, coords, color, line_width, fill_color = None):
        super().__init__("circle", coords, color, line_width, fill_color)

    def resize_end(self):
        bbox = self.bbox()
        self.coords = [ (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]) ]
        self.resizing = None

    def resize_update(self, bbox):
        self.resizing["bbox"] = bbox
        self.coords = [ (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]) ]

    def draw(self, cr, hover=False, selected=False):
        if hover:
            cr.set_line_width(self.line_width + 1)
        else:
            cr.set_line_width(self.line_width)
        cr.set_source_rgb(*self.color)
        x1, y1 = self.coords[0]
        x2, y2 = self.coords[1]
        w, h = (abs(x1 - x2), abs(y1 - y2))
        x0, y0 = (min(x1, x2), min(y1, y2))
        #cr.rectangle(x0, y0, w, h)
        cr.save()
        cr.translate(x0 + w / 2, y0 + h / 2)
        cr.scale(w / 2, h / 2)
        cr.arc(0, 0, 1, 0, 2 * 3.14159)
        if self.fill_color:
            cr.set_source_rgb(*self.fill_color)
            cr.fill_preserve()
        cr.restore()
        cr.stroke()

class Box(Drawable):
    def __init__(self, coords, color, line_width, fill_color = None):
        super().__init__("box", coords, color, line_width, fill_color)

    def resize_end(self):
        bbox = self.bbox()
        self.coords = [ (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]) ]
        self.resizing = None

    def resize_update(self, bbox):
        self.resizing["bbox"] = bbox
        self.coords = [ (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]) ]

    def draw(self, cr, hover=False, selected=False):
        cr.set_source_rgb(*self.color)

        if hover:
            cr.set_line_width(self.line_width + 1)
        else:
            cr.set_line_width(self.line_width)

        x1, y1 = self.coords[0]
        x2, y2 = self.coords[1]
        w, h = (abs(x1 - x2), abs(y1 - y2))
        x0, y0 = (min(x1, x2), min(y1, y2))

        if self.fill_color:
            print("filling with color", self.fill_color)
            cr.set_source_rgb(*self.fill_color)
            cr.rectangle(x0, y0, w, h)
            cr.fill()
            cr.stroke()

        cr.set_source_rgb(*self.color)
        cr.rectangle(x0, y0, w, h)
        cr.stroke()

        if selected:
            cr.set_line_width(0.5)
            cr.arc(x0, y0, 10, 0, 2 * 3.14159)  # Draw a circle
            #cr.fill()  # Fill the circle to make a dot
            cr.stroke()


## ---------------------------------------------------------------------

class HelpDialog(Gtk.Dialog):
    def __init__(self, parent):
        super().__init__(title="Help", transient_for=parent, flags=0)
        self.add_buttons(Gtk.STOCK_OK, Gtk.ResponseType.OK)
        self.set_default_size(900, 300)
        self.set_border_width(10)

        # Example help text with Pango Markup (not Markdown but similar concept)
        help_text = """

<span font="24"><b>screendrawer</b></span>

Draw on the screen with Gnome and Cairo. Quick and dirty.

<b>(Help not complete yet.)</b>

<b>Mouse:</b>

<b>All modes:</b>                          <b>Move mode:</b>
shift-click: Enter text mode               click: Select object
right-button: Move object                  move: Move object
ctrl-click: Change line width              ctrl-a: Select all
                                           Tab: Next object
                                           Shift-Tab: Previous object

Moving object to left lower screen corner deletes it.

<b>Shortcut keys:</b>

<span font_family="monospace">
<b>Drawing modes:</b> (simple key)

<b>d:</b> Draw mode (pencil)                 <b>m:</b> Move mode (move objects around, copy and paste)
<b>t:</b> Text mode (text entry)             <b>b:</b> Box mode  (draw a rectangle)
<b>c:</b> Circle mode (draw an ellipse)      <b>e:</b> Eraser mode (delete objects with a click)

<b>With Ctrl:</b>                    <b>Simple key (not when entering text)</b>
Ctrl-q: Quit                         x: Exit
Ctrl-s: Save drawing                 h, F1, ?: Show this help dialog
Ctrl-l: Clear drawing                l: Clear drawing                 

Ctrl-c: Copy content                 &lt;Del&gt;: Delete selected object (in "move" mode)
Ctrl-v: Paste content                &lt;Esc&gt;: Finish text input
Ctrl-x: Cut content                  &lt;Enter&gt;: New line (in text mode)

Ctrl-k: Select color
Ctrl-plus, Ctrl-minus: Change text size
Ctrl-b: Cycle background transparency

</span>

The state is saved in / loaded from `~/.screendrawer` so you can continue drawing later.
You might want to remove that file if something goes wrong.
        """

        label = Gtk.Label()
        label.set_markup(help_text)
        label.set_justify(Gtk.Justification.LEFT)
        label.set_line_wrap(True)
        
        box = self.get_content_area()
        box.add(label)
        self.show_all()

## ---------------------------------------------------------------------

class TransparentWindow(Gtk.Window):
    def __init__(self):
        super(TransparentWindow, self).__init__()

        self.init_ui()
    
    def init_ui(self):
        self.set_title("Transparent Drawing Window")
        self.set_decorated(False)
        self.connect("destroy", self.exit)
        self.set_default_size(800, 600)

        screen = self.get_screen()
        visual = screen.get_rgba_visual()
        if visual != None and screen.is_composited():
            self.set_visual(visual)

        self.set_app_paintable(True)
        self.connect("draw", self.on_draw)
        self.connect("key-press-event", self.on_key_press)
        self.set_events(Gdk.EventMask.BUTTON_PRESS_MASK | Gdk.EventMask.BUTTON_RELEASE_MASK | Gdk.EventMask.POINTER_MOTION_MASK)

        # Drawing setup
        self.objects = [ ]
        self.current_object = None
        self.changing_line_width = False
        self.selection = None
        self.dragobj   = None
        self.resizeobj = None
        self.mode      = "draw"
        self.current_cursor = None
        self.hover = None
        self.cursor_pos = None
        self.clipboard  = None
        self.clipboard_owner = False # we need to keep track of the clipboard
        self.selection_tool = None
        self.gtk_clipboard = Gtk.Clipboard.get(Gdk.SELECTION_CLIPBOARD)

        # defaults for drawing
        self.transparent = 0
        self.font_size  = 24
        self.line_width = 4
        self.color      = (0.2, 0, 0)

        # distance for selecting objects
        self.max_dist   = 15

        self.load_state()

        self.gtk_clipboard.connect('owner-change', self.on_clipboard_owner_change)
        self.connect("button-press-event", self.on_button_press)
        self.connect("button-release-event", self.on_button_release)
        self.connect("motion-notify-event", self.on_motion_notify)

        self.make_cursors()
        self.set_keep_above(True)
        self.maximize()

    def on_clipboard_owner_change(self, clipboard, event):
        """Handle clipboard owner change events."""

        print("Owner change, removing internal clipboard")
        print("reason:", event.reason)
        if self.clipboard_owner:
            self.clipboard_owner = False
        else:
            self.clipboard = None
        return True

    def exit(self):
        ## close the savefile_f
        print("Exiting")
        self.save_state()
        Gtk.main_quit()
        

    def on_draw(self, widget, cr):
        """Handle draw events."""
        cr.set_source_rgba(1, 1, 1, self.transparent)
        cr.set_operator(cairo.OPERATOR_SOURCE)
        cr.paint()
        cr.set_operator(cairo.OPERATOR_OVER)
        self.draw(cr)
        return True

    def draw(self, cr):
        """Draw the objects."""

        for obj in self.objects:
            hover    = obj == self.hover
            selected = self.selection and self.selection.contains(obj) and self.mode == "move"
            obj.draw(cr, hover=hover, selected=selected)

        # If changing line width, draw a preview of the new line width
        if self.changing_line_width:
            cr.set_line_width(self.line_width)
            cr.set_source_rgb(*self.color)
            self.draw_dot(cr, *self.cur_pos, self.line_width)

    def clear(self):
        """Clear the drawing."""
        self.selection      = None
        self.dragobj        = None
        self.current_object = None
        self.objects = []
        self.queue_draw()

    # Event handlers
    def on_button_press(self, widget, event):
        print("on_button_press: type:", event.type, "button:", event.button, "state:", event.state)
        modifiers = Gtk.accelerator_get_default_mod_mask()
        hover_obj = find_obj_close_to_click(event.x, event.y, self.objects, self.max_dist)
        shift     = event.state & modifiers == Gdk.ModifierType.SHIFT_MASK
        ctrl      = event.state & modifiers == Gdk.ModifierType.CONTROL_MASK
        double    = event.type == Gdk.EventType.DOUBLE_BUTTON_PRESS
        corner_obj = find_corners_next_to_click(event.x, event.y, self.objects, 20)
        pressure = event.get_axis(Gdk.AxisUse.PRESSURE) or 0

        if corner_obj[0]:
            print("corner click:", corner_obj[0].type, corner_obj[1])
        #print("mode:", self.mode)

        # Ignore clicks when text input is active
        if self.current_object and self.current_object.type == "text":
            print("click, but text input active - finishing it first")
            self.finish_text_input()
            return True

        # Start changing line width: single click with ctrl pressed
        if ctrl and event.button == 1 and self.mode == "draw": 
            self.cur_pos = (event.x, event.y)
            self.changing_line_width = True
            return True

        # double click on a text object: start editing
        if event.button == 1 and double and hover_obj and hover_obj.type == "text" and self.mode in ["draw", "text", "move"]:
            # put the cursor in the last line, end of the text
            hover_obj.move_cursor("End")
            self.current_object = hover_obj
            self.queue_draw()
            self.change_cursor("none")
            return True

        # simple click: start drawing
        if event.button == 1 and not self.current_object:
            if self.mode == "text" or (self.mode == "draw" and shift):
                print("new text")
                self.change_cursor("none")
                self.current_object = Text([ (event.x, event.y) ], self.color, self.line_width, content="", size = self.font_size)
                self.current_object.move_cursor("Home")
                self.objects.append(self.current_object)

            elif self.mode == "draw":
                print("starting path")
                self.current_object = Path([ (event.x, event.y) ], self.color, self.line_width, pressure = [ pressure ])
                self.objects.append(self.current_object)

            elif self.mode == "box":
                print("drawing box / circle")
                self.current_object = Box([ (event.x, event.y), (event.x + 1, event.y + 1) ], self.color, self.line_width)
                self.objects.append(self.current_object)

            elif self.mode == "circle":
                print("drawing circle")
                self.current_object = Circle([ (event.x, event.y), (event.x + 1, event.y + 1) ], self.color, self.line_width)
                self.objects.append(self.current_object)

            elif self.mode == "move":
                if corner_obj[0] and corner_obj[0].bbox():
                    print("starting resize")
                    obj    = corner_obj[0]
                    corner = corner_obj[1]
                    self.resizeobj = ResizeEvent(obj, origin = (event.x, event.y), corner = corner)
                elif hover_obj:
                    if shift and self.selection:
                        # create Draw Group with the two objects
                        print("adding to group")
                        self.selection.add(hover_obj)
                    elif not self.selection or not self.selection.contains(hover_obj):
                            self.selection = DrawableGroup([ hover_obj ])

                    self.dragobj = MoveEvent(self.selection, (event.x, event.y))
                else:
                    self.selection = None
                    self.dragobj   = None
                    print("starting selection")
                    self.current_object = Box([ (event.x, event.y), (event.x + 1, event.y + 1) ], (1, 0, 0), 0.3)
                    self.objects.append(self.current_object)
                    self.selection_tool = self.current_object
                    self.queue_draw()

        # moving an object, or erasing it, if an object is underneath the cursor
        if hover_obj:
            #if (event.button == 1 and self.mode == "move") or event.button == 3:
            #    hover_obj.origin_set((event.x, event.y))
            #    self.selection = hover_obj
            #    self.dragobj   = hover_obj

            if event.button == 1 and self.mode == "eraser":
                self.objects.remove(hover_obj)
                self.selection = None
                self.dragobj   = None
                self.revert_cursor()

        self.queue_draw()
        return True

    # Event handlers
    def on_button_release(self, widget, event):
        """Handle mouse button release events."""
        obj = self.current_object
        if obj and obj.type == "path":
            print("finishing path")
            obj.path_append(event.x, event.y, 0)
            if len(obj.coords) != len(obj.pressure):
                print("Pressure and coords don't match")
            if len(obj.coords) < 3:
                self.objects.remove(obj)
            self.queue_draw()

        # this two are for changing line width
        self.cur_pos             = None
        self.changing_line_width = False

        # if the user clicked to create a text, we are not really done yet
        if self.current_object and self.current_object.type != "text":
            self.current_object = None

        # if selection tool is active, finish it
        if self.selection_tool:
            self.objects.remove(self.selection_tool)
            bb = self.selection_tool.bbox()
            obj = find_obj_in_bbox(bb, self.objects)
            self.selection_tool = None
            if len(obj) > 0:
                self.selection = DrawableGroup(obj)
            else:
                self.selection = None
            self.queue_draw()

        if self.resizeobj:
            print("finishing resize")
            self.resizeobj.obj.resize_end()
            self.resizeobj = None
            self.queue_draw()

        if self.dragobj:
            # If the user was dragging a selected object and the drag ends
            # in the lower left corner, delete the object
            # self.dragobj.origin_remove()
            obj = self.dragobj.obj
            if event.x < 10 and event.y > self.get_size()[1] - 10:
                self.objects.remove(obj)
                self.selection = None
                self.queue_draw()
            self.dragobj    = None
        return True


    def on_motion_notify(self, widget, event):
        """Handle mouse motion events."""
        obj = self.current_object
        self.cursor_pos = (event.x, event.y)

        if self.changing_line_width:
            self.line_width = max(3, min(40, self.line_width + (event.x - self.cur_pos[0])/250))
            self.queue_draw()
            return True

        if obj and (obj.type == "box" or obj.type == "circle"):
            obj.coords[1] = (event.x, event.y)
            self.queue_draw()
        elif obj and obj.type == "path":
            pressure = event.get_axis(Gdk.AxisUse.PRESSURE) or 1
            obj.path_append(event.x, event.y, pressure)
            self.queue_draw()
        elif self.resizeobj:
            self.resizeobj.resize_event_update(event.x, event.y)
            self.queue_draw()

        elif self.dragobj is not None:
            dx = event.x - self.dragobj.origin_get()[0]
            dy = event.y - self.dragobj.origin_get()[1]

            # Move the selected object
            self.dragobj.obj.move(dx, dy)

            self.dragobj.origin_set((event.x, event.y))
            self.queue_draw()
        elif self.mode == "move":
            object_underneath = find_obj_close_to_click(event.x, event.y, self.objects, self.max_dist)

            prev_hover = self.hover
            if object_underneath:
                if self.mode == "move":
                    self.change_cursor("moving")
                self.hover = object_underneath
            else:
                if self.mode == "move":
                    self.revert_cursor()
                self.hover = None
            if prev_hover != self.hover:
                self.queue_draw()

        # stop event propagation
        return True

    def finish_text_input(self):
        """Clean up current text and finish text input."""
        print("finishing text input")
        if self.current_object and self.current_object.type == "text":
            self.current_object.cursor_pos = None
            if self.current_object.strlen() == 0:
                self.objects.remove(self.current_object)
            self.current_object = None
        self.revert_cursor()
        self.queue_draw()

    def update_text_input(self, keyname, char):
        """Update the current text input."""
        cur  = self.current_object
    
        if keyname == "BackSpace": # and cur["cursor_pos"] > 0:
            cur.backspace()
        elif keyname in ["Home", "End", "Down", "Up", "Right", "Left"]:
            cur.move_cursor(keyname)
        elif keyname == "Return":
            cur.newline()
        elif char and char.isprintable():
            cur.add_char(char)
        self.queue_draw()

    def cycle_background(self):
        """Cycle through background transparency."""
        if self.transparent == 1:
            self.transparent = 0
        elif self.transparent == 0:
            self.transparent = .5
        else:
            self.transparent = 1
        self.queue_draw()

    def paste_text(self, clip_text):
        """Create a text object from clipboard text."""
        clip_text = clip_text.strip()
        # split by new lines

        if self.current_object and self.current_object.type == "text":
            self.current_object.add_text(clip_text)
            self.queue_draw()
        else:
            pos = self.cursor_pos or (100, 100)
            new_text = Text([ pos ], self.color, self.line_width, content=clip_text, size = self.font_size)
            new_text.move_cursor("End")
            self.objects.append(new_text)
            self.queue_draw()

    def paste_image(self, clip_img):
        """Create an image object from clipboard image."""
        pos = self.cursor_pos or (100, 100)
        self.current_object = Image([ pos ], self.color, self.line_width, clip_img)
        self.objects.append(self.current_object)
        self.queue_draw()

    def object_create_copy(self, obj, bb = None):
        """Copy the current object into a new object."""
        new_obj = copy.deepcopy(obj.to_dict())
        new_obj = Drawable.from_dict(new_obj)

        # move the new object to the current location
        pos = self.cursor_pos or (100, 100)
        if bb is None:
            bb  = new_obj.bbox()
        dx, dy = pos[0] - bb[0], pos[1] - bb[1]
        new_obj.move(dx, dy)

        self.objects.append(new_obj)
        self.queue_draw()

    def paste_content(self):
        """Paste content from clipboard."""

        print("paste_content:", self.clipboard)

        # internal paste
        if self.clipboard:
            print("Pasting content internally")
            if self.clipboard.type != "group":
                Raise("Internal clipboard is not a group")
            bb = self.clipboard.bbox()
            print("clipboard bbox:", bb)
            for obj in self.clipboard.objects:
                self.object_create_copy(obj, bb)
            return

        # external paste
        clipboard = self.gtk_clipboard
        clip_text = clipboard.wait_for_text()
        if clip_text:
            self.paste_text(clip_text)

        clip_img = clipboard.wait_for_image()
        if clip_img:
            self.paste_image(clip_img)

    def copy_content(self, destroy = False):
        """Copy content to clipboard."""
        if not self.selection:
            return

        print("Copying content", self.selection)
        clipboard = self.gtk_clipboard

        if self.selection.length() == 1:
            sel = self.selection.objects[0]
        else:
            sel = self.selection

        if sel.type == "text":
            text = "\n".join(sel.content)
            print("Copying text", text)
            # just copy the text
            clipboard.set_text(text, -1)
            clipboard.store()
        elif sel.type == "image":
            print("Copying image")
            # simply copy the image into clipboard
            clipboard.set_image(sel.image)
            clipboard.store()
        else:
            print("Copying another object")
            # draw a little image and copy it to clipboard
            img_copy = img_object_copy(sel)
            clipboard.set_image(img_copy)
            clipboard.store()

        print("Setting internal clipboard")
        self.clipboard = self.selection
        self.clipboard_owner = True

        if destroy:
            for obj in self.selection.objects:
                self.objects.remove(obj)
            self.selection = None
            self.queue_draw()

    def cut_content(self):
        """Cut content to clipboard."""
        self.copy_content(True)

    def text_size_decrease(self):
        """Decrease text size."""
        self.text_size_change(-1)

    def text_size_increase(self):
        """Increase text size."""
        self.text_size_change(1)

    def text_size_change(self, direction):
        """Change text size up or down."""
        obj = None
        if self.current_object and self.current_object.type == "text":
            obj = self.current_object
        elif self.selection and self.selection.type == "text":
            obj = self.selection

        if obj:
            obj.size += direction
            self.font_size = obj.size
            self.queue_draw()

    def selection_group(self):
        """Group selected objects."""
        if not self.selection or self.selection.length() < 2:
            return
        print("Grouping", self.selection.length(), "objects")
        new_grp_obj = DrawableGroup(self.selection.objects)
        for obj in self.selection.objects:
            self.objects.remove(obj)
        self.objects.append(new_grp_obj)
        self.selection = DrawableGroup([ new_grp_obj ])
        self.queue_draw()

    def selection_ungroup(self):
        """Ungroup selected objects."""
        if not self.selection:
            return
        for obj in self.selection.objects:
            if obj.type == "group":
                print("Ungrouping", obj)
                self.objects.extend(obj.objects)
                self.objects.remove(obj)
                self.queue_draw()
        return

    def select_all(self):
        """Select all objects."""
        if len(self.objects) == 0:
            return

        self.mode = 'move'
        self.default_cursor('move')

        self.selection = DrawableGroup([ self.objects[0] ])
        for obj in self.objects[1:]:
            self.selection.add(obj)
        self.queue_draw()

    def selection_delete(self):
        """Delete selected objects."""
        if self.selection:
            for obj in self.selection.objects:
                self.objects.remove(obj)
            self.selection = None
            self.dragobj   = None
            self.queue_draw()

    def select_next_object(self):
        """Select the next object."""
        if len(self.objects) == 0:
            return
        if not self.selection:
            self.selection = DrawableGroup([ self.objects[0] ])
        idx = self.objects.index(self.selection.objects[-1])
        idx += 1
        if idx >= len(self.objects):
            idx = 0
        self.selection = DrawableGroup([ self.objects[idx] ])
        self.queue_draw()

    def selection_fill(self):
        """Fill the selected object."""
        if self.selection:
            for obj in self.selection.objects:
                obj.fill(self.color)
            self.queue_draw()

    def select_previous_object(self):
        """Select the previous object."""
        if len(self.objects) == 0:
            return
        if not self.selection:
            self.selection = DrawableGroup([ self.objects[-1] ])
        idx = self.objects.index(self.selection.objects[0])
        idx -= 1
        if idx < 0:
            idx = len(self.objects) - 1
        self.selection = DrawableGroup([ self.objects[idx] ])
        self.queue_draw()

    def handle_shortcuts(self, keyname, ctrl, shift):
        """Handle keyboard shortcuts."""
        print(keyname)

        if shift:
            keyname = "Shift-" + keyname

        if ctrl:
            keyname = "Ctrl-" + keyname

        # these are single keystroke mode modifiers
        modes = { 'd': "draw", 't': "text", 'e': "eraser", 'm': "move", 'c': "circle", 'b': "box", 's': "move" }

        # these are single keystroke actions
        actions = {
            'h':                    {'action': self.show_help_dialog},
            'F1':                   {'action': self.show_help_dialog},
            'question':             {'action': self.show_help_dialog},
            'Ctrl-l':               {'action': self.clear},
            'Ctrl-b':               {'action': self.cycle_background},
            'x':                    {'action': self.exit},
            'Ctrl-q':               {'action': self.exit},
            'l':                    {'action': self.clear},
            'f':                    {'action': self.selection_fill, 'modes': ["box", "circle", "draw", "move"]},

            # dialogs
            'Ctrl-s':               {'action': self.save_drawing},
            'Ctrl-k':               {'action': self.select_color},

            # selections and moving objects
            'Tab':                  {'action': self.select_next_object, 'modes': ["move"]},
            'Shift-ISO_Left_Tab':   {'action': self.select_next_object, 'modes': ["move"]},
            'g':                    {'action': self.selection_group,    'modes': ["move"]},
            'u':                    {'action': self.selection_ungroup,  'modes': ["move"]},
            'Delete':               {'action': self.selection_delete,   'modes': ["move"]},
            'Ctrl-c':               {'action': self.copy_content,  'modes': ["move"]},
            'Ctrl-x':               {'action': self.cut_content,   'modes': ["move"]},
            'Ctrl-a':               {'action': self.select_all},
            'Ctrl-v':               {'action': self.paste_content},

            'Ctrl-plus':            {'action': self.text_size_increase, 'modes': ["text", "draw", "move"]},
            'Ctrl-minus':           {'action': self.text_size_decrease, 'modes': ["text", "draw", "move"]},
        }

        if keyname in modes:
            self.mode = modes[keyname]
            self.default_cursor(modes[keyname])
            self.queue_draw()
        elif keyname in actions:
            if not "modes" in actions[keyname] or self.mode in actions[keyname]["modes"]:
                actions[keyname]["action"]()
     
    def on_key_press(self, widget, event):
        """Handle keyboard events."""
        keyname = Gdk.keyval_name(event.keyval)
        char    = chr(Gdk.keyval_to_unicode(event.keyval))
        ctrl    = event.state & Gdk.ModifierType.CONTROL_MASK
        shift   = event.state & Gdk.ModifierType.SHIFT_MASK

        # End text input
        if keyname == "Escape":
            self.finish_text_input()

        # Handle ctrl-keyboard shortcuts 
        elif event.state & ctrl:
            self.handle_shortcuts(keyname, True, shift)
       
        # Handle text input
        elif self.current_object and self.current_object.type == "text":
            self.update_text_input(keyname, char)
        else:
        # handle single keystroke shortcuts
            self.handle_shortcuts(keyname, False, shift)

        return True

    def draw_dot(self, cr, x, y, diameter):
        """Draws a dot at the specified position with the given diameter."""
        cr.arc(x, y, diameter / 2, 0, 2 * 3.14159)  # Draw a circle
        cr.fill()  # Fill the circle to make a dot

    def make_cursors(self):
        """Create cursors for different modes."""
        self.cursors = {
            "hand":      Gdk.Cursor.new_from_name(self.get_display(), "hand1"),
            "move":      Gdk.Cursor.new_from_name(self.get_display(), "hand2"),
            "moving":    Gdk.Cursor.new_from_name(self.get_display(), "move"),
            "text":      Gdk.Cursor.new_from_name(self.get_display(), "text"),
            "eraser":    Gdk.Cursor.new_from_name(self.get_display(), "not-allowed"),
            "pencil":    Gdk.Cursor.new_from_name(self.get_display(), "pencil"),
            "draw":      Gdk.Cursor.new_from_name(self.get_display(), "pencil"),
            "crosshair": Gdk.Cursor.new_from_name(self.get_display(), "crosshair"),
            "circle":    Gdk.Cursor.new_from_name(self.get_display(), "crosshair"),
            "box":       Gdk.Cursor.new_from_name(self.get_display(), "crosshair"),
            "none":      Gdk.Cursor.new_from_name(self.get_display(), "none"),
            "default":   Gdk.Cursor.new_from_name(self.get_display(), "pencil")
        }

    def revert_cursor(self):
        """Revert to the default cursor."""
        if self.current_cursor == "default":
            return
        print("reverting cursor")
        self.get_window().set_cursor(self.cursors["default"])
        self.current_cursor = "default"

    def change_cursor(self, cursor_name):
        """Change the cursor to the specified cursor."""
        if self.current_cursor == cursor_name:
            return
        print("changing cursor to", cursor_name)
        cursor = self.cursors[cursor_name]
        self.get_window().set_cursor(cursor)
        self.current_cursor = cursor_name

    def default_cursor(self, cursor_name):
        """Set the default cursor to the specified cursor."""
        if self.current_cursor == cursor_name:
            return
        print("setting default cursor to", cursor_name)
        self.cursors["default"] = self.cursors[cursor_name]
        self.get_window().set_cursor(self.cursors["default"])
        self.current_cursor = cursor_name

    def select_color(self):
        """Select a color for drawing."""
        # Create a new color chooser dialog
        color_chooser = Gtk.ColorChooserDialog("Select Current Foreground Color", None)

        # Show the dialog
        response = color_chooser.run()

        # Check if the user clicked the OK button
        if response == Gtk.ResponseType.OK:
            color = color_chooser.get_rgba()
            self.color = (color.red, color.green, color.blue)
            if self.selection:
                for obj in self.selection.objects:
                    obj.color_set(self.color)

        # Don't forget to destroy the dialog
        color_chooser.destroy()

    def show_help_dialog(self):
        """Show the help dialog."""
        dialog = HelpDialog(self)
        response = dialog.run()
        dialog.destroy()

    def save_drawing(self):
        """Save the drawing to a file."""
        # Choose where to save the file
        dialog = Gtk.FileChooserDialog("Save as", self, Gtk.FileChooserAction.SAVE,
            (Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, Gtk.STOCK_SAVE, Gtk.ResponseType.OK))
        dialog.set_default_response(Gtk.ResponseType.OK)
        response = dialog.run()
        if response == Gtk.ResponseType.OK:
            filename = dialog.get_filename()
            # Ensure the filename has the correct extension
            if not filename.endswith('.svg'):
                filename += '.svg'
            #self.export_to_png(filename)
            self.export(filename, "svg")
        dialog.destroy()

    def export(self, filename, file_format):
        """Export the drawing to a file."""
        # Create a Cairo surface of the same size as the window content
        width, height = self.get_size()
        if file_format == "png":
            surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        elif file_format == "svg":
            surface = cairo.SVGSurface(filename, width, height)
        else:
            raise ValueError("Invalid file format")

        cr = cairo.Context(surface)
        cr.set_source_rgba(1, 1, 1)
        cr.paint()
        self.draw(cr)

        # Save the surface to the file
        if file_format == "png":
            surface.write_to_png(filename)
        elif file_format == "svg":
            surface.finish()

    def save_state(self): 
        """Save the current drawing state to a file."""
        config = {
                'transparent': self.transparent,
                'font_size': self.font_size,
                'line_width': self.line_width,
                'color': self.color
        }

        objects = [ obj.to_dict() for obj in self.objects ]

        state = { 'config': config, 'objects': objects }
        with open(savefile, 'wb') as f:
            #yaml.dump(state, f)
            pickle.dump(state, f)
        print("Saved drawing to", savefile)

    def load_state(self):
        """Load the drawing state from a file."""
        if not os.path.exists(savefile):
            print("No saved drawing found at", savefile)
            return
        with open(savefile, 'rb') as f:
            state = pickle.load(f)
            #state = yaml.load(f, Loader=yaml.FullLoader)
        self.objects           = [ Drawable.from_dict(d) for d in state['objects'] ]
        self.transparent       = state['config']['transparent']
        self.font_size         = state['config']['font_size']
        self.line_width        = state['config']['line_width']
        self.color             = state['config']['color']


## ---------------------------------------------------------------------

if __name__ == "__main__":
    win = TransparentWindow()
    win.show_all()
    win.present()
    win.change_cursor("default")
    #win.stick()

    Gtk.main()

