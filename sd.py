#!/usr/bin/env python3

## MIT License
## 
## Copyright (c) 2024 January Weiner
## 
## Permission is hereby granted, free of charge, to any person obtaining a copy
## of this software and associated documentation files (the "Software"), to deal
## in the Software without restriction, including without limitation the rights
## to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
## copies of the Software, and to permit persons to whom the Software is
## furnished to do so, subject to the following conditions:
## 
## The above copyright notice and this permission notice shall be included in all
## copies or substantial portions of the Software.
## 
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
## OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
## SOFTWARE.
"""
ScreenDrawer - a simple drawing program that allows you to draw on the screen

Usage:
  sd.py [options] [file.sdrw [file.[png, pdf, svg]]]

See README.md for more information.
"""

import copy
import pickle
import traceback
import colorsys
from sys import exc_info, argv, exit
import os
from os import path
import time
import math
import base64
import tempfile
from io import BytesIO

import warnings
import argparse

import pyautogui
from PIL import ImageGrab

import yaml
import cairo
import appdirs

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GdkPixbuf, Pango, GLib


# ---------------------------------------------------------------------
# These are various classes and utilities for the sd.py script. In the
# "skeleton" variant, they are just imported. In the "full" variant, they
# the files are directly, physically inserted in order to get one big fat
# Python script that can just be copied.

"""
General utility functions for the ScreenDrawer application.
"""


def get_default_savefile(app_name, app_author):
    """Get the default save file for the application."""

    # Get user-specific data directory
    user_data_dir = appdirs.user_data_dir(app_name, app_author)
    print(f"User data directory: {user_data_dir}")
    # Create the directory if it does not exist
    os.makedirs(user_data_dir, exist_ok=True)
    # The filename for the save file: dir + "savefile"
    savefile = os.path.join(user_data_dir, "savefile")
    return savefile


def get_color_under_cursor():
    """Get the color under the cursor."""

    # Get the current position of the cursor
    x, y = pyautogui.position()
    # Grab a single pixel at the cursor's position
    pixel = ImageGrab.grab(bbox=(x, y, x+1, y+1))
    # Retrieve the color of the pixel
    color = pixel.getpixel((0, 0))
    # divide by 255
    color = (color[0] / 255, color[1] / 255, color[2] / 255)
    return color

def rgb_to_hex(rgb):
    """Convert an RGB color to a hexadecimal string."""
    r, g, b = [int(255 * c) for c in rgb]
    return f"#{r:02x}{g:02x}{b:02x}"

def get_screenshot(window, x0, y0, x1, y1):
    """Capture a screenshot of a specific area on the screen."""

    # Get the absolute position of the area to capture
    wpos = window.get_position()
    x0, y0 = wpos[0] + x0, wpos[1] + y0
    x1, y1 = wpos[0] + x1, wpos[1] + y1

    screenshot = ImageGrab.grab(bbox=(x0, y0, x1, y1))

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
        screenshot.save(temp_file, format="PNG")
        temp_file_name = temp_file.name

    print("Saved screenshot to temporary file:", temp_file_name)

    pixbuf = GdkPixbuf.Pixbuf.new_from_file(temp_file_name)
    return pixbuf, temp_file_name

def flatten_and_unique(lst, result_set=None):
    """Flatten a list and remove duplicates."""
    if result_set is None:
        result_set = set()

    for item in lst:
        if isinstance(item, list):
            flatten_and_unique(item, result_set)
        else:
            result_set.add(item)

    return list(result_set)

def sort_by_stack(objs, stack):
    """Sort a list of objects by their position in the stack."""
    # sort the list of objects by their position in the stack
    return sorted(objs, key=stack.index)

def distance(p1, p2):
    """Calculate the Euclidean distance between two points."""
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def bezier_point(t, start, control, end):
    """Calculate a point on a quadratic Bézier curve."""
    # B(t) = (1-t)^2 * P0 + 2(1-t)t * P1 + t^2 * P2, 0 <= t <= 1
    x = (1 - t)**2 * start[0] + 2 * (1 - t) * t * control[0] + t**2 * end[0]
    y = (1 - t)**2 * start[1] + 2 * (1 - t) * t * control[1] + t**2 * end[1]
    return (x, y)

def segment_intersection(p1, p2, p3, p4):
    """
    Calculate the intersection of two line segments.
    
    Parameters:
    - p1, p2: Coordinates of the first line segment's endpoints.
    - p3, p4: Coordinates of the second line segment's endpoints.
    
    Returns:
    - A tuple (True, (x, y)) if segments intersect, with (x, y) being the intersection point.
    - A tuple (False, None) if segments do not intersect.
    """

    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if denom == 0:
        return (False, None)  # Lines are parallel

    t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
    u = ((x1-x3)*(y1-y2) - (y1-y3)*(x1-x2)) / denom

    if 0 <= t <= 1 and 0 <= u <= 1:
        # The segments intersect
        intersect_x = x1 + t * (x2 - x1)
        intersect_y = y1 + t * (y2 - y1)
        return (True, (intersect_x, intersect_y))

    return (False, None)  # No intersection

def pp(p):
    """return point in integers"""
    return [int(p[0]), int(p[1])]

def remove_intersections(outline_l, outline_r):
    """Remove intersections between the left and right outlines."""
    # Does not work yet
    n = len(outline_l)
    if n < 2:
        return outline_l, outline_r
    if n != len(outline_r):
        print("outlines of different length")
        return outline_l, outline_r

    out_ret_l = []
    out_ret_r = []

    for i in range(n - 1):

        out_ret_l.append(outline_l[i])
        for j in range(i + 1, n - 1):
            print("i", i, "left segment: ", pp(outline_l[i]), pp(outline_l[i + 1]))
            print("j", j, "right segment: ", pp(outline_r[j]), pp(outline_r[j + 1]))
            intersect, point = segment_intersection(outline_l[i], outline_l[i + 1],
                                                outline_r[j], outline_r[j + 1])
            if intersect:
                print("FOUND Intersection at", point, "i", i, "j", j)
                #out_ret_l, out_ret_r = out_ret_r, out_ret_l

            out_ret_r.append(outline_r[i])
                # exchange the remainder between outlines
                #tmp = outline_l[(i + 1):]
                #outline_l[(i + 1):] = outline_r[(i + 1):]
                #outline_r[(i + 1):] = tmp

    return out_ret_l, out_ret_r

def calculate_angle2(p0, p1):
    """Calculate angle between vectors given by p0 and p1"""
    x1, y1 = p0
    x2, y2 = p1
    dot = x1 * x2 + y1 * y2
    det = x1 * y2 - y1 * x2
    angle = math.atan2(det, dot)
    return angle

def calculate_angle(p0, p1, p2):
    """Calculate the angle between the line p0->p1 and p1->p2 in degrees."""
    a = math.sqrt((p1[0] - p0[0])**2 + (p1[1] - p0[1])**2)
    b = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    c = math.sqrt((p2[0] - p0[0])**2 + (p2[1] - p0[1])**2)
    if a * b == 0:
        return 0
    cos_angle = (a**2 + b**2 - c**2) / (2 * a * b)
    # Clamp cos_angle to the valid range [-1, 1] to avoid math domain error
    cos_angle = max(-1, min(1, cos_angle))
    angle = math.acos(cos_angle)
    return math.degrees(angle)

def midpoint(p1, p2):
    """Calculate the midpoint between two points."""
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

def smooth_path(coords, pressure=None, threshold=20):
    """Smooth a path using cubic Bézier curves."""

    if len(coords) < 3:
        return coords, pressure  # Not enough points to smooth

    if pressure and len(pressure) != len(coords):
        #raise ValueError("Pressure and coords must have the same length")
        print("Pressure and coords must have the same length:", len(pressure), len(coords))
        return coords, pressure

    #print("smoothing path with", len(coords), "points")
    smoothed_coords = [coords[0]]  # Start with the first point
    if pressure:
        new_pressure    = [pressure[0]]
    else:
        new_pressure = None

    t_values = [t / 10.0 for t in range(1, 5)]

    for i in range(1, len(coords) - 1):
        p0 = coords[i - 1]
        p1 = coords[i]
        p2 = coords[i + 1]

        if pressure:
            prev_pressure = pressure[i - 1]
            current_pressure = pressure[i]
            next_pressure = pressure[i + 1]

        # Calculate distances to determine if smoothing is needed
        dist_to_prev = math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)
        dist_to_next = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

        #angle = calculate_angle(p0, p1, p2)
        #print("angle is", angle)

        if dist_to_prev > threshold or dist_to_next > threshold:
            # Calculate control points for smoother transitions
            control1 = midpoint(p0, p1)
            control2 = midpoint(p1, p2)

            # Generate intermediate points for the cubic Bézier curve
            for t in t_values:
                x, y = calc_bezier_coords(t, p1, p2, control1, control2)
                if pressure:
                    new_pressure.append((1-t) * prev_pressure + t * next_pressure)
                smoothed_coords.append((x, y))
        else:
            # For shorter segments, just add the current point
            smoothed_coords.append(p1)
            if pressure:
                new_pressure.append(current_pressure)

    smoothed_coords.append(coords[-1])  # Ensure the last point is added
    if pressure:
        new_pressure.append(pressure[-1])
    return smoothed_coords, new_pressure

def calc_bezier_coords(t, p1, p2, control1, control2):
    """Calculate a point on a cubic Bézier curve."""
    t0 = (1 - t) ** 3
    t1 = 3 * (1 - t) ** 2 * t
    t2 = 3 * (1 - t) * t ** 2
    t3 = t ** 3
    x = t0 * control1[0] + t1 * p1[0] + t2 * control2[0] + t3 * p2[0]
    y = t0 * control1[1] + t1 * p1[1] + t2 * control2[1] + t3 * p2[1]

    return x, y


def distance_point_to_segment(p, segment):
    """Calculate the distance from a point (px, py) to a line segment (x1, y1) to (x2, y2)."""
    # Calculate the line segment's length squared
    px, py = p
    x1, y1 = segment[0]
    x2, y2 = segment[1]
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

def determine_side_math(p1, p2, p3):
    """Determine the side of a point p3 relative to a line segment p1->p2."""
    det = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
    return 'left' if det > 0 else 'right'

def calc_arc_coords(p1, p2, p3, n = 20):
    """
    Calculate the coordinates of an arc between two points.
    The point p3 is on the opposite side of the arc from the line p1->p2.
    """
    x1, y1 = p1
    x2, y2 = p2
    p0 = ((x1 + x2) / 2, (y1 + y2) / 2)
    x0, y0 = p0
    radius = math.sqrt((x2 - x1)**2 + (y2 - y1)**2) / 2

    side = determine_side_math(p1, p2, p3)

    # calculate the from p0 to p1
    a1 = math.atan2(y1 - p0[1], x1 - p0[0])
    a2 = math.atan2(y2 - p0[1], x2 - p0[0])

    if side == 'left' and a1 > a2:
        a2 += 2 * math.pi
    elif side == 'right' and a1 < a2:
        a1 += 2 * math.pi

    # calculate 20 points on the arc between a1 and a2
    coords = []
    for i in range(n):
        a = a1 + (a2 - a1) * i / (n - 1)
        x = x0 + radius * math.cos(a)
        y = y0 + radius * math.sin(a)
        coords.append((x, y))

    return coords


def calc_rotation_angle(origin, p1, p2):
    """
    Calculate the rotation angle based on the initial (p1) and new (p2)
    cursor position and the rotation centre (origin).

    Arguments:
    origin -- the rotation centre
    p1     -- the initial cursor position
    p2     -- the new cursor position
    """

    # x, y are the new positions of the cursor
    # we need to calculate the angle between two lines:
    # 1. the line between the rotation centre and the origin
    # 2. the line between the rotation centre and the new position
    x0, y0 = origin[0], origin[1]
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]
    angle = math.atan2(y2 - y0, x2 - x0) - math.atan2(y1 - y0, x1 - x0)
    return angle

def coords_rotate(coords, angle, origin):
    """Rotate a set of coordinates around a given origin."""
    ret = []
    for x, y in coords:
        x0, y0 = x - origin[0], y - origin[1]
        x1 = x0 * math.cos(angle) - y0 * math.sin(angle)
        y1 = x0 * math.sin(angle) + y0 * math.cos(angle)
        ret.append((x1 + origin[0], y1 + origin[1]))
    return ret

def normal_vec(p0, p1):
    """Calculate the normal vector of a line segment."""
    #dx, dy = x1 - x0, y1 - y0
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    length = math.sqrt(dx**2 + dy**2)
    dx, dy = dx / length, dy / length
    return -dy, dx

def transform_coords(coords, bb1, bb2):
    """Transform coordinates from one bounding box to another."""
    x0, y0, w0, h0 = bb1
    x1, y1, w1, h1 = bb2
    if w0 == 0 or h0 == 0:
        # issue warning
        warnings.warn("Bounding box has zero width or height")
        return coords
    ret = [
        (x1 + (x - x0) / w0 * w1, y1 + (y - y0) / h0 * h1)
        for x, y in coords
    ]
    return ret

def move_coords(coords, dx, dy):
    """Move a path by a given offset."""
    for i in range(len(coords)):
        coords[i] = (coords[i][0] + dx, coords[i][1] + dy)
    return coords

def path_bbox(coords, lw = 0):
    """Calculate the bounding box of a path."""
    if not coords:
        return (0, 0, 0, 0)

    left = min(p[0] for p in coords) - lw/2
    top = min(p[1] for p in coords) - lw/2
    width  =    max(p[0] for p in coords) - left + lw/2
    height =    max(p[1] for p in coords) - top + lw/2
    return (left, top, width, height)

def find_obj_close_to_click(click_x, click_y, objects, threshold):
    """Find first object that is close to a click."""
    for obj in objects[::-1]: # loop in reverse order to find the topmost object
        if not obj is None and obj.is_close_to_click(click_x, click_y, threshold):
            return obj

    return None

def find_obj_in_bbox(bbox, objects):
    """Find objects that are inside a given bounding box."""
    x, y, w, h = bbox
    ret = []
    for obj in objects:
        x_o, y_o, w_o, h_o = obj.bbox()
        if w_o < 0:
            x_o += w_o
            w_o = -w_o
        if h_o < 0:
            y_o += h_o
            h_o = -h_o
        if x_o >= x and y_o >= y and x_o + w_o <= x + w and y_o + h_o <= y + h:
            ret.append(obj)
    return ret

def is_click_close_to_path(click_x, click_y, path, threshold):
    """Check if a click is close to any segment in the path."""
    point = (click_x, click_y)

    for i in range(len(path) - 1):
        segment_start = path[i]
        segment_end = path[i + 1]
        dist = distance_point_to_segment(point,
                                         [ segment_start, segment_end])
        if dist <= threshold:
            return True
    return False

def is_click_in_bbox(click_x, click_y, bbox):
    """Check if a click is inside a bounding box."""
    x, y, w, h = bbox
    return x <= click_x <= x + w and y <= click_y <= y + h

def is_click_in_bbox_corner(click_x, click_y, bbox, threshold):
    """Check if a click is in the corner of a bounding box."""
    x, y, w, h = bbox

    # make sure that the corner capture area leaves enough space for the
    # grab area in the middle of the bbox
    if w < 2 * threshold:
        w += 2 * threshold
        x -= threshold

    if h < 2 * threshold:
        h += 2 * threshold
        y -= threshold

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

def draw_dot(cr, x, y, diameter):
    """Draws a dot at the specified position with the given diameter."""
    cr.arc(x, y, diameter / 2, 0, 2 * 3.14159)  # Draw a circle
    cr.fill()  # Fill the circle to make a dot

def base64_to_pixbuf(image_base64):
    """Convert a base64 image to a pixbuf."""
    image_binary = base64.b64decode(image_base64)
    image_io = BytesIO(image_binary)
    loader = GdkPixbuf.PixbufLoader.new_with_type('png')  # Specify the image format if known
    loader.write(image_io.getvalue())
    loader.close()  # Finalize the loader
    image = loader.get_pixbuf()  # Get the loaded GdkPixbuf
    return image

"""
This module contains the commands that can be executed on the objects. They
should be undoable and redoable. It is their responsibility to update the
state of the objects they are acting on.
"""



## ---------------------------------------------------------------------
## These are the commands that can be executed on the objects. They should
## be undoable and redoable. It is their responsibility to update the
## state of the objects they are acting on.

class Command:
    """Base class for commands."""
    def __init__(self, mytype, objects):
        self.obj   = objects
        self.__type   = mytype
        self.__undone = False

    def command_type(self):
        """Return the type of the command."""
        return self.__type

    def undo(self):
        """Undo the command."""
        raise NotImplementedError("undo method not implemented")

    def redo(self):
        """Redo the command."""
        raise NotImplementedError("redo method not implemented")

    def undone(self):
        """Return whether the command has been undone."""
        return self.__undone

    def undone_set(self, value):
        """Set the undone status of the command."""
        self.__undone = value

class CommandGroup(Command):
    """Simple class for handling groups of commands."""
    def __init__(self, commands, page=None):
        super().__init__("group", objects=None)
        self.__commands = commands
        self.__page = page

    def undo(self):
        """Undo the command."""
        if self.undone():
            return None
        for cmd in self.__commands:
            cmd.undo()
        self.undone_set(True)
        return self.__page

    def redo(self):
        """Redo the command."""
        if not self.undone():
            return None
        for cmd in self.__commands:
            cmd.redo()
        self.undone_set(False)
        return self.__page

class DeletePageCommand(Command):
    """
    Simple class for handling deleting pages.

    """
    def __init__(self, page, prev_page, next_page):
        super().__init__("delete_page", None)
        self.__prev = prev_page
        self.__next = next_page
        self.__page = page

        if self.__prev:
            # set the previous page's next to our next
            self.__prev.next_set(self.__next)
        if self.__next:
            # set the next page's previous to our previous
            self.__next.prev_set(self.__prev)

    def undo(self):
        """Undo the command."""
        if self.undone():
            return None
        if self.__prev:
            self.__prev.next_set(self.__page)
        if self.__next:
            self.__next.prev_set(self.__page)
        self.undone_set(True)
        return self.__page

    def redo(self):
        """Redo the command."""
        if not self.undone():
            return None
        if self.__prev:
            self.__prev.next_set(self.__next)
        if self.__next:
            self.__next.prev_set(self.__prev)
        self.undone_set(False)
        return self.__page

class DeleteLayerCommand(Command):
    """Simple class for handling deleting layers of a page."""
    def __init__(self, page, layer, layer_pos):
        """Simple class for handling deleting layers of a page."""
        super().__init__("delete_layer", None)
        self.__layer_pos = layer_pos
        self.__layer = layer
        self.__page  = page

    def undo(self):
        """Undo the command."""
        if self.undone():
            return None
        self.__page.layer(self.__layer, self.__layer_pos)
        self.undone_set(True)
        return self.__page

    def redo(self):
        """Redo the command."""
        if not self.undone():
            return None
        self.__page.delete_layer(self.__layer_pos)
        self.undone_set(False)
        return self.__page

class GroupObjectCommand(Command):
    """Simple class for handling grouping objects."""
    def __init__(self, objects, stack, selection_object = None, page = None):
        objects = sort_by_stack(objects, stack)

        super().__init__("group", objects)
        self.__stack      = stack
        self.__stack_copy = stack[:]

        self.__page      = page
        self.__selection = selection_object

        self.__group = DrawableGroup(self.obj)
        #self.__grouped_objects = self.obj[:]

        # position of the last object in stack
        idx = self.__stack.index(self.obj[-1])

        # add group to the stack at the position of the last object
        self.__stack.insert(idx, self.__group)

        for obj in self.obj:
            self.__stack.remove(obj)

        if self.__selection:
            self.__selection.set([ self.__group ])

    def swap_stacks(self):
        """Swap the stack with the copy of the stack."""
        # put the copy of the stack into the stack,
        # and the stack into the copy
        tmp = self.__stack[:]
        self.__stack[:] = self.__stack_copy[:]
        self.__stack_copy = tmp

    def undo(self):
        """Undo the command."""
        if self.undone():
            return None
        self.swap_stacks()
        self.undone_set(True)
        if self.__selection:
            self.__selection.set(self.obj)
        return self.__page

    def redo(self):
        """Redo the command."""
        if not self.undone():
            return None
        self.swap_stacks()
        self.undone_set(False)
        if self.__selection:
            self.__selection.set([ self.__group ])
        return self.__page

class UngroupObjectCommand(Command):
    """Simple class for handling ungrouping objects."""
    def __init__(self, objects, stack, selection_object = None, page = None):
        super().__init__("ungroup", objects)
        self.__stack = stack
        self.__stack_copy = stack[:]
        self.__selection = selection_object
        self.__page = page

        new_objects = []
        n = 0

        for obj in self.obj:
            if not obj.type == "group":
                print("Object is not a group, ignoring:", obj)
                print("object type:", obj.type)
                continue

            n += 1
            # position of the group in the stack
            idx = self.__stack.index(obj)

            # remove the group from the stack
            self.__stack.remove(obj)

            # add the objects back to the stack
            for subobj in obj.objects:
                self.__stack.insert(idx, subobj)
                new_objects.append(subobj)

        if n > 0 and self.__selection:
            self.__selection.set(new_objects)

    def swap_stacks(self):
        """Swap the stack with the copy of the stack."""
        # put the copy of the stack into the stack,
        # and the stack into the copy
        tmp = self.__stack[:]
        self.__stack[:] = self.__stack_copy[:]
        self.__stack_copy = tmp

    def undo(self):
        """Undo the command."""
        if self.undone():
            return None
        self.swap_stacks()
        self.undone_set(True)
        if self.__selection:
            self.__selection.set([ self.obj ])
        return self.__page

    def redo(self):
        """Redo the command."""
        if not self.undone():
            return None
        self.swap_stacks()
        self.undone_set(False)
        if self.__selection:
            self.__selection.set(self.obj)
        return self.__page



class RemoveCommand(Command):
    """Simple class for handling deleting objects."""
    def __init__(self, objects, stack, page = None):
        super().__init__("remove", objects)
        self.__stack = stack
        self.__page = page

        # remove the objects from the stack
        for obj in self.obj:
            self.__stack.remove(obj)

    def undo(self):
        if self.undone():
            return None
        for obj in self.obj:
        # XXX: it should insert at the same position!
            self.__stack.append(obj)
        self.undone_set(True)
        return self.__page

    def redo(self):
        if not self.undone():
            return None
        for obj in self.obj:
            self.__stack.remove(obj)
        self.undone_set(False)
        return self.__page

class AddCommand(Command):
    """Simple class for handling creating objects."""
    def __init__(self, objects, stack, page = None):
        super().__init__("add", objects)
        self.__stack = stack
        self.__stack.append(self.obj)
        self.__page = page

    def undo(self):
        if self.undone():
            return None
        self.__stack.remove(self.obj)
        self.undone_set(True)
        return self.__page

    def redo(self):
        if not self.undone():
            return None
        self.__stack.append(self.obj)
        self.undone_set(False)
        return self.__page

class TransmuteCommand(Command):
    """
    Turning object(s) into another type.

    Internally, for each object we create a new object of the new type, and
    replace the old object with the new one in the stack.

    For undo method, we should store the old object as well as its position in the stack.
    However, we don't. Instead we just slap the old object back onto the stack.
    """

    def __init__(self, objects, stack, new_type, selection_objects = None, page = None):
        super().__init__("transmute", objects)
        #self.__new_type = new_type
        self.__old_objs = [ ]
        self.__new_objs = [ ]
        self.__stack    = stack
        self.__selection_objects = selection_objects
        self.__page = page
        print("executing transmute; undone = ", self.undone())

        for obj in self.obj:
            new_obj = DrawableFactory.transmute(obj, new_type)

            if not obj in self.__stack:
                raise ValueError("TransmuteCommand: Got Object not in stack:", obj)

            if obj == new_obj: # ignore if no transmutation
                continue

            self.__old_objs.append(obj)
            self.__new_objs.append(new_obj)
            self.__stack.remove(obj)
            self.__stack.append(new_obj)

        if self.__selection_objects:
            self.map_selection()

    def map_selection(self):
        """Map the selection objects to the new objects."""
        obj_map = self.obj_map()
        # XXX this should not change the order of the objects
        self.__selection_objects[:] = [ obj_map.get(obj, obj) for obj in self.__selection_objects ]

    def obj_map(self):
        """Return a dictionary mapping old objects to new objects."""
        return { self.__old_objs[i]: self.__new_objs[i] for i in range(len(self.__old_objs)) }

    def undo(self):
        """replace all the new objects with the old ones in the stack"""
        if self.undone():
            return None
        for obj in self.__new_objs:
            self.__stack.remove(obj)
        for obj in self.__old_objs:
            self.__stack.append(obj)
        self.undone_set(True)
        if self.__selection_objects:
            self.map_selection()
        return self.__page

    def redo(self):
        """put the new objects again on the stack and remove the old ones"""
        if not self.undone():
            return None
        for obj in self.__old_objs:
            self.__stack.remove(obj)
        for obj in self.__new_objs:
            self.__stack.append(obj)
        self.undone_set(False)
        if self.__selection_objects:
            self.map_selection()
        return self.__page

class ZStackCommand(Command):
    """Simple class for handling z-stack operations."""
    def __init__(self, objects, stack, operation, page = None):
        super().__init__("z_stack", objects)
        self._operation  = operation
        self.__stack      = stack
        self.__page = page

        for obj in objects:
            if not obj in stack:
                raise ValueError("Object not in stack:", obj)

        self._objects = sort_by_stack(objects, stack)
        # copy of the old stack
        self.__stack_orig = stack[:]

        if operation == "raise":
            self.hoist() # raise is reserved
        elif operation == "lower":
            self.lower()
        elif operation == "top":
            self.top()
        elif operation == "bottom":
            self.bottom()
        else:
            raise ValueError("Invalid operation:", operation)

    ## here is the problem: not all objects that we get exist in the stack.
    ## u

    def hoist(self):
        """Move the objects towards the top of the stack."""
        li = self.__stack.index(self._objects[-1])
        n  = len(self.__stack)

        # if the last element is already on top, we just move everything to
        # the top
        if li == n - 1:
            self.top()
            return

        # otherwise, we move all the objects to the position of the element
        # following the last one. Then, we just copy the elements from the
        # stack to the new stack, and when we see the indicator object, we
        # add our new objects.
        ind_obj = self.__stack[li + 1]

        new_list = []
        for i in range(n):
            o = self.__stack[i]
            if not o in self._objects:
                new_list.append(o)
            if o == ind_obj:
                new_list.extend(self._objects)

        self.__stack[:] = new_list[:]

    def lower(self):
        """Move the objects towards the bottom of the stack."""
        fi = self.__stack.index(self._objects[0])
        n  = len(self.__stack)

        if fi == 0:
            self.bottom()
            return

        # otherwise, we move all the objects to the position of the element
        # preceding the first one. Then, we just copy the elements from the
        # stack to the new stack, and when we see the indicator object, we
        # this could be done more efficiently, but that way it is clearer

        ind_obj = self.__stack[fi - 1]
        new_list = []
        for i in range(n):
            o = self.__stack[i]
            if o == ind_obj:
                new_list.extend(self._objects)
            if not o in self._objects:
                new_list.append(o)

        self.__stack[:] = new_list[:]

    def top(self):
        """Move the objects to the top of the stack."""
        for obj in self._objects:
            self.__stack.remove(obj)
            self.__stack.append(obj)

    def bottom(self):
        """Move the objects to the bottom of the stack."""
        for obj in self.obj[::-1]:
            self.__stack.remove(obj)
            self.__stack.insert(0, obj)

    def undo(self):
        """Undo the command."""
        if self.undone():
            return None
        self.swap_stacks(self.__stack, self.__stack_orig)
        self.undone_set(True)
        return self.__page

    def redo(self):
        """Redo the command."""
        if not self.undone():
            return None
        self.swap_stacks(self.__stack, self.__stack_orig)
        self.undone_set(False)
        return self.__page

    def swap_stacks(self, stack1, stack2):
        """Swap two stacks"""
        stack1[:], stack2[:] = stack2[:], stack1[:]


class MoveResizeCommand(Command):
    """
    Simple class for handling move and resize events.

    Attributes:
        start_point (tuple): the point where the first click was made
        origin (tuple): the original position of the object
        bbox (tuple): the bounding box of the object

    Arguments:
        type (str): the type of the command
        obj (Drawable): the object to be moved or resized
        origin (tuple): the original position of the object
    """

    def __init__(self, mytype, obj, origin):
        super().__init__(mytype, obj)
        self.start_point = origin
        self.origin      = origin
        self.bbox        = obj.bbox()

    def event_update(self, x, y):
        """Update the move or resize event."""
        raise NotImplementedError("event_update method not implemented for type", self.__type)

    def event_finish(self):
        """Finish the move or resize event."""
        raise NotImplementedError("event_finish method not implemented for type", self.__type)

class RotateCommand(MoveResizeCommand):
    """
    Simple class for handling rotate events.

    Attributes:

        corner (str): the corner which is being dragged, e.g. "upper_left"
        _rotation_centre (tuple): the point around which the rotation is done

    Arguments:
        obj (Drawable): object to be rotated
        origin (tuple, optional): where the first click was made
        corner (str, optional): which corner has been clicked
        angle (float, optional): set the rotation angle directly
    """

    def __init__(self, obj, origin=None, corner=None, angle = None, page = None):
        super().__init__("rotate", obj, origin)
        self.corner      = corner
        bb = obj.bbox()
        self.__rotation_centre = (bb[0] + bb[2] / 2, bb[1] + bb[3] / 2)
        obj.rotate_start(self.__rotation_centre)
        self.__page = page

        if not angle is None:
            self.obj.rotate(angle, set = False)

        self._angle = 0

    def event_update(self, x, y):
        angle = calc_rotation_angle(self.__rotation_centre, self.start_point, (x, y))
        d_a = angle - self._angle
        self._angle = angle
        self.obj.rotate(d_a, set_angle = False)

    def event_finish(self):
        self.obj.rotate_end()

    def undo(self):
        if self.undone():
            return None
        self.obj.rotate_start(self.__rotation_centre)
        self.obj.rotate(-self._angle)
        self.obj.rotate_end()
        self.undone_set(True)
        return self.__page

    def redo(self):
        if not self.undone():
            return None
        self.obj.rotate_start(self.__rotation_centre)
        self.obj.rotate(self._angle)
        self.obj.rotate_end()
        self.undone_set(False)
        return self.__page

class MoveCommand(MoveResizeCommand):
    """Simple class for handling move events."""
    def __init__(self, obj, origin, page = None):
        super().__init__("move", obj, origin)
        self.__last_pt = origin
        self.__page = page
        print("MoveCommand: origin is", origin)

    def event_update(self, x, y):
        dx = x - self.__last_pt[0]
        dy = y - self.__last_pt[1]

        self.obj.move(dx, dy)
        self.__last_pt = (x, y)

    def event_finish(self):
        print("MoveCommand: finish")

    def undo(self):
        if self.undone():
            return None
        print("MoveCommand: undo")
        dx = self.start_point[0] - self.__last_pt[0]
        dy = self.start_point[1] - self.__last_pt[1]
        self.obj.move(dx, dy)
        self.undone_set(True)
        return self.__page

    def redo(self):
        if not self.undone():
            return None
        dx = self.start_point[0] - self.__last_pt[0]
        dy = self.start_point[1] - self.__last_pt[1]
        self.obj.move(-dx, -dy)
        self.undone_set(False)
        return self.__page


class ResizeCommand(MoveResizeCommand):
    """Simple class for handling resize events."""
    def __init__(self, obj, origin, corner, proportional = False, page = None):
        super().__init__("resize", obj, origin)
        self.corner = corner
        obj.resize_start(corner, origin)
        self._orig_bb = obj.bbox()
        self._prop    = proportional
        ## XXX check the bb for pitfalls
        self._orig_bb_ratio = self._orig_bb[3] / self._orig_bb[2]
        self.__newbb = None
        self.__page = page


    def undo(self):
        """Undo the command."""
        if self.undone():
            return None
        obj = self.obj
        pt  = (self._orig_bb[0], self._orig_bb[1])
        obj.resize_start(self.corner, pt)
        self.obj.resize_update(self._orig_bb)
        obj.resize_end()
        self.undone_set(True)
        return self.__page

    def redo(self):
        """Redo the command."""
        if not self.undone():
            return None
        obj = self.obj
        obj.resize_start(self.corner, self.start_point)
        obj.resize_update(self.__newbb)
        obj.resize_end()
        self.undone_set(False)
        return self.__page

    def event_finish(self):
        """Finish the resize event."""
        self.obj.resize_end()

    def event_update(self, x, y):
        """Update the resize event."""
        bb = self._orig_bb
        corner = self.corner

        if corner in ["upper_left", "lower_right"]:
            dx = x - self.origin[0]
            dy = y - self.origin[1]
        else:
            dx = (self.origin[0] + bb[2]) - x
            dy = y - self.origin[1] + bb[3]

        if dx == 0 or dy == 0:
            return

        if self._prop:
            if dy / dx > self._orig_bb_ratio:
                dy = dx * self._orig_bb_ratio
            else:
                dx = dy / self._orig_bb_ratio

        if corner == "lower_left":
            newbb = (bb[0] + bb[2] - dx, bb[1], dx, dy)
        elif corner == "upper_right":
            newbb = (bb[0], bb[1] + dy - bb[3], bb[2] * 2 - dx, bb[3] - dy + bb[3])
        elif corner == "upper_left":
            newbb = (bb[0] + dx, bb[1] + dy, bb[2] - dx, bb[3] - dy)
        elif corner == "lower_right":
            newbb = (bb[0], bb[1], bb[2] + dx, bb[3] + dy)
        else:
            raise ValueError("Invalid corner:", corner)

        self.__newbb = newbb
        self.obj.resize_update(newbb)


class SetPropCommand(Command):
    """Simple class for handling color changes."""
    def __init__(self, mytype, objects, prop, get_prop_func, set_prop_func, page = None):
        super().__init__(mytype, objects.get_primitive())
        self.__prop = prop
        self.__page = page
        #self.__get_prop_func = get_prop_func
        self.__set_prop_func = set_prop_func
        self.__undo_dict = { obj: get_prop_func(obj) for obj in self.obj }

        for obj in self.obj:
            set_prop_func(obj, prop)

    def undo(self):
        """Undo the command."""
        if self.undone():
            return None
        for obj in self.obj:
            if obj in self.__undo_dict:
                self.__set_prop_func(obj, self.__undo_dict[obj])
        self.undone_set(True)
        return self.__page

    def redo(self):
        """Redo the command."""
        if not self.undone():
            return None
        for obj in self.obj:
            self.__set_prop_func(obj, self.__prop)
        self.undone_set(False)
        return self.__page

def pen_set_func(obj, prop):
    """Set the pen of the object."""
    obj.pen_set(prop)

def pen_get_func(obj):
    """Get the pen of the object."""
    return obj.pen

class SetPenCommand(SetPropCommand):
    """Simple class for handling color changes."""
    def __init__(self, objects, pen):
        set_prop_func = pen_set_func
        get_prop_func = pen_get_func
        pen = pen.copy()
        super().__init__("set_pen", objects, pen, get_prop_func, set_prop_func)

def color_set_func(obj, prop):
    """Set the color of the object."""
    obj.color_set(prop)

def color_get_func(obj):
    """Get the color of the object."""
    return obj.pen.color

class SetColorCommand(SetPropCommand):
    """Simple class for handling color changes."""
    def __init__(self, objects, color):
        set_prop_func = color_set_func
        get_prop_func = color_get_func
        super().__init__("set_color", objects, color, get_prop_func, set_prop_func)

def set_font_func(obj, prop):
    """Set the font of the object."""
    obj.pen.font_set(prop)

def get_font_func(obj):
    """Get the font of the object."""
    return obj.pen.font_get()

class SetFontCommand(SetPropCommand):
    """Simple class for handling font changes."""
    def __init__(self, objects, font):
        set_prop_func = set_font_func
        get_prop_func = get_font_func
        super().__init__("set_font", objects, font, get_prop_func, set_prop_func)
"""
This module defines the Pen class, which represents a pen with customizable drawing properties.
"""


class Pen:
    """
    Represents a pen with customizable drawing properties.

    This class encapsulates properties like color, line width, and font settings
    that can be applied to drawing operations on a canvas.

    Attributes:
        color (tuple): The RGB color of the pen as a tuple (r, g, b), with
                       each component ranging from 0 to 1.
        line_width (float): The width of the lines drawn by the pen.
        font_size (int): The size of the font when drawing text.
        fill_color (tuple or None): The RGB fill color for shapes. `None` means no fill.
        transparency (float): The transparency level of the pen's color,
                              where 1 is opaque and 0 is fully transparent.
        font_family (str): The name of the font family used for drawing text.
        font_weight (str): The weight of the font ('normal', 'bold', etc.).
        font_style (str): The style of the font ('normal', 'italic', etc.).

    Args:
        color (tuple, optional): Initial color of the pen. Defaults to (0, 0, 0) for black.
        line_width (int, optional): Initial line width. Defaults to 12.
        transparency (float, optional): Initial transparency level. Defaults to 1 (opaque).
        fill_color (tuple, optional): Initial fill color. Defaults to None.
        font_size (int, optional): Initial font size. Defaults to 12.
        font_family (str, optional): Initial font family. Defaults to "Sans".
        font_weight (str, optional): Initial font weight. Defaults to "normal".
        font_style (str, optional): Initial font style. Defaults to "normal".

    Example usage:
        >>> my_pen = Pen(color=(1, 0, 0), line_width=5, transparency=0.5)
        >>> print(my_pen.color)
        (1, 0, 0)

    Note:
        This class does not directly handle drawing operations. It is used to store
        and manage drawing properties that can be applied by a drawing context.
    """

    def __init__(self, color = (0, 0, 0), line_width = 12, transparency = 1,
                 fill_color = None,
                 font_size = 12, font_family = "Sans",
                 font_weight = "normal", font_style = "normal",
                 brush = "rounded"):
        """
        Initializes a new Pen object with the specified drawing properties.
        """
        self.color        = color
        self.line_width   = line_width
        self.fill_color   = fill_color
        self.transparency = transparency
        #self.font_family       = font_family or "Segoe Script"
        self.font_size         = font_size   or 12
        self.font_family       = font_family or "Sans"
        self.font_weight       = font_weight or "normal"
        self.font_style        = font_style  or "normal"
        self.font_description  = Pango.FontDescription.from_string(
                f"{self.font_family} {self.font_style} {self.font_weight} {self.font_size}")
        #self.__brush     = BrushFactory.create_brush(brush)
        self.__brush_type = brush

    def brush_type(self, brush_type = None):
        """Get or set the brush property"""
        if brush_type is not None:
            self.__brush_type = brush_type
       #     print("creating new self", self, "brush", brush_type)
       #     self.__brush = BrushFactory.create_brush(brush_type)
        return self.__brush_type

    def transparency_set(self, transparency):
        """Set pen transparency"""
        self.transparency = transparency

    def fill_set(self, color):
        """Set fill color"""
        self.fill_color = color

    def fill_get(self):
        """Get fill color"""
        return self.fill_color

    def color_set(self, color):
        """Set pen color"""
        self.color = color

    def color_get(self):
        """Get pen color"""
        return self.color

    def line_width_set(self, line_width):
        """Set pen line width"""
        self.line_width = line_width

    def font_get(self):
        """Get the font description"""
        if not self.font_description:
            self.font_description = Pango.FontDescription.from_string(
                    f"{self.font_family} {self.font_style} {self.font_weight} {self.font_size}")
        return self.font_description

    def font_set(self, font):
        """Set the font description"""
        if isinstance(font, str):
            self.font_description = Pango.FontDescription.from_string(font)
            self.font_set_from_description(font)
        elif isinstance(font, Pango.FontDescription):
            self.font_description = font
            self.font_set_from_description(font)
        elif isinstance(font, dict):
            self.font_set_from_dict(font)
        else:
            raise ValueError("font must be a string, a Pango.FontDescription, or a dict")

    def font_set_from_dict(self, font_dict):
        """Set font based on dictionary"""
        self.font_family = font_dict.get("family", "Sans")
        self.font_size   = font_dict.get("size", 12)
        self.font_weight = font_dict.get("weight", "normal")
        self.font_style  = font_dict.get("style", "normal")
        self.font_description = Pango.FontDescription.from_string(f"{self.font_family} {self.font_style} {self.font_weight} {self.font_size}")


    def font_set_from_description(self, font_description):
        """Set font based on a Pango.FontDescription"""
        print("setting font from", font_description)
        self.font_description = font_description
        self.font_family = font_description.get_family()
        self.font_size   = font_description.get_size() / Pango.SCALE
        self.font_weight = "bold"   if font_description.get_weight() == Pango.Weight.BOLD  else "normal"
        self.font_style  = "italic" if font_description.get_style()  == Pango.Style.ITALIC else "normal"

        print("setting font to",
              self.font_family, self.font_size, self.font_weight, self.font_style)

    def stroke_change(self, direction):
        """Change the line width of the pen"""
        # for thin lines, a fine tuned change of line width
        if self.line_width > 2:
            self.line_width += direction
        else:
            self.line_width += direction / 10
        self.line_width = max(0.1, self.line_width)

    def to_dict(self):
        """Convert pen properties to a dictionary"""
        return {
            "color": self.color,
            "line_width": self.line_width,
            "transparency": self.transparency,
            "fill_color": self.fill_color,
            "font_size": self.font_size,
            "font_family": self.font_family,
            "font_weight": self.font_weight,
            "brush": self.__brush_type,
            "font_style": self.font_style
        }

    def copy(self):
        """Create a copy of the pen"""
        return Pen(self.color, self.line_width, self.transparency, self.fill_color, 
                   self.font_size, self.font_family, self.font_weight, self.font_style,
                   brush = self.__brush_type)

    @classmethod
    def from_dict(cls, d):
        """Create a pen object from a dictionary"""
        #def __init__(self, color = (0, 0, 0), line_width = 12, font_size = 12, transparency = 1, fill_color = None, family = "Sans", weight = "normal", style = "normal"):
        return cls(d.get("color"), d.get("line_width"), d.get("transparency"), d.get("fill_color"),
                   d.get("font_size"), d.get("font_family"), d.get("font_weight"), d.get("font_style"), d.get("brush"))
"""
This module contains the MouseEvent class.
It is used to handle mouse events in the drawing area.
"""

## ---------------------------------------------------------------------
class MouseEvent:
    """
    Simple class for handling mouse events.

    Takes the event and computes a number of useful things.

    One advantage of using it: the computationaly intensive stuff is
    computed only once and only if it is needed.
    """
    def __init__(self, event, objects, translate = None, mode = None):
        self.event   = event
        self.objects = objects
        self.__mode = mode
        self.__modifiers = {
                "shift": (event.state & Gdk.ModifierType.SHIFT_MASK) != 0,
                "ctrl": (event.state & Gdk.ModifierType.CONTROL_MASK) != 0,
                "alt": (event.state & Gdk.ModifierType.MOD1_MASK) != 0,
                }
        self.__info = {
                "double":   (event.type == Gdk.EventType.DOUBLE_BUTTON_PRESS),
                "pressure": event.get_axis(Gdk.AxisUse.PRESSURE),
                "hover":    None,
                "corner":   None 
                }

        if self.__info["pressure"] is None:  # note that 0 is perfectly valid
            self.__info["pressure"] = 1

        self.x, self.y = event.x, event.y

        if translate:
            self.x, self.y = self.x - translate[0], self.y - translate[1]

        self.__pos    = (self.x, self.y)


    def hover(self):
        """Return the object that is hovered by the mouse."""
        if not self.__info.get("hover"):
            self.__info["hover"] = find_obj_close_to_click(self.__pos[0],
                                                   self.__pos[1],
                                                   self.objects, 20)
        return self.__info["hover"]

    def corner(self):
        """Return the corner that is hovered by the mouse."""
        if not self.__info.get("corner"):
            self.__info["corner"] = find_corners_next_to_click(self.__pos[0],
                                                       self.__pos[1],
                                                       self.objects, 20)
        return self.__info["corner"][0], self.__info["corner"][1]

    def pos(self):
        """Return the position of the mouse."""
        return self.__pos

    def shift(self):
        """Return True if the shift key is pressed."""
        return self.__modifiers.get("shift")

    def ctrl(self):
        """Return True if the control key is pressed."""
        return self.__modifiers.get("ctrl")

    def alt(self):
        """Return True if the alt key is pressed."""
        return self.__modifiers.get("alt")

    def double(self):
        """Return True if the event is a double click."""
        return self.__info.get("double")

    def pressure(self):
        """Return the pressure of the pen."""
        return self.__info.get("pressure")

    def mode(self):
        """Return the mode in which the event happened."""
        return self.__mode
"""
Dialogs for the ScreenDrawer application.
"""


## ---------------------------------------------------------------------
FORMATS = {
    "All files": { "pattern": "*",      "mime_type": "application/octet-stream", "name": "all" },
    "PNG files":  { "pattern": "*.png",  "mime_type": "image/png",       "name": "png" },
    "JPEG files": { "pattern": "*.jpeg", "mime_type": "image/jpeg",      "name": "jpeg" },
    "PDF files":  { "pattern": "*.pdf",  "mime_type": "application/pdf", "name": "pdf" }
}


## ---------------------------------------------------------------------
class help_dialog(Gtk.Dialog):
    """A dialog to show help information."""
    def __init__(self, parent):
        print("parent:", parent)
        super().__init__(title="Help", transient_for=parent, flags=0)
        self.add_buttons(Gtk.STOCK_OK, Gtk.ResponseType.OK)

        parent_size = parent.get_size()
        self.set_default_size(max(parent_size[0] * 0.8, 400), max(parent_size[1] * 0.9, 300))
        self.set_border_width(10)

        scrolled_window = Gtk.ScrolledWindow()
        scrolled_window.set_min_content_width(380)
        scrolled_window.set_min_content_height(280)
        scrolled_window.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scrolled_window.set_hexpand(True)  # Allow horizontal expansion
        scrolled_window.set_vexpand(True)  # Allow vertical expansion

        help_text = f"""

<span font="24"><b>screendrawer</b></span>

Draw on the screen with Gnome and Cairo. Quick and dirty.

<b>(Help not complete yet.)</b>

<span font_family="monospace">
<b>Mouse:</b>

<b>All modes:</b>                                 <b>Move mode:</b>
shift-click:  Enter text mode              click: Select object             Resizing: click in corner
                                           click and drag: Move object      Rotating: ctrl-shift-click in corner
ctrl-click:   Change line width            ctrl-a: Select all
ctrl-shift-click: Change transparency

Moving object to left lower screen corner deletes it.

<b>Shortcut keys:</b>

<b>Drawing modes:</b> (simple key, when not editing a text)

<b>d:</b> Draw mode (pencil)                 <b>m, |SPACE|:</b> Move mode (move objects around, copy and paste)
<b>t:</b> Text mode (text entry)             <b>r:</b> rectangle mode  (draw a rectangle)
<b>c:</b> Circle mode (draw an ellipse)      <b>e:</b> Eraser mode (delete objects with a click)
<b>s:</b> Shape mode (draw a filled shape)   <b>i:</b> Color p<b>I</b>cker mode (pick a color from the screen)

<b>Works always:</b>                                                             <b>Move mode only:</b>
<b>With Ctrl:</b>              <b>Simple key (not when entering text)</b>               <b>With Ctrl:</b>             <b>Simple key (not when entering text)</b>
Ctrl-q: Quit            x, q: Exit                                        Ctrl-c: Copy content   Tab: Next object
Ctrl-e: Export drawing  h, F1, ?: Show this help dialog                   Ctrl-v: Paste content  Shift-Tab: Previous object
Ctrl-l: Clear drawing   l: Clear drawing                                  Ctrl-x: Cut content    Shift-letter: quick color selection e.g.
                                                                                                 Shift-r for red
Ctrl-i: insert image                                                                             |Del|: Delete selected object(s)
Ctrl-z: undo            |Esc|: Finish text input                                                 g, u: group, ungroup
Ctrl-y: redo            |Enter|: New line (when typing)                   Alt-Up, Alt-Down: Move object up, down
                                                                          Alt-PgUp, Alt-PgDown: Move object to front, back
Ctrl-k: Select color                     f: fill with current color       Alt-s: convert drawing(s) to shape(s)
Ctrl-Shift-k: Select bg color
Ctrl-plus, Ctrl-minus: Change text size  o: toggle outline                Alt-d: convert shape(s) to drawing(s)
Ctrl-b: Cycle background transparency                                     Alt-p: apply pen to selection
Ctrl-p: toggle between two pens                                           Alt-Shift-p: apply pen color to background
Ctrl-g: toggle grid                      1-3: select brush

<b>Taking screenshots:</b>
Ctrl-Shift-f: screenshot: for a screenshot, if you have at least one rectangle
object (r mode) in the drawing, then it will serve as the selection area. The
screenshot will be pasted into the drawing. If there are no rectangles,
then the whole window will be captured.

<b>Saving / importing:</b>
Ctrl-i: Import image from a file (jpeg, png)
Ctrl-o: Open a drawing from a file (.sdrw, that is the "native format") -
        note that the subsequent modifications will be saved to that file only
Ctrl-e: Export selection or whole drawing to a file (png, jpeg, pdf)
Ctrl-Shift-s: "Save as" - save drawing to a file (.sdrw, that is the "native format") - note
        that the subsequent modifications will be saved to that file only

When you copy a selection or individual objects, you can paste them into
other programs as a PNG image.

</span>

The state is saved in / loaded from `{parent.savefile}` so you can continue drawing later.
An autosave happens every minute or so.
        """
        label = Gtk.Label()
        label.set_markup(help_text)
        label.set_justify(Gtk.Justification.LEFT)
        label.set_line_wrap(True)
        scrolled_window.add(label)

        box = self.get_content_area()
        box.pack_start(scrolled_window, True, True, 0)  # Use pack_start with expand and fill

        self.show_all()

## ---------------------------------------------------------------------

def _dialog_add_image_formats(dialog):
    for name, data in FORMATS.items():
        file_filter = Gtk.FileFilter()
        file_filter.set_name(name)
        file_filter.add_pattern(data["pattern"])
        file_filter.add_mime_type(data["mime_type"])
        dialog.add_filter(file_filter)

## ---------------------------------------------------------------------

def export_dialog(parent):
    """Show a file chooser dialog to select a file to save the drawing as
    an image / pdf / svg."""
    print("export_dialog")
    file_name, selected_filter = None, None

    dialog = Gtk.FileChooserDialog(
        title="Export As", parent=parent, action=Gtk.FileChooserAction.SAVE)

    dialog.add_buttons(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                       Gtk.STOCK_SAVE, Gtk.ResponseType.OK)
    dialog.set_modal(True)

    current_directory = os.getcwd()
    dialog.set_current_folder(current_directory)

    _dialog_add_image_formats(dialog)

    # Show the dialog
    response = dialog.run()
    if response == Gtk.ResponseType.OK:
        file_name = dialog.get_filename()
        selected_filter = dialog.get_filter().get_name()
        selected_filter = FORMATS[selected_filter]["name"]
        print(f"Save file as: {file_name}, Format: {selected_filter}")

    dialog.destroy()
    return file_name, selected_filter


def save_dialog(parent):
    """Show a file chooser dialog to set the savefile."""
    print("export_dialog")
    #file_name, selected_filter = None, None
    file_name = None

    dialog = Gtk.FileChooserDialog(
        title="Save As", parent=parent, action=Gtk.FileChooserAction.SAVE)

    dialog.add_buttons(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                       Gtk.STOCK_SAVE, Gtk.ResponseType.OK)
    dialog.set_modal(True)

    current_directory = os.getcwd()
    dialog.set_current_folder(current_directory)

    #_dialog_add_image_formats(dialog)

    # Show the dialog
    response = dialog.run()
    if response == Gtk.ResponseType.OK:
        file_name = dialog.get_filename()
       #selected_filter = dialog.get_filter().get_name()
       #selected_filter = formats[selected_filter]["name"]
       #print(f"Save file as: {file_name}, Format: {selected_filter}")

    dialog.destroy()
    return file_name


def import_image_dialog(parent):
    """Show a file chooser dialog to select an image file."""
    dialog = Gtk.FileChooserDialog(
        title="Select an Image",
        parent=parent,
        action=Gtk.FileChooserAction.OPEN,
        buttons=(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, Gtk.STOCK_OPEN, Gtk.ResponseType.OK)
    )
    dialog.set_modal(True)
    current_directory = os.getcwd()
    dialog.set_current_folder(current_directory)

    # Filter to only show image files
    file_filter = Gtk.FileFilter()
    file_filter.set_name("Image files")
    file_filter.add_mime_type("image/jpeg")
    file_filter.add_mime_type("image/png")
    file_filter.add_mime_type("image/tiff")
    dialog.add_filter(file_filter)

    # Show the dialog and wait for the user response
    response = dialog.run()

    pixbuf = None
    image_path = None
    if response == Gtk.ResponseType.OK:
        image_path = dialog.get_filename()
    elif response == Gtk.ResponseType.CANCEL:
        print("No image selected")

    # Clean up and destroy the dialog
    dialog.destroy()
    return image_path

def open_drawing_dialog(parent):
    """Show a file chooser dialog to select a .sdrw file."""
    dialog = Gtk.FileChooserDialog(
        title="Select an .sdrw file",
        action=Gtk.FileChooserAction.OPEN,
        parent=parent,
        buttons=(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, Gtk.STOCK_OPEN, Gtk.ResponseType.OK)
    )
    dialog.set_modal(True)
    current_directory = os.getcwd()
    dialog.set_current_folder(current_directory)

    # Show the dialog and wait for the user response
    response = dialog.run()

    image_path = None
    if response == Gtk.ResponseType.OK:
        image_path = dialog.get_filename()
    elif response == Gtk.ResponseType.CANCEL:
        print("No image selected")

    # Clean up and destroy the dialog
    dialog.destroy()
    return image_path

def FontChooser(pen, parent):

    # check that pen is an instance of Pen
    if not isinstance(pen, Pen):
        raise ValueError("Pen is not defined or not of class Pen")

    font_dialog = Gtk.FontChooserDialog(title="Select a Font", parent=parent)
    #font_dialog.set_preview_text("Zażółć gęślą jaźń")
    font_dialog.set_preview_text("Sphinx of black quartz, judge my vow.")

    # You can set the initial font for the dialog
    font_dialog.set_font(pen.font_family + " " +
                         pen.font_style + " " +
                         str(pen.font_weight) + " " +
                         str(pen.font_size))

    response = font_dialog.run()

    font_description = None

    if response == Gtk.ResponseType.OK:
        font_description = font_dialog.get_font_desc()

    font_dialog.destroy()
    return font_description


def ColorChooser(parent, title = "Select Pen Color"):
    """Select a color for drawing."""
    # Create a new color chooser dialog
    color_chooser = Gtk.ColorChooserDialog(title, parent = parent)

    # Show the dialog
    response = color_chooser.run()
    color = None

    # Check if the user clicked the OK button
    if response == Gtk.ResponseType.OK:
        color = color_chooser.get_rgba()
        #self.set_color((color.red, color.green, color.blue))

    # Don't forget to destroy the dialog
    color_chooser.destroy()
    return color



## ---------------------------------------------------------------------

class Clipboard:
    """
    Class to handle clipboard operations. Basically, it handles internally
    the clipboard within the app, but sets the gtk clipboard if necessary.

    Atrributes:
        clipboard_owner (bool): True if the app owns the clipboard.
        clipboard (unspecified): The clipboard content.

    """

    def __init__(self, gtk_clipboard=None):
        self._gtk_clipboard = gtk_clipboard or Gtk.Clipboard.get(Gdk.SELECTION_CLIPBOARD)
        self._gtk_clipboard.connect('owner-change', self.on_clipboard_owner_change)
        self.clipboard_owner = False
        self.clipboard = None

    def on_clipboard_owner_change(self, clipboard, event):
        """Handle clipboard owner change events."""

        print(f"Owner change ({clipboard}), removing internal clipboard")
        print("reason:", event.reason)
        if self.clipboard_owner:
            self.clipboard_owner = False
        else:
            self.clipboard = None
        return True

    def set_text(self, text):
        """
        Set text to clipboard and store it.

        Note:
            This is like external copy, so the internal clipboard is set to None.
        """
        self.clipboard_owner = False
        self.clipboard = None
        self._gtk_clipboard.set_text(text, -1)
        self._gtk_clipboard.store()

    def get_content(self):
        # internal paste
        if self.clipboard:
            print("Pasting content internally")
            return "internal", self.clipboard

        # external paste
        clipboard = self._gtk_clipboard
        clip_text = clipboard.wait_for_text()

        if clip_text:
            return "text", clip_text

        clip_img = clipboard.wait_for_image()
        if clip_img:
            return "image", clip_img
        return None, None

    def copy_content(self, selection):
        """
        Copy internal content: object or objects from current selection.

        Args:
            selection (Drawable): The selection to copy, either a group or
                                  a single object.
        """
        clipboard = self._gtk_clipboard

        if selection.length() == 1:
            sel = selection.objects[0]
        else:
            sel = selection

        if sel.type == "text":
            text = sel.to_string()
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
        self.clipboard = DrawableGroup(selection.objects[:])
        self.clipboard_owner = True


"""
CursorManager class to manage the cursor.

Basically, this class is responsible for creating and managing the cursor for the window.
It has methods to set the cursor to different modes.
"""


## ---------------------------------------------------------------------

class CursorManager:
    """
    Class to manage the cursor.

    Attributes:
        _window (Gtk.Window): The window to manage the cursor for.
        _cursors (dict):       A dictionary of premade cursors for different modes.
        _current_cursor (str): The name of the current cursor.
        _default_cursor (str): The name of the default cursor.
        _pos (tuple):          The current position of the cursor.

    """


    def __init__(self, window):
        self._window  = window
        self._cursors = None
        self._current_cursor = "default"
        self._default_cursor = "default"

        self._make_cursors(window)

        self.default("default")

        self._pos = (100, 100)

    def _make_cursors(self, window):
        """Create cursors for different modes."""

        icons = Icons()
        colorpicker = icons.get("colorpicker")

        self._cursors = {
            "hand":        Gdk.Cursor.new_from_name(window.get_display(), "hand1"),
            "move":        Gdk.Cursor.new_from_name(window.get_display(), "hand2"),
            "grabbing":    Gdk.Cursor.new_from_name(window.get_display(), "grabbing"),
            "moving":      Gdk.Cursor.new_from_name(window.get_display(), "grab"),
            "text":        Gdk.Cursor.new_from_name(window.get_display(), "text"),
            #"eraser":      Gdk.Cursor.new_from_name(window.get_display(), "not-allowed"),
            "eraser":      Gdk.Cursor.new_from_pixbuf(window.get_display(),
                                                      icons.get("eraser"), 2, 23),
            "pencil":      Gdk.Cursor.new_from_name(window.get_display(), "pencil"),
            "picker":      Gdk.Cursor.new_from_name(window.get_display(), "crosshair"),
            #"colorpicker": Gdk.Cursor.new_from_name(window.get_display(), "crosshair"),
            "colorpicker": Gdk.Cursor.new_from_pixbuf(window.get_display(), colorpicker, 1, 26),
            "shape":       Gdk.Cursor.new_from_name(window.get_display(), "pencil"),
            "draw":        Gdk.Cursor.new_from_name(window.get_display(), "pencil"),
            "crosshair":   Gdk.Cursor.new_from_name(window.get_display(), "crosshair"),
            "circle":      Gdk.Cursor.new_from_name(window.get_display(), "crosshair"),
            "rectangle":   Gdk.Cursor.new_from_name(window.get_display(), "crosshair"),
            "none":        Gdk.Cursor.new_from_name(window.get_display(), "none"),
            "upper_left":  Gdk.Cursor.new_from_name(window.get_display(), "nw-resize"),
            "upper_right": Gdk.Cursor.new_from_name(window.get_display(), "ne-resize"),
            "lower_left":  Gdk.Cursor.new_from_name(window.get_display(), "sw-resize"),
            "lower_right": Gdk.Cursor.new_from_name(window.get_display(), "se-resize"),
            "default":     Gdk.Cursor.new_from_name(window.get_display(), "pencil")
        }

    def pos(self):
        """Return the current position of the cursor."""
        return self._pos

    def update_pos(self, x, y):
        """Update the current position of the cursor."""
        self._pos = (x, y)

    def default(self, cursor_name):
        """Set the default cursor to the specified cursor."""
        if self._current_cursor == cursor_name:
            return
        print("setting default cursor to", cursor_name)
        self._default_cursor = cursor_name
        self._current_cursor = cursor_name

        self._window.get_window().set_cursor(self._cursors[cursor_name])

    def revert(self):
        """Revert to the default cursor."""
        if self._current_cursor == self._default_cursor:
            return
        self._window.get_window().set_cursor(self._cursors[self._default_cursor])
        self._current_cursor = self._default_cursor

    def set(self, cursor_name):
        """Change the cursor to the specified cursor."""
        if self._current_cursor == cursor_name:
            return
        #print("changing cursor to", cursor_name)
        self._window.get_window().set_cursor(self._cursors[cursor_name])
        self._current_cursor = cursor_name


## ---------------------------------------------------------------------

class GraphicsObjectManager:
    """
    Class to manage graphics objects.

    Attributes:
        _objects (list): The list of objects.
    """

    def __init__(self):

        # private attr
        self.__history    = []
        self.__redo_stack = []
        self.__page = None
        self.page_set(Page())

    def page_set(self, page):
        """Set the current page."""
        self.__page = page

    def next_page(self):
        """Go to the next page."""
        self.page_set(self.__page.next(create = True))

    def prev_page(self):
        """Go to the prev page."""
        self.page_set(self.__page.prev())

    def delete_page(self):
        """Delete the current page."""
        curpage, cmd = self.__page.delete()
        self.__history.append(cmd)
        self.page_set(curpage)

    def next_layer(self):
        """Go to the next layer."""
        print("creating a new layer")
        self.__page.next_layer()

    def prev_layer(self):
        """Go to the previous layer."""
        self.__page.prev_layer()

    def delete_layer(self):
        """Delete the current layer."""
        cmd = self.__page.delete_layer()
        self.__history.append(cmd)

    def page(self):
        """Return the current page."""
        return self.__page

    def set_page_number(self, n):
        """Choose page number n."""
        tot_n = self.number_of_pages()
        if n < 0 or n >= tot_n:
            return
        cur_n = self.current_page_number()

        if n == cur_n:
            return
        if n > cur_n:
            for _ in range(n - cur_n):
                self.next_page()
        else:
            for _ in range(cur_n - n):
                self.prev_page()


    def number_of_pages(self):
        """Return the total number of pages."""
        p = self.start_page()

        n = 1
        while p.next(create = False):
            n += 1
            p = p.next(create = False)
        return n

    def current_page_number(self):
        """Return the current page number."""
        p = self.__page
        n = 0
        while p.prev() != p:
            n += 1
            p = p.prev()
        return n

    def start_page(self):
        """Return the first page."""
        p = self.__page
        while p.prev() != p:
            p = p.prev()
        return p

    def objects(self):
        """Return the list of objects."""
        return self.__page.objects()

    def selection(self):
        """Return the selection object."""
        return self.__page.selection()

    def transmute(self, objects, mode):
        """
        Transmute the object to the given mode.

        This is a dangerous operation, because we are replacing the objects
        and we need to make sure that the old objects are removed from the
        list of objects, selections etc.

        Args:
            objects (list): The list of objects.
            mode (str): The mode to transmute to.
        """
        cmd = TransmuteCommand(objects=objects,
                               stack=self.__page.objects(),
                               new_type=mode,
                               selection_objects=self.__page.selection().objects,
                               page = self.__page)
        self.__history.append(cmd)

    def transmute_selection(self, mode):
        """
        Transmute the selected objects to the given mode.

        Args:
            mode ( str ): The mode to transmute to.
        """
        if self.__page.selection().is_empty():
            return
        self.transmute(self.__page.selection().objects, mode)

    def set_objects(self, objects):
        """Set the list of objects."""
        ## no undo
        print("GOM: setting n=", len(objects), "objects")
        self.__page.objects(objects)

    def set_pages(self, pages):
        """Set the content of pages."""
        self.__page = Page()
        self.__page.import_page(pages[0])
        for p in pages[1:]:
            self.__page = self.__page.next()
            self.__page.import_page(p)

    def add_object(self, obj):
        """Add an object to the list of objects."""
        if obj in self.__page.objects():
            print("object already in list")
            return
        self.__history.append(AddCommand(obj, self.__page.objects(), page=self.__page))

    def export_pages(self):
        """Export all pages."""
        # XXX
        # find the first page
        p = self.__page
        while p.prev() != p:
            p = p.prev()

        # create a list of pages for all pages
        pages = [ ]
        while p:
            pages.append(p.export())
            p = p.next(create = False)
        return pages

    def kill_object(self, obj):
        """Directly remove an object from the list of objects."""
        self.__page.kill_object(obj)

    def selected_objects(self):
        """Return the selected objects."""
        return self.__page.selection().objects

    def remove_selection(self):
        """Remove the selected objects from the list of objects."""
        if self.__page.selection().is_empty():
            return
        self.__history.append(RemoveCommand(self.__page.selection().objects,
                                            self.__page.objects(),
                                            page=self.__page))
        self.__page.selection().clear()

    def remove_objects(self, objects, clear_selection = False):
        """Remove an object from the list of objects."""
        self.__history.append(RemoveCommand(objects, self.__page.objects(), page=self.__page))
        if clear_selection:
            self.__page.selection().clear()

    def remove_all(self):
        """Clear the list of objects."""
        self.__history.append(self.__page.clear())

    def command_append(self, command_list):
        """Append a group of commands to the history."""
        ## append in reverse order!
        self.__history.append(CommandGroup(command_list[::-1]))

    def selection_group(self):
        """Group selected objects."""
        if self.__page.selection().n() < 2:
            return
        print("Grouping", self.__page.selection().n(), "objects")
        self.__history.append(GroupObjectCommand(self.__page.selection().objects,
                                                 self.__page.objects(),
                                                 selection_object=self.__page.selection(),
                                                 page=self.__page))

    def selection_ungroup(self):
        """Ungroup selected objects."""
        if self.__page.selection().is_empty():
            return
        self.__history.append(UngroupObjectCommand(self.__page.selection().objects,
                                                   self.__page.objects(),
                                                   selection_object=self.__page.selection(),
                                                   page=self.__page))

    def select_reverse(self):
        """Reverse the selection."""
        self.__page.selection().reverse()
        # XXX
        #self.__state.mode("move")

    def select_all(self):
        """Select all objects."""
        if not self.__page.objects():
            print("no objects found")
            return

        self.__page.selection().all()
        # XXX
        #self.__state.mode("move")

    def selection_delete(self):
        """Delete selected objects."""
        #self.__page.selection_delete()
        if self.__page.selection().objects:
            self.__history.append(RemoveCommand(self.__page.selection().objects,
                                                self.__page.objects(), page=self.__page))
            self.__page.selection().clear()

    def select_next_object(self):
        """Select the next object."""
        self.__page.selection().next()

    def select_previous_object(self):
        """Select the previous object."""
        self.__page.selection().previous()

    def selection_fill(self):
        """Toggle the fill of the selected objects."""
        for obj in self.__page.selection().objects:
            obj.fill_toggle()

    def selection_color_set(self, color):
        """Set the color of the selected objects."""
        if not self.__page.selection().is_empty():
            self.__history.append(SetColorCommand(self.__page.selection(), color))

    def selection_font_set(self, font_description):
        """Set the font of the selected objects."""
        self.__history.append(SetFontCommand(self.__page.selection(), font_description))

  # XXX! this is not implemented
    def selection_apply_pen(self):
        """Apply the pen to the selected objects."""
  #     if not self.__page.selection().is_empty():
  #         pen = self.__state.pen()
  #         self.__history.append(SetPenCommand(self.__page.selection(), pen))

    def redo(self):
        """Redo the last action."""
        print("Redo stack, size is", len(self.__redo_stack))
        if self.__redo_stack:
            command = self.__redo_stack.pop()
            page = command.redo()
            self.__history.append(command)

            # switch to the relevant page
            if page:
                self.page_set(page)

    def undo(self):
        """Undo the last action."""
        print("Undo, history size is", len(self.__history))
        if self.__history:
            command = self.__history.pop()
            page = command.undo()
            self.__redo_stack.append(command)

            # switch to the relevant page
            if page:
                self.page_set(page)

    def move_obj(self, obj, dx, dy):
        """Move the object by the given amount."""
        event_obj = MoveCommand(obj, (0, 0), page=self.__page)
        event_obj.event_update(dx, dy)
        self.__history.append(event_obj)

    def move_selection(self, dx, dy):
        """Move the selected objects by the given amount."""
        if self.__page.selection().is_empty():
            return
        self.move_obj(self.__page.selection().copy(), dx, dy)

    def rotate_obj(self, obj, angle):
        """Rotate the object by the given angle (degrees)."""
        print("rotating by", angle)
        event_obj = RotateCommand(obj, angle=math.radians(angle), page = self.__page)
        event_obj.event_finish()
        self.__history.append(event_obj)

    def rotate_selection(self, angle):
        """Rotate the selected objects by the given angle (degrees)."""
        if self.__page.selection().is_empty():
            return
        self.rotate_obj(self.__page.selection(), angle)

    def selection_zmove(self, operation):
        """move the selected objects long the z-axis."""
        if self.__page.selection().is_empty():
            return
        self.__history.append(ZStackCommand(self.__page.selection().objects,
                                            self.__page.objects(), operation, page=self.__page))


"""
This module provides functions to import and export drawings in various formats.
"""




def guess_file_format(filename):
    """Guess the file format from the file extension."""
    _, file_format = path.splitext(filename)
    file_format = file_format[1:]
    # lower case
    file_format = file_format.lower()
    # jpg -> jpeg
    if file_format == "jpg":
        file_format = "jpeg"
    # check
    if file_format not in [ "png", "jpeg", "pdf", "svg" ]:
        raise ValueError("Unrecognized file extension")
    return file_format

def convert_file(input_file, output_file, file_format = "all", border = None, page = 0):
    """Convert a drawing from the internal format to another."""
    config, objects, pages = read_file_as_sdrw(input_file)
    print("page = ", page)

    if pages:
        if len(pages) <= page:
            raise ValueError(f"Page number out of range (max. {len(pages) - 1})")
        print("read drawing from", input_file, "with", len(pages), "pages")
        p = Page()
        p.import_page(pages[page])
        objects = p.objects_all_layers()

    print("read drawing from", input_file, "with", len(objects), "objects")
    objects = DrawableGroup(objects)

    bbox = config.get("bbox", None) or objects.bbox()
    if border:
        bbox = objects.bbox()
        bbox = (bbox[0] - border, bbox[1] - border, bbox[2] + 2 * border, bbox[3] + 2 * border)

    bg           = config.get("bg_color", (1, 1, 1))
    transparency = config.get("transparent", 1.0)

    if file_format == "all":
        if output_file is None:
            raise ValueError("No output file format provided")
        file_format = guess_file_format(output_file)
    else:
        if output_file is None:
            # create a file name with the same name but different extension
            output_file = path.splitext(input_file)[0] + "." + file_format

    export_image(objects,
                 output_file, file_format,
                 bg = bg, bbox = bbox,
                 transparency = transparency)


def export_image(objects, filename,
                 file_format = "all",
                 bg = (1, 1, 1),
                 bbox = None,
                 transparency = 1.0):
    """Export the drawing to a file."""

    # if filename is None, we send the output to stdout
    if filename is None:
        print("export_image: no filename provided")
        return

    if file_format == "all":
        # get the format from the file name
        _, file_format = path.splitext(filename)
        file_format = file_format[1:]
        # lower case
        file_format = file_format.lower()
        # jpg -> jpeg
        if file_format == "jpg":
            file_format = "jpeg"
        # check
        if file_format not in [ "png", "jpeg", "pdf", "svg" ]:
            raise ValueError("Unrecognized file extension")
        print("export_image: guessing format from file name:", file_format)

    if bbox is None:
        bbox = objects.bbox()
    print("Bounding box:", bbox)
    # to integers
    width, height = int(bbox[2]), int(bbox[3])


    # Create a Cairo surface of the same size as the bbox
    if file_format == "png":
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    elif file_format == "svg":
        surface = cairo.SVGSurface(filename, width, height)
    elif file_format == "pdf":
        surface = cairo.PDFSurface(filename, width, height)
    else:
        raise ValueError("Invalid file format: " + file_format)

    cr = cairo.Context(surface)
    # translate to the top left corner of the bounding box
    cr.translate(-bbox[0], -bbox[1])

    cr.set_source_rgba(*bg, transparency)
    cr.paint()
    objects.draw(cr)

    # Save the surface to the file
    if file_format == "png":
        surface.write_to_png(filename)
    elif file_format in [ "svg", "pdf" ]:
        surface.finish()

def save_file_as_sdrw(filename, config, objects = None, pages = None):
    """Save the objects to a file in native format."""
    # objects are here for backwards compatibility only
    state = { 'config': config }
    if pages:
        state['pages']   = pages
    if objects:
        state['objects'] = objects
    try:
        with open(filename, 'wb') as f:
            #yaml.dump(state, f)
            pickle.dump(state, f)
        print("Saved drawing to", filename)
        return True
    except OSError as e:
        print(f"Error saving file due to a file I/O error: {e}")
        return False
    except pickle.PicklingError as e:
        print(f"Error saving file because an object could not be pickled: {e}")
        return False

def read_file_as_sdrw(filename):
    """Read the objects from a file in native format."""
    if not path.exists(filename):
        print("No saved drawing found at", filename)
        return None, None, None

    print("READING file as sdrw:", filename)

    config, objects, pages = None, None, None

    try:
        with open(filename, "rb") as file:
            state = pickle.load(file)
            if "objects" in state:
                print("found objects in savefile")
                objects = [ Drawable.from_dict(d) for d in state['objects'] ] or [ ]
            if "pages" in state:
                print("found pages in savefile")
                pages = state['pages']
                for p in pages:
                    # this is for compatibility; newer drawings are saved
                    # with a "layers" key which is then processed by the
                    # page import function - best if page takes care of it
                    if "objects" in p:
                        p['objects'] = [ Drawable.from_dict(d) for d in p['objects'] ] or [ ]
                            
            config = state['config']
    except OSError as e:
        print(f"Error saving file due to a file I/O error: {e}")
        return None, None, None
    except pickle.PicklingError as e:
        print(f"Error saving file because an object could not be pickled: {e}")
        return None, None, None
    return config, objects, pages
"""
EM stands for EventManager. The EventManager class is a singleton that
manages the events and actions of the app. The actions are defined in the
make_actions_dictionary method.

The design of the app is as follows: the EventManager class is a singleton
that manages the events and actions of the app. The actions are defined in
the actions_dictionary method. 

So the EM is a know-it-all class, and the others (GOM, App) are just
listeners to the EM. The EM is the one that knows what to do when an event
happens.
"""


COLORS = {
        "black": (0, 0, 0),
        "white": (1, 1, 1),
        "red": (.7, 0, 0),
        "green": (0, .7, 0),
        "blue": (0, 0, .5),
        "yellow": (1, 1, 0),
        "cyan": (0, 1, 1),
        "magenta": (1, 0, 1),
        "purple": (0.5, 0, 0.5),
        "grey": (0.5, 0.5, 0.5)
}


class EventManager:
    """
    The EventManager class is a singleton that manages the events and actions
    of the app. The actions are defined in the make_actions_dictionary method.
    """
    # singleton pattern
    _instance = None

    def __new__(cls, *args, **kwargs):
        # singleton pattern
        if not cls._instance:
            cls._instance = super(EventManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, gom, app, state, setter):
        # singleton pattern
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self.__state = state
            self.__setter = setter
            self.make_actions_dictionary(gom, app, state, setter)
            self.make_default_keybindings()

    def dispatch_action(self, action_name, **kwargs):
        """
        Dispatches an action by name.
        """
        #print("dispatch_action", action_name)
        if not action_name in self.__actions:
            print(f"action {action_name} not found in actions")
            return

        action = self.__actions[action_name]['action']
        if not callable(action):
            raise ValueError(f"Action is not callable: {action_name}")
        try:
            if 'args' in self.__actions[action_name]:
                args = self.__actions[action_name]['args']
                action(*args)
            else:
                action(**kwargs)
        except Exception as e:
            exc_type, exc_value, exc_traceback = exc_info()
            print("Exception type: ", exc_type)
            print("Exception value:", exc_value)
            print("Traceback:")
            traceback.print_tb(exc_traceback)
            print(f"Error while dispatching action {action_name}: {e}")

    def dispatch_key_event(self, key_event, mode):
        """
        Dispatches an action by key event.
        """
        #print("dispatch_key_event", key_event, mode)

        if not key_event in self.__keybindings:
            print(f"key_event {key_event} not found in keybindings")
            return

        action_name = self.__keybindings[key_event]

        if not action_name in self.__actions:
            print(f"action {action_name} not found in actions")
            return

        # check whether action allowed in the current mode
        if 'modes' in self.__actions[action_name]:
            if not mode in self.__actions[action_name]['modes']:
                print("action not allowed in this mode")
                return

        print("keyevent", key_event, "dispatching action", action_name)
        self.dispatch_action(action_name)

    def on_key_press(self, widget, event):
        """
        This method is called when a key is pressed.
        """

        state = self.__state

        keyname = Gdk.keyval_name(event.keyval)
        char    = chr(Gdk.keyval_to_unicode(event.keyval))
        ctrl    = event.state & Gdk.ModifierType.CONTROL_MASK
        shift   = event.state & Gdk.ModifierType.SHIFT_MASK
        alt_l   = event.state & Gdk.ModifierType.MOD1_MASK
        print("keyname", keyname, "char", char, "ctrl", ctrl, "shift", shift, "alt_l", alt_l)

        mode = state.mode()

        keyfull = keyname

        if char.isupper():
            keyfull = keyname.lower()
        if shift:
            keyfull = "Shift-" + keyfull
        if ctrl:
            keyfull = "Ctrl-" + keyfull
        if alt_l:
            keyfull = "Alt-" + keyfull
        #print("keyfull", keyfull)

        # first, check whether there is a current object being worked on
        # and whether this object is a text object. In that case, we only
        # call the ctrl keybindings and pass the rest to the text object.
        cobj = state.current_obj()

        if cobj and cobj.type == "text" and not(ctrl or keyname == "Escape"):
            print("updating text input")
            cobj.update_by_key(keyname, char)
            state.queue_draw()
            return

        # otherwise, we dispatch the key event
        self.dispatch_key_event(keyfull, mode)

        # XXX this probably should be somewhere else
        state.queue_draw()


    def make_actions_dictionary(self, gom, app, state, setter):
        """
        This dictionary maps key events to actions.
        """
        self.__actions = {
            'mode_draw':             {'action': state.mode, 'args': ["draw"]},
            'mode_rectangle':              {'action': state.mode, 'args': ["rectangle"]},
            'mode_circle':           {'action': state.mode, 'args': ["circle"]},
            'mode_move':             {'action': state.mode, 'args': ["move"]},
            'mode_text':             {'action': state.mode, 'args': ["text"]},
            'mode_select':           {'action': state.mode, 'args': ["select"]},
            'mode_eraser':           {'action': state.mode, 'args': ["eraser"]},
            'mode_shape':            {'action': state.mode, 'args': ["shape"]},
            'mode_colorpicker':      {'action': state.mode, 'args': ["colorpicker"]},

            'finish_text_input':     {'action': setter.finish_text_input},

            'cycle_bg_transparency': {'action': state.cycle_background},
            'toggle_outline':        {'action': state.outline_toggle},

            'clear_page':            {'action': setter.clear},
            'toggle_wiglets':        {'action': state.toggle_wiglets},
            'toggle_grid':           {'action': state.toggle_grid},

            'show_help_dialog':      {'action': app.show_help_dialog},
            'app_exit':              {'action': app.exit},

            'selection_fill':        {'action': gom.selection_fill},
            'transmute_to_shape':    {'action': gom.transmute_selection, 'args': [ "shape" ]},
            'transmute_to_draw':     {'action': gom.transmute_selection, 'args': [ "draw" ]},
            'move_up_10':            {'action': gom.move_selection, 'args': [0, -10],   'modes': ["move"]},
            'move_up_1':             {'action': gom.move_selection, 'args': [0, -1],    'modes': ["move"]},
            'move_up_100':           {'action': gom.move_selection, 'args': [0, -100],  'modes': ["move"]},
            'move_down_10':          {'action': gom.move_selection, 'args': [0, 10],    'modes': ["move"]},
            'move_down_1':           {'action': gom.move_selection, 'args': [0, 1],     'modes': ["move"]},
            'move_down_100':         {'action': gom.move_selection, 'args': [0, 100],   'modes': ["move"]},
            'move_left_10':          {'action': gom.move_selection, 'args': [-10, 0],   'modes': ["move"]},
            'move_left_1':           {'action': gom.move_selection, 'args': [-1, 0],    'modes': ["move"]},
            'move_left_100':         {'action': gom.move_selection, 'args': [-100, 0],  'modes': ["move"]},
            'move_right_10':         {'action': gom.move_selection, 'args': [10, 0],    'modes': ["move"]},
            'move_right_1':          {'action': gom.move_selection, 'args': [1, 0],     'modes': ["move"]},
            'move_right_100':        {'action': gom.move_selection, 'args': [100, 0],   'modes': ["move"]},

            # XXX something is rotten here
            #'f':                    {'action': self.gom.selection_fill, 'modes': ["box", "circle", "draw", "move"]},

            'rotate_selection_ccw_10': {'action': gom.rotate_selection, 'args': [10],  'modes': ["move"]},
            'rotate_selection_ccw_1':  {'action': gom.rotate_selection, 'args': [1],   'modes': ["move"]},
            'rotate_selection_ccw_90': {'action': gom.rotate_selection, 'args': [90],  'modes': ["move"]},
            'rotate_selection_cw_10':  {'action': gom.rotate_selection, 'args': [-10], 'modes': ["move"]},
            'rotate_selection_cw_1':   {'action': gom.rotate_selection, 'args': [-1],  'modes': ["move"]},
            'rotate_selection_cw_90':  {'action': gom.rotate_selection, 'args': [-90], 'modes': ["move"]},

            'zmove_selection_top':    {'action': gom.selection_zmove, 'args': [ "top" ],    'modes': ["move"]},
            'zmove_selection_bottom': {'action': gom.selection_zmove, 'args': [ "bottom" ], 'modes': ["move"]},
            'zmove_selection_raise':  {'action': gom.selection_zmove, 'args': [ "raise" ],  'modes': ["move"]},
            'zmove_selection_lower':  {'action': gom.selection_zmove, 'args': [ "lower" ],  'modes': ["move"]},

            'set_color_white':       {'action': setter.set_color, 'args': [COLORS["white"]]},
            'set_color_black':       {'action': setter.set_color, 'args': [COLORS["black"]]},
            'set_color_red':         {'action': setter.set_color, 'args': [COLORS["red"]]},
            'set_color_green':       {'action': setter.set_color, 'args': [COLORS["green"]]},
            'set_color_blue':        {'action': setter.set_color, 'args': [COLORS["blue"]]},
            'set_color_yellow':      {'action': setter.set_color, 'args': [COLORS["yellow"]]},
            'set_color_cyan':        {'action': setter.set_color, 'args': [COLORS["cyan"]]},
            'set_color_magenta':     {'action': setter.set_color, 'args': [COLORS["magenta"]]},
            'set_color_purple':      {'action': setter.set_color, 'args': [COLORS["purple"]]},
            'set_color_grey':        {'action': setter.set_color, 'args': [COLORS["grey"]]},

            'set_brush_rounded':     {'action': setter.set_brush, 'args': ["rounded"] },
            'set_brush_marker':      {'action': setter.set_brush, 'args': ["marker"] },
            'set_brush_slanted':     {'action': setter.set_brush, 'args': ["slanted"] },
            'set_brush_pencil':      {'action': setter.set_brush, 'args': ["pencil"] },

            'apply_pen_to_bg':       {'action': state.apply_pen_to_bg,        'modes': ["move"]},
            'toggle_pens':           {'action': state.switch_pens},

            # dialogs
            "export_drawing":        {'action': app.export_drawing},
            "save_drawing_as":       {'action': app.save_drawing_as},
            "select_color":          {'action': app.select_color},
            "select_color_bg":       {'action': app.select_color_bg},
            "select_font":           {'action': app.select_font},
            "import_image":          {'action': app.select_image_and_create_pixbuf},
            "open_drawing":          {'action': app.open_drawing},

            # selections and moving objects
            'select_next_object':     {'action': gom.select_next_object,     'modes': ["move"]},
            'select_previous_object': {'action': gom.select_previous_object, 'modes': ["move"]},
            'select_all':             {'action': gom.select_all},
            'select_reverse':         {'action': gom.select_reverse},
            'selection_group':        {'action': gom.selection_group,   'modes': ["move"]},
            'selection_ungroup':      {'action': gom.selection_ungroup, 'modes': ["move"]},
            'selection_delete':       {'action': gom.selection_delete,  'modes': ["move"]},
            'redo':                   {'action': gom.redo},
            'undo':                   {'action': gom.undo},

            'next_page':              {'action': gom.next_page},
            'prev_page':              {'action': gom.prev_page},
            'delete_page':            {'action': gom.delete_page},

            'next_layer':             {'action': gom.next_layer},
            'prev_layer':             {'action': gom.prev_layer},
            'delete_layer':           {'action': gom.delete_layer},

            'apply_pen_to_selection': {'action': gom.selection_apply_pen,    'modes': ["move"]},

            'copy_content':           {'action': app.copy_content,        },
            'cut_content':            {'action': app.cut_content,         'modes': ["move"]},
            'paste_content':          {'action': app.paste_content},
            'screenshot':             {'action': app.screenshot},

            'stroke_increase':        {'action': setter.stroke_change, 'args': [1]},
            'stroke_decrease':        {'action': setter.stroke_change, 'args': [-1]},
        }

    def make_default_keybindings(self):
        """
        This dictionary maps key events to actions.
        """
        self.__keybindings = {
            'm':                    "mode_move",
            'r':                    "mode_rectangle",
            'c':                    "mode_circle",
            'd':                    "mode_draw",
            't':                    "mode_text",
            'e':                    "mode_eraser",
            's':                    "mode_shape",
            'i':                    "mode_colorpicker",
            'space':                "mode_move",

            'h':                    "show_help_dialog",
            'F1':                   "show_help_dialog",
            'question':             "show_help_dialog",
            'Shift-question':       "show_help_dialog",
            'Escape':               "finish_text_input",
            'Ctrl-l':               "clear_page",
            'Ctrl-b':               "cycle_bg_transparency",
            'x':                    "app_exit",
            'q':                    "app_exit",
            'Ctrl-q':               "app_exit",
            'l':                    "clear_page",
            'o':                    "toggle_outline",
            'w':                    "toggle_wiglets",
            'Ctrl-g':              "toggle_grid",
            'Alt-s':                "transmute_to_shape",
            'Alt-d':                "transmute_to_draw",
            'f':                    "selection_fill",

            'Up':                   "move_up_10",
            'Shift-Up':             "move_up_1",
            'Ctrl-Up':              "move_up_100",
            'Down':                 "move_down_10",
            'Shift-Down':           "move_down_1",
            'Ctrl-Down':            "move_down_100",
            'Left':                 "move_left_10",
            'Shift-Left':           "move_left_1",
            'Ctrl-Left':            "move_left_100",
            'Right':                "move_right_10",
            'Shift-Right':          "move_right_1",
            'Ctrl-Right':           "move_right_100",
            'Page_Up':              "rotate_selection_ccw_10",
            'Shift-Page_Up':        "rotate_selection_ccw_1",
            'Ctrl-Page_Up':         "rotate_selection_ccw_90",
            'Page_Down':            "rotate_selection_cw_10",
            'Shift-Page_Down':      "rotate_selection_cw_1",
            'Ctrl-Page_Down':       "rotate_selection_cw_90",

            'Alt-Page_Up':          "zmove_selection_top",
            'Alt-Page_Down':        "zmove_selection_bottom",
            'Alt-Up':               "zmove_selection_raise",
            'Alt-Down':             "zmove_selection_lower",

            'Shift-w':              "set_color_white",
            'Shift-b':              "set_color_black",
            'Shift-r':              "set_color_red",
            'Shift-g':              "set_color_green",
            'Shift-l':              "set_color_blue",
            'Shift-e':              "set_color_grey",
            'Shift-y':              "set_color_yellow",
#            'Shift-p':              "set_color_purple",

            '1':                    "set_brush_rounded",
            '2':                    "set_brush_marker",
            '3':                    "set_brush_slanted",
            '4':                    "set_brush_pencil",

            'Shift-p':              "prev_page",
            'Shift-n':              "next_page",
            'Shift-d':              "delete_page",
            'Ctrl-Shift-p':         "prev_layer",
            'Ctrl-Shift-n':         "next_layer",
            'Ctrl-Shift-d':         "delete_layer",

            'Ctrl-e':               "export_drawing",
            'Ctrl-Shift-s':         "save_drawing_as",
            'Ctrl-k':               "select_color",
            'Ctrl-Shift-k':         "select_color_bg",
            'Ctrl-f':               "select_font",
            'Ctrl-i':               "import_image",
            'Ctrl-p':               "toggle_pens",
            'Ctrl-o':               "open_drawing",


            'Tab':                  "select_next_object",
            'Shift-ISO_Left_Tab':   "select_previous_object",
            'g':                    "selection_group",
            'u':                    "selection_ungroup",
            'Delete':               "selection_delete",

            'Alt-p':                "apply_pen_to_selection",
            'Alt-Shift-p':          "apply_pen_to_bg",

            'Ctrl-a':               "select_all",
            'Ctrl-r':               "select_reverse",
            'Ctrl-y':               "redo",
            'Ctrl-z':               "undo",
            'Ctrl-c':               "copy_content",
            'Ctrl-x':               "cut_content",
            'Ctrl-v':               "paste_content",
            'Ctrl-Shift-f':         "screenshot",
            'Ctrl-plus':            "stroke_increase",
            'Ctrl-minus':           "stroke_decrease",
        }
"""
This module holds the MenuMaker class, which is a singleton class that creates menus.
"""


class MenuMaker:
    """A class holding methods to create menus. Singleton."""
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(MenuMaker, cls).__new__(cls)
        return cls._instance

    def __init__(self, em, app):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self.__app = app # App
            self.__em  = em  # EventManager
            self.__context_menu = None
            self.__object_menu = None

    def build_menu(self, menu_items):
        """Build a menu from a list of menu items."""
        menu = Gtk.Menu()
        menu.set_name("myMenu")

        for m in menu_items:
            if "separator" in m:
                menu_item = Gtk.SeparatorMenuItem()
            else:
                menu_item = Gtk.MenuItem(label=m["label"])
                menu_item.connect("activate", m["callback"], m["action"])
            menu.append(menu_item)
        menu.show_all()
        return menu

    def on_menu_item_activated(self, widget, action):
        """Callback for when a menu item is activated."""
        print("Menu item activated:", action, "from", widget)

        self.__em.dispatch_action(action)
        self.__app.queue_draw()


    def context_menu(self):
        """Build the main context menu"""
        # general menu for everything
        ## build only once
        if self.__context_menu:
            return self.__context_menu

        menu_items = [
                { "label": "Move         [m]",          "callback": self.on_menu_item_activated, "action": "mode_move" },
                { "label": "Pencil       [d]",          "callback": self.on_menu_item_activated, "action": "mode_draw" },
                { "label": "Shape        [s]",          "callback": self.on_menu_item_activated, "action": "mode_shape" },
                { "label": "Text         [t]",          "callback": self.on_menu_item_activated, "action": "mode_text" },
                { "label": "Rectangle    [r]",          "callback": self.on_menu_item_activated, "action": "mode_box" },
                { "label": "Circle       [c]",          "callback": self.on_menu_item_activated, "action": "mode_circle" },
                { "label": "Eraser       [e]",          "callback": self.on_menu_item_activated, "action": "mode_eraser" },
                { "label": "Color picker [i]",          "callback": self.on_menu_item_activated, "action": "mode_colorpicker" },
                { "separator": True },
                { "label": "Select all    (Ctrl-a)",    "callback": self.on_menu_item_activated, "action": "select_all" },
                { "label": "Paste         (Ctrl-v)",    "callback": self.on_menu_item_activated, "action": "paste_content" },
                { "label": "Clear drawing (Ctrl-l)",    "callback": self.on_menu_item_activated, "action": "clear_page" },
                { "separator": True },
                { "label": "Next Page     (Shift-n)",    "callback": self.on_menu_item_activated, "action": "next_page" },
                { "label": "Prev Page     (Shift-p)",    "callback": self.on_menu_item_activated, "action": "prev_page" },
                { "label": "Delete Page   (Shift-d)",    "callback": self.on_menu_item_activated, "action": "delete_page" },
                { "label": "Next Layer    (Ctrl-Shift-n)",    "callback": self.on_menu_item_activated, "action": "next_layer" },
                { "label": "Prev Layer    (Ctrl-Shift-p)",    "callback": self.on_menu_item_activated, "action": "prev_layer" },
                { "label": "Delete Layer  (Ctrl-Shift-d)",    "callback": self.on_menu_item_activated, "action": "delete_layer" },
                { "separator": True },
                { "label": "Bg transparency (Ctrl-b)",  "callback": self.on_menu_item_activated, "action": "cycle_bg_transparency" },
                { "label": "Tggl outline         [o]",  "callback": self.on_menu_item_activated, "action": "toggle_outline" },
                { "separator": True },
                { "label": "Color           (Ctrl-k)",  "callback": self.on_menu_item_activated, "action": "select_color" },
                { "label": "Bg Color        (Ctrl-k)",  "callback": self.on_menu_item_activated, "action": "select_color_bg" },
                { "label": "Font            (Ctrl-f)",  "callback": self.on_menu_item_activated, "action": "select_font" },
                { "separator": True },
                { "label": "Open drawing    (Ctrl-o)",  "callback": self.on_menu_item_activated, "action": "open_drawing" },
                { "label": "Image from file (Ctrl-i)",  "callback": self.on_menu_item_activated, "action": "import_image" },
                { "label": "Screenshot      (Ctrl-Shift-f)",  "callback": self.on_menu_item_activated, "action": "screenshot" },
                { "label": "Save as         (Ctrl-s)",  "callback": self.on_menu_item_activated, "action": "save_drawing_as" },
                { "label": "Export          (Ctrl-e)",  "callback": self.on_menu_item_activated, "action": "export_drawing" },
                { "label": "Help            [F1]",      "callback": self.on_menu_item_activated, "action": "show_help_dialog" },
                { "label": "Quit            (Ctrl-q)",  "callback": self.on_menu_item_activated, "action": "app_exit" },
        ]

        self.__context_menu = self.build_menu(menu_items)
        return self.__context_menu

    def object_menu(self, objects):
        """Build the object context menu"""
        # when right-clicking on an object
        menu_items = [
                { "label": "Copy (Ctrl-c)",        "callback": self.on_menu_item_activated, "action": "copy_content" },
                { "label": "Cut (Ctrl-x)",         "callback": self.on_menu_item_activated, "action": "cut_content" },
                { "separator": True },
                { "label": "Delete (|Del|)",       "callback": self.on_menu_item_activated, "action": "selection_delete" },
                { "label": "Group (g)",            "callback": self.on_menu_item_activated, "action": "selection_group" },
                { "label": "Ungroup (u)",          "callback": self.on_menu_item_activated, "action": "selection_ungroup" },
                { "label": "Export (Ctrl-e)",      "callback": self.on_menu_item_activated, "action": "export_drawing" },
                { "separator": True },
                { "label": "Move to top (Alt-Page_Up)", "callback": self.on_menu_item_activated, "action": "zmove_selection_top" },
                { "label": "Raise (Alt-Up)",       "callback": self.on_menu_item_activated, "action": "zmove_selection_up" },
                { "label": "Lower (Alt-Down)",     "callback": self.on_menu_item_activated, "action": "zmove_selection_down" },
                { "label": "Move to bottom (Alt-Page_Down)", "callback": self.on_menu_item_activated, "action": "zmove_selection_bottom" },
                { "separator": True },
                { "label": "To shape   (Alt-s)",   "callback": self.on_menu_item_activated, "action": "transmute_to_shape" },
                { "label": "To drawing (Alt-d)",   "callback": self.on_menu_item_activated, "action": "transmute_to_drawing" },
                #{ "label": "Fill       (f)",       "callback": self.on_menu_item_activated, "action": "f" },
                { "separator": True },
                { "label": "Color (Ctrl-k)",       "callback": self.on_menu_item_activated, "action": "select_color" },
                { "label": "Font (Ctrl-f)",        "callback": self.on_menu_item_activated, "action": "select_font" },
                { "label": "Help [F1]",            "callback": self.on_menu_item_activated, "action": "show_help_dialog" },
                { "label": "Quit (Ctrl-q)",        "callback": self.on_menu_item_activated, "action": "app_exit" },
        ]

        # if there is only one object, remove the group menu item
        if len(objects) == 1:
            print("only one object")
            menu_items = [m for m in menu_items if "action" not in m or "selection_group" not in m["action"]]

        group_found = [o for o in objects if o.type == "group"]
        if not group_found:
            print("no group found")
            menu_items = [m for m in menu_items if "action" not in m or "selection_ungroup" not in m["action"]]

        self.__object_menu = self.build_menu(menu_items)

        return self.__object_menu
"""
Wiglets are small dialog-like objects that can be drawn on the canvas.
They are used to provide interactive controls for changing drawing properties
such as line width, color, and transparency.
"""

def draw_rhomb(cr, bbox, fg = (0, 0, 0), bg = (1, 1, 1)):
    """
    Draw a rhombus shape
    """
    x0, y0, w, h = bbox
    cr.set_source_rgb(*bg)
    cr.move_to(x0, y0 + h/2)
    cr.line_to(x0 + w/2, y0)
    cr.line_to(x0 + w, y0 + h/2)
    cr.line_to(x0 + w/2, y0 + h)
    cr.close_path()
    cr.fill()

    cr.set_source_rgb(*fg)
    cr.set_line_width(1)
    cr.move_to(x0, y0 + h/2)
    cr.line_to(x0 + w/2, y0)
    cr.line_to(x0 + w, y0 + h/2)
    cr.line_to(x0 + w/2, y0 + h)
    cr.close_path()
    cr.stroke()


def adjust_color_brightness(rgb, factor):
    """
    Adjust the color brightness.
    :param rgb: A tuple of (r, g, b) in the range [0, 1]
    :param factor: Factor by which to adjust the brightness (>1 to make lighter, <1 to make darker)
    :return: Adjusted color as an (r, g, b) tuple in the range [0, 1]
    """
    # Convert RGB to HLS
    h, l, s = colorsys.rgb_to_hls(*rgb)
    #print("r,g,b:", *rgb, "h, l, s:", h, l, s)

    # Adjust lightness
    l = max(min(l * factor, 1), 0)  # Ensure lightness stays within [0, 1]
    newrgb = colorsys.hls_to_rgb(h, l, s)

    # Convert back to RGB
    return newrgb


## ---------------------------------------------------------------------
class Wiglet:
    """drawable dialog-like objects on the canvas"""
    def __init__(self, mytype, coords):
        self.wiglet_type   = mytype
        self.coords = coords

    def update_size(self, width, height):
        """update the size of the widget"""
        raise NotImplementedError("update size method not implemented")

    def draw(self, cr):
        """draw the widget"""
        raise NotImplementedError("draw method not implemented")

    def event_update(self, x, y):
        """update on mouse move"""
        raise NotImplementedError("event_update method not implemented")

    def event_finish(self):
        """update on mouse release"""
        raise NotImplementedError("event_finish method not implemented")

class WigletColorPicker(Wiglet):
    """Invisible wiglet that processes clicks in the color picker mode."""
    def __init__(self, func_color, clipboard):
        super().__init__("colorpicker", None)

        self.__func_color = func_color
        self.__clipboard = clipboard
                 

    def draw(self, cr):
        """ignore drawing the widget"""

    def update_size(self, width, height):
        """ignore resizing"""

    def event_update(self, x, y):
        """ignore on mouse move"""

    def on_click(self, x, y, ev):
        """handle the click event"""

        # only works in color picker mode
        if ev.mode() != "colorpicker":
            return False

        if ev.shift() or ev.alt() or ev.ctrl():
            return

        color = get_color_under_cursor()
        self.__func_color(color)

        color_hex = rgb_to_hex(color)
        self.__clipboard.set_text(color_hex)
        #self.__state.queue_draw()
        return True


class WigletTransparency(Wiglet):
    """Wiglet for changing the transparency."""
    def __init__(self, coords, pen):
        super().__init__("transparency", coords)

        if not pen or not isinstance(pen, Pen):
            raise ValueError("Pen is not defined or not of class Pen")

        self.__pen      = pen
        self.__initial_transparency = pen.transparency
        print("initial transparency:", self.__initial_transparency)

    def draw(self, cr):
        """draw the widget"""
        cr.set_source_rgba(*self.__pen.color, self.__pen.transparency)
        draw_dot(cr, *self.coords, 50)

    def event_update(self, x, y):
        """update on mouse move"""
        dx = x - self.coords[0]
        ## we want to change the transparency by 0.1 for every 20 pixels
        self.__pen.transparency = max(0, min(1, self.__initial_transparency + dx/500))

    def update_size(self, width, height):
        """ignoring the update the size of the widget"""

    def event_finish(self):
        """update on mouse release"""

class WigletLineWidth(Wiglet):
    """
    Wiglet for changing the line width.
    directly operates on the pen of the object
    """
    def __init__(self, coords, pen):
        super().__init__("line_width", coords)

        if not pen or not isinstance(pen, Pen):
            raise ValueError("Pen is not defined or not of class Pen")
        self.__pen      = pen
        self.__initial_width = pen.line_width

    def draw(self, cr):
        cr.set_source_rgb(*self.__pen.color)
        draw_dot(cr, *self.coords, self.__pen.line_width)

    def event_update(self, x, y):
        dx = x - self.coords[0]
        print("changing line width", dx)
        self.__pen.line_width = max(1, min(60, self.__initial_width + dx/20))
        return True

    def update_size(self, width, height):
        """ignoring the update the size of the widget"""

    def event_finish(self):
        """ignoring the update on mouse release"""

## ---------------------------------------------------------------------
class WigletPageSelector(Wiglet):
    """Wiglet for selecting the page."""
    def __init__(self, coords = (500, 0), gom = None,
                 width = 20, height = 35, screen_wh_func = None,
                 set_page_func = None):
        if not gom:
            raise ValueError("GOM is not defined")
        super().__init__("page_selector", coords)

        self.__width, self.__height = width, height
        self.__height_per_page = height
        self.__gom = gom
        self.__screen_wh_func = screen_wh_func
        self.__set_page_func  = set_page_func

        # we need to recalculate often because the pages might have been
        # changing a lot
        self.recalculate()

    def recalculate(self):
        """recalculate the position of the widget"""
        if self.__screen_wh_func and callable(self.__screen_wh_func):
            w, _ = self.__screen_wh_func()
            self.coords = (w - self.__width, 0)
        self.__page_n = self.__gom.number_of_pages()
        self.__height = self.__height_per_page * self.__page_n
        self.__bbox = (self.coords[0], self.coords[1], self.__width, self.__height)
        self.__current_page_n = self.__gom.current_page_number()

    def on_click(self, x, y, ev):
        """handle the click event"""
        self.recalculate()
        if not is_click_in_bbox(x, y, self.__bbox):
            return False

        # which page is at this position?
        print("clicked inside the bbox, event", ev)
        dy = y - self.__bbox[1]

        page_no = int(dy / self.__height_per_page)
        print("selected page:", page_no)

        page_in_range = 0 <= page_no < self.__page_n
        if page_in_range and self.__set_page_func and callable(self.__set_page_func):
            print("setting page to", page_no)
            self.__set_page_func(page_no)

        return True

    def draw(self, cr):
        """draw the widget"""
        self.recalculate()

        wpos = self.__bbox[0]
        hpos = self.__bbox[1]

        for i in range(self.__page_n):
            cr.set_source_rgb(0.5, 0.5, 0.5)
            cr.rectangle(wpos, hpos, self.__width, self.__height_per_page)
            cr.fill()

            if i == self.__current_page_n:
                cr.set_source_rgb(0, 0, 0)
            else:
                cr.set_source_rgb(1, 1, 1)

            cr.rectangle(wpos + 1, hpos + 1,
                        self.__width - 2, self.__height_per_page - 2)
            cr.fill()
            if i == self.__current_page_n:
                cr.set_source_rgb(1, 1, 1)
            else:
                cr.set_source_rgb(0, 0, 0)
            cr.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
            cr.set_font_size(14)
            cr.move_to(wpos + 5, hpos + 20)
            cr.show_text(str(i + 1))

            hpos += self.__height_per_page

            # draw layer symbols for the current page
            if i == self.__current_page_n:
                page = self.__gom.page()
                n_layers = page.number_of_layers()
                cur_layer = page.layer_no()
                cr.set_source_rgb(0.5, 0.5, 0.5)
                cr.rectangle(wpos, hpos, self.__width, 
                             n_layers * 5 + 5)
                cr.fill()

                hpos = hpos + n_layers * 5 + 5
                for j in range(n_layers):
                    # draw a small rhombus for each layer
                    curpos = hpos - j * 5 - 10
                    if j == cur_layer:
                        # inverted for the current layer
                        draw_rhomb(cr, (wpos, curpos, self.__width, 10),
                                   (1, 1, 1), (0, 0, 0))
                    else:
                        draw_rhomb(cr, (wpos, curpos, self.__width, 10))



    def update_size(self, width, height):
        """update the size of the widget"""


## ---------------------------------------------------------------------
class WigletToolSelector(Wiglet):
    """Wiglet for selecting the tool."""
    def __init__(self, coords = (50, 0), width = 1000, height = 35, func_mode = None):
        super().__init__("tool_selector", coords)

        self.__width, self.__height = width, height
        self.__icons_only = True

        self.__modes = [ "move", "draw", "shape", "rectangle",
                        "circle", "text", "eraser", "colorpicker" ]
        self.__modes_dict = { "move": "Move", "draw": "Draw", "shape": "Shape",
                              "rectangle": "Rectangle", "circle": "Circle", "text": "Text",
                              "eraser": "Eraser", "colorpicker": "Col.Pick" }

        if self.__icons_only and width > len(self.__modes) * 35:
            self.__width = len(self.__modes) * 35

        self.__bbox = (coords[0], coords[1], self.__width, self.__height)
        self.recalculate()
        self.__mode_func = func_mode
        self.__icons = { }

        self._init_icons()

    def _init_icons(self):
        """initialize the icons"""
        icons = Icons()
        self.__icons = { mode: icons.get(mode) for mode in self.__modes }
        print("icons:", self.__icons)

    def recalculate(self):
        """recalculate the position of the widget"""
        self.__bbox = (self.coords[0], self.coords[1], self.__width, self.__height)
        self.__dw   = self.__width / len(self.__modes)

    def draw(self, cr):
        """draw the widget"""
        cr.set_source_rgb(0.5, 0.5, 0.5)
        cr.rectangle(*self.__bbox)
        cr.fill()

        cur_mode = None
        if self.__mode_func and callable(self.__mode_func):
            cur_mode = self.__mode_func()

        for i, mode in enumerate(self.__modes):
            label = self.__modes_dict[mode]
            # white rectangle
            if mode == cur_mode:
                cr.set_source_rgb(0, 0, 0)
            else:
                cr.set_source_rgb(1, 1, 1)

            cr.rectangle(self.__bbox[0] + 1 + i * self.__dw,
                         self.__bbox[1] + 1,
                         self.__dw - 2,
                         self.__height - 2)
            cr.fill()
            # black text

            if mode == cur_mode:
                cr.set_source_rgb(1, 1, 1)
            else:
                cr.set_source_rgb(0, 0, 0)
            # select small font

            icon = self.__icons.get(mode)
            if not self.__icons_only:
                cr.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
                cr.set_font_size(14)
                x_bearing, y_bearing, t_width, t_height, _, _ = cr.text_extents(label)
                x0 = self.__bbox[0]
                if icon:
                    iw = icon.get_width()
                    x0 += i * self.__dw + (self.__dw - t_width - iw) / 2 - x_bearing + iw
                else:
                    x0 += i * self.__dw + (self.__dw - t_width) / 2 - x_bearing

                cr.move_to(x0, self.__bbox[1] + (self.__height - t_height) / 2 - y_bearing)
                cr.show_text(label)
            if icon:
                Gdk.cairo_set_source_pixbuf(cr,
                                            self.__icons[mode],
                                            self.__bbox[0] + i * self.__dw + 5,
                                            self.__bbox[1] + 5)
                cr.paint()


    def on_click(self, x, y, ev):
        """handle the click event"""

        if not is_click_in_bbox(x, y, self.__bbox):
            return False

        # which mode is at this position?
        print("clicked inside the bbox, event", ev)
        dx = x - self.__bbox[0]
        print("dx:", dx)
        sel_mode = None
        i = int(dx / self.__dw)
        sel_mode = self.__modes[i]
        print("selected mode:", sel_mode)
        if self.__mode_func and callable(self.__mode_func):
            self.__mode_func(sel_mode)


        return True

    def update_size(self, width, height):
        """update the size of the widget"""

class WigletColorSelector(Wiglet):
    """Wiglet for selecting the color."""
    def __init__(self, coords = (0, 0),
                 width = 15,
                 height = 500,
                 func_color = None,
                 func_bg = None):
        super().__init__("color_selector", coords)
        print("height:", height)

        self.__width, self.__height = width, height
        self.__bbox = (coords[0], coords[1], width, height)
        self.__colors = self.generate_colors()
        self.__dh = 25
        self.__func_color = func_color
        self.__func_bg    = func_bg
        self.recalculate()

    def recalculate(self):
        """recalculate the position of the widget"""
        self.__bbox = (self.coords[0], self.coords[1], self.__width, self.__height)
        self.__color_dh = (self.__height - self.__dh) / len(self.__colors)
        self.__colors_hpos = { color : self.__dh + i * self.__color_dh
                              for i, color in enumerate(self.__colors) }

    def update_size(self, width, height):
        """update the size of the widget"""
        _, self.__height = width, height
        self.recalculate()

    def draw(self, cr):
        """draw the widget"""
        # draw grey rectangle around my bbox
        cr.set_source_rgb(0.5, 0.5, 0.5)
        cr.rectangle(*self.__bbox)
        cr.fill()

        bg, fg = (0, 0, 0), (1, 1, 1)
        if self.__func_bg and callable(self.__func_bg):
            bg = self.__func_bg()
        if self.__func_color and callable(self.__func_color):
            fg = self.__func_color()

        cr.set_source_rgb(*bg)
        cr.rectangle(self.__bbox[0] + 4, self.__bbox[1] + 9, self.__width - 5, 23)
        cr.fill()
        cr.set_source_rgb(*fg)
        cr.rectangle(self.__bbox[0] + 1, self.__bbox[1] + 1, self.__width - 5, 14)
        cr.fill()

        # draw the colors
        dh = 25
        h = (self.__height - dh)/ len(self.__colors)
        for color in self.__colors:
            cr.set_source_rgb(*color)
            cr.rectangle(self.__bbox[0] + 1, self.__colors_hpos[color], self.__width - 2, h)
            cr.fill()

    def on_click(self, x, y, ev):
        """handle the click event"""
        if not is_click_in_bbox(x, y, self.__bbox):
            return False
        print("clicked inside the bbox")

        dy = y - self.__bbox[1]
        # which color is at this position?
        sel_color = None
        for color, ypos in self.__colors_hpos.items():
            if ypos <= dy <= ypos + self.__color_dh:
                print("selected color:", color)
                sel_color = color
        if ev.shift():
            print("setting bg to color", sel_color)
            if sel_color and self.__func_bg and callable(self.__func_bg):
                self.__func_bg(sel_color)
        else:
            print("setting fg to color", sel_color)
            if sel_color and self.__func_color and callable(self.__func_color):
                self.__func_color(sel_color)
        return True


    def event_update(self, x, y):
        """update on mouse move"""

    def event_finish(self):
        """update on mouse release"""

    def generate_colors(self):
        """Generate a rainbow of 24 colors."""

        # list of 24 colors forming a rainbow
        colors = [  #(0.88, 0.0, 0.83),
                     (0.29, 0.0, 0.51),
                     (0.0, 0.0, 1.0),
                     (0.0, 0.6, 1.0),
                     (0.0, 0.7, 0.5),
                     (0.0, 1.0, 0.0),
                     (1.0, 1.0, 0.0),
                     (1.0, 0.6, 0.0),
                     (0.776, 0.612, 0.427),
                     (1.0, 0.3, 0.0),
                     (1.0, 0.0, 0.0)]

       #colors = [ ]

       #for i in range(1, 21):
       #    h = i/20
       #    rgb = colorsys.hls_to_rgb(h, 0.5, 1)
       #    print("h=", h, "rgb:", *rgb)
       #    colors.append(rgb)

        newc = [ ]
        for i in range(11):
            newc.append((i/10, i/10, i/10))

        for c in colors:
            for dd in range(30, 180, 15): #
                d = dd / 100
                newc.append(adjust_color_brightness(c, d))

        return newc
"""
DrawManager is a class that manages the drawing on the canvas.
"""




class DrawManager:
    """
    DrawManager is a class that manages the drawing canvas.
    It holds information about the state, the mouse events, the position of
    the cursor, whether there is a current object being generated, whether
    a resize operation is in progress, whether the view is paned or zoomed etc.

    DrawManager must be aware of GOM, because GOM holds all the objects

    methods from app used:
    app.mm.object_menu()     # used for right click context menu
    app.clipboard.set_text() # used for color picker

    methods from state used:
    state.cursor()
    state.show_wiglets()
    state.mode()
    state.current_obj()
    state.current_obj_clear()
    state.queue_draw()
    state.pen()
    state.get_win_size()
    state.hover_obj()
    state.hover_obj_clear()

    methods from gom used:
    gom.set_page_number()
    gom.page()
    gom.draw()
    gom.selected_objects()
    gom.selection()
    gom.remove_objects()
    gom.selection().set()
    gom.selection().add()
    gom.selection().clear()
    gom.add_object()
    gom.kill_object()
    gom.remove_all()
    gom.command_append()
    """
    def __init__(self, gom, mm, state, wiglets, setter):
        self.__state = state
        self.__gom = gom
        self.__mm = mm
        self.__cursor = state.cursor()
        self.__wiglets = wiglets
        self.__setter = setter

        # objects that indicate the state of the drawing area
        self.__wiglet_active = None
        self.__resizeobj = None
        self.__selection_tool = None
        self.__paning = None
        self.__show_wiglets = True
        # drawing parameters

    def selection_tool(self):
        """Get the selection tool."""
        return self.__selection_tool

    # ---------------------------------------------------------------------
    #                              Event handlers

    def on_pan(self, gesture, direction, offset):
        """Handle panning events."""
        print(f"Panning: Direction: {direction}, Offset: {offset}, Gesture: {gesture}")

    def on_zoom(self, gesture, scale):
        """Handle zoom events."""
        print(f"Zooming: Scale: {scale}, gesture: {gesture}")

    def on_right_click(self, event, hover_obj):
        """Handle right click events - context menus."""
        if hover_obj:
            self.__state.mode("move")

            if not self.__gom.selection().contains(hover_obj):
                self.__gom.selection().set([ hover_obj ])

            # XXX - this should happen directly?
            sel_objects = self.__gom.selected_objects()
            self.__mm.object_menu(sel_objects).popup(None, None,
                                                         None, None,
                                                         event.button, event.time)
        else:
            self.__mm.context_menu().popup(None, None,
                                               None, None,
                                               event.button, event.time)
        self.__state.queue_draw()

    # ---------------------------------------------------------------------

    def __move_resize_rotate(self, ev):
        """Process events for moving, resizing and rotating objects."""

        if self.__state.mode() != "move":
            return False

        corner_obj, corner = ev.corner()
        hover_obj  = ev.hover()
        pos = ev.pos()
        shift, ctrl = ev.shift(), ev.ctrl()

        if corner_obj and corner_obj.bbox():
            print("starting resize")
            print("ctrl:", ctrl, "shift:", shift)
            # XXX this code here is one of the reasons why rotating or resizing the
            # whole selection does not work. The other reason is that the
            # selection object itself is not considered when detecting
            # hover or corner objects.
            if ctrl and shift:
                self.__resizeobj = RotateCommand(corner_obj, origin = pos,
                                                 corner = corner)
            else:
                self.__resizeobj = ResizeCommand(corner_obj, origin = pos,
                                                 corner = corner, proportional = ctrl)
            self.__gom.selection().set([ corner_obj ])
            self.__cursor.set(corner)
            return True

        if hover_obj:
            if ev.shift():
                # add if not present, remove if present
                print("adding object to selection:", hover_obj)
                self.__gom.selection().add(hover_obj)
            if not self.__gom.selection().contains(hover_obj):
                print("object not in selection, setting selection to it:", hover_obj)
                self.__gom.selection().set([ hover_obj ])
            # we are using the .selection().copy() because the selection
            # object can change in time
            self.__resizeobj = MoveCommand(self.__gom.selection().copy(), pos)
            self.__cursor.set("grabbing")
            return True

        if not ctrl and not shift:
            # start selection tool
            self.__gom.selection().clear()
            self.__resizeobj   = None
            print("starting selection tool")
            self.__selection_tool = SelectionTool([ pos, (pos[0] + 1, pos[1] + 1) ])
            self.__state.queue_draw()
            return True

        return False

    def create_object(self, ev):
        """Create an object based on the current mode."""

        # not managed by GOM: first create, then decide whether to add to GOM
        mode = self.__state.mode()

        if ev.shift() and not ev.ctrl():
            mode = "text"

        if not mode in [ "draw", "shape", "rectangle", "circle", "text" ]:
            return False

        obj = DrawableFactory.create_drawable(mode, pen = self.__state.pen(), ev=ev)

        if obj:
            self.__state.current_obj(obj)
        else:
            print("No object created for mode", mode)

        return True

    # Button press event handlers -------------------------------------------
    def on_button_press(self, widget, event):
        """Handle mouse button press events."""
        print("on_button_press: type:", event.type, "button:", event.button, "state:", event.state)
        self.__state.modified(True)
        ev = MouseEvent(event, self.__gom.objects(),
                        translate = self.__gom.page().translate(),
                        mode = self.__state.mode())

        # Ignore clicks when text input is active
        if self.__state.current_obj():
            if  self.__state.current_obj().type == "text":
                print("click, but text input active - finishing it first")
                self.finish_text_input()

        # right click: open context menu
        if event.button == 3:
            if self.__handle_button_3(event, ev):
                return True

        elif event.button == 1:
            if self.__handle_button_1(event, ev):
                return True

        return True

    def __handle_button_3(self, event, ev):
        """Handle right click events, unless shift is pressed."""
        if ev.shift():
            return False
        self.on_right_click(event, ev.hover())
        return True

    def __handle_button_1(self, event, ev):
        """Handle left click events."""
        print("handling button 1")

        # check whether any wiglet wants to process the event
        # processing in the order reverse to the drawing order,
        # so top wiglets are processed first
        if self.__handle_wiglets_on_click(event, ev):
            return True

        if self.__handle_text_input_on_click(ev):
            return True

        if ev.alt():
            self.__paning = (event.x, event.y)
            return True

        if self.__move_resize_rotate(ev):
            return True

        if self.__handle_mode_specials_on_click(event, ev):
            return True

        # simple click: create modus
        self.create_object(ev)
        self.__state.queue_draw()

        return True

    def __handle_wiglets_on_click(self, event, ev):
        """Pass the event to the wiglets"""
        if self.__show_wiglets:
            for w in self.__wiglets[::-1]:
                if w.on_click(event.x, event.y, ev):
                    self.__state.queue_draw()
                    return True
        return False


    def __handle_mode_specials_on_click(self, event, ev):
        """Handle special events for the current mode."""

        # Start changing line width: single click with ctrl pressed
        if ev.ctrl(): # and self.__mode == "draw":
            print("current line width, transparency:", self.__state.pen().line_width, self.__state.pen().transparency)
            if not ev.shift():
                self.__wiglet_active = WigletLineWidth((event.x, event.y), self.__state.pen())
                self.__wiglets.append(self.__wiglet_active)
            else:
                self.__wiglet_active = WigletTransparency((event.x, event.y), self.__state.pen())
                self.__wiglets.append(self.__wiglet_active)
            self.__state.queue_draw()
            return True

        if self.__handle_eraser_on_click(ev):
            return True

        return False

    def __handle_eraser_on_click(self, ev):
        """Handle eraser on click events."""
        if not self.__state.mode() == "eraser":
            return False

        hover_obj = ev.hover()
        if not hover_obj:
            return False

        self.__gom.remove_objects([ hover_obj ], clear_selection = True)
        self.__resizeobj   = None
        self.__cursor.revert()
        self.__state.queue_draw()
        return True

    def __handle_text_input_on_click(self, ev):
        """Check whether text object should be activated."""
        hover_obj = ev.hover()

        if not hover_obj or hover_obj.type != "text":
            return False

        if not self.__state.mode() in ["draw", "text", "move"]:
            return False

        if not ev.double():
            return False

        if ev.shift() or ev.ctrl():
            return False

        # only when double clicked with no modifiers - start editing the hover obj
        print("starting text editing existing object")
        hover_obj.move_caret("End")
        self.__state.current_obj(hover_obj)
        self.__state.queue_draw()
        self.__cursor.set("none")
        return True


    # Event handlers
    # XXX same comment as above
    # Button release handlers ------------------------------------------------
    def on_button_release(self, widget, event):
        """Handle mouse button release events."""
        print("button release: type:", event.type, "button:", event.button, "state:", event.state)
        ev = MouseEvent(event, self.__gom.objects(),
                        translate = self.__gom.page().translate(),
                        mode = self.__state.mode())

        if self.__paning:
            self.__paning = None
            return True

        if self.__handle_current_object_on_release(ev):
            return True

        if self.__handle_wiglets_on_release():
            return True

        if self.__handle_selection_on_release():
            return True

        if self.__handle_drag_on_release(event):
            return True

        return True

    def __handle_current_object_on_release(self, ev):
        """Handle the current object on mouse release."""
        obj = self.__state.current_obj()
        if not obj:
            return False

        if obj.type in [ "shape", "path" ]:
            print("finishing path / shape")
            obj.path_append(ev.x, ev.y, 0)
            obj.finish()
            # remove paths that are too small
            if len(obj.coords) < 3:
                print("removing object of type", obj.type, "because too small")
                self.__state.current_obj_clear()
                obj = None

        # remove objects that are too small
        if obj:
            bb = obj.bbox()
            if bb and obj.type in [ "rectangle", "box", "circle" ] and bb[2] == 0 and bb[3] == 0:
                print("removing object of type", obj.type, "because too small")
                self.__state.current_obj_clear()
                obj = None

        if obj:
            self.__gom.add_object(obj)
            # with text, we are not done yet! Need to keep current object
            # such that keyboard events can update it
            if self.__state.current_obj().type != "text":
                self.__gom.selection().clear()
                self.__state.current_obj_clear()

        self.__state.queue_draw()
        return True

    def __handle_wiglets_on_release(self):
        """Handle wiglets on mouse release."""
        if not self.__wiglet_active:
            return False

        self.__wiglet_active.event_finish()
        self.__wiglets.remove(self.__wiglet_active)
        self.__wiglet_active = None
        self.__state.queue_draw()
        return True

    def __handle_selection_on_release(self):
        """Handle selection on mouse release."""
        if not self.__selection_tool:
            return False

        print("finishing selection tool")
        objects = self.__selection_tool.objects_in_selection(self.__gom.objects())

        if len(objects) > 0:
            self.__gom.selection().set(objects)
        else:
            self.__gom.selection().clear()
        self.__selection_tool = None
        self.__state.queue_draw()
        return True

    def __handle_drag_on_release(self, event):
        """Handle mouse release on drag events."""
        if not self.__resizeobj:
            return False

        # If the user was dragging a selected object and the drag ends
        # in the lower left corner, delete the object
        self.__resizeobj.event_finish()
        # self.__resizeobj is a command object
        # self.__resizeobj.obj is a copy of the selection group
        # self.__resizeobj.obj.objects is the list of objects in the copy of the selection group

        obj = self.__resizeobj.obj

        _, height = self.__state.get_win_size()
        if self.__resizeobj.command_type() == "move" and  event.x < 10 and event.y > height - 10:
            # command group because these are two commands: first move,
            # then remove
            self.__gom.command_append([ self.__resizeobj,
                                       RemoveCommand(obj.objects,
                                                     self.__gom.objects()) ])
            self.__gom.selection().clear()
        else:
            self.__gom.command_append([ self.__resizeobj ])
        self.__resizeobj    = None
        self.__cursor.revert()
        self.__state.queue_draw()

        return True

    # ---------------------------------------------------------------------
    # motion event handlers

    def on_motion_notify(self, widget, event):
        """Handle mouse motion events."""

        ev = MouseEvent(event, self.__gom.objects(),
                        translate = self.__gom.page().translate(),
                        mode = self.__state.mode())
        x, y = ev.pos()
        self.__cursor.update_pos(x, y)

        if self.__on_motion_wiglet(event.x, event.y):
            return True

        # we are paning
        if self.__on_motion_paning(event):
            return True

        if self.__on_motion_update_object(ev):
            return True

        if self.__on_motion_update_resize(ev):
            return True

        self.__on_motion_process_hover(ev)
        # stop event propagation
        return True

    def __on_motion_process_hover(self, ev):
        """Process hover events."""

        if not self.__state.mode() == "move":
            return False

        object_underneath = ev.hover()

        if object_underneath:
            self.__cursor.set("moving")
            self.__state.hover_obj(object_underneath)
        else:
            self.__cursor.revert()
            self.__state.hover_obj_clear()

        corner_obj, corner = ev.corner()

        if corner_obj and corner_obj.bbox():
            self.__cursor.set(corner)
            self.__state.hover_obj(corner_obj)
            self.__state.queue_draw()

        self.__state.queue_draw()
        return True

    def __on_motion_update_resize(self, event):
        """Handle on motion update for resizing."""
        if not self.__resizeobj:
            return False

        self.__resizeobj.event_update(event.x, event.y)
        self.__state.queue_draw()
        return True

    def __on_motion_update_object(self, event):
        """Handle on motion update for an object."""
        obj = self.__state.current_obj() or self.__selection_tool
        if not obj:
            return False

        obj.update(event.x, event.y, event.pressure())
        self.__state.queue_draw()
        return True

    def __on_motion_paning(self, event):
        """Handle on motion update when paning"""
        if not self.__paning:
            return False

        tr = self.__gom.page().translate()
        if not tr:
            tr = self.__gom.page().translate((0, 0))
        dx, dy = event.x - self.__paning[0], event.y - self.__paning[1]
        tr = (tr[0] + dx, tr[1] + dy)
        self.__gom.page().translate(tr)
        #self.__translate = (self.__translate[0] + dx, self.__translate[1] + dy)
        self.__paning = (event.x, event.y)
        self.__state.queue_draw()
        return True

    def __on_motion_wiglet(self, x, y):
        """Handle on motion update when a wiglet is active."""
        if not self.__wiglet_active:
            return False

        self.__wiglet_active.event_update(x, y)
        self.__state.queue_draw()
        return True
## singleton class for serving icon pixbufs



class Icons:
    """
    Singleton class for serving
    icon pixbufs.
    """
    __new_instance = None

    def __new__(cls, *args, **kwargs):
        if cls.__new_instance is None:
            cls.__new_instance = super(Icons, cls).__new__(cls)
        return cls.__new_instance

    def __init__(self):
        self._icons = {}
        self._load_icons()

    def get(self, name):
        """Get the icon with the given name."""
        if name not in self._icons:
            print("icon not found")
            return None
        return self._icons.get(name)

    def _load_icons(self):
        """
        Load all icons from stored base64 strings.
        """

        icons = { "colorpicker": "iVBORw0KGgoAAAANSUhEUgAAABwAAAAbCAYAAABvCO8sAAACm0lEQVRIx7XWT4hVVRgA8N+ZGRuFnkgoEhMTITbZJsEg7I8WVIsQ0UELFxGNk6KbmFYtEqIsApNW4mbmRRRJi+i1UrNCFyahhbkIF9ngFKMp40yMohX2tTkjt/G+8s17c+Bu7v3O/XG+e75zv2SWRkTE9HsppZUds4WllL7GZQxgCVZHxLGOWcKOYSyqlY1YC6lv8hC0NfPi4ihgx3EqY8UxHzqaWMUr2IyfUMv3DuBoVCtvlkxbDKkJ8Aj2YAPOYByrolpZc9Nm6Zv8AQsjoqutic/1I3bhT9yN3jrYW5gbEV0ppZSa3Bz7sDTvhe/xYFQrywvY53gkIhY2Y03V2Bpcwa/4Aidziv+KaiUwgi/L6nEmWC+uZ/BevJjRU9iBg9jfKmwFruEqni48fgz7MJQ30gc4NGM0T1yGCzmND5SEPZdX+Q6ewi8zAvOkroz9hnvqhPbgW5zHaYw0DOYJnfgZO9FdJ3QAYzm+He/i44bAHNyNsxmrN7oLmOlHXiPYC/nE3wn9ajF1FULvK6Su/f/em/6jqPtzWjbgq2kIGLTuCXyUT5HOfOo0BmbsNTyLT/B2GTa0YPtnMTF6f0T0pJTa8fetZK6tBHs519RDnds+/LQMe7/79RMxMbooY3NvFfsXmLEBrMfG+YNx1/N7K6dvwnreO3t95OSliHg0pXQb/mhk16cC9hL68GS/2pWy4KFFAxfi4vDRiFjfSBrrpXQrnqmHVe989VpcHK5lLM0Eg468ut040a82Xrqy2zdPxrmxXRHxRlO/mUKLsVZq61VSpoNzNv3u8tXDGZvXikZr+MYHXfr4xFRhz1m95TspfYPRXNSpFVjCJdxRaBvm4RwWRMSy3MC2BLtRDu1LHh6f+pVMOwc7W923/gMDooiVvplNJAAAAABJRU5ErkJggg==",
                 "rectangle": "iVBORw0KGgoAAAANSUhEUgAAABwAAAAaCAMAAACTisy7AAACdlBMVEW+gFX///8AAAD///////////////////////////////////////////////////////////////////////////////////////////////////////////////////7//////v3///////////////////////////////////////////////////////7///////////////////////////////////////+0tLRwcXByc3FxcnB/gH7s7et0dHQAAAANABoPAB0NABwjEDHCrtDy6Pl1dnULABdVAJZeAKVdAKRdAKVfAqZqC7K1hdkMABlcAKJlALJoB7NqCrRnBrFhAKtjAK9hAK+vfNZbAKFmA7Lbw+3ZwOzcxO+agaweAjRTAJJjALGwfNdnBLLRs+f///+rrq0DAQhMAIevfNjPsef/+/b/1pj/zID/zYH/z4PXpVmEUBBrE4phALOvY5n/z4H/zYP/3q3/+O3/rjX/mwf/nQn/nAn/ngr7mRWIJItfALWvTGD/nwf/mwX/vlz/rzj/nQr/ng34lxWHJItgALWvTGH/oAv/v17Qsef/+e7Fptzy9PLx6t/8rTVcAKFkAa9BJFgwNC5BOy7cjRH/ogr/oQr6mRKIJIp0dXMKABdkAK86AGgmAEYzBEmiSlG+W1O9WlO9W1K5Vld4FZtgALTCw8GDc5BmCaxiAK9jALBiALKuS2Lu4fiMQcSALr6BL7+BLruBIJqBHZJ/G5S+W1L9/P717/r07fr07vv05un0o0H0khb0kxj0kxn5mBP/sDj/nwz/rjb/nAf/mwb//Pb/2J7/z4j/0In/z4f/4LGPZS+MAAAAAXRSTlMAQObYZgAAATVJREFUKM9jYMALODi50AA3QpLPytoGGdja2XPAJQUdHJ2ckYCLq5u7EFzSw9PL2wcBfP38A0QQkoFBwSGhMBAWHhEZhSwZHRMQGwcB8QmJSckposiSqWnpMJCRmZWcLcUpLSMjC5PMSc/Nyy8AgsKi4pLSsvKKioJKeYRkVXVNLRDU1Tc0NjW31Na2tikiSbZ3dIJBV3dPb19nZ10/FSQnpE/ELTlp8pSpOCSnTZ8xc9bsOXPBYN58ZMkFC6MXLV6ydNnyFSCwctXqNQhJ5bXr1m/YuDG5FAY2rdkMl1RV27J12/btO3bugoHde+CSDBp79+3ff+DgocNHoODosU64JIOmlpa2zsTj9SdgoBNJEgzggQAFqJInT3Ugg9NtSJJnzp47jwwuXNRHSBoaoQEpY/xJGgDtYfX0n9HvoQAAAABJRU5ErkJggg==",
                 "move":"iVBORw0KGgoAAAANSUhEUgAAABwAAAAZCAYAAAAiwE4nAAAFKUlEQVRIx82WfWyV5RnGf8973p6ecvr9QVtKi622sdDSxiGooFCUDzdHYgzOZRMTY/xgLINsSkwUzdRADMZkC8YxNH5MplUajMygVqAM6BCC8Qs0tlAp3dmhHNrz0fac03Peyz/s6QrSQmJcdv/1PHnu57ne676u+84L/+MwAJL0gx4xxlxqrh2Px1PJL6Y+YCQmATYgoB/oBM6O7BNAEDgFHJSkSwU1khSLxQL/bD1YEDyUTTKcjjAgGI4mcRLCSjPklXnIK/NguQzGBRnZdjJvqieSX57R7c1P+wL4xaWwtgHe3fmPgt//+imui6zh8pI+CopOgwy2PYzLlSSZtDkZzGcomolkkCySjscld25OztS8nPLG3Lor5uZSWJkxUHplVn9mofuV8VjbAGm2u7/f6ckNE2B6XYez/I5XLclgjDBGSIZ43E089h17xzHEY+mcOVPAya4KOturad46nVgix7vs8Rrv0oeqbxhXQ4AZ02cECktzckOdvUSjGQnLctLG6mmM8GQM4ckYGr0Yi3qIxdMpLg0wMFDA8eNR3PmTmDYrByAwEWB2ZWVlqKSsmP7OUwSDNe5Ewsa2E+dm6r/VkQxvb7+NvXsWEhyO0KfTFDcEuOuJmcO1CyfvApaNp6FljAkD1NRU08dx/uMvJBzKAaMJ2kBMmdKD2x0npCCNd6fz6Pbl3TfcMmsVsGQix1qpRUNDAyFO4e+38ftLJgQEmHd9G/ev/BMLaodIfjCDfRsG8o619t54sb62U4v6uplhV1Y0KxAO4esp58q6Ty/aUzW1R5la3s2Rwz+hbWdT5hc7625vvLU8fN2Ksu0p0PPZjgJWVFT48oqzs4Lh05z2l4BjXbyLHYtJ3gjzFuyh8aqPOXTwana/sTjrWOv0O+f8qtw3+47S77VHCnBSVVXVYMmUyYQ6/AQC00gmbFyu5DlGMZZzjnlGzSRDZlaIpkWtzGz4lL175rNnY1Pp0fer1y5cddmxEbbZxphwikYUYEpZKYMECIczGR5OG9Vx965F/P21u+g7WwCWc2G2MuBYFBT1cuvyt1j5m40Mfb2P7Y9+Vdv/76gDPCsp2xqpswDy8/MZZohY3E0y6QLL4ZuuSna881N27LqWl164j+6uSjDO+KaSASMkMRDNpqapgMxCtwXcDFSeI1Sm10ucMIND6cRjHjq+rOVvL9+DzzVAm1nPzqNxNv15NR++/zMi4ezv2J4PbEQk7KX5jdsobGhk2bpqbLcFsBU4Zo/NraiYRoRt9IYMe9sW0t4+C1NVyoOv1jK5pYPXXthMT+AaTr5+M4cPzWF+UysNjUfI8EZGTSbH4oP3FtEbuYaVT9aQVeQB6AZeN8bEx5hCam8/ECvKLdXPWa/7TIs2Lm3TyQ5fUtKmaDT63LaWbYMLFsxXoatKV3OvVrhe1DO1G/TZmnlKbsmRXvLqwD2LtTq3Wbuf69JIJCT97vull+Tz+Trq6+t1FSu0tuo9dR3uC0p6RGPC7/cf+cvm5zV79hwVmRrNY40eyHhZby65X5+snqu1hVvU/IejSsSd1JUWSZkXAkyTpHWPrdOc+vna1fJRQtKD50+NkUce7unp+XrD0+tVWXG5pjFXt7NJv3Vv1TM37VfkbDwF9rmkGeP2sSQNDg6q90yvkk5ix3gjagzhPx5oP3BqydLFyucy/bLqSXX+K5A680m6acLBIal4bPkuNmhSeSdOnGj+65bNw/v37xur26of7U9sBOShEVZJSVsvqNuPAJqKIv5f4lttCtvxW3g8aQAAAABJRU5ErkJggg==",
                 "text":"iVBORw0KGgoAAAANSUhEUgAAABwAAAAbCAQAAADFASenAAACtklEQVQ4y42US2iTWRiGn5M//RuTXmK1l4xVK0gRb2hxiooywwiKNwQXdlEVQUUUEaSIIIK3zUCHKY6goqAuRzfKaMWFFUXd1GEQaqFQ8JK0SUNsQo1JTW3yuqi1Nhc7z/blPec73/d+xwAE8fGMo5RymjjjbKaYvbTioZ9a8vIK8e63dExx5ZBcIJTHY+AtXsq9xKrCPLVGDWTIZBwOBxqqXrm6w/W5ZCghqrKMDijFhlhLMlL9flns04dAZE14Z0VDoi/sTjb853o8VPTRzWi+QoPISI2i3TFfkpw3bR28JElswn9SkgjkuBxQAnOhp4/j6d5fDAhIAz6jdlo7gOFZ7nxGC0bB1aIuY5LfSSGM0V+Jw5DJmByjEyzog/DN5VmqH7BMWuAOZfIZw6SxjLhDXY7cR78RjjwDccJcCvMTUcQIMV4DCcqBDHVjbyxMBOGhoqhGC5lHJ4tlT/fiGm9OYWbyELuEkZbHEmrSrnPV0VIbAmOlFsbPovn0bhvs+NX/cs3zuws6T7QOH0uVmFjujZMb+y++3uVb/aes1D9lbV1P7scT1zrBk33+H0ARknXL1v6LUhX7GAt6TdJ7RZI826UebmXnSNhIbKAbzbvPCslGnEXyfZxxRapBksrYMj6OcQZJ6WDb+ntppzAbl666fSSluLEBRwYDv2MMwL3sUqPkbuMnLiDVxmdelW5wniD9ud2IMp0Qn7+FuAI3KVxIlbJeDDQCBIs1UssUBBlFTmlZrzNkqTK6sP3AOekdU/KWZqTNz1lfW+8+PaNjbZME3VMbB5HaHlU2T7z4/NedmSJyRcCAJ/IC6hvLzixpgN3AnIntKISAYpsy8B56syf9M/Twv4igadKKPyVpyd/rHkhdk/exEG4Cw7PZ4Vt3PfS6es7Dpskfyw8IcJmXE6FABL9TvwD2d2QO1hAt8wAAAABJRU5ErkJggg==",
                 "eraser":"iVBORw0KGgoAAAANSUhEUgAAABwAAAAaCAYAAACkVDyJAAAFj0lEQVRIx62Wa1CUVRzGn3PevS+wCywXXdmFdVPMuClU462UFEUJrzGNqTM0jFPQ2DSmH5ypkWo0bZokGx0LY6ycIacMAnWUBtAmLxiChHKTqwqoqMCy7L67+/770GqIkKidj+cyv/M85/wvDE8xiIhGW2OMsZHmZU8DYyx1tlzw5E4z9wtygXraelS+Xb0KhcvDj927zHCw7ClgC/w14nebk9vDN8y/1qtSSL4DDsFx5oqf36lGXWacee71qnafA8PB7Mlgy2LDDbb8bStbJqcl3CDGIYKgBABwAhjo9zr9rfmfRcPhYi0AdhIVHWKMMdnjw5ZPnGa+m/fx8pbJi6J7AGLsPgwAJAZwYjf65G6PW8784Rdsh337Y1vqtTFx3czOvPeSOmZHh9kAGsEgTrh8TYuthWbDVIrpj0OMphOdmhi25TUiItnYYa/uTo27+fPWpa06c9AgILFB74XbACgAmMCJrt9Wiht/jFC0dphrliHKIIPMFIYwuOBKG5NCL6zSFOBI2rayReWF9QNoAWAEcAnAi+CEM4066Z2DE90tLZbBJMwNVUAxDoCjC129NaixPhLohTUqBOnsb5uquTXE/iYk5gHgADDJez4ZnGTHa/zx1vdWp617UnsiZobroNMDQBvabpWi1OOCWAYgRvYIWM0zIfZzlR/+6fRRuzNATPJCgoa+2V8dPtInB2dyn+7I1jl4TiWHXA0AfehDBSoCXXDmEBVvHvWX/gNL2fdy5N1Vh7NqXT5qTxzoocxBYGCDTgGfFkzmPp2JjkiEBEmQgggEF1yoQpV4B7e3ERVn34tDPspv3PiCpW/1/vR6f73WPQ30QLwOAKgHYAcj5J8z4Gh1yF09fCQCBQKABInO47y7AQ0FREXZQ7MNH0HZ5yq5+/0tS9o9EcGD9SD0DdlSDmALgAFw0pZe0uOjQotodEXWaqB2E4hz8O4udN1qRKNNosJVw1MbH5ZBIgG8YfARlXFmmxycnGAkDNkfAoZ0MIoprgqwv/ttPIJuzkU8YqcD8GNgHa1o3VqK0itO+knPWYpyuIPCPZggMCVR81qNVhs+wTTlSOt1J7t+S5iuU0uOAF93OwRygGGCy82xvdh054v8l3wn9i6QWWAWOLhMgiRWo/rcKZyc46LDCYylRAFFnQ9VkX/VsQ0mk2lnTs4uZ2rq0saGphudG7J2W2ovVBvUrL9kaujdukWxNhw4HWo+02BYlIzFulAEayRI8MCDSlRKF3GxU6JCI2MpBqCoZ8Sy5YXNZ4zl7tmzx3/9+vW9AL4GEDM4KCb8cbpu/OVLHcKZsw3iLwWlfQP9zoEYPH8hAfEJAMo5eHwzmieVoazDRYfDRquDD7wh53yXUqnUNzQ03Kyvr9cAWA5AUKsVNYnzoo9nZS2uyv0mU/HKPKVBjnZ7NKJjGNivHPyYA47AJjT1jgV2H2i1Wi2ZmZl+FRUVloULF6rS0tJ0BQUFJlEU1QCMoigGHzlSDLv9zqBWpTL1oTccQJId9t1XcbWuhb7UMTZFGEteZkREERERTWvWrGHjx48Pr66u5iUlJVebmpqcGo3mbHJy8hXOeWZ6enrgrFkz7Tt2HHbvyz4p6KEX7LDvb6Ov3h6LsuF9yb4ZM2bYc3JyKDc3l/bu3Xs7Kyur1mg0NgMYyMjIcNhsNjcRieXlF3sE2RIbkFz6Xz3NaEPGGGNERCqV6gOZTPZ6QkLCFKPR6B8VFeVvsVi68/Ly+IoVK5RarRYOh0M4ejTf1+NuPEhUt+6xlA0twPegjLGw2traVQEBAau1Wu10q9Xa3N/fL544ceJZq9UqZmdnGw8dOlRGZH8i2Ij2em0aB2CTt9b1CILQrNPpagD88CQ2PhT4o/WbjLF4AEsBxAK4TURrn1YZe5xG9/+w8W8l9atrgG4MXQAAAABJRU5ErkJggg==",
                 "circle":"iVBORw0KGgoAAAANSUhEUgAAABwAAAAbCAYAAABvCO8sAAAEqElEQVRIx8WWWWxVVRSGv33OHdrblk400FKwKTMyyZQYxYggEFGJyoMx0YBiYhAeMFETSdQHNGpiDIk4oJEHEzFAMBgkBgICKkRFi2KZhNILtpSWTtzeofees38fbkOYLpHBuN7W3jn7y17r//c6cB0hyeW/DlnrSpquS2OBZAtuPUw2X9Lrsn6nOo9Ip7ZKrT9LXqJX0jpJ1bfwZp6RtEKZhKcDb0tfDpfWlkifV0m7npF6TkvSBskWX8+5Ts4d494DeoFDH7jUvQnxJnDzINMDx7+An16CdPdjYBbdNFBSEHiOtt/KqF8NNg3Vc2H2Jhi7DNwwRL+BE+sN8OxNl1bSRMm2au9y6dOwtH6M1FFvJR1UOpbQtgXZ9c3TpeQ5K+npmy3pvcSbKmjeCRioeQRKxzQCiwgW7mTkQggUQuchOLffALNvGNjntffoqIee0xAqhuo5ANuMMfuBjVRMtRQPBa8HWvYC3ClpsaQ5kkZKCl/PDbOq6zwIXgIKqqFkuAV29+3vI69/K2UTQBba/8BLJodkUnaN9bUZ2AmkJM2XVejywwNXAZYDzZxvqEI+FN0GodJu4Gjf/miME6J4BDgu3SdOsP7j3XR39TNF/YPh26aUVI17YACDxvZbh2GtrF41jmm/FrAEmy4geTabRSrBDZcCZyXNAN6P1sXLTq/3uHt4AKW6aNgTpaV9IAbxy4Ymdqxq4L5ltfkzl9UuCRW4EVktNY6J5yppBJsJkolnBRMqvrjUK88cjg367Klfqdsax8olGPLIy8/g4OBiMBg6m5N8teIwW1Yexff0JIYnrtVDF8lBXt8DcKEIs3xP07595zjRP7surBsjHMde+mZgsFZ8t/okx3adc4GFkopzATMYx8cJAsqavs8q7Y2JwOHtrRgMgYCHMcJag+9fOUQMEI9lOLC5Jdt3qM0FjOEE0wT7ZbPeDkAtwNBYay+JrgwGQ6SgB+NYMukQvekwBl3VBu3RBH5G+UBZLmAnTrCUSGU2izeBlyoCCsOFAYJ5LiDKy9vB9emJF5JM5Oc0eqQ0iBMwGSCnaDqyEhmR7VMsCqm2PKC2ojbiD76jGJc0gwb9DUa0n6sglcq/6g0DjsPomRUYwyng5FWBxpgYsIbycRAqgkQzdBx0AcKFgaZZy4dTMypBdXUUrEO0sYaMH7wCZhHj5w1gwsMDATYZY85e6y39npLRln7DIBOH6BaQ3wnUj51b4S16o5uysjaSsQhHDo1CWXkhhI9wXMOk+ZU8vmockZLgD8BH1zJ+FpjXv4Eh84bRXgent8KZPaVUzbjLSTU7VeGNEEjTk5mGO3AiZQmwniVcGKBydBGTF1Qx6dFK8ouDe4DnjTHNF6s314h6i/MNL7N9AXQdhrLxMGoxNO/IzsJAPpr+CcnSh+hpS+JlLHlFAYoqwl4wz2kE1gEfGmPOXG6XXMBa4GsaNt7OvuWQasuKSD44IRi7FCa/1o0TWgO8eNnnNcaY6I0M4gdl/TNq3CxtuV9aN0zaNFX6/V0pHeuV9IrS580t/nPTHEk/qrcrpe6/pESLJ+mYpCWyfuh6zzP/EloOTAEGA53A/hsq2f8R/wC95leyPYReegAAAABJRU5ErkJggg==",
                 "shape":"iVBORw0KGgoAAAANSUhEUgAAABkAAAAcCAYAAACUJBTQAAAFb0lEQVRIx62VW2xU5RbHf2vvmenM0CmdWlo6XEppBUUqImgo5ByVEOEBRF9UDAovXhBjjCbknCcviU9eIlFiUHMSjXjUqMFLVBCNRQUVMEALVqBjLS2X0pmh7dw6M/tbPsxMKdNiSPSffMneX75v/ff6r7X+Gy4TqmrpBcRVNaqqO1X1DlW1+LswRu1C8KiTNZoazGp22CkSRlV1k6qWXeq+XGYWCpzv+Ka/cvfWLvq7kpRP8nDNrTW03DsNf9CdATaIyP/+Fsnet3p4/4k2BvqHEUABG2H+7XWsebmZ4FTf58AdIpIpvW9dDkG0O3X2280HSfQnsbCwEOzC9+3ffoodz51AlRAwYbwY1qhgs/ViiKoqZrjf2/1q7drV/2XtujepnBhDRwkgQOfeKIlIxhQSHANXgcAHdBhH6Tk8SN/xeHLhnVMMsIfj2xb7w8/ir08xrT4MwP+3rSOXc40EySQdyqs914vI+b/KJAhw5Ms+Nq/Yy2trDvifmtdqune1LqbtecilwdigwqKWH5jbfAgHeyRIRa0vV5S2AHs8EgU4/l2EWF8KNUrv4ZjV+8FmSHaBFI6p4PGmWXnbdmZMD6MIgs28m9MWx7YOcfiFDH0/9gO5gjoX5AKiAMbRtCBeRQhWxmiobwdT0oDGon5GmA0bN3P0yFyMullQ02mxpzOAyUH59GoWPO3QeFdSVX0ikpZRhQ8Ag/fLxzsM1qIpdd0TN/3nWcorBkHH6XRRsExeAyMXRFEHvDWw5OVh6leViYiMdJeIDImIvK6rlwenVtzdtDjYhe0GMfmAY3pbwLHztRo9CWJD6gz88WlZqVyMIhNV1dSp8Err4E0vxX77ocltp5hQHkcsM35W4824ZWtx2F2XOuYLzfxs0Hl+4/tb9mzp/aWHude2s3TZTqprzoKxLpYNk5csL0wWT0U/vsnbgQ1jbKXgUbx+z4FVhz45bW55pHHpsd2Rx4/tjYqFMOvKDu5b/wZ1od58RqIkExMYkvmDtf7vKy76yvVxRERGSFS1CojseO7Evp/e6Zl1rjPhH044iIWtjlpaKInBYs7VbTzw0CsEJg4wNFDJO2+v48TJJaZhcV344VX/biolAHCpahCI7H6ti8+e+e2GZDyLIHkTNKVKK12/NxKJVBMIxvj5xxb27VsE5KzYRyebNu54l5sebBi6EwKqWll0AAuIxnpSzq6XwiTjWawCwYiECAYLRfC4h5nbfIja2jNgLOLxAAYLCndSCcPB944GTrfuBIgV5XcB9LYP2ec6E1glzq8Ifl+C+dcfYEZDmFCoh6nTTuLzJ0GFJf9q5cyZOtra5pHLupk543dW3/4BNZ0noK8mwbxNW4pOa3758JRsvWs/xtGLCCoCA9yz9k0WLNyHuLL5YhdXobMyaS/d3fWk0z6mTuumsiqS7zQ1MGVZhuWfeFyAeCa4EFtgFIlBaL6unYU3/JTfcOwSa3UDgqcsRdPsjnwrG2tUe1sQbfcUa5KcPKuc4BQvTuF3YFAmTvLSsqYK7DGjDrYvi2ag5cU2QkuHMQK5os1qYTngrR5x4f3VM/2sevIqJjeW4/baVIV8maWPNn591YrZGdQ9dppNMr85a10zyz8tY87D56m6FjyV4K7IL38I5jxUUFV1GfAVQKQryblwIudk9ZVrltc8RrrvFD1fB8nFXRzZ4mbgeHGquxGdzvo45HN1ON/xBqmzi/K/YFU8lcIV1zUANxaH8WrgaIksZSDDI69frIDTrSCuC4Y5auiK7VqCgIjErYIp/ipjYGWKTwB4J4Fx8lZuDBjdVmqs4yD+lwY5Bs2PK3aZEP8DXOUQnDPIPwlVDamqqpPNamZQNZsYuIQ84+JPZVirv1LpTsoAAAAASUVORK5CYII=",
                 "draw":"iVBORw0KGgoAAAANSUhEUgAAABwAAAAaCAYAAACkVDyJAAAFAUlEQVRIx92WWWxUZRTHf9+9t8y0nel02s6UQmkZ28rSBbugCEUIi4IokWAUMS4oD8aYiDwQouEB9YEXDIkmJJAIQRNME8FEgQASVmVrrRTaQgtCO3SGdjrTZdpZeufO50OhQC1Y0Sf/yffy3XPO/5x7vrPA/x1ipEsppXyokhDiUQm1B5K5D9TiOT6ZQF0iER9ICaY0SeaMS3/n0MOc0u4hsgBB2o400rxrCm1HSweJ4iCUO1KCzppiQl4PBW/WErzmItiSQTRgxogqqOYIic4OrK7Ld5waTixuk2UBHmo2GjTvUulvAyUBUvLAOQNsBQaKpnDzsMB7HJQxYLJDtAti4UGnkCDE4LcEK6Q8BtnPVlO2oeJeUk1KaQI8nF0HDVtVpAG2SZC/Mkrui+ewT50NqAB0nB3UiusQbocxtkHDpjRQTaD3QcgLoXbwVUP3lQr0/moppbxDqgERWvcHafrGCsDE5ZLiD8/iqJgBzAagp6GaM2sq8J4ERQXikJAC0zcFyJpTR4KlE0UzMAacRHwuuurH07hdpbMG2n8t+2sO3futRAPgKIfS9TuwF74zlNtw1z7/oa+XtBwdoMO3DN1IxmoJkD/JTXZZ7JSwTFg6ZC0BMKdD6uRXSU4VNOzUMTlVoOouYbClF191CqoZxla230tWt69jf92e5sX1PxYS6JyJLjUkAgWw27pYEM2ZPv8T3u73RyKB1vDUPv9Aph42khRNLLA4ym+mT5rfkJZF1VBKACGllBx7awN6SKHgdQ8TX9ru/r3ni192tq6qrvLYA94oIEiyKKSON6NpBsE2Pz3dSSTZxlDygnOgxxtVA+6QGumNYegSRRUk2jTs2Ynx7Gkp7Su2FGfdebFieKE3n/Tv3ftx47KmU37iCDLsnVTM9VC8ekV3ZklOp3mg0XV+0xa1audioroJiSSORCDQhEDRBHEDYvE4cSQqAkduMovW5dfPfd9VKIYX/dbl5zi/x4PZBNOfPMPC+T+QPefpXubuSAG48F21UfVRrdp+yznUpibPy6DouUxSs80DpmRNj0UNzd8SNrXWdnPlmJ+AN8y4x618fmX+fYUvq6vaGusPdkzRFJXn19pZPO0QargVrK+5gcK6n279tnt9V1nHLSdWm0ZMl0RDMUqWjI0sXJtnBsbcPkOo+d5z7vDma1GhIIDK+1pb0wl/QV+/Tm6RnWfey0O9kAh9QE+9K3Byt3//p6llHS0DZDjCvLK5iCPburl8ykfnjZAYqatIKbXy5eP08uXjhu6UewVC3bp3bK7levEix/WUCWM7cFYCkvj1g0mHPzuZ3nw+SmKiZOn69L6KN4owWzQkklg0rj6gn8bEMAxFGO030ld/W+6/T6Pwg3b0QKb7RBNnTs8EFGatyqFy7RMWgLgRB8TdVvtPpoXZogWGjysppaRyO1eOnaa37xYZOWYq33XVAqUAAXcYBUFqllkfafKMBGU0I8bdZMZAMmGajZwyWynAz1uueX1X+0myJJBTZvOONsJR/Qw9EhsUVoUEOLHtRu2RL//IisYM8mel4XrKfvyRB/BIyMy3SAUhGo/4xMaSo3TeCJX2BwdIz0piwZo8n9VhWjXaLWBUEZa/PO6qqyyN/qDOjYvdRIIxXKV2Vn5VEi5a5HSM1s4Dd5qR1o6bF3tbLx1oz4n0xkjPTYpPnpfR4MhLLgIShBCx/5TwQYvVv1mm/r/4E+uvH1a7XvVXAAAAAElFTkSuQmCC",
                 }
        self._icons = { name: base64_to_pixbuf(data) for name, data in icons.items() }


"""
This module contains the Page class, which is a container for objects.
"""



class Layer:
    """
    A layer is a container for objects.
    """
    def __init__(self):
        self.__objects = []
        self.__selection = SelectionObject(self.__objects)

    def objects(self, objects = None):
        """Return or set the list of objects on the layer."""
        if objects:
            self.__objects = objects
            self.__selection = SelectionObject(self.__objects)
        return self.__objects

    def objects_import(self, object_list):
        """Import objects from a dict"""
        self.objects([ Drawable.from_dict(d) for d in object_list ] or [ ])

    def selection(self):
        """Return the selection object."""
        return self.__selection

    def kill_object(self, obj):
        """Directly remove an object from the list of objects."""
        if obj in self.__objects:
            self.__objects.remove(obj)

    def export(self):
        """Exports the layer as a dict"""
        return [ obj.to_dict() for obj in self.__objects ]

class Page:
    """
    A page is a container for layers.

    It serves as an interface between layers and whatever wants to
    manipulate objects or selection on a layer by choosing the current
    layer and managing layers.
    """
    def __init__(self, prev = None, layers = None):
        self.__prev    = prev
        self.__next = None
        self.__layers = [ layers or Layer() ]
        self.__current_layer = 0
        self.__translate = None
        self.__drawer = Drawer()

    def next(self, create = True):
        """
        Return the next page.

        If create is True, create a new page if it doesn't exist.
        """
        if not self.__next and create:
            print("Creating new page")
            self.__next = Page(self)
        return self.__next

    def prev(self):
        """Return the previous page."""
        return self.__prev or self # can't go beyond first page

    def next_set(self, page):
        """Set the next page."""
        self.__next = page

    def prev_set(self, page):
        """Set the previous page."""
        self.__prev = page

    def delete(self):
        """Delete the page and create links between prev and next pages."""
        if not self.__prev and not self.__next:
            print("only one page remaining")
            return self

        cmd = DeletePageCommand(self, self.__prev, self.__next)
        ret = self.__prev or self.__next

        return ret, cmd

    def objects(self, objects = None):
        """Return or set the list of objects on the page."""
        layer = self.__layers[self.__current_layer]
        return layer.objects(objects)

    def objects_all_layers(self):
        """Return all objects on all layers."""
        objects = [ obj for layer in self.__layers for obj in layer.objects() ]
        return objects

    def selection(self):
        """Return the selection object."""
        layer = self.__layers[self.__current_layer]
        return layer.selection()

    def kill_object(self, obj):
        """Directly remove an object from the list of objects."""
        layer = self.__layers[self.__current_layer]
        layer.kill_object(obj)

    def number_of_layers(self):
        """Return the number of layers."""
        return len(self.__layers)

    def next_layer(self):
        """Switch to the next layer."""
        print("appending a new layer")
        self.__current_layer += 1
        if self.__current_layer == len(self.__layers):
            self.__layers.append(Layer())
        self.__layers[self.__current_layer].selection().all()
        return self.__current_layer

    def prev_layer(self):
        """Switch to the previous layer."""
        self.__current_layer = max(0, self.__current_layer - 1)
        self.__layers[self.__current_layer].selection().all()
        return self.__current_layer

    def layer(self, new_layer = None, pos = None):
        """
        Get or insert the current layer.

        Arguments:
        new_layer -- if not None, insert new_layer and set
                     the current layer to new_layer.
        pos -- if not None, insert a new layer at pos.
        """
        if new_layer is not None:
            if pos is not None and pos < len(self.__layers):
                self.__layers.insert(pos, new_layer)
                self.__current_layer = pos
            else:
                self.__layers.append(new_layer)
                self.__current_layer = len(self.__layers) - 1

        return self.__layers[self.__current_layer]

    def layer_no(self, layer_no = None):
        """Get or set the current layer."""
        if layer_no is None:
            return self.__current_layer

        layer_no = max(0, layer_no)

        if layer_no >= len(self.__layers):
            self.__layers.append(Layer())

        self.__current_layer = layer_no
        return self.__current_layer

    def delete_layer(self, layer_no = None):
        """Delete the current layer."""

        if len(self.__layers) == 1:
            return None

        if layer_no is None or layer_no < 0 or layer_no >= len(self.__layers):
            layer_no = self.__current_layer

        layer = self.__layers[layer_no]
        pos   = layer_no

        del self.__layers[layer_no]

        self.__current_layer = max(0, layer_no - 1)
        self.__current_layer = min(self.__current_layer,
                                   len(self.__layers) - 1)

        cmd = DeleteLayerCommand(self, layer, pos)

        return cmd

    def translate(self, new_val = None):
        """Get or set the translate"""
        if new_val is not None:
            self.__translate = new_val
        return self.__translate

    def translate_set(self, new_val):
        """Set the translate"""
        self.__translate = new_val
        return self.__translate

    def export(self):
        """Exports the page with all layers as a dict"""
        layers = [ l.export() for l in self.__layers ]
        ret = {
                 "layers": layers,
                 "translate": self.translate(),
                 "cur_layer": self.__current_layer
              }
        return ret

    def import_page(self, page_dict):
        """Imports a dict to self"""
        print("importing pages")
        self.__translate = page_dict.get("translate")
        if "objects" in page_dict:
            self.objects(page_dict["objects"])
        elif "layers" in page_dict:
            print(len(page_dict["layers"]), "layers found")
            print("however, we only have", len(self.__layers), "layers")
            print("creating", len(page_dict["layers"]) - len(self.__layers), "new layers")
            self.__current_layer = 0
            for _ in range(len(page_dict["layers"]) - len(self.__layers)):
                self.next_layer()
            self.__current_layer = 0
            for l_list in page_dict["layers"]:
                layer = self.__layers[self.__current_layer]
                layer.objects_import(l_list)
                self.__current_layer += 1

        cl = page_dict.get("cur_layer")
        self.__current_layer = cl if cl is not None else 0

    def clear(self):
        """
        Remove all objects from all layers.

        Returns a CommandGroup object that can be used to undo the
        operation.
        """
        ret_commands = []
        for layer in self.__layers:
            cmd = RemoveCommand(layer.objects()[:], layer.objects(), page = self)
            ret_commands.append(cmd)
        return CommandGroup(ret_commands[::-1])

    def draw(self, cr, state):
        """Draw the objects on the page."""

        self.__drawer.draw(cr, self, state)
"""
Canvas for drawing shapes and text.

This is simpler then you might think. This is just a container for
background color, transparency, current pen and public methods to get or
set them.

The actual objects are managed by the GOM, graphical object manager.
The drawing on screen is realized mainly through DM, Draw Manager that also
holds information about other stuff that needs to be drawn, like the
currently selected object, wiglets etc.
"""


class Canvas:
    """
    Canvas for drawing shapes and text.
    """
    def __init__(self, state, dm, wiglets):
        self.__state = state
        self.__grid = Grid()
        self.__dm = dm
        self.__wiglets = wiglets

    def on_draw(self, widget, cr):
        """Main draw method of the whole app."""
        if self.__state.hidden():
            return
        page = self.__state.current_page()
        tr = page.translate()

        cr.save()

        if tr:
            cr.translate(*tr)

        self.draw_bg(cr, tr)
        page.draw(cr, self.__state)

        cobj = self.__state.current_obj()
        if cobj and not cobj in page.objects_all_layers():
            self.__state.current_obj().draw(cr)

        if self.__dm.selection_tool():
            self.__dm.selection_tool().draw(cr)

        cr.restore()

        #self.__dm.draw(None, cr)

        if self.__state.show_wiglets():
            for w in self.__wiglets:
                w.update_size(*self.__state.get_win_size())
                w.draw(cr)

       # XXX this does not work.
       #if self.__wiglet_active:
       #    self.__wiglet_active.draw(cr)

        return True



    def draw_bg(self, cr, tr):
        """
        Draw the objects on the page.

        :param objects: The objects to draw.
        :param cr: The context on which to draw.
        :param tr: The translation (paning).
        """
        pass

        bg_color     = self.__state.bg_color()
        transparency = self.__state.alpha()
        show_grid    = self.__state.show_grid()
        size         = self.__state.get_win_size()

        cr.set_source_rgba(*bg_color, transparency)
        cr.set_operator(cairo.OPERATOR_SOURCE)
        cr.paint()
        cr.set_operator(cairo.OPERATOR_OVER)

        if show_grid:
            tr = tr or (0, 0)
            self.__grid.draw(cr, tr, size)
"""Class for different brushes."""


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

def calc_normal_outline2(coords, pressure, line_width, rounded = False):
    """
    Calculate the normal outline of a path.

    This one is used in the pencil brush.
    """
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

        outline_l.append(point_mean(l_seg1_e, l_seg2_s))
        outline_r.append(point_mean(r_seg1_e, r_seg2_s))

        if i == n - 3:
            outline_l.append(l_seg2_e)
            outline_r.append(r_seg2_e)
            if rounded:
                arc_coords = calc_arc_coords(l_seg2_e, r_seg2_e, p1, 10)
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

        outline_l, outline_r = calc_normal_outline2(coords, pressure, lwd, False)
       #print("outline lengths:", len(outline_l), len(outline_r))

        self.__outline_l = outline_l
        self.__outline_r = outline_r
        self.__pressure  = pressure
        self.__coords    = coords
        self.outline(outline_l + outline_r[::-1])

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
"""Grid class for drawing a grid on screen"""

class Grid:
    """
    Grid object holds information about how tight a grid is, and how it is drawn.

    app is a necessary argument, because Grid needs to know the current size of 
    the screen to draw the grid properly.
    """
    def __init__(self):
        self.__spacing = 50
        self.__small_ticks = 5
        self.__color = (.2, .2, .2, .75)
        self.__line_width = 0.2
        self.__cache = None
        self.__state = [ (0, 0), (100, 100) ]

    def __cache_new(self, tr, size):
        """Cache the grid for the current size"""

        x, y = tr
        width, height = size

        surface = cairo.ImageSurface(cairo.Format.ARGB32, int(width) + 1, int(height) + 1)
        cr = cairo.Context(surface)
        cr.translate(x, y)
        self.__cache = {
                "surface": surface,
                "cr": cr,
                "x": x,
                "y": y,
                }
        self.__draw(cr, tr, size)

    def draw(self, cr, tr, size):
        """Draw the grid on the screen"""

        if self.__cache is None or self.__state != [tr, size]:
            self.__cache_new(tr, size)
            self.__state = [tr, size]

        cr.set_source_surface(self.__cache["surface"], 
                              -self.__cache["x"], 
                              -self.__cache["y"])
        cr.paint()

    def __draw(self, cr, tr, size):
        """Draw grid in the current cairo context"""

        width, height = size
        dx, dy = tr
        ticks = self.__small_ticks

        x0 =  - int(dx / self.__spacing) * self.__spacing
        y0 =  - int(dy / self.__spacing) * self.__spacing

        cr.set_source_rgba(*self.__color)
        cr.set_line_width(self.__line_width/2)

        # draw vertical lines
        x = x0
        i = 1
        while x < width + x0:
            if i == ticks:
                cr.set_line_width(self.__line_width)
                cr.move_to(x, y0)
                cr.line_to(x, height + y0)
                cr.stroke()
                cr.set_line_width(self.__line_width/2)
                i = 1
            else:
                cr.move_to(x, y0)
                cr.line_to(x, height + y0)
                cr.stroke()
                i += 1
            x += self.__spacing / ticks

        # draw horizontal lines
        y = y0
        while y < height + y0:
            if i == ticks:
                cr.set_line_width(self.__line_width)
                cr.move_to(x0, y)
                cr.line_to(width + x0, y)
                cr.stroke()
                cr.set_line_width(self.__line_width/2)
                i = 1
            else:
                cr.move_to(x0, y)
                cr.line_to(width + x0, y)
                cr.stroke()
                i += 1
            y += self.__spacing / ticks
"""
Class for editing text.
"""

class TextEditor:
    """
    Class for editing text.
    """

    def __init__(self, text = ""):
        self.__cont = text.split("\n")
        self.__line = 0
        self.__caret_pos = 0

    def __backspace(self):
        """Remove the last character from the text."""
        cnt = self.__cont
        lno = self.__line
        cpos = self.__caret_pos

        if cpos > 0:
            cnt[lno] = cnt[lno][:cpos - 1] + cnt[lno][cpos:]
            self.__caret_pos -= 1
        elif lno > 0:
            self.__caret_pos = len(cnt[lno - 1])
            cnt[lno - 1] += cnt[lno]
            cnt.pop(lno)
            self.__line -= 1

    def __newline(self):
        """Add a newline to the text."""
        self.__cont.insert(self.__line + 1,
                            self.__cont[self.__line][self.__caret_pos:])
        self.__cont[self.__line] = self.__cont[self.__line][:self.__caret_pos]
        self.__line += 1
        self.__caret_pos = 0

    def __add_char(self, char):
        """Add a character to the text."""
        lno, cpos = self.__line, self.__caret_pos
        before_caret = self.__cont[lno][:cpos]
        after_caret  = self.__cont[lno][cpos:]
        self.__cont[lno] = before_caret + char + after_caret
        self.__caret_pos += 1

    def __move_end(self):
        """Move the caret to the end of the last line."""
        self.__line = len(self.__cont) - 1
        self.__caret_pos = len(self.__cont[self.__line])

    def __move_home(self):
        """Move the caret to the beginning of the first line."""
        self.__line = 0
        self.__caret_pos = 0

    def __move_right(self):
        """Move the caret to the right."""
        if self.__caret_pos < len(self.__cont[self.__line]):
            self.__caret_pos += 1
        elif self.__line < len(self.__cont) - 1:
            self.__line += 1
            self.__caret_pos = 0

    def __move_left(self):
        """Move the caret to the left."""
        if self.__caret_pos > 0:
            self.__caret_pos -= 1
        elif self.__line > 0:
            self.__line -= 1
            self.__caret_pos = len(self.__cont[self.__line])

    def __move_down(self):
        """Move the caret down."""
        if self.__line < len(self.__cont) - 1:
            self.__line += 1
            self.__caret_pos = min(self.__caret_pos, len(self.__cont[self.__line]))

    def __move_up(self):
        """Move the caret up."""
        if self.__line > 0:
            self.__line -= 1
            self.__caret_pos = min(self.__caret_pos, len(self.__cont[self.__line]))

    def move_caret(self, direction):
        """Move the caret in the text."""
        { "End":   self.__move_end,
          "Home":  self.__move_home,
          "Right": self.__move_right,
          "Left":  self.__move_left,
          "Down":  self.__move_down,
          "Up":    self.__move_up }[direction]()

    def to_string(self):
        """Return the text as a string."""
        return "\n".join(self.__cont)

    def lines(self):
        """Return the text split by lines."""
        return self.__cont

    def strlen(self):
        """Return the length of the text."""
        return len(self.to_string())

    def caret_line(self, new_line = None):
        """Return the current line."""
        if new_line is not None:
            self.__line = new_line
        return self.__line

    def caret_pos(self, new_pos = None):
        """Return the caret position."""
        if new_pos is not None:
            self.__caret_pos = new_pos
        return self.__caret_pos

    def add_text(self, text):
        """Add text to the text."""
        # split text by newline
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if i == 0:
                self.__cont[self.__line] += line
                self.__caret_pos += len(text)
            else:
                self.__cont.insert(self.__line + i, line)
                self.__caret_pos = len(line)

    def update_by_key(self, keyname, char):
        """Update the text by key press."""
        if keyname == "BackSpace": # and cur["caret_pos"] > 0:
            self.__backspace()
        elif keyname in ["Home", "End", "Down", "Up", "Right", "Left"]:
            self.move_caret(keyname)
        elif keyname == "Return":
            self.__newline()
        elif char and char.isprintable():
            self.__add_char(char)
"""
Very simple class that holds the pixbuf along with some additional
information.
"""


class ImageObj:
    """Simple class to hold an image object."""
    def __init__(self, pixbuf, base64):

        if base64:
            self.__base64 = base64
            pixbuf = base64_to_pixbuf(base64)
        else:
            self.__base64 = None

        self.__pixbuf = pixbuf
        self.__size = (pixbuf.get_width(), pixbuf.get_height())

    def pixbuf(self):
        """Return the pixbuf."""
        return self.__pixbuf

    def size(self):
        """Return the size of the image."""
        return self.__size

    def encode_base64(self):
        """Encode the image to base64."""
        with tempfile.NamedTemporaryFile(delete = True) as temp:
            self.__pixbuf.savev(temp.name, "png", [], [])
            with open(temp.name, "rb") as f:
                image_base64 = base64.b64encode(f.read()).decode("utf-8")
        return image_base64

    def base64(self):
        """Return the base64 encoded image."""
        if self.__base64 is None:
            self.__base64 = self.encode_base64()
        return self.__base64


"""Status singleton class for holding key app information."""

class State:
    """Singleton class for holding key app information."""
    __new_instance = None

    def __new__(cls, *args, **kwargs):
        if not cls.__new_instance:
            cls.__new_instance = super(State, cls).__new__(cls)
        return cls.__new_instance

    def __init__(self, app, gom, cursor):
        self.__mode = 'draw'
        self.__app = app
        self.__gom = gom
        self.__cursor = cursor
        self.__modified = True

        self.__gr = {
                "bg_color": (.8, .75, .65),
                "transparency": 0,
                "outline": False,
                }

        self.__show = {
                "hidden": False,
                "grid": False,
                "wiglets": True,
                }

        self.__objs = {
                "hover": None,
                "current": None,
                "resize": None,
                }

        self.__pens = [ Pen(line_width = 4,  color = (0.2, 0, 0),
                            font_size = 24, transparency  = 1),
                        Pen(line_width = 40, color = (1, 1, 0),
                            font_size = 24, transparency = .2) ]

    # -------------------------------------------------------------------------
    def current_obj(self, obj = None):
        """Get or set the current object."""
        if obj:
            self.__objs["current"] = obj
        return self.__objs["current"]

    def current_obj_clear(self):
        """Clear the current object."""
        self.__objs["current"] = None

    def hover_obj(self, obj = None):
        """Get or set the hover object."""
        if obj:
            self.__objs["hover"] = obj
        return self.__objs["hover"]

    def hover_obj_clear(self):
        """Clear the hover object."""
        self.__objs["hover"] = None

    def current_page(self):
        """Get the current page object from gom."""
        return self.__gom.page()

    def modified(self, value = None):
        """Get or set the modified flag."""
        if value is not None:
            self.__modified = value
        return self.__modified

    # -------------------------------------------------------------------------
    def mode(self, mode = None):
        """Get or set the cursor mode."""
        if mode:
            self.__mode = mode
            self.__cursor.default(mode)
        return self.__mode

    # -------------------------------------------------------------------------
    def cursor(self):
        """expose the cursor manager."""
        return self.__cursor

    # -------------------------------------------------------------------------
    def show_grid(self):
        """What is the show grid status."""
        return self.__show["grid"]

    def toggle_grid(self):
        """Toggle the grid."""
        self.__show["grid"] = not self.__show["grid"]

    def show_wiglets(self, value = None):
        """Show or hide the wiglets."""
        if value is not None:
            self.__show["wiglets"] = value
        return self.__show["wiglets"]

    def toggle_wiglets(self):
        """Toggle the wiglets."""
        self.__show["wiglets"] = not self.__show["wiglets"]

    # -------------------------------------------------------------------------
    def __pen_set(self, pen, alternate = False):
        """Set the pen."""
        if alternate:
            self.__pens[1] = pen
        else:
            self.__pens[0] = pen

    def pen(self, alternate = False, pen = None):
        """Get or set the pen."""
        if pen:
            self.__pen_set(pen, alternate)
        return self.__pens[1] if alternate else self.__pens[0]

    def switch_pens(self):
        """Switch between pens."""
        self.__pens = [self.__pens[1], self.__pens[0]]

    def apply_pen_to_bg(self):
        """Apply the pen to the background."""
        self.__gr["bg_color"] = self.__pens[0].color
    # -------------------------------------------------------------------------

    def cycle_background(self):
        """Cycle through background transparency."""
        self.__gr["transparency"] = {1: 0, 0: 0.5, 0.5: 1}[self.__gr["transparency"]]

    def alpha(self, value=None):
        """Get or set the transparency."""
        if value:
            self.__gr["transparency"] = value
        return self.__gr["transparency"]

    def outline(self):
        """Get the outline mode."""
        return self.__gr["outline"]

    def outline_toggle(self):
        """Toggle outline mode."""
        self.__gr["outline"] = not self.__gr["outline"]

    def bg_color(self, color=None):
        """Get or set the background color."""
        if color:
            self.__gr["bg_color"] = color
        return self.__gr["bg_color"]

    # -------------------------------------------------------------------------
    def get_win_size(self):
        """Get the window size."""
        return self.__app.get_size()

    def queue_draw(self):
        """Queue a draw."""
        self.__app.queue_draw()

    def hidden(self, value = None):
        """Hide or show the drawing."""
        if not value is None:
            self.__show["hidden"] = value
        return self.__show["hidden"]

# -----------------------------------------------------------------------------
class Setter:
    """
    Class for setting the state.


    The purpose is to pack a bunch of setter methods into a single class
    so that the state class doesn't get too cluttered.
    """
    __new_instance = None

    def __new__(cls, *args, **kwargs):
        if not cls.__new_instance:
            cls.__new_instance = super(Setter, cls).__new__(cls)
        return cls.__new_instance

    def __init__(self, app, gom, cursor):
        self.__state = State(app, gom, cursor)
        self.__app = app
        self.__gom = gom
        self.__cursor = cursor


    # -------------------------------------------------------------------------
    def set_font(self, font_description):
        """Set the font."""
        self.__state.pen().font_set_from_description(font_description)
        self.__gom.selection_font_set(font_description)

        obj = self.__state.current_obj()
        if obj and obj.type == "text":
            obj.pen.font_set_from_description(font_description)

    def set_brush(self, brush = None):
        """Set the brush."""
        if brush is not None:
            print("setting pen", self.__state.pen(), "brush to", brush)
            self.__state.pen().brush_type(brush)
        return self.__state.pen().brush_type()

    def set_color(self, color = None):
        """Get or set the color."""
        if color is None:
            return self.__state.pen().color
        self.__state.pen().color_set(color)
        self.__gom.selection_color_set(color)
        return color

    def stroke_change(self, direction):
        """Modify the line width or text size."""
        print("Changing stroke", direction)
        cobj = self.__state.current_obj()
        if cobj and cobj.type == "text":
            print("Changing text size")
            cobj.stroke_change(direction)
            self.__state.pen().font_size = cobj.pen.font_size
        else:
            for obj in self.__gom.selected_objects():
                obj.stroke_change(direction)

        # without a selected object, change the default pen, but only if in the correct mode
        if self.__state.mode() == "draw":
            self.__state.pen().line_width = max(1, self.__state.pen().line_width + direction)
        elif self.__state.mode() == "text":
            self.__state.pen().font_size = max(1, self.__state.pen().font_size + direction)

    # ---------------------------------------------------------------------
    def finish_text_input(self):
        """Clean up current text and finish text input."""
        print("finishing text input")
        obj = self.__state.current_obj()

        if obj and obj.type == "text":
            obj.show_caret(False)

            if obj.strlen() == 0:
                print("kill object because empty")
                self.__gom.kill_object(obj)

            self.__state.current_obj_clear()
        self.__cursor.revert()


    # ---------------------------------------------------------------------
    def clear(self):
        """Clear the drawing."""
        self.__gom.selection().clear()
        #self.__resizeobj      = None
        self.__state.current_obj_clear()
        self.__gom.remove_all()
        self.__state.queue_draw()
"""
These are the objects that can be displayed. It includes groups, but
also primitives like boxes, paths and text.
"""



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

    # ------------ Drawable attribute methods ------------------
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
        if obj_type not in type_map:
            raise ValueError("Invalid type:", obj_type)

        if "pen" in d:
            d["pen"] = Pen.from_dict(d["pen"])

        return type_map.get(obj_type)(**d)
"""Class which draws the actual objects and caches them."""

def draw_on_surface(cr, objects, selection, state):
    """
    Draw the objects on the given graphical context.
    """

    for obj in objects:

        hover    = obj == state.hover_obj() and state.mode() == "move"
        selected = selection.contains(obj) and state.mode() == "move"

        obj.draw(cr, hover=hover,
                 selected=selected,
                 outline = state.outline())

def obj_status(obj, selection, state):
    """Calculate the status of an object."""

    hover_obj = state.hover_obj()
    hover    = obj == hover_obj and state.mode() == "move"
    selected = selection.contains(obj) and state.mode() == "move"
    is_cur_obj = obj == state.current_obj()

    return (obj.mod, hover, selected, is_cur_obj)

def create_cache_surface(objects):
    """
    Create a cache surface.

    :param objects: The objects to cache.
    """

    if not objects:
        return None

    grp = DrawableGroup(objects)
    bb  = grp.bbox(actual = True)

    if not bb:
        return None

    # create a surface that fits the bounding box of the objects
    x, y, width, height = bb
    surface = cairo.ImageSurface(cairo.Format.ARGB32, int(width) + 1, int(height) + 1)
    cr = cairo.Context(surface)
    cr.translate(-x, -y)
    ret = {
            "surface": surface,
            "cr": cr,
            "x": x,
            "y": y,
            }
    return ret



class Drawer:
    """Singleton Class which draws the actual objects and caches them."""
    __new_instance = None

    def __new__(cls, *args, **kwargs):
        if cls.__new_instance is None:
            cls.__new_instance = super(Drawer, cls).__new__(cls)
        return cls.__new_instance

    def __init__(self):
        self.__cache        = None
        self.__obj_mod_hash = { }

    def new_cache(self, groups, selection, state):
        """
        Generate the new cache when the objects have changed.

        :param groups: The groups of objects to cache.
        """

        self.__cache = { "groups": groups,
                         "surfaces": [ ],}

        cur = groups["first_is_same"]

        # for each non-empty group of objects that remained the same,
        # generate a cache surface and draw the group on it
        for obj_grp in groups["groups"]:
            if not cur or not obj_grp:
                cur = not cur
                continue

            surface = create_cache_surface(obj_grp)
            self.__cache["surfaces"].append(surface)
            cr = surface["cr"]
            draw_on_surface(cr, obj_grp, selection, state)
            cur = not cur

    def update_cache(self, objects, selection, state):
        """
        Update the cache.

        :param objects: The objects to update.
        :param selection: The selection.
        :param state: The state.
        """

        groups = self.__find_groups(objects, selection, state)

        if not self.__cache or self.__cache["groups"] != groups:
            self.new_cache(groups, selection, state)

    def __find_groups(self, objects, selection, state):
        """
        Method to detect which objects changed from the previous time.
        These objects are then split into groups separated by objects that
        did change, so when drawing, the stacking order is maintained
        despite cacheing.

        :param objects: The objects to find groups for.
        :param selection: The selection object to determine whether an
                           object is selected (selected state is drawn
                           differently, so that counts as a change).
        :param state: The state object, holding information about the
                      drawing mode and hover object.
        
        Two values are returned. First, a list of groups, alternating
        between groups that have changed and groups that haven't. Second,
        a boolean indicating whether the first group contains objects
        that have changed.
        """
        modhash = self.__obj_mod_hash

        cur_grp       = [ ]
        groups        = [ cur_grp ]
        first_is_same = None

        is_same = True
        prev    = None

        # The goal of this method is to ensure correct stacking order
        # of the drawn active objects and cached groups.
        for obj in objects:
            status = obj_status(obj, selection, state)

            is_same = obj in modhash and modhash[obj] == status and not status[3]

            if first_is_same is None:
                first_is_same = is_same

            # previous group type was different
            if prev is not None and prev != is_same:
                cur_grp = [ ]
                groups.append(cur_grp)

            cur_grp.append(obj)

            modhash[obj] = status
            prev = is_same

        ret = { "groups": groups, "first_is_same": first_is_same }
        return ret

    def draw_cache(self, cr, selection, state):
        """
        Process the cache. Draw the cached objects as surfaces and the rest
        normally.
        """

        is_same = self.__cache["groups"]["first_is_same"]
        i = 0

        for obj_grp in self.__cache["groups"]["groups"]:

            # ignore empty groups (that might happen in edge cases)
            if not obj_grp:
                is_same = not is_same
                continue

            # objects in this group have changed: draw it normally on the surface
            if not is_same:
                draw_on_surface(cr, obj_grp, selection, state)
                is_same = not is_same
                continue

            # objects in this group remained the same: draw the cached surface
            if is_same:
                surface = self.__cache["surfaces"][i]
                cr.set_source_surface(surface["surface"], surface["x"], surface["y"])
                cr.paint()
                i += 1
                is_same = not is_same

    def draw(self, cr, page, state):
        """
        Draw the objects on the page.

        :param objects: The objects to draw.
        :param cr: The context on which to draw.
        :param page: The page object.
        :param state: The state object.
        """

        # extract objects from the page provided
        objects = page.objects_all_layers()

        # extract the selection object that we need
        # to determine whether an object in selected state
        selection = page.selection()

        # check if the cache needs to be updated
        self.update_cache(objects, selection, state)

        # draw the cache
        self.draw_cache(cr, selection, state)

        return True
"""Factory for drawable objects"""


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
"""
Classes that represent groups of drawable objects.
"""


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

        #print("Creating DrawableGroup with ", len(objects), "objects")
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
        self.mod += 1

    def stroke_change(self, direction):
        """Change the stroke size of the objects in the group."""
        for obj in self.objects:
            obj.stroke_change(direction)
        self.mod += 1

    def transmute_to(self, mode):
        """Transmute all objects within the group to a new type."""
        print("transmuting group to", mode)
       #for i in range(len(self.objects)):
       #    self.objects[i] = DrawableFactory.transmute(self.objects[i], mode)
        self.mod += 1

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
        self.mod += 1

    def get_primitive(self):
        """Return the primitives of the objects in the group."""
        primitives = [ obj.get_primitive() for obj in self.objects ]
        return flatten_and_unique(primitives)

    def rotate_start(self, origin):
        """Start the rotation operation."""
        self.rot_origin = origin
        for obj in self.objects:
            obj.rotate_start(origin)
        self.mod += 1

    def rotate(self, angle, set_angle = False):
        """Rotate the objects in the group."""
        if set_angle:
            self.rotation = angle
        else:
            self.rotation += angle
        for obj in self.objects:
            obj.rotate(angle, set_angle)
        self.mod += 1

    def rotate_end(self):
        """Finish the rotation operation."""
        for obj in self.objects:
            obj.rotate_end()
        self.rot_origin = None
        self.rotation = 0
        self.mod += 1

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
        self.mod += 1

    def resize_end(self):
        """Finish the resizing operation."""
        self.resizing = None
        for obj in self.objects:
            obj.resize_end()
        self.mod += 1

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
        self.mod += 1

    def remove(self, obj):
        """Remove an object from the group."""
        self.objects.remove(obj)
        self.mod += 1

    def move(self, dx, dy):
        """Move the group by dx, dy."""
        for obj in self.objects:
            obj.move(dx, dy)
        self.mod += 1

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

Drawable.register_type("group", DrawableGroup)
"""
These classes are the primitives for drawing: text, shapes, paths
"""



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
        cr.set_source_rgba(*self.pen.color, self.pen.transparency)

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
"""
These classes are the primitives for drawing: text, shapes, paths
"""



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
        cr.set_source_rgba(*self.pen.color, self.pen.transparency)

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
"""Path is a drawable that is like a shape, but not closed and has an outline"""



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

        if outline:
            print("Warning: outline is not used in Path")

        if len(self.coords) > 3 and not self.__brush.outline():
            self.outline_recalculate()

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
        #print("drawing path", self, "with brush", self.__brush, "of type",
        # self.__brush.brush_type())
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

Drawable.register_type("path", Path)


# ---------------------------------------------------------------------
# defaults


COLORS = {
        "black": (0, 0, 0),
        "white": (1, 1, 1),
        "red": (.7, 0, 0),
        "green": (0, .7, 0),
        "blue": (0, 0, .5),
        "yellow": (1, 1, 0),
        "cyan": (0, 1, 1),
        "magenta": (1, 0, 1),
        "purple": (0.5, 0, 0.5),
        "grey": (0.5, 0.5, 0.5)
}

AUTOSAVE_INTERVAL = 30 * 1000 # 30 seconds




## ---------------------------------------------------------------------

class TransparentWindow(Gtk.Window):
    """Main app window. Holds all information and everything that exists.
       One window to rule them all."""

    def __init__(self, save_file = None):
        super().__init__()

        self.savefile            = save_file
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        self.set_title("Transparent Drawing Window")
        self.set_decorated(False)
        self.connect("destroy", self.exit)
        self.set_default_size(800, 800)
        self.set_keep_above(True)
        self.maximize()


        # transparency
        screen = self.get_screen()
        visual = screen.get_rgba_visual()
        if visual is not None and screen.is_composited():
            self.set_visual(visual)
        self.set_app_paintable(True)

        # autosave
        GLib.timeout_add(AUTOSAVE_INTERVAL, self.__autosave)

        # Drawing setup
        self.cursor             = CursorManager(self)
        self.gom                = GraphicsObjectManager()

        # we pass the app to the state, because it has the queue_draw
        # method
        self.state              = State(app = self, 
                                        gom = self.gom,
                                        cursor = self.cursor)

        self.clipboard           = Clipboard()

        self.setter = Setter(app = self, gom = self.gom, cursor = self.cursor)

        wiglets = [ WigletColorSelector(height = self.state.get_win_size()[1],
                                               func_color = self.setter.set_color,
                                               func_bg = self.state.bg_color),
                           WigletToolSelector(func_mode = self.state.mode),
                           WigletPageSelector(gom = self.gom, screen_wh_func = self.state.get_win_size,
                                              set_page_func = self.gom.set_page_number),
                           WigletColorPicker(func_color = self.setter.set_color, clipboard = self.clipboard),
                          ]


        # em has to know about all that to link actions to methods
        em  = EventManager(gom = self.gom, app = self,
                                state  = self.state,
                                setter = self.setter)
        mm  = MenuMaker(em, self)

        # dm needs to know about gom because gom manipulates the selection
        # and history (object creation etc)
        self.dm                  = DrawManager(gom = self.gom, mm = mm,
                                               state = self.state, wiglets = wiglets,
                                               setter = self.setter)

        # canvas orchestrates the drawing
        self.canvas              = Canvas(state = self.state, dm = self.dm,
                                          wiglets = wiglets)

        # distance for selecting objects
        self.max_dist   = 15

        # load the drawing from the savefile
        self.load_state()

        # connecting events

       #XXX doesn't work
       #self.gesture_pan = Gtk.GesturePan.new(self, orientation=Gtk.Orientation.VERTICAL)
       #self.gesture_pan.connect('pan', self.dm.on_pan)
       #self.gesture_pan.set_propagation_phase(Gtk.PropagationPhase.CAPTURE)

       ## Gesture for zoom
       #self.gesture_zoom = Gtk.GestureZoom.new(self)
       #self.gesture_zoom.connect('begin', self.dm.on_zoom)
       #self.gesture_zoom.set_propagation_phase(Gtk.PropagationPhase.CAPTURE)

        self.set_events(Gdk.EventMask.BUTTON_PRESS_MASK |
                        Gdk.EventMask.BUTTON_RELEASE_MASK |
                        Gdk.EventMask.POINTER_MOTION_MASK |
                        Gdk.EventMask.TOUCH_MASK)

        self.connect("key-press-event",      em.on_key_press)
        self.connect("draw",                 self.canvas.on_draw)
        self.connect("button-press-event",   self.dm.on_button_press)
        self.connect("button-release-event", self.dm.on_button_release)
        self.connect("motion-notify-event",  self.dm.on_motion_notify)

    def exit(self):
        """Exit the application."""
        ## close the savefile_f
        print("Exiting")
        self.__save_state()
        Gtk.main_quit()

    # ---------------------------------------------------------------------

    def paste_text(self, clip_text):
        """Enter some text in the current object or create a new object."""

        cobj = self.state.current_obj()
        if cobj and cobj.type == "text":
            cobj.add_text(clip_text.strip())
        else:
            new_text = Text([ self.cursor.pos() ],
                            pen = self.state.pen(), content=clip_text.strip())
            self.gom.add_object(new_text)

    def paste_image(self, clip_img):
        """Create an image object from a pixbuf image."""
        obj = Image([ self.cursor.pos() ], self.state.pen(), clip_img)
        self.gom.add_object(obj)

    def __object_create_copy(self, obj, bb = None):
        """Copy the given object into a new object."""
        new_obj = copy.deepcopy(obj.to_dict())
        new_obj = Drawable.from_dict(new_obj)

        # move the new object to the current location
        x, y = self.cursor.pos()
        if bb is None:
            bb  = new_obj.bbox()
        new_obj.move(x - bb[0], y - bb[1])

        self.gom.add_object(new_obj)

    def copy_content(self, destroy = False):
        """Copy content to clipboard."""
        content = self.gom.selection()
        if content.is_empty():
            # nothing selected
            print("Nothing selected, selecting all objects")
            content = DrawableGroup(self.gom.objects())

        print("Copying content", content)
        self.clipboard.copy_content(content)

        if destroy:
            self.gom.remove_selection()

    def paste_content(self):
        """Paste content from clipboard."""
        clip_type, clip = self.clipboard.get_content()

        if not clip:
            return

        # internal paste
        if clip_type == "internal":
            print("Pasting content internally")
            if clip.type != "group":
                raise ValueError("Internal clipboard is not a group")
            bb = clip.bbox()
            print("clipboard bbox:", bb)
            for obj in clip.objects:
                self.__object_create_copy(obj, bb)
            return

        if clip_type == "text":
            self.paste_text(clip)
        elif clip_type == "image":
            self.paste_image(clip)

    def cut_content(self):
        """Cut content to clipboard."""
        self.copy_content(True)

    def select_color_bg(self):
        """Select a color for the background using ColorChooser."""
        color = ColorChooser(self, "Select Background Color")
        if color:
            self.state.bg_color((color.red, color.green, color.blue))

    def select_color(self):
        """Select a color for drawing using ColorChooser dialog."""
        color = ColorChooser(self)
        if color:
            self.setter.set_color((color.red, color.green, color.blue))

    def select_font(self):
        """Select a font for drawing using FontChooser dialog."""
        font_description = FontChooser(self.state.pen(), self)

        if font_description:
            self.setter.set_font(font_description)

    def show_help_dialog(self):
        """Show the help dialog."""
        dialog = help_dialog(self)
        dialog.run()
        dialog.destroy()

    def save_drawing_as(self):
        """Save the drawing to a file."""
        print("opening save file dialog")
        file = save_dialog(self)
        if file:
            self.savefile = file
            print("setting savefile to", file)
            self.__save_state()

    def export_drawing(self):
        """Save the drawing to a file."""
        # Choose where to save the file
        #    self.export(filename, "svg")
        bbox = None
        if self.gom.selected_objects():
            # only export the selected objects
            obj = self.gom.selected_objects()
        else:
            # set bbox so we export the whole screen
            obj = self.gom.objects()
            bbox = (0, 0, *self.get_size())

        if not obj:
            print("Nothing to draw")
            return

        obj = DrawableGroup(obj)
        file_name, file_format = export_dialog(self)

        if file_name:
            export_image(obj, file_name, file_format,
                         bg = self.state.bg_color(),
                         bbox = bbox, transparency = self.state.alpha())

    def select_image_and_create_pixbuf(self):
        """Select an image file and create a pixbuf from it."""

        image_file = import_image_dialog(self)
        pixbuf = None

        if image_file:
            try:
                pixbuf = GdkPixbuf.Pixbuf.new_from_file(image_file)
                print(f"Loaded image: {image_file}")
            except Exception as e:
                print(f"Failed to load image: {e}")

            if pixbuf is not None:
                pos = self.cursor.pos()
                img = Image([ pos ], self.state.pen(), pixbuf)
                self.gom.add_object(img)
                self.queue_draw()

        return pixbuf

    def __screenshot_finalize(self, bb):
        """Finish up the screenshot."""
        print("Taking screenshot now")
        frame = (bb[0] - 3, bb[1] - 3, bb[0] + bb[2] + 6, bb[1] + bb[3] + 6)
        pixbuf, filename = get_screenshot(self, *frame)
        self.state.hidden(False)
        self.queue_draw()

        # Create the image and copy the file name to clipboard
        if pixbuf is not None:
            img = Image([ (bb[0], bb[1]) ], self.state.pen(), pixbuf)
            self.gom.add_object(img)
            self.queue_draw()
            self.clipboard.set_text(filename)

    def __find_screenshot_box(self):
        """Find a box suitable for selecting a screenshot."""
        cobj = self.state.current_obj()
        if cobj and cobj.type == "rectangle":
            return cobj
        for obj in self.gom.selected_objects():
            if obj.type == "rectangle":
                return obj
        for obj in self.gom.objects()[::-1]:
            if obj.type == "rectangle":
                return obj
        return None

    def screenshot(self):
        """Take a screenshot and add it to the drawing."""
        obj = self.__find_screenshot_box()
        if not obj:
            print("no suitable box found")
            # use the whole screen
            bb = (0, 0, *self.get_size())
        else:
            bb = obj.bbox()
            print("bbox is", bb)
        #self.hidden = True
        self.state.hidden(True)
        self.queue_draw()
        while Gtk.events_pending():
            Gtk.main_iteration_do(False)
        GLib.timeout_add(100, self.__screenshot_finalize, bb)

    def __autosave(self):
        """Autosave the drawing state."""
        if not self.state.modified():
            return

        if self.state.current_obj(): # not while drawing!
            return

        print("Autosaving")
        self.__save_state()
        self.state.modified(False)

    def __save_state(self):
        """Save the current drawing state to a file."""
        if not self.savefile:
            print("No savefile set")
            return

        print("savefile:", self.savefile)
        config = {
                'bg_color':    self.state.bg_color(),
                'transparent': self.state.alpha(),
                'show_wiglets': self.state.show_wiglets(),
                'bbox':        (0, 0, *self.get_size()),
                'pen':         self.state.pen().to_dict(),
                'pen2':        self.state.pen(alternate = True).to_dict(),
                'page':        self.gom.current_page_number()
        }

        #objects = self.gom.export_objects()
        pages   = self.gom.export_pages()

        save_file_as_sdrw(self.savefile, config, pages = pages)

    def open_drawing(self):
        """Open a drawing from a file in native format."""
        file_name = open_drawing_dialog(self)
        if self.read_file(file_name):
            print("Setting savefile to", file_name)
            self.savefile = file_name
            self.state.modified(True)

    def read_file(self, filename, load_config = True):
        """Read the drawing state from a file."""
        config, objects, pages = read_file_as_sdrw(filename)

        if pages:
            self.gom.set_pages(pages)
        elif objects:
            self.gom.set_objects(objects)

        if config and load_config:
            self.state.bg_color(config.get('bg_color') or (.8, .75, .65))
            self.state.alpha(config.get('transparent') or 0)
            show_wiglets = config.get('show_wiglets')
            if show_wiglets is None:
                show_wiglets = True
            self.state.show_wiglets(show_wiglets)
            self.state.pen(pen = Pen.from_dict(config['pen']))
            self.state.pen(pen = Pen.from_dict(config['pen2']), alternate = True)
            self.gom.set_page_number(config.get('page') or 0)
        if config or objects:
            self.state.modified(True)
            return True
        return False

    def load_state(self):
        """Load the drawing state from a file."""
        self.read_file(self.savefile)


## ---------------------------------------------------------------------

def main():
    """Main function for the application."""
    APP_NAME   = "ScreenDrawer"
    APP_AUTHOR = "JanuaryWeiner"  # Optional; used on Windows

    # Get user-specific config directory
    user_config_dir = appdirs.user_config_dir(APP_NAME, APP_AUTHOR)
    print(f"User config directory: {user_config_dir}")

    #user_cache_dir = appdirs.user_cache_dir(APP_NAME, APP_AUTHOR)
    #user_log_dir   = appdirs.user_log_dir(APP_NAME, APP_AUTHOR)


# ---------------------------------------------------------------------
# Parsing command line
# ---------------------------------------------------------------------

    parser = argparse.ArgumentParser(
            description="Drawing on the screen",
            epilog=f"Alternative use: {argv[0]} file.sdrw file.[png, pdf, svg]")
    parser.add_argument("-l", "--loadfile", help="Load drawing from file")
    parser.add_argument("-c", "--convert",
                        help="""
Convert screendrawer file to given format (png, pdf, svg) and exit
(use -o to specify output file, otherwise a default name is used)
""",
                        metavar = "FORMAT"
    )
    parser.add_argument("-p", "--page", help="Page to convert (default: 1)", 
                        type=int, default = 1)

    parser.add_argument("-b", "--border", 
                        help="""
                             Border width for conversion. If not
                             specified, whole page will be converted.
                             """, 
                        type=int)
    parser.add_argument("-o", "--output", help="Output file for conversion")
    parser.add_argument("files", nargs="*")
    args     = parser.parse_args()

    if args.convert:
        if not args.convert in [ "png", "pdf", "svg" ]:
            print("Invalid conversion format")
            exit(1)
        OUTPUT = None
        if args.output:
            OUTPUT = args.output

        if not args.files:
            print("No input file provided")
            exit(1)
        convert_file(args.files[0], 
                     OUTPUT, 
                     args.convert, 
                     border = args.border, 
                     page = args.page - 1)
        exit(0)

    if args.files:
        if len(args.files) > 2:
            print("Too many files provided")
            exit(1)
        elif len(args.files) == 2:
            convert_file(args.files[0], args.files[1], 
                         border = args.border, 
                         page = args.page - 1)
            exit(0)
        else:
            savefile = args.files[0]
    else:
        savefile = get_default_savefile(APP_NAME, APP_AUTHOR)
    print("Save file is:", savefile)

# ---------------------------------------------------------------------

    win = TransparentWindow(save_file = savefile)
    if args.loadfile:
        win.read_file(args.loadfile)

    CSS = b"""
    #myMenu {
        background-color: rgba(255, 255, 255, 0.9); /* Semi-transparent white */
        font-family: Monospace, 'Monospace regular', monospace, 'Courier New'; /* Use 'Courier New', falling back to any monospace font */
    }
    """

    style_provider = Gtk.CssProvider()
    style_provider.load_from_data(CSS)
    Gtk.StyleContext.add_provider_for_screen(
        Gdk.Screen.get_default(),
        style_provider,
        Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
    )

    win.show_all()
    win.present()
    win.cursor.set(win.state.mode())
    win.stick()

    Gtk.main()

if __name__ == "__main__":
    main()
