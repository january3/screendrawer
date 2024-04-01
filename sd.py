#!/usr/bin/env python3

##  MIT License
##
##  Copyright (c) 2024 January Weiner
##
##  Permission is hereby granted, free of charge, to any person obtaining a copy
##  of this software and associated documentation files (the "Software"), to deal
##  in the Software without restriction, including without limitation the rights
##  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
##  copies of the Software, and to permit persons to whom the Software is
##  furnished to do so, subject to the following conditions:
##
##  The above copyright notice and this permission notice shall be included in all
##  copies or substantial portions of the Software.
##
##  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
##  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
##  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
##  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
##  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
##  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
##  SOFTWARE.

# ---------------------------------------------------------------------
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
    """Calculate the intersection of two line segments."""
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
        raise ValueError("Pressure and coords must have the same length")

    print("smoothing path with", len(coords), "points")
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
                t0 = (1 - t) ** 3
                t1 = 3 * (1 - t) ** 2 * t
                t2 = 3 * (1 - t) * t ** 2
                t3 = t ** 3
                x = t0 * control1[0] + t1 * p1[0] + t2 * control2[0] + t3 * p2[0]
                y = t0 * control1[1] + t1 * p1[1] + t2 * control2[1] + t3 * p2[1]
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

def path_bbox(coords):
    """Calculate the bounding box of a path."""
    if not coords:
        return (0, 0, 0, 0)

    left, top = min(p[0] for p in coords), min(p[1] for p in coords)
    width  =    max(p[0] for p in coords) - left
    height =    max(p[1] for p in coords) - top
    return (left, top, width, height)

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
        self.__get_prop_func = get_prop_func
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
                 font_weight = "normal", font_style = "normal"):
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

    def transparency_set(self, transparency):
        """Set pen transparency"""
        self.transparency = transparency

    def fill_set(self, color):
        """Set fill color"""
        self.fill_color = color

    def color_set(self, color):
        """Set pen color"""
        self.color = color

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
            "font_style": self.font_style
        }

    def copy(self):
        """Create a copy of the pen"""
        return Pen(self.color, self.line_width, self.transparency, self.fill_color, self.font_size, self.font_family, self.font_weight, self.font_style)

    @classmethod
    def from_dict(cls, d):
        """Create a pen object from a dictionary"""
        #def __init__(self, color = (0, 0, 0), line_width = 12, font_size = 12, transparency = 1, fill_color = None, family = "Sans", weight = "normal", style = "normal"):
        return cls(d.get("color"), d.get("line_width"), d.get("transparency"), d.get("fill_color"),
                   d.get("font_size"), d.get("font_family"), d.get("font_weight"), d.get("font_style"))
"""
These are the objects that can be displayed. It includes groups, but
also primitives like boxes, paths and text.
"""



class DrawableFactory:
    """
    Factory class for creating drawable objects.
    """
    @classmethod
    def create_drawable(cls, mode, pen, ev):
        """
        Create a drawable object of the specified type.
        """
        shift, ctrl, pressure = ev.shift(), ev.ctrl(), ev.pressure()
        pos = ev.pos()
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

        elif mode == "box":
            print("creating box object")
            ret_obj = Box([ pos, (pos[0], pos[1]) ], pen = pen)

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
            # XXX for now, we do not pass transmutations to groups, because
            # we then cannot track the changes.
            return obj

        if mode == "text":
            obj = Text.from_object(obj)
        elif mode == "draw":
            obj = Path.from_object(obj)
        elif mode == "box":
            obj = Box.from_object(obj)
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
        self.type       = mytype
        self.coords     = coords
        self.origin     = None
        self.resizing   = None
        self.rotation   = 0
        self.rot_origin = None
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

    def unfill(self):
        """Remove the fill from the object."""
        self.pen.fill_set(None)

    def fill(self, color = None):
        """Fill the object with a color."""
        self.pen.fill_set(color)

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
    def origin_remove(self):
        """Remove the origin point."""
        self.origin = None

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

    def bbox(self):
        """Return the bounding box of the object."""
        if self.resizing:
            return self.resizing["bbox"]
        left, top = min(p[0] for p in self.coords), min(p[1] for p in self.coords)
        width =    max(p[0] for p in self.coords) - left
        height =   max(p[1] for p in self.coords) - top
        return (left, top, width, height)

    def bbox_draw(self, cr, bb=None, lw=0.2):
        """Draw the bounding box of the object."""
        if not bb:
            bb = self.bbox()
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
        """ Create a drawable object from a dictionary. """
        type_map = {
            "path": Path,
            "polygon": Shape, #back compatibility
            "shape": Shape,
            "circle": Circle,
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
    def __init__(self, objects = [ ], objects_dict = None, mytype = "group"):

        if objects_dict:
            objects = [ Drawable.from_dict(d) for d in objects_dict ]

        print("Creating DrawableGroup with ", len(objects), "objects")
        # XXX better if type would be "drawable_group"
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

    def color_set(self, color):
        """Set the color of the objects in the group."""
        for obj in self.objects:
            obj.color_set(color)

    def font_set(self, size, family, weight, style):
        """Set the font of the objects in the group."""
        for obj in self.objects:
            obj.font_set(size, family, weight, style)

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

    def bbox(self):
        """Return the bounding box of the group."""
        if self.resizing:
            return self.resizing["bbox"]
        if not self.objects:
            return None

        left, top, width, height = self.objects[0].bbox()
        bottom, right = top + height, left + width

        for obj in self.objects[1:]:
            x, y, w, h = obj.bbox()
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

    def toggle(self, obj):
        """Toggle the selection of an object."""
        if obj in self.objects:
            self.objects.remove(obj)
        else:
            self.objects.append(obj)

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

        if image_base64:
            self.image_base64 = image_base64
            image = base64_to_pixbuf(image_base64)
        else:
            self.image_base64 = None

        self.image_size = (image.get_width(), image.get_height())
        self.transform = transform or (1, 1)

        width  = self.image_size[0] * self.transform[0]
        height = self.image_size[1] * self.transform[1]

        coords = [ (coords[0][0], coords[0][1]),
                   (coords[0][0] + width, coords[0][1] + height) ]
        super().__init__("image", coords, pen)
        self.image = image
        self._orig_bbox = None

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

        Gdk.cairo_set_source_pixbuf(cr, self.image, 0, 0)
        cr.paint()

        cr.restore()

        cr.set_source_rgb(*self.pen.color)
        if selected:
            self.bbox_draw(cr, lw=1.5)
        if hover:
            self.bbox_draw(cr, lw=.5)

    def bbox(self):
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
        self._orig_bbox = self.bbox()
        self.resizing = {
            "corner": corner,
            "origin": origin,
            "bbox":   self.bbox(),
            "transform": self.transform
            }

    def resize_update(self, bbox):
        """Update during the resize of the object."""
        self.resizing["bbox"] = bbox
        coords = self.coords

        x1, y1, w1, h1 = bbox

        # calculate scale relative to the old bbox
        print("old bbox is", self._orig_bbox)
        print("new bbox is", bbox)

        w_scale = w1 / self._orig_bbox[2]
        h_scale = h1 / self._orig_bbox[3]

        print("resizing image", w_scale, h_scale)
        print("old transform is", self.resizing["transform"])

        self.coords[0] = (x1, y1)
        self.coords[1] = (x1 + w1, y1 + h1)
        self.transform = (w_scale * self.resizing["transform"][0],
                          h_scale * self.resizing["transform"][1])

    def resize_end(self):
        """Finish the resizing operation."""
        self.coords[1] = (self.coords[0][0] + self.image_size[0] * self.transform[0],
                          self.coords[0][1] + self.image_size[1] * self.transform[1])
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
        if click_x >= x and click_x <= x + width and click_y >= y and click_y <= y + height:
            return True
        return False

    def encode_base64(self):
        """Encode the image to base64."""
        with tempfile.NamedTemporaryFile(delete = True) as temp:
            self.image.savev(temp.name, "png", [], [])
            with open(temp.name, "rb") as f:
                image_base64 = base64.b64encode(f.read()).decode("utf-8")
        return image_base64

    def base64(self):
        """Return the base64 encoded image."""
        if self.image_base64 is None:
            self.image_base64 = self.encode_base64()
        return self.image_base64

    def to_dict(self):
        """Convert the object to a dictionary."""

        return {
            "type": self.type,
            "coords": self.coords,
            "pen": self.pen.to_dict(),
            "image": None,
            "rotation": self.rotation,
            "transform": self.transform,
            "image_base64": self.base64(),
        }


class Text(Drawable):
    """Class for Text objects"""
    def __init__(self, coords, pen, content, rotation = None, rot_origin = None):
        super().__init__("text", coords, pen)

        # split content by newline
        content = content.split("\n")
        self.content = content
        self.line    = 0
        self.caret_pos    = None
        self.bb           = None
        self.font_extents = None

        if rotation:
            self.rotation = rotation
            self.rot_origin = rot_origin

    def is_close_to_click(self, click_x, click_y, threshold):
        """Check if a click is close to the object."""
        if self.bb is None:
            return False
        x, y, width, height = self.bb
        if click_x >= x and click_x <= x + width and click_y >= y and click_y <= y + height:
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
        old_bbox   = self.bb

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
            current_bbox = self.bb
            print("drawn, bbox is", self.bb)
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
            "content": self.as_string()
        }

    def bbox(self):
        if self.resizing:
            return self.resizing["bbox"]
        if not self.bb:
            bb = (self.coords[0][0], self.coords[0][1], 50, 50)
        else:
            bb = self.bb
        if self.rotation:
            # first, calculate position bb after rotation relative to the
            # text origin
            x, y, w, h = bb
            x1, y1 = x + w, y + h
            bb = coords_rotate([(x, y), (x, y1), (x1, y), (x1, y1)], self.rotation, self.rot_origin)
            bb = path_bbox(bb)

        return bb

    def as_string(self):
        """Return the text as a single string."""
        return "\n".join(self.content)

    def strlen(self):
        """Return the length of the text."""
        return len(self.as_string())

    def add_text(self, text):
        """Add text to the object."""
        # split text by newline
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if i == 0:
                self.content[self.line] += line
                self.caret_pos += len(text)
            else:
                self.content.insert(self.line + i, line)
                self.caret_pos = len(line)

    def backspace(self):
        """Remove the last character from the text."""
        cnt = self.content
        if self.caret_pos > 0:
            cnt[self.line] = cnt[self.line][:self.caret_pos - 1] + cnt[self.line][self.caret_pos:]
            self.caret_pos -= 1
        elif self.line > 0:
            self.caret_pos = len(cnt[self.line - 1])
            cnt[self.line - 1] += cnt[self.line]
            cnt.pop(self.line)
            self.line -= 1

    def newline(self):
        """Add a newline to the text."""
        self.content.insert(self.line + 1,
                            self.content[self.line][self.caret_pos:])
        self.content[self.line] = self.content[self.line][:self.caret_pos]
        self.line += 1
        self.caret_pos = 0

    def add_char(self, char):
        """Add a character to the text."""
        before_caret = self.content[self.line][:self.caret_pos]
        after_caret  = self.content[self.line][self.caret_pos:]
        self.content[self.line] = before_caret + char + after_caret
        self.caret_pos += 1

    def move_caret(self, direction):
        """Move the caret in the text."""
        if direction == "End":
            self.line = len(self.content) - 1
            self.caret_pos = len(self.content[self.line])
        elif direction == "Home":
            self.line = 0
            self.caret_pos = 0
        elif direction == "Right":
            if self.caret_pos < len(self.content[self.line]):
                self.caret_pos += 1
            elif self.line < len(self.content) - 1:
                self.line += 1
                self.caret_pos = 0
        elif direction == "Left":
            if self.caret_pos > 0:
                self.caret_pos -= 1
            elif self.line > 0:
                self.line -= 1
                self.caret_pos = len(self.content[self.line])
        elif direction == "Down":
            if self.line < len(self.content) - 1:
                self.line += 1
                if self.caret_pos > len(self.content[self.line]):
                    self.caret_pos = len(self.content[self.line])
        elif direction == "Up":
            if self.line > 0:
                self.line -= 1
                if self.caret_pos > len(self.content[self.line]):
                    self.caret_pos = len(self.content[self.line])
        else:
            raise ValueError("Invalid direction:", direction)

    def update_by_key(self, keyname, char):
        """Update the text object by keypress."""
        if keyname == "BackSpace": # and cur["caret_pos"] > 0:
            self.backspace()
        elif keyname in ["Home", "End", "Down", "Up", "Right", "Left"]:
            self.move_caret(keyname)
        elif keyname == "Return":
            self.newline()
        elif char and char.isprintable():
            self.add_char(char)

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
        content, pen, caret_pos = self.content, self.pen, self.caret_pos

        # get font info
        cr.select_font_face(pen.font_family,
                            pen.font_style == "italic" and
                                cairo.FONT_SLANT_ITALIC or cairo.FONT_SLANT_NORMAL,
                            pen.font_weight == "bold"  and
                                cairo.FONT_WEIGHT_BOLD  or cairo.FONT_WEIGHT_NORMAL)
        cr.set_font_size(pen.font_size)

        font_extents      = cr.font_extents()
        self.font_extents = font_extents
        ascent, height    = font_extents[0], font_extents[2]

        dy   = 0

        # new bounding box
        bb_x = position[0]
        bb_y = position[1] - ascent
        bb_w = 0
        bb_h = 0

        if self.rotation:
            cr.save()
            cr.translate(self.rot_origin[0], self.rot_origin[1])
            cr.rotate(self.rotation)
            cr.translate(-self.rot_origin[0], -self.rot_origin[1])

        for i in range(len(content)):
            fragment = content[i]

            #x_bearing, y_bearing, t_width, t_height, x_advance, y_advance
            x_bearing, _, t_width, _, _, _ = cr.text_extents(fragment)

            bb_w = max(bb_w, t_width + x_bearing)
            bb_h += height

            cr.set_font_size(pen.font_size)
            cr.move_to(position[0], position[1] + dy)
            cr.set_source_rgba(*pen.color, pen.transparency)
            cr.show_text(fragment)
            cr.stroke()

            # draw the caret
            if not caret_pos is None and i == self.line:
                x_bearing, _, t_width, _, _, _ = cr.text_extents("|" +
                                                        fragment[:caret_pos] + "|")
                _, _, t_width2, _, _, _ = cr.text_extents("|")
                cr.set_source_rgb(1, 0, 0)
                xx0 = position[0] - x_bearing + t_width - 2 * t_width2
                yy0 = position[1] + dy - ascent
                self.draw_caret(cr, xx0, yy0, height)

            dy += height

        self.bb = (bb_x, bb_y, bb_w, bb_h)

        if self.rotation:
            cr.restore()
        if selected:
            self.bbox_draw(cr, lw=1.5)
        if hover:
            self.bbox_draw(cr, lw=.5)

class Shape(Drawable):
    """Class for shapes (closed paths with no outline)."""
    def __init__(self, coords, pen):
        super().__init__("shape", coords, pen)
        self.bb = None

    def finish(self):
        """Finish the shape."""
        print("finishing shape")
        self.coords, _ = smooth_path(self.coords)
        #self.outline_recalculate_new()

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
        if click_x >= x and click_x <= x + width and click_y >= y and click_y <= y + height:
            return True
        return False

    def path_append(self, x, y, pressure = None):
        """Append a new point to the path."""
        self.coords.append((x, y))
        self.bb = None

    def bbox(self):
        """Calculate the bounding box of the shape."""
        if self.resizing:
            return self.resizing["bbox"]
        if not self.bb:
            self.bb = path_bbox(self.coords)
        return self.bb

    def resize_end(self):
        """recalculate the coordinates after resizing"""
        old_bbox = self.bb or path_bbox(self.coords)
        self.coords = transform_coords(self.coords, old_bbox, self.resizing["bbox"])
        self.resizing  = None
        self.bb = path_bbox(self.coords)


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
        self.draw_simple(cr, res_bb)

        if outline:
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
    def from_path(cls, path):
        """Create a shape from a path."""
        return cls(path.coords, path.pen)

    @classmethod
    def from_object(cls, obj):
        """Create a shape from an object."""
        print("Shape.from_object", obj)
        if obj.coords and len(obj.coords) > 2 and obj.pen:
            return cls(obj.coords, obj.pen)

        # issue a warning
        print("Shape.from_object: invalid object")
        return obj

class Path(Drawable):
    """ Path is like shape, but not closed and has an outline that depends on
        line width and pressure."""
    def __init__(self, coords, pen, outline = None, pressure = None):
        super().__init__("path", coords, pen = pen)
        self.outline   = outline  or []
        self.pressure  = pressure or []
        self.outline_l = []
        self.outline_r = []
        self.bb        = []

        if len(self.coords) > 3 and not self.outline:
            self.outline_recalculate_new()

    def finish(self):
        self.outline_recalculate_new()
        if len(self.coords) != len(self.pressure):
            raise ValueError("Pressure and coords don't match")

    def update(self, x, y, pressure):
        self.path_append(x, y, pressure)

    def move(self, dx, dy):
        move_coords(self.coords, dx, dy)
        move_coords(self.outline, dx, dy)
        self.bb = None

    def rotate_end(self):
        # rotate all coords and outline
        self.coords  = coords_rotate(self.coords,  self.rotation, self.rot_origin)
        self.outline = coords_rotate(self.outline, self.rotation, self.rot_origin)
        self.rotation   = 0
        self.rot_origin = None
        # recalculate bbox
        self.bb = path_bbox(self.coords)

    def is_close_to_click(self, click_x, click_y, threshold):
        return is_click_close_to_path(click_x, click_y, self.coords, threshold)

    def to_dict(self):
        return {
            "type": self.type,
            "coords": self.coords,
            "outline": self.outline,
            "pressure": self.pressure,
            "pen": self.pen.to_dict()
        }

    def stroke_change(self, direction):
        """Change the stroke size."""
        self.pen.stroke_change(direction)
        self.outline_recalculate_new()

    def smoothen(self, threshold=20):
        """Smoothen the path."""
        if len(self.coords) < 3:
            return
        print("smoothening path")
        self.coords, self.pressure = smooth_path(self.coords, self.pressure, 1)
        self.outline_recalculate_new()

    def pen_set(self, pen):
        """Set the pen for the path."""
        self.pen = pen.copy()
        self.outline_recalculate_new()

    def outline_recalculate_new(self, coords = None, pressure = None):
        """Recalculate the outline of the path."""
        if not coords:
            coords = self.coords
        if not pressure:
            pressure = self.pressure or [1] * len(coords)

        lwd = self.pen.line_width

        if len(coords) < 3:
            return
        print("recalculating outline")

        print("1.length of coords and pressure:", len(coords), len(pressure))
        coords, pressure = smooth_path(coords, pressure, 20)
        print("2.length of coords and pressure:", len(coords), len(pressure))

        outline_l = []
        outline_r = []

        n = len(coords)

        for i in range(n - 2):
            p0, p1, p2 = coords[i], coords[i + 1], coords[i + 2]
            nx, ny = normal_vec(p0, p1)
            mx, my = normal_vec(p1, p2)

            width  = lwd * pressure[i] / 2
            #width  = self.line_width / 2

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
                outline_l.append(left_segment1_start)
                outline_r.append(right_segment1_start)

            outline_l.append(left_segment1_end)
            outline_l.append(left_segment2_start)
            outline_r.append(right_segment1_end)
            outline_r.append(right_segment2_start)

            if i == n - 2:
                print("last segment")
                outline_l.append(left_segment2_end)
                outline_r.append(right_segment2_end)

            #self.outline_l.append((p1[0] + nx * width, p1[1] + ny * width))
            #self.outline_r.append((p1[0] - nx * width, p1[1] - ny * width))

        self.outline_l, _ = smooth_path(outline_l, None, 20)
        self.outline_r, _ = smooth_path(outline_r, None, 20)
        self.outline  = outline_l + outline_r[::-1]
        self.coords   = coords
        self.pressure = pressure


    def path_append(self, x, y, pressure = 1):
        """Append a point to the path, calculating the outline of the
           shape around the path. Only used when path is created to
           allow for a good preview. Later, the path is smoothed and recalculated."""
        coords = self.coords
        width  = self.pen.line_width * pressure

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

        if len(coords) < 2:
            return

        p1, p2 = coords[-2], coords[-1]
        nx, ny = normal_vec(p1, p2)

        if len(coords) == 2:
            ## append the points for the first coord
            self.outline_l.append((p1[0] + nx * width, p1[1] + ny * width))
            self.outline_r.append((p1[0] - nx * width, p1[1] - ny * width))

        self.outline_l.append((p2[0] + nx * width, p2[1] + ny * width))
        self.outline_r.append((p2[0] - nx * width, p2[1] - ny * width))
        self.outline = self.outline_l + self.outline_r[::-1]
        self.bb = None

    # XXX not efficient, this should be done in path_append and modified
    # upon move.
    def bbox(self):
        if self.resizing:
            return self.resizing["bbox"]
        if not self.bb:
            self.bb = path_bbox(self.outline or self.coords)
        return self.bb

    def resize_end(self):
        """recalculate the outline after resizing"""
        print("length of coords and pressure:", len(self.coords), len(self.pressure))
        old_bbox = self.bb or path_bbox(self.coords)
        new_coords = transform_coords(self.coords, old_bbox, self.resizing["bbox"])
        pressure   = self.pressure
        self.outline_recalculate_new(coords=new_coords, pressure=pressure)
        self.resizing  = None
        self.bb = path_bbox(self.outline or self.coords)

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


    def draw_simple(self, cr, bbox=None):
        """draws the path as a single line. Useful for resizing."""

        if len(self.coords) < 2:
            return

        if bbox:
            old_bbox = path_bbox(self.outline or self.coords)
            coords = transform_coords(self.coords, old_bbox, bbox)
        else:
            coords = self.coords

        cr.set_source_rgb(*self.pen.color)
        cr.set_line_width(0.5)
        cr.move_to(coords[0][0], coords[0][1])
        for point in coords[1:]:
            cr.line_to(point[0], point[1])
        cr.stroke()


    def draw_standard(self, cr):
        """standard drawing of the path."""
        cr.set_fill_rule(cairo.FillRule.WINDING)

        cr.move_to(self.outline[0][0], self.outline[0][1])
        for point in self.outline[1:]:
            cr.line_to(point[0], point[1])
        cr.close_path()


    def draw(self, cr, hover=False, selected=False, outline = False):
        """Draw the path."""
        if len(self.outline) < 4 or len(self.coords) < 3:
            return

        if self.rotation != 0:
            cr.save()
            bb = self.bbox()
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


class Circle(Drawable):
    """Class for creating circles."""
    def __init__(self, coords, pen):
        super().__init__("circle", coords, pen)

    def update(self, x, y, pressure):
        self.coords[1] = (x, y)

    def resize_end(self):
        bbox = self.bbox()
        self.coords = [ (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]) ]
        self.resizing = None

    def resize_update(self, bbox):
        self.resizing["bbox"] = bbox
        self.coords = [ (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]) ]

    def draw(self, cr, hover=False, selected=False, outline=False):
        if hover:
            cr.set_line_width(self.pen.line_width + 1)
        else:
            cr.set_line_width(self.pen.line_width)

        cr.set_source_rgba(*self.pen.color, self.pen.transparency)
        x1, y1 = self.coords[0]
        x2, y2 = self.coords[1]
        w, h = (abs(x1 - x2), abs(y1 - y2))

        if w != 0 and h != 0:
            x0, y0 = (min(x1, x2), min(y1, y2))
            #cr.rectangle(x0, y0, w, h)
            cr.save()
            cr.translate(x0 + w / 2, y0 + h / 2)
            cr.scale(w / 2, h / 2)
            cr.arc(0, 0, 1, 0, 2 * 3.14159)
            cr.restore()

            if self.pen.fill_color:
                cr.fill()
            else:
                cr.stroke()

        if selected:
            cr.set_source_rgba(1, 0, 0)
            self.bbox_draw(cr, lw=.35)

        if hover:
            self.bbox_draw(cr, lw=.35)


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
    def __init__(self, event, objects, translate = None):
        self.event   = event
        self.objects = objects
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
Ctrl-b: Cycle background transparency
Ctrl-p: toggle between two pens
Ctrl-Shift-f: screenshot: for a screenshot, you need at least one rectangle
object (r mode) in the drawing which serves as the selection area. The
screenshot will be pasted into the drawing.


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

        print("Owner change, removing internal clipboard")
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
            text = sel.as_string()
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
            "box":         Gdk.Cursor.new_from_name(window.get_display(), "crosshair"),
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

    def __init__(self, app, canvas):

        # private attr
        self.__app = app
        self.__canvas = canvas
        self.__history    = []
        self.__redo_stack = []
        self.__page = None
        self.__selection = None
        self.page_set(Page())

    def page_set(self, page):
        """Set the current page."""
        self.__page = page
        self.__selection = self.__page.selection()

    def next_page(self):
        """Go to the next page."""
        self.page_set(self.__page.next(create = True))

    def prev_page(self):
        """Go to the prev page."""
        self.page_set(self.__page.prev())

    def delete_page(self):
        """Delete the current page."""
        self.page_set(self.__page.delete())

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
        return self.__selection

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
                               selection_objects=self.__selection.objects,
                               page = self.__page)
        self.__history.append(cmd)

    def transmute_selection(self, mode):
        """
        Transmute the selected objects to the given mode.

        Args:
            mode ( str ): The mode to transmute to.
        """
        if self.__selection.is_empty():
            return
        self.transmute(self.__selection.objects, mode)

    def set_objects(self, objects):
        """Set the list of objects."""
        ## no undo
        print("GOM: setting n=", len(objects), "objects")
        self.__page.objects(objects)

    def set_pages(self, pages):
        """Set the content of pages."""
        self.__page = Page()
        self.__page.objects(pages[0]['objects'])
        for p in pages[1:]:
            self.__page = self.__page.next()
            self.__page.objects(p['objects'])
        self.__selection = self.__page.selection()

    def add_object(self, obj):
        """Add an object to the list of objects."""
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
            objects = [ obj.to_dict() for obj in p.objects() ]
            pages.append({ "objects": objects })
            p = p.next(create = False)
        return pages

    def export_objects(self):
        """Just the objects from the current page."""
        objects = [ obj.to_dict() for obj in self.__page.objects() ]
        return objects

    def kill_object(self, obj):
        """Directly remove an object from the list of objects."""
        self.__page.kill_object(obj)

    def selected_objects(self):
        """Return the selected objects."""
        return self.__selection.objects

    def remove_selection(self):
        """Remove the selected objects from the list of objects."""
        if self.__selection.is_empty():
            return
        self.__history.append(RemoveCommand(self.__selection.objects,
                                            self.__page.objects(),
                                            page=self.__page))
        self.__selection.clear()

    def remove_objects(self, objects, clear_selection = False):
        """Remove an object from the list of objects."""
        self.__history.append(RemoveCommand(objects, self.__page.objects(), page=self.__page))
        if clear_selection:
            self.__selection.clear()

    def remove_all(self):
        """Clear the list of objects."""
        self.__history.append(RemoveCommand(self.__page.objects()[:],
                                            self.__page.objects(),
                                            page = self.__page))

    def command_append(self, command_list):
        """Append a group of commands to the history."""
        ## append in reverse order
        self.__history.append(CommandGroup(command_list[::-1]))

    def selection_group(self):
        """Group selected objects."""
        if self.__selection.n() < 2:
            return
        print("Grouping", self.__selection.n(), "objects")
        self.__history.append(GroupObjectCommand(self.__selection.objects,
                                                 self.__page.objects(),
                                                 selection_object=self.__selection,
                                                 page=self.__page))

    def selection_ungroup(self):
        """Ungroup selected objects."""
        if self.__selection.is_empty():
            return
        self.__history.append(UngroupObjectCommand(self.__selection.objects,
                                                   self.__page.objects(),
                                                   selection_object=self.__selection,
                                                   page=self.__page))

    def select_reverse(self):
        """Reverse the selection."""
        self.__selection.reverse()
        self.__app.dm.mode("move")

    def select_all(self):
        """Select all objects."""
        if not self.__page.objects():
            return

        self.__selection.all()
        self.__app.dm.mode("move")

    def selection_delete(self):
        """Delete selected objects."""
        #self.__page.selection_delete()
        if self.__selection.objects:
            self.__history.append(RemoveCommand(self.__selection.objects,
                                                self.__page.objects(), page=self.__page))
            self.__selection.clear()

    def select_next_object(self):
        """Select the next object."""
        self.__selection.next()

    def select_previous_object(self):
        """Select the previous object."""
        self.__selection.previous()

    def selection_fill(self):
        """Fill the selected object."""
        # XXX gom should not call dm directly
        # this code should be gone!
        color = self.__canvas.pen().color
        for obj in self.__selection.objects:
            obj.fill(color)

    def selection_color_set(self, color):
        """Set the color of the selected objects."""
        if not self.__selection.is_empty():
            self.__history.append(SetColorCommand(self.__selection, color))

    def selection_font_set(self, font_description):
        """Set the font of the selected objects."""
        # XXX: no undo!
        self.__history.append(SetFontCommand(self.__selection, font_description))
       #for obj in self.__selection.objects:
       #    obj.pen.font_set_from_description(font_description)

    def selection_apply_pen(self):
        """Apply the pen to the selected objects."""
        if not self.__selection.is_empty():
            pen = self.__canvas.pen()
            self.__history.append(SetPenCommand(self.__selection, pen))

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
        if self.__selection.is_empty():
            return
        self.move_obj(self.__selection.copy(), dx, dy)

    def rotate_obj(self, obj, angle):
        """Rotate the object by the given angle (degrees)."""
        print("rotating by", angle)
        event_obj = RotateCommand(obj, angle=math.radians(angle), page = self.__page)
        event_obj.event_finish()
        self.__history.append(event_obj)

    def rotate_selection(self, angle):
        """Rotate the selected objects by the given angle (degrees)."""
        if self.__selection.is_empty():
            return
        self.rotate_obj(self.__selection, angle)

    def selection_zmove(self, operation):
        """move the selected objects long the z-axis."""
        if self.__selection.is_empty():
            return
        self.__history.append(ZStackCommand(self.__selection.objects,
                                            self.__page.objects(), operation, page=self.__page))

    def draw(self, cr, hover_obj = None, mode = None):
        """Draw the objects in the given context. Used also by export functions."""

        for obj in self.__page.objects():
            hover    = obj == hover_obj and mode == "move"
            selected = self.__selection.contains(obj) and mode == "move"
            obj.draw(cr, hover=hover, selected=selected, outline = self.__canvas.outline())
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

    if pages:
        if len(pages) < page:
            raise ValueError(f"Page number out of range (max. {len(pages)})")
        print("read drawing from", input_file, "with", len(pages), "pages")
        objects = pages[page]['objects']

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

    def __init__(self, gom, app, dm, canvas):
        # singleton pattern
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self.__app = app
            self.__dm  = dm
            self.__canvas = canvas
            self.make_actions_dictionary(gom, app, dm, canvas)
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

        dm, app = self.__dm, self.__app

        keyname = Gdk.keyval_name(event.keyval)
        char    = chr(Gdk.keyval_to_unicode(event.keyval))
        ctrl    = event.state & Gdk.ModifierType.CONTROL_MASK
        shift   = event.state & Gdk.ModifierType.SHIFT_MASK
        alt_l   = event.state & Gdk.ModifierType.MOD1_MASK
        print("keyname", keyname, "char", char, "ctrl", ctrl, "shift", shift, "alt_l", alt_l)

        mode = dm.mode()

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
        cobj = dm.current_object()
        if cobj and cobj.type == "text" and not(ctrl or keyname == "Escape"):
            print("updating text input")
            cobj.update_by_key(keyname, char)
            app.queue_draw()
            return

        # otherwise, we dispatch the key event
        self.dispatch_key_event(keyfull, mode)

        # XXX this probably should be somewhere else
        app.queue_draw()


    def make_actions_dictionary(self, gom, app, dm, canvas):
        """
        This dictionary maps key events to actions.
        """
        self.__actions = {
            'mode_draw':             {'action': dm.mode, 'args': ["draw"]},
            'mode_box':              {'action': dm.mode, 'args': ["box"]},
            'mode_circle':           {'action': dm.mode, 'args': ["circle"]},
            'mode_move':             {'action': dm.mode, 'args': ["move"]},
            'mode_text':             {'action': dm.mode, 'args': ["text"]},
            'mode_select':           {'action': dm.mode, 'args': ["select"]},
            'mode_eraser':           {'action': dm.mode, 'args': ["eraser"]},
            'mode_shape':            {'action': dm.mode, 'args': ["shape"]},
            'mode_colorpicker':      {'action': dm.mode, 'args': ["colorpicker"]},

            'finish_text_input':     {'action': dm.finish_text_input},

            'cycle_bg_transparency': {'action': canvas.cycle_background},
            'toggle_outline':        {'action': canvas.outline_toggle},

            'clear_page':            {'action': dm.clear},
            'toggle_wiglets':        {'action': dm.toggle_wiglets},

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

            'set_color_white':       {'action': dm.set_color, 'args': [COLORS["white"]]},
            'set_color_black':       {'action': dm.set_color, 'args': [COLORS["black"]]},
            'set_color_red':         {'action': dm.set_color, 'args': [COLORS["red"]]},
            'set_color_green':       {'action': dm.set_color, 'args': [COLORS["green"]]},
            'set_color_blue':        {'action': dm.set_color, 'args': [COLORS["blue"]]},
            'set_color_yellow':      {'action': dm.set_color, 'args': [COLORS["yellow"]]},
            'set_color_cyan':        {'action': dm.set_color, 'args': [COLORS["cyan"]]},
            'set_color_magenta':     {'action': dm.set_color, 'args': [COLORS["magenta"]]},
            'set_color_purple':      {'action': dm.set_color, 'args': [COLORS["purple"]]},
            'set_color_grey':        {'action': dm.set_color, 'args': [COLORS["grey"]]},

            'apply_pen_to_bg':       {'action': canvas.apply_pen_to_bg,        'modes': ["move"]},
            'toggle_pens':           {'action': canvas.switch_pens},

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

            'apply_pen_to_selection': {'action': gom.selection_apply_pen,    'modes': ["move"]},

#            'Ctrl-m':               {'action': self.smoothen,           'modes': ["move"]},
            'copy_content':          {'action': app.copy_content,        },
            'cut_content':           {'action': app.cut_content,         'modes': ["move"]},
            'paste_content':         {'action': app.paste_content},
            'screenshot':            {'action': app.screenshot},

            'stroke_increase':       {'action': dm.stroke_change, 'args': [1]},
            'stroke_decrease':       {'action': dm.stroke_change, 'args': [-1]},
        }

    def get_keybindings(self):
        """
        Returns the keybindings dictionary.
        """
        return self.__keybindings

    def make_default_keybindings(self):
        """
        This dictionary maps key events to actions.
        """
        self.__keybindings = {
            'm':                    "mode_move",
            'b':                    "mode_box",
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

            'Shift-p':              "prev_page",
            'Shift-n':              "next_page",
            'Shift-d':              "delete_page",

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
            menu_items = [m for m in menu_items if not "action" in m or "selection_group" not in m["action"]]

        group_found = [o for o in objects if o.type == "group"]
        if not group_found:
            print("no group found")
            menu_items = [m for m in menu_items if not "action" in m or "selection_ungroup" not in m["action"]]

        self.__object_menu = self.build_menu(menu_items)

        return self.__object_menu
"""
Wiglets are small dialog-like objects that can be drawn on the canvas.
They are used to provide interactive controls for changing drawing properties
such as line width, color, and transparency.
"""

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

class WigletTransparency(Wiglet):
    """Wiglet for changing the transparency."""
    def __init__(self, coords, pen):
        super().__init__("transparency", coords)

        if not pen or not isinstance(pen, Pen):
            raise ValueError("Pen is not defined or not of class Pen")

        self.pen      = pen
        self._last_pt = coords[0]
        self._initial_transparency = pen.transparency
        print("initial transparency:", self._initial_transparency)

    def draw(self, cr):
        """draw the widget"""
        cr.set_source_rgba(*self.pen.color, self.pen.transparency)
        draw_dot(cr, *self.coords, 50)

    def event_update(self, x, y):
        """update on mouse move"""
        dx = x - self.coords[0]
        #print("changing transparency", dx)
        ## we want to change the transparency by 0.1 for every 20 pixels
        self.pen.transparency = max(0, min(1, self._initial_transparency + dx/500))
        #print("new transparency:", self.pen.transparency)

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
        self.pen      = pen
        self._last_pt = coords[0]
        self._initial_width = pen.line_width

    def draw(self, cr):
        cr.set_source_rgb(*self.pen.color)
        draw_dot(cr, *self.coords, self.pen.line_width)

    def event_update(self, x, y):
        dx = x - self.coords[0]
        print("changing line width", dx)
        self.pen.line_width = max(1, min(60, self._initial_width + dx/20))
        return True

    def event_finish(self):
        pass

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
        self.__current_page = self.__gom.current_page_number()

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
        cr.set_source_rgb(0.5, 0.5, 0.5)
        cr.rectangle(*self.__bbox)
        cr.fill()

        for i in range(self.__page_n):
            if i == self.__current_page:
                cr.set_source_rgb(0, 0, 0)
            else:
                cr.set_source_rgb(1, 1, 1)
            cr.rectangle(self.__bbox[0] + 1, self.__bbox[1] + i * self.__height_per_page + 1,
                        self.__width - 2, self.__height_per_page - 2)
            cr.fill()
            if i == self.__current_page:
                cr.set_source_rgb(1, 1, 1)
            else:
                cr.set_source_rgb(0, 0, 0)
            cr.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
            cr.set_font_size(14)
            cr.move_to(self.__bbox[0] + 5, self.__bbox[1] + i * self.__height_per_page + 20)
            cr.show_text(str(i + 1))

    def update_size(self, width, height):
        """update the size of the widget"""


## ---------------------------------------------------------------------
class WigletToolSelector(Wiglet):
    """Wiglet for selecting the tool."""
    def __init__(self, coords = (50, 0), width = 1000, height = 35, func_mode = None):
        super().__init__("tool_selector", coords)

        self.__width, self.__height = width, height
        self.__icons_only = True

        self.__modes = [ "move", "draw", "shape", "box",
                        "circle", "text", "eraser", "colorpicker" ]
        self.__modes_dict = { "move": "Move", "draw": "Draw", "shape": "Shape",
                              "box": "Rectangle", "circle": "Circle", "text": "Text",
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
    """
    def __init__(self, gom, app, cursor, canvas):
        self.__canvas = canvas
        self.__current_object = None
        self.__gom = gom
        self.__app = app
        self.__cursor = cursor
        self.__mode = "draw"

        # objects that indicate the state of the drawing area
        self.__hover = None
        self.__wiglet_active = None
        self.__resizeobj = None
        self.__selection_tool = None
        self.__current_object = None
        self.__paning = None
        self.__show_wiglets = True
        self.__wiglets = [ WigletColorSelector(height = app.get_size()[1],
                                               func_color = self.set_color,
                                               func_bg = self.__canvas.bg_color),
                           WigletToolSelector(func_mode = self.mode),
                           WigletPageSelector(gom = gom, screen_wh_func = app.get_size,
                                              set_page_func = gom.set_page_number),
                          ]

        # drawing parameters
        self.__hidden = False
        self.__modified = False
        self.__translate = None

        # defaults for drawing
    def toggle_wiglets(self):
        """Toggle the wiglets."""
        self.__show_wiglets = not self.__show_wiglets

    def show_wiglets(self, value = None):
        """Show or hide the wiglets."""
        if value is not None:
            self.__show_wiglets = value
        return self.__show_wiglets

    def hide(self, value = None):
        """Hide or show the drawing."""
        if not value is None:
            self.__hidden = value
        return self.__hidden

    def current_object(self):
        """Get the current object."""
        return self.__current_object

    def mode(self, mode = None):
        """Get or set the mode."""
        if mode:
            self.__mode = mode
            self.__cursor.default(self.__mode)
        return self.__mode

    def modified(self, value = None):
        """Get or set the modified flag."""
        if value is not None:
            self.__modified = value
        return self.__modified

    def on_draw(self, widget, cr):
        """Handle draw events."""
        if self.__hidden:
            print("I am hidden!")
            return True

        cr.save()
        if self.__translate:
            cr.translate(*self.__translate)

        cr.set_source_rgba(*self.__canvas.bg_color(), self.__canvas.transparent())
        cr.set_operator(cairo.OPERATOR_SOURCE)
        cr.paint()
        cr.set_operator(cairo.OPERATOR_OVER)
        self.__gom.draw(cr, self.__hover, self.__mode)

        if self.__current_object:
            self.__current_object.draw(cr)

        if self.__selection_tool:
            self.__selection_tool.draw(cr)

        cr.restore()

        if self.__show_wiglets:
            for w in self.__wiglets:
                w.update_size(*self.__app.get_size())
                w.draw(cr)

        if self.__wiglet_active:
            self.__wiglet_active.draw(cr)

        return True


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
            self.__mode = "move"
            self.__cursor.default(self.__mode)

            if not self.__gom.selection().contains(hover_obj):
                self.__gom.selection().set([ hover_obj ])

            # XXX - this should happen directly?
            sel_objects = self.__gom.selected_objects()
            self.__app.mm.object_menu(sel_objects).popup(None, None,
                                                         None, None,
                                                         event.button, event.time)
        else:
            self.__app.mm.context_menu().popup(None, None,
                                               None, None,
                                               event.button, event.time)
        self.__app.queue_draw()

    # ---------------------------------------------------------------------

    def __move_resize_rotate(self, ev):
        """Process events for moving, resizing and rotating objects."""

        if self.__mode != "move":
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
            self.__app.queue_draw()
            return True

        return False

    def create_object(self, ev):
        """Create an object based on the current mode."""

        # not managed by GOM: first create, then decide whether to add to GOM
        mode = self.__mode

        if ev.shift() and not ev.ctrl():
            mode = "text"

        if not mode in [ "draw", "shape", "box", "circle", "text" ]:
            return False

        obj = DrawableFactory.create_drawable(mode, pen = self.__canvas.pen(), ev=ev)

        if obj:
            self.__current_object = obj
        else:
            print("No object created for mode", mode)

        return True

    # Button press event handlers -------------------------------------------
    def on_button_press(self, widget, event):
        """Handle mouse button press events."""
        print("on_button_press: type:", event.type, "button:", event.button, "state:", event.state)
        self.__modified = True # better safe than sorry
        ev = MouseEvent(event, self.__gom.objects(), translate = self.__translate)

        # Ignore clicks when text input is active
        if self.__current_object:
            print("Current object exists and has type", self.__current_object.type)
            if  self.__current_object.type == "text":
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
        self.__app.queue_draw()

        return True

    def __handle_wiglets_on_click(self, event, ev):
        """Pass the event to the wiglets"""
        if self.__show_wiglets:
            for w in self.__wiglets[::-1]:
                if w.on_click(event.x, event.y, ev):
                    self.__app.queue_draw()
                    return True
        return False


    def __handle_mode_specials_on_click(self, event, ev):
        """Handle special events for the current mode."""

        # Start changing line width: single click with ctrl pressed
        if ev.ctrl(): # and self.__mode == "draw":
            if not ev.shift():
                self.__wiglet_active = WigletLineWidth((event.x, event.y), self.__canvas.pen())
            else:
                self.__wiglet_active = WigletTransparency((event.x, event.y), self.__canvas.pen())
            return True

        if self.__handle_color_picker_on_click():
            return True

        if self.__handle_eraser_on_click(ev):
            return True

        return False

    def __handle_color_picker_on_click(self):
        """Handle color picker on click events."""
        if not self.__mode == "colorpicker":
            return False

        color = get_color_under_cursor()
        self.set_color(color)
        color_hex = rgb_to_hex(color)
        self.__app.clipboard.set_text(color_hex)
        self.__app.queue_draw()
        return True

    def __handle_eraser_on_click(self, ev):
        """Handle eraser on click events."""
        if not self.__mode == "eraser":
            return False

        hover_obj = ev.hover()
        if not hover_obj:
            return False

        # self.history.append(RemoveCommand([ hover_obj ], self.objects))
        self.__gom.remove_objects([ hover_obj ], clear_selection = True)
        self.__resizeobj   = None
        self.__cursor.revert()
        self.__app.queue_draw()
        return True

    def __handle_text_input_on_click(self, ev):
        """Check whether text object should be activated."""
        hover_obj = ev.hover()

        if not hover_obj or hover_obj.type != "text":
            return False

        if not self.__mode in ["draw", "text", "move"]:
            return False

        if not ev.double(): #or self.__mode == "text":
            return False

        if ev.shift() or ev.ctrl():
            return False

        # only when double clicked with no modifiers - start editing the hover obj
        print("starting text editing existing object")
        hover_obj.move_caret("End")
        self.__current_object = hover_obj
        self.__app.queue_draw()
        self.__cursor.set("none")
        return True


    # Event handlers
    # XXX same comment as above
    # Button release handlers ------------------------------------------------
    def on_button_release(self, widget, event):
        """Handle mouse button release events."""
        print("button release: type:", event.type, "button:", event.button, "state:", event.state)
        ev = MouseEvent(event, self.__gom.objects(), translate = self.__translate)

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
        obj = self.__current_object
        if not obj:
            return False

        if obj.type in [ "shape", "path" ]:
            print("finishing path / shape")
            obj.path_append(ev.x, ev.y, 0)
            obj.finish()
            # remove paths that are too small
            if len(obj.coords) < 3:
                print("removing object of type", obj.type, "because too small")
                self.__current_object = None
                obj = None

        # remove objects that are too small
        if obj:
            bb = obj.bbox()
            if bb and obj.type in [ "box", "circle" ] and bb[2] == 0 and bb[3] == 0:
                print("removing object of type", obj.type, "because too small")
                self.__current_object = None
                obj = None

        if obj:
            self.__gom.add_object(obj)
            # with text, we are not done yet! Need to keep current object
            # such that keyboard events can update it
            if self.__current_object.type != "text":
                #self.__current_object.caret_pos = None

                self.__gom.selection().clear()
                self.__current_object = None

        self.__app.queue_draw()
        return True

    def __handle_wiglets_on_release(self):
        """Handle wiglets on mouse release."""
        if not self.__wiglet_active:
            return False

        self.__wiglet_active.event_finish()
        self.__wiglet_active = None
        self.__app.queue_draw()
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
        self.__app.queue_draw()
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

        #
        obj = self.__resizeobj.obj

        _, width = self.__app.get_size()
        if self.__resizeobj.command_type() == "move" and  event.x < 10 and event.y > width - 10:
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
        self.__app.queue_draw()

        return True

    # ---------------------------------------------------------------------
    # motion event handlers

    def on_motion_notify(self, widget, event):
        """Handle mouse motion events."""

        ev = MouseEvent(event, self.__gom.objects(), translate = self.__translate)
        x, y = ev.pos()
        self.__cursor.update_pos(x, y)

        if self.__on_motion_wiglet(x, y):
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

        if not self.__mode == "move":
            return False

        object_underneath = ev.hover()
        #prev_hover        = self.__hover

        if object_underneath:
            self.__cursor.set("moving")
            self.__hover = object_underneath
        else:
            self.__cursor.revert()
            self.__hover = None

        corner_obj, corner = ev.corner()

        if corner_obj and corner_obj.bbox():
            self.__cursor.set(corner)
            self.__hover = corner_obj
            self.__app.queue_draw()

        #if prev_hover != self.__hover:
        #    self.__app.queue_draw()

        self.__app.queue_draw()
        return True

    def __on_motion_update_resize(self, event):
        """Handle on motion update for resizing."""
        if not self.__resizeobj:
            return False

        self.__resizeobj.event_update(event.x, event.y)
        self.__app.queue_draw()
        return True

    def __on_motion_update_object(self, event):
        """Handle on motion update for an object."""
        obj = self.__current_object or self.__selection_tool
        if not obj:
            return False

        obj.update(event.x, event.y, event.pressure())
        self.__app.queue_draw()
        return True

    def __on_motion_paning(self, event):
        """Handle on motion update when paning"""
        if not self.__paning:
            return False

        if not self.__translate:
            self.__translate = (0, 0)
        dx, dy = event.x - self.__paning[0], event.y - self.__paning[1]
        self.__translate = (self.__translate[0] + dx, self.__translate[1] + dy)
        self.__paning = (event.x, event.y)
        self.__app.queue_draw()
        return True

    def __on_motion_wiglet(self, x, y):
        """Handle on motion update when a wiglet is active."""
        if not self.__wiglet_active:
            return False

        self.__wiglet_active.event_update(x, y)
        self.__app.queue_draw()
        return True


    # ---------------------------------------------------------------------
    def finish_text_input(self):
        """Clean up current text and finish text input."""
        print("finishing text input")
        if self.__current_object and self.__current_object.type == "text":
            self.__current_object.caret_pos = None
            if self.__current_object.strlen() == 0:
                print("kill object because empty")
                self.__gom.kill_object(self.__current_object)
            self.__current_object = None
        self.__cursor.revert()
    # ---------------------------------------------------------------------

    def stroke_change(self, direction):
        """Modify the line width or text size."""
        print("Changing stroke", direction)
        cobj = self.__current_object
        if cobj and cobj.type == "text":
            print("Changing text size")
            cobj.stroke_change(direction)
            self.__canvas.pen().font_size = cobj.pen.font_size
        else:
            for obj in self.__gom.selected_objects():
                obj.stroke_change(direction)

        # without a selected object, change the default pen, but only if in the correct mode
        if self.__mode == "draw":
            self.__canvas.pen().line_width = max(1, self.__canvas.pen().line_width + direction)
        elif self.__mode == "text":
            self.__canvas.pen().font_size = max(1, self.__canvas.pen().font_size + direction)

    def set_font(self, font_description):
        """Set the font."""
        self.__canvas.pen().font_set_from_description(font_description)
        self.__gom.selection_font_set(font_description)
        if self.__current_object and self.__current_object.type == "text":
            self.__current_object.pen.font_set_from_description(font_description)

#   def smoothen(self):
#       """Smoothen the selected object."""
#       if self.selection.n() > 0:
#           for obj in self.selection.objects:
#               obj.smoothen()
    def set_color(self, color = None):
        """Set the color."""
        if color is None:
            return self.__canvas.pen().color
        self.__canvas.pen().color_set(color)
        self.__gom.selection_color_set(color)
        return color

    def clear(self):
        """Clear the drawing."""
        self.__gom.selection().clear()
        self.__resizeobj      = None
        self.__current_object = None
        self.__gom.remove_all()
        self.__app.queue_draw()
## singleton class for serving icon pixbufs



class Icons:
    """
    Singleton class for serving
    icon pixbufs.
    """
    __new_instance = None

    def __new__(cls, *args, **kwargs):
        if cls.__new_instance is None:
            print("creating new instance of Icons")
            cls.__new_instance = super(Icons, cls).__new__(cls)
        else:
            print("returning existing instance of Icons")
        return cls.__new_instance

    def __init__(self):
        print("initializing Icons")
        self._icons = {}
        self._load_icons()

    def get(self, name):
        """Get the icon with the given name."""
        print("getting icon", name)
        if name not in self._icons:
            print("icon not found")
            return None
        return self._icons.get(name)

    def _load_icons(self):
        """
        Load all icons from stored base64 strings.
        """

        icons = { "colorpicker": "iVBORw0KGgoAAAANSUhEUgAAABwAAAAbCAYAAABvCO8sAAACm0lEQVRIx7XWT4hVVRgA8N+ZGRuFnkgoEhMTITbZJsEg7I8WVIsQ0UELFxGNk6KbmFYtEqIsApNW4mbmRRRJi+i1UrNCFyahhbkIF9ngFKMp40yMohX2tTkjt/G+8s17c+Bu7v3O/XG+e75zv2SWRkTE9HsppZUds4WllL7GZQxgCVZHxLGOWcKOYSyqlY1YC6lv8hC0NfPi4ihgx3EqY8UxHzqaWMUr2IyfUMv3DuBoVCtvlkxbDKkJ8Aj2YAPOYByrolpZc9Nm6Zv8AQsjoqutic/1I3bhT9yN3jrYW5gbEV0ppZSa3Bz7sDTvhe/xYFQrywvY53gkIhY2Y03V2Bpcwa/4Aidziv+KaiUwgi/L6nEmWC+uZ/BevJjRU9iBg9jfKmwFruEqni48fgz7MJQ30gc4NGM0T1yGCzmND5SEPZdX+Q6ewi8zAvOkroz9hnvqhPbgW5zHaYw0DOYJnfgZO9FdJ3QAYzm+He/i44bAHNyNsxmrN7oLmOlHXiPYC/nE3wn9ajF1FULvK6Su/f/em/6jqPtzWjbgq2kIGLTuCXyUT5HOfOo0BmbsNTyLT/B2GTa0YPtnMTF6f0T0pJTa8fetZK6tBHs519RDnds+/LQMe7/79RMxMbooY3NvFfsXmLEBrMfG+YNx1/N7K6dvwnreO3t95OSliHg0pXQb/mhk16cC9hL68GS/2pWy4KFFAxfi4vDRiFjfSBrrpXQrnqmHVe989VpcHK5lLM0Eg468ut040a82Xrqy2zdPxrmxXRHxRlO/mUKLsVZq61VSpoNzNv3u8tXDGZvXikZr+MYHXfr4xFRhz1m95TspfYPRXNSpFVjCJdxRaBvm4RwWRMSy3MC2BLtRDu1LHh6f+pVMOwc7W923/gMDooiVvplNJAAAAABJRU5ErkJggg==",
                 "box": "iVBORw0KGgoAAAANSUhEUgAAABwAAAAaCAMAAACTisy7AAACdlBMVEW+gFX///8AAAD///////////////////////////////////////////////////////////////////////////////////////////////////////////////////7//////v3///////////////////////////////////////////////////////7///////////////////////////////////////+0tLRwcXByc3FxcnB/gH7s7et0dHQAAAANABoPAB0NABwjEDHCrtDy6Pl1dnULABdVAJZeAKVdAKRdAKVfAqZqC7K1hdkMABlcAKJlALJoB7NqCrRnBrFhAKtjAK9hAK+vfNZbAKFmA7Lbw+3ZwOzcxO+agaweAjRTAJJjALGwfNdnBLLRs+f///+rrq0DAQhMAIevfNjPsef/+/b/1pj/zID/zYH/z4PXpVmEUBBrE4phALOvY5n/z4H/zYP/3q3/+O3/rjX/mwf/nQn/nAn/ngr7mRWIJItfALWvTGD/nwf/mwX/vlz/rzj/nQr/ng34lxWHJItgALWvTGH/oAv/v17Qsef/+e7Fptzy9PLx6t/8rTVcAKFkAa9BJFgwNC5BOy7cjRH/ogr/oQr6mRKIJIp0dXMKABdkAK86AGgmAEYzBEmiSlG+W1O9WlO9W1K5Vld4FZtgALTCw8GDc5BmCaxiAK9jALBiALKuS2Lu4fiMQcSALr6BL7+BLruBIJqBHZJ/G5S+W1L9/P717/r07fr07vv05un0o0H0khb0kxj0kxn5mBP/sDj/nwz/rjb/nAf/mwb//Pb/2J7/z4j/0In/z4f/4LGPZS+MAAAAAXRSTlMAQObYZgAAATVJREFUKM9jYMALODi50AA3QpLPytoGGdja2XPAJQUdHJ2ckYCLq5u7EFzSw9PL2wcBfP38A0QQkoFBwSGhMBAWHhEZhSwZHRMQGwcB8QmJSckposiSqWnpMJCRmZWcLcUpLSMjC5PMSc/Nyy8AgsKi4pLSsvKKioJKeYRkVXVNLRDU1Tc0NjW31Na2tikiSbZ3dIJBV3dPb19nZ10/FSQnpE/ELTlp8pSpOCSnTZ8xc9bsOXPBYN58ZMkFC6MXLV6ydNnyFSCwctXqNQhJ5bXr1m/YuDG5FAY2rdkMl1RV27J12/btO3bugoHde+CSDBp79+3ff+DgocNHoODosU64JIOmlpa2zsTj9SdgoBNJEgzggQAFqJInT3Ugg9NtSJJnzp47jwwuXNRHSBoaoQEpY/xJGgDtYfX0n9HvoQAAAABJRU5ErkJggg==",
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


class Page:
    """
    A page is a container for objects.
    """
    def __init__(self, prev = None):
        self.__prev    = prev
        self.__next = None
        self.__objects = []
        self.__selection = SelectionObject(self.__objects)

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

        if self.__prev:
            self.__prev.next_set(self.__next)
        if self.__next:
            self.__next.prev_set(self.__prev)
        return self.__prev or self.__next

    def objects(self, objects = None):
        """Return or set the list of objects on the page."""
        if objects:
            self.__objects = objects
            self.__selection = SelectionObject(self.__objects)
        return self.__objects

    def selection(self):
        """Return the selection object."""
        return self.__selection

    def kill_object(self, obj):
        """Directly remove an object from the list of objects."""
        if obj in self.__objects:
            self.__objects.remove(obj)
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
    def __init__(self):
        self.__bg_color = (.8, .75, .65)
        self.__transparency = 0
        self.__outline = False

        self.__pen  = Pen(line_width = 4,  color = (0.2, 0, 0), font_size = 24, transparency  = 1)
        self.__pen2 = Pen(line_width = 40, color = (1, 1, 0),   font_size = 24, transparency = .2)


    def pen(self, alternate = False, pen = None):
        """Get or set the pen."""
        if pen:
            self.__pen_set(pen, alternate)
        return self.__pen2 if alternate else self.__pen

    def __pen_set(self, pen, alternate = False):
        """Set the pen."""
        if alternate:
            self.__pen2 = pen
        else:
            self.__pen = pen

    def switch_pens(self):
        """Switch between pens."""
        self.__pen, self.__pen2 = self.__pen2, self.__pen

    def apply_pen_to_bg(self):
        """Apply the pen to the background."""
        self.__bg_color = self.__pen.color

    def cycle_background(self):
        """Cycle through background transparency."""
        self.__transparency = {1: 0, 0: 0.5, 0.5: 1}[self.__transparency]

    def outline(self):
        """Get the outline mode."""
        return self.__outline

    def outline_toggle(self):
        """Toggle outline mode."""
        self.__outline = not self.__outline

    def bg_color(self, color=None):
        """Get or set the background color."""
        if color:
            self.__bg_color = color
        return self.__bg_color

    def transparent(self, value=None):
        """Get or set the transparency."""
        if value:
            self.__transparency = value
        return self.__transparency




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
        self.clipboard           = Clipboard()
        self.canvas              = Canvas()
        self.gom                 = GraphicsObjectManager(self, canvas=self.canvas)
        self.cursor              = CursorManager(self)

        # dm needs to know about gom because gom manipulates the selection
        # and history (object creation etc)
        # it needs to know about the cursor to change the cursor if needed
        # it needs to know about the canvas to get the pen
        # it needs to know about the app to get the size of the window and
        # queue up redraws
        self.dm                  = DrawManager(gom = self.gom,  app = self,
                                               cursor = self.cursor, canvas = self.canvas)

        # em has to know about all that to link actions to methods
        self.em                  = EventManager(gom = self.gom, app = self,
                                                dm = self.dm, canvas = self.canvas)
        self.mm                  = MenuMaker(self.em, self)

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

        self.connect("key-press-event",      self.em.on_key_press)
        self.connect("draw",                 self.dm.on_draw)
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

        cobj = self.dm.current_object()
        if cobj and cobj.type == "text":
            cobj.add_text(clip_text.strip())
        else:
            new_text = Text([ self.cursor.pos() ],
                            pen = self.canvas.pen(), content=clip_text.strip())
            self.gom.add_object(new_text)

    def paste_image(self, clip_img):
        """Create an image object from a pixbuf image."""
        obj = Image([ self.cursor.pos() ], self.canvas.pen(), clip_img)
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
            self.canvas.bg_color((color.red, color.green, color.blue))

    def select_color(self):
        """Select a color for drawing using ColorChooser dialog."""
        color = ColorChooser(self)
        if color:
            self.dm.set_color((color.red, color.green, color.blue))

    def select_font(self):
        """Select a font for drawing using FontChooser dialog."""
        font_description = FontChooser(self.canvas.pen(), self)

        if font_description:
            self.dm.set_font(font_description)

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
                         bg = self.canvas.bg_color(), 
                         bbox = bbox, transparency = self.canvas.transparent())

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
                img = Image([ pos ], self.canvas.pen(), pixbuf)
                self.gom.add_object(img)
                self.queue_draw()

        return pixbuf

    def __screenshot_finalize(self, bb):
        """Finish up the screenshot."""
        print("Taking screenshot now")
        frame = (bb[0] - 3, bb[1] - 3, bb[0] + bb[2] + 6, bb[1] + bb[3] + 6)
        pixbuf, filename = get_screenshot(self, *frame)
        self.dm.hide(False)
        self.queue_draw()

        # Create the image and copy the file name to clipboard
        if pixbuf is not None:
            img = Image([ (bb[0], bb[1]) ], self.canvas.pen(), pixbuf)
            self.gom.add_object(img)
            self.queue_draw()
            self.clipboard.set_text(filename)

    def __find_screenshot_box(self):
        """Find a box suitable for selecting a screenshot."""
        cobj = self.dm.current_object()
        if cobj and cobj.type == "box":
            return cobj
        for obj in self.gom.selected_objects():
            if obj.type == "box":
                return obj
        for obj in self.gom.objects()[::-1]:
            if obj.type == "box":
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
        self.dm.hide(True)
        self.queue_draw()
        while Gtk.events_pending():
            Gtk.main_iteration_do(False)
        GLib.timeout_add(100, self.__screenshot_finalize, bb)

    def __autosave(self):
        """Autosave the drawing state."""
        if not self.dm.modified():
            return

        if self.dm.current_object(): # not while drawing!
            return

        print("Autosaving")
        self.__save_state()
        self.dm.modified(False)

    def __save_state(self):
        """Save the current drawing state to a file."""
        if not self.savefile:
            print("No savefile set")
            return

        print("savefile:", self.savefile)
        config = {
                'bg_color':    self.canvas.bg_color(),
                'transparent': self.canvas.transparent(),
                'show_wiglets': self.dm.show_wiglets(),
                'bbox':        (0, 0, *self.get_size()),
                'pen':         self.canvas.pen().to_dict(),
                'pen2':        self.canvas.pen(alternate = True).to_dict()
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
            self.dm.modified(True)

    def read_file(self, filename, load_config = True):
        """Read the drawing state from a file."""
        config, objects, pages = read_file_as_sdrw(filename)

        if pages:
            self.gom.set_pages(pages)
        elif objects:
            self.gom.set_objects(objects)

        if config and load_config:
            self.canvas.bg_color(config.get('bg_color') or (.8, .75, .65))
            self.canvas.transparent(config.get('transparent') or 0)
            show_wiglets = config.get('show_wiglets')
            if show_wiglets is None:
                show_wiglets = True
            self.dm.show_wiglets(show_wiglets)
            self.canvas.pen(pen = Pen.from_dict(config['pen']))
            self.canvas.pen(pen = Pen.from_dict(config['pen2']), alternate = True)
        if config or objects:
            self.dm.modified(True)
            return True
        return False

    def load_state(self):
        """Load the drawing state from a file."""
        self.read_file(self.savefile)


## ---------------------------------------------------------------------

if __name__ == "__main__":
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
"""
    )

    parser.add_argument("-b", "--border", help="Border width for conversion", type=int)
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
        convert_file(args.files[0], OUTPUT, args.convert, border = args.border)
        exit(0)

    if args.files:
        if len(args.files) > 2:
            print("Too many files provided")
            exit(1)
        elif len(args.files) == 2:
            convert_file(args.files[0], args.files[1], border = args.border)
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
    win.cursor.set(win.dm.mode())
    win.stick()

    Gtk.main()
