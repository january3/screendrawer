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
import gi
import copy
import yaml
import pickle
gi.require_version('Gtk', '3.0')

from gi.repository import Gtk, Gdk, GdkPixbuf, Pango, GLib
import cairo
import os
import time
import math
import base64
import tempfile
from io import BytesIO

import warnings
import appdirs
import argparse

import pyautogui
from PIL import ImageGrab
from os import path

# ---------------------------------------------------------------------
# These are various classes and utilities for the sd.py script. In the
# "skeleton" variant, they are just imported. In the "full" variant, they
# the files are directly, physically inserted in order to get one big fat 
# Python script that can just be copied.

## ---------------------------------------------------------------------
                                                                     

def get_default_savefile(app_name, app_author):
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
    return "#{:02x}{:02x}{:02x}".format(*[int(255 * c) for c in rgb])

def get_screenshot(window, x0, y0, x1, y1):

    # Get the absolute position of the area to capture
    window_position = window.get_position()
    x0 = window_position[0] + x0
    y0 = window_position[1] + y0
    x1 = window_position[0] + x1
    y1 = window_position[1] + y1

    screenshot = ImageGrab.grab(bbox=(x0, y0, x1, y1))

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    screenshot.save(temp_file, format="PNG")
    temp_file_name = temp_file.name  
    temp_file.close()
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
    return sorted(objs, key=lambda x: stack.index(x))


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
    else:
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
                x = (1-t)**3 * control1[0] + 3*(1-t)**2 * t * p1[0] + 3*(1-t) * t**2 * control2[0] + t**3 * p2[0]
                y = (1-t)**3 * control1[1] + 3*(1-t)**2 * t * p1[1] + 3*(1-t) * t**2 * control2[1] + t**3 * p2[1]
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
    if not coords:
        ValueError("No coordinates to move")
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

def draw_dot(cr, x, y, diameter):
    """Draws a dot at the specified position with the given diameter."""
    cr.arc(x, y, diameter / 2, 0, 2 * 3.14159)  # Draw a circle
    cr.fill()  # Fill the circle to make a dot




## ---------------------------------------------------------------------
## These are the commands that can be executed on the objects. They should
## be undoable and redoable. It is their responsibility to update the
## state of the objects they are acting on.

class Command:
    """Base class for commands."""
    def __init__(self, type, objects):
        self.obj   = objects
        self._type = type
        self._undone = False

    def command_type(self):
        return self._type

    def undo(self):
        raise NotImplementedError("undo method not implemented")

    def redo(self):
        raise NotImplementedError("redo method not implemented")

class CommandGroup(Command):
    """Simple class for handling groups of commands."""
    def __init__(self, commands):
        self._commands = commands

    def undo(self):
        for cmd in self._commands:
            cmd.undo()
        self._undone = True

    def redo(self):
        if not self._undone:
            return
        for cmd in self._commands:
            cmd.redo()
        self._undone = False
        
class SetColorCommand(Command):
    """Simple class for handling color changes."""
    # XXX: what happens if an object is added to group after the command,
    # but before the undo? well, bad things happen
    def __init__(self, objects, color):
        super().__init__("set_color", objects.get_primitive())
        self._color = color
        self._undo_color = { obj: obj.pen.color for obj in self.obj }

        for obj in self.obj:
            obj.color_set(color)

    def undo(self):
        for obj in self.obj:
            if obj in self._undo_color:
                obj.color_set(self._undo_color[obj])
        self._undone = True

    def redo(self):
        if not self._undone:
            return
        for obj in self.obj:
            obj.color_set(self._color)
        self._undone = False

class RemoveCommand(Command):
    """Simple class for handling deleting objects."""
    def __init__(self, objects, stack):
        super().__init__("remove", objects)
        self._stack = stack

        # remove the objects from the stack
        for obj in self.obj:
            self._stack.remove(obj)

    def undo(self):
        for obj in self.obj:
        # XXX: it should insert at the same position!
            self._stack.append(obj)
        self._undone = True

    def redo(self):
        if not self._undone:
            return
        for obj in self.obj:
            self._stack.remove(obj)
        self._undone = False

class AddCommand(Command):
    """Simple class for handling creating objects."""
    def __init__(self, objects, stack):
        super().__init__("add", objects)
        self._stack = stack
        self._stack.append(self.obj)

    def undo(self):
        self._stack.remove(self.obj)
        self._undone = True

    def redo(self):
        if not self._undone:
            return
        self._stack.append(self.obj)
        self._undone = False

class TransmuteCommand(Command):
    """
    Turning object(s) into another type.

    Internally, for each object we create a new object of the new type, and
    replace the old object with the new one in the stack.

    For undo method, we should store the old object as well as its position in the stack.
    However, we don't. Instead we just slap the old object back onto the stack.
    """

    def __init__(self, objects, stack, new_type, selection_objects = None):
        super().__init__("transmute", objects)
        self._new_type = new_type
        self._old_objs = [ ]
        self._new_objs = [ ]
        self._stack    = stack
        self._selection_objects = selection_objects

        for obj in self.obj:
            new_obj = DrawableFactory.transmute(obj, new_type)

            if not obj in self._stack:
                raise ValueError("TransmuteCommand: Got Object not in stack:", obj)
                continue

            if obj == new_obj: # ignore if no transmutation
                continue

            self._old_objs.append(obj)
            self._new_objs.append(new_obj)
            self._stack.remove(obj)
            self._stack.append(new_obj)

        if self._selection_objects:
            self.map_selection()

    def map_selection(self):
        obj_map = self.obj_map()
        # XXX this should not change the order of the objects
        self._selection_objects[:] = [ obj_map.get(obj, obj) for obj in self._selection_objects ]

    def obj_map(self):
        """Return a dictionary mapping old objects to new objects."""
        return { self._old_objs[i]: self._new_objs[i] for i in range(len(self._old_objs)) }

    def undo(self):
        """replace all the new objects with the old ones in the stack"""
        if self._undone:
            return
        for obj in self._new_objs:
            self._stack.remove(obj)
        for obj in self._old_objs:
            self._stack.append(obj)
        self._undone = True
        if self._selection_objects:
            self.map_selection()

    def redo(self):
        """put the new objects again on the stack and remove the old ones"""
        if not self._undone:
            return
        for obj in self._old_objs:
            self._stack.remove(obj)
        for obj in self._new_objs:
            self._stack.append(obj)
        self._undone = False
        if self._selection_objects:
            self.map_selection()

class ZStackCommand(Command):
    """Simple class for handling z-stack operations."""
    def __init__(self, objects, stack, operation):
        super().__init__("z_stack", objects)
        self._operation  = operation
        self._stack      = stack

        for obj in objects:
            if not obj in stack:
                raise ValueError("Object not in stack:", obj)

        self._objects = sort_by_stack(objects, stack)
        # copy of the old stack
        self._stack_orig = stack[:]

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
        li = self._stack.index(self._objects[-1])
        n  = len(self._stack)

        # if the last element is already on top, we just move everything to
        # the top
        if li == n - 1:
            self.top()
            return

        # otherwise, we move all the objects to the position of the element
        # following the last one. Then, we just copy the elements from the
        # stack to the new stack, and when we see the indicator object, we
        # add our new objects.
        ind_obj = self._stack[li + 1]

        new_list = []
        for i in range(n):
            o = self._stack[i]
            if not o in self._objects:
                new_list.append(o)
            if o == ind_obj:
                new_list.extend(self._objects)

        self._stack[:] = new_list[:]

    def lower(self):
        fi = self._stack.index(self._objects[0])
        n  = len(self._stack)

        if fi == 0:
            self.bottom()
            return

        # otherwise, we move all the objects to the position of the element
        # preceding the first one. Then, we just copy the elements from the
        # stack to the new stack, and when we see the indicator object, we
        # this could be done more efficiently, but that way it is clearer

        ind_obj = self._stack[fi - 1]
        new_list = []
        for i in range(n):
            o = self._stack[i]
            if o == ind_obj:
                new_list.extend(self._objects)
            if not o in self._objects:
                new_list.append(o)

        self._stack[:] = new_list[:]

    def top(self):
        for obj in self._objects:
            self._stack.remove(obj)
            self._stack.append(obj)

    def bottom(self):
        for obj in self.obj[::-1]:
            self._stack.remove(obj)
            self._stack.insert(0, obj)

    def undo(self):
        self.swap_stacks(self._stack, self._stack_orig)
        self._undone = True

    def redo(self):
        if not self._undone:
            return
        self.swap_stacks(self._stack, self._stack_orig)
        self._undone = False

    def swap_stacks(self, stack1, stack2):
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

    def __init__(self, type, obj, origin):
        super().__init__("move", obj)
        self.start_point = origin
        self.origin      = origin
        self.bbox        = obj.bbox()

    def event_update(self, x, y):
        raise NotImplementedError("event_update method not implemented for type", self._type)

    def event_finish(self):
        raise NotImplementedError("event_finish method not implemented for type", self._type)

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

    def __init__(self, obj, origin=None, corner=None, angle = None):
        super().__init__("rotate", obj, origin)
        self.corner      = corner
        bb = obj.bbox()
        self._rotation_centre = (bb[0] + bb[2] / 2, bb[1] + bb[3] / 2)
        obj.rotate_start(self._rotation_centre)

        if not angle is None:
            self.obj.rotate(angle, set = False)

        self._angle = 0

    def event_update(self, x, y):
        angle = calc_rotation_angle(self._rotation_centre, self.start_point, (x, y))
        d_a = angle - self._angle
        self._angle = angle
        self.obj.rotate(d_a, set = False)

    def event_finish(self):
        self.obj.rotate_end()

    def undo(self):
        self.obj.rotate_start(self._rotation_centre)
        self.obj.rotate(-self._angle)
        self.obj.rotate_end()
        self._undone = True

    def redo(self):
        if not self._undone:
            return
        self.obj.rotate_start(self._rotation_centre)
        self.obj.rotate(self._angle)
        self.obj.rotate_end()
        self._undone = False

class MoveCommand(MoveResizeCommand):
    """Simple class for handling move events."""
    def __init__(self, obj, origin):
        super().__init__("move", obj, origin)
        self._last_pt = origin
        print("MoveCommand: origin is", origin)

    def event_update(self, x, y):
        dx = x - self._last_pt[0]
        dy = y - self._last_pt[1]

        self.obj.move(dx, dy)
        self._last_pt = (x, y)

    def event_finish(self):
        print("MoveCommand: finish")
        pass

    def undo(self):
        if self._undone:
            return
        print("MoveCommand: undo")
        dx = self.start_point[0] - self._last_pt[0]
        dy = self.start_point[1] - self._last_pt[1]
        self.obj.move(dx, dy)
        self._undone = True

    def redo(self):
        if not self._undone:
            return
        dx = self.start_point[0] - self._last_pt[0]
        dy = self.start_point[1] - self._last_pt[1]
        self.obj.move(-dx, -dy)
        self._undone = False


class ResizeCommand(MoveResizeCommand):
    """Simple class for handling resize events."""
    def __init__(self, obj, origin, corner, proportional = False):
        super().__init__("resize", obj, origin)
        self.corner = corner
        obj.resize_start(corner, origin)
        self._orig_bb = obj.bbox()
        self._prop    = proportional
        ## XXX check the bb for pitfalls
        self._orig_bb_ratio = self._orig_bb[3] / self._orig_bb[2]


    def undo(self):
        obj = self.obj
        pt  = (self._orig_bb[0], self._orig_bb[1])
        obj.resize_start(self.corner, pt)
        self.obj.resize_update(self._orig_bb)
        obj.resize_end()
        self._undone = True

    def redo(self):
        if not self._undone:
            return
        obj = self.obj
        obj.resize_start(self.corner, self.start_point)
        obj.resize_update(self._newbb)
        obj.resize_end()
        self._undone = False

    def event_finish(self):
        self.obj.resize_end()

    def event_update(self, x, y):
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

        self._newbb = newbb
        self.obj.resize_update(newbb)



class Pen:
    """
    Represents a pen with customizable drawing properties.

    This class encapsulates properties like color, line width, and font settings
    that can be applied to drawing operations on a canvas.

    Attributes:
        color (tuple): The RGB color of the pen as a tuple (r, g, b), with each component ranging from 0 to 1.
        line_width (float): The width of the lines drawn by the pen.
        font_size (int): The size of the font when drawing text.
        fill_color (tuple or None): The RGB fill color for shapes. `None` means no fill.
        transparency (float): The transparency level of the pen's color, where 1 is opaque and 0 is fully transparent.
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

    def __init__(self, color = (0, 0, 0), line_width = 12, transparency = 1, fill_color = None, 
                 font_size = 12, font_family = "Sans", font_weight = "normal", font_style = "normal"):
        """
        Initializes a new Pen object with the specified drawing properties.
        """
        self.color        = color
        self.line_width   = line_width
        self.font_size    = font_size
        self.fill_color   = fill_color
        self.transparency = transparency
        #self.font_family       = font_family or "Segoe Script"
        self.font_family       = font_family or "Sans"
        self.font_weight       = font_weight or "normal"
        self.font_style        = font_style  or "normal"

    def color_set(self, color):
        self.color = color

    def font_set_from_description(self, font_description):
        self.font_family = font_description.get_family()
        self.font_size   = font_description.get_size() / Pango.SCALE
        self.font_weight = "bold"   if font_description.get_weight() == Pango.Weight.BOLD  else "normal"
        self.font_style  = "italic" if font_description.get_style()  == Pango.Style.ITALIC else "normal"

        print("setting font to", self.font_family, self.font_size, self.font_weight, self.font_style)


    def transparency_set(self, transparency):
        self.transparency = transparency
    
    def fill_set(self, color):
        self.fill_color = color

    def stroke_change(self, direction):
        # for thin lines, a fine tuned change of line width
        if self.line_width > 2:
            self.line_width += direction
        else:
            self.line_width += direction / 10
        self.line_width = max(0.1, self.line_width)

    def to_dict(self):
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
        return Pen(self.color, self.line_width, self.transparency, self.fill_color, self.font_size, self.font_family, self.font_weight, self.font_style)

    @classmethod
    def from_dict(cls, d):
        #def __init__(self, color = (0, 0, 0), line_width = 12, font_size = 12, transparency = 1, fill_color = None, family = "Sans", weight = "normal", style = "normal"):
        return cls(d.get("color"), d.get("line_width"), d.get("transparency"), d.get("fill_color"),
                   d.get("font_size"), d.get("font_family"), d.get("font_weight"), d.get("font_style"))


## ---------------------------------------------------------------------
## These are the objects that can be displayed. It includes groups, but
## also primitives like boxes, paths and text.



class DrawableFactory:
    @classmethod
    def create_drawable(cls, mode, pen, ev):
        print("create object of type", mode)
        shift, ctrl, pressure = ev.shift(), ev.ctrl(), ev.pressure()
        pos = ev.pos()
        corner_obj = ev.corner()
        hover_obj  = ev.hover()

        ret_obj = None

        if mode == "text" or (mode == "draw" and shift and not ctrl and not corner_obj[0] and not hover_obj):
            #manager.cursor.set("none")
            ret_obj = Text([ pos ], pen = pen, content = "")
            ret_obj.move_caret("Home")

        elif mode == "draw":
            ret_obj = Path([ pos ], pen = pen, pressure = [ pressure ])

        elif mode == "box":
            ret_obj = Box([ pos, (pos[0], pos[1]) ], pen = pen)

        elif mode == "shape":
            ret_obj = Shape([ pos ], pen = pen)

        elif mode == "circle":
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
        elif mode == "text":
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
    def __init__(self, type, coords, pen):
        self.type       = type
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
        pass

    def finish(self):
        """Called when building (drawing, typing etc.) is concluded."""
        pass

    def get_primitive(self):
        """This is for allowing to distinguish between primitives and groups."""
        return self

    # ------------ Drawable rotation methods ------------------
    def rotate_start(self, origin):
        self.rot_origin = origin

    def rotate(self, angle, set = False):
        # the self.rotation variable is for drawing while rotating
        if set:
            self.rotation = angle
        else:
            self.rotation += angle

    def rotate_end(self):
        raise NotImplementedError("rotate_end method not implemented")

    # ------------ Drawable resizing methods ------------------
    def resize_start(self, corner, origin):
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
        self.pen = pen.copy()

    def stroke_change(self, direction):
        """Change the stroke size of the object."""
        self.pen.stroke_change(direction)

    def smoothen(self, threshold=20):
        print("smoothening not implemented")

    def unfill(self):
        """Remove the fill from the object."""
        self.pen.fill_set(None)

    def fill(self, color = None):
        """Fill the object with a color."""
        self.pen.fill_set(color)

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
        return {
            "type": self.type,
            "coords": self.coords,
            "pen": self.pen.to_dict()
        }

    def move(self, dx, dy):
        move_coords(self.coords, dx, dy)
        if self.rotation:
            self.rot_origin = (self.rot_origin[0] + dx, self.rot_origin[1] + dy)

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

    def draw(self, cr, hover=False, selected=False, outline=False):
        raise NotImplementedError("draw method not implemented")

    # ------------ Drawable conversion methods ------------------
    @classmethod
    def from_dict(cls, d):
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

        type = d.pop("type")
        if type not in type_map:
            raise ValueError("Invalid type:", type)

        if "pen" in d:
            d["pen"] = Pen.from_dict(d["pen"])
        #print("generating object of type", type, "with data", d)
        return type_map.get(type)(**d)

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
    def __init__(self, objects = [ ], objects_dict = None):

        if objects_dict:
            objects = [ Drawable.from_dict(d) for d in objects_dict ]

        print("Creating DrawableGroup with ", len(objects), "objects")
        super().__init__("drawable_group", [ (None, None) ], None)
        self.objects = objects
        self.type = "group"

    def contains(self, obj):
        return obj in self.objects

    def is_close_to_click(self, click_x, click_y, threshold):
        for obj in self.objects:
            if obj.is_close_to_click(click_x, click_y, threshold):
                return True
        return False

    def stroke_change(self, direction):
        for obj in self.objects:
            obj.stroke_change(direction)

    def transmute_to(self, mode):
        """Transmute all objects within the group to a new type."""
        print("transmuting group to", mode)
        for i in range(len(self.objects)):
            self.objects[i] = DrawableFactory.transmute(self.objects[i], mode)

    def to_dict(self):
        return {
            "type": self.type,
            "objects_dict": [ obj.to_dict() for obj in self.objects ],
        }

    def color_set(self, color):
        for obj in self.objects:
            obj.color_set(color)

    def font_set(self, size, family, weight, style):
        for obj in self.objects:
            obj.font_set(size, family, weight, style)

    def resize_start(self, corner, origin):
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
        primitives = [ obj.get_primitive() for obj in self.objects ]
        return flatten_and_unique(primitives)


    def rotate_start(self, origin):
        self.rot_origin = origin
        for obj in self.objects:
            obj.rotate_start(origin)

    def rotate(self, angle, set = False):
        if set:
            self.rotation = angle
        else:
            self.rotation += angle
        for obj in self.objects:
            obj.rotate(angle, set)

    def rotate_end(self):
        for obj in self.objects:
            obj.rotate_end()
        self.rot_origin = None
        self.rotation = 0
 
    def resize_update(self, bbox):
        """Resize the group of objects. we need to calculate the new
           bounding box for each object within the group"""
        orig_bbox = self.resizing["orig_bbox"]

        dx, dy           = bbox[0] - orig_bbox[0], bbox[1] - orig_bbox[1]
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
            left, top = min(left, x, x + w), min(top, y, y + h)
            bottom, right = max(bottom, y, y + h), max(right, x, x + w)

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

    def draw(self, cr, hover=False, selected=False, outline=False):
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
    """

    def __init__(self, all_objects):
        super().__init__([ ], None)

        self._all_objects = all_objects

    def n(self):
        return len(self.objects)

    def is_empty(self):
        return not self.objects

    def clear(self):
        print("clearing selection")
        self.objects = [ ]

    def toggle(self, obj):
        if obj in self.objects:
            self.objects.remove(obj)
        else:
            self.objects.append(obj)

    def set(self, objects):
        print("setting selection to", objects)
        self.objects = objects

    def add(self, obj):
        print("adding object to selection:", obj, "selection is", self.objects)
        if not obj in self.objects:
            self.objects.append(obj)

    def all(self):
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
            image = self.decode_base64(image_base64)
        else:
            self.image_base64 = None

        self.image_size = (image.get_width(), image.get_height())
        self.transform = transform or (1, 1)
        width, height = self.image_size[0] * self.transform[0], self.image_size[1] * self.transform[1]
        coords = [ (coords[0][0], coords[0][1]), (coords[0][0] + width, coords[0][1] + height) ]
        super().__init__("image", coords, pen)
        self.image = image
        self._orig_bbox = None

        if rotation:
            self.rotation = rotation
            self.rotate_start((coords[0][0] + width / 2, coords[0][1] + height / 2))

    def _bbox_internal(self):
        x, y = self.coords[0]
        w, h = self.coords[1]
        return (x, y, w - x, h - y)

    def draw(self, cr, hover=False, selected=False, outline=False):
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
        self._orig_bbox = self.bbox()
        self.resizing = {
            "corner": corner,
            "origin": origin,
            "bbox":   self.bbox(),
            "transform": self.transform
            }

    def resize_update(self, bbox):
        self.resizing["bbox"] = bbox
        old_bbox = self.bbox()
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
        self.transform = (w_scale * self.resizing["transform"][0], h_scale * self.resizing["transform"][1])

    def resize_end(self):
        self.coords[1] = (self.coords[0][0] + self.image_size[0] * self.transform[0], self.coords[0][1] + self.image_size[1] * self.transform[1])
        self.resizing = None

    def rotate_end(self):
        bb = self._bbox_internal()
        center_x, center_y = bb[0] + bb[2] / 2, bb[1] + bb[3] / 2
        new_center = coords_rotate([(center_x, center_y)], self.rotation, self.rot_origin)[0]
        self.move(new_center[0] - center_x, new_center[1] - center_y)
        self.rot_origin = new_center

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
            "pen": self.pen.to_dict(),
            "image": None,
            "rotation": self.rotation,
            "transform": self.transform,
            "image_base64": self.base64(),
        }


class Text(Drawable):
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
        if self.bb is None:
            return False
        x, y, width, height = self.bb
        if click_x >= x and click_x <= x + width and click_y >= y and click_y <= y + height:
            return True

    def move(self, dx, dy):
        move_coords(self.coords, dx, dy)
        if self.rotation:
            self.rot_origin = (self.rot_origin[0] + dx, self.rot_origin[1] + dy)

    def rotate_end(self):
        if self.bb:
            center_x, center_y = self.bb[0] + self.bb[2] / 2, self.bb[1] + self.bb[3] / 2
            new_center = coords_rotate([(center_x, center_y)], self.rotation, self.rot_origin)[0]
            self.move(new_center[0] - center_x, new_center[1] - center_y)
        self.rot_origin = new_center
        pass

    def stroke_change(self, direction):
        """Change text size up or down."""
        self.pen.font_size += direction
        self.pen.font_size = max(8, min(128, self.pen.font_size))
 
    def resize_update(self, bbox):
        print("resizing text", bbox)
        if(bbox[2] < 0):
            bbox = (bbox[0], bbox[1], 10, bbox[3])
        if(bbox[3] < 0):
            print("flipping y")
            bbox = (bbox[0], bbox[1], bbox[2], 10)
        self.resizing["bbox"] = bbox

    def resize_end(self):
        new_bbox   = self.resizing["bbox"]
        old_bbox   = self.bb
        old_coords = self.coords

        if not self.font_extents:
            return None

        # create a surface with the new size
        surface = cairo.ImageSurface(cairo.Format.ARGB32, 
                                     2 * math.ceil(new_bbox[2]), 
                                     2 * math.ceil(new_bbox[3]))
        cr = cairo.Context(surface)
        min_fs, max_fs = 8, 154

        if new_bbox[2] < old_bbox[2] or new_bbox[3] < old_bbox[3]:
            dir = -1
        else:
            dir = 1

        self.coords = [ (0, 0), (old_bbox[2], old_bbox[3]) ]
        # loop while font size not larger than max_fs and not smaller than
        # min_fs
        print("resizing text, dir=", dir, "font size is", self.pen.font_size)
        while True:
            self.pen.font_size += dir
            print("trying font size", self.pen.font_size)
            self.draw(cr, False, False)
            if (self.pen.font_size < min_fs and dir < 0) or (self.pen.font_size > max_fs and dir > 0):
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
        return "\n".join(self.content)

    def strlen(self):
        return len(self.as_string())

    def add_text(self, text):
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
        self.content.insert(self.line + 1,
                            self.content[self.line][self.caret_pos:])
        self.content[self.line] = self.content[self.line][:self.caret_pos]
        self.line += 1
        self.caret_pos = 0

    def add_char(self, char):
        self.content[self.line] = self.content[self.line][:self.caret_pos] + char + self.content[self.line][self.caret_pos:]
        self.caret_pos += 1

    def move_caret(self, direction):
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
        if keyname == "BackSpace": # and cur["caret_pos"] > 0:
            self.backspace()
        elif keyname in ["Home", "End", "Down", "Up", "Right", "Left"]:
            self.move_caret(keyname)
        elif keyname == "Return":
            self.newline()
        elif char and char.isprintable():
            self.add_char(char)


    def draw_caret(self, cr, xx0, yy0, height):
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
        position = self.coords[0]
        content, pen, caret_pos = self.content, self.pen, self.caret_pos
        
        # get font info
        cr.select_font_face(pen.font_family, 
                            pen.font_style == "italic" and cairo.FONT_SLANT_ITALIC or cairo.FONT_SLANT_NORMAL,
                            pen.font_weight == "bold"  and cairo.FONT_WEIGHT_BOLD  or cairo.FONT_WEIGHT_NORMAL)
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

            x_bearing, y_bearing, t_width, t_height, x_advance, y_advance = cr.text_extents(fragment)

            bb_w = max(bb_w, t_width + x_bearing)
            bb_h += height

            cr.set_font_size(pen.font_size)
            cr.move_to(position[0], position[1] + dy)
            cr.set_source_rgba(*pen.color, pen.transparency)
            cr.show_text(fragment)
            cr.stroke()

            # draw the caret
            if caret_pos != None and i == self.line:
                x_bearing, y_bearing, t_width, t_height, x_advance, y_advance = cr.text_extents("|" + fragment[:caret_pos] + "|")
                x_bearing2, y_bearing2, t_width2, t_height2, x_advance2, y_advance2 = cr.text_extents("|")
                cr.set_source_rgb(1, 0, 0)
                xx0, yy0 = position[0] - x_bearing + t_width - 2 * t_width2, position[1] + dy - ascent
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
        print("finishing shape")
        self.coords, pressure = smooth_path(self.coords)
        #self.outline_recalculate_new()

    def update(self, x, y, pressure):
        self.path_append(x, y, pressure)

    def move(self, dx, dy):
        move_coords(self.coords, dx, dy)

    def rotate_end(self):
        self.coords = coords_rotate(self.coords, self.rotation, self.rot_origin)
        self.rotation = 0
        self.rot_origin = None
        self.bb = path_bbox(self.coords)

    def move(self, dx, dy):
        move_coords(self.coords, dx, dy)
        self.bb = None

    def is_close_to_click(self, click_x, click_y, threshold):
        bb = self.bbox()
        if bb is None:
            return False
        x, y, width, height = bb
        if click_x >= x and click_x <= x + width and click_y >= y and click_y <= y + height:
            return True
    
    def path_append(self, x, y, pressure):
        self.coords.append((x, y))
        self.bb = None

    def bbox(self):
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

    def rotate_end(self):
        # rotate all coords and outline
        self.coords  = coords_rotate(self.coords,  self.rotation, self.rot_origin)
        self.rotation   = 0
        self.rot_origin = None
        # recalculate bbox
        self.bb = path_bbox(self.coords)


    def draw_outline(self, cr):
        """draws each segment separately and makes a dot at each coord."""

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
        if len(self.coords) < 3:
            return

        if self.rotation != 0:
            cr.save()
            bb = self.bbox()
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
        return cls(path.coords, path.pen)

    @classmethod
    def from_object(cls, obj):
        print("Shape.from_object", obj)
        if obj.coords and len(obj.coords) > 2 and obj.pen:
            return cls(obj.coords, obj.pen)
        else:
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
        self.pen.stroke_change(direction)
        self.outline_recalculate_new()

    def smoothen(self, threshold=20):
        if len(self.coords) < 3:
            return
        print("smoothening path")
        self.coords, self.pressure = smooth_path(self.coords, self.pressure, 1)
        self.outline_recalculate_new()

    def outline_recalculate_new(self, coords = None, pressure = None):
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
        outline   = []

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

        self.outline_l, whatever = smooth_path(outline_l, None, 20)
        self.outline_r, whatever = smooth_path(outline_r, None, 20)
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
            self.bb = path_bbox(self.coords)
        return self.bb

    def resize_end(self):
        """recalculate the outline after resizing"""
        print("length of coords and pressure:", len(self.coords), len(self.pressure))
        old_bbox = self.bb or path_bbox(self.coords)
        new_coords = transform_coords(self.coords, old_bbox, self.resizing["bbox"])
        pressure   = self.pressure
        self.outline_recalculate_new(coords=new_coords, pressure=pressure)
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


    def draw_simple(self, cr, bbox=None):
        """draws the path as a single line. Useful for resizing."""

        if len(self.coords) < 2:
            return

        if bbox:
            old_bbox = path_bbox(self.coords)
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
        cr.set_fill_rule(cairo.FillRule.WINDING)

        cr.move_to(self.outline[0][0], self.outline[0][1])
        for point in self.outline[1:]:
            cr.line_to(point[0], point[1])
        cr.close_path()


    def draw(self, cr, hover=False, selected=False, outline = False):
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
        else:
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

        cr.set_source_rgb(*self.pen.color)
        x1, y1 = self.coords[0]
        x2, y2 = self.coords[1]
        w, h = (abs(x1 - x2), abs(y1 - y2))
        x0, y0 = (min(x1, x2), min(y1, y2))
        #cr.rectangle(x0, y0, w, h)
        cr.save()
        cr.translate(x0 + w / 2, y0 + h / 2)
        cr.scale(w / 2, h / 2)
        cr.arc(0, 0, 1, 0, 2 * 3.14159)

        if self.pen.fill_color:
            cr.set_source_rgb(*self.pen.fill_color)
            cr.fill_preserve()
        cr.restore()
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
        cr.set_source_rgb(*self.pen.color)

        if hover:
            cr.set_line_width(self.pen.line_width + 1)
        else:
            cr.set_line_width(self.pen.line_width)

        x1, y1 = self.coords[0]
        x2, y2 = self.coords[1]
        w, h = (abs(x1 - x2), abs(y1 - y2))
        x0, y0 = (min(x1, x2), min(y1, y2))

        if self.pen.fill_color:
            cr.set_source_rgb(*self.pen.fill_color)
            cr.rectangle(x0, y0, w, h)
            cr.fill()
            cr.stroke()

        cr.set_source_rgb(*self.pen.color)
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



## ---------------------------------------------------------------------
class MouseEvent:
    """
    Simple class for handling mouse events.

    Takes the event and computes a number of useful things.

    One advantage of using it: the computationaly intensive stuff is
    computed only once and only if it is needed.
    """
    def __init__(self, event, objects):
        self.event   = event
        self.objects = objects
        self._pos    = (event.x, event.y)
        self._hover  = None
        self._corner = None
        self._shift  = (event.state & Gdk.ModifierType.SHIFT_MASK) != 0
        self._ctrl   = (event.state & Gdk.ModifierType.CONTROL_MASK) != 0
        self._double = event.type == Gdk.EventType.DOUBLE_BUTTON_PRESS
        self._pressure = event.get_axis(Gdk.AxisUse.PRESSURE)
        if self._pressure is None:  # note that 0 is perfectly valid
            self._pressure = 1


    def hover(self):
        if not self._hover:
            self._hover = find_obj_close_to_click(self._pos[0], self._pos[1], self.objects, 20)
        return self._hover

    def corner(self):
        if not self._corner:
            self._corner = find_corners_next_to_click(self._pos[0], self._pos[1], self.objects, 20)
        return self._corner

    def pos(self):
        return self._pos

    def shift(self):
        return self._shift

    def ctrl(self):
        return self._ctrl

    def double(self):
        return self._double

    def pressure(self):
        return self._pressure


## ---------------------------------------------------------------------
class Wiglet:
    """drawable dialog-like objects on the canvas"""
    def __init__(self, type, coords):
        self.wiglet_type   = type
        self.coords = coords

    def draw(self, cr):
        raise NotImplementedError("draw method not implemented")

    def event_update(self, x, y):
        raise NotImplementedError("event_update method not implemented")

    def event_finish(self):
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
        cr.set_source_rgba(*self.pen.color, self.pen.transparency)
        draw_dot(cr, *self.coords, 50)

    def event_update(self, x, y):
        dx = x - self.coords[0]
        #print("changing transparency", dx)
        ## we want to change the transparency by 0.1 for every 20 pixels
        self.pen.transparency = max(0, min(1, self._initial_transparency + dx/500))
        #print("new transparency:", self.pen.transparency)

    def event_finish(self):
        pass

class WigletLineWidth(Wiglet):
    """Wiglet for changing the line width."""
    """directly operates on the pen of the object"""
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



## ---------------------------------------------------------------------
class help_dialog(Gtk.Dialog):
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
shift-click:  Enter text mode              click: Select object   Resizing: click in corner
                                           move: Move object      Rotating: ctrl-shift-click in corner
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
Ctrl-e: Export drawing to a file (png, jpeg, pdf)

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
    formats = {
        "All files": { "pattern": "*",      "mime_type": "application/octet-stream", "name": "all" },
        "PNG files":  { "pattern": "*.png",  "mime_type": "image/png",       "name": "png" },
        "JPEG files": { "pattern": "*.jpeg", "mime_type": "image/jpeg",      "name": "jpeg" },
        "PDF files":  { "pattern": "*.pdf",  "mime_type": "application/pdf", "name": "pdf" }
    }

    for name, data in formats.items():
        filter = Gtk.FileFilter()
        filter.set_name(name)
        filter.add_pattern(data["pattern"])
        filter.add_mime_type(data["mime_type"])
        dialog.add_filter(filter)

## ---------------------------------------------------------------------

def export_dialog(widget, parent = None):
    """Show a file chooser dialog to select a file to save the drawing as
    an image / pdf / svg."""
    file_name, selected_filter = None, None

    dialog = Gtk.FileChooserDialog(
        title="Save As", parent=parent, action=Gtk.FileChooserAction.SAVE)
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
        selected_filter = formats[selected_filter]["name"]
        print(f"Save file as: {file_name}, Format: {selected_filter}")

    dialog.destroy()
    return file_name, selected_filter

def import_image_dialog(widget, parent = None):
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

def FontChooser(pen, parent = None):

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


def ColorChooser(parent = None):
    """Select a color for drawing."""
    # Create a new color chooser dialog
    color_chooser = Gtk.ColorChooserDialog("Select Current Foreground Color", parent = parent)

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
        self._cursors = {
            "hand":        Gdk.Cursor.new_from_name(window.get_display(), "hand1"),
            "move":        Gdk.Cursor.new_from_name(window.get_display(), "hand2"),
            "grabbing":    Gdk.Cursor.new_from_name(window.get_display(), "grabbing"),
            "moving":      Gdk.Cursor.new_from_name(window.get_display(), "grab"),
            "text":        Gdk.Cursor.new_from_name(window.get_display(), "text"),
            "eraser":      Gdk.Cursor.new_from_name(window.get_display(), "not-allowed"),
            "pencil":      Gdk.Cursor.new_from_name(window.get_display(), "pencil"),
            "picker":      Gdk.Cursor.new_from_name(window.get_display(), "crosshair"),
            "colorpicker": Gdk.Cursor.new_from_name(window.get_display(), "crosshair"),
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
        return self._pos

    def update_pos(self, x, y):
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
        print("reverting cursor")
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
        _history_stack (list): The list of commands in the history.
        _hidden (bool): True if the objects are hidden.
        _current_object (Drawable): The current object.
        _wiglet_active (Drawable): The active wiglet.
        _resizeobj (Drawable): The object being resized.
        _mode (str): The current mode.
        _cursor (CursorManager): The cursor manager.
        _hover (Drawable): The object being hovered over.
        _clipboard (Clipboard): The clipboard.
        _selection_tool (SelectionTool): The selection tool.
    """

    def __init__(self, window):
        # public attr
        self.clipboard = Clipboard()

        # private attr
        self._objects    = []
        self._history    = []
        self._redo_stack = []
        self.selection = SelectionObject(self._objects)

        self._hidden = False
        self._current_object = None
        self._wiglet_active = None
        self._resizeobj = None
        self._hover = None
        self._selection_tool = None

    def objects(self):
        """Return the list of objects."""
        return self._objects

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
        self._history.append(TransmuteCommand(objects, self._objects, mode, self.selection.objects))
        # XXX the problem is that we need to remove the old objects from the
        # selection as well. However, it turns out to be more complicated than

    def transmute_selection(self, mode):
        """
        Transmute the selected objects to the given mode.

        Args:
            mode ( str ): The mode to transmute to.
        """
        if self.selection.is_empty():
            return
        self.transmute(self.selection.objects, mode)

    def set_objects(self, objects):
        """Set the list of objects."""
        ## no undo
        print("GOM: setting n=", len(objects), "objects")
        self._objects = objects
        self.selection = SelectionObject(self._objects)

    def add_object(self, obj):
        """Add an object to the list of objects."""
        self._history.append(AddCommand(obj, self._objects))
        ##self._objects.append(obj)

    def export_objects(self):
        objects = [ obj.to_dict() for obj in self._objects ]
        return objects

    def kill_object(self, obj):
        """Directly remove an object from the list of objects."""
        self._objects.remove(obj)
        ##self._objects.remove(obj)

    def selected_objects(self):
        """Return the selected objects."""
        return self.selection.objects

    def remove_selection(self):
        """Remove the selected objects from the list of objects."""
        if self.selection.is_empty():
            return
        self._history.append(RemoveCommand(self.selection.objects, self._objects))
        self.selection.clear()

    def remove_objects(self, objects, clear_selection = False):
        """Remove an object from the list of objects."""
        self._history.append(RemoveCommand(objects, self._objects))
        if clear_selection:
            self.selection.clear()
        ##self._objects.remove(obj)

    def remove_all(self):
        """Clear the list of objects."""
        self._history.append(RemoveCommand(self._objects[:], self._objects))

    def command_append(self, command_list):
        """Append a group of commands to the history."""
        ## append in reverse order
        self._history.append(CommandGroup(command_list[::-1]))

    def hide(self):
        """Hide the objects."""
        self._hidden = True

    def show(self):
        """Show the objects."""
        self._hidden = False

    def toggle_visibility(self):
        """Toggle the visibility of the objects."""
        self._hidden = not self._hidden

    def selection_group(self):
        """Group selected objects."""
        if self.selection.n() < 2:
            return
        print("Grouping", self.selection.n(), "objects")
        objects = sort_by_stack(self.selection.objects, self._objects)
        new_grp_obj = DrawableGroup(objects)

        for obj in self.selection.objects:
            self._objects.remove(obj)

        # XXX history append CommandGroup: Remove obj + add group
        self._objects.append(new_grp_obj)
        self.selection.set([ new_grp_obj ])

    def selection_ungroup(self):
        """Ungroup selected objects."""
        if self.selection.is_empty():
            return
        for obj in self.selection.objects:
            if obj.type == "group":
                print("Ungrouping", obj)
                self._objects.extend(obj.objects)
                self._objects.remove(obj)
        return

    def select_reverse(self):
        """Reverse the selection."""
        self.selection.reverse()

    def select_all(self):
        """Select all objects."""
        if not self._objects:
            return

        self.selection.all()

    def selection_delete(self):
        """Delete selected objects."""
        if self.selection.objects:
            self._history.append(RemoveCommand(self.selection.objects, self._objects))
            self.selection.clear()

    def select_next_object(self):
        """Select the next object."""
        self.selection.next()

    def selection_fill(self, color):
        """Fill the selected object."""
        for obj in self.selection.objects:
            obj.fill(color)

    def select_previous_object(self):
        """Select the previous object."""
        self.selection.previous()

    def selection_color_set(self, color):
        """Set the color of the selected objects."""
        if not self.selection.is_empty():
            self._history.append(SetColorCommand(self.selection, color))

    def selection_font_set(self, font_description):
        for obj in self.selection.objects:
            obj.pen.font_set_from_description(font_description)

    def do(self, command):
        """Do a command."""
        self._history.append(command)

    def redo(self):
        """Redo the last action."""
        print("Redo stack, size is", len(self._redo_stack))
        if self._redo_stack:
            command = self._redo_stack.pop()
            command.redo()
            self._history.append(command)

    def undo(self):
        """Undo the last action."""
        print("Undo, history size is", len(self._history))
        if self._history:
            command = self._history.pop()
            command.undo()
            self._redo_stack.append(command)

    def move_obj(self, obj, dx, dy):
        """Move the object by the given amount."""
        eventObj = MoveCommand(obj, (0, 0))
        eventObj.event_update(dx, dy)
        self._history.append(eventObj)

    def move_selection(self, dx, dy):
        """Move the selected objects by the given amount."""
        if self.selection.is_empty():
            return
        self.move_obj(self.selection, dx, dy)

    def rotate_obj(self, obj, angle):
        """Rotate the object by the given angle (degrees)."""
        print("rotating by", angle)
        eventObj = RotateCommand(obj, angle=math.radians(angle))
        eventObj.event_finish()
        self._history.append(eventObj)

    def rotate_selection(self, angle):
        """Rotate the selected objects by the given angle (degrees)."""
        if self.selection.is_empty():
            return
        self.rotate_obj(self.selection, angle)

    def selection_zmove(self, operation):
        """move the selected objects long the z-axis."""
        if self.selection.is_empty():
            return
        self._history.append(ZStackCommand(self.selection.objects, self._objects, operation))



def export_image(width, height, filename, draw_func, file_format = "all"):
    """Export the drawing to a file."""
    # Create a Cairo surface of the same size as the window content

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

    if file_format == "png":
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    elif file_format == "svg":
        surface = cairo.SVGSurface(filename, width, height)
    elif file_format == "pdf":
        surface = cairo.PDFSurface(filename, width, height)
    else:
        raise ValueError("Invalid file format: " + file_format)

    cr = cairo.Context(surface)
    cr.set_source_rgba(1, 1, 1)
    cr.paint()
    draw_func(cr)

    # Save the surface to the file
    if file_format == "png":
        surface.write_to_png(filename)
    elif file_format in [ "svg", "pdf" ]:
        surface.finish()

def save_file_as_sdrw(filename, config, objects):
    """Save the objects to a file in native format."""
    state = { 'config': config, 'objects': objects }
    try:
        with open(filename, 'wb') as f:
            #yaml.dump(state, f)
            pickle.dump(state, f)
        print("Saved drawing to", filename)
        return True
    except Exception as e:
        print("Error saving file:", e)
        return False

def read_file_as_sdrw(filename):
    """Read the objects from a file in native format."""
    if not path.exists(filename):
        print("No saved drawing found at", filename)
        return None, None

    config, objects = None, None

    try:
        with open(filename, "rb") as file:
            state = pickle.load(file)
            objects = [ Drawable.from_dict(d) for d in state['objects'] ] or [ ]
            config = state['config']
    except Exception as e:
        print("Error reading file:", e)
        return None, None
    return config, objects

# the design of the app is as follows: the EventManager class is a singleton
# that manages the events and actions of the app. The actions are defined in
# the actions_dictionary method. 
#
# So the EM is a know-it-all class, and the others (GOM, App) are just
# listeners to the EM. The EM is the one that knows what to do when an event
# happens.


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

    def __init__(self, gom, app):
        # singleton pattern
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self.__gom = gom
            self.__app = app
            self.make_actions_dictionary(gom, app)
            self.make_default_keybindings()

    def dispatch_action(self, action_name, **kwargs):
        """
        Dispatches an action by name.
        """
        print("dispatch_action", action_name)
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
            print(f"Error while dispatching action {action_name}: {e}")

    def dispatch_key_event(self, key_event, mode):
        """
        Dispatches an action by key event.
        """
        print("dispatch_key_event", key_event, mode)

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

        print("dispatching action", action_name)
        self.dispatch_action(action_name)

    def on_key_press(self, widget, event):
        """
        This method is called when a key is pressed.
        """

        gom, app = self.__gom, self.__app

        keyname = Gdk.keyval_name(event.keyval)
        char    = chr(Gdk.keyval_to_unicode(event.keyval))
        ctrl    = event.state & Gdk.ModifierType.CONTROL_MASK
        shift   = event.state & Gdk.ModifierType.SHIFT_MASK
        alt_l   = event.state & Gdk.ModifierType.MOD1_MASK
        print("keyname", keyname, "char", char, "ctrl", ctrl, "shift", shift, "alt_l", alt_l)

        mode = app.get_mode()

        keyfull = keyname

        if char.isupper():
            keyfull = keyname.lower()
        if shift:
            keyfull = "Shift-" + keyfull
        if ctrl:
            keyfull = "Ctrl-" + keyfull
        if alt_l:
            keyfull = "Alt-" + keyfull
        print("keyfull", keyfull)

        # first, check whether there is a current object being worked on
        # and whether this object is a text object. In that case, we only
        # call the ctrl keybindings and pass the rest to the text object.
        if app.current_object and app.current_object.type == "text" and not(ctrl or keyname == "Escape"):
            print("updating text input")
            app.current_object.update_by_key(keyname, char)
            app.queue_draw()
            return

        # otherwise, we dispatch the key event
        self.dispatch_key_event(keyfull, mode)

        # XXX this probably should be somewhere else
        app.queue_draw()


    def make_actions_dictionary(self, gom, app):
        """
        This dictionary maps key events to actions.
        """
        self.__actions = {
            'mode_draw':             {'action': app.set_mode, 'args': ["draw"]},
            'mode_box':              {'action': app.set_mode, 'args': ["box"]},
            'mode_circle':           {'action': app.set_mode, 'args': ["circle"]},
            'mode_move':             {'action': app.set_mode, 'args': ["move"]},
            'mode_text':             {'action': app.set_mode, 'args': ["text"]},
            'mode_select':           {'action': app.set_mode, 'args': ["select"]},
            'mode_eraser':           {'action': app.set_mode, 'args': ["eraser"]},
            'mode_shape':            {'action': app.set_mode, 'args': ["shape"]},
            'mode_colorpicker':      {'action': app.set_mode, 'args': ["colorpicker"]},

            'finish_text_input':     {'action': app.finish_text_input},

            'show_help_dialog':      {'action': app.show_help_dialog},
            'app_exit':              {'action': app.exit},

            'clear_page':            {'action': app.clear},
            'cycle_bg_transparency': {'action': app.cycle_background},
            'toggle_outline':        {'action': app.outline_toggle},

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

            'set_color_white':       {'action': app.set_color, 'args': [COLORS["white"]]},
            'set_color_black':       {'action': app.set_color, 'args': [COLORS["black"]]},
            'set_color_red':         {'action': app.set_color, 'args': [COLORS["red"]]},
            'set_color_green':       {'action': app.set_color, 'args': [COLORS["green"]]},
            'set_color_blue':        {'action': app.set_color, 'args': [COLORS["blue"]]},
            'set_color_yellow':      {'action': app.set_color, 'args': [COLORS["yellow"]]},
            'set_color_cyan':        {'action': app.set_color, 'args': [COLORS["cyan"]]},
            'set_color_magenta':     {'action': app.set_color, 'args': [COLORS["magenta"]]},
            'set_color_purple':      {'action': app.set_color, 'args': [COLORS["purple"]]},
            'set_color_grey':        {'action': app.set_color, 'args': [COLORS["grey"]]},

            # dialogs
            "export_drawing":        {'action': app.export_drawing},
            "select_color":          {'action': app.select_color},
            "select_font":           {'action': app.select_font},
            "import_image":          {'action': app.select_image_and_create_pixbuf},
            "toggle_pens":           {'action': app.switch_pens},
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

#            'Ctrl-m':               {'action': self.smoothen,           'modes': ["move"]},
            'copy_content':          {'action': app.copy_content,        'modes': ["move"]},
            'cut_content':           {'action': app.cut_content,         'modes': ["move"]},
            'paste_content':         {'action': app.paste_content},
            'screenshot':            {'action': app.screenshot},

            'stroke_increase':       {'action': app.stroke_increase},
            'stroke_decrease':       {'action': app.stroke_decrease},
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
            'Shift-p':              "set_color_purple",

            'Ctrl-e':               "export_drawing",
            'Ctrl-k':               "select_color",
            'Ctrl-f':               "select_font",
            'Ctrl-i':               "import_image",
            'Ctrl-p':               "toggle_pens",
            'Ctrl-o':               "open_drawing",

            'Tab':                  "select_next_object",
            'Shift-ISO_Left_Tab':   "select_previous_object",
            'g':                    "selection_group",
            'u':                    "selection_ungroup",
            'Delete':               "selection_delete",

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



class MenuMaker:
    """A class holding methods to create menus. Singleton."""
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(MenuMaker, cls).__new__(cls)
        return cls._instance

    def __init__(self, gom, em, app):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self.__gom = gom # GraphicObjectManager
            self.__app = app # App
            self.__em  = em  # EventManager
            self.__context_menu = None
            self.__object_menu = None

    def build_menu(self, menu_items):
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
        print("Menu item activated:", action)

        self.__em.dispatch_action(action)
        self.__app.queue_draw()


    def context_menu(self):
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
                { "label": "Tggl outline    (Ctrl-b)",  "callback": self.on_menu_item_activated, "action": "toggle_outline" },
                { "separator": True },
                { "label": "Color           (Ctrl-k)",  "callback": self.on_menu_item_activated, "action": "select_color" },
                { "label": "Font            (Ctrl-f)",  "callback": self.on_menu_item_activated, "action": "select_font" },
                { "separator": True },
                { "label": "Open drawing    (Ctrl-o)",  "callback": self.on_menu_item_activated, "action": "open_drawing" },
                { "label": "Image from file (Ctrl-i)",  "callback": self.on_menu_item_activated, "action": "import_image" },
                { "label": "Screenshot      (Ctrl-Shift-f)",  "callback": self.on_menu_item_activated, "action": "screenshot" },
                { "label": "Export drawing  (Ctrl-e)",  "callback": self.on_menu_item_activated, "action": "export_drawing" },
                { "label": "Help            [F1]",      "callback": self.on_menu_item_activated, "action": "show_help_dialog" },
                { "label": "Quit            (Ctrl-q)",  "callback": self.on_menu_item_activated, "action": "app_exit" },
        ]

        self.__context_menu = self.build_menu(menu_items)
        return self.__context_menu

    def object_menu(self, objects):
        # when right-clicking on an object
        menu_items = [
                { "label": "Copy (Ctrl-c)",        "callback": self.on_menu_item_activated, "action": "copy_content" },
                { "label": "Cut (Ctrl-x)",         "callback": self.on_menu_item_activated, "action": "cut_content" },
                { "separator": True },
                { "label": "Delete (|Del|)",       "callback": self.on_menu_item_activated, "action": "selection_delete" },
                { "label": "Group (g)",            "callback": self.on_menu_item_activated, "action": "selection_group" },
                { "label": "Ungroup (u)",          "callback": self.on_menu_item_activated, "action": "selection_ungroup" },
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

    def __init__(self, savefile = None):
        super(TransparentWindow, self).__init__()

        self.savefile            = savefile
        self.init_ui()
    
    def init_ui(self):
        self.set_title("Transparent Drawing Window")
        self.set_decorated(False)
        self.connect("destroy", self.exit)
        self.set_default_size(800, 600)

        # transparency
        screen = self.get_screen()
        visual = screen.get_rgba_visual()
        if visual != None and screen.is_composited():
            self.set_visual(visual)
        self.set_app_paintable(True)

        # autosave
        GLib.timeout_add(AUTOSAVE_INTERVAL, self.autosave)

        # Drawing setup
        self.mode                = "draw"
        self.gom                 = GraphicsObjectManager(self)
        self.em                  = EventManager(gom = self.gom, app = self)
        self.clipboard           = Clipboard()
        self.cursor              = CursorManager(self)
        self.mm                  = MenuMaker(self.gom, self.em, self)
        self.hidden              = False

        self.current_object      = None
        self.wiglet_active       = None
        self.resizeobj           = None
        self.hover               = None
        self.selection_tool      = None

        # defaults for drawing
        self.pen  = Pen(line_width = 4,  color = (0.2, 0, 0), font_size = 24, transparency  = 1)
        self.pen2 = Pen(line_width = 40, color = (1, 1, 0),   font_size = 24, transparency = .2)
        self.transparent = 0
        self.outline     = False

        # distance for selecting objects
        self.max_dist   = 15

        self.objects = [ ]
        self.load_state()
        self.modified = False # for autosave

        # connecting events
        self.connect("draw", self.on_draw)
        self.connect("key-press-event", self.em.on_key_press)
        self.set_events(Gdk.EventMask.BUTTON_PRESS_MASK | Gdk.EventMask.BUTTON_RELEASE_MASK | Gdk.EventMask.POINTER_MOTION_MASK)
        self.connect("button-press-event",   self.on_button_press)
        self.connect("button-release-event", self.on_button_release)
        self.connect("motion-notify-event",  self.on_motion_notify)

        self.set_keep_above(True)
        self.maximize()



    def exit(self):
        ## close the savefile_f
        print("Exiting")
        self.save_state()
        Gtk.main_quit()

    def on_draw(self, widget, cr):
        """Handle draw events."""
        if self.hidden:
            print("I am hidden!")
            return True

        cr.set_source_rgba(.8, .75, .65, self.transparent)
        cr.set_operator(cairo.OPERATOR_SOURCE)
        cr.paint()
        cr.set_operator(cairo.OPERATOR_OVER)
        self.draw(cr)

        if self.current_object:
            self.current_object.draw(cr)

        if self.wiglet_active:
            self.wiglet_active.draw(cr)
        return True

    def draw(self, cr):
        """Draw the objects in the given context. Used also by export functions."""

        for obj in self.gom.objects():
            hover    = obj == self.hover and self.mode == "move"
            selected = self.gom.selection.contains(obj) and self.mode == "move"
            obj.draw(cr, hover=hover, selected=selected, outline = self.outline)
    #   if self.current_object:
    #       print("drawing current object:", self.current_object, "mode:", self.mode)
    #       self.current_object.draw(cr)

        # If changing line width, draw a preview of the new line width
      
    def clear(self):
        """Clear the drawing."""
        self.gom.selection.clear()
        self.resizeobj      = None
        self.current_object = None
        self.gom.remove_all()
        self.queue_draw()

    # ---------------------------------------------------------------------
    #                              Event handlers

    def on_right_click(self, event, hover_obj):
        """Handle right click events - context menus."""
        if hover_obj:
            self.mode = "move"
            self.cursor.default(self.mode)

            if not self.gom.selection.contains(hover_obj):
                self.gom.selection.set([ hover_obj ])

            self.mm.object_menu(self.gom.selected_objects()).popup(None, None, None, None, event.button, event.time)
        else:
            self.mm.context_menu().popup(None, None, None, None, event.button, event.time)
        self.queue_draw()

    # ---------------------------------------------------------------------

    def move_resize_rotate(self, ev):
        """Process events for moving, resizing and rotating objects."""
        corner_obj = ev.corner()
        hover_obj  = ev.hover()
        pos = ev.pos()
        shift, ctrl = ev.shift(), ev.ctrl()

        if corner_obj[0] and corner_obj[0].bbox():
            print("starting resize")
            obj    = corner_obj[0]
            corner = corner_obj[1]
            print("ctrl:", ctrl, "shift:", shift)
            # XXX this code here is one of the reasons why rotating or resizing the
            # whole selection does not work. The other reason is that the
            # selection object itself is not considered when detecting
            # hover or corner objects.
            if ctrl and shift:
                self.resizeobj = RotateCommand(obj, origin = pos, corner = corner)
            else:
                self.resizeobj = ResizeCommand(obj, origin = pos, corner = corner, proportional = ctrl)
            self.gom.selection.set([ obj ])
            # XXX - this should happen through GOM and upon mouse release 
            # self.history.append(self.resizeobj)
            self.cursor.set(corner)
        elif hover_obj:
            if ev.shift():
                # add if not present, remove if present
                print("adding object", hover_obj)
                self.gom.selection.add(hover_obj)
            if not self.gom.selection.contains(hover_obj):
                print("object not in selection, setting it", hover_obj)
                self.gom.selection.set([ hover_obj ])
            self.resizeobj = MoveCommand(self.gom.selection, pos)
            # XXX - this should happen through GOM and upon mouse release 
            # self.history.append(self.resizeobj)
            self.cursor.set("grabbing")
        else:
            self.gom.selection.clear()
            self.resizeobj   = None
            print("starting selection tool")
            self.selection_tool = SelectionTool([ pos, (pos[0] + 1, pos[1] + 1) ])
            self.current_object = self.selection_tool
            self.queue_draw()
        return True

    def create_object(self, ev):
        """Create an object based on the current mode."""
        # not managed by GOM: first create, then decide whether to add to GOM
        obj = DrawableFactory.create_drawable(self.mode, pen = self.pen, ev=ev)
        if obj:
            self.current_object = obj

    def set_mode(self, mode):
        """Set the mode."""
        self.mode = mode
        self.cursor.default(self.mode)

    def get_mode(self):
        """Get the current mode."""
        return self.mode

    # XXX this code should be completely rewritten, cleaned up, refactored
    # and god knows what else. It's a mess.
    def on_button_press(self, widget, event):
        print("on_button_press: type:", event.type, "button:", event.button, "state:", event.state)
        self.modified = True # better safe than sorry

        ev = MouseEvent(event, self.gom.objects())
        shift, ctrl, pressure = ev.shift(), ev.ctrl(), ev.pressure()
        hover_obj = ev.hover()

        # double click on a text object: start editing
        if event.button == 1 and ev.double() and hover_obj and hover_obj.type == "text" and self.mode in ["draw", "text", "move"]:
            # put the cursor in the last line, end of the text
            # this should be a Command event
            hover_obj.move_caret("End")
            self.current_object = hover_obj
            self.queue_draw()
            self.cursor.set("none")
            return True

        # Ignore clicks when text input is active
        if self.current_object:
            if  self.current_object.type == "text":
                print("click, but text input active - finishing it first")
                self.finish_text_input()
            else:
                print("click, but text input active - ignoring it; object=", self.current_object)
            return True

        # right click: open context menu
        if event.button == 3 and not shift:
            self.on_right_click(event, hover_obj)
            return True

        if event.button != 1:
            return True

        # Start changing line width: single click with ctrl pressed
        if ctrl and event.button == 1 and self.mode == "draw": 
            if not shift: 
                self.wiglet_active = WigletLineWidth((event.x, event.y), self.pen)
            else:
                self.wiglet_active = WigletTransparency((event.x, event.y), self.pen)
            return True

        if self.mode == "colorpicker":
            #print("picker mode")
            color = get_color_under_cursor()
            self.set_color(color) 
            color_hex = rgb_to_hex(color)
            self.clipboard.set_text(color_hex)
            return True

        elif self.mode == "move":
            return self.move_resize_rotate(ev)

        # erasing an object, if an object is underneath the cursor
        elif self.mode == "eraser" and hover_obj: 
                ## XXX -> GOM 
                # self.history.append(RemoveCommand([ hover_obj ], self.objects))
                self.gom.remove_objects([ hover_obj ], clear_selection = True)
                self.resizeobj   = None
                self.cursor.revert()

        # simple click: create modus
        else:
            self.create_object(ev)

        self.queue_draw()

        return True

    # Event handlers
    # XXX same comment as above
    def on_button_release(self, widget, event):
        """Handle mouse button release events."""
        obj = self.current_object

        if obj and obj.type in [ "shape", "path" ]:
            print("finishing path / shape")
            obj.path_append(event.x, event.y, 0)
            obj.finish()
            if len(obj.coords) < 3:
                obj = None
            self.queue_draw()

        if obj:
            # remove objects that are too small
            bb = obj.bbox()
            if bb and obj.type in [ "box", "circle" ] and bb[2] == 0 and bb[3] == 0:
                obj = None

        if obj:
            if obj != self.selection_tool:
                self.gom.add_object(obj)
            else:
                self.current_object = None

        if self.wiglet_active:
            self.wiglet_active.event_finish()
            self.wiglet_active = None
            self.queue_draw()
            return True

        # if selection tool is active, finish it
        if self.selection_tool:
            print("finishing selection tool")
            #self.objects.remove(self.selection_tool)
            #bb = self.selection_tool.bbox()
            objects = self.selection_tool.objects_in_selection(self.gom.objects())
            if len(objects) > 0:
                self.gom.selection.set(objects)
            else:
                self.gom.selection.clear()
            self.selection_tool = None
            self.queue_draw()
            return True

        # if the user clicked to create a text, we are not really done yet
        if self.current_object and self.current_object.type != "text":
            print("there is a current object: ", self.current_object)
            self.gom.selection.clear()
            self.current_object = None
            self.queue_draw()
            return True

        if self.resizeobj:
            # If the user was dragging a selected object and the drag ends
            # in the lower left corner, delete the object
            self.resizeobj.event_finish()
            obj = self.resizeobj.obj
            if self.resizeobj.command_type == "move" and  event.x < 10 and event.y > self.get_size()[1] - 10:
                # command group because these are two commands: first move,
                # then remove
                self.gom.command_append([ self.resizeobj, RemoveCommand([ obj ], self.gom.objects()) ])
                self.selection.clear()
            else:
                self.gom.command_append([ self.resizeobj ])
            self.resizeobj    = None
            self.cursor.revert()
            self.queue_draw()
        return True


    def on_motion_notify(self, widget, event):
        """Handle mouse motion events."""

        ev = MouseEvent(event, self.gom.objects())
        x, y = ev.pos()
        self.cursor.update_pos(x, y)

        if self.wiglet_active:
            self.wiglet_active.event_update(x, y)
            self.queue_draw()
            return True

        obj = self.current_object or self.selection_tool

        if obj:
            obj.update(x, y, ev.pressure())
            self.queue_draw()
        elif self.resizeobj:
            self.resizeobj.event_update(x, y)
            self.queue_draw()
        elif self.mode == "move":
            object_underneath = ev.hover()
            prev_hover = self.hover

            if object_underneath:
                if object_underneath.type == "text":
                    self.cursor.set("text")
                else:
                    self.cursor.set("moving")
                self.hover = object_underneath
            else:
                self.cursor.revert()
                self.hover = None

            corner_obj = ev.corner()

            if corner_obj[0] and corner_obj[0].bbox():
                self.cursor.set(corner_obj[1])
                self.hover = corner_obj[0]
                self.queue_draw()

            if prev_hover != self.hover:
                self.queue_draw()

        # stop event propagation
        return True

    # ---------------------------------------------------------------------

    def finish_text_input(self):
        """Clean up current text and finish text input."""
        print("finishing text input")
        if self.current_object and self.current_object.type == "text":
            self.current_object.caret_pos = None
            if self.current_object.strlen() == 0:
                self.gom.kill_object(self.current_object)
            self.current_object = None
        self.cursor.revert()

    def update_text_input(self, keyname, char):
        """Update the current text input."""
        cur  = self.current_object
        if not cur:
            raise ValueError("No current object")
        cur.update_by_key(keyname, char)

    def cycle_background(self):
        """Cycle through background transparency."""
        self.transparent = {1: 0, 0: 0.5, 0.5: 1}[self.transparent]

    def paste_text(self, clip_text):
        """Enter some text in the current object or create a new object."""

        if self.current_object and self.current_object.type == "text":
            self.current_object.add_text(clip_text.strip())
        else:
            new_text = Text([ self.cursor.pos() ], 
                            pen = self.pen, content=clip_text.strip())
            self.gom.add_object(new_text)

    def paste_image(self, clip_img):
        """Create an image object from a pixbuf image."""
        obj = Image([ self.cursor.pos() ], self.pen, clip_img)
        self.gom.add_object(obj)

    def object_create_copy(self, obj, bb = None):
        """Copy the given object into a new object."""
        new_obj = copy.deepcopy(obj.to_dict())
        new_obj = Drawable.from_dict(new_obj)

        # move the new object to the current location
        x, y = self.cursor.pos()
        if bb is None:
            bb  = new_obj.bbox()
        new_obj.move(x - bb[0], y - bb[1])

        self.gom.add_object(new_obj)

    def paste_content(self):
        """Paste content from clipboard."""
        clip_type, clip = self.clipboard.get_content()

        if not clip:
            return

        # internal paste
        if clip_type == "internal":
            print("Pasting content internally")
            if clip.type != "group":
                Raise("Internal clipboard is not a group")
            bb = clip.bbox()
            print("clipboard bbox:", bb)
            for obj in clip.objects:
                self.object_create_copy(obj, bb)
            return

        elif clip_type == "text":
            self.paste_text(clip)
        elif clip_type == "image":
            self.paste_image(clip)

    def copy_content(self, destroy = False):
        """Copy content to clipboard."""
        if self.gom.selection.is_empty():
            # nothing selected
            print("Nothing selected")
            return

        print("Copying content", self.gom.selection)
        self.clipboard.copy_content(self.gom.selection)

        if destroy:
            self.gom.remove_selection()

    def cut_content(self):
        """Cut content to clipboard."""
        self.copy_content(True)
   
    def stroke_increase(self):
        """Increase whatever is selected."""
        self.stroke_change(1)

    def stroke_decrease(self):
        """Decrease whatever is selected."""
        self.stroke_change(-1)

    def stroke_change(self, direction):
        """Modify the line width or text size."""
        print("Changing stroke", direction)
        if self.current_object and self.current_object.type == "text":
            print("Changing text size")
            self.current_object.stroke_change(direction)
            self.pen.font_size = self.current_object.pen.font_size
        else: 
            for obj in self.gom.selected_objects():
                obj.stroke_change(direction)

        # without a selected object, change the default pen, but only if in the correct mode
        if self.mode == "draw":
            self.pen.line_width = max(1, self.pen.line_width + direction)
        elif self.mode == "text":
            self.pen.font_size = max(1, self.pen.font_size + direction)

    def outline_toggle(self):
        """Toggle outline mode."""
        self.outline = not self.outline

    def set_color(self, color):
        self.pen.color_set(color)
        self.gom.selection_color_set(color)

    def set_font(self, font_description):
        """Set the font."""
        self.pen.font_set_from_description(font_description)
        self.gom.selection_font_set(font_description)
        if self.current_object and self.current_object.type == "text":
            self.current_object.pen.font_set_from_description(font_description)

    def transmute(self, mode):
        """Change the selected object(s) to a shape."""
        print("transmuting to", mode)
        sel = self.gom.selected_objects()
        # note to self: sel is a list, not the selection
        if sel:
            self.gom.transmute(sel, mode)

#   def smoothen(self):
#       """Smoothen the selected object."""
#       if self.selection.n() > 0:
#           for obj in self.selection.objects:
#               obj.smoothen()

    def switch_pens(self):
        """Switch between pens."""
        self.pen, self.pen2 = self.pen2, self.pen
        self.queue_draw()

    def select_color(self):
        """Select a color for drawing."""
        color = ColorChooser(self)
        if color:
            self.set_color((color.red, color.green, color.blue))

    def select_font(self):
        font_description = FontChooser(self.pen, self)

        if font_description:
            self.set_font(font_description)

    def show_help_dialog(self):
        """Show the help dialog."""
        dialog = help_dialog(self)
        response = dialog.run()
        dialog.destroy()

    def export_drawing(self):
        """Save the drawing to a file."""
        # Choose where to save the file
        #    self.export(filename, "svg")
        file_name, file_format = export_dialog(self)
        width, height = self.get_size()
        export_image(width, height, file_name, self.draw, file_format)

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
                img = Image([ pos ], self.pen, pixbuf)
                self.gom.add_object(img)
                self.queue_draw()
        
        return pixbuf

    def screenshot_finalize(self, bb):
        print("Taking screenshot now")
        pixbuf, filename = get_screenshot(self, bb[0] - 3, bb[1] - 3, bb[0] + bb[2] + 6, bb[1] + bb[3] + 6)
        self.hidden = False
        self.queue_draw()

        # Create the image and copy the file name to clipboard
        if pixbuf is not None:
            img = Image([ (bb[0], bb[1]) ], self.pen, pixbuf)
            self.gom.add_object(img)
            self.queue_draw()
            self.clipboard.set_text(filename)

    def find_screenshot_box(self):
        if self.current_object and self.current_object.type == "box":
            return self.current_object
        if self.gom.selection.n() == 1 and self.gom.selected_objects()[0].type == "box":
            return self.gom.selection.objects()[0]

        for obj in self.gom.objects()[::-1]:
            if obj.type == "box":
                return obj
        return None

    def screenshot(self):
        obj = self.find_screenshot_box()
        if not obj:
            print("no suitable box found")
            return

        bb = obj.bbox()
        print("bbox is", bb)
        self.hidden = True
        self.queue_draw()
        while Gtk.events_pending():
            Gtk.main_iteration_do(False)
        GLib.timeout_add(100, self.screenshot_finalize, bb)

    def autosave(self):
        # XXX: not implemented, tracking modifications of state
        if not self.modified:
           return

        if self.current_object: # not while drawing!
            return

        print("Autosaving")
        self.save_state()
        self.modified = False

    def save_state(self): 
        """Save the current drawing state to a file."""
        if not self.savefile:
            print("No savefile set")
            return

        print("savefile:", self.savefile)
        config = {
                'transparent': self.transparent,
                'pen': self.pen.to_dict(),
                'pen2': self.pen2.to_dict()
        }

        objects = self.gom.export_objects()
        save_file_as_sdrw(self.savefile, config, objects)

    def open_drawing(self):
        file_name = open_drawing_dialog(self)
        if self.read_file(file_name):
            print("Setting savefile to", file_name)
            self.savefile = file_name
            self.modified = True

    def read_file(self, filename, load_config = True):
        """Read the drawing state from a file."""
        config, objects = read_file_as_sdrw(filename)

        if objects:
            self.gom.set_objects(objects)

        if config and load_config:
            self.transparent       = config['transparent']
            self.pen               = Pen.from_dict(config['pen'])
            self.pen2              = Pen.from_dict(config['pen2'])
        if config and objects:
            self.modified = True
            return True
        return False

    def load_state(self):
        """Load the drawing state from a file."""
        self.read_file(self.savefile)


## ---------------------------------------------------------------------

if __name__ == "__main__":
    app_name   = "ScreenDrawer"
    app_author = "JanuaryWeiner"  # Optional; used on Windows

    # Get user-specific config directory
    user_config_dir = appdirs.user_config_dir(app_name, app_author)
    print(f"User config directory: {user_config_dir}")

    #user_cache_dir = appdirs.user_cache_dir(app_name, app_author)
    #user_log_dir   = appdirs.user_log_dir(app_name, app_author)


# ---------------------------------------------------------------------

    parser = argparse.ArgumentParser(description="Drawing on the screen")
    parser.add_argument("-s", "--savefile", help="File for automatic save upon exit")
    parser.add_argument("files", nargs="*")

    args     = parser.parse_args()
    savefile = args.savefile or get_default_savefile(app_name, app_author)
    files    = args.files
    print("Save file is:", savefile)

# ---------------------------------------------------------------------

    win = TransparentWindow(savefile = savefile)
    if files:
        win.read_file(files[0])

    css = b"""
    #myMenu {
        background-color: rgba(255, 255, 255, 0.9); /* Semi-transparent white */
        font-family: Monospace, 'Monospace regular', monospace, 'Courier New'; /* Use 'Courier New', falling back to any monospace font */
    }
    """

    style_provider = Gtk.CssProvider()
    style_provider.load_from_data(css)
    Gtk.StyleContext.add_provider_for_screen(
        Gdk.Screen.get_default(),
        style_provider,
        Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
    )

    win.show_all()
    win.present()
    win.cursor.set(win.mode)
    win.stick()

    Gtk.main()

