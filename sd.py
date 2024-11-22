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

# pylint: disable=unused-import
# pylint: disable=unused-wildcard-import
# pylint: disable=wildcard-import

import copy
import pickle
import traceback
import colorsys
import sys
from sys import exc_info, argv
import os
from os import path
import time
import math
import base64
import tempfile
import logging
import hashlib
from io import BytesIO

import warnings
import argparse

import pyautogui
from PIL import ImageGrab

import yaml
import cairo
import appdirs
import numpy as np

import gi
gi.require_version('Gtk', '3.0') # pylint: disable=wrong-import-position
from gi.repository import Gtk, Gdk, GdkPixbuf, Pango, GLib


# ---------------------------------------------------------------------
# These are various classes and utilities for the sd.py script. In the
# "skeleton" variant, they are just imported. In the "full" variant, they
# the files are directly, physically inserted in order to get one big fat
# Python script that can just be copied.


"""
General utility functions for the ScreenDrawer application.
"""

FRAC_FWD = np.array([1/50, 2/10, 1/2])
FRAC_BCK = np.array([-1/50, -2/10])

def get_default_savefile(app_name, app_author):
    """Get the default save file for the application."""

    # Get user-specific data directory
    user_data_dir = appdirs.user_data_dir(app_name, app_author)
    log.debug("User data directory: %s", user_data_dir)
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

def get_cursor_position(window):
    """Get the cursor position relative to the window."""

    screen = window.get_screen()
    window = window.get_window()
    display = screen.get_display()
    seat = display.get_default_seat()
    device = seat.get_pointer()

    # Get the cursor position
    _, x, y, _ = window.get_device_position(device)
    log.debug("cursor pos %s %s", x, y)

    return (x, y)

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

    log.debug("Saved screenshot to temporary file: %s", temp_file_name)

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

def swap_stacks(stack1, stack2):
    """Swap two stacks"""
    stack1[:], stack2[:] = stack2[:], stack1[:]

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
        return True, (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

    return (False, None)  # No intersection

def pp(p):
    """return point in integers"""
    return [int(p[0]), int(p[1])]

def calculate_length(coords):
    """Sum up the lengths of the segments in a path."""
    length = 0
    for i in range(len(coords) - 1):
        length += distance(coords[i], coords[i + 1])
    return length

def first_point_after_length(coords, length):
    """Return the index of the first point after a given length."""
    total_length = 0
    for i in range(len(coords) - 1):
        total_length += distance(coords[i], coords[i + 1])
        if total_length >= length:
            return i + 1
    return len(coords) - 1

def calculate_angle2(p0, p1):
    """Calculate angle between vectors given by p0 and p1"""
    x1, y1 = p0
    x2, y2 = p1
    dot = x1 * x2 + y1 * y2
    det = x1 * y2 - y1 * x2
    angle = math.atan2(det, dot)
    return angle

def calc_intersect(seg1, seg2):
    """Calculate intersection point between two line segments.
       Each segment consists of two points, start and end"""

    x1, y1 = seg1[0]
    x2, y2 = seg1[1]
    x3, y3 = seg2[0]
    x4, y4 = seg2[1]

    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if denom == 0:
        return None  # Lines are parallel

    t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
    u = ((x1-x3)*(y1-y2) - (y1-y3)*(x1-x2)) / denom

    if 0 <= t <= 1 and 0 <= u <= 1:
        # The segments intersect
        intersect_x = x1 + t * (x2 - x1)
        intersect_y = y1 + t * (y2 - y1)
        return (intersect_x, intersect_y)

    return None  # No intersection

def calc_intersect_2(seg1, seg2):
    """Calculate intersection of two infinite lines given by two points each."""

    x1, y1 = seg1[0]
    x2, y2 = seg1[1]
    x3, y3 = seg2[0]
    x4, y4 = seg2[1]

    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if denom == 0:
        return None  # Lines are parallel

    t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
    # u = ((x1-x3)*(y1-y2) - (y1-y3)*(x1-x2)) / denom

    intersect_x = x1 + t * (x2 - x1)
    intersect_y = y1 + t * (y2 - y1)
    return (intersect_x, intersect_y)

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

def vec_to_np(vec):
    """Convert vec to numpy array if necessary"""

    if vec is not None and not isinstance(vec, np.ndarray):
        vec = np.array(vec)

    return vec

def coords_unique(coords, pressure = None):
    """Select unique coordinates from coords and pressure vector"""

    _, unique_indices = np.unique(coords, axis=0, return_index=True)
    uniq = np.sort(unique_indices)

    coords = coords[uniq]

    if pressure is not None:
        pressure = pressure[uniq]

    return coords, pressure


def smooth_vector(coords, smoothing_factor, pressure = None):
    """
    Smooth a vector
    """

    coords, pressure = coords_unique(coords, pressure)

    # Create a parameterized spline representation of the curve
    ret = splprep(coords.T, s = smoothing_factor, k = 2)
    tck, u = ret[0], ret[1]

    delta_u = np.diff(u)

    # Calculate additional points
    new_u_fwd = np.array([u[:-1][:, None] + delta_u[:, None] * f
                          for f in FRAC_FWD]).reshape(-1)
    new_u_bck = np.array([u[1: ][:, None] + delta_u[:, None] * f
                          for f in FRAC_BCK]).reshape(-1)

    # Combine original and additional points
    u_new = np.sort(np.unique(np.concatenate((u, new_u_fwd,
                                              new_u_bck))))
    # Generate new points along the spline
    new_points = splev(u_new, tck)

    # Convert the smoothed coordinates back to a list of tuples

    print("coords:\n", np.column_stack((coords, pressure)))
    print("pressure:\n", pressure)
    # interpolate the pressure
    if pressure is not None:
        pressure_interp = interp1d(u, pressure, kind='linear')
        pressure = pressure_interp(u_new)

    print("new_points:\n", np.column_stack((pressure, *new_points)))

    smoothed_coords = list(zip(new_points[0], new_points[1]))
    return smoothed_coords, pressure

def smooth_coords(coords, pressure=None, smoothing_factor=0):
    """Smooth a path using scipy"""

    if len(coords) < 5:
        return coords, pressure

    if pressure:
        if len(pressure) != len(coords):
            log.warning("Pressure and coords lengths differ: %d %d",
                len(pressure), len(coords))
            log.warning("shortening pressure")
            pressure = pressure[:len(coords)]
            raise ValueError("incorrect length of coords and pressure")

    coords   = vec_to_np(coords)
    pressure = vec_to_np(pressure)

    smoothed_coords, pressure = smooth_vector(coords, smoothing_factor, pressure)

    return smoothed_coords, pressure

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

def calc_arc_coords(p1, p2, c, n = 20):
    """
    Calculate the coordinates of an arc between two points.
    The point p3 is on the opposite side of the arc from the line p1->p2.
    """
    x1, y1 = p1
    x2, y2 = p2
    xc, yc = c

    p0 = ((x1 + x2) / 2, (y1 + y2) / 2)
    x0, y0 = p0
    radius = math.sqrt((x2 - x1)**2 + (y2 - y1)**2) / 2

    side = determine_side_math(p1, p2, p0)

    # calculate the from p0 to p1
    a1 = np.arctan2(y1 - yc, x1 - xc)
    a2 = np.arctan2(y2 - yc, x2 - xc)

    if side == 'left' and a1 > a2:
        a2 += 2 * np.pi
    elif side == 'right' and a1 < a2:
        a1 += 2 * np.pi

    # calculate 20 points on the arc between a1 and a2
    angles = np.linspace(a1, a2, n)

    x_coords = x0 + radius * np.cos(angles)
    y_coords = y0 + radius * np.sin(angles)

    coords = np.column_stack((x_coords, y_coords))
    return coords.tolist()

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

def coords_rotate_np(coords, angle, origin):
    """Rotate a set of coordinates around a given origin."""
    
    coords = coords - origin
    
    # create  the rotation matrix
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    rotation_m = np.array([[cos_angle, -sin_angle],
                                [sin_angle, cos_angle]]).T
    
    coords = np.dot(coords, rotation_m.T)
    
    return coords + origin

def normal_vec(p0, p1):
    """Calculate the normal vector of a line segment."""

    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    length = np.sqrt(dx**2 + dy**2)
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

def transform_coords_np(coords, bb1, bb2):
    """Transform coordinates from one bounding box to another."""
    x0, y0, w0, h0 = bb1
    x1, y1, w1, h1 = bb2

    if w0 == 0 or h0 == 0:
        # issue warning
        warnings.warn("Bounding box has zero width or height")
        return coords
    
    # Ensure coords is a NumPy array
    if not isinstance(coords, np.ndarray):
        coords = np.array(coords)
    
    # Calculate the transformed coordinates
    transformed_coords = np.zeros_like(coords)
    transformed_coords[:, 0] = x1 + (coords[:, 0] - x0) / w0 * w1
    transformed_coords[:, 1] = y1 + (coords[:, 1] - y0) / h0 * h1

    return transformed_coords

def move_coords(coords, dx, dy):
    """Move a path by a given offset."""
    for i, (x, y) in enumerate(coords):
        coords[i] = (x + dx, y + dy)
    return coords

def path_bbox(coords, lw=0):
    """Calculate the bounding box of a path."""
    # now with numpy

    if len(coords) == 0:
        return (0, 0, 0, 0)

    coords = np.array(coords)
    left = np.min(coords[:, 0]) - lw / 2
    top = np.min(coords[:, 1]) - lw / 2
    width = np.max(coords[:, 0]) - left + lw / 2
    height = np.max(coords[:, 1]) - top + lw / 2

    return (left, top, width, height)

def path_bbox_old(coords, lw = 0):
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

def bbox_is_overlap(bbox0, bbox1):
    """Check whether two boxes overlap"""
    x0, y0, w0, h0 = bbox0
    x1, y1, w1, h1 = bbox1

   #log.debug("x0, y0, w0, h0: %d %d %d %d", x0, y0, w0, h0)
   #log.debug("x1, y1, w1, h1: %d %d %d %d", x1, y1, w1, h1)
   #log.debug("is_overlap: %s", (x0 < x1 + w1 and x0 + w0 > x1 and y0 < y1 + h1 and y0 + h0 > y1))

    return (x0 < x1 + w1 and x0 + w0 > x1 and y0 < y1 + h1 and y0 + h0 > y1)

def bbox_overlap(bbox1, bbox2):
    """Calculate the bbox of the overlap between two bboxes"""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    x = max(x1, x2)
    y = max(y1, y2)
    w = min(x1 + w1, x2 + w2) - x
    h = min(y1 + h1, y2 + h2) - y

    return (x, y, w, h)

def objects_bbox(objects, actual = True):
    """Calculate the bounding box of a list of objects."""
    if not objects:
        return (0, 0, 0, 0)

    bb = objects[0].bbox(actual = actual)

    if not bb:
        return (0, 0, 0, 0)

    left, top, width, height = objects[0].bbox(actual = actual)
    bottom, right = top + height, left + width

    for obj in objects[1:]:
        x, y, w, h = obj.bbox(actual = actual)
        left, top = min(left, x, x + w), min(top, y, y + h)
        bottom, right = max(bottom, y, y + h), max(right, x, x + w)

    width, height = right - left, bottom - top
    return (left, top, width, height)


def is_click_in_bbox(click_x, click_y, bbox):
    """Check if a click is inside a bounding box."""
    x, y, w, h = bbox
    log.debug("Checking click, bbox: %d,%d,%d,%d", int(x), int(y), int(x + w), int(y + h))
    log.debug("Click: %d,%d", int(click_x), int(click_y))
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
        bb = obj.bbox(actual = True)
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

def bus_listeners_on(bus, listeners):
    """group switch on for listeners"""

    for event, listener in listeners.items():
        if "priority" in listener:
            bus.on(event, listener["listener"], priority = listener["priority"])
        else:
            bus.on(event, listener["listener"])

def bus_listeners_off(bus, listeners):
    """group switch off for listeners"""

    for event, listener in listeners.items():
        bus.off(event, listener["listener"])
"""Utilities for calculation of brush strokes."""


NP_VEC = { 7: np.linspace(0, 1, 7) }
TIME_OLD = 0
TIME_NEW = 0

def get_current_color_and_alpha(ctx):
    """Get the current color and alpha from the Cairo context."""
    pattern = ctx.get_source()
    if isinstance(pattern, cairo.SolidPattern):
        r, g, b, a = pattern.get_rgba()
        return r, g, b, a
    # For non-solid patterns, return a default or indicate it's not solid
    return None, None, None, None  # Or raise an exception or another appropriate response

def bin_values(values, n_bins):
    """Bin values into n_bin bins between 0 and 1"""

    bin_size = 1 / n_bins
    ret = [ ]

    for value in values:
        ret.append(int(value / bin_size) * bin_size)

    return ret

def min_pr(val):
    """Scale the pressure such that it does not go below .2"""
    return .2 + .8 * val

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

def smooth_pressure(pressure, window_size = 3):
    """Smooth the pressure values."""
    n = len(pressure)
    smoothed = [ ]
    for i in range(n):
        start = max(0, i - window_size)
        end = min(n, i + window_size)
        smoothed.append(sum(pressure[start:end]) / (end - start))
    return smoothed

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

def calc_segments_2(p0, p1, width):
    """Calculate the segments of an outline."""
    nx, ny = normal_vec_scaled(p0, p1, width)

    l_seg_s = (p0[0] + nx, p0[1] + ny)
    r_seg_s = (p0[0] - nx, p0[1] - ny)
    l_seg_e = (p1[0] + nx, p1[1] + ny)
    r_seg_e = (p1[0] - nx, p1[1] - ny)

    return (l_seg_s, l_seg_e), (r_seg_s, r_seg_e)

def calc_segments_3(p0, p1, w1, w2):
    """Calculate the segments of an outline."""
    nx1, ny1 = normal_vec_scaled(p0, p1, w1)
    nx2, ny2 = normal_vec_scaled(p0, p1, w2)

    l_seg_s = (p0[0] + nx1, p0[1] + ny1)
    r_seg_s = (p0[0] - nx1, p0[1] - ny1)
    l_seg_e = (p1[0] + nx2, p1[1] + ny2)
    r_seg_e = (p1[0] - nx2, p1[1] - ny2)

    return (l_seg_s, l_seg_e), (r_seg_s, r_seg_e)

def calc_outline_short_generic(coords, pressure, line_width, rounded = False):
    """Calculate the normal outline for a 2-coordinate path"""

    outline_l = []
    outline_r = []
    line_width = line_width or 1

    p0, p1 = coords[0], coords[1]
    width  = line_width * pressure[0] / 2


    l_seg1, r_seg1 = calc_segments_2(p0, p1, width)

    if rounded:
        arc_coords = calc_arc_coords2(l_seg1[0], r_seg1[0], p1, 10)
        outline_r.extend(arc_coords)

    outline_l.append(l_seg1[0])
    outline_l.append(l_seg1[1])
    outline_r.append(r_seg1[0])
    outline_r.append(r_seg1[1])

    if rounded:
        arc_coords = calc_arc_coords2(l_seg1[1], r_seg1[1], p1, 10)
        outline_l.extend(arc_coords)

    return outline_l, outline_r

def calc_normal_outline_short(coords, widths, rounded = False):
    """Calculate the normal outline for a 2-coordinate path"""

    outline_l = []
    outline_r = []

    p0, p1 = coords[0], coords[1]
    width  = widths[0] / 2

    l_seg1, r_seg1 = calc_segments_2(p0, p1, width)

    if rounded:
        arc_coords = calc_arc_coords2(l_seg1[0], r_seg1[0], p1, 10)
        outline_r.extend(arc_coords)

    outline_l.append(l_seg1[0])
    outline_l.append(l_seg1[1])
    outline_r.append(r_seg1[0])
    outline_r.append(r_seg1[1])

    if rounded:
        arc_coords = calc_arc_coords2(l_seg1[1], r_seg1[1], p1, 10)
        outline_l.extend(arc_coords)

    return outline_l, outline_r


def calc_normal_segments(coords):
    """Calculate the normal segments of a path using numpy."""

    # Calculate differences between consecutive points
    dx = np.diff(coords[:, 0])
    dy = np.diff(coords[:, 1])

    # Calculate lengths of the segments
    lengths = np.sqrt(dx**2 + dy**2)

    # Normalize the vectors
    dx_normalized = dx / lengths
    dy_normalized = dy / lengths

    # Calculate normal vectors
    nx = -dy_normalized
    ny = dx_normalized

    ret = np.column_stack((nx, ny))
    return ret

def calc_normal_segments_scaled(normal_segments, widths):
    """multiply the normal segments by weights"""

    widths = np.array(widths) / 2
    nw0 = normal_segments * widths[:-1, np.newaxis]
    nw1 = normal_segments * widths[1:, np.newaxis]

    return nw0, nw1

def determine_side_math(p1, p2, p3):
    """Determine the side of a point p3 relative to a line segment p1->p2."""
    det = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
    return 'left' if det > 0 else 'right'

def calc_arc_coords2(p1, p2, c, n = 20):
    """Calculate the coordinates of an arc between points p1 and p2.
       The arc is a fragment of a circle with centre in c."""
    # pylint: disable=too-many-locals
    x1, y1 = p1
    x2, y2 = p2
    xc, yc = c

    if n not in NP_VEC:
        NP_VEC[n] = np.linspace(0, 1, n)

    # calculate the radius of the circle
    radius = np.sqrt((x2 - xc)**2 + (y2 - yc)**2)
    side = determine_side_math(p1, p2, c)

    # calculate the angle between the line p1->c and p1->p2
    a1 = np.arctan2(y1 - c[1], x1 - c[0])
    a2 = np.arctan2(y2 - c[1], x2 - c[0])

    if side == 'left' and a1 > a2:
        a2 += 2 * np.pi
    elif side == 'right' and a1 < a2:
        a1 += 2 * np.pi

    #angles = np.linspace(a1, a2, n)
    angles = a1 + (a2 - a1) * NP_VEC[n]

    # Calculate the arc points
    x_coords = xc + radius * np.cos(angles)
    y_coords = yc + radius * np.sin(angles)

    # Combine x and y coordinates
    #coords = np.column_stack((x_coords, y_coords))
    coords = list(zip(x_coords, y_coords))
    return coords

def calc_segments(coords, nw0, nw1):
    """calculate starting and ending points of the segments"""

    l_seg_s = coords[:-1] + nw0
    r_seg_s = coords[:-1] - nw0
    l_seg_e = coords[1:] + nw1
    r_seg_e = coords[1:] - nw1

    return [l_seg_s, l_seg_e], [r_seg_s, r_seg_e]

def calc_intersections(lseg_s, lseg_e):
    """
    Calculate intersection points between consecutive line segments.
    Each segment consists of two points, start and end.
    """
    # pylint: disable=too-many-locals

    x1, y1 = lseg_s[:-1].T  # Start points of segments
    x2, y2 = lseg_e[:-1].T  # End points of segments
    x3, y3 = lseg_s[1:].T   # Start points of the next segments
    x4, y4 = lseg_e[1:].T   # End points of the next segments

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    # Avoid division by zero
    denom_nonzero = denom != 0

    # Calculate t and u only where denom is non-zero
    t = np.zeros_like(denom)
    u = np.zeros_like(denom)

    t[denom_nonzero] = ((x1[denom_nonzero] - x3[denom_nonzero]) *
                        (y3[denom_nonzero] - y4[denom_nonzero]) -
                        (y1[denom_nonzero] - y3[denom_nonzero]) *
                        (x3[denom_nonzero] - x4[denom_nonzero])) / denom[denom_nonzero]

    u[denom_nonzero] = ((x1[denom_nonzero] - x3[denom_nonzero]) *
                        (y1[denom_nonzero] - y2[denom_nonzero]) -
                        (y1[denom_nonzero] - y3[denom_nonzero]) *
                        (x1[denom_nonzero] - x2[denom_nonzero])) / denom[denom_nonzero]
    # Conditions for intersection
    intersect_cond = (0 <= t) & (t <= 1) & (0 <= u) & (u <= 1)

    # Calculate intersection points
    intersect_x = x1 + t * (x2 - x1)
    intersect_y = y1 + t * (y2 - y1)

    # Mask out non-intersections
    intersect_x[~intersect_cond] = np.nan
    intersect_y[~intersect_cond] = np.nan

    # Stack results
    intersections = np.stack((intersect_x, intersect_y), axis=-1)

    return intersections

def construct_outline(segments, intersections):
    """Construct the left or right outline from segments and intersections."""
    starts, ends = segments
    is_intersect = ~np.isnan(intersections[:, 0]) & ~np.isnan(intersections[:, 1])

    n = len(intersections)
    is_s = np.sum(~is_intersect) # is segment

    # joints count double
    l_out = n + is_s

    result = np.zeros((l_out + 2, 2), dtype=segments[0].dtype)

    seg_int = np.where(~is_intersect)[0]
    selector_seg = seg_int + np.arange(is_s)

    result[selector_seg + 1] = ends[:-1][~is_intersect]
    result[selector_seg + 2] = starts[1:][~is_intersect]

    selector = np.zeros(l_out + 2, dtype=bool)
    selector[:] = True
    selector[[0, -1]] = False

    selector[selector_seg + 1] = False
    selector[selector_seg + 2] = False
    result[selector] = intersections[is_intersect]
    result[0] = starts[0]
    result[-1] = ends[-1]
    return result

def calc_arc_coords_np(p1, p2, c, n = 20):
    """Calculate the coordinates of an arc between points p1 and p2.
       The arc is a fragment of a circle with centre in c."""
    # pylint: disable=too-many-locals
    x1, y1 = p1
    x2, y2 = p2
    xc, yc = c

    if n not in NP_VEC:
        NP_VEC[n] = np.linspace(0, 1, n)

    # calculate the radius of the circle
    radius = np.sqrt((x2 - xc)**2 + (y2 - yc)**2)
    side = determine_side_math(p1, p2, c)

    # calculate the angle between the line p1->c and p1->p2
    a1 = np.arctan2(y1 - c[1], x1 - c[0])
    a2 = np.arctan2(y2 - c[1], x2 - c[0])

    if side == 'left' and a1 > a2:
        a2 += 2 * np.pi
    elif side == 'right' and a1 < a2:
        a1 += 2 * np.pi

    angles = a1 + (a2 - a1) * NP_VEC[n]

    # Calculate the arc points
    x_coords = xc + radius * np.cos(angles)
    y_coords = yc + radius * np.sin(angles)

    return x_coords, y_coords

def construct_outline_round(segments, intersections, coords, n = 3,
                            ret_segments = False):
    """Construct the left or right outline from segments and intersections."""
    # pylint: disable=too-many-locals

    starts, ends = segments
    n_segments = len(starts)
    #print("starts", starts)
    #print("ends", ends)
    is_intersect = ~np.isnan(intersections[:, 0]) & ~np.isnan(intersections[:, 1])

    #print(is_intersect)
    n_int = len(intersections)

    # number of rounded joints
    is_rj = np.sum(~is_intersect)

    # round joints count n + 1 times
    l_out = n_int + is_rj * (n - 1)
    #print("n_segments = ", n_segments, "n =", n, "n_int = ", n_int, "is_rj = ", is_rj, "l_out = ", l_out)

    result = np.zeros((l_out + 2, 2), dtype=segments[0].dtype)
    
    # first and last point
    result[0] = starts[0]
    result[-1] = ends[-1]

    seg_int = np.where(~is_intersect)[0]
    #print("seg_int", seg_int)
    #print("intersections at:", np.where(is_intersect)[0])

    # selector_seg selects the segment joints that need to be an arc
    # it contains the positions in the resulting array of the segment joints
    selector_seg = seg_int + np.arange(is_rj) * (n - 1)
    #print("selector_seg", selector_seg)

    # only these which are not intersections
    starts = starts[1:][~is_intersect]
    ends = ends[:-1][~is_intersect]

    point_selector = np.arange(len(coords))[1:-1]
    #print("point_selector", point_selector)

    # fill in the arcs
    # i is the index of the segment joint
    # ind is the position of the joint in the result array
    # idx is the index of the point in the coords array
    for i, ind in enumerate(selector_seg):
        idx = point_selector[~is_intersect][i]
        ac_x, ac_y = calc_arc_coords_np(ends[i], starts[i], coords[idx], n)
        result[ind + 1:ind + n + 1, 0] = ac_x
        result[ind + 1:ind + n + 1, 1] = ac_y
        #print("i =", i, "idx=", idx, "ind=", ind, "p=", coords[idx], "arc_coords =", arc_coords)
    index_column = np.arange(result.shape[0]).reshape(-1, 1)
    #print("after adding arcs\n", np.hstack((index_column, result)).astype(int))

    # selector indicates where in the array are the intersections
    selector = np.zeros(l_out + 2, dtype=bool)
    selector[:] = True
    selector[[0, -1]] = False

    for i in range(n):
        selector[selector_seg + i + 1] = False

    #result[selector_seg + 1] = ends
    #print("after adding ends\n", result[:10])
    #result[selector_seg + n] = starts
    #print("after adding starts\n", result[:10])
    result[selector] = intersections[is_intersect]
    #print("after adding intersections\n", result[:10])
    if not ret_segments:
        return result, None

    seg_positions = np.zeros((n_segments, 2), dtype=int)
    seg_positions[np.where(is_intersect)[0], 0] = np.where(selector)[0] - 1

    # number of points corresponding to the segment including start and end
    seg_positions[np.where(is_intersect)[0], 1] = 2

    seg_positions[np.where(~is_intersect)[0], 0] = selector_seg
    # number of points corresponding to the segment including start and arc
    seg_positions[np.where(~is_intersect)[0], 1] = n + 1

    seg_positions[-1, 0] = l_out
    seg_positions[-1, 1] = 2
    #print("seg_positions\n", seg_positions)

    return result, seg_positions

def round_tip(rseg, lseg, coords, n = 3):
    """Round tip of the brush."""

    p_s = coords[0]
    r_s = rseg[0][0]
    l_s = lseg[0][0]
    ac_x_s, ac_y_s = calc_arc_coords_np(r_s, l_s, p_s, n)
    arc_coords_s = np.column_stack((ac_x_s, ac_y_s))

    p_e = coords[-1]
    r_e = rseg[1][-1]
    l_e = lseg[1][-1]
    ac_x_e, ac_y_e = calc_arc_coords_np(l_e, r_e, p_e, n)
    arc_coords_e = np.column_stack((ac_x_e, ac_y_e))

    return arc_coords_s, arc_coords_e

def calc_normal_outline(coords, widths, rounded = False):
    """Calculate the normal outline of a path."""
    n = len(coords)

    if n < 2:
        return [], []

    if n == 2:
        return calc_normal_outline_short(coords, widths, rounded)

    t1 = GLib.get_monotonic_time()
    coords_np = np.array(coords)

    # normal vectors
    n_segm = calc_normal_segments(coords_np)

    # normal vectors scaled by the widths
    nw0, nw1 = calc_normal_segments_scaled(n_segm, widths)

    # calculate the outline segments
    lseg, rseg = calc_segments(coords_np, nw0, nw1)

    # figure whether and if yes, where the segments intersect
    l_intersect = calc_intersections(*lseg)
    r_intersect = calc_intersections(*rseg)

    if rounded:
        outline_l, _ = construct_outline_round(lseg, l_intersect,
                                            coords, n = 3)
        outline_r, _ = construct_outline_round(rseg, r_intersect,
                                            coords, n = 3)
        tip_s, tip_e = round_tip(rseg, lseg, coords, n = 10)
        outline_l = np.concatenate((tip_s, outline_l, tip_e))
    else:
        outline_l = construct_outline(lseg, l_intersect)
        outline_r = construct_outline(rseg, r_intersect)

    t2 = GLib.get_monotonic_time()

    log.debug("outline lengths: %d, %d", len(outline_l), len(outline_r))
    log.debug("coords length: %d", len(coords))
    global TIME_NEW
    TIME_NEW += t2 - t1
    log.debug("time, new: %s", TIME_NEW)
    return outline_l, outline_r

def point_mean(p1, p2):
    """Calculate the mean of two points."""
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

def __consecutive_runs(values):
    """Find the lengths of consecutive identical numbers."""

    # Convert the list to a NumPy array
    array = np.array(values)

    # Find the indices where the value changes
    change_indices = np.where(np.diff(array) != 0)[0] + 1

    # Add the start and end indices
    change_indices = np.concatenate(([0], change_indices, [len(array)]))

    # Calculate the lengths of consecutive identical numbers
    lengths = np.diff(change_indices)

    # Get the unique values corresponding to each segment
    unique_values = array[change_indices[:-1]]

    return lengths, unique_values

def __consecutive_runs_2(values):
    """
    Find the lengths of consecutive identical pairs of numbers.
    """

    a0 = values[:,0]
    a1 = values[:,1]

    a0_diff = np.diff(a0) != 0
    a1_diff = np.diff(a1) != 0

    #print("values:", values)
    #print("a0_diff:", a0_diff)
    #print("a1_diff:", a1_diff)

    # Find the indices where the value changes
    change_indices = np.where(a0_diff | a1_diff)[0] + 1

    # Add the start and end indices
    change_indices = np.concatenate(([0], change_indices, [len(values)]))

    # Calculate the lengths of consecutive identical numbers
    lengths = np.diff(change_indices)

    return np.column_stack((change_indices[:-1], lengths))

def __pressure_to_bins(pressure, n_bins = 5):
    """Convert pressure values to bins."""

    #print("pressure:", pressure)
    pressure = np.array(pressure)
    pressure = (pressure[:-1] + pressure[1:]) / 2
    #print("pressure:", pressure)

    bin_edges = np.linspace(0, 1, n_bins + 1, endpoint=False)[1:]
    #print("bin_edges:", bin_edges)
    bin_edges = np.append(bin_edges, 1.0)
    #print("bin_edges:", bin_edges)

    # Find the bin index for each number
    pressure = np.digitize(pressure, bins=bin_edges, right=True)
    #print("pressure:", pressure)

    return pressure

def __make_segments(coords, ledge, redge, l_intersect, r_intersect,
                    join_lengths):
    """
    Make segments from the left and right outlines.

    Returns two arrays. First array contains the coordinates of all
    segments in a continuous manner (so that scaling or moving would become
    easy). This is the same as the outline, but the coordinates are
    rearranged to form distinct segments.

    The second array contains information about the segments: starting
    coordinate, length, coordinate at which the segment ends are, and the
    transparency value.
    """

    l_is_intersect = ~np.isnan(l_intersect[:, 0]) & ~np.isnan(l_intersect[:, 1])
    r_is_intersect = ~np.isnan(r_intersect[:, 0]) & ~np.isnan(r_intersect[:, 1])
    b_is_intersect = l_is_intersect & r_is_intersect

    ## we will return the outline as a single numpy object
    # this will have n - 1 segments
    # each segment will have 4 coordinates with no intersections,
    # 2 + arc coordinates with 1 intersection and 
    # 2 * arc coordinates with 2 intersections

    #join_lengths, join_vals = __consecutive_runs(pressure) 

    # create np array with n_coord rows and three columns

    outline_l, l_seg_pos = construct_outline_round(ledge, l_intersect,
                                        coords, n = 3, ret_segments = True)
    outline_r, r_seg_pos = construct_outline_round(redge, r_intersect,
                                        coords, n = 3, ret_segments = True)

    #print("outline_l: \n", outline_l[:], "\nl_seg_pos: \n", l_seg_pos)
    #print("outline_r: \n", outline_r[:], "\nr_seg_pos: \n", r_seg_pos)

    #print("join_lengths: ", join_lengths, "sum:", np.sum(join_lengths))
    #print("l_seg_pos length: ", len(l_seg_pos), "r_seg_pos length: ", len(r_seg_pos))

    tot_seg_length = len(outline_l) + len(outline_r) + len(join_lengths) * 2 - 4
    #print("outline_l length: ", len(outline_l), "outline_r length: ", len(outline_r))
    #print("tot_seg_length = ", tot_seg_length)

    # first column: the coordinate at which the segment starts in the
    # resulting array
    # second column: number of coordinates in the segment
    # third column: the coordinate at which the actual segment ends are
    # found (note: the outline continues beyond this point, since there is
    # the right outline after the left. the second column shows where the
    # left outline ends and the right outline starts)
    # fourth column: the binned pressure value for this segment
    # XXX: each segment should have a starting and ending pressure value
    # or actually how are the pressure values used?
    segments = np.zeros((tot_seg_length, 2))

    seg_info = np.zeros((len(join_lengths), 5), dtype = int)

    left  = None
    right = None

    pos = 0
    seg = 0
    n_r = 0
    n_l = 0

    for jseg, j_len in enumerate(join_lengths):
        # j_len gives the number of segments to join
        # jseg gives the index of the joint segment

        n_coords_l = np.sum(l_seg_pos[seg:seg + j_len, 1]) - j_len + 1
        n_coords_r = np.sum(r_seg_pos[seg:seg + j_len, 1]) - j_len + 1

        n_coords = n_coords_l + n_coords_r
        n_l += n_coords_l
        n_r += n_coords_r

        seg_info[jseg, 0] = pos
        seg_info[jseg, 1] = n_coords
        seg_info[jseg, 3] = seg
        seg_info[jseg, 4] = seg + j_len - 1

       #print("building joined segment jseg = ", jseg, 
       #      "join_lengths[jseg] = ", j_len,
       #      "n_coords_l = ", n_coords_l, "n_coords_r = ", n_coords_r,
       #      "n_coords = ", n_coords, "n_l = ", n_l, "n_r = ", n_r)

        # first the left segment: from start to end
        pos0 = pos
        for j in range(j_len):
            if j == 0:
                coord_start = l_seg_pos[seg + j, 0]
                coord_n     = l_seg_pos[seg + j, 1]
            else:
                coord_start = l_seg_pos[seg + j, 0] + 1
                coord_n     = l_seg_pos[seg + j, 1] - 1

            segments[pos0:pos0 + coord_n, 0:2] = outline_l[coord_start:coord_start + coord_n, 0:2]
            #print("j = ", j, "coord_start = ", coord_start, "coord_n = ", coord_n, "seg + j = ", seg + j, "pos0 = ", pos0)
            #print("outline_l:\n", outline_l[coord_start:coord_start + coord_n, 0:2])
            #segments[pos0:pos0 + coord_n,2] = jseg
            #segments[pos0:pos0 + coord_n,3] = join_vals[jseg]

            #seg += 1
            pos0 += coord_n

        seg_info[jseg, 2] = pos0 - 1
        #print("value at pos0: ", segments[pos0 - 1])
        #print("segments:\n", segments[:20].astype(int))
        # then the right segment: from end to start
        pos0 = 0
        for j in range(j_len):
            if j == 0:
                coord_start = r_seg_pos[seg + j, 0]
                coord_n     = r_seg_pos[seg + j, 1]
            else:
                coord_start = r_seg_pos[seg + j, 0] + 1
                coord_n     = r_seg_pos[seg + j, 1] - 1

            s_to   = pos + n_coords - pos0
            s_from = s_to - coord_n
            #print("j = ", j, "coord_start = ", coord_start, "coord_n = ", coord_n, "seg + j = ", seg + j, "pos = ", pos)
            #print("outline_r:\n", outline_r[coord_start:coord_start + coord_n, 0:2])
            #print("outline_r:\n", outline_r[coord_start:coord_start + coord_n, 0:2][::-1])
            segments[s_from:s_to, 0:2] = outline_r[coord_start:coord_start + coord_n, 0:2][::-1]
            #segments[s_from:s_to,2] = jseg
            #segments[s_from:s_to,3] = join_vals[jseg]

            pos0 += coord_n

        pos += n_coords
        seg += j_len

    #print("segment coords:\n", segments.astype(int))
    #print("number of joint segments: ", len(join_lengths))
    #print("n_l = ", n_l, "n_r = ", n_r, "pos = ", pos)

    #print("outline_l length: ", len(outline_l), "outline_r length: ", len(outline_r), "sum: ", len(outline_l) + len(outline_r))
    #print("how about: ", len(outline_l) + len(outline_r) + (len(join_lengths) - 2)* 2)
    #print("len segments: ", len(segments))
    #print("seg_info:\n", seg_info.astype(int))
    return segments, seg_info, None

def __calc_midpoints(seg_info, lseg, rseg, pressure):
    """
    Calculate the segment midpoints used to generate the pencil gradient
    """

    midpoints = np.zeros((len(seg_info), 6), dtype = lseg[0].dtype)
    midpoints[:,4:6] = pressure
    sel = seg_info[:,3]
    midpoints[:,0:2] = (lseg[0][sel] + rseg[0][sel])/2
    midpoints[:,2:4] = (lseg[1][sel] + rseg[1][sel])/2
    print("midpoints:\n", midpoints)
    
    return midpoints

def calc_pencil_segments(coords, widths, pressure):
    """
    Calculate the normal outline of a path.

    This one is used in the pencil brush v3.

    The main difference from normal outlines is that we are creating two
    outlines and a pressure vector, all with exactly the same number of
    points. This allows to create segments with different transparency
    values.
    """


    n = len(coords)

    if n < 2:
        return [], []

    # normal vectors
    #print("len coords: ", len(coords), "len pressure: ", len(pressure))
    coords_np = np.array(coords)
    n_segm = calc_normal_segments(coords_np)

    # normal vectors scaled by the widths
    nw0, nw1 = calc_normal_segments_scaled(n_segm, widths)

    # calculate the outline segments
    lseg, rseg = calc_segments(coords_np, nw0, nw1)

    # figure whether and if yes, where the segments intersect
    l_intersect = calc_intersections(*lseg)
    r_intersect = calc_intersections(*rseg)

    pressure = __pressure_to_bins(pressure, n_bins = 15)
    pressure = pressure / 5 * 0.8 + 0.2
    print("pressure:\n", pressure)
    #print("len pressure now: ", len(pressure))

    pp = np.column_stack((pressure[:-1], pressure[1:]))
    #print("pp: \n", pp)
    runs = __consecutive_runs_2(pp)
    pp = pp[runs[:,0]]
    #print("runs:\n", runs)

    segments, seg_info, pressure = __make_segments(coords, lseg, rseg, 
                               l_intersect, r_intersect,
                                                   runs[:,1])
    #print("len coords:", len(coords), "len lseg:", len(lseg[0]))

    midpoints = __calc_midpoints(seg_info, lseg, rseg, pp)

    return segments, seg_info, midpoints


def calc_pencil_outline(coords, pressure, line_width):
    """
    Calculate the normal outline of a path.

    This one is used in the pencil brush v3.

    The main difference from normal outlines is that we are creating two
    outlines and a pressure vector, all with exactly the same number of
    points. This allows to create segments with different transparency
    values.
    """

    n = len(coords)

    if n < 2:
        return [], [], []

    if n == 2:
        outline_l, outline_r = calc_outline_short_generic(coords, pressure, line_width, True)
        return outline_l, outline_r, pressure

    pressure_ret = [ ]
    line_width = line_width or 1
    outline_l = []
    outline_r = []

    p0, p1 = coords[0], coords[1]
    width  = line_width / 5 #* pressure[0] / 2
    l_seg1, r_seg1 = calc_segments_2(p0, p1, width)

    ## append the points for the first coord
    arc_coords = calc_arc_coords2(l_seg1[0], r_seg1[0], p1, 10)
    outline_r.extend(arc_coords[:5][::-1])
    outline_l.extend(arc_coords[5:])
    pressure_ret = pressure_ret + [pressure[0]] * 5

    outline_l.append(l_seg1[0])
    outline_r.append(r_seg1[0])
    pressure_ret.append(pressure[0])

    npt = 7

    for i in range(n - 2):
        p2 = coords[i + 2]
        width  = line_width / 3#* pressure[i] / 2

        l_seg2, r_seg2 = calc_segments_2(p1, p2, width)

        # in the following, if the two outline segments intersect, we simplify
        # them; if they don't, we add a curve
        intersect_l = calc_intersect(l_seg1, l_seg2)

        if intersect_l is None:
            arc_coords = calc_arc_coords2(l_seg1[1], l_seg2[0], p1, npt)
            outline_l.extend(arc_coords)
        else:
            outline_l.extend([intersect_l] * npt)

        # in the following, if the two outline segments intersect, we simplify
        # them; if they don't, we add a curve
        intersect_r = calc_intersect(r_seg1, r_seg2)

        if intersect_r is None:
            arc_coords = calc_arc_coords2(r_seg1[1], r_seg2[0], p1, npt)
            outline_r.extend(arc_coords)
        else:
            outline_r.extend([intersect_r] * npt)

        pressure_ret.extend([ pressure[i] ] * npt)

        l_seg1, r_seg1 = l_seg2, r_seg2
        p0, p1 = p1, p2

    outline_l.append(l_seg1[1])
    outline_r.append(r_seg1[1])
    pressure_ret.append(pressure[-1])

    arc_coords = calc_arc_coords2(l_seg1[1], r_seg1[1], p0, 10)
    outline_r.extend(arc_coords[:5])
    outline_l.extend(arc_coords[5:][::-1])
    pressure_ret = pressure_ret + [pressure[-1]] * 5

    log.debug("outline lengths: %d, %d pres=%d", len(outline_l), len(outline_r), len(pressure))
    return outline_l, outline_r, pressure_ret

def calc_normal_outline_tapered(coords, pressure, line_width, taper_pos, taper_length):
    """Calculate the normal outline of a path for tapered brush."""
    n = len(coords)

    if n < 2:
        return [], []

    if n == 2:
        return calc_outline_short_generic(coords, pressure, line_width, False)

    line_width = line_width or 1

    outline_l = []
    outline_r = []

    taper_l_cur = 0
    p0, p1 = coords[0], coords[1]

    for i in range(n - 2):
        p2 = coords[i + 2]

        if i >= taper_pos:
            taper_l_cur += distance(p0, p1)
            taper_l_cur = min(taper_l_cur, taper_length)

            w0  = line_width * pressure[i] / 2 * (1 - taper_l_cur / taper_length)
            w1  = line_width * pressure[i + 1] / 2 * (1 - taper_l_cur / taper_length)
            w2  = line_width * pressure[i + 2] / 2 * (1 - taper_l_cur / taper_length)
        else:
            w0  = line_width * pressure[i] / 2
            w1  = line_width * pressure[i + 1] / 2
            w2  = line_width * pressure[i + 2] / 2

        l_seg1, r_seg1 = calc_segments_3(p0, p1, w0, w1)
        l_seg2, r_seg2 = calc_segments_3(p1, p2, w1, w2)

        if i == 0:
        ## append the points for the first coord
            arc_coords = calc_arc_coords2(l_seg1[0], r_seg1[0], p1, 10)
            outline_r.extend(arc_coords)
            outline_l.append(l_seg1[0])
            outline_r.append(r_seg1[0])

        outline_l.append(l_seg1[1])
        outline_l.append(l_seg2[0])
        outline_r.append(r_seg1[1])
        outline_r.append(r_seg2[0])

        if i == n - 3:
            outline_l.append(point_mean(l_seg2[1], r_seg2[1]))
        p0, p1 = p1, p2

    return outline_l, outline_r
"""
This module contains the commands that can be executed on the objects. They
should be undoable and redoable. It is their responsibility to update the
state of the objects they are acting on.
"""

## ---------------------------------------------------------------------
## These are the commands that can be executed on the objects. They should
## be undoable and redoable. It is their responsibility to update the
## state of the objects they are acting on.

## The hash is used to identify the command. By default, it is unique for
## for a type of command and the objects it is acting on.
## That way, a command affecting a certain group of primitives can be
## joined with another command that does the same thing on the same group.
## ---------------------------------------------------------------------

def compute_id_hash(objects):
    """calculate a unique hash based on the drawables carreid by the object"""
    # Extract IDs and concatenate them into a single string
    if isinstance(objects, list):
        ids_concatenated = ''.join(str(id(obj)) for obj in objects)
    else:
        ids_concatenated = str(id(objects))

    # Compute the hash of the concatenated string
    hash_object = hashlib.md5(ids_concatenated.encode())
    hash_hex    = hash_object.hexdigest()

    return hash_hex

class Command:
    """Base class for commands."""
    def __init__(self, mytype, objects):
        self.obj   = objects
        self.__type   = mytype
        self.__undone = False

        if objects:
            self.__hash = compute_id_hash(objects)
        else:
            self.__hash = compute_id_hash(self)

        self.__hash = mytype + ':' + self.__hash

    def __eq__(self, other):
        """Return whether the command is equal to another command."""
        return self.hash() == other.hash()

    def __gt__(self, other):
        """Return whether the command is a group that contains commands with identical hashes."""
        return self.hash() ==  'group:' + other.hash()

    def __add__(self, other):
        """Add two commands together."""

        if other.com_type() == "group":
            other.add(self)
            return other

        return CommandGroup([ self, other ])

    def com_type(self):
        """Return my type"""
        return self.__type

    def hash(self):
        """Return a hash of the command."""
        return self.__hash

    def type(self):
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
    def __init__(self, commands):
        super().__init__("group", objects=None)
        self.__commands = commands

        self.__hash = compute_id_hash([ self ])
        self.__hash = "group" + ':' + self.__hash

    def __add__(self, other):
        """Add two commands together."""
        if other.type() == "group":
            return CommandGroup(self.__commands + other.commands())
        return CommandGroup(self.__commands + [ other ])

    def hash(self):
        """Return a hash of the command."""
        cmds = self.__commands
        hashes = [ cmd.hash() for cmd in cmds ]

        ## how many unique values in the hashes array?
        unique_hashes = set(hashes)
        if len(unique_hashes) == 1:
            return 'group:' + hashes[0]

        return self.__hash

    def commands(self, cmd = None):
        """Return or set the commands in the group."""
        if cmd:
            self.__commands = cmd
        return self.__commands

    def add(self, cmd):
        """Add a command to the group."""
        self.__commands.append(cmd)
        return self

    def undo(self):
        """Undo the command."""
        if self.undone():
            return
        for cmd in self.__commands[::-1]:
            cmd.undo()
        self.undone_set(True)
        return

    def redo(self):
        """Redo the command."""
        if not self.undone():
            return
        for cmd in self.__commands:
            cmd.redo()
        self.undone_set(False)
        return

class InsertPageCommand(Command):
    """
    Handling inserting pages.
    """
    def __init__(self, page):
        super().__init__("insert_page", None)
        self.__prev = page
        self.__next = page.next(create = False)

        # create the new page
        page.next_set(None)
        self.__page = page.next(create = True)
        self.__page.prev_set(page)
        self.__page.next_set(self.__next)
        page.next_set(self.__page)

        if self.__next:
            self.__next.prev_set(self.__page)

    def undo(self):
        """Undo the command."""
        if self.undone():
            return None
        self.__prev.next_set(self.__next)
        if self.__next:
            self.__next.prev_set(self.__prev)
        self.undone_set(True)
        return self.__prev

    def redo(self):
        """Redo the command."""
        if not self.undone():
            return None
        self.__prev.next_set(self.__page)
        if self.__next:
            self.__next.prev_set(self.__page)
        self.undone_set(False)
        return self.__page

class DeletePageCommand(Command):
    """
    Handling deleting pages.
    """
    def __init__(self, page):
        super().__init__("delete_page", None)
        prev_page = page.prev()

        if prev_page == page:
            prev_page = None

        next_page = page.next(create = False)

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
    def __init__(self, page, layer_pos = None):
        """Simple class for handling deleting layers of a page."""
        super().__init__("delete_layer", None)

        if layer_pos is None:
            layer_pos = page.layer_no()
        self.__layer, self.__layer_pos = page.delete_layer(layer_pos)
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

class ClipCommand(Command):
    """Simple class for handling clipping objects."""
    def __init__(self, clip, objects, stack, selection_object = None):
        super().__init__("clip", objects)
        self.__selection = selection_object
        self.__stack = stack
        self.__stack_copy = stack[:]

        # position of the last object in stack
        idx = self.__stack.index(self.obj[-1])

        self.__group = ClippingGroup(clip, self.obj)
        # add group to the stack at the position of the last object
        self.__stack.insert(idx, self.__group)

        for obj in self.obj:
            if not obj in stack:
                raise ValueError("Object not in stack:", obj)
            stack.remove(obj)
        stack.remove(clip)

    def undo(self):
        """Undo the command."""
        if self.undone():
            return
        swap_stacks(self.__stack, self.__stack_copy)
        self.undone_set(True)
        if self.__selection:
            self.__selection.set(self.obj)

    def redo(self):
        """Redo the command."""
        if not self.undone():
            return
        swap_stacks(self.__stack, self.__stack_copy)
        self.undone_set(False)
        if self.__selection:
            self.__selection.set([ self.__group ])

class UnClipCommand(Command):
    """Simple class for handling clipping objects."""
    def __init__(self, objects, stack, selection_object = None):
        super().__init__("unclip", objects)
        self.__stack = stack
        self.__stack_copy = stack[:]
        self.__selection = selection_object

        new_objects = []
        n = 0

        for obj in self.obj:
            if not obj.type == "clipping_group":
                log.warning("Object is not a clipping_group, ignoring: %s", obj)
                log.warning("object type: %s", obj.type)
                continue

            n += 1
            # position of the group in the stack
            idx = self.__stack.index(obj)

            # remove the group from the stack
            self.__stack.remove(obj)

            # add the objects back to the stack
            for subobj in obj.objects[::-1]:
                self.__stack.insert(idx, subobj)
                new_objects.append(subobj)

        if n > 0 and self.__selection:
            self.__selection.set(new_objects)

        self.__group = new_objects

    def undo(self):
        """Undo the command."""
        if self.undone():
            return
        swap_stacks(self.__stack, self.__stack_copy)
        self.undone_set(True)
        if self.__selection:
            self.__selection.set(self.obj)

    def redo(self):
        """Redo the command."""
        if not self.undone():
            return
        swap_stacks(self.__stack, self.__stack_copy)
        self.undone_set(False)
        if self.__selection:
            self.__selection.set(self.__group)

class TextEditCommand(Command):
    """Simple class for handling text editing."""

    def __init__(self, obj, oldtext, newtext):
        super().__init__("text_edit", obj)
        self.__oldtext = oldtext
        self.__newtext = newtext

    def undo(self):
        """Undo the command."""
        if self.undone():
            return
        self.obj.set_text(self.__oldtext)
        self.undone_set(True)

    def redo(self):
        """Redo the command."""
        if not self.undone():
            return
        self.obj.set_text(self.__newtext)
        self.undone_set(False)

class AddToGroupCommand(Command):
    """ Add an object to an existing group """

    def __init__(self, group, obj, page=None):
        super().__init__("add_to_group", objects=None)
        self.__page      = page
        self.__group     = group
        self.__obj       = obj

        group.add(obj)

    def undo(self):
        """Undo the command."""
        if self.undone():
            return None
        self.__group.remove(self.__obj)
        self.undone_set(True)

        return self.__page

    def redo(self):
        """Redo the command."""
        if not self.undone():
            return None
        self.__group.add(self.__obj)
        self.undone_set(False)

        return self.__page

class GroupObjectCommand(Command):
    """Simple class for handling grouping objects."""
    def __init__(self, objects, stack, selection_object = None):
        objects = sort_by_stack(objects, stack)

        super().__init__("group", objects)
        self.__stack      = stack
        self.__stack_copy = stack[:]

        self.__selection = selection_object

        self.__group = DrawableGroup(self.obj)

        # position of the last object in stack
        idx = self.__stack.index(self.obj[-1])

        # add group to the stack at the position of the last object
        self.__stack.insert(idx, self.__group)

        for obj in self.obj:
            if not obj in stack:
                raise ValueError("Object not in stack:", obj)
            self.__stack.remove(obj)

        if self.__selection:
            self.__selection.set([ self.__group ])

    def undo(self):
        """Undo the command."""
        if self.undone():
            return
        swap_stacks(self.__stack, self.__stack_copy)
        self.undone_set(True)
        if self.__selection:
            self.__selection.set(self.obj)

    def redo(self):
        """Redo the command."""
        if not self.undone():
            return
        swap_stacks(self.__stack, self.__stack_copy)
        self.undone_set(False)
        if self.__selection:
            self.__selection.set([ self.__group ])

class UngroupObjectCommand(Command):
    """
    Class for handling ungrouping objects.

    :param objects: Objects to be ungrouped (objects which are not groups
                    will be ignored)
    """
    def __init__(self, objects, stack, selection_object = None):
        super().__init__("ungroup", objects)
        self.__stack = stack
        self.__stack_copy = stack[:]
        self.__selection = selection_object

        new_objects = []
        n = 0

        for obj in self.obj:
            if not obj.type == "group":
                log.warning("Object is not a group, ignoring: %s", obj)
                log.warning("object type: %s", obj.type)
                continue

            n += 1
            # position of the group in the stack
            idx = self.__stack.index(obj)

            # remove the group from the stack
            self.__stack.remove(obj)

            # add the objects back to the stack
            for subobj in obj.objects[::-1]:
                self.__stack.insert(idx, subobj)
                new_objects.append(subobj)

        if n > 0 and self.__selection:
            self.__selection.set(new_objects)

    def undo(self):
        """Undo the command."""
        if self.undone():
            return
        swap_stacks(self.__stack, self.__stack_copy)
        self.undone_set(True)
        if self.__selection:
            self.__selection.set([ self.obj ])

    def redo(self):
        """Redo the command."""
        if not self.undone():
            return
        swap_stacks(self.__stack, self.__stack_copy)
        self.undone_set(False)
        if self.__selection:
            self.__selection.set(self.obj)

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
        #self.__new_type = new_type
        self.__old_objs = [ ]
        self.__new_objs = [ ]
        self.__stack    = stack
        self.__selection_objects = selection_objects
        log.debug("executing transmute; undone = %s", self.undone())

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
        self.__selection_objects[:] = [ obj_map.get(obj, obj) for obj in self.__selection_objects ]

    def obj_map(self):
        """Return a dictionary mapping old objects to new objects."""
        return { self.__old_objs[i]: self.__new_objs[i] for i in range(len(self.__old_objs)) }

    def undo(self):
        """replace all the new objects with the old ones in the stack"""
        if self.undone():
            return
        for obj in self.__new_objs:
            self.__stack.remove(obj)
        for obj in self.__old_objs:
            self.__stack.append(obj)
        self.undone_set(True)
        if self.__selection_objects:
            self.map_selection()
        return

    def redo(self):
        """put the new objects again on the stack and remove the old ones"""
        if not self.undone():
            return
        for obj in self.__old_objs:
            self.__stack.remove(obj)
        for obj in self.__new_objs:
            self.__stack.append(obj)
        self.undone_set(False)
        if self.__selection_objects:
            self.map_selection()
        return

class ZStackCommand(Command):
    """Simple class for handling z-stack operations."""
    def __init__(self, objects, stack, operation):
        super().__init__("z_stack", objects)
        self._operation  = operation
        self.__stack      = stack

        for obj in objects:
            if not obj in stack:
                raise ValueError("Object not in stack:", obj)

        self._objects = sort_by_stack(objects, stack)
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
        for obj in self._objects[::-1]:
            self.__stack.remove(obj)
            self.__stack.insert(0, obj)

    def undo(self):
        """Undo the command."""
        if self.undone():
            return
        swap_stacks(self.__stack, self.__stack_orig)
        self.undone_set(True)

    def redo(self):
        """Redo the command."""
        if not self.undone():
            return
        swap_stacks(self.__stack, self.__stack_orig)
        self.undone_set(False)

class FlushCommand(Command):
    """
    Class for flushing a group of object to left / right / top / bottom.
    """

    def __init__(self, objects, flush_direction):
        name = "flush_" + flush_direction
        super().__init__(name, objects)
        log.debug("flushing objects %s to %s", objects, flush_direction)

        self.__undo_dict = None
        self.__redo_dict = None

        # call the appropriate method
        if flush_direction == "left":
            self.flush_left()
        elif flush_direction == "right":
            self.flush_right()
        elif flush_direction == "top":
            self.flush_top()
        elif flush_direction == "bottom":
            self.flush_bottom()
        else:
            raise ValueError("Invalid flush direction:", flush_direction)

    def flush_left(self):
        """Flush the objects to the left."""
        log.debug("flushing left")
        self.__undo_dict = { obj: obj.bbox() for obj in self.obj }
        min_x = min(obj.bbox()[0] for obj in self.obj)

        for obj in self.obj:
            obj.move(min_x - obj.bbox()[0], 0)

        self.__redo_dict = { obj: obj.bbox() for obj in self.obj }

    def flush_right(self):
        """Flush the objects to the right."""
        self.__undo_dict = { obj: obj.bbox() for obj in self.obj }
        max_x = max(obj.bbox()[2] + obj.bbox()[0] for obj in self.obj)

        for obj in self.obj:
            obj.move(max_x - (obj.bbox()[2] + obj.bbox()[0]), 0)

        self.__redo_dict = { obj: obj.bbox() for obj in self.obj }

    def flush_top(self):
        """Flush the objects to the top."""
        self.__undo_dict = { obj: obj.bbox() for obj in self.obj }
        min_y = min(obj.bbox()[1] for obj in self.obj)

        for obj in self.obj:
            obj.move(0, min_y - obj.bbox()[1])

        self.__redo_dict = { obj: obj.bbox() for obj in self.obj }

    def flush_bottom(self):
        """Flush the objects to the bottom."""
        self.__undo_dict = { obj: obj.bbox() for obj in self.obj }
        max_y = max(obj.bbox()[3] + obj.bbox()[1] for obj in self.obj)

        for obj in self.obj:
            obj.move(0, max_y - (obj.bbox()[3] + obj.bbox()[1]))

        self.__redo_dict = { obj: obj.bbox() for obj in self.obj }

    def move_to_bb(self, obj, bb):
        """Move the object to the bounding box."""
        bb_obj = obj.bbox()
        dx = bb[0] - bb_obj[0]
        dy = bb[1] - bb_obj[1]
        obj.move(dx, dy)

    def undo(self):
        """Undo the command."""
        if self.undone():
            return
        for obj in self.obj:
            self.move_to_bb(obj, self.__undo_dict[obj])
        self.undone_set(True)

    def redo(self):
        """Redo the command."""
        if not self.undone():
            return
        for obj in self.obj:
            self.move_to_bb(obj, self.__redo_dict[obj])
        self.undone_set(False)
"""Commands for setting properties of objects."""

# -------------------------------------------------------------------------------------
class SetPropCommand(Command):
    """
    Superclass for handling property changes of drawing primitives.

    The superclass handles everything, while the subclasses set up the
    functions that do the actual manipulation of the primitives.

    In principle, we need one function to extract the current property, and
    one to set the property.
    """
    def __init__(self, mytype, objects, prop, prop_func):
        super().__init__(mytype, objects.get_primitive())
        self.__prop = prop
        self.__prop_func = prop_func
        self.__undo_dict = { obj: prop_func(obj) for obj in self.obj }
        log.debug("undo_dict: %s", self.__undo_dict)

        for obj in self.obj:
            log.debug("setting prop type %s for %s", mytype, obj)
            prop_func(obj, prop)
            obj.modified(True)

    def __add__(self, other):

        if self == other:
            self.__prop = other.prop()
            return self

        return super().__add__(other)

    def prop(self):
        """Return the property"""
        return self.__prop

    def undo(self):
        """Undo the command."""
        if self.undone():
            return
        for obj in self.obj:
            if obj in self.__undo_dict:
                self.__prop_func(obj, self.__undo_dict[obj])
                obj.modified(True)
        self.undone_set(True)

    def redo(self):
        """Redo the command."""
        if not self.undone():
            return
        for obj in self.obj:
            self.__prop_func(obj, self.__prop)
            obj.modified(True)
        self.undone_set(False)

def set_pen(obj, prop = None):
    """Set the pen property."""
    if prop:
        obj.pen_set(prop)
    return obj.pen

class SetPenCommand(SetPropCommand):
    """Simple class for handling color changes."""
    def __init__(self, objects, pen):
        pen = pen.copy()
        super().__init__("set_pen", objects, pen, set_pen)

def set_transparency(obj, prop = None):
    """Set the transparency property."""
    if prop:
        obj.pen.transparency_set(prop)
    return obj.pen.transparency

class SetTransparencyCommand(SetPropCommand):
    """Simple class for handling line width changes."""
    def __init__(self, objects, width):
        super().__init__("set_transparency", objects, width,
                         set_transparency)

def set_line_width(obj, prop = None):
    """Set the line width property."""
    if prop:
        obj.stroke(prop)
    return obj.stroke()

class SetLineWidthCommand(SetPropCommand):
    """Simple class for handling line width changes."""
    def __init__(self, objects, width):
        super().__init__("set_line_width", objects, width,
                         set_line_width)

def set_color(obj, prop = None):
    """Set the color property."""
    if prop:
        obj.pen.color_set(prop)
    return obj.pen.color

class SetColorCommand(SetPropCommand):
    """Simple class for handling color changes."""
    def __init__(self, objects, color):
        super().__init__("set_color", objects, color, set_color)

def set_font(obj, prop = None):
    """Set the font property."""
    if prop:
        obj.pen.font_set(prop)
    return obj.pen.font_get()

class SetFontCommand(SetPropCommand):
    """Simple class for handling font changes."""
    def __init__(self, objects, font):
        super().__init__("set_font", objects, font, set_font)

class ToggleFillCommand(Command):
    """Simple class for handling toggling fill."""
    def __init__(self, objects):
        super().__init__("fill_toggle", objects.get_primitive())

        for obj in self.obj:
            obj.fill_toggle()

    def undo(self):
        """Undo the command."""
        if self.undone():
            return
        for obj in self.obj:
            obj.fill_toggle()
        self.undone_set(True)

    def redo(self):
        """Redo the command."""
        if not self.undone():
            return
        for obj in self.obj:
            obj.fill_toggle()
        self.undone_set(False)

class ChangeStrokeCommand(Command):
    """Simple class for handling line width changes."""
    def __init__(self, objects, direction):
        super().__init__("change_stroke", objects.get_primitive())

        self.__direction = direction
        self.__undo_dict = { obj: obj.stroke_change(direction) for obj in self.obj }

    def __add__(self, other):
        """Add two commands together."""
        if self == other:
            self.__direction += other.direction()
            return self
        return super().__add__(other)

    def direction(self):
        """Return the direction"""
        return self.__direction

    def undo(self):
        """Undo the command."""
        if self.undone():
            return
        for obj in self.obj:
            if obj in self.__undo_dict:
                obj.stroke(self.__undo_dict[obj])
        self.undone_set(True)

    def redo(self):
        """Redo the command."""
        if not self.undone():
            return
        for obj in self.obj:
            obj.stroke_change(self.__direction)
        self.undone_set(False)
"""Add, Move, Resize, Rotate, and Delete commands for drawable objects."""

class AddCommand(Command):
    """
    Class for handling creating objects.

    :param objects: a list of objects to be removed.
    """
    def __init__(self, objects, stack):
        super().__init__("add", objects)
        self.__stack = stack
        self.__add_objects()

    def __add_objects(self):
        for o in self.obj:
            self.__stack.append(o)

    def undo(self):
        if self.undone():
            return
        for o in self.obj:
            self.__stack.remove(o)
        self.undone_set(True)

    def redo(self):
        if not self.undone():
            return
        self.__add_objects()
        self.undone_set(False)

class RemoveCommand(Command):
    """
    Class for handling deleting objects.

    :param objects: a list of objects to be removed.
    """
    def __init__(self, objects, stack):
        super().__init__("remove", objects)
        self.__stack = stack
        self.__stack_copy = self.__stack[:]

        # remove the objects from the stack
        for obj in self.obj:
            self.__stack.remove(obj)

    def undo(self):
        if self.undone():
            return
        swap_stacks(self.__stack, self.__stack_copy)
        self.undone_set(True)

    def redo(self):
        if not self.undone():
            return
        swap_stacks(self.__stack, self.__stack_copy)
        self.undone_set(False)


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

    This class is different from other classes because it takes a single
    object as the argument. This is because Move or Resize commands need to
    react to continuous updates. It is therefore the responsibility of the
    caller to ensure that a list of objects is grouped as a DrawableGroup.

    Also, the subclasses need to implement two methods: event_update and
    event_finish, which have to handle the changes during move / resize and
    call on objects to finalize the command.
    """

    def __init__(self, mytype, obj, origin):
        super().__init__(mytype, obj)
        self.start_point = origin
        self.origin      = origin

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

    def __init__(self, obj, origin=None, corner=None, angle = None):
        super().__init__("rotate", obj, origin)
        self.corner      = corner
        bb = obj.bbox(actual = True)
        self.bbox        = obj.bbox()
        self.__rotation_centre = (bb[0] + bb[2] / 2, bb[1] + bb[3] / 2)
        obj.rotate_start(self.__rotation_centre)

        self.__angle = 0

        if not angle is None:
            self.obj.rotate(angle, set_angle = False)
            self.__angle = angle

    def event_update(self, x, y):
        angle = calc_rotation_angle(self.__rotation_centre, self.start_point, (x, y))
        d_a = angle - self.__angle
        self.__angle = angle
        self.obj.rotate(d_a, set_angle = False)

    def event_finish(self):
        self.obj.rotate_end()

    def undo(self):
        if self.undone():
            return
        if not self.__angle:
            return
        self.obj.rotate_start(self.__rotation_centre)
        self.obj.rotate(0 - self.__angle)
        self.obj.rotate_end()
        self.undone_set(True)

    def redo(self):
        if not self.undone():
            return
        self.obj.rotate_start(self.__rotation_centre)
        self.obj.rotate(self.__angle)
        self.obj.rotate_end()
        self.undone_set(False)

class MoveCommand(MoveResizeCommand):
    """Simple class for handling move events."""
    def __init__(self, obj, origin):
        obj = obj.objects
        super().__init__("move", obj, origin)
        self.__last_pt = origin
        log.debug("MoveCommand: origin is %s hash %s",
               [int(x) for x in origin], self.hash())

    def __add__(self, other):
        """Add two move commands"""
        if not isinstance(other, MoveCommand):
            return super().__add__(other)

        dx = other.last_pt()[0] - other.start_point[0]
        dy = other.last_pt()[1] - other.start_point[1]
        self.__last_pt = (
                self.__last_pt[0] + dx,
                self.__last_pt[1] + dy
                )
        return self

    def last_pt(self):
        """Return last point"""
        return self.__last_pt

    def event_update(self, x, y):
        """Update the move event."""
        dx = x - self.__last_pt[0]
        dy = y - self.__last_pt[1]

        for obj in self.obj:
            obj.move(dx, dy)
        self.__last_pt = (x, y)

    def event_finish(self):
        """Finish the move event."""
        log.debug("MoveCommand: finish")

    def undo(self):
        """Undo the command."""
        if self.undone():
            return
        log.debug("MoveCommand: undo")
        dx = self.start_point[0] - self.__last_pt[0]
        dy = self.start_point[1] - self.__last_pt[1]
        for obj in self.obj:
            obj.move(dx, dy)
        self.undone_set(True)

    def redo(self):
        """Redo the command."""
        if not self.undone():
            return
        dx = self.start_point[0] - self.__last_pt[0]
        dy = self.start_point[1] - self.__last_pt[1]
        for obj in self.obj:
            obj.move(-dx, -dy)
        self.undone_set(False)

class ResizeCommand(MoveResizeCommand):
    """Simple class for handling resize events."""
    def __init__(self, obj, origin, corner, proportional = False):
        super().__init__("resize", obj, origin)
        self.corner = corner
        self.bbox     = obj.bbox(actual = True)
        obj.resize_start(corner, origin)
        self._orig_bb = obj.bbox(actual = True)
        self._prop    = proportional
        if self._orig_bb[2] == 0:
            raise ValueError("Bounding box with no width")

        self._orig_bb_ratio = self._orig_bb[3] / self._orig_bb[2]
        self.__newbb = None

    def undo(self):
        """Undo the command."""
        if self.undone():
            return
        obj = self.obj
        pt  = (self._orig_bb[0], self._orig_bb[1])
        obj.resize_start(self.corner, pt)
        self.obj.resize_update(self._orig_bb)
        obj.resize_end()
        self.undone_set(True)

    def redo(self):
        """Redo the command."""
        if not self.undone():
            return
        obj = self.obj
        obj.resize_start(self.corner, self.start_point)
        obj.resize_update(self.__newbb)
        obj.resize_end()
        self.undone_set(False)

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

    def stroke(self, line_width = None):
        """Set pen line width"""
        if line_width is not None:
            self.line_width = line_width
        return self.line_width

    def stroke_change(self, direction):
        """Change the line width of the pen"""
        # for thin lines, a fine tuned change of line width
        if self.line_width > 2:
            self.line_width += direction
        else:
            self.line_width += direction / 10
        self.line_width = max(0.1, self.line_width)

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
            self.font_set_from_description(self.font_description)
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
        fstr = f"{self.font_family} {self.font_style} {self.font_weight} {self.font_size}"
        self.font_description = Pango.FontDescription.from_string(fstr)


    def font_set_from_description(self, font_description):
        """Set font based on a Pango.FontDescription"""
        log.debug("setting font from %s", font_description)
        self.font_description = font_description
        self.font_family = font_description.get_family()
        self.font_size   = font_description.get_size() / Pango.SCALE
        wgt = font_description.get_weight()
        self.font_weight = "bold"   if wgt == Pango.Weight.BOLD  else "normal"
        sty = font_description.get_style()
        self.font_style  = "italic" if sty == Pango.Style.ITALIC else "normal"

        log.debug("setting font to %s %s %s %s",
              self.font_family, self.font_size, self.font_weight, self.font_style)

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
        return cls(d.get("color"), d.get("line_width"), d.get("transparency"),
                   d.get("fill_color"), d.get("font_size"), d.get("font_family"),
                   d.get("font_weight"), d.get("font_style"), d.get("brush"))
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
    def __init__(self, event, state):
        self.event = event
        self.state = state

        self.x_abs, self.y_abs = event.x, event.y

        self.x, self.y = state.pos_abs_to_rel((event.x, event.y))


        self.__info = {
                "mode": state.mode(),
                "shift": (event.state & Gdk.ModifierType.SHIFT_MASK) != 0,
                "ctrl": (event.state & Gdk.ModifierType.CONTROL_MASK) != 0,
                "alt": (event.state & Gdk.ModifierType.MOD1_MASK) != 0,
                "double":   (event.type == Gdk.EventType.DOUBLE_BUTTON_PRESS),
                "pressure": event.get_axis(Gdk.AxisUse.PRESSURE),
                "hover":    None,
                "corner":   [ ], 
                "pos": (self.x, self.y),
                "pos_abs": (self.x_abs, self.y_abs)
                }

        if self.__info["pressure"] is None:  # note that 0 is perfectly valid
            self.__info["pressure"] = 1


    def hover(self):
        """Return the object that is hovered by the mouse."""
        objects = self.state.objects()
        pos = self.__info["pos"]
        if not self.__info.get("hover"):
            self.__info["hover"] = find_obj_close_to_click(pos[0],
                                                   pos[1],
                                                   objects, 20)
        return self.__info["hover"]

    def corner(self):
        """Return the corner that is hovered by the mouse."""
        objects = self.state.objects()
        pos = self.__info["pos"]
        if not self.__info.get("corner"):
            self.__info["corner"] = find_corners_next_to_click(pos[0],
                                                       pos[1],
                                                       objects, 20)
        return self.__info["corner"][0], self.__info["corner"][1]

    def pos_abs(self):
        """Return the position of the mouse."""
        return self.__info["pos_abs"]

    def pos(self):
        """Return the position of the mouse."""
        return self.__info["pos"]

    def shift(self):
        """Return True if the shift key is pressed."""
        return self.__info.get("shift")

    def ctrl(self):
        """Return True if the control key is pressed."""
        return self.__info.get("ctrl")

    def alt(self):
        """Return True if the alt key is pressed."""
        return self.__info.get("alt")

    def double(self):
        """Return True if the event is a double click."""
        return self.__info.get("double")

    def pressure(self):
        """Return the pressure of the pen."""
        return self.__info.get("pressure")

    def mode(self):
        """Return the mode in which the event happened."""
        return self.__info.get("mode")


class MouseCatcher:
    """
    Class that catches mouse events, creates the MouseEvent and sends it
    back to the bus.

    """
    def __init__(self, bus, state):
        self.__bus = bus
        self.__state = state

        # objects that indicate the state of the drawing area
        # drawing parameters

    # ---------------------------------------------------------------------
    #                              Event handlers

    def on_pan(self, gesture, direction, offset):
        """Handle panning events."""

    def on_zoom(self, gesture, scale):
        """Handle zoom events."""

    # ---------------------------------------------------------------------

    # Button press event handlers -------------------------------------------
    def on_button_press(self, widget, event): # pylint: disable=unused-argument
        """Handle mouse button press events."""
        log.debug("type:{event.type} button:{event.button} state:%s", event.state)
        self.__state.graphics().modified(True)
        ev = MouseEvent(event, state = self.__state)

        if event.button == 3:
            if self.__handle_button_3(ev):
                return True

        elif event.button == 1:
            if self.__handle_button_1(event, ev):
                return True

        return True

    def __handle_button_3(self, ev):
        """Handle right click events, unless shift is pressed."""
        if self.__bus.emit("right_mouse_click", True, ev):
            return True

        return False

    def __handle_button_1(self, event, ev):
        """Handle left click events."""

        if ev.double():
            log.debug("dblclick (%d, %d) raw (%d, %d)",
                      int(ev.x), int(ev.y), int(event.x), int(event.y))
            self.__bus.emit("cancel_left_mouse_single_click", True, ev)
            self.__bus.emit("left_mouse_double_click", True, ev)
            return True

        log.debug("sngle clck (%d, %d) raw (%d, %d)",
                  int(ev.x), int(ev.y), int(event.x), int(event.y))

        if self.__bus.emit("left_mouse_click", True, ev):
            log.debug("bus event caught the click")
            self.__bus.emit("queue_draw")

        return False

    def on_button_release(self, widget, event): # pylint: disable=unused-argument
        """Handle mouse button release events."""
        log.debug("button release: type:%s button:%s state:%s",
                  event.type, event.button, event.state)
        ev = MouseEvent(event, state = self.__state)

        if self.__bus.emit("mouse_release", True, ev):
            self.__bus.emit("queue_draw")
            return True

        return True

    # ---------------------------------------------------------------------
    # motion event handlers

    def on_motion_notify(self, widget, event): # pylint: disable=unused-argument
        """Handle mouse motion events."""

        ev = MouseEvent(event, state = self.__state)

        self.__bus.emit_mult("cursor_pos_update", ev.pos(), ev.pos_abs())

        if self.__bus.emit_once("mouse_move", ev):
            self.__bus.emit("queue_draw")
            return True

        return True
"""
Dialogs for the ScreenDrawer application.
"""


## ---------------------------------------------------------------------
FORMATS = {
    "All files": { "pattern": "*",      "mime_type": "application/octet-stream", "name": "any" },
    "PNG files":  { "pattern": "*.png",  "mime_type": "image/png",       "name": "png" },
    "JPEG files": { "pattern": "*.jpeg", "mime_type": "image/jpeg",      "name": "jpeg" },
    "PDF files":  { "pattern": "*.pdf",  "mime_type": "application/pdf", "name": "pdf" }
}


## ---------------------------------------------------------------------
class HelpDialog(Gtk.Dialog):
    """A dialog to show help information."""
    def __init__(self, parent):
        log.debug("parent: {parent}")
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
<b>General / Interface:</b>
w: Toggle UI (hide / show widgets)         F1, h, ?: Show this help dialog
Ctrl-q, x, q: Quit

<b>UI:</b> (toggle with 'w')
Color selector: click to select pen color, shift-click to select background color
Tool selector: click to select tool
Page selector: click to select page, click on '+' to add a new page

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
<b>s:</b> Shape mode (draw a filled shape)   <b>i:</b> Color p<b>i</b>cker mode (pick a color from the screen)
<b>Shift-s:</b> Segmented path mode (double-click to finish a segmented path)

<b>Works always:</b>                                                             <b>Move mode only:</b>
<b>With Ctrl:</b>              <b>Simple key (not when entering text)</b>               <b>With Ctrl:</b>             <b>Simple key (not when entering text)</b>
Ctrl-e: Export drawing                                                    Ctrl-c: Copy content   Tab: Next object
Ctrl-l: Clear drawing   l: Clear drawing                                  Ctrl-v: Paste content  Shift-Tab: Previous object
                                                                          Ctrl-x: Cut content    Shift-letter: quick color selection e.g.
                                                                                                 Shift-r for red
Ctrl-i: insert image                                                                             |Del|: Delete selected object(s)
Ctrl-z: undo            |Esc|: Finish text input
Ctrl-y: redo            |Enter|: New line (when typing)                   Alt-Up, Alt-Down: Move object up, down in stack
                                                                          Alt-PgUp, Alt-PgDown: Move object to front, back
Ctrl-k: Select color                     f: fill with current color       Alt-s: convert drawing(s) to shape(s)
Ctrl-Shift-k: Select bg color
Ctrl-plus, Ctrl-minus: Change text size  o: toggle outline                Alt-d: convert shape(s) to drawing(s)
Ctrl-b: Cycle background transparency                                     Alt-p: apply pen to selection
Ctrl-p: toggle between two pens                                           Alt-Shift-p: apply pen color to background
Ctrl-g: toggle grid                      1-5: select brush

Ctrl-Shift-g: toggle "group while drawing" mode

<b>Brushes:</b> (select with 1-5)

1: general rounded brush
2: marker
3: calligraphy
4: pencil (pressure changes transparency)
5: tapered brush

<b>Group operations:</b>
g, u: group, ungroup (move mode only)
Shift-c, Shift-u: clip / unclip group (need a rectangle, circle or shape as the last selected object)

Alt-Shift-Arrow: flush objects in a group or selection:
    Alt-Shift-Left:      to the left
    Alt-Shift-Right:     to the right
    Alt-Shift-Up:        to the top
    Alt-Shift-Down:      to the bottom

<b>Pages and layers:</b>

Pages (= slides) can hold multiple layers. When you select and move
objects, you are always acting on the current layer.

Shift-n: Next / new page                  Ctrl-Shift-n: Next / new layer
Shift-p: Previous page                    Ctrl-Shift-p: Previous layer
Shift-d: Delete current page              Ctrl-Shift-d: Delete layer
Shift-i: Insert new page after current

If you have more than one page, exporting to PDF will create a multipage PDF
if you check the "multipage" checkbox.

<b>Taking screenshots:</b>
Ctrl-Shift-f: screenshot: for a screenshot, if you have at least one rectangle                    This is likely to change in the future.
object (r mode) selected, then it will serve as the selection area. The
screenshot will be pasted into the drawing. If no rectangle is selected, then
the mode will change to "rectangle" and the next rectangle you draw will be
used as the capture area.

<b>Saving / importing:</b>
Ctrl-i: Import image from a file (jpeg, png)
Ctrl-o: Open a drawing from a file (.sdrw, that is the "native format") -
        note that the subsequent modifications will be saved to that file only
Ctrl-e: Export selection or whole drawing to a file (png, jpeg, pdf)
Ctrl-Shift-s: "Save as" - save drawing to a file (.sdrw, that is the "native format") - note
        that the subsequent modifications will be saved to that file only
Ctrl-c, Ctrl-v: copy and paste objects

When you copy a selection or individual objects, you can paste them into
other programs as a PNG image.

If you have more than one page, exporting to PDF will create a multipage PDF
if you check the "multipage" checkbox.

<b>Group while drawing mode:</b>
To make drawing complex objects (or handwriting, for that matter) easier,
you can press Ctrl-Shift-g to toggle "group while drawing" mode. In this mode,
all objects you draw are automatically grouped together. Stop by pressing
either the escape key or Ctrl-Shift-g again.

</span>

The state is saved in / loaded from `{parent.state.config().savefile()}` so you can continue drawing later.
An autosave happens every minute or so. Press ESC to exit this help screen.
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

def export_dialog_extra_widgets(ui_opts = None):
    """Options and format chooser for the export dialog"""

    format_selector = Gtk.ComboBoxText()
    formats = [ "By extension", "PDF", "SVG", "PNG"]
    for fmt in formats:
        format_selector.append_text(fmt)

    # Set the default selection to either
    # ui_opts["format"] or if that is None, to 0
    if ui_opts and ui_opts.get("format"):
        format_selector.set_active(formats.index(ui_opts["format"]))
    else:
        format_selector.set_active(0)

    # Create a checkbox for "export all pages as PDF"
    export_all_checkbox = Gtk.CheckButton(label="Export all pages as PDF")
    if ui_opts and ui_opts.get("all_pages_pdf"):
        export_all_checkbox.set_active(ui_opts["all_pages_pdf"])

    # Checkbox for "export screen view"
    export_screen_checkbox = Gtk.CheckButton(label="Export screen view")
    if ui_opts and ui_opts.get("export_screen"):
        export_screen_checkbox.set_active(ui_opts["export_screen"])

    # Function to update checkbox sensitivity
    def update_checkbox_sensitivity(combo):
        selected_format = combo.get_active_text()
        export_all_checkbox.set_sensitive(selected_format in [ "PDF", "By extension"])

    # Connect the combo box's changed signal to update the checkbox
    format_selector.connect("changed", update_checkbox_sensitivity)

    # Initial update of checkbox sensitivity
    update_checkbox_sensitivity(format_selector)

    # Create a label for the selector
    label = Gtk.Label(label="File Format:")

    # Create a horizontal box to hold the label and selector
    hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
    hbox.pack_start(label, False, False, 0)
    hbox.pack_start(format_selector, False, False, 0)
    hbox.pack_start(export_all_checkbox, False, False, 0)
    hbox.pack_start(export_screen_checkbox, False, False, 0)

    return hbox, format_selector, export_all_checkbox, export_screen_checkbox

def __mk_export_dialog(parent, ui_opts):
    """Create a file chooser dialog for exporting."""

    selected = ui_opts.get("selected", False)
    ## doesn't really work because we don't have a standalone window with
    ## its own title bar
    title = "Export selected objects As" if selected else "Export all objects As"

    dialog = Gtk.FileChooserDialog(
        title=title, parent=parent, action=Gtk.FileChooserAction.SAVE)

    dialog.add_buttons(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                       Gtk.STOCK_SAVE, Gtk.ResponseType.OK)
    dialog.set_modal(True)

    return dialog

def __export_dialog_set_extras(dialog, ui_opts):
    """Set the extra widgets for the export dialog."""

    ret =  export_dialog_extra_widgets(ui_opts = ui_opts)
    hbox, fmt_selector, exp_all_cb, exp_screen_cb = ret

    dialog.set_extra_widget(hbox)
    hbox.show_all()

    return fmt_selector, exp_all_cb, exp_screen_cb

def __export_dialog_init_state(dialog, export_dir, filename):
    """Initialize the state of the export dialog."""

    current_directory = export_dir or os.getcwd()
    log.debug("current_directory: {current_directory}, filename: %s", filename)

    if filename:
        dialog.set_filename(filename)

    dialog.set_current_folder(current_directory)
    _dialog_add_image_formats(dialog)

    log.debug("filter: %s", dialog.get_filter())

def export_dialog(parent, export_dir = None, filename=None, ui_opts = None):
    """Show a file chooser dialog to select a file to save the drawing as
    an image / pdf / svg."""
    log.debug("export_dialog")

    dialog = __mk_export_dialog(parent, ui_opts)
    ret    = __export_dialog_set_extras(dialog, ui_opts)

    __export_dialog_init_state(dialog, export_dir, filename)

    all_pages_pdf = False

    # Show the dialog
    response = dialog.run()

    if not response == Gtk.ResponseType.OK:
        dialog.destroy()
        return None, None, None, None

    fmt_selector, exp_all_cb, exp_screen_cb = ret

    file_name = dialog.get_filename()
    selected_format = fmt_selector.get_active_text()
    all_pages_pdf = exp_all_cb.get_active()
    export_screen = exp_screen_cb.get_active()

    log.debug("Save file as: %s; Format: %s",
              file_name, selected_format)
    log.debug("all pages as PDF: %s; export_screen: %s",
              all_pages_pdf, export_screen)

    dialog.destroy()
    return file_name, selected_format, all_pages_pdf, export_screen

def save_dialog(parent):
    """Show a file chooser dialog to set the savefile."""
    log.debug("save_dialog")
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

def import_image_dialog(parent, import_dir = None):
    """Show a file chooser dialog to select an image file."""
    dialog = Gtk.FileChooserDialog(
        title="Select an Image",
        parent=parent,
        action=Gtk.FileChooserAction.OPEN,
        buttons=(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, Gtk.STOCK_OPEN, Gtk.ResponseType.OK)
    )
    dialog.set_modal(True)
    current_directory = import_dir or os.getcwd()
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

    image_path = None
    if response == Gtk.ResponseType.OK:
        image_path = dialog.get_filename()
    elif response == Gtk.ResponseType.CANCEL:
        log.warning("No image selected")

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

def font_chooser(pen, parent):
    """Dialog to choose a Font."""

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

def color_chooser(parent, title = "Select Pen Color"):
    """Select a color for drawing."""
    # Create a new color chooser dialog
    __color_chooser = Gtk.ColorChooserDialog(title, parent = parent)

    # Show the dialog
    response = __color_chooser.run()
    color = None

    # Check if the user clicked the OK button
    if response == Gtk.ResponseType.OK:
        color = __color_chooser.get_rgba()

    __color_chooser.destroy()
    return color
"""Handling the clipboard operations."""
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
        self.__gtk_clipboard = gtk_clipboard or Gtk.Clipboard.get(Gdk.SELECTION_CLIPBOARD)
        self.__gtk_clipboard.connect('owner-change', self.on_clipboard_owner_change)
        self.clipboard_owner = False
        self.clipboard = None


    def on_clipboard_owner_change(self, clipboard, event):
        """Handle clipboard owner change events."""

        log.debug("Owner change (%s), removing internal clipboard, reason: %s",
                 clipboard, event.reason)
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
        self.__gtk_clipboard.set_text(text, -1)
        self.__gtk_clipboard.store()

    def get_content(self):
        """Return the clipboard content."""

        # internal paste
        if self.clipboard:
            log.debug("Pasting content internally")
            return "internal", self.clipboard

        # external paste
        clipboard = self.__gtk_clipboard
        clip_text = clipboard.wait_for_text()

        if clip_text:
            return "text", clip_text

        clip_img = clipboard.wait_for_image()
        if clip_img:
            return "image", clip_img
        return None, None

    def copy_content(self, selection, cut = False):
        """
        Copy internal content: object or objects from current selection.

        Args:
            selection (Drawable): The selection to copy, either a group or
                                  a single object.
        """
        clipboard = self.__gtk_clipboard

        # don't like the code below
        if selection.length() == 1:
            sel = selection.objects[0]
        else:
            sel = selection

        if sel.type == "text":
            text = sel.to_string()
            log.debug("Copying text %s", text)
            # just copy the text
            clipboard.set_text(text, -1)
            clipboard.store()
        elif sel.type == "image":
            log.debug("Copying image")
            # simply copy the image into clipboard
            clipboard.set_image(sel.image())
            clipboard.store()
        else:
            log.debug("Copying another object")
            # draw a little image and copy it to clipboard
            img_copy = img_object_copy(sel)
            clipboard.set_image(img_copy)
            clipboard.store()

        log.debug("Setting internal clipboard")
        self.clipboard = ClipboardGroup(selection.objects[:], cut = cut)
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
        __window (Gtk.Window): The window to manage the cursor for.
        __cursors (dict):       A dictionary of premade cursors for different modes.
        __current_cursor (str): The name of the current cursor.
        __default_cursor (str): The name of the default cursor.
        __pos (tuple):          The current position of the cursor.

    """


    def __init__(self, window, bus):
        self.__window  = window
        self.__cursors = None
        self.__crossline = False
        self.__current_cursor = "default"
        self.__default_cursor = "default"

        self.__make_cursors(window)

        self.default("default")

        self.__pos     = None
        self.__pos_abs = None

        self.__bus = bus
        self.__bus.on("cursor_pos_update", self.update_pos, 999)
        self.__bus.on("cursor_abs_pos_update", self.update_pos_abs, 999)
        self.__bus.on("cursor_set", self.set)
        self.__bus.on("cursor_revert", self.revert)
        self.__bus.on("toggle_crosslines", self.toggle_crosslines)

    def __make_cursors(self, window):
        """Create cursors for different modes."""

        icons = Icons()
        colorpicker = icons.get("colorpicker")

        self.__cursors = {
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
            "segment":     Gdk.Cursor.new_from_name(window.get_display(), "crosshair"),
            "circle":      Gdk.Cursor.new_from_name(window.get_display(), "crosshair"),
            "rectangle":   Gdk.Cursor.new_from_name(window.get_display(), "crosshair"),
            "none":        Gdk.Cursor.new_from_name(window.get_display(), "none"),
            "upper_left":  Gdk.Cursor.new_from_name(window.get_display(), "nw-resize"),
            "upper_right": Gdk.Cursor.new_from_name(window.get_display(), "ne-resize"),
            "lower_left":  Gdk.Cursor.new_from_name(window.get_display(), "sw-resize"),
            "lower_right": Gdk.Cursor.new_from_name(window.get_display(), "se-resize"),
            "default":     Gdk.Cursor.new_from_name(window.get_display(), "pencil")
        }

    def toggle_crosslines(self):
        """Toggle the crossline under the cursor."""
        self.__crossline = not self.__crossline

        if self.__crossline:
            self.__bus.on("draw", self.draw)
        else:
            self.__bus.off("draw", self.draw)
        self.__bus.emit("queue_draw")

    def draw(self, cr, state):
        """Draw a crossline under the cursor."""

        if self.__pos_abs is None:
            x, y = state.cursor_pos_abs()
        else:
            x, y = self.__pos_abs

        w, h = state.get_win_size()
        cr.save()
        cr.set_source_rgba(0.4, 0.4, 0.4, 0.4)
        cr.set_line_width(1)
        cr.move_to(0, y)
        cr.line_to(w, y)
        cr.stroke()
        cr.move_to(x, 0)
        cr.line_to(x, h)
        cr.stroke()
        cr.restore()

    def pos_absolute(self):
        """Return the current position of the cursor."""
        if self.__pos_abs is None:
            log.debug("no cursor position")
            self.__bus.emit_once("query_cursor_pos")

        if self.__pos_abs is None:
            return (0, 0)
        x, y = self.__pos_abs
        return (x, y)

    def pos(self):
        """Return the current position in draw coordinates."""
        if self.__pos is None:
            return (0, 0)
        x, y = self.__pos
        return (x, y)

    def update_pos_abs(self, pos_abs):
        """Update the current absolute position of the cursor."""

        self.__pos_abs = pos_abs
        self.__bus.emit("queue_draw")
        return False

    def update_pos(self, pos, pos_abs):
        """Update the current position of the cursor."""

        self.__pos     = pos
        self.__pos_abs = pos_abs
        self.__bus.emit("queue_draw")
        return False

    def default(self, cursor_name):
        """Set the default cursor to the specified cursor."""
        if self.__current_cursor == cursor_name:
            return
        log.debug("setting default cursor to %s", cursor_name)
        self.__default_cursor = cursor_name
        self.__current_cursor = cursor_name

        self.__window.get_window().set_cursor(self.__cursors.get(cursor_name))

    def revert(self):
        """Revert to the default cursor."""
        if self.__current_cursor == self.__default_cursor:
            return
        self.__window.get_window().set_cursor(self.__cursors[self.__default_cursor])
        self.__current_cursor = self.__default_cursor

    def set(self, cursor_name):
        """Change the cursor to the specified cursor."""
        if self.__current_cursor == cursor_name:
            return
        self.__window.get_window().set_cursor(self.__cursors[cursor_name])
        self.__current_cursor = cursor_name
"""
Graphics Object Manager is the top level for handling the graphic objects.
The hierarchy is GOM -> Page -> Layer. Layer does all the footwork, Page
handles Layers, and GOM handles Pages.
"""




## ---------------------------------------------------------------------

class GraphicsObjectManager:
    """
    Class to manage graphics objects - in practice, to manage pages.

    Attributes:
        _objects (list): The list of objects.
    """

    def __init__(self, bus):

        # private attr
        self.__bus        = bus
        self.__page = None
        self.page_set(Page())
        self.__add_bus_listeners()

    def __add_bus_listeners(self):
        """Add listeners to the bus."""
        self.__bus.on("next_page", self.next_page)
        self.__bus.on("prev_page", self.prev_page)
        self.__bus.on("insert_page", self.insert_page)
        self.__bus.on("page_set", self.page_set)
        self.__bus.on("delete_page", self.delete_page)

    def page_set(self, page):
        """Set the current page."""
        self.__bus.emit("page_changed", False, page)
        self.__page = page
        self.__page.activate(self.__bus)

    def next_page(self):
        """Go to the next page."""
        self.page_set(self.__page.next(create = True))

    def insert_page(self):
        """Insert a new page."""
        curpage, cmd = self.__page.insert()
        self.__bus.emit_once("history_append", cmd)
        self.page_set(curpage)

    def prev_page(self):
        """Go to the prev page."""
        self.page_set(self.__page.prev())

    def delete_page(self):
        """Delete the current page."""
        curpage, cmd = self.__page.delete()

        if curpage == self.page():
            return

        self.page_set(curpage)
        self.__bus.emit("history_append", True, cmd)

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
        return self.__page.layer().objects()

    def selection(self):
        """Return the selection object."""
        return self.__page.layer().selection()

    def set_objects(self, objects):
        """Set the list of objects."""
        ## no undo
        log.debug("GOM: setting n=%d objects", len(objects))
        self.__page.layer().objects(objects)

    def set_pages(self, pages):
        """Set the content of pages."""
        self.__page = Page()
        self.__page.import_page(pages[0])
        for p in pages[1:]:
            self.__page = self.__page.next()
            self.__page.import_page(p)
        self.__page.activate(self.__bus)

    def get_all_pages(self):
        """Return all pages."""
        p = self.__page
        while p.prev() != p:
            p = p.prev()

        pages = [ ]

        while p:
            pages.append(p)
            p = p.next(create = False)

        return pages

    def export_pages(self):
        """Export all pages."""

        pages = [ p.export() for p in self.get_all_pages() ]
        return pages

    def selected_objects(self):
        """Return the selected objects."""
        return self.__page.layer().selection().objects

    def remove_selection(self):
        """Remove the selected objects from the list of objects."""
        if self.__page.layer().selection().is_empty():
            return
        cmd = RemoveCommand(self.__page.layer().selection().objects,
                                            self.__page.layer().objects())
        self.__bus.emit("history_append", True, cmd)
        self.__page.layer().selection().clear()

    def command_append(self, command_list):
        """Append a group of commands to the history."""
        ## append in reverse order!
        cmd = CommandGroup(command_list[::-1])
        self.__bus.emit("history_append", True, cmd)
"""
This module provides functions to import and export drawings in various formats.
"""




def __draw_object(cr, obj, bg, bbox, transparency):
    """Draw an object on a Cairo context."""

    cr.save()

    # we translate the object to the origin of the context
    cr.translate(-bbox[0], -bbox[1])

    # paint the background
    cr.set_source_rgba(*bg, transparency)
    cr.paint()

    obj.draw(cr)

    cr.restore()


def __find_max_width_height(obj_list):
    """Find the maximum width and height of a list of objects."""
    width, height = None, None

    for o in obj_list:
        bb = o.bbox()

        if bb is None:
            continue

        if not width or not height:
            width, height = bb[2], bb[3]
            continue

        width  = max(width, bb[2])
        height = max(height, bb[3])

    return width, height

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
    if file_format not in [ "png", "jpeg", "pdf", "svg", "yaml" ]:
        raise ValueError("Unrecognized file extension")

    return file_format

def convert_file(input_file, output_file, file_format = "any", border = None, page_no = None):
    """
    Convert a drawing from the internal format to another format.

    :param input_file: The name of the file to read from.
    :param output_file: The name of the file to save to.
    :param file_format: The format of the file to save to. If "any", the
    format will be guessed from the file extension.
    :param border: The border around the objects.
    :param page_no: The page number to export (if the drawing has multiple pages).
    """
    log.debug("Converting file %s to %s as %s page_no=%s",
              input_file, output_file, file_format, page_no)
    if file_format == "any":
        if output_file is None:
            raise ValueError("No output file format provided")
        file_format = guess_file_format(output_file)
    else:
        if output_file is None:
            # create a file name with the same name but different extension
            output_file = path.splitext(input_file)[0] + "." + file_format

    # if we have a page specified, we need to convert to pdf using the
    # one-page converter
    if file_format in [ "png", "jpeg", "svg" ] or (file_format == "pdf"
                                                   and page_no is not None):
        if not page_no:
            page_no = 0
        convert_file_to_image(input_file, output_file, file_format, border, page_no)

   #elif file_format == "yaml":
   #    print("Exporting to yaml")
   #    export_file_as_yaml(output_file, config, objects=objects.to_dict())

    elif file_format == "pdf":
        log.debug("Converting to multipage PDF")
        convert_to_multipage_pdf(input_file, output_file, border)
    else:
        raise NotImplementedError("Conversion to " + file_format + " is not implemented")

def convert_file_to_image(input_file, output_file, file_format = "png", border = None, page_no = 0):
    """
    Convert a drawing to an image file: png, jpeg, svg.

    :param input_file: The name of the file to read from.
    :param output_file: The name of the file to save to.
    :param file_format: The format of the file to save to.
    :param border: The border around the objects.
    :param page_no: The page number to export (if the drawing has multiple pages).
    """
    if page_no is None:
        page_no = 0

    # yeah so we read the file twice, shoot me
    config, objects, pages = read_file_as_sdrw(input_file)

    if pages:
        if len(pages) <= page_no:
            raise ValueError(f"Page number out of range (max. {len(pages) - 1})")
        log.debug("read drawing from %s with %d pages",
                  input_file, len(pages))
        p = Page()
        p.import_page(pages[page_no])
        objects = p.objects_all_layers()

    log.debug("read drawing from %s with %d objects",
              input_file, len(objects))

    if not objects:
        log.warning("No objects found in the input file on page %s", page_no)
        return

    objects = DrawableGroup(objects)


    bbox = objects.bbox()
    if border:
        bbox = objects.bbox()
        bbox = (bbox[0] - border, bbox[1] - border, bbox[2] + 2 * border, bbox[3] + 2 * border)

    bg           = config.get("bg_color", (1, 1, 1))
    transparency = config.get("transparent", 1.0)

    cfg = { "bg": bg, "bbox": bbox, "transparency": transparency, "border": border }

    export_image(objects,
                 output_file, file_format, cfg)

def convert_to_multipage_pdf(input_file, output_file, border = None):
    """
    Convert a native drawing to a multipage PDF file.

    :param input_file: The name of the file to read from.
    :param output_file: The name of the file to save to.
    :param border: The border around the objects.
    """
    log.debug("Converting to multipage PDF")
    config, _, pages = read_file_as_sdrw(input_file)
    if not pages:
        raise ValueError("No multiple pages found in the input file")

    page_obj = []

    for p in pages:
        page = Page()
        page.import_page(p)
        obj_grp = DrawableGroup(page.objects_all_layers())
        page_obj.append(obj_grp)

    export_objects_to_multipage_pdf(page_obj, output_file, config, border)

def export_objects_to_multipage_pdf(obj_list, output_file, config, border = 10):
    """
    Export a list of objects to a multipage PDF file.

    :param obj_list: A list of objects to export.
    :param output_file: The name of the file to save to.
    :param config: The configuration of the drawing (dict).
    :param border: The border around the objects.

    Each object in the list will be drawn on a separate page.
    """
    if not border:
        border = 0

    log.debug("Exporting %s objects to multipage PDF with border %s",
              len(obj_list), border)

    width, height = __find_max_width_height(obj_list)

    bg           = config.get("bg_color", (1, 1, 1))
    transparency = config.get("transparent", 1.0)

    width, height = int(width + 2 * border), int(height + 2 * border)

    surface = cairo.PDFSurface(output_file, width, height)
    cr = cairo.Context(surface)

    cr.set_source_rgba(*bg, transparency)
    cr.paint()

    nobj = len(obj_list)

    # each object is a DrawableGroup for a single page
    for i, o in enumerate(obj_list):
        bb = o.bbox()

        # some pages might be empty
        if bb:
            cr.save()
            cr.translate(border - bb[0], border - bb[1])
            o.draw(cr)
            cr.restore()

        # do not show_page on the last page.
        if i < nobj - 1:
            surface.show_page()
    surface.finish()

def export_image_jpg(obj, output_file, bg = (1, 1, 1), bbox = None, transparency = 1.0):
    """Export the drawing to a JPEG file."""
    raise NotImplementedError("JPEG export is not implemented")

def export_image_pdf(obj, output_file, cfg):
    """
    Export the drawing to a single-page PDF file.

    :param obj: The object to export. This is a single object, since
    generating a DrawableGroup object from multiple objects is trivial.
    :param output_file: The name of the file to save to.
    :param bg: The background color of the image.
    :param bbox: The bounding box of the image. If None, it will be calculated
    from the object.
    :param transparency: The transparency of the image.
    """

    bbox = cfg.get("bbox", None)
    bg   = cfg.get("bg", (1, 1, 1))
    transparency = cfg.get("transparency", 1.0)

    if bbox is None:
        bbox = obj.bbox()

    # to integers
    width, height = int(bbox[2]), int(bbox[3])
    surface = cairo.PDFSurface(output_file, width, height)
    cr = cairo.Context(surface)
    __draw_object(cr, obj, bg, bbox, transparency)
    surface.finish()

def export_image_svg(obj, output_file, cfg):
    """
    Export the drawing to a SVG file.

    :param obj: The object to export. This is a single object, since
    generating a DrawableGroup object from multiple objects is trivial.
    :param output_file: The name of the file to save to.
    :param bg: The background color of the image.
    :param bbox: The bounding box of the image. If None, it will be calculated
    from the object.
    :param transparency: The transparency of the image.
    """

    bbox = cfg.get("bbox", None)
    bg   = cfg.get("bg", (1, 1, 1))
    transparency = cfg.get("transparency", 1.0)

    if bbox is None:
        bbox = obj.bbox()

    # to integers
    width, height = int(bbox[2]), int(bbox[3])
    surface = cairo.SVGSurface(output_file, width, height)
    cr = cairo.Context(surface)
    __draw_object(cr, obj, bg, bbox, transparency)

    surface.finish()

def export_image_png(obj, output_file, cfg):
    """
    Export the drawing to a PNG file.

    :param obj: The object to export. This is a single object, since
    generating a DrawableGroup object from multiple objects is trivial.
    :param output_file: The name of the file to save to.
    :param bg: The background color of the image.
    :param bbox: The bounding box of the image. If None, it will be calculated
    from the object.
    :param transparency: The transparency of the image.
    """

    log.debug("Exporting image as PNG to file %s", output_file)
    res_scale = 5

    bbox = cfg.get("bbox", None)
    log.debug("Bounding box: %s", bbox)
    bg   = cfg.get("bg", (1, 1, 1))
    transparency = cfg.get("transparency", 1.0)

    if bbox is None:
        bbox = obj.bbox()

    # to integers
    width, height = int(bbox[2]), int(bbox[3])
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, res_scale * width,
                                 res_scale * height)
    cr = cairo.Context(surface)
    cr.scale(res_scale, res_scale)
    __draw_object(cr, obj, bg, bbox, transparency)

    surface.write_to_png(output_file)

def export_image(obj, output_file, file_format = "any", config = None, all_pages_pdf = False):
    """
    Export the drawing to a file.

    :param obj: The object to export. This is a single object, since
    generating a DrawableGroup object from multiple objects is trivial.
    :param output_file: The name of the file to save to.
    :param file_format: The format of the file to save to. If "any", the
    format will be guessed from the file extension.
    :param bg: The background color of the image.
    :param bbox: The bounding box of the image. If None, it will be calculated
    from the object.
    :param transparency: The transparency of the image.
    :param all_pages_pdf: If True, all pages will be exported to a single PDF file.
    """
    if not config:
        config = { "bg": (1, 1, 1), "bbox": None, "transparency": 1.0, "border": None }

    log.debug("exporting to %s file %s all_pages_pdf: %s",
        file_format, output_file, all_pages_pdf)

    # if output_file is None, we send the output to stdout
    if output_file is None:
        log.debug("export_image: no output_file provided")
        return

    if file_format == "any":
        # get the format from the file name
        _, file_format = path.splitext(output_file)
        file_format = file_format[1:]
        # lower case
        file_format = file_format.lower()
        # jpg -> jpeg
        if file_format == "jpg":
            file_format = "jpeg"
        # check
        if file_format not in [ "png", "jpeg", "pdf", "svg" ]:
            raise ValueError("Unrecognized file extension")
        log.debug("guessing format from file name: %s",
            file_format)

    # Create a Cairo surface of the same size as the bbox
    if file_format == "png":
        export_image_png(obj, output_file, config)
    elif file_format == "svg":
        export_image_svg(obj, output_file, config)
    elif file_format == "pdf":
        if all_pages_pdf:
            export_objects_to_multipage_pdf(obj, output_file, config, border=10)
        else:
            export_image_pdf(obj, output_file, config)
    else:
        raise NotImplementedError("Export to " + file_format + " is not implemented")

def export_file_as_yaml(output_file, config, objects = None, pages = None):
    """
    Save the objects to a YAML file.

    :param output_file: The name of the file to save to.
    :param config: The configuration of the drawing (dict).
    :param objects: The objects to save (dict).
    :param pages: The pages to save (dict).

    Pages and Drawable objects need to be converted to dictionaries before
    saving them to a file using their to_dict() method.
    """

    state = { 'config': config }
    if pages:
        state['pages']   = pages
    if objects:
        state['objects'] = objects
    try:
        with open(output_file, 'w', encoding = 'utf-8') as f:
            yaml.dump(state, f)
        log.debug("Saved drawing to %s", output_file)
        return True
    except OSError as e:
        log.warning("Error saving file due to a file I/O error: %s", e)
        return False
    except yaml.YAMLError as e:
        log.warning("Error saving file because: %s", e)
        return False


# ------------------- handling of the native format -------------------

def save_file_as_sdrw(output_file, config, objects = None, pages = None):
    """
    Save the objects to a file in native format.

    :param output_file: The name of the file to save to.
    :param config: The configuration of the drawing (dict).
    :param objects: The objects to save (dict).
    :param pages: The pages to save (dict).

    Pages and Drawable objects need to be converted to dictionaries before
    saving them to a file using their to_dict() method.
    """
    # objects are here for backwards compatibility only
    state = { 'config': config }
    if pages:
        state['pages']   = pages
    if objects:
        state['objects'] = objects
    try:
        with open(output_file, 'wb') as f:
            #yaml.dump(state, f)
            pickle.dump(state, f)
        log.debug("Saved drawing to %s", output_file)
        return True
    except OSError as e:
        log.warning("Error saving file due to a file I/O error: %s", e)
        return False
    except pickle.PicklingError as e:
        log.warning("Error saving file, an object could not be pickled: %s", e)
        return False

def read_file_as_sdrw(input_file):
    """
    Read the objects from a file in native format.

    :param input_file: The name of the file to read from.
    """
    if not path.exists(input_file):
        log.warning("No saved drawing found at %s", input_file)
        return None, None, None

    log.debug("READING file as sdrw: %s", input_file)

    config, objects, pages = None, None, None

    try:
        with open(input_file, "rb") as file:
            state = pickle.load(file)
            if "objects" in state:
                log.debug("found objects in savefile")
                objects = [ Drawable.from_dict(d) for d in state['objects'] ] or [ ]
            if "pages" in state:
                log.debug("found pages in savefile")
                pages = state['pages']
                for p in pages:
                    # this is for compatibility; newer drawings are saved
                    # with a "layers" key which is then processed by the
                    # page import function - best if page takes care of it
                    if "objects" in p:
                        p['objects'] = [ Drawable.from_dict(d) for d in p['objects'] ] or [ ]

            config = state['config']
    except OSError as e:
        log.warning("Error saving file due to a file I/O error: %s", e)
        return None, None, None
    except pickle.PicklingError as e:
        log.warning("Error saving file because an object could not be pickled: %s", e)
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

# pylint: disable=line-too-long

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

    def __init__(self, bus, state):
        # singleton pattern
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self.__state = state
            self.__bus = bus
            self.make_actions_dictionary()
            self.make_default_keybindings()
            bus.on("key_press_event", self.process_key_event)

    def dispatch_action(self, action_name, **kwargs):
        """
        Dispatches an action by name.
        """
        #print("dispatch_action", action_name)
        if not action_name in self.__actions:
            log.warning("action %s not found in actions", action_name)
            return

        action = self.__actions[action_name].get('action') or self.__bus.emit

        if not callable(action):
            raise ValueError(f"Action is not callable: {action_name}")
        try:
            if 'args' in self.__actions[action_name]:
                args = self.__actions[action_name]['args']
                action(*args)
            else:
                action(**kwargs)
        except KeyError as ke:
            log.error("Action name %s not found in actions: %s", action_name, ke)
        except TypeError as te:
            log.error("Type error in calling action %s: %s", action_name, te)
        except Exception as e: # pylint: disable=broad-except
            exc_type, exc_value, exc_traceback = exc_info()
            traceback.print_tb(exc_traceback)
            log.error("Error while dispatching action %s %s",
                      action_name, e)
            log.warning("Exception type: %s", exc_type)
            log.warning("Traceback:")
            log.warning("Exception value: %s", exc_value)

    def dispatch_key_event(self, key_event, keyname, mode):
        """
        Dispatches an action by key event.
        """
        #print("dispatch_key_event", key_event, mode)

        if keyname in self.__keybindings:
            key_event = keyname

        if not key_event in self.__keybindings:
            return

        action_name = self.__keybindings[key_event]

        if not action_name in self.__actions:
            log.warning("action %s not found in actions",
                        action_name)
            return

        # check whether action allowed in the current mode
        if 'modes' in self.__actions[action_name]:
            if not mode in self.__actions[action_name]['modes']:
                log.warning("action not allowed in this mode")
                return

        log.debug("keyevent %s dispatching action %s",
                  key_event, action_name)
        self.dispatch_action(action_name)

    def on_key_press(self, _, event):
        """
        This method is called when a key is pressed.
        """

        log.debug("key pressed %s state %s",
                  event.keyval, event.state)
        self.__bus.emit("key_press_event", True, event)

    def process_key_event(self, event):
        """Process the key event and send message to the bus"""
        state = self.__state

        keyname = Gdk.keyval_name(event.keyval)
        char    = chr(Gdk.keyval_to_unicode(event.keyval))
        ctrl    = event.state & Gdk.ModifierType.CONTROL_MASK
        shift   = event.state & Gdk.ModifierType.SHIFT_MASK
        alt_l   = event.state & Gdk.ModifierType.MOD1_MASK # pylint: disable=no-member

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
        log.debug("keyname %s char %s ctrl %s shift %s alt_l %s keyfull %s",
                  keyname, char, ctrl, shift, alt_l, keyfull)

        # first, check whether there is a current object being worked on
        # and whether this object is a text object. In that case, we only
        # call the ctrl keybindings and pass the rest to the text object.
        cobj = state.current_obj()

        if cobj and cobj.type == "text" and not keyname == "Escape":
            log.debug("updating text input, keyname=%s char=%s keyfull=%s mode=%s",
                      keyname, char, keyfull, mode)
            cobj.update_by_key(keyfull, char)
            self.__bus.emit("queue_draw")
            return

        # otherwise, we dispatch the key event
        self.dispatch_key_event(keyfull, 'keyname:' + str(keyname), mode)
        self.__bus.emit("queue_draw")

    def make_actions_dictionary(self):
        """
        This dictionary maps key events to actions.
        """
        self.__actions = {
            'mode_draw':             {'args': [ "mode_set", False, "draw"]},
            'mode_rectangle':        {'args': [ "mode_set", False, "rectangle"]},
            'mode_circle':           {'args': [ "mode_set", False, "circle"]},
            'mode_move':             {'args': [ "mode_set", False, "move"]},
            'mode_text':             {'args': [ "mode_set", False, "text"]},
            'mode_select':           {'args': [ "mode_set", False, "select"]},
            'mode_eraser':           {'args': [ "mode_set", False, "eraser"]},
            'mode_shape':            {'args': [ "mode_set", False, "shape"]},
            'mode_zoom':             {'args': [ "mode_set", False, "zoom"]},
            'mode_segment':          {'args': [ "mode_set", False, "segment"]},
            'mode_colorpicker':      {'args': [ "mode_set", False, "colorpicker"]},

            'escape':                {'args': ["escape"]},
            'toggle_grouping':       {'args': ["toggle_grouping"]},
            'toggle_outline':        {'args': ["toggle_outline"]},

            'cycle_bg_transparency': {'args': ["cycle_bg_transparency"]},
            'toggle_wiglets':        {'args': ["toggle_wiglets"]},
            'toggle_crosslines':     {'args': ["toggle_crosslines"]},
            'toggle_grid':           {'args': ["toggle_grid"]},
            'switch_pens':           {'args': ["switch_pens"]},
            'apply_pen_to_bg':       {'args': ["apply_pen_to_bg"], 'modes': ["move"]},

            'clear_page':            {'args': ["clear_page"]},

            'zoom_reset':            {'args': ["zoom_reset"]},
            'zoom_in':               {'args': ["zoom_in"]},
            'zoom_out':              {'args': ["zoom_out"]},

            # dialogs and app events
            'app_exit':              {'args': ["app_exit"]},
            'show_help_dialog':      {'args': ["show_help_dialog"]},
            "export_drawing":        {'args': ["export_drawing"]},
            "save_drawing_as":       {'args': ["save_drawing_as"]},
            "select_color":          {'args': ["select_color"]},
            "select_color_bg":       {'args': ["select_color_bg"]},
            "select_font":           {'args': ["select_font"]},
            "import_image":          {'args': ["import_image"]},
            "open_drawing":          {'args': ["open_drawing"]},

            'copy_content':          {'args': ["copy_content"]},
            'cut_content':           {'args': ["cut_content"]},
            'paste_content':         {'args': ["paste_content"]},
            'duplicate_content':     {'args': ["duplicate_content"]},
            'screenshot':            {'args': ["screenshot"]},

            'selection_fill':        {'args': [ "selection_fill" ], 'modes': ["move"]},

            'transmute_to_shape':    {'args': [ "transmute_selection", True, "shape" ]},
            'transmute_to_draw':     {'args': [ "transmute_selection", True, "draw" ]},

            'move_up_10':            {'args': [ "move_selection", True, 0, -10],   'modes': ["move"]},
            'move_up_1':             {'args': [ "move_selection", True, 0, -1],    'modes': ["move"]},
            'move_up_100':           {'args': [ "move_selection", True, 0, -100],  'modes': ["move"]},
            'move_down_10':          {'args': [ "move_selection", True, 0, 10],    'modes': ["move"]},
            'move_down_1':           {'args': [ "move_selection", True, 0, 1],     'modes': ["move"]},
            'move_down_100':         {'args': [ "move_selection", True, 0, 100],   'modes': ["move"]},
            'move_left_10':          {'args': [ "move_selection", True, -10, 0],   'modes': ["move"]},
            'move_left_1':           {'args': [ "move_selection", True, -1, 0],    'modes': ["move"]},
            'move_left_100':         {'args': [ "move_selection", True, -100, 0],  'modes': ["move"]},
            'move_right_10':         {'args': [ "move_selection", True, 10, 0],    'modes': ["move"]},
            'move_right_1':          {'args': [ "move_selection", True, 1, 0],     'modes': ["move"]},
            'move_right_100':        {'args': [ "move_selection", True, 100, 0],   'modes': ["move"]},

            'rotate_selection_ccw_10': {'args': [ "rotate_selection", True, 10],  'modes': ["move"]},
            'rotate_selection_ccw_1':  {'args': [ "rotate_selection", True, 1],   'modes': ["move"]},
            'rotate_selection_ccw_90': {'args': [ "rotate_selection", True, 90],  'modes': ["move"]},
            'rotate_selection_cw_10':  {'args': [ "rotate_selection", True, -10], 'modes': ["move"]},
            'rotate_selection_cw_1':   {'args': [ "rotate_selection", True, -1],  'modes': ["move"]},
            'rotate_selection_cw_90':  {'args': [ "rotate_selection", True, -90], 'modes': ["move"]},

            'zmove_selection_top':     {'args': [ "selection_zmove", True, "top" ],    'modes': ["move"]},
            'zmove_selection_bottom':  {'args': [ "selection_zmove", True, "bottom" ], 'modes': ["move"]},
            'zmove_selection_raise':   {'args': [ "selection_zmove", True, "raise" ],  'modes': ["move"]},
            'zmove_selection_lower':   {'args': [ "selection_zmove", True, "lower" ],  'modes': ["move"]},

            'flush_left':            {'args': [ "flush_selection", True, "left" ],   'modes': ["move"]},
            'flush_right':           {'args': [ "flush_selection", True, "right" ],  'modes': ["move"]},
            'flush_top':             {'args': [ "flush_selection", True, "top" ],    'modes': ["move"]},
            'flush_bottom':          {'args': [ "flush_selection", True, "bottom" ], 'modes': ["move"]},

            'set_color_white':       {'args': [ "set_color", False, COLORS["white"]]},
            'set_color_black':       {'args': [ "set_color", False, COLORS["black"]]},
            'set_color_red':         {'args': [ "set_color", False, COLORS["red"]]},
            'set_color_green':       {'args': [ "set_color", False, COLORS["green"]]},
            'set_color_blue':        {'args': [ "set_color", False, COLORS["blue"]]},
            'set_color_yellow':      {'args': [ "set_color", False, COLORS["yellow"]]},
            'set_color_cyan':        {'args': [ "set_color", False, COLORS["cyan"]]},
            'set_color_magenta':     {'args': [ "set_color", False, COLORS["magenta"]]},
            'set_color_purple':      {'args': [ "set_color", False, COLORS["purple"]]},
            'set_color_grey':        {'args': [ "set_color", False, COLORS["grey"]]},

            'set_brush_rounded':     {'args': [ "set_brush", True, "rounded"] },
            'set_brush_marker':      {'args': [ "set_brush", True, "marker"] },
            'set_brush_slanted':     {'args': [ "set_brush", True, "slanted"] },
            'set_brush_pencil':      {'args': [ "set_brush", True, "pencil"] },
            'set_brush_tapered':     {'args': [ "set_brush", True, "tapered"] },
            'set_brush_simple':      {'args': [ "set_brush", True, "simple"] },

            'stroke_increase':       {'args': [ "stroke_change", True, 1]},
            'stroke_decrease':       {'args': [ "stroke_change", True, -1]},

            # selections and moving objects
            'select_next_object':     {'args': [ "set_selection", True, "next_object" ],     'modes': ["move"]},
            'select_previous_object': {'args': [ "set_selection", True, "previous_object" ], 'modes': ["move"]},
            'select_all':             {'args': [ "set_selection", True, "all" ]},
            'select_reverse':         {'args': [ "set_selection", True, "reverse" ]},

            'selection_clip':         {'args': [ "selection_clip"   ], 'modes': ["move"]},
            'selection_unclip':       {'args': [ "selection_unclip" ], 'modes': ["move"]},
            'selection_group':        {'args': [ "selection_group"  ], 'modes': ["move"]},
            'selection_ungroup':      {'args': [ "selection_ungroup"], 'modes': ["move"]},
            'selection_delete':       {'args': [ "selection_delete" ], 'modes': ["move"]},
            'redo':                   {'args': [ "history_redo" ]},
            'undo':                   {'args': [ "history_undo" ]},
            'next_page':              {'args': [ "next_page" ]},
            'prev_page':              {'args': [ "prev_page" ]},
            'insert_page':            {'args': [ "insert_page" ]},
            'delete_page':            {'args': [ "delete_page" ]},
            'next_layer':             {'args': [ "next_layer" ]},
            'prev_layer':             {'args': [ "prev_layer" ]},
            'delete_layer':           {'args': [ "delete_layer" ]},
            'apply_pen_to_selection': {'args': [ "selection_apply_pen" ],    'modes': ["move"]},
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
            'z':                    "mode_zoom",
            'Shift-s':              "mode_segment",
            'i':                    "mode_colorpicker",
            'space':                "mode_move",

            'h':                    "show_help_dialog",
            'F1':                   "show_help_dialog",
            'question':             "show_help_dialog",
            'Shift-question':       "show_help_dialog",
            'Escape':               "escape",
            'Ctrl-l':               "clear_page",
            'Ctrl-b':               "cycle_bg_transparency",
            'x':                    "app_exit",
            'q':                    "app_exit",
            'Ctrl-q':               "app_exit",
            'l':                    "clear_page",
            'o':                    "toggle_outline",
            'w':                    "toggle_wiglets",
            'k':                    "toggle_crosslines",
            'Ctrl-g':               "toggle_grid",
            'Alt-s':                "transmute_to_shape",
            'Alt-d':                "transmute_to_draw",
            'f':                    "selection_fill",

            'keyname:equal':        "zoom_reset",
            'plus':                 "zoom_in",
            'minus':                "zoom_out",

            'Ctrl-Shift-g':         "toggle_grouping",

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

            'Alt-Shift-Left':             "flush_left",
            'Alt-Shift-Right':            "flush_right",
            'Alt-Shift-Up':               "flush_top",
            'Alt-Shift-Down':             "flush_bottom",

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
            '5':                    "set_brush_tapered",
            '6':                    "set_brush_simple",

            'Shift-p':              "prev_page",
            'Shift-n':              "next_page",
            'Shift-i':              "insert_page",
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
            'Ctrl-p':               "switch_pens",
            'Ctrl-o':               "open_drawing",


            'Tab':                  "select_next_object",
            'Shift-ISO_Left_Tab':   "select_previous_object",
            'g':                    "selection_group",
            'u':                    "selection_ungroup",
            'Shift-c':              "selection_clip",
            'Shift-u':              "selection_unclip",
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
            'Ctrl-d':               "duplicate_content",
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

    def __init__(self, bus, state):
        if not hasattr(self, '_initialized'):
            self.__bus = bus
            self._initialized = True
            self.__state = state
            self.__context_menu = None
            self.__object_menu = None
            self.__bus.on("right_mouse_click", self.on_right_mouse_click)

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

    def on_menu_item_activated(self, widget, params):
        """Callback for when a menu item is activated."""
        log.debug("Menu item activated: %s from %s", params, widget)

        self.__bus.emit_mult("queue_draw")
        action = params[0]
        args   = params[1:]
        self.__bus.emit(action, *args)


    def context_menu(self):
        """Build the main context menu"""
        # general menu for everything
        ## build only once
        if self.__context_menu:
            return self.__context_menu

        # pylint: disable=line-too-long
        cb = self.on_menu_item_activated
        menu_items = [
                { "label": "Toggle UI    [w]",                "callback": cb, "action": [ "toggle_wiglets" ] },
                { "separator": True },
                { "label": "Move         [m]",                "callback": cb, "action": [ "mode_set", False, "move" ] },
                { "label": "Pencil       [d]",                "callback": cb, "action": [ "mode_set", False, "draw" ] },
                { "label": "Shape        [s]",                "callback": cb, "action": [ "mode_set", False, "shape" ] },
                { "label": "Text         [t]",                "callback": cb, "action": [ "mode_set", False, "text" ] },
                { "label": "Rectangle    [r]",                "callback": cb, "action": [ "mode_set", False, "box" ] },
                { "label": "Circle       [c]",                "callback": cb, "action": [ "mode_set", False, "circle" ] },
                { "label": "Eraser       [e]",                "callback": cb, "action": [ "mode_set", False, "eraser" ] },
                { "label": "Color picker [i]",                "callback": cb, "action": [ "mode_set", False, "colorpicker" ] },
                { "separator": True },
                { "label": "Select all    (Ctrl-a)",          "callback": cb, "action": [ "set_selection", True, "all" ] },
                { "label": "Paste         (Ctrl-v)",          "callback": cb, "action": [ "paste_content" ] },
                { "label": "Clear drawing (Ctrl-l)",          "callback": cb, "action": [ "clear_page" ] },
                { "separator": True },
                { "label": "Next Page     (Shift-n)",         "callback": cb, "action": [ "next_page" ] },
                { "label": "Prev Page     (Shift-p)",         "callback": cb, "action": [ "prev_page" ] },
                { "label": "Delete Page   (Shift-d)",         "callback": cb, "action": [ "delete_page" ] },
                { "label": "Next Layer    (Ctrl-Shift-n)",    "callback": cb, "action": [ "next_layer" ] },
                { "label": "Prev Layer    (Ctrl-Shift-p)",    "callback": cb, "action": [ "prev_layer" ] },
                { "label": "Delete Layer  (Ctrl-Shift-d)",    "callback": cb, "action": [ "delete_layer" ] },
                { "separator": True },
                { "label": "Bg transparency (Ctrl-b)",        "callback": cb, "action": [ "cycle_bg_transparency" ] },
                { "label": "Toggle outline       [o]",        "callback": cb, "action": [ "toggle_outline" ] },
                { "separator": True },
                { "label": "Color           (Ctrl-k)",        "callback": cb, "action": [ "select_color" ] },
                { "label": "Bg Color  (Shift-Ctrl-k)",        "callback": cb, "action": [ "select_color_bg" ] },
                { "label": "Font            (Ctrl-f)",        "callback": cb, "action": [ "select_font" ] },
                { "separator": True },
                { "label": "Open drawing    (Ctrl-o)",        "callback": cb, "action": [ "open_drawing" ] },
                { "label": "Image from file (Ctrl-i)",        "callback": cb, "action": [ "import_image" ] },
                { "label": "Screenshot      (Ctrl-Shift-f)",  "callback": cb, "action": [ "screenshot" ] },
                { "label": "Save as         (Ctrl-s)",        "callback": cb, "action": [ "save_drawing_as" ] },
                { "label": "Export          (Ctrl-e)",        "callback": cb, "action": [ "export_drawing" ] },
                { "label": "Help            [F1]",            "callback": cb, "action": [ "show_help_dialog" ] },
                { "label": "Quit            (Ctrl-q)",        "callback": cb, "action": [ "app_exit" ] },
        ]

        self.__context_menu = self.build_menu(menu_items)
        return self.__context_menu

    def object_menu(self, objects):
        """Build the object context menu"""
        # when right-clicking on an object
        cb = self.on_menu_item_activated

        # pylint: disable=line-too-long
        menu_items = [
                { "label": "Copy (Ctrl-c)",                  "callback": cb, "action": [ "copy_content" ] },
                { "label": "Cut (Ctrl-x)",                   "callback": cb, "action": [ "cut_content" ] },
                { "separator": True },
                { "label": "Delete (|Del|)",                 "callback": cb, "action": [ "selection_delete" ] },
                { "label": "Group (g)",                      "callback": cb, "action": [ "selection_group" ] },
                { "label": "Ungroup (u)",                    "callback": cb, "action": [ "selection_ungroup" ] },
                { "label": "Export (Ctrl-e)",                "callback": cb, "action": [ "export_drawing" ] },
                { "separator": True },
                { "label": "Move to top (Alt-Page_Up)",      "callback": cb, "action": [ "selection_zmove", False, "top" ] },
                { "label": "Raise (Alt-Up)",                 "callback": cb, "action": [ "selection_zmove", False, "raise" ] },
                { "label": "Lower (Alt-Down)",               "callback": cb, "action": [ "selection_zmove", False, "lower" ] },
                { "label": "Move to bottom (Alt-Page_Down)", "callback": cb, "action": [ "selection_zmove", False, "bottom" ] },
                { "separator": True },
                { "label": "To shape   (Alt-s)",             "callback": cb, "action": [ "transmute_selection", True, "shape" ] },
                { "label": "To drawing (Alt-d)",             "callback": cb, "action": [ "transmute_selection", True, "draw" ] },
                { "label": "Fill toggle (f)",                "callback": cb, "action": [ "selection_fill" ] },
                { "separator": True },
                { "label": "Color (Ctrl-k)",                 "callback": cb, "action": [ "select_color" ] },
                { "label": "Font (Ctrl-f)",                  "callback": cb, "action": [ "select_font" ] },
                { "label": "Help [F1]",                      "callback": cb, "action": [ "show_help_dialog" ] },
                { "label": "Quit (Ctrl-q)",                  "callback": cb, "action": [ "app_exit" ] },
        ]

        # if there is only one object, remove the group menu item
        if len(objects) == 1:
            log.debug("only one object")
            menu_items = [m for m in menu_items if "action" not in m or "selection_group" not in m["action"]]

        group_found = [o for o in objects if o.type == "group"]
        if not group_found:
            log.debug("no group found")
            menu_items = [m for m in menu_items if "action" not in m or "selection_ungroup" not in m["action"]]

        self.__object_menu = self.build_menu(menu_items)

        return self.__object_menu

    def on_right_mouse_click(self, ev):
        """Catch the right mouse click and display the menu"""

        event     = ev.event
        hover_obj = ev.hover()

        if hover_obj:
            self.__state.mode("move")

            if not self.__state.selection().contains(hover_obj):
                self.__state.selection().set([ hover_obj ])

            sel_objects = self.__state.selection().objects
            self.object_menu(sel_objects).popup(None, None,
                                                         None, None,
                                                         event.button, event.time)
        else:
            self.context_menu().popup(None, None,
                                               None, None,
                                               event.button, event.time)
        self.__bus.emit("queue_draw")
        return True
"""
Wiglets are small dialog-like objects that can be drawn on the canvas.
They are used to provide interactive controls for changing drawing properties
such as line width, color, and transparency.
"""

#log.setLevel(logging.INFO)

class Wiglet:
    """drawable dialog-like objects on the canvas"""
    def __init__(self, mytype, coords):
        self.wiglet_type   = mytype
        self.coords = coords

    def update_size(self, width, height): # pylint: disable=unused-argument
        """ignore update the size of the widget"""
        return False

    def draw(self, cr, state): # pylint: disable=unused-argument
        """do not draw the widget"""

    def on_click(self, ev): # pylint: disable=unused-argument
        """ignore the click event"""
        return False

    def on_release(self, ev): # pylint: disable=unused-argument
        """ignore the release event"""
        return False

    def on_move(self, ev): # pylint: disable=unused-argument
        """ignore update on mouse move"""
        return False

class WigletResizeRotate(Wiglet):
    """Catch resize events and update the size of the object."""

    def __init__(self, bus, state):
        super().__init__("resize", None)
        self.__bus = bus
        self.__cmd = None
        self.__state = state # pylint: disable=unused-private-member

        bus.on("left_mouse_click", self.on_click)

    def on_click(self, ev):
        log.debug("resize widget clicked")
        if ev.mode() != "move" or ev.alt():
            log.debug("resizing - wrong modifiers")
            return False

        corner_obj, corner = ev.corner()

        if not corner_obj or not corner_obj.bbox() or ev.double():
            log.debug("widget resizing wrong event hover: %s, corner: %s, double: %s",
                    ev.hover, ev.corner, ev.double)
            return False

        log.debug("widget resizing object. corner: %s", corner_obj)

        if ev.ctrl() and ev.shift():
            log.debug("rotating with both shift and ctrl")
            self.__cmd = RotateCommand(corner_obj, origin = ev.pos(),
                                             corner = corner)
        else:
            self.__cmd = ResizeCommand(corner_obj, origin = ev.pos(),
                                   corner = corner,
                                   proportional = ev.ctrl())

        self.__bus.on("mouse_move", self.on_move)
        self.__bus.on("mouse_release", self.on_release)
        self.__bus.emit_once("set_selection", [ corner_obj ])
        self.__bus.emit_once("cursor_set", corner)
        self.__bus.emit("queue_draw")

        return True

    def on_move(self, ev):
        if not self.__cmd:
            return False

        self.__cmd.event_update(*ev.pos())
        self.__bus.emit("queue_draw")
        return True

    def on_release(self, ev):
        if not self.__cmd:
            return False
        self.__cmd.event_update(*ev.pos())
        self.__cmd.event_finish()
        self.__bus.emit_once("history_append", self.__cmd)
        self.__bus.emit_once("cursor_revert")
        self.__cmd = None
        self.__bus.off("mouse_move", self.on_move)
        self.__bus.off("mouse_release", self.on_release)
        self.__bus.emit("queue_draw")
        return True

class WigletHover(Wiglet):
    """Change cursor when moving over objects."""

    def __init__(self, bus, state):
        super().__init__("hover", None)

        self.__bus   = bus
        self.__state = state

        bus.on("mouse_move", self.on_move, priority = -9)

    def on_move(self, ev):
        """When cursor hovers over an object"""

        if not ev.mode() == "move":
            return False

        corner_obj, corner = ev.corner()
        object_underneath  = ev.hover()

        if corner_obj and corner_obj.bbox():
            self.__bus.emit_once("cursor_set", corner)
        elif object_underneath:
            self.__bus.emit_once("cursor_set", "moving")
            self.__state.hover_obj(object_underneath)
        else:
            self.__bus.emit_once("cursor_revert")
            self.__state.hover_obj_clear()


        self.__bus.emit("queue_draw")
        return True

class WigletMove(Wiglet):
    """Catch moving events and update the position of the object."""

    def __init__(self, bus, state):
        super().__init__("move", None)
        self.__bus = bus
        self.__cmd = None
        self.__state = state
        self.__obj = None

        bus.on("left_mouse_click", self.on_click)

    def on_click(self, ev):
        """Start moving object"""
        obj = ev.hover()

        if ev.mode() != "move" or ev.alt() or ev.ctrl():
            log.debug("wrong modifiers")
            return False

        if not obj or ev.corner()[0] or ev.double():
            log.debug("widget moving wrong event")
            return False

        # first, update the current selection
        selection = self.__state.selection()

        if ev.shift():
            selection.add(obj)
        if not selection.contains(obj):
            selection.set([obj])

        self.__bus.on("cancel_left_mouse_single_click", self.cancel)
        self.__bus.on("mouse_move", self.on_move)
        self.__bus.on("mouse_release", self.on_release)

        self.__obj = selection.copy()
        log.debug("moving object: %s", self.__obj)
        self.__cmd = MoveCommand(self.__obj, ev.pos())
        self.__bus.emit_once("cursor_set", "grabbing")
        self.__bus.emit("queue_draw")

        return True

    def on_move(self, ev):
        """Update moving object"""
        if not self.__cmd:
            return False

        self.__cmd.event_update(*ev.pos())
        self.__bus.emit("queue_draw")
        return True

    def cancel(self, ev): # pylint: disable=unused-argument
        """Cancel the move operation"""
        if not self.__cmd:
            return False

        self.__bus.off("cancel_left_mouse_single_click", self.cancel)
        self.__bus.off("mouse_move", self.on_move)
        self.__bus.off("mouse_release", self.on_release)

        self.__cmd = None
        self.__obj = None

        return True

    def on_release(self, ev):
        """Finish moving object"""
        if not self.__cmd:
            return False

        self.__bus.off("cancel_left_mouse_single_click", self.cancel)
        self.__bus.off("mouse_move", self.on_move)
        self.__bus.off("mouse_release", self.on_release)

        _, height = self.__state.get_win_size()
        cmd = self.__cmd
        cmd.event_update(*ev.pos())
        cmd.event_finish()

        obj = self.__obj
        page = self.__state.page()

        if ev.event.x < 10 and ev.event.y > height - 10:
            # command group because these are two commands: first move,
            # then remove
            cmd = CommandGroup([ cmd,
                                 RemoveCommand(obj.objects,
                                               page.layer().objects()) ])
            self.__bus.emit_once("history_append", cmd)
            self.__bus.emit_once("set_selection", "nothing")
        else:
            self.__bus.emit_once("history_append", cmd)

        self.__bus.emit_once("cursor_revert")
        self.__cmd = None
        self.__obj = None
        self.__bus.emit("queue_draw")
        return True

class WigletSelectionTool(Wiglet):
    """Draw the selection tool when activated."""

    def __init__(self, bus, state):
        super().__init__("selection_tool", None)

        self.__selection_tool = None
        self.__bus = bus
        self.__state   = state
        bus.on("left_mouse_click", self.on_click, priority = -1)

    def draw(self, cr, state):
        """draw the widget"""
        if not self.__selection_tool:
            return
        self.__selection_tool.draw(cr, state)

    def on_click(self, ev):
        """handle the click event"""
        log.debug("receiving call")
        # ev.shift() means "add to current selection"
        if ev.mode() != "move" or ev.alt() or ev.ctrl():
            log.debug("wrong modifiers")
            return False

        if ev.hover() or ev.corner()[0] or ev.double():
            log.debug("wrong event; hover=%s corner=%s double=%s",
                      ev.hover(), ev.corner(), ev.double())
            return False

        log.debug("taking the call")

        self.__bus.on("mouse_move",       self.on_move)
        self.__bus.on("mouse_release",   self.on_release)
        self.__bus.on("obj_draw", self.draw)
        self.__bus.emit_once("set_selection", "nothing")

        x, y, = ev.x, ev.y
        self.coords = (x, y)
        self.__selection_tool = SelectionTool([ (x, y), (x + 1, y + 1) ])
        return True

    def on_move(self, ev):
        """update on mouse move"""
        if not self.__selection_tool:
            return False

        x, y = ev.x, ev.y
        obj = self.__selection_tool
        obj.update(x, y, ev.pressure())
        self.__bus.emit("queue_draw")
        return True

    def on_release(self, ev):
        """handle the release event"""
        if not self.__selection_tool:
            return False

        page = self.__state.page()
        objects = self.__selection_tool.objects_in_selection(page.layer().objects())

        if len(objects) > 0:
            self.__bus.emit_once("set_selection", objects)
        else:
            self.__bus.emit_once("set_selection", "nothing")

        self.__bus.off("mouse_move",       self.on_move)
        self.__bus.off("mouse_release",   self.on_release)
        self.__bus.off("obj_draw", self.draw)
        self.__bus.emit("queue_draw")

        self.__selection_tool = None
        return True

class WigletPan(Wiglet):
    """Paning the page, i.e. tranposing it with alt-click"""
    def __init__(self, bus):
        super().__init__("pan", None)

        self.__bus    = bus
        self.__origin = None
        bus.on("left_mouse_click", self.on_click, priority = 9)

    def on_click(self, ev):
        """Start paning"""
        if ev.shift() or ev.ctrl() or not ev.alt():
            return False
        self.__origin = (ev.event.x, ev.event.y)
        self.__bus.on("mouse_move",     self.on_move, priority = 99)
        self.__bus.on("mouse_release",  self.on_release, priority = 99)
        return True

    def on_move(self, ev):
        """Update paning"""

        if not self.__origin:
            return False

        dx, dy = ev.event.x - self.__origin[0], ev.event.y - self.__origin[1]
        self.__bus.emit_once("page_translate", (dx, dy))
        self.__origin = (ev.event.x, ev.event.y)
        self.__bus.emit("force_redraw")
        return True

    def on_release(self, ev):
        """Handle mouse release event"""
        self.__bus.off("mouse_move",     self.on_move)
        self.__bus.off("mouse_release",  self.on_release)

        if not self.__origin:
            log.warning("no origin")
            return False

        self.__origin = None
        return True

class WigletEditText(Wiglet):
    """Create or edit text objects"""
    def __init__(self, bus, state):
        super().__init__("pan", None)

        self.__bus   = bus
        self.__state = state
        self.__obj   = None
        self.__active = False
        self.__edit_existing = False
        bus.on("left_mouse_click",  self.on_click, priority = 99)
        bus.on("left_mouse_double_click",
               self.on_double_click, priority = 9)

        self.__listeners = {
            "mouse_move": { "listener":        self.on_move, "priority": 99},
            "mouse_release": { "listener":     self.on_release, "priority": 99},
            "mode_set": { "listener":          self.finish_text_input, "    priority": 99},
            "finish_text_input": { "listener": self.finish_text_input, "priority": 99},
            "escape": { "listener":            self.finish_text_input, "priority": 99},
            }

    def start_listening(self):
        """Start listening to events"""
        bus_listeners_on(self.__bus, self.__listeners)

    def stop_listening(self):
        """Stop listening to events"""
        bus_listeners_off(self.__bus, self.__listeners)

    def on_double_click(self, ev):
        """Double click on text launches text editing"""

        if self.__active: # currently editing
            self.__bus.emit("finish_text_input")
            return True

        if ev.shift() or ev.ctrl() or ev.alt():
            return False

        obj = ev.hover()
        if not (obj and obj.type == "text"):
            return False

        self.__edit_existing = obj.to_string()
        self.__obj = obj
        self.__active = True
        self.__state.current_obj(obj)
        self.__obj.move_caret("End")
        self.__bus.emit("queue_draw")
        self.start_listening()
        return True

    def on_click(self, ev):
        """Start typing text"""

        if self.__active: # currently editing
            self.__bus.emit("finish_text_input")
            return False

        mode = self.__state.mode()
        log.debug("mode %s", mode)

        if ev.shift() and not ev.ctrl() and mode != "move":
            mode = "text"

        if mode != "text":
            return False

        log.debug("Creating a new text object")
        self.__edit_existing = False

        obj = DrawableFactory.create_drawable(mode, pen = self.__state.pen(), ev=ev)

        if obj:
            self.__state.current_obj(obj)
            self.__active = True
            self.__obj = obj
        else:
            log.debug("No object created for mode %s", mode)
            return False

        self.start_listening()
        return True

    def on_release(self, ev):
        """Finish drawing object"""

        obj = self.__obj

        if not obj:
            return False

        self.__bus.emit("queue_draw")
        return True

    def finish_text_input(self, new_mode = False): #pylint: disable=unused-argument
        """Finish text input"""
        if not self.__active:
            return False

        log.debug("finishing text input")

        obj = self.__obj
        obj.show_caret(False)

        if self.__edit_existing:
            page = self.__state.page()

            if obj.strlen() > 0:
                # create a command to allow undo
                cmd = TextEditCommand(obj, self.__edit_existing, obj.to_string())
                self.__bus.emit("history_append", True, cmd)
            else:
                # remove the object
                cmd1 = TextEditCommand(obj, self.__edit_existing, obj.to_string())
                cmd2 = RemoveCommand([ obj ], page.layer().objects())
                self.__bus.emit("history_append", True, CommandGroup([ cmd1, cmd2 ]))
        elif obj.strlen() > 0:
            self.__bus.emit("add_object", True, obj)

        self.__state.current_obj_clear()
        self.__bus.emit_once("cursor_revert")
        self.__active = False
        self.__obj = None

        self.__bus.emit("queue_draw")
        self.stop_listening()
        return True

class WigletCreateSegments(Wiglet):
    """Create a segmented path"""

    def __init__(self, bus, state):
        super().__init__("pan", None)

        self.__bus   = bus
        self.__state = state
        self.__obj   = None
        bus.on("left_mouse_click", self.on_click,   priority = 90)
        bus.on("mode_set",         self.cancel,     priority = 99)
        bus.on("escape", self.cancel,   priority = 99)
        bus.on("left_mouse_double_click", self.on_finish,   priority = 99)
        bus.on("mouse_move",       self.on_move,    priority = 99)
        bus.on("obj_draw",         self.draw_obj,   priority = 99)

    def cancel(self, new_mode = None):
        """Cancel creating a segmented path"""
        mode = self.__state.mode()

        if new_mode is not None and self.__obj:
            self.__bus.emit("add_object", True, self.__obj)
            self.__state.selection().clear()

            self.__obj = None
            self.__bus.emit("queue_draw")
            return False

        if self.__obj:
            log.debug("WigletCreateSegments: cancel")
            self.__obj = None

        if mode != "segment":
            return False

        return True

    def on_move(self, ev):
        """Update drawing object"""
        obj = self.__obj

        if not obj:
            return False

        obj.path_pop()
        obj.update(ev.x, ev.y, ev.pressure())
        self.__bus.emit("queue_draw")
        return True

    def draw_obj(self, cr, state):
        """Draw the object currently being created"""
        if not self.__obj:
            return False

        self.__obj.draw(cr)
        return True

    def on_click(self, ev):
        """Start drawing"""

        if ev.ctrl() or ev.alt():
            return False

        mode = self.__state.mode()
        log.debug("segment on_click here")

        if mode != "segment" or ev.shift() or ev.ctrl():
            return False

        if self.__obj:
            self.__obj.path_pop()
            ## append twice, once the "actual" point, once the moving end
            self.__obj.path_append(ev.x, ev.y, 1)
            self.__obj.path_append(ev.x, ev.y, 1)
        else:
            obj = DrawableFactory.create_drawable("segmented_path", pen = self.__state.pen(), ev=ev)

            if obj:
                self.__obj = obj
                self.__obj.path_append(ev.x, ev.y, 1)

        self.__bus.emit("queue_draw")
        return True

    def on_finish(self, ev):
        """Finish drawing object"""

        obj = self.__obj

        if not obj:
            return False

        obj.path_append(ev.x, ev.y, 0)
        obj.finish()

        self.__bus.emit("add_object", True, obj)
        self.__state.selection().clear()

        self.__obj = None
        self.__bus.emit("queue_draw")
        return True

class WigletCreateGroup(Wiglet):
    """
    Create a groups of objects while drawing

    Basically, by default, objects are grouped automatically
    until you change the mode or press escape.
    """

    def __init__(self, bus, state, grouping = True):
        super().__init__("pan", None)

        self.__bus   = bus
        self.__state = state
        self.__group_obj   = None
        self.__added = False
        self.__grouping = grouping

        # the first command in the group. In case we need to abort before
        # the second element is added, we will undo this command and
        # instead add a single object to the page.
        self.__first_cmd = None

        # the logic is as follows: listen to all events. If we catch an
        # event which is not in the ignore list, we finish the group. This
        # ensures that weird stuff doesn't happen.
        self.__ignore_events = [ "queue_draw", "mouse_move",
                                 "history_append", "add_object",
                                 "draw", "obj_draw",
                                 "left_mouse_click",
                                 "cursor_pos_update",
                                 "cursor_revert", "cursor_set",
                                 "update_win_size",
                                 "mouse_release" ]

        bus.on("toggle_grouping",  self.toggle_grouping, priority = 0)

        if self.__grouping:
            self.start_grouping()

    def toggle_grouping(self):
        """Toggle automatic grouping of objects"""

        self.__grouping = not self.__grouping

        if self.__grouping:
            self.start_grouping()
        else:
            self.end_grouping()

        return True

    def start_grouping(self, mode = None): # pylint: disable=unused-argument
        """Start automatic grouping of objects"""

        if self.__group_obj:
            raise ValueError("Group object already exists")

        self.__group_obj = DrawableGroup()
        self.__added = False
        self.__bus.on("add_object", self.add_object, priority = 99)
        self.__bus.on("escape",     self.end_grouping, priority = 99)
        self.__bus.on("clear_page", self.end_grouping, priority = 200)
        self.__bus.on("mode_set",   self.end_grouping, priority = 200)
        self.__bus.on("*",          self.abort)
        self.__bus.off("toggle_grouping",  self.start_grouping)
        return True

    def abort(self, event, *args, **kwargs):
        """Abort grouping if event is not in the ignore list"""
        if event in self.__ignore_events:
            return False
        log.debug("event: {event} %s %s, aborting grouping", args, kwargs)
        self.end_grouping()
        return False

    def end_grouping(self, mode = None): # pylint: disable=unused-argument
        """End automatic grouping of objects"""

        if not self.__group_obj:
            log.warning("end_grouping: no group object")

        self.__bus.off("add_object", self.add_object)
        self.__bus.off("escape",     self.end_grouping)
        self.__bus.off("clear_page", self.end_grouping)
        self.__bus.off("mode_set",   self.end_grouping)
        self.__bus.off("*",          self.abort)

        n = self.__group_obj.length()

        if n == 1:
            page = self.__state.current_page()
            obj = self.__group_obj.objects[0]
            #cmd1 = RemoveCommand([ self.__group_obj ], page.layer().objects())
            self.__bus.emit("history_undo_cmd", True, self.__first_cmd)
            cmd2 = AddCommand([ obj ], page.layer().objects())
            #cmd = CommandGroup([ cmd1, cmd2 ])
            self.__bus.emit("history_append", True, cmd2)

        self.__group_obj = None

        if self.__grouping:
            self.start_grouping()

        return True

    def add_object(self, obj):
        """Add object to the group"""

        mode = self.__state.mode()

        if not mode == "draw":
            return False

        if not self.__group_obj:
            return False

        if obj.type != "path":
            log.warning("object of type %s cannot be added to automatic path group",
                        obj.type)
            return False

        if not self.__added:
            page = self.__state.current_page()
            cmd1 = AddCommand([ self.__group_obj ], page.layer().objects())
            cmd2 = AddToGroupCommand(self.__group_obj, obj)
            cmd  = CommandGroup([ cmd1, cmd2 ])
            self.__bus.emit("history_append", True, cmd)
            self.__first_cmd = cmd
            self.__added = True
        else:
            cmd = AddToGroupCommand(self.__group_obj, obj)
            self.__bus.emit("history_append", True, cmd)

        return True

class WigletCreateObject(Wiglet):
    """Create object when clicked"""

    def __init__(self, bus, state):
        super().__init__("pan", None)

        self.__bus   = bus
        self.__state = state
        self.__obj   = None
        bus.on("left_mouse_click", self.on_click,   priority = 0)

    def draw_obj(self, cr, state): # pylint: disable=unused-argument
        """Draw the object currently being created"""
        if not self.__obj:
            return False
        self.__obj.draw(cr)
        return True

    def on_click(self, ev):
        """Start drawing"""

        if ev.ctrl() or ev.alt():
            return False

        mode = self.__state.mode()

        if ev.shift() and not ev.ctrl():
            mode = "text"

        if mode not in [ "draw", "shape", "rectangle", "circle" ]:
            return False

        log.debug("WigletCreateObject: creating a new object at %s, %s pressure %s",
                int(ev.x), int(ev.y), int(ev.pressure() * 1000))
        obj = DrawableFactory.create_drawable(mode, pen = self.__state.pen(), ev=ev)

        if obj:
            self.__obj = obj
            self.__bus.on("mouse_move",       self.on_move,    priority = 99)
            self.__bus.on("mouse_release",    self.on_release, priority = 99)
            self.__bus.on("obj_draw",         self.draw_obj,   priority = 99)
        else:
            log.debug("No object created for mode %s", mode)
        return True

    def on_move(self, ev):
        """Update drawing object"""
        obj = self.__obj
        if not obj:
            return False
        obj.update(ev.x, ev.y, ev.pressure())
        self.__bus.emit("queue_draw")
        return True

    def on_release(self, ev):
        """Finish drawing object"""

        obj = self.__obj

        if not obj:
            return False

        if obj.type in [ "shape", "path" ]:
            log.debug("finishing path / shape")
            obj.path_append(ev.x, ev.y, 0)
            obj.finish()
            # remove paths that are too small
            if len(obj.coords) < 3:
                obj = None

        # remove objects that are too small
        if obj:
            bb = obj.bbox()
            if bb and obj.type in [ "rectangle", "box", "circle" ] and bb[2] == 0 and bb[3] == 0:
                obj = None

        if obj:
            self.__bus.emit("add_object", True, obj)

            if self.__obj.type == "text":
                raise ValueError("Text object should not be finished here")
            self.__state.selection().clear()

        self.__obj = None
        self.__bus.off("mouse_move",    self.on_move)
        self.__bus.off("mouse_release", self.on_release)
        self.__bus.off("obj_draw",      self.draw_obj)
        self.__bus.emit("queue_draw")
        return True

class WigletEraser(Wiglet):
    """Erase mode. Removes objects."""

    def __init__(self, bus, state):
        super().__init__("pan", None)

        self.__bus    = bus
        self.__state  = state
        self.__active = False
        bus.on("left_mouse_click", self.on_click, priority = 10)
        bus.on("mouse_move",       self.on_move, priority = 99)
        bus.on("mouse_release",   self.on_release, priority = 99)

    def on_click(self, ev):
        """Clicking above an object removes it"""

        if not self.__state.mode() == "eraser":
            return False

        self.__active = True
        self.__delete_hover(ev)
        return True

    def __delete_hover(self, ev):
        """Delete object underneath"""
        hover_obj = ev.hover()

        if not hover_obj:
            return

        self.__bus.emit_once("remove_objects", [ hover_obj ], clear_selection = True)
        self.__bus.emit_once("cursor_revert")
        self.__bus.emit("queue_draw")

    def on_move(self, ev):
        """Process move: if active, delete everything underneath"""

        if not self.__active:
            return False

        self.__delete_hover(ev)
        return True

    def on_release(self, ev):
        """Stop being active"""
        if not self.__active:
            return False

        self.__active = False
        return True

class WigletColorPicker(Wiglet):
    """Invisible wiglet that processes clicks in the color picker mode."""
    def __init__(self, bus, func_color, clipboard):
        super().__init__("colorpicker", None)

        self.__func_color = func_color
        self.__clipboard = clipboard
        bus.on("left_mouse_click", self.on_click, priority = 1)

    def on_click(self, ev):
        """handle the click event"""

        # only works in color picker mode
        if ev.mode() != "colorpicker":
            return False

        if ev.shift() or ev.alt() or ev.ctrl():
            return False

        color = get_color_under_cursor()
        self.__func_color(color)

        color_hex = rgb_to_hex(color)
        self.__clipboard.set_text(color_hex)
        #self.__state.queue_draw()
        return True

class WigletZoom(Wiglet):
    """Zoom in and out"""

    def __init__(self, bus):
        super().__init__("zoom", None)

        self.__bus   = bus
        self.__wsize = (100, 100)
        self.__start_pos = None
        self.__zoom_tool = None

        # listeners that are active when zooming
        self.__active_listeners = {
                "mouse_release": { "listener": self.on_release, "priority": 99},
                "mouse_move": { "listener": self.on_move, "priority": 99},
                "obj_draw": { "listener": self.draw},
                }

        listeners = {
            "zoom_reset": { "listener":  self.zoom_reset, "priority": 99},
            "zoom_in": { "listener":  self.zoom_in, "priority": 99},
            "zoom_out": { "listener":  self.zoom_out, "priority": 99},
            "update_win_size": { "listener": self.update_size},
            "left_mouse_click": { "listener": self.on_click, "priority": 1},
        }

        bus_listeners_on(bus, listeners)

    def draw(self, cr, state):
        """draw the widget"""
        if not self.__zoom_tool:
            return
        self.__zoom_tool.draw(cr, state)

    def on_click(self, ev):
        """handle the click event"""

        if ev.mode() != "zoom":
            return False
        x, y, = ev.x, ev.y
        self.__zoom_tool = SelectionTool([ (x, y), (x + 1, y + 1) ])

        bus_listeners_on(self.__bus, self.__active_listeners)
        log.debug("zooming in or out")
        return True

    def on_release(self, ev):
        """handle the release event"""
        bus_listeners_off(self.__bus, self.__active_listeners)

        bb = self.__zoom_tool.bbox()
        x, y, w, h = bb

        self.__bus.emit_once("page_zoom_reset")
        self.__bus.emit_once("page_translate", (-x, -y))

        z1 = self.__wsize[0] / w
        z2 = self.__wsize[1] / h
        zoom = min(z1, z2, 14)
        log.debug("zooming to %s", zoom)
        self.__bus.emit_once("page_zoom",
                            (0, 0), zoom)
        self.__bus.emit("force_redraw")
        return True

    def on_move(self, ev):
        """handle the move event"""

        if not self.__start_pos:
            self.__start_pos = ev.pos()
            return False

        x, y = ev.x, ev.y
        obj = self.__zoom_tool
        obj.update(x, y, ev.pressure())
        self.__bus.emit("queue_draw")
        return True

    def update_size(self, width, height):
        """update the size of the widget"""
        self.__wsize = (width, height)

    def zoom_reset(self):
        """Reset zoom to 100%"""

        self.__bus.emit_once("page_zoom_reset")
        self.__bus.emit("force_redraw")
        return True

    def zoom_out(self):
        """Zoom out"""

        pos = (self.__wsize[0]/2, self.__wsize[1]/2)
        self.__bus.emit_once("page_zoom", pos, 0.9)
        self.__bus.emit("force_redraw")
        return True

    def zoom_in(self):
        """Zoom in"""

        pos = (self.__wsize[0]/2, self.__wsize[1]/2)
        self.__bus.emit_once("page_zoom", pos, 1.1)
        self.__bus.emit("force_redraw")
        return True
"""Wiglets which constitute visible UI elements"""


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



class WigletTransparency(Wiglet):
    """Wiglet for changing the transparency."""
    def __init__(self, bus, state):
        super().__init__("transparency", None)

        self.__state = state
        self.__pen    = None
        self.__initial_transparency = None
        self.__active = False
        self.__bus    = bus
        log.debug("initial transparency: %s", self.__initial_transparency)

        bus.on("left_mouse_click", self.on_click)

    def draw(self, cr, state):
        """draw the widget"""
        if not self.__active:
            return

        log.debug("drawing transparency widget")
        cr.set_source_rgba(*self.__pen.color, self.__pen.transparency)
        draw_dot(cr, *self.coords, 50)

    def on_click(self, ev):
        """handle the click event"""
        # make sure we are in the correct mode
        if not ev.shift() or ev.alt() or not ev.ctrl():
            return False

        if ev.hover() or ev.corner()[0] or ev.double():
            return False

        self.__bus.on("mouse_move", self.on_move)
        self.__bus.on("mouse_release", self.on_release)
        self.__bus.on("draw", self.draw)

        self.coords = (ev.x_abs, ev.y_abs)

        self.__pen    = self.__state.pen()
        self.__initial_transparency = self.__pen.transparency

        self.__active = True
        return True

    def on_move(self, ev):
        """update on mouse move"""
        if not self.__active:
            return False
        x = ev.x_abs
        dx = x - self.coords[0]

        ## we want to change the transparency by 0.1 for every 20 pixels
        transparency = max(0, min(1, self.__initial_transparency + dx/500))
        self.__bus.emit_mult("set_transparency", transparency)

        return True

    def on_release(self, ev):
        """handle the release event"""

        if not self.__active:
            return False

        log.debug("got release event")
        self.__active = False
        self.__bus.off("mouse_move", self.on_move)
        self.__bus.off("mouse_release", self.on_release)
        self.__bus.off("draw", self.draw)
        return True

class WigletLineWidth(Wiglet):
    """
    Wiglet for changing the line width.
    directly operates on the pen of the object
    """
    def __init__(self, bus, state):
        super().__init__("line_width", None)

        if not state:
            raise ValueError("Need state object")
        self.__state = state
        self.__pen    = None
        self.__initial_width = None
        self.__active = False
        self.__bus = bus

        bus.on("left_mouse_click", self.on_click)

    def draw(self, cr, state):
        """draw the widget"""
        if not self.__active:
            return

        cr.set_source_rgb(*self.__pen.color)
        draw_dot(cr, *self.coords, self.__pen.line_width)

    def on_click(self, ev):
        """handle the click event"""
        # make sure we are in the correct mode

        if ev.shift() or ev.alt() or not ev.ctrl():
            return False
        if ev.hover() or ev.corner()[0] or ev.double():
            return False

        self.coords = (ev.x_abs, ev.y_abs)
        self.__pen    = self.__state.pen()
        self.__initial_width = self.__pen.line_width

        self.__bus.on("mouse_move", self.on_move)
        self.__bus.on("mouse_release", self.on_release)
        self.__bus.on("draw", self.draw)

        log.debug("activating lw at %d, %d, initial width %s",
                  int(self.coords[0]), int(self.coords[1]),
                  self.__initial_width)
        self.__active = True
        return True

    def on_release(self, ev):
        """handle the release event"""
        if not self.__active:
            return False
        self.__active = False

        self.__bus.off("mouse_move", self.on_move)
        self.__bus.off("mouse_release", self.on_release)
        self.__bus.off("draw", self.draw)

        return True

    def on_move(self, ev):
        """update on mouse move"""
        if not self.__active:
            return False
        x = ev.x_abs
        dx = x - self.coords[0]
        width = max(1, min(60, self.__initial_width + dx/20))

        self.__bus.emit_mult("set_line_width", width)
        return True

## ---------------------------------------------------------------------
class WigletPageSelector(Wiglet):
    """Wiglet for selecting the page."""

    # we need five things for talking to to the outside world:
    # 0. getting the height and width of the screen
    # 1. getting the number of pages
    # 2. getting the current page number
    # 3. setting the current page number
    # 4. adding a new page
    # one possibility: get a separate function for each of these
    # or: use gom as a single object that can do all of these, but then we
    # need to be aware of the gom object
    def __init__(self, state, bus):

        coords = (500, 0)
        wh = (20, 35)

        super().__init__("page_selector", coords)

        self.__bbox = (coords[0], coords[1], wh[0], wh[1])
        self.__height_per_page = wh[1]
        self.__gom = state.gom()
        self.__page_screen_pos = [ ]

        # we need to recalculate often because the pages might have been
        # changing a lot
        self.recalculate()
        bus.on("left_mouse_click", self.on_click, priority = 99)
        bus.on("update_win_size", self.update_size)
        bus.on("draw", self.draw)

    def recalculate(self):
        """recalculate the position of the widget"""
        self.__page_n = self.__gom.number_of_pages()     # <- outside info

        tot_h = sum(x for x, _ in self.__page_screen_pos)

        self.__bbox = (self.coords[0], self.coords[1],
                       self.__bbox[2], tot_h)

        self.__current_page_n = self.__gom.current_page_number() # <- outside info

    def update_size(self, width, height):
        """update the size of the widget"""

        self.coords = (width - self.__bbox[2], 0)
        self.__bbox = (self.coords[0], self.coords[1],
                       self.__bbox[2], self.__bbox[3])

        self.recalculate()

    def on_click(self, ev):
        """handle the click event"""
        if not ev.state.graphics().show_wiglets():
            return False

        x, y = ev.x_abs, ev.y_abs

        self.recalculate()

        if not is_click_in_bbox(x, y, self.__bbox):
            return False

        # which page is at this position?
        log.debug("page_selector: clicked inside the bbox, event %s", ev)
        dy = y - self.__bbox[1]

        page_no = self.__page_n

        for i, (y0, y1) in enumerate(self.__page_screen_pos):
            if y0 <= dy <= y1:
                page_no = i
                break
        log.debug("selected page: %s", page_no)

        page_in_range = 0 <= page_no < self.__page_n

        if page_in_range:
            log.debug("setting page to %s", page_no)
            self.__gom.set_page_number(page_no)     # <- outside info

        if page_no == self.__page_n:
            log.debug("adding a new page")
            self.__gom.set_page_number(page_no - 1) # <- outside info
            self.__gom.next_page()                  # <- outside info

        return True

    def draw(self, cr, state):
        """
        Draw the widget on cr.

        For each page, make a little white rectangle; current page is
        highlighted by inverted colors.  If the current page is selected,
        draw a little symbol for the layers.

        Finally, draw a little "+" symbol for adding a new page.
        """
        if not state.graphics().show_wiglets():
            return

        self.recalculate()

        wpos  = self.__bbox[0]
        hpos  = self.__bbox[1]
        width = self.__bbox[2]

        # page_screen_pos records the exact screen positions of the pages,
        # so when the widget is clicked, we know on which page
        self.__page_screen_pos = [ ]

        for i in range(self.__page_n + 1):
            page_pos = hpos
            self.__draw_page(cr, i, hpos, self.__bbox)

            hpos += self.__height_per_page

            # draw layer symbols for the current page
            if i == self.__current_page_n:
                page      = self.__gom.page()
                n_layers  = page.number_of_layers()
                cur_layer = page.layer_no()

                hpos = hpos + n_layers * 5 + 5
                self.__draw_layers(cr, n_layers, cur_layer,
                                   (wpos, hpos, width, None))

            self.__page_screen_pos.append((page_pos, hpos))

    def __draw_page(self, cr, page_no, hpos, bbox):
        """draw a page"""

        wpos, _, width, _ = bbox

        # grey background
        cr.set_source_rgb(0.5, 0.5, 0.5)
        cr.rectangle(wpos, hpos, width,
                     self.__height_per_page)
        cr.fill()

        # the rectangle representing the page
        if page_no == self.__current_page_n:
            cr.set_source_rgb(0, 0, 0)
        else:
            cr.set_source_rgb(1, 1, 1)

        cr.rectangle(wpos + 1, hpos + 1,
                     width - 2,
                     self.__height_per_page - 2)
        cr.fill()

        # the page number or "+" symbol
        if page_no == self.__current_page_n:
            cr.set_source_rgb(1, 1, 1)
        else:
            cr.set_source_rgb(0, 0, 0)
        cr.select_font_face("Sans",
                            cairo.FONT_SLANT_NORMAL,
                            cairo.FONT_WEIGHT_NORMAL)
        cr.set_font_size(14)
        cr.move_to(wpos + 5, hpos + 20)
        if page_no == self.__page_n:
            cr.show_text("+")
        else:
            cr.show_text(str(page_no + 1))


    def __draw_layers(self, cr, n_layers, cur_layer, bbox):
        """
        Draw n_layers layers with current layer cur_layer starting from
        bottom at position hpos and left at position wpos, with width
        width.
        """
        wpos, hpos, width, _ = bbox
        cr.set_source_rgb(0.5, 0.5, 0.5)
        cr.rectangle(wpos, hpos, width,
                     n_layers * 5 + 5)
        cr.fill()

        for j in range(n_layers):
            # draw a small rhombus for each layer
            curpos = hpos - j * 5 - 10
            if j == cur_layer:
                # inverted for the current layer
                draw_rhomb(cr, (wpos, curpos, width, 10),
                           (1, 1, 1), (1, 0, 0))
            else:
                draw_rhomb(cr, (wpos, curpos, width, 10))

## ---------------------------------------------------------------------

## ---------------------------------------------------------------------
class WigletToolSelector(Wiglet):
    """Wiglet for selecting the tool."""
    def __init__(self, bus, coords = (50, 0), func_mode = None):
        super().__init__("tool_selector", coords)

        width, height = 1000, 35
        self.__icons_only = True

        self.__modes = [ "move", "draw", "segment", "shape", "rectangle",
                        "circle", "text", "eraser", "colorpicker", "zoom" ]
        self.__modes_dict = { "move": "Move", "draw": "Draw", "segment": "Seg.Path",
                             "shape": "Shape", "rectangle": "Rectangle",
                             "circle": "Circle", "text": "Text", "eraser": "Eraser",
                             "colorpicker": "Col.Pick", "zoom": "Zoom"}

        if self.__icons_only and width > len(self.__modes) * 35:
            width = len(self.__modes) * 35

        self.__bbox = (coords[0], coords[1], width, height)
        self.__mode_func = func_mode
        self.__icons = { }

        self._init_icons()
        bus.on("left_mouse_click", self.on_click, priority = 99)
        bus.on("draw", self.draw)

    def _init_icons(self):
        """initialize the icons"""
        icons = Icons()
        self.__icons = { mode: icons.get(mode) for mode in self.__modes }

    def draw(self, cr, state):
        """draw the widget"""
        if not state.graphics().show_wiglets():
            return

        cr.set_source_rgb(0.5, 0.5, 0.5)
        cr.rectangle(*self.__bbox)
        cr.fill()

        height = self.__bbox[3]
        dw   = self.__bbox[2] / len(self.__modes)

        cur_mode = None

        if self.__mode_func and callable(self.__mode_func):
            cur_mode = self.__mode_func()

        for i, mode in enumerate(self.__modes):
            # white rectangle
            icon  = self.__icons.get(mode)
            label = self.__modes_dict[mode] if not self.__icons_only else None

            bb = (self.__bbox[0] + i * dw,
                  self.__bbox[1],
                  dw, height)

            self.__draw_label(cr, bb, label, icon, mode == cur_mode)

    def __draw_label(self, cr, bbox, label, icon, inverse = False):
        """Paint one button within the bounding box"""

        x0, y0 = bbox[0], bbox[1]
        iw = 0

        if inverse:
            cr.set_source_rgb(0, 0, 0)
        else:
            cr.set_source_rgb(1, 1, 1)

        cr.rectangle(bbox[0] + 1, bbox[1] + 1, bbox[2] - 2, bbox[3] - 2)
        cr.fill()

        if icon:
            iw = icon.get_width()
            Gdk.cairo_set_source_pixbuf(cr, icon, x0 + 5, y0 + 5)
            cr.paint()

        if not label:
            return

        if inverse:
            cr.set_source_rgb(1, 1, 1)
        else:
            cr.set_source_rgb(0, 0, 0)
        # select small font

        cr.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        cr.set_font_size(14)

        x_bearing, y_bearing, t_width, t_height, _, _ = cr.text_extents(label)

        xpos = x0 + (bbox[2] - t_width - iw) / 2 - x_bearing + iw
        ypos = y0 + (bbox[3] - t_height) / 2 - y_bearing

        cr.move_to(xpos, ypos)
        cr.show_text(label)

    def on_click(self, ev):
        """handle the click event"""

        if not ev.state.graphics().show_wiglets():
            return False

        x, y = ev.x_abs, ev.y_abs

        dw   = self.__bbox[2] / len(self.__modes)

        if not is_click_in_bbox(x, y, self.__bbox):
            return False

        # which mode is at this position?
        dx = x - self.__bbox[0]
        sel_mode = None
        i = int(dx / dw)
        sel_mode = self.__modes[i]
        if self.__mode_func and callable(self.__mode_func):
            self.__mode_func(sel_mode)

        return True

class WigletStatusLine(Wiglet):
    """Show file name on the bottom of the screen"""

    def __init__(self, bus, state, bbox = (100, 100, 15, 500)):
        coords = (bbox[0], bbox[1])
        super().__init__("status_line", coords)

        self.__state = state
        self.__params = {
                "screen_wh": (100, 100),
                "text_par": [ None, None, None, None ],
                "bbox": bbox,
                "moves": [ ],
                "moves_per_second": 0,
                "zoom": [ 1, 1 ]
                }

        self.zoom_calc()

        bus.on("mouse_move", self.rec_move, 1999)
        bus.on("update_win_size", self.update_size)
        bus.on("draw", self.draw)
        bus.on("page_zoom", self.zoom_calc, priority = -1)
        bus.on("page_zoom_reset", self.zoom_calc, priority = -1)
        bus.on("page_translate", self.zoom_calc, priority = -1)

    def zoom_calc(self, *args): # pylint: disable=unused-argument
        """recalculate the current zoom"""
        trafo = self.__state.page().trafo()
        zx, zy = trafo.calc_zoom()
        self.__params["zoom"] = (zx, zy)
        return False

    def rec_move(self, ev):
        """record the move and calculate moves / second"""

        ## current time in seconds
        t = ev.event.time
        ## convert it to hh:mm:ss
        #log.debug("time: %s", t)
        moves = self.__params["moves"]
        moves.append(t)

        if len(moves) > 100:
            # remove first element
            moves.pop(0)
            #self.__param["moves"] = moves[1:]
            #log.debug("time for 100 moves: %.2f", (self.__moves[-1] - self.__moves[0]) / 1000)
            self.__params["moves_per_second"] = 1000 * 100 / (moves[-1] - moves[0])
        return False

    def update_size(self, width, height):
        """update the size of the widget"""

        self.__params["screen_wh"] = (width, height)
        return True

    def calc_size(self):
        """Calculate the vertical size of the widget"""
        p = self.__params
        _, height = p["screen_wh"]

        # we can only update the size if we have the text parameters
        if not p["text_par"]:
            return False

        (dx, dy, tw, th) = p["text_par"]
        x0 = 5 - dx
        y0 = height - 5 - th - dy
        self.coords = (x0, y0)
        p["bbox"] = (x0 + dx - 5, y0 + dy - 5, tw + 10, th + 10)
        return True

    def draw(self, cr, state):
        """draw the widget"""
        if not state.graphics().show_wiglets():
            return

        p = self.__params
        state = self.__state
        status_line = state.config().savefile() + f" |mode: {state.mode()}|"

        status_line += ' (!)' if state.graphics().modified() else ''

        bg_cols = [ int(x * 100) for x in state.graphics().bg_color()]
        tr      = int(state.graphics().alpha() * 100)
        bg_cols.append(tr)
        status_line += f"  bg: {bg_cols}"

        pen = state.pen()
        status_line += f"  pen: col={pen.color} lw={int(100*pen.line_width)/100} "
        status_line += f"tr={int(100*pen.transparency)} type: {pen.brush_type()}"
        status_line += f'| zoom: {int(p["zoom"][0] * 100)}%'

        hov = state.hover_obj()
        status_line += f"  hover: {hov.type}" if hov else ''

        status_line += f'  moves/s: {p["moves_per_second"]:.2f}'

        cr.select_font_face("monospace", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        cr.set_font_size(14)

        # x_bearing: The horizontal distance from the origin to the leftmost part of the glyphs.
        # y_bearing: The vertical distance from the origin to the topmost part of the glyphs.
        # width: The width of the text.
        # height: The height of the text.
        # x_advance: The distance to advance horizontally after drawing the text.
        # y_advance: The distance to advance vertically after drawing the text
        #            (usually 0 for horizontal text).
        dx, dy, tw, th, _, _ = cr.text_extents(status_line)
        p["text_par"] = (dx, dy, tw, th)
        self.calc_size()

        cr.set_source_rgba(1, 1, 1, 0.5)
        cr.rectangle(*p["bbox"])
        cr.fill()

        cr.set_source_rgb(0.2, 0.2, 0.2)
        cr.move_to(*self.coords)
        cr.show_text(status_line)

class WigletColorSelector(Wiglet):
    """Wiglet for selecting the color."""
    def __init__(self, bus, bbox = (0, 0, 15, 500),
                 func_color = None,
                 func_bg = None):

        coords = (bbox[0], bbox[1])
        super().__init__("color_selector", coords)

        self.__bbox = bbox
        self.__colors = self.generate_colors()
        self.__dh = 25
        self.__func_color = func_color
        self.__func_bg    = func_bg
        self.__bus = bus
        self.recalculate()
        bus.on("left_mouse_click", self.on_click, priority = 999)
        bus.on("update_win_size", self.update_size)
        bus.on("draw", self.draw)

    def recalculate(self):
        """recalculate the position of the widget"""
        self.__color_dh = (self.__bbox[3] - self.__dh) / len(self.__colors)
        self.__colors_hpos = { color : self.__dh + i * self.__color_dh
                              for i, color in enumerate(self.__colors) }

    def update_size(self, width, height):
        """update the size of the widget"""
        self.__bbox = (self.coords[0], self.coords[1], self.__bbox[2], height - 50)
        self.recalculate()

    def draw(self, cr, state):
        """draw the widget"""
        # draw grey rectangle around my bbox
        if not state.graphics().show_wiglets():
            return

        cr.set_source_rgb(0.5, 0.5, 0.5)
        cr.rectangle(*self.__bbox)
        cr.fill()

        bg, fg = (0, 0, 0), (1, 1, 1)
        if self.__func_bg and callable(self.__func_bg):
            bg = self.__func_bg()
        if self.__func_color and callable(self.__func_color):
            fg = self.__func_color()

        cr.set_source_rgb(*bg)
        cr.rectangle(self.__bbox[0] + 4,
                     self.__bbox[1] + 9,
                     self.__bbox[2] - 5, 23)
        cr.fill()
        cr.set_source_rgb(*fg)
        cr.rectangle(self.__bbox[0] + 1,
                     self.__bbox[1] + 1,
                     self.__bbox[2] - 5, 14)
        cr.fill()

        # draw the colors
        dh = 25
        h = (self.__bbox[3] - dh)/ len(self.__colors)
        for color in self.__colors:
            cr.set_source_rgb(*color)
            cr.rectangle(self.__bbox[0] + 1,
                         self.__colors_hpos[color],
                         self.__bbox[2] - 2, h)
            cr.fill()

    def on_click(self, ev):
        """handle the click event"""
        if not ev.state.graphics().show_wiglets():
            return False

        x, y = ev.x_abs, ev.y_abs

        if not is_click_in_bbox(x, y, self.__bbox):
            return False

        log.debug("color_selector: clicked inside the bbox")

        dy = y - self.__bbox[1]
        # which color is at this position?
        sel_color = None
        for color, ypos in self.__colors_hpos.items():
            if ypos <= dy <= ypos + self.__color_dh:
                log.debug("selected color: %s", color)
                sel_color = color

        if not sel_color:
            log.debug("no color selected")
            return True

        if ev.shift():
            log.debug("setting bg to color %s", sel_color)
            self.__bus.emit("set_bg_color", False, sel_color)
        else:
            log.debug("setting fg to color %s", sel_color)
            self.__bus.emit("set_color", False, sel_color)

        self.__bus.emit("queue_draw")
        return True

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
"""Module for the icons singleton class providing icons to tools etc."""



class Icons:
    """
    Singleton class for serving
    icon pixbufs.

    Usage: Icons().get("icon_name")
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
            log.warning("icon for %s not found", name)
            return None
        return self._icons.get(name)

    def _load_icons(self):
        """
        Load all icons from stored base64 strings.
        """

        # pylint: disable=line-too-long
        icons = { "colorpicker": "iVBORw0KGgoAAAANSUhEUgAAABwAAAAbCAYAAABvCO8sAAACm0lEQVRIx7XWT4hVVRgA8N+ZGRuFnkgoEhMTITbZJsEg7I8WVIsQ0UELFxGNk6KbmFYtEqIsApNW4mbmRRRJi+i1UrNCFyahhbkIF9ngFKMp40yMohX2tTkjt/G+8s17c+Bu7v3O/XG+e75zv2SWRkTE9HsppZUds4WllL7GZQxgCVZHxLGOWcKOYSyqlY1YC6lv8hC0NfPi4ihgx3EqY8UxHzqaWMUr2IyfUMv3DuBoVCtvlkxbDKkJ8Aj2YAPOYByrolpZc9Nm6Zv8AQsjoqutic/1I3bhT9yN3jrYW5gbEV0ppZSa3Bz7sDTvhe/xYFQrywvY53gkIhY2Y03V2Bpcwa/4Aidziv+KaiUwgi/L6nEmWC+uZ/BevJjRU9iBg9jfKmwFruEqni48fgz7MJQ30gc4NGM0T1yGCzmND5SEPZdX+Q6ewi8zAvOkroz9hnvqhPbgW5zHaYw0DOYJnfgZO9FdJ3QAYzm+He/i44bAHNyNsxmrN7oLmOlHXiPYC/nE3wn9ajF1FULvK6Su/f/em/6jqPtzWjbgq2kIGLTuCXyUT5HOfOo0BmbsNTyLT/B2GTa0YPtnMTF6f0T0pJTa8fetZK6tBHs519RDnds+/LQMe7/79RMxMbooY3NvFfsXmLEBrMfG+YNx1/N7K6dvwnreO3t95OSliHg0pXQb/mhk16cC9hL68GS/2pWy4KFFAxfi4vDRiFjfSBrrpXQrnqmHVe989VpcHK5lLM0Eg468ut040a82Xrqy2zdPxrmxXRHxRlO/mUKLsVZq61VSpoNzNv3u8tXDGZvXikZr+MYHXfr4xFRhz1m95TspfYPRXNSpFVjCJdxRaBvm4RwWRMSy3MC2BLtRDu1LHh6f+pVMOwc7W923/gMDooiVvplNJAAAAABJRU5ErkJggg==",
                 "segment": "iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAYAAAByDd+UAAAABmJLR0QAAAAAAAD5Q7t/AAAFe0lEQVRIx+2WWWyUVRTHf/f7ZqbTGShDpzPF0o1NSIvdtASlWrBglAZEIIq+iDEao4kaog/EGE2MUR/UxOCDxg0xKhBrBWK0sgjYUsBSFKnFVijTZegMM12ms3SW7/jA0FSNC1HfPMmX+3C//H/5n3vuPQf+j3851JX8LCIakAlkATnAVUAhMCv9FQMFgBsYA14DXlJKxf4QKCIWwA44Jonmp4XygTwgF8hOjhtTIyMJa8g3bgp6ovjPRvB1jaGbNWofKsY91x4F7lNKbb+sb0pD7EA9UJcWnpGGTQMyEzHDHB1OMOobJ+iJ4OsO4+sK4/s5zFBvlFHfONHRJMl4CgMQhERMuGfLwkzgbhHjU6W0OIBJRAqBF4H1qYRhDvnjjF4YJ+CJ4O8OM9gV5uLZMMG+KCFfnFgoSTJxSVhNpEmh0qsOGCgsga+grx9mLl+E0ucCHZcdPgPcPeKN0bC5g65vgoQDcWJjSVJJ43fCoBB0NASF/Po4UJj0BDU3NLOiegc0Z8DyT2bgrFg0GbgGoGOvn2+2eiaEBQXoaZygqxQZ1ihTp4SY5hgmGHQSDDgnYAYa2Y4A9as+Y0nNIczmGCQckIorwDb5DHUA1xw7znwbIwNhrNYo9ikhpk8fIifHjzt3ELd7EJfLhyVjnGOt1/P1gboJVwAL5newdt0O5sz7CYwUZLig7AlwlvcAhyeqVESagBUi0Putj6HdT+GglSzHKHZbGItlHLQk6MJgfyGNDes50VZNytARFFZLjJtqD3Bb/W6yHEFIKXBXQfWzkHszwHZgM9CjlBIT0AAsVQpzYbWTwqQJznRfMq/bwDoTMbtoP+Sm8eM6+r0FaAiCItd9gdW3N1C9qBVdjwNWmH8Xw67H6f/OSV5pjOkzrWuBRUCDiLyiRGQe8AUwG4DIAPQ1gTLDtGLGwk6atgxx4M2LRMZkomzKyk9yx9qd5Bf1QMoAez6UP0nEtYG3N3Zy6nMvpbe4eeCjamwO/XJGt5uAs8C+CaAtD67eCMCoP8W2Te2cbPSnPSmm2EPUrWiibvmX2OwhMDSYUQvXPQPuxXTtGaSj6QIpQ+g9McjYuTPYKhcAGsAdmlIqBbwM7AWiQAToB453H/YNfL9nYFJpgMvto7T0FJmZEbBMhZKHYdl74F6MCJxs9BKPpwCN/NxTOH64C87vuuwwbgJQSp0RkTuBsvRGPzDkLMrcWnStI+/noyPpiyL0nJvNltc2UV7ZTs2DC5hVeS+axQJAoCdC535/Ou0GFRVtWJJnYOAgFK0BaDJNlKtSQ8DB37yrngc/rOTIC+9wdJeZC758BMVoKItDh5bSftpCWdNpau4vZF5NNp37/QR6IoBGdvZFSkp+AM0KzgqAUeBd0180iLdyZk+9adWjidLr57/OsZYqWltr8XrzEBRjgRjN75/nu11eFt7qxn82giGCoDN/wY+4XINgL4KragGOAQf/FKiUOiEi65m38dGcDMeGlQt3TF9846scP1LOkZZa+vsLEHTCwwlaP+5Lv6kKkz5ORWUbSo9fgk0pEmCnUiqk/mYf1IElRLyP4d27ku6PrMOdXXzbeg0tLUvp6y0kJToaBgYaBfnn2fTEi2Rlx2DpNii4rQtYrpTyXGkDzgRWEgs8hnffErq3aSOdP9J+tITm5mV4PMUYhsbqNQ2sWr0TnNfCLY1gdb6qlNp0xR1/Ejgb2EB85BG8B0ro3kqo8xSnT84imcqgqqoNm20EKjZD1dPDQL1SquUfzyciUiQiz0k81CeePSL714l8kCPydobIh8Ui/jYRkd0iYv1XByMRKRORNyQRDkrvlyJHN4ucaxAxkhERWf+fTGNipEwiUi8izSJiiMiYiDwvYlj/n1X/0/gFCbx4aQa42HAAAAAASUVORK5CYII=",
                 "rectangle": "iVBORw0KGgoAAAANSUhEUgAAABwAAAAaCAMAAACTisy7AAACdlBMVEW+gFX///8AAAD///////////////////////////////////////////////////////////////////////////////////////////////////////////////////7//////v3///////////////////////////////////////////////////////7///////////////////////////////////////+0tLRwcXByc3FxcnB/gH7s7et0dHQAAAANABoPAB0NABwjEDHCrtDy6Pl1dnULABdVAJZeAKVdAKRdAKVfAqZqC7K1hdkMABlcAKJlALJoB7NqCrRnBrFhAKtjAK9hAK+vfNZbAKFmA7Lbw+3ZwOzcxO+agaweAjRTAJJjALGwfNdnBLLRs+f///+rrq0DAQhMAIevfNjPsef/+/b/1pj/zID/zYH/z4PXpVmEUBBrE4phALOvY5n/z4H/zYP/3q3/+O3/rjX/mwf/nQn/nAn/ngr7mRWIJItfALWvTGD/nwf/mwX/vlz/rzj/nQr/ng34lxWHJItgALWvTGH/oAv/v17Qsef/+e7Fptzy9PLx6t/8rTVcAKFkAa9BJFgwNC5BOy7cjRH/ogr/oQr6mRKIJIp0dXMKABdkAK86AGgmAEYzBEmiSlG+W1O9WlO9W1K5Vld4FZtgALTCw8GDc5BmCaxiAK9jALBiALKuS2Lu4fiMQcSALr6BL7+BLruBIJqBHZJ/G5S+W1L9/P717/r07fr07vv05un0o0H0khb0kxj0kxn5mBP/sDj/nwz/rjb/nAf/mwb//Pb/2J7/z4j/0In/z4f/4LGPZS+MAAAAAXRSTlMAQObYZgAAATVJREFUKM9jYMALODi50AA3QpLPytoGGdja2XPAJQUdHJ2ckYCLq5u7EFzSw9PL2wcBfP38A0QQkoFBwSGhMBAWHhEZhSwZHRMQGwcB8QmJSckposiSqWnpMJCRmZWcLcUpLSMjC5PMSc/Nyy8AgsKi4pLSsvKKioJKeYRkVXVNLRDU1Tc0NjW31Na2tikiSbZ3dIJBV3dPb19nZ10/FSQnpE/ELTlp8pSpOCSnTZ8xc9bsOXPBYN58ZMkFC6MXLV6ydNnyFSCwctXqNQhJ5bXr1m/YuDG5FAY2rdkMl1RV27J12/btO3bugoHde+CSDBp79+3ff+DgocNHoODosU64JIOmlpa2zsTj9SdgoBNJEgzggQAFqJInT3Ugg9NtSJJnzp47jwwuXNRHSBoaoQEpY/xJGgDtYfX0n9HvoQAAAABJRU5ErkJggg==",
                 "move":"iVBORw0KGgoAAAANSUhEUgAAABwAAAAZCAYAAAAiwE4nAAAFKUlEQVRIx82WfWyV5RnGf8973p6ecvr9QVtKi622sdDSxiGooFCUDzdHYgzOZRMTY/xgLINsSkwUzdRADMZkC8YxNH5MplUajMygVqAM6BCC8Qs0tlAp3dmhHNrz0fac03Peyz/s6QrSQmJcdv/1PHnu57ne676u+84L/+MwAJL0gx4xxlxqrh2Px1PJL6Y+YCQmATYgoB/oBM6O7BNAEDgFHJSkSwU1khSLxQL/bD1YEDyUTTKcjjAgGI4mcRLCSjPklXnIK/NguQzGBRnZdjJvqieSX57R7c1P+wL4xaWwtgHe3fmPgt//+imui6zh8pI+CopOgwy2PYzLlSSZtDkZzGcomolkkCySjscld25OztS8nPLG3Lor5uZSWJkxUHplVn9mofuV8VjbAGm2u7/f6ckNE2B6XYez/I5XLclgjDBGSIZ43E089h17xzHEY+mcOVPAya4KOturad46nVgix7vs8Rrv0oeqbxhXQ4AZ02cECktzckOdvUSjGQnLctLG6mmM8GQM4ckYGr0Yi3qIxdMpLg0wMFDA8eNR3PmTmDYrByAwEWB2ZWVlqKSsmP7OUwSDNe5Ewsa2E+dm6r/VkQxvb7+NvXsWEhyO0KfTFDcEuOuJmcO1CyfvApaNp6FljAkD1NRU08dx/uMvJBzKAaMJ2kBMmdKD2x0npCCNd6fz6Pbl3TfcMmsVsGQix1qpRUNDAyFO4e+38ftLJgQEmHd9G/ev/BMLaodIfjCDfRsG8o619t54sb62U4v6uplhV1Y0KxAO4esp58q6Ty/aUzW1R5la3s2Rwz+hbWdT5hc7625vvLU8fN2Ksu0p0PPZjgJWVFT48oqzs4Lh05z2l4BjXbyLHYtJ3gjzFuyh8aqPOXTwana/sTjrWOv0O+f8qtw3+47S77VHCnBSVVXVYMmUyYQ6/AQC00gmbFyu5DlGMZZzjnlGzSRDZlaIpkWtzGz4lL175rNnY1Pp0fer1y5cddmxEbbZxphwikYUYEpZKYMECIczGR5OG9Vx965F/P21u+g7WwCWc2G2MuBYFBT1cuvyt1j5m40Mfb2P7Y9+Vdv/76gDPCsp2xqpswDy8/MZZohY3E0y6QLL4ZuuSna881N27LqWl164j+6uSjDO+KaSASMkMRDNpqapgMxCtwXcDFSeI1Sm10ucMIND6cRjHjq+rOVvL9+DzzVAm1nPzqNxNv15NR++/zMi4ezv2J4PbEQk7KX5jdsobGhk2bpqbLcFsBU4Zo/NraiYRoRt9IYMe9sW0t4+C1NVyoOv1jK5pYPXXthMT+AaTr5+M4cPzWF+UysNjUfI8EZGTSbH4oP3FtEbuYaVT9aQVeQB6AZeN8bEx5hCam8/ECvKLdXPWa/7TIs2Lm3TyQ5fUtKmaDT63LaWbYMLFsxXoatKV3OvVrhe1DO1G/TZmnlKbsmRXvLqwD2LtTq3Wbuf69JIJCT97vull+Tz+Trq6+t1FSu0tuo9dR3uC0p6RGPC7/cf+cvm5zV79hwVmRrNY40eyHhZby65X5+snqu1hVvU/IejSsSd1JUWSZkXAkyTpHWPrdOc+vna1fJRQtKD50+NkUce7unp+XrD0+tVWXG5pjFXt7NJv3Vv1TM37VfkbDwF9rmkGeP2sSQNDg6q90yvkk5ix3gjagzhPx5oP3BqydLFyucy/bLqSXX+K5A680m6acLBIal4bPkuNmhSeSdOnGj+65bNw/v37xur26of7U9sBOShEVZJSVsvqNuPAJqKIv5f4lttCtvxW3g8aQAAAABJRU5ErkJggg==",
                 "text":"iVBORw0KGgoAAAANSUhEUgAAABwAAAAbCAQAAADFASenAAACtklEQVQ4y42US2iTWRiGn5M//RuTXmK1l4xVK0gRb2hxiooywwiKNwQXdlEVQUUUEaSIIIK3zUCHKY6goqAuRzfKaMWFFUXd1GEQaqFQ8JK0SUNsQo1JTW3yuqi1Nhc7z/blPec73/d+xwAE8fGMo5RymjjjbKaYvbTioZ9a8vIK8e63dExx5ZBcIJTHY+AtXsq9xKrCPLVGDWTIZBwOBxqqXrm6w/W5ZCghqrKMDijFhlhLMlL9flns04dAZE14Z0VDoi/sTjb853o8VPTRzWi+QoPISI2i3TFfkpw3bR28JElswn9SkgjkuBxQAnOhp4/j6d5fDAhIAz6jdlo7gOFZ7nxGC0bB1aIuY5LfSSGM0V+Jw5DJmByjEyzog/DN5VmqH7BMWuAOZfIZw6SxjLhDXY7cR78RjjwDccJcCvMTUcQIMV4DCcqBDHVjbyxMBOGhoqhGC5lHJ4tlT/fiGm9OYWbyELuEkZbHEmrSrnPV0VIbAmOlFsbPovn0bhvs+NX/cs3zuws6T7QOH0uVmFjujZMb+y++3uVb/aes1D9lbV1P7scT1zrBk33+H0ARknXL1v6LUhX7GAt6TdJ7RZI826UebmXnSNhIbKAbzbvPCslGnEXyfZxxRapBksrYMj6OcQZJ6WDb+ntppzAbl666fSSluLEBRwYDv2MMwL3sUqPkbuMnLiDVxmdelW5wniD9ud2IMp0Qn7+FuAI3KVxIlbJeDDQCBIs1UssUBBlFTmlZrzNkqTK6sP3AOekdU/KWZqTNz1lfW+8+PaNjbZME3VMbB5HaHlU2T7z4/NedmSJyRcCAJ/IC6hvLzixpgN3AnIntKISAYpsy8B56syf9M/Twv4igadKKPyVpyd/rHkhdk/exEG4Cw7PZ4Vt3PfS6es7Dpskfyw8IcJmXE6FABL9TvwD2d2QO1hAt8wAAAABJRU5ErkJggg==",
                 "eraser":"iVBORw0KGgoAAAANSUhEUgAAABwAAAAaCAYAAACkVDyJAAAFj0lEQVRIx62Wa1CUVRzGn3PevS+wCywXXdmFdVPMuClU462UFEUJrzGNqTM0jFPQ2DSmH5ypkWo0bZokGx0LY6ycIacMAnWUBtAmLxiChHKTqwqoqMCy7L67+/770GqIkKidj+cyv/M85/wvDE8xiIhGW2OMsZHmZU8DYyx1tlzw5E4z9wtygXraelS+Xb0KhcvDj927zHCw7ClgC/w14nebk9vDN8y/1qtSSL4DDsFx5oqf36lGXWacee71qnafA8PB7Mlgy2LDDbb8bStbJqcl3CDGIYKgBABwAhjo9zr9rfmfRcPhYi0AdhIVHWKMMdnjw5ZPnGa+m/fx8pbJi6J7AGLsPgwAJAZwYjf65G6PW8784Rdsh337Y1vqtTFx3czOvPeSOmZHh9kAGsEgTrh8TYuthWbDVIrpj0OMphOdmhi25TUiItnYYa/uTo27+fPWpa06c9AgILFB74XbACgAmMCJrt9Wiht/jFC0dphrliHKIIPMFIYwuOBKG5NCL6zSFOBI2rayReWF9QNoAWAEcAnAi+CEM4066Z2DE90tLZbBJMwNVUAxDoCjC129NaixPhLohTUqBOnsb5uquTXE/iYk5gHgADDJez4ZnGTHa/zx1vdWp617UnsiZobroNMDQBvabpWi1OOCWAYgRvYIWM0zIfZzlR/+6fRRuzNATPJCgoa+2V8dPtInB2dyn+7I1jl4TiWHXA0AfehDBSoCXXDmEBVvHvWX/gNL2fdy5N1Vh7NqXT5qTxzoocxBYGCDTgGfFkzmPp2JjkiEBEmQgggEF1yoQpV4B7e3ERVn34tDPspv3PiCpW/1/vR6f73WPQ30QLwOAKgHYAcj5J8z4Gh1yF09fCQCBQKABInO47y7AQ0FREXZQ7MNH0HZ5yq5+/0tS9o9EcGD9SD0DdlSDmALgAFw0pZe0uOjQotodEXWaqB2E4hz8O4udN1qRKNNosJVw1MbH5ZBIgG8YfARlXFmmxycnGAkDNkfAoZ0MIoprgqwv/ttPIJuzkU8YqcD8GNgHa1o3VqK0itO+knPWYpyuIPCPZggMCVR81qNVhs+wTTlSOt1J7t+S5iuU0uOAF93OwRygGGCy82xvdh054v8l3wn9i6QWWAWOLhMgiRWo/rcKZyc46LDCYylRAFFnQ9VkX/VsQ0mk2lnTs4uZ2rq0saGphudG7J2W2ovVBvUrL9kaujdukWxNhw4HWo+02BYlIzFulAEayRI8MCDSlRKF3GxU6JCI2MpBqCoZ8Sy5YXNZ4zl7tmzx3/9+vW9AL4GEDM4KCb8cbpu/OVLHcKZsw3iLwWlfQP9zoEYPH8hAfEJAMo5eHwzmieVoazDRYfDRquDD7wh53yXUqnUNzQ03Kyvr9cAWA5AUKsVNYnzoo9nZS2uyv0mU/HKPKVBjnZ7NKJjGNivHPyYA47AJjT1jgV2H2i1Wi2ZmZl+FRUVloULF6rS0tJ0BQUFJlEU1QCMoigGHzlSDLv9zqBWpTL1oTccQJId9t1XcbWuhb7UMTZFGEteZkREERERTWvWrGHjx48Pr66u5iUlJVebmpqcGo3mbHJy8hXOeWZ6enrgrFkz7Tt2HHbvyz4p6KEX7LDvb6Ov3h6LsuF9yb4ZM2bYc3JyKDc3l/bu3Xs7Kyur1mg0NgMYyMjIcNhsNjcRieXlF3sE2RIbkFz6Xz3NaEPGGGNERCqV6gOZTPZ6QkLCFKPR6B8VFeVvsVi68/Ly+IoVK5RarRYOh0M4ejTf1+NuPEhUt+6xlA0twPegjLGw2traVQEBAau1Wu10q9Xa3N/fL544ceJZq9UqZmdnGw8dOlRGZH8i2Ij2em0aB2CTt9b1CILQrNPpagD88CQ2PhT4o/WbjLF4AEsBxAK4TURrn1YZe5xG9/+w8W8l9atrgG4MXQAAAABJRU5ErkJggg==",
                 "circle":"iVBORw0KGgoAAAANSUhEUgAAABwAAAAbCAYAAABvCO8sAAAEqElEQVRIx8WWWWxVVRSGv33OHdrblk400FKwKTMyyZQYxYggEFGJyoMx0YBiYhAeMFETSdQHNGpiDIk4oJEHEzFAMBgkBgICKkRFi2KZhNILtpSWTtzeofees38fbkOYLpHBuN7W3jn7y17r//c6cB0hyeW/DlnrSpquS2OBZAtuPUw2X9Lrsn6nOo9Ip7ZKrT9LXqJX0jpJ1bfwZp6RtEKZhKcDb0tfDpfWlkifV0m7npF6TkvSBskWX8+5Ts4d494DeoFDH7jUvQnxJnDzINMDx7+An16CdPdjYBbdNFBSEHiOtt/KqF8NNg3Vc2H2Jhi7DNwwRL+BE+sN8OxNl1bSRMm2au9y6dOwtH6M1FFvJR1UOpbQtgXZ9c3TpeQ5K+npmy3pvcSbKmjeCRioeQRKxzQCiwgW7mTkQggUQuchOLffALNvGNjntffoqIee0xAqhuo5ANuMMfuBjVRMtRQPBa8HWvYC3ClpsaQ5kkZKCl/PDbOq6zwIXgIKqqFkuAV29+3vI69/K2UTQBba/8BLJodkUnaN9bUZ2AmkJM2XVejywwNXAZYDzZxvqEI+FN0GodJu4Gjf/miME6J4BDgu3SdOsP7j3XR39TNF/YPh26aUVI17YACDxvZbh2GtrF41jmm/FrAEmy4geTabRSrBDZcCZyXNAN6P1sXLTq/3uHt4AKW6aNgTpaV9IAbxy4Ymdqxq4L5ltfkzl9UuCRW4EVktNY6J5yppBJsJkolnBRMqvrjUK88cjg367Klfqdsax8olGPLIy8/g4OBiMBg6m5N8teIwW1Yexff0JIYnrtVDF8lBXt8DcKEIs3xP07595zjRP7surBsjHMde+mZgsFZ8t/okx3adc4GFkopzATMYx8cJAsqavs8q7Y2JwOHtrRgMgYCHMcJag+9fOUQMEI9lOLC5Jdt3qM0FjOEE0wT7ZbPeDkAtwNBYay+JrgwGQ6SgB+NYMukQvekwBl3VBu3RBH5G+UBZLmAnTrCUSGU2izeBlyoCCsOFAYJ5LiDKy9vB9emJF5JM5Oc0eqQ0iBMwGSCnaDqyEhmR7VMsCqm2PKC2ojbiD76jGJc0gwb9DUa0n6sglcq/6g0DjsPomRUYwyng5FWBxpgYsIbycRAqgkQzdBx0AcKFgaZZy4dTMypBdXUUrEO0sYaMH7wCZhHj5w1gwsMDATYZY85e6y39npLRln7DIBOH6BaQ3wnUj51b4S16o5uysjaSsQhHDo1CWXkhhI9wXMOk+ZU8vmockZLgD8BH1zJ+FpjXv4Eh84bRXgent8KZPaVUzbjLSTU7VeGNEEjTk5mGO3AiZQmwniVcGKBydBGTF1Qx6dFK8ouDe4DnjTHNF6s314h6i/MNL7N9AXQdhrLxMGoxNO/IzsJAPpr+CcnSh+hpS+JlLHlFAYoqwl4wz2kE1gEfGmPOXG6XXMBa4GsaNt7OvuWQasuKSD44IRi7FCa/1o0TWgO8eNnnNcaY6I0M4gdl/TNq3CxtuV9aN0zaNFX6/V0pHeuV9IrS580t/nPTHEk/qrcrpe6/pESLJ+mYpCWyfuh6zzP/EloOTAEGA53A/hsq2f8R/wC95leyPYReegAAAABJRU5ErkJggg==",
                 "shape":"iVBORw0KGgoAAAANSUhEUgAAABkAAAAcCAYAAACUJBTQAAAFb0lEQVRIx62VW2xU5RbHf2vvmenM0CmdWlo6XEppBUUqImgo5ByVEOEBRF9UDAovXhBjjCbknCcviU9eIlFiUHMSjXjUqMFLVBCNRQUVMEALVqBjLS2X0pmh7dw6M/tbPsxMKdNiSPSffMneX75v/ff6r7X+Gy4TqmrpBcRVNaqqO1X1DlW1+LswRu1C8KiTNZoazGp22CkSRlV1k6qWXeq+XGYWCpzv+Ka/cvfWLvq7kpRP8nDNrTW03DsNf9CdATaIyP/+Fsnet3p4/4k2BvqHEUABG2H+7XWsebmZ4FTf58AdIpIpvW9dDkG0O3X2280HSfQnsbCwEOzC9+3ffoodz51AlRAwYbwY1qhgs/ViiKoqZrjf2/1q7drV/2XtujepnBhDRwkgQOfeKIlIxhQSHANXgcAHdBhH6Tk8SN/xeHLhnVMMsIfj2xb7w8/ir08xrT4MwP+3rSOXc40EySQdyqs914vI+b/KJAhw5Ms+Nq/Yy2trDvifmtdqune1LqbtecilwdigwqKWH5jbfAgHeyRIRa0vV5S2AHs8EgU4/l2EWF8KNUrv4ZjV+8FmSHaBFI6p4PGmWXnbdmZMD6MIgs28m9MWx7YOcfiFDH0/9gO5gjoX5AKiAMbRtCBeRQhWxmiobwdT0oDGon5GmA0bN3P0yFyMullQ02mxpzOAyUH59GoWPO3QeFdSVX0ikpZRhQ8Ag/fLxzsM1qIpdd0TN/3nWcorBkHH6XRRsExeAyMXRFEHvDWw5OVh6leViYiMdJeIDImIvK6rlwenVtzdtDjYhe0GMfmAY3pbwLHztRo9CWJD6gz88WlZqVyMIhNV1dSp8Err4E0vxX77ocltp5hQHkcsM35W4824ZWtx2F2XOuYLzfxs0Hl+4/tb9mzp/aWHude2s3TZTqprzoKxLpYNk5csL0wWT0U/vsnbgQ1jbKXgUbx+z4FVhz45bW55pHHpsd2Rx4/tjYqFMOvKDu5b/wZ1od58RqIkExMYkvmDtf7vKy76yvVxRERGSFS1CojseO7Evp/e6Zl1rjPhH044iIWtjlpaKInBYs7VbTzw0CsEJg4wNFDJO2+v48TJJaZhcV344VX/biolAHCpahCI7H6ti8+e+e2GZDyLIHkTNKVKK12/NxKJVBMIxvj5xxb27VsE5KzYRyebNu54l5sebBi6EwKqWll0AAuIxnpSzq6XwiTjWawCwYiECAYLRfC4h5nbfIja2jNgLOLxAAYLCndSCcPB944GTrfuBIgV5XcB9LYP2ec6E1glzq8Ifl+C+dcfYEZDmFCoh6nTTuLzJ0GFJf9q5cyZOtra5pHLupk543dW3/4BNZ0noK8mwbxNW4pOa3758JRsvWs/xtGLCCoCA9yz9k0WLNyHuLL5YhdXobMyaS/d3fWk0z6mTuumsiqS7zQ1MGVZhuWfeFyAeCa4EFtgFIlBaL6unYU3/JTfcOwSa3UDgqcsRdPsjnwrG2tUe1sQbfcUa5KcPKuc4BQvTuF3YFAmTvLSsqYK7DGjDrYvi2ag5cU2QkuHMQK5os1qYTngrR5x4f3VM/2sevIqJjeW4/baVIV8maWPNn591YrZGdQ9dppNMr85a10zyz8tY87D56m6FjyV4K7IL38I5jxUUFV1GfAVQKQryblwIudk9ZVrltc8RrrvFD1fB8nFXRzZ4mbgeHGquxGdzvo45HN1ON/xBqmzi/K/YFU8lcIV1zUANxaH8WrgaIksZSDDI69frIDTrSCuC4Y5auiK7VqCgIjErYIp/ipjYGWKTwB4J4Fx8lZuDBjdVmqs4yD+lwY5Bs2PK3aZEP8DXOUQnDPIPwlVDamqqpPNamZQNZsYuIQ84+JPZVirv1LpTsoAAAAASUVORK5CYII=",
                 "draw":"iVBORw0KGgoAAAANSUhEUgAAABwAAAAaCAYAAACkVDyJAAAFAUlEQVRIx92WWWxUZRTHf9+9t8y0nel02s6UQmkZ28rSBbugCEUIi4IokWAUMS4oD8aYiDwQouEB9YEXDIkmJJAIQRNME8FEgQASVmVrrRTaQgtCO3SGdjrTZdpZeufO50OhQC1Y0Sf/yffy3XPO/5x7vrPA/x1ipEsppXyokhDiUQm1B5K5D9TiOT6ZQF0iER9ICaY0SeaMS3/n0MOc0u4hsgBB2o400rxrCm1HSweJ4iCUO1KCzppiQl4PBW/WErzmItiSQTRgxogqqOYIic4OrK7Ld5waTixuk2UBHmo2GjTvUulvAyUBUvLAOQNsBQaKpnDzsMB7HJQxYLJDtAti4UGnkCDE4LcEK6Q8BtnPVlO2oeJeUk1KaQI8nF0HDVtVpAG2SZC/Mkrui+ewT50NqAB0nB3UiusQbocxtkHDpjRQTaD3QcgLoXbwVUP3lQr0/moppbxDqgERWvcHafrGCsDE5ZLiD8/iqJgBzAagp6GaM2sq8J4ERQXikJAC0zcFyJpTR4KlE0UzMAacRHwuuurH07hdpbMG2n8t+2sO3futRAPgKIfS9TuwF74zlNtw1z7/oa+XtBwdoMO3DN1IxmoJkD/JTXZZ7JSwTFg6ZC0BMKdD6uRXSU4VNOzUMTlVoOouYbClF191CqoZxla230tWt69jf92e5sX1PxYS6JyJLjUkAgWw27pYEM2ZPv8T3u73RyKB1vDUPv9Aph42khRNLLA4ym+mT5rfkJZF1VBKACGllBx7awN6SKHgdQ8TX9ru/r3ni192tq6qrvLYA94oIEiyKKSON6NpBsE2Pz3dSSTZxlDygnOgxxtVA+6QGumNYegSRRUk2jTs2Ynx7Gkp7Su2FGfdebFieKE3n/Tv3ftx47KmU37iCDLsnVTM9VC8ekV3ZklOp3mg0XV+0xa1audioroJiSSORCDQhEDRBHEDYvE4cSQqAkduMovW5dfPfd9VKIYX/dbl5zi/x4PZBNOfPMPC+T+QPefpXubuSAG48F21UfVRrdp+yznUpibPy6DouUxSs80DpmRNj0UNzd8SNrXWdnPlmJ+AN8y4x618fmX+fYUvq6vaGusPdkzRFJXn19pZPO0QargVrK+5gcK6n279tnt9V1nHLSdWm0ZMl0RDMUqWjI0sXJtnBsbcPkOo+d5z7vDma1GhIIDK+1pb0wl/QV+/Tm6RnWfey0O9kAh9QE+9K3Byt3//p6llHS0DZDjCvLK5iCPburl8ykfnjZAYqatIKbXy5eP08uXjhu6UewVC3bp3bK7levEix/WUCWM7cFYCkvj1g0mHPzuZ3nw+SmKiZOn69L6KN4owWzQkklg0rj6gn8bEMAxFGO030ld/W+6/T6Pwg3b0QKb7RBNnTs8EFGatyqFy7RMWgLgRB8TdVvtPpoXZogWGjysppaRyO1eOnaa37xYZOWYq33XVAqUAAXcYBUFqllkfafKMBGU0I8bdZMZAMmGajZwyWynAz1uueX1X+0myJJBTZvOONsJR/Qw9EhsUVoUEOLHtRu2RL//IisYM8mel4XrKfvyRB/BIyMy3SAUhGo/4xMaSo3TeCJX2BwdIz0piwZo8n9VhWjXaLWBUEZa/PO6qqyyN/qDOjYvdRIIxXKV2Vn5VEi5a5HSM1s4Dd5qR1o6bF3tbLx1oz4n0xkjPTYpPnpfR4MhLLgIShBCx/5TwQYvVv1mm/r/4E+uvH1a7XvVXAAAAAElFTkSuQmCC",
                 "app_icon":"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",
                 }

        self._icons = { name: base64_to_pixbuf(data) for name, data in icons.items() }
"""
This module contains the Page class, which is a container for Layers,
and Layer class, which actually handles the objects.
"""


class LayerSelectionPropertyHandler:
    """
    Base class for handling properties of selected objects.
    """
    def __init__(self, layer, bus):
        self.__layer = layer
        self.__bus   = bus

    def selection_color_set(self, color):
        """Set the color of the selected objects."""
        log.debug("setting color selection")
        if not self.__layer.selection().is_empty():
            cmd = SetColorCommand(self.__layer.selection(), color)
            self.__layer.selection().modified(True)
            self.__bus.emit("history_append", True, cmd)

    def selection_font_set(self, font_description):
        """Set the font of the selected objects."""
        if not self.__layer.selection().is_empty():
            cmd = SetFontCommand(self.__layer.selection(), font_description)
            self.__layer.selection().modified(True)
            self.__bus.emit("history_append", True, cmd)

    def selection_set_transparency(self, transparency):
        """Set the line width of the selected objects."""
        if not self.__layer.selection().is_empty():
            cmd = SetTransparencyCommand(self.__layer.selection(), transparency)
            self.__layer.selection().modified(True)
            self.__bus.emit("history_append", True, cmd)

    def selection_set_line_width(self, width):
        """Set the line width of the selected objects."""
        if not self.__layer.selection().is_empty():
            cmd = SetLineWidthCommand(self.__layer.selection(), width)
            self.__layer.selection().modified(True)
            self.__bus.emit("history_append", True, cmd)

    def selection_change_stroke(self, direction):
        """Change the stroke size of the selected objects."""
        if not self.__layer.selection().is_empty():
            cmd = ChangeStrokeCommand(self.__layer.selection(), direction)
            self.__layer.selection().modified(True)
            self.__bus.emit("history_append", True, cmd)

  # XXX! this is not implemented
    def selection_apply_pen(self):
        """Apply the pen to the selected objects."""
  #     if not self.__layer.selection().is_empty():
  #         pen = self.__state.pen()
  #         self.__history.append(SetPenCommand(self.__layer.selection(), pen))

    def selection_fill(self):
        """Toggle the fill of the selected objects."""
        if not self.__layer.selection().is_empty():
            cmd = ToggleFillCommand(self.__layer.selection())
            self.__layer.selection().modified(True)
            self.__bus.emit("history_append", True, cmd)


class LayerEventHandler:
    """
    Base class for handling events on a layer.
    """
    def __init__(self, layer, bus):
        self.__layer = layer
        self.__ph    = LayerSelectionPropertyHandler(layer, bus)
        self.__bus   = bus

        # Dictionary containing the event signal name, listener, and priority (if given)
        self.__event_listeners = {
            "selection_group": {"listener": self.selection_group},
            "selection_ungroup": {"listener": self.selection_ungroup},
            "selection_delete": {"listener": self.selection_delete},
            "selection_clip": {"listener": self.selection_clip},
            "selection_unclip": {"listener": self.selection_unclip},
            "rotate_selection": {"listener": self.rotate_selection},
            "move_selection": {"listener": self.move_selection},
            "selection_zmove": {"listener": self.selection_zmove},
            "transmute_selection": {"listener": self.transmute_selection},
            "remove_objects": {"listener": self.remove_objects},
            "set_selection": {"listener": self.selection_set},
            "add_object": {"listener": self.add_object, "priority": 1},
            "clear_page": {"listener": self.clear, "priority": 9},
            "flush_selection": {"listener": self.flush_selection},
            "set_color": {"listener": self.__ph.selection_color_set},
            "set_font": {"listener": self.__ph.selection_font_set},
            "set_transparency": {"listener": self.__ph.selection_set_transparency},
            "set_line_width": {"listener": self.__ph.selection_set_line_width},
            "stroke_change": {"listener": self.__ph.selection_change_stroke},
            # "apply_pen_to_selection": {"listener": self.__ph.selection_apply_pen},
            "selection_fill": {"listener": self.__ph.selection_fill}
        }

        self.activate()


    def activate(self):
        """Activate the event handler."""
        bus_listeners_on(self.__bus, self.__event_listeners)

    def deactivate(self):
        """Deactivate the event handler."""
        bus_listeners_off(self.__bus, self.__event_listeners)

    def selection_group(self):
        """Group selected objects."""
        if self.__layer.selection().n() < 2:
            return
        log.debug("Grouping n=%s objects", self.__layer.selection().n())
        cmd = GroupObjectCommand(self.__layer.selection().objects,
                                                 self.__layer.objects(),
                                                 selection_object=self.__layer.selection())
        self.__bus.emit("history_append", True, cmd)

    def selection_ungroup(self):
        """Ungroup selected objects."""
        if self.__layer.selection().is_empty():
            return
        cmd = UngroupObjectCommand(self.__layer.selection().objects,
                                                   self.__layer.objects(),
                                                   selection_object=self.__layer.selection())
        self.__bus.emit("history_append", True, cmd)

    def selection_delete(self):
        """Delete selected objects."""

        if self.__layer.selection().objects:
            cmd = RemoveCommand(self.__layer.selection().objects,
                                                self.__layer.objects())
            self.__bus.emit("history_append", True, cmd)
            self.__layer.selection().clear()

    def selection_clip(self):
        """Clip the selected objects."""
        if self.__layer.selection().is_empty():
            return

        obj = self.__layer.selection().objects

        if len(obj) < 2:
            log.warning("need at least two objects to clip")
            return

        log.debug("object: %s", obj[-1].type)
        if not obj[-1].type in [ "rectangle", "shape", "circle" ]:
            log.warning("Need a shape, rectangle or circle to clip, not %s", obj[-1].type)
            return

        cmd = ClipCommand(obj[-1], obj[:-1],
                                       self.__layer.objects(),
                                       selection_object=self.__layer.selection())
        self.__bus.emit("history_append", True, cmd)

    def selection_unclip(self):
        """Unclip the selected objects."""
        if self.__layer.selection().is_empty():
            return
        cmd = UnClipCommand(self.__layer.selection().objects,
                                         self.__layer.objects(),
                                         selection_object=self.__layer.selection())
        self.__bus.emit("history_append", True, cmd)

    def rotate_selection(self, angle):
        """Rotate the selected objects by the given angle (degrees)."""
        if self.__layer.selection().is_empty():
            return

        obj = self.__layer.selection()
        event_obj = RotateCommand(obj, angle=radians(angle))
        event_obj.event_finish()
        self.__bus.emit("history_append", True, event_obj)

    def move_selection(self, dx, dy):
        """Move the selected objects by the given amount."""
        if self.__layer.selection().is_empty():
            return

        obj = self.__layer.selection().copy()
        event_obj = MoveCommand(obj, (0, 0))
        event_obj.event_update(dx, dy)
        self.__bus.emit("history_append", True, event_obj)

    def selection_zmove(self, operation):
        """move the selected objects long the z-axis."""
        if self.__layer.selection().is_empty() or not operation:
            return

        cmd = ZStackCommand(self.__layer.selection().objects,
                                            self.__layer.objects(), operation)
        self.__bus.emit("history_append", True, cmd)

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

    def transmute_selection(self, mode):
        """
        Transmute the selected objects to the given mode.

        Args:
            mode (str): The mode to transmute to.
        """
        if self.__layer.selection().is_empty():
            return
        objects = self.__layer.selection().objects
        cmd = TransmuteCommand(objects=objects,
                               stack=self.__layer.objects(),
                               new_type=mode,
                               selection_objects=self.__layer.selection().objects)
        self.__bus.emit("history_append", True, cmd)

    def remove_objects(self, objects, clear_selection = False):
        """Remove an object from the list of objects."""
        cmd = RemoveCommand(objects, self.__layer.objects())
        self.__bus.emit("history_append", True, cmd)
        if clear_selection:
            self.__layer.selection().clear()

    def selection_set(self, what):
        """Dispatch to the correct selection function"""

        if not what:
            return False

        if what == "all":
            self.__layer.selection().all()
        elif what == "next_object":
            self.__layer.selection().next()
        elif what == "previous_object":
            self.__layer.selection().prev()
        elif what == "reverse":
            self.__layer.selection().reverse()
        elif what == "nothing":
            self.__layer.selection().clear()
        else:
            log.debug("Setting selection to %s", what)
            self.__layer.selection().set(what)

        self.__bus.emit("mode_set", False, "move")
        return True

    def add_object(self, obj):
        """Add an object to the list of objects."""

        log.debug("Adding object %s", obj)

        if obj in self.__layer.objects():
            log.warning("object %s already in list", obj)
            return None
        if not isinstance(obj, Drawable):
            raise ValueError("Only Drawables can be added to the stack")

        cmd = AddCommand([obj], self.__layer.objects())
        self.__bus.emit("history_append", True, cmd)

        return obj

    def clear(self):
        """Clear the list of objects."""
        self.__layer.selection().clear()

    def flush_selection(self, flush_direction):
        """Flush the selection in the given direction."""
        if self.__layer.selection().n() < 2:
            return

        cmd = FlushCommand(self.__layer.selection().objects, flush_direction)
        self.__bus.emit("history_append", True, cmd)



class Layer:
    """
    A layer is a container for objects.
    """
    def __init__(self):
        self.__objects = []
        self.__selection = SelectionObject(self.__objects)
        self.__bus = None
        self.__layer_handler = None

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

    def export(self):
        """Exports the layer as a dict"""
        return [ obj.to_dict() for obj in self.__objects ]

    def activate(self, bus):
        """Activate the layer."""
        log.debug("layer %s activating", self)
        bus.emit_mult("layer_deactivate")
        bus.on("layer_deactivate", self.deactivate)
        self.__bus = bus
        self.__layer_handler = LayerEventHandler(self, bus)

    def deactivate(self):
        """Deactivate the layer."""
        log.debug("layer %s dectivating", self)

        self.__bus.off("layer_deactivate", self.deactivate)
        self.__layer_handler.deactivate()
        self.__layer_handler = None
        self.__bus = None

## ---------------------------------------------------------------------
##
##       Page class for handling entire pages with multiple layers
##
## ---------------------------------------------------------------------

class PageChain:
    """Base class for Page. Handles the linked list structure of pages."""

    def __init__(self, prev = None, next_p = None):
        self.__prev = prev
        self.__next = next_p

    def next(self, create = True):
        """
        Return the next page.

        If create is True, create a new page if it doesn't exist.
        """
        if not self.__next and create:
            log.debug("Creating new page")
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

    def insert(self):
        """Insert a new page after the current page."""
        # we are already the last page
        cmd = InsertPageCommand(self)
        ret = self.__next
        return ret, cmd

    def delete(self):
        """Delete the page and create links between prev and next pages."""
        if not self.__prev and not self.__next:
            log.debug("only one page remaining")
            return self, None

        cmd = DeletePageCommand(self)
        ret = self.__prev or self.__next

        return ret, cmd

class PageView(PageChain):
    """Page Chain augmented by transformations"""
    def __init__(self, prev = None):
        super().__init__(prev)

        self.__trafo = Trafo()

    def trafo(self):
        """Return the transformation object."""
        return self.__trafo

    def translate(self, new_val):
        """set the translate"""

        self.__trafo.add_trafo(("move", new_val))

    def pos_abs_to_rel(self, pos):
        """recalculate the absolute position to relative"""
        pos = self.__trafo.apply_reverse([ pos ])[0]

        return pos

    def reset_trafo(self):
        """Reset the transformation."""
        self.__trafo = Trafo()

    def zoom(self, pos, factor):
        """Zoom in and out the view."""
        self.__trafo.add_trafo(("resize", (*pos, factor, factor)))


class Page(PageView):
    """
    A page is a container for layers.

    It serves as an interface between layers and whatever wants to
    manipulate objects or selection on a layer by choosing the current
    layer and managing layers.
    """
    def __init__(self, prev = None, layers = None):
        super().__init__(prev)

        self.__layers = [ layers or Layer() ]
        self.__current_layer = 0
        self.__drawer = Drawer()
        self.__bus = None

        self.__listeners = {
            "page_deactivate": { "listener": self.deactivate},
            "next_layer": { "listener": self.next_layer},
            "prev_layer": { "listener": self.prev_layer},
            "delete_layer": { "listener": self.delete_layer_cmd},
            "clear_page": { "listener": self.clear, "priority": 8},
            "page_zoom_reset": { "listener": self.reset_trafo},
            "page_translate": { "listener": self.translate},
            "page_zoom":    { "listener": self.zoom},
        }


    def activate(self, bus):
        """Activate the page so that it responds to the bus"""
        # the active page responds to signals requesting layer manipulation
        log.debug("page %s activating", self)

        # shout out to the previous current page to get lost
        bus.emit_once("page_deactivate")
        bus_listeners_on(bus, self.__listeners)

        self.layer().activate(bus)

        self.__bus = bus

    def deactivate(self):
        """Stop reacting to the bus"""
        bus = self.__bus

        if bus is None:
            return

        log.debug("page %s deactivating", self)

        bus_listeners_off(bus, self.__listeners)
        self.layer().deactivate()

        self.__bus = None

    def objects_all_layers(self):
        """Return all objects on all layers."""
        objects = [ obj for layer in self.__layers for obj in layer.objects() ]
        return objects

    def number_of_layers(self):
        """Return the number of layers."""
        return len(self.__layers)

    def next_layer(self):
        """Switch to the next layer."""
        self.__current_layer += 1
        if self.__current_layer == len(self.__layers):
            self.__layers.append(Layer())
            log.debug("appending a new layer, total now %s", len(self.__layers))
        self.__layers[self.__current_layer].selection().all()
        if self.__bus:
            self.__layers[self.__current_layer].activate(self.__bus)
        return self.__current_layer

    def prev_layer(self):
        """Switch to the previous layer."""
        self.__current_layer = max(0, self.__current_layer - 1)
        self.__layers[self.__current_layer].selection().all()
        if self.__bus:
            self.__layers[self.__current_layer].activate(self.__bus)
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
            if self.__bus:
                self.__layers[self.__current_layer].activate(self.__bus)

        if pos is not None:
            return self.__layers[pos]

        return self.__layers[self.__current_layer]

    def layer_no(self, layer_no = None):
        """Get or set the current layer number."""
        if layer_no is None:
            return self.__current_layer

        layer_no = max(0, layer_no)

        if layer_no >= len(self.__layers):
            self.__layers.append(Layer())

        self.__current_layer = layer_no
        if self.__bus:
            self.__layers[self.__current_layer].activate(self.__bus)
        return self.__current_layer

    def delete_layer_cmd(self):
        """Delete the current layer."""
        # the "logic", if you can call it thusly, is as follows.
        # the delete_layer is actually called by the DeleteLayerCommand.
        # Therefore, we need a wrapper around DeleteLayerCommand that does
        # not actually delete the layer.

        cmd = DeleteLayerCommand(self, self.layer_no())
        self.__bus.emit("history_append", True, cmd)

    def delete_layer(self, layer_no = None):
        """Delete the current layer."""
        log.debug("deleting layer %s", layer_no)

        if len(self.__layers) == 1:
            return None, None

        if layer_no is None or layer_no < 0 or layer_no >= len(self.__layers):
            layer_no = self.__current_layer

        layer = self.__layers[layer_no]
        pos   = layer_no

        del self.__layers[layer_no]

        # make sure layer is within boundaries
        self.__current_layer = max(0, layer_no - 1)
        self.__current_layer = min(self.__current_layer,
                                   len(self.__layers) - 1)

        if self.__bus:
            self.__layers[self.__current_layer].activate(self.__bus)
        return layer, pos

    def export(self):
        """Exports the page with all layers as a dict"""
        layers = [ l.export() for l in self.__layers ]
        ret = {
                 "layers": layers,
                 "view": self.trafo().trafos(),
                 "cur_layer": self.__current_layer
              }
        return ret

    def import_page(self, page_dict):
        """Imports a dict to self"""
        log.debug("importing pages")

        if "objects" in page_dict:
            self.layer().objects(page_dict["objects"])
        elif "layers" in page_dict:
            log.debug('%s layers found', len(page_dict["layers"]))
            log.debug("however, we only have %s layers", len(self.__layers))
            log.debug('creating %s new layers', len(page_dict["layers"]) - len(self.__layers))
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

        # import the transformations
        trafo = page_dict.get("view")
        if trafo:
            self.trafo().trafos(trafo)

        # backwards compatibility
        translate = page_dict.get("translate")
        if translate:
            self.trafo().add_trafo(("move", translate))

        if self.__bus:
            self.__layers[self.__current_layer].activate(self.__bus)

    def clear(self):
        """
        Remove all objects from all layers.

        Returns a CommandGroup object that can be used to undo the
        operation.
        """
        ret_commands = []
        for layer in self.__layers:
            cmd = RemoveCommand(layer.objects()[:], layer.objects())
            ret_commands.append(cmd)
        cmd = CommandGroup(ret_commands[::-1])
        self.__bus.emit("history_append", True, cmd)

    def draw(self, cr, state, force_redraw = False):
        """Draw the objects on the page."""

        self.__drawer.draw(cr, self, state, force_redraw)
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
    def __init__(self, bus, state):
        self.__state = state
        self.__grid = Grid()
        self.__bus = bus
        self.__force_redraw = False
        self.__bus.on("force_redraw", self.force_redraw)
        self.__winsize = (0, 0)

    def force_redraw(self):
        """Set the marker to refresh the cache."""
        self.__force_redraw = True

    def on_draw(self, _, cr):
        """Main draw method of the whole app."""

        if self.__state.graphics().hidden():
            return False

        page = self.__state.current_page()

        cr.save()
        page.trafo().transform_context(cr)
        self.draw_bg(cr, (0, 0))
        cr.restore()

        page.draw(cr, self.__state, force_redraw = self.__force_redraw)

        # emit the draw signal for objects that wish to be drawn in draw
        # coordinates
        cr.save()
        page.trafo().transform_context(cr)
        self.__bus.emit("obj_draw", exclusive = False, cr = cr, state = self.__state)

        cobj = self.__state.current_obj()

        if cobj and not cobj in page.objects_all_layers():
            self.__state.current_obj().draw(cr)

        cr.restore()

        ws = self.__state.get_win_size()

        if ws != self.__winsize:
            self.__winsize = ws
            self.__bus.emit_mult("update_win_size", width = ws[0], height = ws[1])

        self.__bus.emit("draw", exclusive = False, cr = cr, state = self.__state)
        self.__force_redraw = False
        return False

    def draw_bg(self, cr, tr):
        """
        Draw the objects on the page.

        :param objects: The objects to draw.
        :param cr: The context on which to draw.
        :param tr: The translation (paning).
        """

        outline      = self.__state.graphics().outline()
        bg_color     = self.__state.graphics().bg_color()
        transparency = self.__state.graphics().alpha()
        show_grid    = self.__state.graphics().show_grid()
        size         = self.__state.get_win_size()

        if not outline:
            cr.set_source_rgba(*bg_color, transparency)
        else:
            cr.set_source_rgba(0.4, 0.4, 0.4, transparency)

        cr.set_operator(cairo.OPERATOR_SOURCE)
        cr.paint()
        cr.set_operator(cairo.OPERATOR_OVER)

        if show_grid:
            tr = tr or (0, 0)
            self.__grid.draw(cr, tr, size)
"""Class for different brushes."""


# ----------- Brush factory ----------------

class BrushFactory:
    """
    Factory class for creating brushes.
    """
    @classmethod
    def all_brushes(cls, **kwargs):
        """
        Create dict of all brushes.
        """

        brushes = {
                "rounded": BrushRound(**kwargs),
                "slanted": BrushSlanted(**kwargs),
                "marker":  BrushMarker(**kwargs),
                "pencil":  BrushPencil(**kwargs),
                "tapered": BrushTapered(**kwargs),
                }

        return brushes

    @classmethod
    def create_brush(cls, brush_type = "marker", **kwargs):
        """
        Create a brush of the specified type.
        """

        if not brush_type:
            brush_type = "rounded"

        log.debug("Selecting BRUSH %s, kwargs %s", brush_type, kwargs)

        brushes = {
                "rounded": BrushRound,
                "slanted": BrushSlanted,
                "marker":  BrushMarker,
                "pencil":  BrushPencil,
                "tapered": BrushTapered,
                "simple":  BrushSimple,
                }

        if brush_type in brushes:
            return brushes[brush_type](**kwargs)

        raise NotImplementedError("Brush type", brush_type, "not implemented")

# ----------- Brushes ----------------

class Brush:
    """Base class for brushes."""
    def __init__(self, rounded = False, brush_type = "marker", outline = None, smooth_path = True):
        self.__outline = outline or [ ]
        self.__coords = [ ]
        self.__pressure = [ ]
        self.__rounded = rounded
        self.__outline = [ ]
        self.__coords = [ ]
        self.__brush_type = brush_type
        self.__smooth_path = smooth_path
        self.__bbox = None

    def to_dict(self):
        """Return a dictionary representation of the brush."""
        return {
                "brush_type": self.__brush_type,
                "smooth_path": self.smooth_path(),
               }

    def bbox_move(self, dx, dy):
        """Move the bbox by dx, xy"""
        if self.__bbox is None:
            self.bbox(force = True)
            return

        x, y, w, h = self.__bbox
        self.__bbox = (x + dx, y + dy, w, h)

    def bbox(self, force = False):
        """Get bounding box of the brush."""

        if self.__outline is None or len(self.__outline) < 4:
            return None
        if not self.__bbox or force:
            xy_max = np.max(self.__outline, axis = 0)
            xy_min = np.min(self.__outline, axis = 0)
            self.__bbox = [xy_min[0], xy_min[1], 
                           xy_max[0] - xy_min[0], 
                           xy_max[1] - xy_min[1]]
        return self.__bbox

    def brush_type(self):
        """Get brush type."""
        return self.__brush_type

    def smooth_path(self, smooth_path = None):
        """Set or get smooth path."""
        if smooth_path is not None:
            self.__smooth_path = smooth_path
        return self.__smooth_path

    def coords(self, coords = None):
        """Set or get brush coordinates."""
        if coords is not None:
            self.__coords = coords
        return self.__coords

    def outline(self, new_outline = None):
        """Get brush outline."""
        if new_outline is not None:
            self.__outline = new_outline
        if self.__outline is None or len(self.__outline) < 4:
            return None
        return self.__outline

    def pressure(self, pressure = None):
        """Set or get brush pressure."""
        if pressure is not None:
            self.__pressure = pressure
        return self.__pressure

    def draw(self, cr, outline = False):
        """Draw the brush on the Cairo context."""
        if self.__outline is None or len(self.__outline) < 4:
            return

        cr.move_to(self.__outline[0][0], self.__outline[0][1])

        for point in self.__outline[1:]:
            cr.line_to(point[0], point[1])
        cr.close_path()

        if outline:
            cr.set_source_rgb(0, 1, 1)
            cr.set_line_width(0.1)
            coords = self.coords()

            for i in range(len(coords) - 1):
                cr.move_to(coords[i][0], coords[i][1])
                cr.line_to(coords[i + 1][0], coords[i + 1][1])
                cr.stroke()
                # make a dot at each coord
                cr.arc(coords[i][0], coords[i][1], .8, 0, 2 * 3.14159)
                cr.fill()

    def move(self, dx, dy):
        """Move the outline."""
        self.__outline = [ (x + dx, y + dy) for x, y in self.__outline ]
        self.bbox_move(dx, dy)
        #print("bbox now 1:", self.bbox())
        #self.bbox(force = True)
        #print("bbox now 2:", self.bbox())

    def rotate(self, angle, rot_origin):
        """Rotate the outline."""
        self.__outline = coords_rotate(self.__outline, angle, rot_origin)
        self.bbox(force = True)

    def scale(self, old_bbox, new_bbox):
        """Scale the outline."""
        self.__outline = transform_coords(self.__outline, old_bbox, new_bbox)
        self.bbox(force = True)

    def calc_width(self, pressure, lwd):
        """Calculate the width of the brush."""

        widths = [ lwd * (0.25 + p) for p in pressure ]
        return widths

    def coords_add(self, point, pressure):
        """Add a point to the brush coords."""

    def calculate(self, line_width, coords, pressure = None):
        """Recalculate the outline of the brush."""

        pressure = pressure if pressure is not None else [1] * len(coords)

        if coords is not None and pressure is not None:
            if len(coords) != len(pressure):
                log.warning("Pressure and coords don't match (%d <> %d)",
                    len(coords), len(pressure))
                pressure = None

        nc = len(coords)

        if nc < 2:
            return None

        # we are smoothing only shorter lines to avoid generating too many
        # points
        if self.smooth_path() and nc < 100:
            coords, pressure = smooth_coords(coords, pressure)

        if len(coords) != len(pressure):
            log.warning("Pressure and coords don't match (%d <> %d)",
                    len(coords), len(pressure))

        self.coords(coords)

        widths = self.calc_width(pressure, line_width)
        outline_l, outline_r = calc_normal_outline(coords, widths, self.__rounded)
        outline  = np.vstack((outline_l, outline_r[::-1]))

        self.__outline = outline
        self.bbox(force = True)
        return outline

class BrushMarker(Brush):
    """Marker brush."""
    def __init__(self, outline = None, smooth_path = True):
        super().__init__(rounded = False, brush_type = "marker",
                         outline = outline, smooth_path = smooth_path)

class BrushRound(Brush):
    """Round brush."""
    def __init__(self, outline = None, smooth_path = True):
        super().__init__(rounded = True, brush_type = "rounded",
                         outline = outline, smooth_path = smooth_path)

class BrushTapered(Brush):
    """Tapered brush."""
    def __init__(self, outline = None, smooth_path = True):
        super().__init__(rounded = False, brush_type = "tapered",
                         outline = outline, smooth_path = smooth_path)

    def calculate(self, line_width, coords, pressure = None):
        """Recalculate the outline of the brush."""

        if coords is not None and pressure is not None:
            if len(coords) != len(pressure):
                raise ValueError("Pressure and coords don't match")

        pressure = pressure or [1] * len(coords)

        lwd = line_width

        if len(coords) < 2:
            return None

        n_taper = 5
        if len(coords) < n_taper:
            n_taper = len(coords) - 2

        length_to_taper = calculate_length(coords[:(len(coords) - n_taper)])

        if self.smooth_path():
            coords, pressure = smooth_coords(coords, pressure, 20)

        taper_pos = first_point_after_length(coords, length_to_taper)
        taper_length = calculate_length(coords[taper_pos:])

        outline_l, outline_r = calc_normal_outline_tapered(coords, pressure, lwd,
                                                    taper_pos, taper_length)

        outline  = outline_l + outline_r[::-1]

        if len(coords) != len(pressure):
            #raise ValueError("Pressure and coords don't match")
            log.warning("Pressure and coords don't match (%d <> %d)",
                    len(coords), len(pressure))
        self.outline(outline)
        self.bbox(force = True)
        return outline

class BrushPencil(Brush):
    """
    Pencil brush, v3.

    This is more or less an experimental pencil.

    This version attempts to draw with the same stroke, but with
    transparency varying depending on pressure. The idea is to create short
    segments with different transparency. Each segment is drawn using a
    gradient, from starting transparency to ending transparency.
    """
    def __init__(self, outline = None, bins = None, smooth_path = True): # pylint: disable=unused-argument
        super().__init__(rounded = True, brush_type = "pencil",
                         outline = outline, smooth_path = smooth_path)
        self.__pressure  = [ ]
        self.__coords = [ ]
        self.__gradients = None
        self.__midpoints = None
        self.__seg_info = None

    def draw(self, cr, outline = False):
        """Draw the brush on the Cairo context."""
        if self.__coords is None or len(self.__coords) < 2:
            return

        if len(self.__pressure) != len(self.__coords):
            log.warning("Pressure and coords don't match (%d <> %d)",
                    len(self.__pressure), len(self.__coords))
            return

        if outline:
            cr.set_line_width(0.4)
            cr.set_source_rgba(0, 1, 1, 1)

        mp    = self.__midpoints
        segs  = self.outline()
        sinfo = self.__seg_info

        rgba = get_current_color_and_alpha(cr)

        for seg_i in range(len(sinfo)):

            seg_pos = sinfo[seg_i, 0]
            seg_len = sinfo[seg_i, 1]
            cr.move_to(segs[seg_pos][0], segs[seg_pos][1])

            for i in range(seg_pos + 1, seg_pos + seg_len):
                cr.line_to(segs[i][0], segs[i][1])
            cr.close_path()

            if outline:
                cr.stroke()
            else:
                gr = self.get_gradient(rgba, mp[seg_i])
                cr.set_source(gr)
                cr.fill()

        if outline:
            self.draw_outline(cr)

    def get_gradient(self, c, info):
        """Get a gradient for the brush segment."""

        gr = cairo.LinearGradient(info[0], info[1],
                                  info[2], info[3])
        gr.add_color_stop_rgba(0, c[0], c[1], c[2], c[3] * info[4])
        gr.add_color_stop_rgba(1, c[0], c[1], c[2], c[3] * info[5])
        return gr

    def draw_outline(self, cr):
        """Draw the outline of the brush."""

        mp    = self.__midpoints
        segs  = self.outline()
        sinfo = self.__seg_info

        cr.set_source_rgba(0, 1, 1, 1)
        cr.set_line_width(0.04)
        #cr.stroke()

        # segment points
        for seg_i in range(len(sinfo)):
            seg_pos = sinfo[seg_i, 0]
            seg_len = sinfo[seg_i, 1]
            cr.move_to(segs[seg_pos][0], segs[seg_pos][1])

            for i in range(seg_pos, seg_pos + seg_len):
                #cr.line_to(self.__coords[i, 0], self.__coords[i, 1])
                cr.arc(segs[i][0], segs[i][1], .7, 0, 2 * 3.14159)
                cr.fill()

        # segment midpoints, start and end
        for seg_i in range(len(sinfo)):
            cr.arc(mp[seg_i, 0], mp[seg_i, 1], .7, 0, 2 * 3.14159)
            cr.fill()
            cr.arc(mp[seg_i, 2], mp[seg_i, 3], .7, 0, 2 * 3.14159)
            cr.fill()

    def calculate(self, line_width, coords, pressure = None):
        """Recalculate the outline of the brush."""

        if coords is not None and pressure is not None:
            if len(coords) != len(pressure):
                raise ValueError("Pressure and coords don't match")

        pressure = pressure or [1] * len(coords)

        lwd = line_width
        nc = len(coords)

        if nc < 4:
            return None

        if self.smooth_path() and nc < 125:
            coords, pressure = smooth_coords(coords, pressure)

        self.__pressure  = pressure
        self.__coords    = coords

        widths = np.full(len(coords), lwd * .67)
        segments, self.__seg_info, self.__midpoints = calc_pencil_segments(coords, widths, pressure)

        self.outline(segments)

        self.bbox(force = True)
        return self.outline()

class BrushSlanted(Brush):
    """Slanted brush."""
    def __init__(self, slant = None, smooth_path = True):
        super().__init__(brush_type = "slanted", outline = None, smooth_path = smooth_path)

        self.__slant = slant or [(-0.4, 0.6), (0.3, - 0.6)]

    def to_dict(self):
        """Return a dictionary representation of the brush."""
        return {
                "brush_type": self.brush_type(),
                "smooth_path": self.smooth_path(),
                "slant": self.__slant
                }

    def rotate(self, angle, rot_origin):
        """Rotate the outline."""
        #self.outline(coords_rotate(self.outline(), angle, rot_origin))
        self.__slant = coords_rotate(self.__slant, angle, (0, 0))
        self.bbox(force = True)

    def calculate(self, line_width = 1, coords = None, pressure = None):
        """Recalculate the outline of the brush."""

        if coords is not None and pressure is not None:
            if len(coords) != len(pressure):
                raise ValueError("Pressure and coords don't match")

        if self.smooth_path():
            coords, pressure = smooth_coords(coords, pressure)

        # we need to multiply everything by line_width
        dx0, dy0, dx1, dy1 = [ line_width * x for x in self.__slant[0] + self.__slant[1] ]

        outline_l, outline_r = calculate_slant_outlines(coords, dx0, dy0, dx1, dy1)
        outline = outline_l + outline_r[::-1]
        self.outline(outline)

        #print("done calculating slanted brush")
        self.bbox(force = True)
        return outline

class BrushSimple(Brush):
    """Simplistic brush for testing purposes."""
    def __init__(self, smooth_path = True):
        super().__init__(brush_type = "simple", outline = None, smooth_path = True)

        self.__line_width = 1
        self.__bbox = None

    def catmull_rom_spline(self, p0, p1, p2, p3, t):
        """Calculate a point on the Catmull-Rom spline."""

        t2 = t * t
        t3 = t2 * t

        part1 = 2 * p1
        part2 = -p0 + p2
        part3 = 2 * p0 - 5 * p1 + 4 * p2 - p3
        part4 = -p0 + 3 * p1 - 3 * p2 + p3

        return 0.5 * (part1 + part2 * t + part3 * t2 + part4 * t3)

    def calculate_control_points(self, points):
        """Calculate the control points for a smooth curve using Catmull-Rom splines."""
        n = len(points)
        p0 = points[:-3]
        p1 = points[1:-2]
        p2 = points[2:-1]
        p3 = points[3:]

        t = 0.5  # Tension parameter for Catmull-Rom spline

        c1 = p1 + (p2 - p0) * t / 3
        c2 = p2 - (p3 - p1) * t / 3

        return np.squeeze(c1), np.squeeze(c2)

    def draw (self, cr, outline = False):
        """Draw a smooth spline through the given points using Catmull-Rom splines."""
        points = self.outline()

        if len(points) < 4 or outline:
            self.draw_simple(cr, outline)

        points = np.array(points)

        # Move to the first point
        cr.move_to(points[0][0], points[0][1])

        for i in range(1, len(points) - 2):
            p1 = points[i]
            p2 = points[i + 1]
            c1, c2 = self.calculate_control_points(points[i-1:i+3])

            # Draw the cubic Bezier curve using the calculated control points
            cr.curve_to(c1[0], c1[1], c2[0], c2[1], p2[0], p2[1])

        # Draw the path
        cr.set_line_width(self.__line_width / 3)
        cr.stroke()


    def draw_simple(self, cr, outline = False):
        """Draw the brush on the Cairo context."""

        coords = self.outline()
        cr.set_line_width(self.__line_width / 3)

        cr.move_to(coords[0][0], coords[0][1])

        for point in coords[1:]:
            cr.line_to(point[0], point[1])

        cr.stroke()

    def bbox(self, force = False):
        """Get bounding box of the brush."""

        outline = self.outline()

        log.debug("bbox brush simple")
        if not outline:
            log.debug("no outline, returning None")
            return None
        if not self.__bbox or force:
            log.debug("recalculating bbox")
            self.__bbox = path_bbox(outline,
                                    lw = self.__line_width)
        return self.__bbox

    def calculate(self, line_width = 1, coords = None, pressure = None):
        """Recalculate the outline of the brush."""
        log.debug("calculate calling")
        coords, _ = smooth_coords(coords)
        self.__line_width = line_width

        self.outline(coords)
        self.bbox(force = True)
        return coords
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

    def __move_right_word(self):
        """Move the caret to the right."""
        log.debug("moving right one word")
        if self.__caret_pos == len(self.__cont[self.__line]):
            if self.__line < len(self.__cont) - 1:
                self.__line += 1
                self.__caret_pos = 0
            else:
                return

        line = self.__cont[self.__line]
        while self.__caret_pos < len(line) and line[self.__caret_pos].isspace():
            self.__caret_pos += 1

        while self.__caret_pos < len(line) and not line[self.__caret_pos].isspace():
            self.__caret_pos += 1

    def __move_left_word(self):
        """Move the caret to the left."""
        log.debug("moving left one word")
        if self.__caret_pos == 0:
            if self.__line > 0:
                self.__line -= 1
                self.__caret_pos = len(self.__cont[self.__line])
            else:
                return
        while self.__caret_pos > 0 and self.__cont[self.__line][self.__caret_pos - 1].isspace():
            self.__caret_pos -= 1
        while self.__caret_pos > 0 and not self.__cont[self.__line][self.__caret_pos - 1].isspace():
            self.__caret_pos -= 1

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
          "Ctrl-Right": self.__move_right_word,
          "Ctrl-Left":  self.__move_left_word,
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

    def set_text(self, text):
        """Set the text."""

        lines = text.split("\n")
        self.__cont = lines
        self.__line = len(lines) - 1
        self.__caret_pos = len(lines[-1])

    def update_by_key(self, keyname, char):
        """Update the text by key press."""
        if keyname == "BackSpace": # and cur["caret_pos"] > 0:
            self.__backspace()
        elif keyname in ["Home", "End", "Down", "Up", "Right", "Left", "Ctrl-Left", "Ctrl-Right"]:
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
    def __init__(self, pixbuf, base64_enc):

        if base64_enc:
            self.__base64 = base64_enc
            pixbuf = base64_to_pixbuf(base64_enc)
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
from os import path



def object_create_copies(objects, move = False, bb = None):
    """Create copies of given objects, possibly shifted"""

    new_objects = [ ]

    for obj in objects:
        new_obj = obj.duplicate()
        new_objects.append(new_obj)

        # move the new object to the current location
        if move:
            x, y = move
            if bb is None:
                bb  = new_obj.bbox()
            new_obj.move(x - bb[0], y - bb[1])

    return new_objects


class StateGraphics:
    """
    Base class for holding key app graphics state information.

    Essentially options regarding whether to show UI elements, background
    etc.
    """

    def __init__(self):

        self.__gr = {
                "mode": "draw",             # drawing mode
                "modified": True,           # modified flag
                "bg_color": (.8, .75, .65), # bg color
                "transparency": 0,          # bg alpha
                "outline": False,           # outline mode
                "hidden": False,            # hide drawing
                                            # for screenshots
                "grid": False,              # show grid
                "wiglets": True,            # show wiglets
                "win_size": (100, 100),
                }

    def win_size(self, width = None, height = None):
        """Get or set the window size"""

        if not (width is None or height is None):
            self.__gr["win_size"] = (width, height)

        return self.__gr["win_size"]

    def bg_color(self, color=None):
        """Get or set the background color."""
        if color:
            self.__gr["bg_color"] = color
        return self.__gr["bg_color"]

    def cycle_background(self):
        """Cycle through background transparency."""
        self.__gr["transparency"] = {1: 0, 0: 0.5, 0.5: 1}[self.__gr["transparency"]]

    def alpha(self, value=None):
        """Get or set the bg transparency."""
        if value:
            self.__gr["transparency"] = value
        return self.__gr["transparency"]

    def outline(self, value = None):
        """Get the outline mode."""
        if value is not None:
            self.__gr["outline"] = value
        return self.__gr["outline"]

    def show_grid(self):
        """What is the show grid status."""
        return self.__gr["grid"]

    def toggle_grid(self):
        """Toggle the grid."""
        self.__gr["grid"] = not self.__gr["grid"]

    def show_wiglets(self, value = None):
        """Show or hide the wiglets."""
        if value is not None:
            self.__gr["wiglets"] = value
        return self.__gr["wiglets"]

    def toggle_wiglets(self):
        """Toggle the wiglets."""
        self.__gr["wiglets"] = not self.__gr["wiglets"]

    def hidden(self, value = None):
        """Hide or show the drawing."""
        if not value is None:
            self.__gr["hidden"] = value
        return self.__gr["hidden"]

    def mode(self, mode = None):
        """Get or set the cursor mode."""
        if mode:
            if mode == self.__gr["mode"]:
                return mode
            log.debug("setting mode to %s", mode)
            self.__gr["mode"] = mode
        return self.__gr["mode"]

    def modified(self, value = None):
        """Get or set the modified flag."""
        if value is not None:
            self.__gr["modified"] = value
        return self.__gr["modified"]

class StateConfig:
    """import export dirs"""

    def __init__(self):
        self.__vars = {
                "savefile": None,
                "cur_dir": None,
                "import_dir": None,
                "export_dir": None,
                "export_fn": None,
                }

    def savefile(self, name = None):
        """Get or set the savefile."""
        if name:
            self.__vars["savefile"] = name
        return self.__vars["savefile"]

    def cur_dir(self, name = None):
        """Get or set the current directory."""
        if name:
            self.__vars["cur_dir"] = name
        return self.__vars["cur_dir"]

    def import_dir(self, name = None):
        """Get or set the import directory."""
        if name:
            self.__vars["import_dir"] = name
        return self.__vars["import_dir"]

    def export_dir(self, name = None):
        """Get or set the export directory."""
        if name:
            self.__vars["export_dir"] = name
        return self.__vars["export_dir"]

    def export_fn(self, name = None):
        """Get or set the export file name."""
        if name:
            self.__vars["export_fn"] = name
        return self.__vars["export_fn"]


class StateRoot:
    """Base class for holding key app information."""

    def __init__(self):
        self.__gr_state = StateGraphics()
        self.__config   = StateConfig()

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
    def graphics(self):
        """Return the graphics state."""
        return self.__gr_state

    def config(self):
        """Return the config state."""
        return self.__config

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
        self.__gr_state.bg_color(self.__pens[0].color)

    def set_font(self, font_description):
        """Set the font."""
        self.pen().font_set_from_description(font_description)

        obj = self.current_obj()
        if obj and obj.type == "text":
            obj.pen.font_set_from_description(font_description)

    def set_brush(self, brush = None):
        """Set the brush."""
        if brush is not None:
            log.debug("setting pen %s brush to {brush}", self.pen())
            self.pen().brush_type(brush)
        return self.pen().brush_type()

    def set_color(self, color = None):
        """Get or set the pen color."""

        if color is not None:
            log.debug("Setting color to %s", color)
            self.pen().color_set(color)

        return self.pen().color

    def set_transparency(self, transparency = None):
        """Set the line width."""

        if transparency is not None:
            self.pen().transparency = transparency

        return self.pen().line_width

    def set_line_width(self, width = None):
        """Set the line width."""

        if width is not None:
            log.debug("Setting line width to %s", width)
            self.pen().line_width = width

        return self.pen().line_width

    def stroke_change(self, direction):
        """Modify the line width or text size."""
        log.debug("Changing stroke %s", direction)
        cobj = self.current_obj()
        # without a selected object, change the default pen, but only if in the correct mode
        if self.graphics().mode() == "draw":
            self.pen().line_width = max(1, self.pen().line_width + direction)
        elif self.graphics().mode() == "text":
            self.pen().font_size = max(1, self.pen().font_size + direction)

        if cobj and cobj.type == "text":
            log.debug("Changing text size")
            cobj.stroke_change(direction)
            self.pen().font_size = cobj.pen.font_size
            return True

        return False


# -----------------------------------------------------------------------------
class StateObj(StateRoot):

    """
    Adds big object handling to state
    """

    def __init__(self, app, bus):

        super().__init__()
        self.__bus      = bus

        history = History(bus)
        gom = GraphicsObjectManager(self.__bus)
        cursor = CursorManager(app, bus)

        self.__obj = {
                "gom": gom,
                "app": app,
                "cursor": cursor,
                "history": history,
                "clipboard": None,
                "mouse": None,
                }

    def mouse(self, mouse = None):
        """Return the mouse object."""
        if mouse:
            self.__obj["mouse"] = mouse
        return self.__obj["mouse"]

    def cursor(self):
        """expose cursor"""
        return self.__obj["cursor"]

    def clipboard(self, clipboard = None):
        """Return the clipboard."""
        if clipboard:
            self.__obj["clipboard"] = clipboard
        return self.__obj["clipboard"]

   #def cursor_pos(self):
   #    """Return the cursor position."""
   #    return self.__obj["cursor"].pos()

    def history(self):
        """Return the history."""
        return self.__obj["history"]

    def app(self):
        """Return the app."""
        return self.__obj["app"]

    def bus(self):
        """Return the bus."""
        return self.__bus

    def gom(self):
        """Return GOM"""
        return self.__obj["gom"]

    def current_page(self):
        """Get the current page object from gom."""
        return self.gom().page()

    def page(self):
        """Current page"""
        return self.gom().page()

    def pos_abs_to_rel(self, pos):
        """Convert absolute position to real position."""
        return self.page().pos_abs_to_rel(pos)

    def selection(self):
        """Current selection"""
        return self.gom().selection()

    def selected_objects(self):
        """Return the selected objects."""
        return self.gom().selected_objects()

    def objects(self):
        """Return the objects of the current layer."""
        return self.page().layer().objects()

    def objects_all_layers(self):
        """Return the objects of all layers."""
        return self.page().objects_all_layers()

    def get_win_size(self):
        """Get the window size."""
        return self.__obj["app"].get_size()

    def queue_draw(self):
        """Queue a draw."""
        self.__obj["app"].queue_draw()


# -----------------------------------------------------------------------------
class State(StateObj):
    """
    Class for setting the state.


    The purpose is to pack a bunch of setter methods into a single class
    so that the state class doesn't get too cluttered.
    """
    __new_instance = None

    def __new__(cls, *args, **kwargs):
        if not cls.__new_instance:
            cls.__new_instance = super(State, cls).__new__(cls)
        return cls.__new_instance

    def __init__(self, app, bus):

        super().__init__(app, bus)

        clipboard = Clipboard()
        self.clipboard(clipboard)
        self.__init_signals()

    def __init_signals(self):
        """Initialize the signals."""

        bus_signals = {
            "queue_draw": {"listener": self.queue_draw},
            "mode_set": {"listener": self.mode, "priority": 999},
            "update_win_size": { "listener": self.graphics().win_size},

            "cycle_bg_transparency": {"listener": self.graphics().cycle_background, "priority": 0},
            "set_bg_color": {"listener": self.graphics().bg_color, "priority": 0},
            "toggle_wiglets": {"listener": self.graphics().toggle_wiglets, "priority": 0},
            "toggle_grid": {"listener": self.graphics().toggle_grid, "priority": 0},

            "set_savefile": {"listener": self.config().savefile, "priority": 0},
            "set_export_dir": {"listener": self.config().export_dir, "priority": 0},
            "set_import_dir": {"listener": self.config().import_dir, "priority": 0},
            "set_export_fn": {"listener": self.config().export_fn, "priority": 0},

            "switch_pens": {"listener": self.switch_pens, "priority": 0},
            "apply_pen_to_bg": {"listener": self.apply_pen_to_bg, "priority": 0},
            "clear_page": {"listener": self.current_obj_clear, "priority": 0},

            "set_color": {"listener": self.set_color},
            "set_brush": {"listener": self.set_brush},
            "set_font": {"listener": self.set_font},
            "set_line_width": {"listener": self.set_line_width},
            "set_transparency": {"listener": self.set_transparency},
            "toggle_outline": {"listener": self.toggle_outline},
            "toggle_hide": {"listener": self.toggle_hide},
            "stroke_change": {"listener": self.stroke_change, "priority": 90},
            "query_cursor_pos": {"listener": self.get_cursor_pos},

            "copy_content": {"listener": self.copy_content},
            "cut_content": {"listener": self.cut_content},
            "duplicate_content": {"listener": self.duplicate_content},
            "paste_content": {"listener": self.paste_content},
        }

        bus = self.bus()
        for signal, params in bus_signals.items():
            if params.get("priority"):
                bus.on(signal, params["listener"], priority = params["priority"])
            else:
                bus.on(signal, params["listener"])

    # -------------------------------------------------------------------------
    def cursor_pos(self):
        """Report the screen (user) coordinates"""
        pos_abs = self.cursor_pos_abs()
        return self.page().trafo().apply_reverse([pos_abs])[0]

    def cursor_pos_abs(self):
        """Report the absolute (window) coordinates"""
        pos = self.get_cursor_pos()
        return pos

    def get_cursor_pos(self):
        """Get the cursor position"""
        x, y = get_cursor_position(self.app())
        return (x, y)

    def visible_bbox(self):
        """Get the visible bbox."""
        # Drawing coordinates of the visible screen
        w, h = self.app().get_size()
        p1, p2 = self.page().trafo().apply_reverse([(0, 0), (w, h)])
        return (p1[0], p1[1], p2[0] - p1[0], p2[1] - p1[1])

    def toggle_outline(self):
        """Toggle outline mode."""
        self.graphics().outline(not self.graphics().outline())
        self.bus().emit("force_redraw")

    def toggle_hide(self, hide_state = None):
        """Toggle hide mode."""
        if hide_state is not None:
            self.graphics().hidden(hide_state)
        else:
            self.graphics().hidden(not self.graphics().hidden())

        self.bus().emit("force_redraw")

    def mode(self, mode = None):
        """Get or set the mode."""
        # wrapper, because this is used frequently, and also because
        # graphics state does not hold the cursor object
        if mode is not None and mode != self.graphics().mode():
            self.cursor().default(mode)

        return self.graphics().mode(mode)

    def cut_content(self):
        """Cut content to clipboard."""
        self.copy_content(True)

    def copy_content(self, destroy = False):
        """Copy content to clipboard."""
        content = self.selection()

        if content.is_empty():
            return

        log.debug("Copying content %s", content)
        self.clipboard().copy_content(content, cut = destroy)

        if destroy:
            self.gom().remove_selection()

    def __paste_text(self, clip_text):
        """Enter some text in the current object or create a new object."""

        obj = Text([ self.cursor_pos() ],
                        pen = self.pen(), content=clip_text.strip())
        return [ obj ]

    def __paste_image(self, clip_img):
        """Create an image object from a pixbuf image."""
        obj = Image([ self.cursor_pos() ], self.pen(), clip_img)
        return [ obj ]

    def __paste_internal(self, clip):
        """Paste internal content."""
        log.debug("Pasting internal content")

        if clip.type != "clipboard_group":
            raise ValueError("Internal clipboard is not a clipboard_group")

        bb = clip.bbox()
        log.debug("clipboard bbox %s", bb)

        if not clip.is_cut():
            move = self.cursor_pos()
        else:
            move = None

        new_objects = object_create_copies(clip.objects, move, bb)

        return new_objects

    def paste_content(self):
        """Paste content from clipboard."""
        clip_type, clip = self.clipboard().get_content()

        if not clip:
            return

        # internal paste
        if clip_type == "internal":
            new_objects = self.__paste_internal(clip)
        elif clip_type == "text":
            new_objects = self.__paste_text(clip)
        elif clip_type == "image":
            new_objects = self.__paste_image(clip)

        if new_objects:
            for obj in new_objects:
                self.bus().emit("add_object", True, obj)
            self.bus().emit("set_selection", True, new_objects)

    def duplicate_content(self):
        """Duplicate the selected content."""
        content = self.selection()

        if content.is_empty():
            return

        for obj in content.objects:
            new_obj = obj.duplicate()
            self.bus().emit_once("add_object", new_obj)
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

    def modified(self, mod=None):
        """Was the object modified?"""
        if mod:
            self.mod += 1
        return self.mod

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
        log.warning("resize_end not implemented")
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

    def stroke(self, lw = None):
        """Set the line width of the object."""
        self.mod += 1
        if lw is not None:
            self.pen.stroke(lw)
        return self.pen.stroke()

    def stroke_change(self, direction):
        """Change the stroke size of the object."""
        self.pen.stroke_change(direction)
        self.mod += 1
        return self.pen.stroke()

    def smoothen(self, threshold=20):
        """Smoothen the object."""
        log.warning("smoothening not implemented (threshold %s)", threshold)
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

    def duplicate(self):
        """Duplicate the object."""
        new_obj = copy.deepcopy(self.to_dict())
        new_obj = self.from_dict(new_obj)

        return new_obj

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
        log.debug("Generating object of type: %s", obj_type)
        if obj_type not in type_map:
            raise ValueError("Invalid type:", obj_type)

        if "pen" in d:
            d["pen"] = Pen.from_dict(d["pen"])
        if "rotation" in d:
            log.warning("rotation keyword obsolete, ignoring")
            d.pop("rotation")
        if "rot_origin" in d:
            log.warning("rot_origin keyword obsolete, ignoring")
            d.pop("rot_origin")

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
                 outline = state.graphics().outline())

def obj_status(obj, selection, state):
    """Calculate the status of an object."""

    hover_obj = state.hover_obj()
    hover    = obj == hover_obj and state.mode() == "move"
    selected = selection.contains(obj) and state.mode() == "move"
    is_cur_obj = obj == state.current_obj()

    return (obj.mod, hover, selected, is_cur_obj)

def create_cache_surface(objects, mask = None, trafo = None):
    """
    Create a cache surface.

    :param objects: The objects to cache.
    """

    if not objects:
        return None

    if mask:
        grp = ClippingGroup(mask, objects)
    else:
        grp = DrawableGroup(objects)

    bb  = grp.bbox(actual = True)

    if not bb:
        return None

    x, y, width, height = bb

    if width <= 0 or height <= 0:
        #log.debug("no bb overlap with mask, skipping")
        #log.debug("clipgroup bbox: %s", grp.bbox(actual = True))
        return None

    if trafo:
        bb = [ (x, y), (x + width, y + height) ]
        bb = trafo.apply(bb)
        x, y, width, height = bb[0][0], bb[0][1], bb[1][0] - bb[0][0], bb[1][1] - bb[0][1]
        #log.debug("surface size: %d x %d", int(width), int(height))

    # create a surface that fits the bounding box of the objects
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
        self.__trafo        = None
        self.__win_size     = None
        self.__mask         = None
        self.__mask_bbox    = None
        self.__obj_bb_cache = { }

    def new_cache(self, groups, selection, state):
        """
        Generate the new cache when the objects have changed.

        :param groups: The groups of objects to cache.
        """

        #log.debug("generating new cache")
        self.__cache = { "groups": groups,
                         "surfaces": [ ],}

        cur = groups["first_is_same"]

        # for each non-empty group of objects that changed
        # generate a cache surface and draw the group on it
        for obj_grp in groups["groups"]:
            if not cur or not obj_grp:
                cur = not cur
                continue

            surface = create_cache_surface(obj_grp, mask = self.__mask, trafo = self.__trafo)
            self.__cache["surfaces"].append(surface)
            if surface:
                cr = surface["cr"]
                self.__trafo.transform_context(cr)
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

       #if self.__cache:
       #    if self.__cache["groups"] != groups:
       #        log.debug("groups have changed!")
       #        log.debug("cached: %s", self.__cache["groups"])
       #        log.debug("new: %s", groups)

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
        bbhash = self.__obj_bb_cache

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
            is_same = is_same and not obj.modified()

            if not is_same or not obj in bbhash:
                bb = obj.bbox(actual = True)
                bbhash[obj] = bbox_is_overlap(bb, self.__mask_bbox)

            if not bbhash[obj]:
                continue

            #log.debug("object of type %s is same: %s (status=%s)", obj.type, is_same, status)

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
        n_cached = 0
        n_groups = 0

        for obj_grp in self.__cache["groups"]["groups"]:
            n_groups += 1

            # ignore empty groups (that might happen in edge cases)
            if not obj_grp:
                is_same = not is_same
                continue

            # objects in this group have changed: draw it normally on the surface
            if not is_same:
                cr.save()
                self.__trafo.transform_context(cr)
                draw_on_surface(cr, obj_grp, selection, state)
                cr.restore()
                is_same = not is_same
                continue

            # objects in this group remained the same: draw the cached surface
            if is_same:
                #print("Drawing cached surface")
                surface = self.__cache["surfaces"][i]
                if surface:
                    cr.set_source_surface(surface["surface"], surface["x"], surface["y"])
                    cr.paint()
                i += 1
                is_same = not is_same
                n_cached += 1
        cr.save()
       #self.__trafo.transform_context(cr)
       #self.__mask.draw(cr)
       #cr.restore()
       #log.debug("Cached %d groups out of %d", n_cached, n_groups)

    def draw(self, cr, page, state, force_redraw=False):
        """
        Draw the objects on the page.

        :param objects: The objects to draw.
        :param cr: The context on which to draw.
        :param page: The page object.
        :param state: The state object.
        """

        if force_redraw:
            log.debug("Forced redraw")
            self.__cache = None
            self.__obj_mod_hash = { }
            self.__obj_bb_cache = { }

        self.__win_size = state.graphics().win_size()
        self.__trafo = page.trafo()
        wrelpos = self.__trafo.apply_reverse([(0, 0),
                                              self.__win_size])
        x0, y0 = wrelpos[0]
        x1, y1 = wrelpos[1]
        self.__mask_bbox = [ x0, y0, x1 - x0, y1 - y0 ]
        self.__mask = Rectangle([ (x0, y0), (x1, y0),
                                 (x1, y1), (x0, y1),
                                 (x0, y0) ],
                                pen = Pen(color = (0, 1, 1)))
        #log.debug("wsize: %s wrelpos: %s",
                  #self.__win_size,
                  #wrelpos)


        #log.debug("Drawing objects on the page, force_redraw=%s", force_redraw)
        # extract objects from the page provided
        objects = page.objects_all_layers()

        # extract the selection object that we need
        # to determine whether an object in selected state
        selection = page.layer().selection()

        # check if the cache needs to be updated
        self.update_cache(objects, selection, state)


        # draw the cache
        cr.save()
        #page.trafo().transform_context(cr)
        self.draw_cache(cr, selection, state)
        cr.restore()

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

        log.debug("create object in mode %s", mode)
        #if mode == "text" or (mode == "draw" and shift_click and no_active_area):

        if mode == "text":
            log.debug("creating text object")
            ret_obj = Text([ pos ], pen = pen, content = "")
            ret_obj.move_caret("Home")

        elif mode == "draw":
            log.debug("creating path object")
            ret_obj = Path([ pos ], pen = pen, pressure = [ pressure ])

        elif mode == "segmented_path":
            log.debug("creating segmented path object")
            ret_obj = SegmentedPath([ pos ], pen = pen, pressure = [ pressure ])

        elif mode == "rectangle":
            log.debug("creating rectangle object")
            ret_obj = Rectangle([ pos ], pen = pen)

        elif mode == "shape":
            log.debug("creating shape object")
            ret_obj = Shape([ pos ], pen = pen)

        elif mode == "circle":
            log.debug("creating circle object")
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
        log.debug("transmuting object to %s", mode)

        if obj.type == "group":
            # for now, we do not pass transmutations to groups, because
            # we then cannot track the changes.
            return obj

        if mode == "draw":
            obj = Path.from_object(obj)
        elif mode == "rectangle":
            obj = Rectangle.from_object(obj)
        elif mode == "shape":
            log.debug("calling Shape.from_object")
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

        super().__init__(mytype, [ (None, None) ], None)
        self.objects = objects
        self.__modif = None

    def modified(self, mod=False):
        """Was the object modified?"""
        ret = False

        for obj in self.objects:
            ret = obj.modified(mod) or ret

        if mod:
            self.mod += 1

        status = ret or self.mod != self.__modif
        self.__modif = self.mod

        return status

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
        log.debug("transmuting group to %s", mode)
       #for i in range(len(self.objects)):
       #    self.objects[i] = DrawableFactory.transmute(self.objects[i], mode)
        self.mod += 1

    def to_dict(self):
        """Convert the group to a dictionary."""
        return {
            "type": self.type,
            "objects_dict": [ obj.to_dict() for obj in self.objects ],
        }

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

    def resize_start(self, corner, origin):
        """Start the resizing operation."""
        self.resizing = {
            "corner": corner,
            "origin": origin,
            "bbox":   self.bbox(actual = True),
            "orig_bbox": self.bbox(actual = True),
            "objects": { obj: obj.bbox(actual = True) for obj in self.objects }
            }

        for obj in self.objects:
            obj.resize_start(corner, origin)
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

        return objects_bbox(self.objects)

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
            obj.draw(cr, hover=False, selected=selected,
                     outline=outline)

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

class ClippingGroup(DrawableGroup):
    """
    Class for creating groups of drawable objects that are clipped.
    """
    def __init__(self, clip_obj=None, objects=None, objects_dict = None):
        if objects_dict:
            clip_obj = Drawable.from_dict(objects_dict[0])
            objects  = Drawable.from_dict(objects_dict[1]).objects

        self.__clip_obj  = clip_obj
        self.__obj_group = DrawableGroup(objects)
        objlist = objects[:]
        objlist.append(clip_obj)

        super().__init__(objlist,
                         mytype = "clipping_group")

    def draw_clip(self, cr):
        """Draw the clipping path."""
        if self.rotation != 0:
            cr.save()
            cr.translate(self.rot_origin[0], self.rot_origin[1])
            cr.rotate(self.rotation)
            cr.translate(-self.rot_origin[0], -self.rot_origin[1])

        co = self.__clip_obj
        coords = co.coords
        res_bb = self.resizing and self.resizing["bbox"] or None
        if res_bb:
            old_bbox = path_bbox(coords)
            coords = transform_coords(coords, old_bbox, res_bb)

        cr.move_to(coords[0][0], coords[0][1])
        for point in coords[1:]:
            cr.line_to(point[0], point[1])
        cr.close_path()
        if self.rotation != 0:
            cr.restore()

    def draw(self, cr, hover=False, selected=False, outline=False):
        cr.save()
        #self.__clip_obj.draw(cr,  hover=False, selected=selected, outline=True)
        #self.__obj_group.draw(cr, hover=False, selected=selected, outline=True)

        self.draw_clip(cr)

        cr.clip()
        self.__obj_group.draw(cr, hover=False,
                              selected=selected, outline = outline)
        if self.rotation:
            x, y = self.rot_origin[0], self.rot_origin[1]
            cr.translate(x, y)
            cr.rotate(self.rotation)
            cr.translate(-x, -y)

        cr.restore()

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

    def bbox_draw(self, cr, lw=0.2):
        """Draw the bounding box of the object."""
        bb = self.bbox(actual = True)
        x, y, w, h = bb
        cr.set_line_width(lw)
        cr.rectangle(x, y, w, h)
        cr.stroke()

    def bbox(self, actual = False):
        """Return the bounding box of the group."""
        if not actual:
            return objects_bbox(self.objects)
        return bbox_overlap(
            self.__clip_obj.bbox(),
            self.__obj_group.bbox())

    def is_close_to_click(self, click_x, click_y, threshold):
        """Check if a click is close to one of the objects."""

        bb = self.bbox(actual = True)
        return (bb[0] - threshold < click_x < bb[0] + bb[2] + threshold and
                bb[1] - threshold < click_y < bb[1] + bb[3] + threshold)

    def to_dict(self):
        """Convert the group to a dictionary."""
        log.debug("my type is %s", self.type)
        return {
            "type": self.type,
            "objects_dict": [ self.__clip_obj.to_dict(), self.__obj_group.to_dict() ]
        }


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

        log.debug("Selection Object with %s objects", len(all_objects))
        self._all_objects = all_objects

    def copy(self):
        """Return a copy of the selection object."""
        # the copy can be used for undo operations
        log.debug("copying selection to a new selection object")
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
        log.debug("setting selection to %s", objects)
        self.objects = objects

    def add(self, obj):
        """Add an object to the selection."""
        log.debug("adding object to selection: %s selection is %s", obj, self.objects)
        if not obj in self.objects:
            self.objects.append(obj)

    def all(self):
        """Select all objects."""
        log.debug("selecting everything")
        self.objects = self._all_objects[:]
        log.debug("selection has now %s objects", len(self.objects))
        log.debug("all objects have %s objects", len(self._all_objects))

    def next(self):
        """
        Return a selection object with the next object in the list,
        relative to the current selection.
        """

        log.debug("selecting next object")
        all_objects = self._all_objects

        if not all_objects:
            return

        if not self.objects:
            log.debug("no selection yet, selecting first object: %s", all_objects[0])
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
            log.debug("no selection yet, selecting everything")
            self.objects = self._all_objects[:]
            return

        new_sel = [ ]
        for obj in self._all_objects:
            if not self.contains(obj):
                new_sel.append(obj)

        self.objects = new_sel

class ClipboardGroup(DrawableGroup):
    """Basically same as drawable group, but for copy and paste operations."""

    def __init__(self, objects=None, cut=False):
        super().__init__(objects, mytype = "clipboard_group")

        self.__cut = cut

    def is_cut(self):
        """Is the clipboard group a copy?"""
        return self.__cut

Drawable.register_type("group", DrawableGroup)
Drawable.register_type("clipping_group", ClippingGroup)
"""
These classes are the primitives for drawing: text, shapes, paths
"""



class DrawableTrafo(Drawable):
    """
    Class for objects that are transformed using cairo transformations.

    Rather than recalculating the coordinates upon tranformation, we record the transformations
    and apply them when drawing.

    This has some advantages, but would mess with the pens in case of
    paths, and might be less efficient than recalculating the coordinates
    once and for all.

    The coordinates here hold the original bounding box of the object, with
    coords[0] being the left upper, and coords[1] the right lower corner.

    This information is only used for calculating the bounding box of the
    transformed object.
    """

    def __init__(self, mytype, coords, pen, transform = None):

        log.debug("initializing DrawableTrafo %s, %s, %s", mytype, coords, transform)
        super().__init__(mytype, coords, pen)
        self.__bbox = None

        self.__trafo = Trafo(transform)
        self.bbox_recalculate()

    def draw(self, cr, hover=False, selected=False, outline=False):
        """Draw the object on the Cairo context."""
        raise NotImplementedError("draw method not implemented")

    def trafo(self):
        """Return the transformations."""
        return self.__trafo

    def apply_trafos(self, cr):
        """Apply the transformations to the Cairo context."""

        self.__trafo.transform_context(cr)

    def move(self, dx, dy):
        """Move the image by dx, dy."""
        self.__trafo.add_trafo(("move", (dx, dy)))
        self.bbox_recalculate()

    def bbox_recalculate(self, mod = True):
        """Return the bounding box of the object."""
        log.debug("recalculating bbox of %s", self.type)
        if mod:
            self.mod += 1
        coords = self.coords
        x0, y0 = coords[0]
        x1, y1 = coords[1]
        coords = [ (x0, y0), (x1, y0), (x1, y1), (x0, y1) ]

        coords = self.__trafo.apply(coords)

        # get the bounding box
        self.__bbox = path_bbox(coords)
        return self.__bbox

    def bbox(self, actual = False):
        """Return the bounding box of the object."""
        return self.__bbox

    def resize_start(self, corner, origin):
        """Start the resizing operation."""
        self.resizing = {
            "corner": corner,
            "origin": origin,
            "orig_bbox": self.bbox(),
            "bbox":   self.bbox(),
            }
        self.mod += 1
        # dummy transformation x 2
        # needed because resize_update first removes the previous two
        self.__trafo.add_trafo(("dummy", (0, 0, 1, 1)))
        self.__trafo.add_trafo(("dummy", (0, 0, 1, 1)))

    def resize_update(self, bbox):
        """Update during the resize of the object."""

        # remove the 2 last transformations
        # (the resizing might involve a move!)
        self.__trafo.pop_trafo()
        self.__trafo.pop_trafo()

        # this is the new bounding box
        x1, y1, w1, h1 = bbox

        # this is the old bounding box
        x0, y0, w0, h0 = self.resizing["orig_bbox"]

        # calculate the new transformation
        w_scale = w1 / w0
        h_scale = h1 / h0

        # apply the transformation. No merging, because they are temporary
        self.__trafo.add_trafo(("move", (x1 - x0, y1 - y0)), merge = False)
        self.__trafo.add_trafo(("resize", (x1, y1, w_scale, h_scale)), merge = False)
        self.bbox_recalculate()

    def resize_end(self):
        """Finish the resizing operation."""
        self.resizing = None
        self.mod += 1

    def rotate(self, angle, set_angle = False):
        """Rotate the object by the specified angle."""
        # if the previous trafo is a rotation, simply add the new angle
        self.__trafo.add_trafo(("rotate", (self.rot_origin[0], self.rot_origin[1], angle)))
        self.bbox_recalculate()

    def rotate_end(self):
        """Finish the rotation operation."""

    def is_close_to_click(self, click_x, click_y, threshold):
        """Check if a click is close to the object."""
        bb = self.bbox()
        if bb is None:
            return False
        x, y, width, height = bb
        if x <= click_x <= x + width and y <= click_y <= y + height:
            return True
        return False

class Image(DrawableTrafo):
    """
    Class for Images
    """
    def __init__(self, coords, pen, image, image_base64 = None, transform = None):

        #log.debug("CREATING IMAGE, pos %s, trafo %s", coords, transform)
        self.__image = ImageObj(image, image_base64)

        width, height = self.__image.size()
        coords = [ (coords[0][0], coords[0][1]),
                   (coords[0][0] + width, coords[0][1] + height) ]

        super().__init__("image", coords, pen, transform)

        self.bbox_recalculate()

    def draw(self, cr, hover=False, selected=False, outline=False):
        """Draw the object on the Cairo context."""
        cr.save()
        self.trafo().transform_context(cr)

        w, h = self.__image.size()
        x, y = self.coords[0]
        cr.rectangle(x, y, w, h)
        cr.clip()
        Gdk.cairo_set_source_pixbuf(cr, self.__image.pixbuf(), x, y)
        cr.paint()

        cr.restore()

        #cr.set_source_rgb(*self.pen.color)
        cr.set_source_rgb(1, 0, 0)
        if selected:
            self.bbox_draw(cr, lw=1.5)
        if hover:
            self.bbox_draw(cr, lw=.5)

    def image(self):
        """Return the image."""
        return self.__image.pixbuf()

    def to_dict(self):
        """Convert the object to a dictionary."""

        log.debug("transformations saved: %s", self.trafo().trafos())
        return {
            "type": self.type,
            "coords": self.coords,
            "pen": self.pen.to_dict(),
            "image": None,
            "image_base64": self.__image.base64(),
            "transform": self.trafo().trafos(),
        }


class Text(DrawableTrafo):
    """Class for Text objects"""
    def __init__(self, coords, pen, content, transform = None):


        # split content by newline
        # content = content.split("\n")
        self.__text = TextEditor(content)
        self.font_extents = None
        self.__show_caret = False

        coords = [ (coords[0][0], coords[0][1]), (50, 50) ]

        super().__init__("text", coords, pen, transform)

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

    def stroke(self, font_size = None):
        """Return the stroke of the text."""
        if font_size is not None:
            self.pen.font_size = font_size
            self.mod += 1
        return self.pen.font_size

    def stroke_change(self, direction):
        """Change text size up or down."""
        self.pen.font_size += direction
        self.pen.font_size = max(8, min(128, self.pen.font_size))
        self.mod += 1
        return self.pen.font_size

    def to_dict(self):
        return {
            "type": self.type,
            "coords": self.coords,
            "pen": self.pen.to_dict(),
            "content": self.__text.to_string(),
            "transform": self.trafo().trafos(),
        }

    def to_string(self):
        """Return the text as a single string."""
        return self.__text.to_string()

    def strlen(self):
        """Return the length of the text."""
        return self.__text.strlen()

    def set_text(self, text):
        """Set the text of the object."""
        self.__text.set_text(text)
        self.mod += 1

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
        dy_top = self.font_extents[0]
        bb = [position[0],
              position[1], # - self.font_extents[0],
              0, 0]

        cr.save()
        self.trafo().transform_context(cr)

        for i, fragment in enumerate(content):

            cr.set_source_rgba(*self.pen.color, self.pen.transparency)
            #x_bearing, y_bearing, t_width, t_height, x_advance, y_advance
            x_bearing, _, t_width, _, _, _ = cr.text_extents(fragment)

            bb[2] = max(bb[2], t_width + x_bearing)
            bb[3] += self.font_extents[2]

            cr.set_font_size(self.pen.font_size)
            cr.move_to(position[0], position[1] + dy + dy_top)
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
                                position[1] + dy,
                                self.font_extents[2])

            dy += self.font_extents[2]

        new_coords = [ (bb[0], bb[1]), (bb[0] + bb[2], bb[1] + bb[3]) ]

        if new_coords != self.coords:
            self.coords = new_coords
            self.bbox_recalculate(mod = False)

        #cr.rectangle(bb[0], bb[1], bb[2], bb[3])
        #cr.stroke()

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
        log.debug("finishing shape")
        self.coords, _ = smooth_coords(self.coords)
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

    def draw_points(self, cr):
        """Draw a dot at each of the coordinate pairs"""
        for x, y in self.coords:
            cr.arc(x, y, 1, 0, 2 * 3.14159)
            cr.fill()

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
            cr.set_source_rgba(0, 1, 1)
            self.draw_simple(cr, res_bb)
            cr.set_line_width(0.5)
            cr.stroke()
            self.draw_points(cr)
        elif self.fill():
            self.draw_simple(cr, res_bb)
            cr.fill()
        else:
            self.draw_simple(cr, res_bb)
            cr.set_line_width(self.pen.line_width)
            cr.stroke()

        if selected:
            cr.set_source_rgba(1, 0, 0)
            self.bbox_draw(cr, lw=.5)

        if hover:
            self.bbox_draw(cr, lw=.3)

        if self.rotation != 0:
            cr.restore()

    @classmethod
    def from_object(cls, obj):
        """Create a shape from an object."""
        log.debug("Shape.from_object %s", obj)
        if obj.coords and len(obj.coords) > 2 and obj.pen:
            return cls(obj.coords, obj.pen)

        # issue a warning
        log.warning("Shape.from_object: invalid object")
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
        log.debug("finishing rectangle")
        #self.coords, _ = smooth_coords(self.coords)

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



class DrawableTrafo(Drawable):
    """
    Class for objects that are transformed using cairo transformations.

    Rather than recalculating the coordinates upon tranformation, we record the transformations
    and apply them when drawing.

    This has some advantages, but would mess with the pens in case of
    paths, and might be less efficient than recalculating the coordinates
    once and for all.

    The coordinates here hold the original bounding box of the object, with
    coords[0] being the left upper, and coords[1] the right lower corner.

    This information is only used for calculating the bounding box of the
    transformed object.
    """

    def __init__(self, mytype, coords, pen, transform = None):

        log.debug("initializing DrawableTrafo %s, %s, %s", mytype, coords, transform)
        super().__init__(mytype, coords, pen)
        self.__bbox = None

        self.__trafo = Trafo(transform)
        self.bbox_recalculate()

    def draw(self, cr, hover=False, selected=False, outline=False):
        """Draw the object on the Cairo context."""
        raise NotImplementedError("draw method not implemented")

    def trafo(self):
        """Return the transformations."""
        return self.__trafo

    def apply_trafos(self, cr):
        """Apply the transformations to the Cairo context."""

        self.__trafo.transform_context(cr)

    def move(self, dx, dy):
        """Move the image by dx, dy."""
        self.__trafo.add_trafo(("move", (dx, dy)))
        self.bbox_recalculate()

    def bbox_recalculate(self, mod = True):
        """Return the bounding box of the object."""
        log.debug("recalculating bbox of %s", self.type)
        if mod:
            self.mod += 1
        coords = self.coords
        x0, y0 = coords[0]
        x1, y1 = coords[1]
        coords = [ (x0, y0), (x1, y0), (x1, y1), (x0, y1) ]

        coords = self.__trafo.apply(coords)

        # get the bounding box
        self.__bbox = path_bbox(coords)
        return self.__bbox

    def bbox(self, actual = False):
        """Return the bounding box of the object."""
        return self.__bbox

    def resize_start(self, corner, origin):
        """Start the resizing operation."""
        self.resizing = {
            "corner": corner,
            "origin": origin,
            "orig_bbox": self.bbox(),
            "bbox":   self.bbox(),
            }
        self.mod += 1
        # dummy transformation x 2
        # needed because resize_update first removes the previous two
        self.__trafo.add_trafo(("dummy", (0, 0, 1, 1)))
        self.__trafo.add_trafo(("dummy", (0, 0, 1, 1)))

    def resize_update(self, bbox):
        """Update during the resize of the object."""

        # remove the 2 last transformations
        # (the resizing might involve a move!)
        self.__trafo.pop_trafo()
        self.__trafo.pop_trafo()

        # this is the new bounding box
        x1, y1, w1, h1 = bbox

        # this is the old bounding box
        x0, y0, w0, h0 = self.resizing["orig_bbox"]

        # calculate the new transformation
        w_scale = w1 / w0
        h_scale = h1 / h0

        # apply the transformation. No merging, because they are temporary
        self.__trafo.add_trafo(("move", (x1 - x0, y1 - y0)), merge = False)
        self.__trafo.add_trafo(("resize", (x1, y1, w_scale, h_scale)), merge = False)
        self.bbox_recalculate()

    def resize_end(self):
        """Finish the resizing operation."""
        self.resizing = None
        self.mod += 1

    def rotate(self, angle, set_angle = False):
        """Rotate the object by the specified angle."""
        # if the previous trafo is a rotation, simply add the new angle
        self.__trafo.add_trafo(("rotate", (self.rot_origin[0], self.rot_origin[1], angle)))
        self.bbox_recalculate()

    def rotate_end(self):
        """Finish the rotation operation."""

    def is_close_to_click(self, click_x, click_y, threshold):
        """Check if a click is close to the object."""
        bb = self.bbox()
        if bb is None:
            return False
        x, y, width, height = bb
        if x <= click_x <= x + width and y <= click_y <= y + height:
            return True
        return False

class Image(DrawableTrafo):
    """
    Class for Images
    """
    def __init__(self, coords, pen, image, image_base64 = None, transform = None):

        #log.debug("CREATING IMAGE, pos %s, trafo %s", coords, transform)
        self.__image = ImageObj(image, image_base64)

        width, height = self.__image.size()
        coords = [ (coords[0][0], coords[0][1]),
                   (coords[0][0] + width, coords[0][1] + height) ]

        super().__init__("image", coords, pen, transform)

        self.bbox_recalculate()

    def draw(self, cr, hover=False, selected=False, outline=False):
        """Draw the object on the Cairo context."""
        cr.save()
        self.trafo().transform_context(cr)

        w, h = self.__image.size()
        x, y = self.coords[0]
        cr.rectangle(x, y, w, h)
        cr.clip()
        Gdk.cairo_set_source_pixbuf(cr, self.__image.pixbuf(), x, y)
        cr.paint()

        cr.restore()

        #cr.set_source_rgb(*self.pen.color)
        cr.set_source_rgb(1, 0, 0)
        if selected:
            self.bbox_draw(cr, lw=1.5)
        if hover:
            self.bbox_draw(cr, lw=.5)

    def image(self):
        """Return the image."""
        return self.__image.pixbuf()

    def to_dict(self):
        """Convert the object to a dictionary."""

        log.debug("transformations saved: %s", self.trafo().trafos())
        return {
            "type": self.type,
            "coords": self.coords,
            "pen": self.pen.to_dict(),
            "image": None,
            "image_base64": self.__image.base64(),
            "transform": self.trafo().trafos(),
        }


class Text(DrawableTrafo):
    """Class for Text objects"""
    def __init__(self, coords, pen, content, transform = None):


        # split content by newline
        # content = content.split("\n")
        self.__text = TextEditor(content)
        self.font_extents = None
        self.__show_caret = False

        coords = [ (coords[0][0], coords[0][1]), (50, 50) ]

        super().__init__("text", coords, pen, transform)

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

    def stroke(self, font_size = None):
        """Return the stroke of the text."""
        if font_size is not None:
            self.pen.font_size = font_size
            self.mod += 1
        return self.pen.font_size

    def stroke_change(self, direction):
        """Change text size up or down."""
        self.pen.font_size += direction
        self.pen.font_size = max(8, min(128, self.pen.font_size))
        self.mod += 1
        return self.pen.font_size

    def to_dict(self):
        return {
            "type": self.type,
            "coords": self.coords,
            "pen": self.pen.to_dict(),
            "content": self.__text.to_string(),
            "transform": self.trafo().trafos(),
        }

    def to_string(self):
        """Return the text as a single string."""
        return self.__text.to_string()

    def strlen(self):
        """Return the length of the text."""
        return self.__text.strlen()

    def set_text(self, text):
        """Set the text of the object."""
        self.__text.set_text(text)
        self.mod += 1

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
        dy_top = self.font_extents[0]
        bb = [position[0],
              position[1], # - self.font_extents[0],
              0, 0]

        cr.save()
        self.trafo().transform_context(cr)

        for i, fragment in enumerate(content):

            cr.set_source_rgba(*self.pen.color, self.pen.transparency)
            #x_bearing, y_bearing, t_width, t_height, x_advance, y_advance
            x_bearing, _, t_width, _, _, _ = cr.text_extents(fragment)

            bb[2] = max(bb[2], t_width + x_bearing)
            bb[3] += self.font_extents[2]

            cr.set_font_size(self.pen.font_size)
            cr.move_to(position[0], position[1] + dy + dy_top)
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
                                position[1] + dy,
                                self.font_extents[2])

            dy += self.font_extents[2]

        new_coords = [ (bb[0], bb[1]), (bb[0] + bb[2], bb[1] + bb[3]) ]

        if new_coords != self.coords:
            self.coords = new_coords
            self.bbox_recalculate(mod = False)

        #cr.rectangle(bb[0], bb[1], bb[2], bb[3])
        #cr.stroke()

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
        log.debug("finishing shape")
        self.coords, _ = smooth_coords(self.coords)
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

    def draw_points(self, cr):
        """Draw a dot at each of the coordinate pairs"""
        for x, y in self.coords:
            cr.arc(x, y, 1, 0, 2 * 3.14159)
            cr.fill()

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
            cr.set_source_rgba(0, 1, 1)
            self.draw_simple(cr, res_bb)
            cr.set_line_width(0.5)
            cr.stroke()
            self.draw_points(cr)
        elif self.fill():
            self.draw_simple(cr, res_bb)
            cr.fill()
        else:
            self.draw_simple(cr, res_bb)
            cr.set_line_width(self.pen.line_width)
            cr.stroke()

        if selected:
            cr.set_source_rgba(1, 0, 0)
            self.bbox_draw(cr, lw=.5)

        if hover:
            self.bbox_draw(cr, lw=.3)

        if self.rotation != 0:
            cr.restore()

    @classmethod
    def from_object(cls, obj):
        """Create a shape from an object."""
        log.debug("Shape.from_object %s", obj)
        if obj.coords and len(obj.coords) > 2 and obj.pen:
            return cls(obj.coords, obj.pen)

        # issue a warning
        log.warning("Shape.from_object: invalid object")
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
        log.debug("finishing rectangle")
        #self.coords, _ = smooth_coords(self.coords)

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



class DrawableTrafo(Drawable):
    """
    Class for objects that are transformed using cairo transformations.

    Rather than recalculating the coordinates upon tranformation, we record the transformations
    and apply them when drawing.

    This has some advantages, but would mess with the pens in case of
    paths, and might be less efficient than recalculating the coordinates
    once and for all.

    The coordinates here hold the original bounding box of the object, with
    coords[0] being the left upper, and coords[1] the right lower corner.

    This information is only used for calculating the bounding box of the
    transformed object.
    """

    def __init__(self, mytype, coords, pen, transform = None):

        log.debug("initializing DrawableTrafo %s, %s, %s", mytype, coords, transform)
        super().__init__(mytype, coords, pen)
        self.__bbox = None

        self.__trafo = Trafo(transform)
        self.bbox_recalculate()

    def draw(self, cr, hover=False, selected=False, outline=False):
        """Draw the object on the Cairo context."""
        raise NotImplementedError("draw method not implemented")

    def trafo(self):
        """Return the transformations."""
        return self.__trafo

    def apply_trafos(self, cr):
        """Apply the transformations to the Cairo context."""

        self.__trafo.transform_context(cr)

    def move(self, dx, dy):
        """Move the image by dx, dy."""
        self.__trafo.add_trafo(("move", (dx, dy)))
        self.bbox_recalculate()

    def bbox_recalculate(self, mod = True):
        """Return the bounding box of the object."""
        log.debug("recalculating bbox of %s", self.type)
        if mod:
            self.mod += 1
        coords = self.coords
        x0, y0 = coords[0]
        x1, y1 = coords[1]
        coords = [ (x0, y0), (x1, y0), (x1, y1), (x0, y1) ]

        coords = self.__trafo.apply(coords)

        # get the bounding box
        self.__bbox = path_bbox(coords)
        return self.__bbox

    def bbox(self, actual = False):
        """Return the bounding box of the object."""
        return self.__bbox

    def resize_start(self, corner, origin):
        """Start the resizing operation."""
        self.resizing = {
            "corner": corner,
            "origin": origin,
            "orig_bbox": self.bbox(),
            "bbox":   self.bbox(),
            }
        self.mod += 1
        # dummy transformation x 2
        # needed because resize_update first removes the previous two
        self.__trafo.add_trafo(("dummy", (0, 0, 1, 1)))
        self.__trafo.add_trafo(("dummy", (0, 0, 1, 1)))

    def resize_update(self, bbox):
        """Update during the resize of the object."""

        # remove the 2 last transformations
        # (the resizing might involve a move!)
        self.__trafo.pop_trafo()
        self.__trafo.pop_trafo()

        # this is the new bounding box
        x1, y1, w1, h1 = bbox

        # this is the old bounding box
        x0, y0, w0, h0 = self.resizing["orig_bbox"]

        # calculate the new transformation
        w_scale = w1 / w0
        h_scale = h1 / h0

        # apply the transformation. No merging, because they are temporary
        self.__trafo.add_trafo(("move", (x1 - x0, y1 - y0)), merge = False)
        self.__trafo.add_trafo(("resize", (x1, y1, w_scale, h_scale)), merge = False)
        self.bbox_recalculate()

    def resize_end(self):
        """Finish the resizing operation."""
        self.resizing = None
        self.mod += 1

    def rotate(self, angle, set_angle = False):
        """Rotate the object by the specified angle."""
        # if the previous trafo is a rotation, simply add the new angle
        self.__trafo.add_trafo(("rotate", (self.rot_origin[0], self.rot_origin[1], angle)))
        self.bbox_recalculate()

    def rotate_end(self):
        """Finish the rotation operation."""

    def is_close_to_click(self, click_x, click_y, threshold):
        """Check if a click is close to the object."""
        bb = self.bbox()
        if bb is None:
            return False
        x, y, width, height = bb
        if x <= click_x <= x + width and y <= click_y <= y + height:
            return True
        return False

class Image(DrawableTrafo):
    """
    Class for Images
    """
    def __init__(self, coords, pen, image, image_base64 = None, transform = None):

        #log.debug("CREATING IMAGE, pos %s, trafo %s", coords, transform)
        self.__image = ImageObj(image, image_base64)

        width, height = self.__image.size()
        coords = [ (coords[0][0], coords[0][1]),
                   (coords[0][0] + width, coords[0][1] + height) ]

        super().__init__("image", coords, pen, transform)

        self.bbox_recalculate()

    def draw(self, cr, hover=False, selected=False, outline=False):
        """Draw the object on the Cairo context."""
        cr.save()
        self.trafo().transform_context(cr)

        w, h = self.__image.size()
        x, y = self.coords[0]
        cr.rectangle(x, y, w, h)
        cr.clip()
        Gdk.cairo_set_source_pixbuf(cr, self.__image.pixbuf(), x, y)
        cr.paint()

        cr.restore()

        #cr.set_source_rgb(*self.pen.color)
        cr.set_source_rgb(1, 0, 0)
        if selected:
            self.bbox_draw(cr, lw=1.5)
        if hover:
            self.bbox_draw(cr, lw=.5)

    def image(self):
        """Return the image."""
        return self.__image.pixbuf()

    def to_dict(self):
        """Convert the object to a dictionary."""

        log.debug("transformations saved: %s", self.trafo().trafos())
        return {
            "type": self.type,
            "coords": self.coords,
            "pen": self.pen.to_dict(),
            "image": None,
            "image_base64": self.__image.base64(),
            "transform": self.trafo().trafos(),
        }


class Text(DrawableTrafo):
    """Class for Text objects"""
    def __init__(self, coords, pen, content, transform = None):


        # split content by newline
        # content = content.split("\n")
        self.__text = TextEditor(content)
        self.font_extents = None
        self.__show_caret = False

        coords = [ (coords[0][0], coords[0][1]), (50, 50) ]

        super().__init__("text", coords, pen, transform)

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

    def stroke(self, font_size = None):
        """Return the stroke of the text."""
        if font_size is not None:
            self.pen.font_size = font_size
            self.mod += 1
        return self.pen.font_size

    def stroke_change(self, direction):
        """Change text size up or down."""
        self.pen.font_size += direction
        self.pen.font_size = max(8, min(128, self.pen.font_size))
        self.mod += 1
        return self.pen.font_size

    def to_dict(self):
        return {
            "type": self.type,
            "coords": self.coords,
            "pen": self.pen.to_dict(),
            "content": self.__text.to_string(),
            "transform": self.trafo().trafos(),
        }

    def to_string(self):
        """Return the text as a single string."""
        return self.__text.to_string()

    def strlen(self):
        """Return the length of the text."""
        return self.__text.strlen()

    def set_text(self, text):
        """Set the text of the object."""
        self.__text.set_text(text)
        self.mod += 1

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
        dy_top = self.font_extents[0]
        bb = [position[0],
              position[1], # - self.font_extents[0],
              0, 0]

        cr.save()
        self.trafo().transform_context(cr)

        for i, fragment in enumerate(content):

            cr.set_source_rgba(*self.pen.color, self.pen.transparency)
            #x_bearing, y_bearing, t_width, t_height, x_advance, y_advance
            x_bearing, _, t_width, _, _, _ = cr.text_extents(fragment)

            bb[2] = max(bb[2], t_width + x_bearing)
            bb[3] += self.font_extents[2]

            cr.set_font_size(self.pen.font_size)
            cr.move_to(position[0], position[1] + dy + dy_top)
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
                                position[1] + dy,
                                self.font_extents[2])

            dy += self.font_extents[2]

        new_coords = [ (bb[0], bb[1]), (bb[0] + bb[2], bb[1] + bb[3]) ]

        if new_coords != self.coords:
            self.coords = new_coords
            self.bbox_recalculate(mod = False)

        #cr.rectangle(bb[0], bb[1], bb[2], bb[3])
        #cr.stroke()

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
        log.debug("finishing shape")
        self.coords, _ = smooth_coords(self.coords)
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

    def draw_points(self, cr):
        """Draw a dot at each of the coordinate pairs"""
        for x, y in self.coords:
            cr.arc(x, y, 1, 0, 2 * 3.14159)
            cr.fill()

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
            cr.set_source_rgba(0, 1, 1)
            self.draw_simple(cr, res_bb)
            cr.set_line_width(0.5)
            cr.stroke()
            self.draw_points(cr)
        elif self.fill():
            self.draw_simple(cr, res_bb)
            cr.fill()
        else:
            self.draw_simple(cr, res_bb)
            cr.set_line_width(self.pen.line_width)
            cr.stroke()

        if selected:
            cr.set_source_rgba(1, 0, 0)
            self.bbox_draw(cr, lw=.5)

        if hover:
            self.bbox_draw(cr, lw=.3)

        if self.rotation != 0:
            cr.restore()

    @classmethod
    def from_object(cls, obj):
        """Create a shape from an object."""
        log.debug("Shape.from_object %s", obj)
        if obj.coords and len(obj.coords) > 2 and obj.pen:
            return cls(obj.coords, obj.pen)

        # issue a warning
        log.warning("Shape.from_object: invalid object")
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
        log.debug("finishing rectangle")
        #self.coords, _ = smooth_coords(self.coords)

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



class PathRoot(Drawable):
    """ Path is like shape, but not closed and has an outline that depends on
        line width and pressure."""
    def __init__(self, mytype, coords, pen, outline = None, pressure = None, brush = None): # pylint: disable=unused-argument
        super().__init__(mytype, coords, pen = pen)
        self.__pressure  = pressure or [1] * len(coords)
        self.__bb        = []
        self.__brush     = None
        self.__n_appended = 0

        if outline:
            log.warning("outline is not used in Path")

    def brush(self, brush = None):
        """Set the brush for the path."""
        if not brush:
            return self.__brush
        self.__brush = brush
        return brush

    def outline_recalculate(self):
        """Recalculate the outline of the path."""
        if len(self.coords) != len(self.__pressure):
            log.error("coords and pressure have different lengths")
            log.error("length coords: %s", len(self.coords))
            log.error("length pressure: %s", len(self.__pressure))
            raise ValueError("coords and pressure have different lengths")

        self.__brush.calculate(self.pen.line_width,
                                 coords = self.coords,
                                 pressure = self.__pressure)
        self.__bb = self.__brush.bbox() or path_bbox(self.coords)
        self.mod += 1

    def finish(self):
        """Finish the path."""
        self.outline_recalculate()

    def update(self, x, y, pressure):
        """Update the path with a new point."""
       #if len(x) != len(pressure) or len(y) != len(x):
       #    log.error("incorrect values sent to the update function")
       #    log.error("length x: %s", len(x))
       #    log.error("length y: %s", len(y))
       #    log.error("length pressure: %s", len(pressure))
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

    def stroke(self, lw = None):
        """Change the stroke size."""
        if lw:
            self.pen.line_width = lw
        self.outline_recalculate()
        return self.pen.stroke()

    def stroke_change(self, direction):
        """Change the stroke size."""
        self.pen.stroke_change(direction)
        self.outline_recalculate()
        return self.pen.stroke()

    def smoothen(self, threshold=20):
        """Smoothen the path."""
        if len(self.coords) < 3:
            return
        self.coords, self.__pressure = smooth_coords(self.coords, self.__pressure, 1)
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

        if len(coords) < 2:
            return

        self.__n_appended = self.__n_appended - 1

        if self.__n_appended >= len(self.coords):
            return

        self.__pressure.pop()
        coords.pop()

        if len(coords) < 2:
            return

        self.outline_recalculate()
        self.__bb = None

    def bbox(self, actual = False):
        """Return the bounding box"""
        if self.resizing:
            return self.resizing["bbox"]
        if not self.__bb:
            self.__bb = self.__brush.bbox() or path_bbox(self.coords)
        return self.__bb

    def resize_end(self):
        """recalculate the outline after resizing"""
        #print("length of coords and pressure:", len(self.coords), len(self.__pressure))
        old_bbox = self.__bb or path_bbox(self.coords)
        self.coords = transform_coords(self.coords, old_bbox, self.resizing["bbox"])
        self.outline_recalculate()
        self.resizing  = None

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
            cr.arc(coords[i][0], coords[i][1], .4, 0, 2 * 3.14159)  # Draw a circle at each point
            cr.fill()
        cr.arc(coords[-1][0], coords[-1][1], .4, 0, 2 * 3.14159)  # Draw a circle at each point
        cr.fill()

    def draw_simple(self, cr, bbox=None):
        """draws the path as a single line. Useful for resizing."""

        if len(self.coords) < 2:
            return

        if bbox:
            old_bbox = self.__brush.bbox() or self.coords
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
        if self.__brush.outline() is None:
            return

        if len(self.__brush.outline()) < 2 or len(self.coords) < 2:
            return

        if self.__brush.outline() is None:
            log.warning("no outline for brush %s",
                        self.__brush.brush_type())
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
            self.bbox_draw(cr, lw=.5)

        if hover:
            self.bbox_draw(cr, lw=.3)

        if self.rotation != 0:
            cr.restore()

class Path(PathRoot):
    """ Path is like shape, but not closed and has an outline that depends on
        line width and pressure."""
    def __init__(self, coords, pen, pressure = None, brush = None):
        super().__init__("path", coords, pen = pen, pressure = pressure)

        if brush:
            self.brush(BrushFactory.create_brush(**brush))
        else:
            brush_type = pen.brush_type()
            self.brush(BrushFactory.create_brush(brush_type))

        if len(self.coords) > 3 and not self.brush().outline():
            self.outline_recalculate()

    @classmethod
    def from_object(cls, obj):
        """Generate path from another object."""
        log.debug("Path.from_object %s", obj)
        if obj.coords and len(obj.coords) > 2 and obj.pen:
            return cls(obj.coords, obj.pen)
        # issue a warning
        log.warning("Path.from_object: invalid object")
        return obj


class SegmentedPath(PathRoot):
    """Path with no smoothing at joints."""
    def __init__(self, coords, pen, pressure = None, brush = None):
        super().__init__("segmented_path", coords, pen = pen, pressure = pressure)

        if brush:
            brush['smooth_path'] = False
            self.brush(BrushFactory.create_brush(**brush))
        else:
            brush_type = pen.brush_type()
            self.brush(BrushFactory.create_brush(brush_type, smooth_path = False))

        if len(self.coords) > 1 and not self.brush().outline():
            self.outline_recalculate()


Drawable.register_type("path", Path)
Drawable.register_type("segmented_path", SegmentedPath)
"""Module for history tracking."""

class History:
    """
    Class for history tracking.

    Keeps track of the undo / redo stacks.
    """

    def __init__(self, bus):
        self.__history = []
        self.__redo = []
        self.__bus = bus
        self.__cur_page = "None"
        bus.on("page_changed",     self.set_page)
        bus.on("history_redo",     self.redo)
        bus.on("history_undo",     self.undo)
        bus.on("history_undo_cmd", self.undo_command)
        bus.on("history_append",   self.add)
        bus.on("history_remove",   self.history_remove)

    def length(self):
        """Return the number of items in the history."""
        return len(self.__history)

    def set_page(self, page):
        """Set the current page."""
        log.debug("setting page to %s", page)
        self.__cur_page = page

    def add(self, cmd):
        """Add item to history."""
        log.debug("appending {cmd.type()} on page={self.__cur_page} new cmd hash=%s", cmd.hash())

        oldcmd  = self.__history[-1]['cmd']  if self.__history else None
        oldpage = self.__history[-1]['page'] if self.__history else None
        log.debug("oldcmd hash={oldcmd.hash() if oldcmd else None} oldpage=%s", oldpage)

        if oldcmd and oldpage == self.__cur_page:
            if oldcmd == cmd or oldcmd > cmd:
                log.debug("cmd hash=%s", cmd.hash() if cmd else None)
                log.debug("merging commands, {cmd.type()} and %s", oldcmd.type())
                self.__history.pop()
                cmd = oldcmd + cmd
                log.debug("new command hash=%s", cmd.hash())

        self.__history.append({'cmd': cmd, 'page': self.__cur_page})
        self.__redo = []

    def history_remove(self, cmd):
        """Remove an item from the history."""
        log.debug("removing %s from history", cmd.type())
        n = len(self.__history)
        self.__history = [ item for item in self.__history if item['cmd'] != cmd ]
        if n == len(self.__history):
            log.warning("could not remove %s from history", cmd.type())

    def undo_command(self, cmd):
        """
        Undo a specific command.

        Dangerous! Use with caution. Make sure that the command does not
        have any side effects.
        """
        log.debug("undoing specific command, type %s", cmd.type())
        if not self.__history:
            log.warning("Nothing to undo")
            return None

        if self.__history[-1]['cmd'] == cmd:
            return self.undo()

        log.warning("Undoing cmd %s which is not the last in history, beware", cmd.type())

        for i, item in enumerate(self.__history):
            if item['cmd'] == cmd:
                self.__history.pop(i)
                self.__redo.append(item)
                if item['page'] != self.__cur_page:
                    self.__bus.emit("page_set", False, item['page'])
                return cmd.undo()
        log.warning("Command %s not found in  history", cmd.type())
        return None

    def undo(self):
        """Undo the last action."""
        if not self.__history:
            log.warning("Nothing to undo")
            return None

        item = self.__history.pop()
        cmd = item['cmd']
        log.debug("undoing %s", cmd.type())
        ret = cmd.undo()
        self.__redo.append(item)
        if item['page'] != self.__cur_page:
            self.__bus.emit("page_set", False, item['page'])
        return ret

    def redo(self):
        """Redo the last action."""
        if not self.__redo:
            return None

        item = self.__redo.pop()
        cmd = item['cmd']
        log.debug("redoing %s", cmd.type())
        ret = cmd.redo()
        self.__history.append(item)
        if item['page'] != self.__cur_page:
            self.__bus.emit("page_set", False, item['page'])
        return ret
"""An event bus class for dispatching events between objects."""

class Bus:
    """A simple event bus for dispatching events between objects."""

    def __init__(self):
        self.__listeners = {}

    def on(self, event, listener, priority = 0):
        """Add a listener for an event."""
        if listener is None:
            raise ValueError("Listener cannot be None")

        if not callable(listener):
            raise ValueError("Listener must be callable")

        if event is None:
            raise ValueError("Event cannot be None")

        if event not in self.__listeners:
            self.__listeners[event] = []

        self.__listeners[event].append((listener, priority))
        self.__listeners[event].sort(key = lambda x: -x[1])

    def off(self, event, listener):
        """Remove a listener for an event."""
        if event in self.__listeners:
            self.__listeners[event][:] = [x for x in self.__listeners[event] if x[0] != listener]

    def call(self, listener, event, include_event, args, kwargs):
        """Call the listener with the specified arguments."""
        try:
            if include_event:
                ret = listener(event, *args, **kwargs)
            else:
                ret = listener(*args, **kwargs)
        except Exception: #pylint: disable=broad-except
            ret = None
            exc_type, exc_value, exc_traceback = exc_info()
            log.warning("Traceback:")
            traceback.print_tb(exc_traceback)
            log.warning("Exception value: %s", exc_value)
            log.error("Exception type: %s", exc_type)
            log.warning("Error while dispatching signal %s to %s:", event, listener)
        return ret

    def emit_once(self, event, *args, **kwargs):
        """Emit an exclusive event - stops dispatching if a listener returns a truthy value."""

        return self.emit(event, True, *args, **kwargs)

    def emit_mult(self, event, *args, **kwargs):
        """Emit a non-exclusive event - dispatches to all listeners regardless of return value."""

        return self.emit(event, False, *args, **kwargs)

    def emit(self, event, exclusive = False, *args, **kwargs):
        """
        Dispatch an event to all listeners.

        Exclusive events will stop dispatching if a listener returns a truthy value.
        """

        log.debug("emitting event %s exclusive=%s with %s and %s",
                  event, exclusive, args, kwargs)

        # completely ignore events that have no listeners
        if not event in self.__listeners:
            return False

        # call the promiscous listeners first, but they don't stop the event
        for listener, _ in self.__listeners.get('*', []):
            ret = self.call(listener, event, True, args, kwargs)

        caught = False
        for listener, _ in self.__listeners.get(event, []):
            ret = self.call(listener, event, False, args, kwargs)
            if ret:
                caught = True
                if exclusive:
                    return ret

        return caught
"""Class which produces dialogs and UI elements"""
from os import path








class UIBuilder():
    """Builds the UI."""

    __new_instance = None

    def __new__(cls, *args, **kwargs):
        if not cls.__new_instance:
            cls.__new_instance = super(UIBuilder, cls).__new__(cls)
        return cls.__new_instance

    def __init__(self, state):

        self.__state = state
        self.__app = state.app()
        self.__bus = state.bus()

        self.__init_wiglets()
        self.__register_bus_events()
        self.__ui_state = {
                "export_options": {
                    "format": None,
                    "all_pages_pdf": False,
                    "export_screen": False
                    }
                }

    def __register_bus_events(self):
        """Register the bus events."""

        listeners = {
            "show_help_dialog":  self.show_help_dialog,
            "import_image":      self.import_image,
            "select_font":       self.select_font,
            "select_color_bg":   self.select_color_bg,
            "select_color":      self.select_color,
            "open_drawing":      self.open_drawing,
            "save_drawing_as":   self.save_drawing_as,
            "export_drawing":    self.export_drawing,
            "screenshot":        self.screenshot,
        }

        for event, listener in listeners.items():
            self.__bus.on(event, listener)

    def __init_wiglets(self):
        """Initialize the wiglets."""
        bus = self.__bus
        state = self.__state

        WigletZoom(bus = bus)
        WigletPan(bus = bus)
        WigletStatusLine(bus = bus, state = state)
        WigletEraser(bus = bus, state = state)
        WigletCreateObject(bus = bus, state = state)
        WigletCreateGroup(bus = bus, state = state)
        WigletCreateSegments(bus = bus, state = state)
        WigletEditText(bus = bus, state = state)
        WigletHover(bus = bus, state = state)
        WigletSelectionTool(bus = bus, state = state)
        WigletResizeRotate(bus = bus, state = state)
        WigletMove(bus = bus, state = state)
        WigletColorSelector(bus = bus, func_color = state.set_color,
                             func_bg = state.graphics().bg_color)
        WigletToolSelector(bus = bus, func_mode = state.mode)
        WigletPageSelector(bus = bus, state = state)
        WigletColorPicker(bus = bus, func_color = state.set_color,
                          clipboard = state.clipboard())
        WigletTransparency(bus = bus, state = state)
        WigletLineWidth(bus = bus, state = state)

    def show_help_dialog(self):
        """Show the help dialog."""

        dialog = HelpDialog(self.__app)
        dialog.run()
        dialog.destroy()

    def select_font(self):
        """Select a font for drawing using FontChooser dialog."""
        font_description = font_chooser(self.__state.pen(), self.__app)

        if font_description:
            self.__bus.emit_mult("set_font", font_description)

    def select_color_bg(self):
        """Select a color for the background using color_chooser."""

        color = color_chooser(self.__app, "Select Background Color")

        if color:
            self.__bus.emit_once("set_bg_color", (color.red, color.green, color.blue))

    def select_color(self):
        """Select a color for drawing using color_chooser dialog."""
        color = color_chooser(self.__app)
        if color:
            self.__bus.emit_mult("set_color", (color.red, color.green, color.blue))

    def import_image(self):
        """Select an image file and create a pixbuf from it."""

        bus = self.__bus

        import_dir = self.__state.config().import_dir()
        image_file = import_image_dialog(self.__app, import_dir = import_dir)
        dirname, _ = path.split(image_file)
        bus.emit_mult("set_import_dir", dirname)

        pixbuf = None

        if image_file:
            try:
                pixbuf = GdkPixbuf.Pixbuf.new_from_file(image_file)
                log.debug("Loaded image: %s", image_file)
            except Exception as e: #pylint: disable=broad-except
                log.error("Failed to load image: %s", e)

            if pixbuf is not None:
                pos = self.__state.cursor_pos()
                img = Image([ pos ], self.__state.pen(), pixbuf)
                bus.emit_once("add_object", img)

        return pixbuf

    def open_drawing(self):
        """Open a drawing from a file in native format."""
        file_name = open_drawing_dialog(self.__app)

        if file_name and self.read_file(file_name):
            log.debug("Setting savefile to %s", file_name)

            self.__bus.emit("set_savefile", False, file_name)
            self.__state.graphics().modified(True)

    def read_file(self, filename, load_config = True):
        """Read the drawing state from a file."""
        if not filename:
            raise ValueError("No filename provided")

        state = self.__state
        config, objects, pages = read_file_as_sdrw(filename)

        if pages:
            state.gom().set_pages(pages)
        elif objects:
            state.gom().set_objects(objects)

        if config and load_config:
            state.graphics().bg_color(config.get('bg_color') or (.8, .75, .65))
            state.graphics().alpha(config.get('transparent') or 0)
            show_wiglets = config.get('show_wiglets')

            if show_wiglets is None:
                show_wiglets = True

            state.graphics().show_wiglets(show_wiglets)
            state.pen(pen = Pen.from_dict(config['pen']))
            state.pen(pen = Pen.from_dict(config['pen2']), alternate = True)
            state.gom().set_page_number(config.get('page') or 0)
        if config or objects:
            state.graphics().modified(True)
            return True
        return False

    def autosave(self):
        """Autosave the drawing state."""
        state = self.__state

        if not state.graphics().modified():
            return

        if state.current_obj(): # not while drawing!
            return

        log.debug("Autosaving")

        self.save_state()
        state.graphics().modified(False)

    def save_state(self):
        """Save the current drawing state to a file."""
        state = self.__state

        savefile = state.config().savefile()

        if not savefile:
            log.debug("No savefile set")
            return

        log.debug("savefile: %s", savefile)
        config = {
                'bg_color':     state.graphics().bg_color(),
                'transparent':  state.graphics().alpha(),
                'show_wiglets': state.graphics().show_wiglets(),
                'bbox':         (0, 0, *self.__app.get_size()),
                'pen':          state.pen().to_dict(),
                'pen2':         state.pen(alternate = True).to_dict(),
                'page':         state.gom().current_page_number()
        }

        pages   = state.gom().export_pages()

        save_file_as_sdrw(savefile, config, pages = pages)

    def save_drawing_as(self):
        """Save the drawing to a file."""
        log.debug("opening save file dialog")
        file = save_dialog(self.__app)

        if file:
            log.debug("setting savefile to %s", file)
            self.__bus.emit("set_savefile", False, file)
            self.save_state()

    def __normalize_export_format(self, file_format):
        """Normalize the export format."""
        formats = { "By extension": "any",
                    "PDF": "pdf",
                    "SVG": "svg",
                    "PNG": "png",
                   }
        return formats[file_format]

    def __get_export_cfg(self, bbox):
        """Get the export configuration."""
        state = self.__state
        cfg = { "bg": state.graphics().bg_color(),
               "bbox": bbox,
               "transparency": state.graphics().alpha() }
        return cfg

    def __emit_export_dir(self, file_name):
        """Emit the export directory."""
        # extract dirname and base name from file name
        dirname, base_name = path.split(file_name)
        log.debug("dirname: %s base_name: %s", dirname, base_name)

        self.__bus.emit_mult("set_export_dir", dirname)
        self.__bus.emit_mult("set_export_fn", base_name)

    def __prepare_export_objects(self, all_as_pdf, exp_screen, selected):
        """Prepare the objects for export."""
        state = self.__state

        if all_as_pdf:
            # get all objects from all pages and layers
            # create drawable group for each page
            obj = state.gom().get_all_pages()
            obj = [ p.objects_all_layers() for p in obj ]
            obj = [ DrawableGroup(o) for o in obj ]
            bbox = None
        elif exp_screen:
            log.debug("Exporting screen")
            obj = DrawableGroup(state.objects_all_layers())
            bbox = state.visible_bbox()
        else:
            log.debug("Exporting selected objects")
            if selected:
                obj = DrawableGroup(selected)
            else:
                obj = DrawableGroup(state.objects_all_layers())
            bbox = obj.bbox()

        return obj, bbox

    def export_drawing(self):
        """Save the drawing to a file."""
        # Choose where to save the file
        #    self.export(filename, "svg")
        bbox = None
        state = self.__state

        selected = state.selected_objects()

        self.__ui_state["export_options"]["selected"] = selected

        export_dir = state.config().export_dir()
        file_name  = state.config().export_fn()

        ret = export_dialog(self.__app, export_dir = export_dir,
                            filename = file_name,
                            ui_opts = self.__ui_state["export_options"]
                            )

        file_name, file_format, all_as_pdf, exp_screen = ret

        if not file_name:
            return

        self.__ui_state["export_options"]["format"] = file_format
        self.__ui_state["export_options"]["all_pages_pdf"] = all_as_pdf
        self.__ui_state["export_options"]["export_screen"] = exp_screen

        file_format = self.__normalize_export_format(file_format)
        obj, bbox = self.__prepare_export_objects(all_as_pdf, exp_screen,
                                                  selected)

        self.__emit_export_dir(file_name)
        cfg = self.__get_export_cfg(bbox)

        export_image(obj, file_name, file_format, cfg, all_as_pdf)

    def __screenshot_finalize(self, bb):
        """Finish up the screenshot."""

        state = self.__state
        trafo = self.__state.page().trafo()
        bus   = self.__bus
        log.debug("Taking screenshot now")

        frame = trafo.apply([ (bb[0] - 3, bb[1] - 3),
                                     (bb[0] + bb[2] + 6, bb[1] + bb[3] + 6) ])
        frame = (frame[0][0], frame[0][1], frame[1][0], frame[1][1])
        log.debug("frame is %s", [ int(x) for x in frame ])

        pixbuf, filename = get_screenshot(self.__app, *frame)
        bus.emit_once("toggle_hide", False)
        bus.emit_once("queue_draw")

        # Create the image and copy the file name to clipboard
        if pixbuf is not None:
            img = Image([ (bb[0], bb[1]) ], state.pen(), pixbuf)
            bus.emit_once("add_object", img)
            bus.emit_once("queue_draw")
            state.clipboard().set_text(filename)

    def __find_screenshot_box(self):
        """Find a box suitable for selecting a screenshot."""

        for obj in self.__state.selected_objects():
            if obj.type == "rectangle":
                return obj

        return None

    def screenshot(self, obj = None):
        """Take a screenshot and add it to the drawing."""

        # function called twice: once, when requesting a screenshot
        # and second, when the screenshot has been made

        state = self.__state
        bus   = self.__bus

        if not obj:
            obj = self.__find_screenshot_box()

        bus.off("add_object", self.screenshot)

        if not obj:
            log.debug("no suitable box found")
            state.mode("rectangle")
            bus.on("add_object", self.screenshot, priority = 999)
            return

        bb = obj.bbox()
        log.debug("bbox is %s", [int(x) for x in bb])

        bus.emit_once("toggle_hide", True)
        bus.emit_once("queue_draw")

        while Gtk.events_pending():
            Gtk.main_iteration_do(False)
        GLib.timeout_add(100, self.__screenshot_finalize, bb)
"""A class which holds a set of transformations."""


# ---------- trafos ----------------------
def trafos_apply(coords, trafos):
    """Apply transformations to a set of coordinates."""

    for trafo in trafos:
        trafo_type, trafo_args = trafo
        if trafo_type == "rotate":
            coords = coords_rotate(coords, trafo_args[2], (trafo_args[0], trafo_args[1]))
        elif trafo_type == "resize":
            coords = [ (trafo_args[0] + (p[0] - trafo_args[0]) * trafo_args[2],
                        trafo_args[1] + (p[1] - trafo_args[1]) * trafo_args[3]) for p in coords ]
        elif trafo_type == "move":
            coords = [ (p[0] + trafo_args[0], p[1] + trafo_args[1]) for p in coords ]
        elif trafo_type == "dummy":
            pass
        else:
            log.error("unknown trafo %s", trafo_type)

    return coords

def trafos_reverse(coords, trafos):
    """Reverse transformations on a set of coordinates."""

    for trafo in reversed(trafos):
        trafo_type, trafo_args = trafo
        if trafo_type == "rotate":
            coords = coords_rotate(coords, -trafo_args[2], (trafo_args[0], trafo_args[1]))
        elif trafo_type == "resize":
            coords = [ (trafo_args[0] + (p[0] - trafo_args[0]) / trafo_args[2],
                        trafo_args[1] + (p[1] - trafo_args[1]) / trafo_args[3]) for p in coords ]
        elif trafo_type == "move":
            coords = [ (p[0] - trafo_args[0], p[1] - trafo_args[1]) for p in coords ]
        elif trafo_type == "dummy":
            pass
        else:
            log.error("unknown trafo %s", trafo_type)

    return coords

def trafos_on_cairo(cr, trafos):
    """Apply transformations to a cairo context."""

    for i, trafo in enumerate(reversed(trafos)):

        trafo_type, trafo_args = trafo

        if trafo_type == "rotate":
            cr.translate(trafo_args[0], trafo_args[1])
            cr.rotate(trafo_args[2])
            cr.translate(-trafo_args[0], -trafo_args[1])
        elif trafo_type == "resize":
            cr.translate(trafo_args[0], trafo_args[1])
            cr.scale(trafo_args[2], trafo_args[3])
            cr.translate(-trafo_args[0], -trafo_args[1])
        elif trafo_type == "move":
            cr.translate(trafo_args[0], trafo_args[1])
        elif trafo_type == "dummy":
            pass
        else:
            log.error("unknown trafo %s [%d]", trafo_type, i)

class Trafo():
    """
    Class to hold a set of transformations.

    Attributes:
        trafo (list): The list of transformations.
    """

    def __init__(self, trafo = None):

        log.debug("initializing with %s", trafo)

        if not trafo:
            self.__trafo = []
        else:
            assert isinstance(trafo, list)
            self.__trafo = trafo

    def n(self):
        """Return the number of transformations."""
        return len(self.__trafo)

    def add_trafo(self, trafo, merge = True):
        """Add a transformation to the list."""
        prev_trafo_type = self.__trafo[-1][0] if self.__trafo else None

        if merge and prev_trafo_type == "move" and trafo[0] == "move":
            x, y = self.__trafo.pop()[1]
            dx, dy = trafo[1]
            self.__trafo.append(("move", (x + dx, y + dy)))
            return

        if merge and prev_trafo_type == "rotate" and trafo[0] == "rotate":
            x0, y0, a0 = self.__trafo.pop()[1]
            x1, y1, a1 = trafo[1]
            if x0 == x1 and y0 == y1:
                self.__trafo.append(("rotate", (x1, y1, a0 + a1)))
                return

        self.__trafo.append(trafo)

    def pop_trafo(self):
        """Remove the last transformation from the list."""
        return self.__trafo.pop()

    def trafos(self, trafo = None):
        """Return or set the transformations."""
        if trafo:
            self.__trafo = trafo
        return self.__trafo

    def apply(self, coords):
        """Apply the transformations to the coordinates."""

        return trafos_apply(coords, self.__trafo)

    def apply_reverse(self, coords):
        """Apply the reverse transformations to the coordinates."""

        return trafos_reverse(coords, self.__trafo)

    def transform_context(self, cr):
        """Transform the cairo context."""

        if self.__trafo:
            trafos_on_cairo(cr, self.__trafo)

    def calc_zoom(self):
        """Calculate the zoom factor of the transformations."""

        zoom_x, zoom_y = 1, 1
        for trafo in self.__trafo:
            trafo_type, trafo_args = trafo
            if trafo_type == "resize":
                zoom_x *= trafo_args[2]
                zoom_y *= trafo_args[3]
        return zoom_x, zoom_y

    def __str__(self):
        return f"Trafo({self.__trafo})"

    def __repr__(self):
        return self.__str__()


# ---------------------------------------------------------------------
# defaults

logging.basicConfig(level=logging.INFO,
                    #format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    format='%(levelname)s %(filename)s:%(lineno)d %(funcName)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
log = logging.getLogger(__name__)
log.info("Application is starting")

APP_NAME   = "ScreenDrawer"
APP_AUTHOR = "JanuaryWeiner"  # used on Windows

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

        self.set_title("Transparent Drawing Window")
        self.set_decorated(False)
        self.connect("destroy", self.exit)
        self.set_default_size(800, 800)
        self.set_keep_above(True)
        self.maximize()

        screen = self.get_screen()
        visual = screen.get_rgba_visual()

        if visual is not None and screen.is_composited():
            self.set_visual(visual)

        self.set_app_paintable(True)

        self.fixed = Gtk.Fixed()
        self.add(self.fixed)

        self.init_ui(save_file = save_file)

    def init_ui(self, save_file):
        """Initialize the user interface."""

        # Drawing setup
        self.bus                = Bus()

        # we pass the app to the state, because it has the queue_draw
        # method
        self.state              = State(app = self,
                                        bus = self.bus)

        # em has to know about all that to link actions to methods
        em  = EventManager(bus = self.bus, state  = self.state)
        MenuMaker(self.bus, self.state)

        # mouse gets the mouse events
        self.mouse = MouseCatcher(bus = self.bus, state = self.state)

        # canvas orchestrates the drawing
        self.canvas = Canvas(bus = self.bus, state = self.state)

        self.uibuilder = UIBuilder(state = self.state)
        # autosave
        GLib.timeout_add(AUTOSAVE_INTERVAL, self.uibuilder.autosave)

        # connecting events
        self.__add_bus_events()

        self.set_events(Gdk.EventMask.BUTTON_PRESS_MASK |
                        Gdk.EventMask.BUTTON_RELEASE_MASK |
                        Gdk.EventMask.POINTER_MOTION_MASK |
                        Gdk.EventMask.TOUCH_MASK)

        self.connect("key-press-event",      em.on_key_press)
        self.connect("draw",                 self.canvas.on_draw)
        self.connect("button-press-event",   self.mouse.on_button_press)
        self.connect("button-release-event", self.mouse.on_button_release)
        self.connect("motion-notify-event",  self.mouse.on_motion_notify)

        # load the drawing from the savefile
        self.bus.emit("set_savefile", False, save_file)
        self.uibuilder.read_file(save_file)


    def exit(self, event = None): # pylint: disable=unused-argument
        """Exit the application."""
        ## close the savefile_f
        log.info("Exiting")
        self.uibuilder.save_state()
        Gtk.main_quit()

    # ---------------------------------------------------------------------

    def __add_bus_events(self):
        """Add bus events."""

        self.bus.on("app_exit", self.exit)


## ---------------------------------------------------------------------

def parse_arguments():
    """Handle command line arguments."""
    parser = argparse.ArgumentParser(
            description="Drawing on the screen",
            epilog=f"Alternative use: {argv[0]} file.sdrw file.[png, pdf, svg, yaml]")
    parser.add_argument("-l", "--loadfile", help="Load drawing from file")
    parser.add_argument("-c", "--convert",
                        help="""
Convert screendrawer file to given format (png, pdf, svg, yaml) and exit
(use -o to specify output file, otherwise a default name is used)
""",
                        metavar = "FORMAT"
    )
    parser.add_argument("-p", "--page", help="Page to convert (default: 1)",
                        type=int)

    parser.add_argument("-b", "--border",
                        help="""
                             Border width for conversion. If not
                             specified, the default is 10 px.
                             """,
                        default=10,
                        type=int)
    parser.add_argument("-o", "--output", help="Output file for conversion")
    parser.add_argument("-d", "--debug", help="Debug mode", action="store_true")
    parser.add_argument("-s", "--sticky", help="Sticky window (yes or no)",
                        choices = [ "yes", "no" ], default = "yes")
    parser.add_argument("files", nargs="*")
    args     = parser.parse_args()
    return args

def process_args(args, app_name, app_author):
    """Process command line arguments."""
    page_no = None
    _savefile = None

    if args.page is not None:
        page_no = args.page - 1

    # explicit conversion
    if args.convert:
        if not args.convert in [ "png", "pdf", "svg", "yaml" ]:
            print("Invalid conversion format")
            sys.exit(1)
        output = None
        if args.output:
            output = args.output

        if not args.files:
            print("No input file provided")
            sys.exit(1)

        convert_file(args.files[0],
                     output,
                     args.convert,
                     border = args.border,
                     page_no = page_no)
        sys.exit(0)

    # convert if exactly two file names are provided
    if args.files:
        if len(args.files) > 2:
            print("Too many files provided")
            sys.exit(1)
        elif len(args.files) == 2:
            convert_file(args.files[0], args.files[1],
                         border = args.border,
                         page_no = page_no)
            sys.exit(0)
        else:
            _savefile = args.files[0]
    else:
        _savefile = get_default_savefile(app_name, app_author)
    return _savefile

def main():
    """Main function for the application."""

    # Get user-specific config directory
    user_config_dir = appdirs.user_config_dir(APP_NAME, APP_AUTHOR)
    log.debug("User config directory: %s", user_config_dir)

    #user_cache_dir = appdirs.user_cache_dir(APP_NAME, APP_AUTHOR)
    #user_log_dir   = appdirs.user_log_dir(APP_NAME, APP_AUTHOR)

# ---------------------------------------------------------------------
# Parsing command line
# ---------------------------------------------------------------------

    args = parse_arguments()
    log.info(args.debug)
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        log.info("Setting logging level to DEBUG")
    savefile = process_args(args, APP_NAME, APP_AUTHOR)
    log.debug("Save file is: %s", savefile)

# ---------------------------------------------------------------------

    win = TransparentWindow(save_file = savefile)
    if args.loadfile:
        win.uibuilder.read_file(args.loadfile)

    css = b"""
    #myMenu {
        background-color: rgba(255, 255, 255, 0.9); /* Semi-transparent white */
        font-family: 'Ubuntu Mono', Monospace, 'Monospace Regular', monospace, 'Courier New'; /* Use 'Courier New', falling back to any monospace font */
    }
    """

    style_provider = Gtk.CssProvider()
    style_provider.load_from_data(css)
    Gtk.StyleContext.add_provider_for_screen(
        Gdk.Screen.get_default(),
        style_provider,
        Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
    )

    win.set_icon(Icons().get("app_icon"))
    win.show_all()
    win.present()
    win.state.cursor().set(win.state.mode())
    if args.sticky == "yes":
        win.stick()

    Gtk.main()

if __name__ == "__main__":
    main()
