"""
General utility functions for the ScreenDrawer application.
"""
import os                                                           #<remove>
import math                                                         #<remove>
import base64                                                       #<remove>
import tempfile                                                     #<remove>
import warnings                                                     #<remove>
from io import BytesIO                                              # <remove>
import gi                                                  # <remove>
gi.require_version('Gtk', '3.0')                           # <remove> pylint: disable=wrong-import-position
gi.require_version('Gdk', '3.0')                           # <remove> pylint: disable=wrong-import-position
from gi.repository import Gdk, GdkPixbuf                            #<remove>
import cairo                                                        #<remove>
import appdirs                                                      #<remove>
import pyautogui                                                    #<remove>
from PIL import ImageGrab                                           #<remove>
import logging                                                   # <remove>
log = logging.getLogger(__name__)                                # <remove>

gi.require_version('Gtk', '3.0')                                    #<remove>

def get_default_savefile(app_name, app_author):
    """Get the default save file for the application."""

    # Get user-specific data directory
    user_data_dir = appdirs.user_data_dir(app_name, app_author)
    log.debug(f"User data directory: {user_data_dir}")
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

    log.debug(f"Saved screenshot to temporary file: {temp_file_name}")

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
        log.warning("outlines of different length")
        return outline_l, outline_r

    out_ret_l = []
    out_ret_r = []

    for i in range(n - 1):

        out_ret_l.append(outline_l[i])
        for j in range(i + 1, n - 1):
            #print("i", i, "left segment: ", pp(outline_l[i]), pp(outline_l[i + 1]))
            #print("j", j, "right segment: ", pp(outline_r[j]), pp(outline_r[j + 1]))
            intersect, point = segment_intersection(outline_l[i], outline_l[i + 1],
                                                outline_r[j], outline_r[j + 1])
            #if intersect:
                #print("FOUND Intersection at", point, "i", i, "j", j)
                #out_ret_l, out_ret_r = out_ret_r, out_ret_l

            out_ret_r.append(outline_r[i])
                # exchange the remainder between outlines
                #tmp = outline_l[(i + 1):]
                #outline_l[(i + 1):] = outline_r[(i + 1):]
                #outline_r[(i + 1):] = tmp

    return out_ret_l, out_ret_r

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
    u = ((x1-x3)*(y1-y2) - (y1-y3)*(x1-x2)) / denom

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

def smooth_path(coords, pressure=None, threshold=20):
    """Smooth a path using cubic Bézier curves."""

    if len(coords) < 3:
        return coords, pressure  # Not enough points to smooth

    if pressure and len(pressure) != len(coords):
        #raise ValueError("Pressure and coords must have the same length")
        log.warning(f"Pressure and coords must have the same length, however p={len(pressure)} c={len(coords)}")
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

def calc_arc_coords2(p1, p2, c, n = 20):
    """Calculate the coordinates of an arc between points p1 and p2.
       The arc is a fragment of a circle with centre in c."""
    x1, y1 = p1
    x2, y2 = p2
    xc, yc = c

    # calculate the radius of the circle
    radius = math.sqrt((x2 - xc)**2 + (y2 - yc)**2)
    side = determine_side_math(p1, p2, c)

    # calculate the angle between the line p1->c and p1->p2
    a1 = math.atan2(y1 - c[1], x1 - c[0])
    a2 = math.atan2(y2 - c[1], x2 - c[0])

    if side == 'left' and a1 > a2:
        a2 += 2 * math.pi
    elif side == 'right' and a1 < a2:
        a1 += 2 * math.pi

    # calculate 20 points on the arc between a1 and a2
    coords = []
    for i in range(n):
        a = a1 + (a2 - a1) * i / (n - 1)
        x = c[0] + radius * math.cos(a)
        y = c[1] + radius * math.sin(a)
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

