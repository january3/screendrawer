#!/usr/bin/env python3

import gi
import yaml
gi.require_version('Gtk', '3.0')

from gi.repository import Gtk, Gdk
import cairo
import os
import time
import math

savefile = os.path.expanduser("~/.screendrawer")
print(savefile)
# open file for appending if exists, or create if not

import numpy as np

def perpendicular_vector(v):
    """Calculate a vector perpendicular to the given vector."""
    return np.array([-v[1], v[0]])

def unit_vector(v):
    """Normalize the given vector."""
    return v / np.linalg.norm(v)

import numpy as np

def calculate_outline(points, lwd):
    polygon_outline = []

    cleaned_up = []

    for i in range(len(points) - 1):
        p0, p1 = points[i], points[i + 1]
        x0, y0 = p0
        x1, y1 = p1
        w      = lwd[i]

        # Calculate the direction vector of the segment
        dx, dy = x1 - x0, y1 - y0
        length = np.sqrt(dx**2 + dy**2)
        if(length > 0):
            dx, dy = dx / length, dy / length
            cleaned_up.append((x0, y0, x1, y1, -dy, dx, w))

    print("points length: ", len(points), " cleaned up length: ", len(cleaned_up))

    for i in range(len(cleaned_up) - 1):
        x0, y0, x1, y1, nx, ny, w = cleaned_up[i]

        # Calculate the offset due to the line width
        offset = w / 2.0 * np.array([nx, ny])

        # Calculate the four corners of the outline for this segment
        corner1 = np.array([x0, y0]) + offset
        corner2 = np.array([x0, y0]) - offset
        corner3 = np.array([x1, y1]) - offset
        corner4 = np.array([x1, y1]) + offset

        print(p0, p1, length, nx, ny, corner1, corner2, corner3, corner4, w)

        # Append corners to the outline
        # Ensure we're not introducing NaNs and the format is correct
        if not np.any(np.isnan(corner1)) and not np.any(np.isnan(corner2)):
            polygon_outline.extend([tuple(corner1), tuple(corner2), tuple(corner3), tuple(corner4)])
        else:
            print("NaNs in corner coordinates")

    return polygon_outline


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

def is_click_close_to_path(click_x, click_y, path, threshold):
    """Check if a click is close to any segment in the path."""
    for i in range(len(path) - 1):
        segment_start = path[i]
        segment_end = path[i + 1]
        distance = distance_point_to_segment(click_x, click_y, segment_start[0], segment_start[1], segment_end[0], segment_end[1])
        if distance <= threshold:
            return True
    return False

def is_text_close_to_click(click_x, click_y, text, threshold):
    if "bb" not in text:
        return False
    x, y, width, height = text["bb"]
    if click_x >= x and click_x <= x + width and click_y >= y and click_y <= y + height:
        return True

def is_box_close_to_click(click_x, click_y, coords, threshold):
    x1, y1 = coords[0]
    x2, y2 = coords[1]

    path = [ (x1, y1), (x1, y2), (x2, y2), (x2, y1), (x1, y1) ]
    return is_click_close_to_path(click_x, click_y, path, threshold)

def find_obj_close_to_click(click_x, click_y, objects, threshold):
    for obj in objects:
        if obj["type"] == "path":
            if is_click_close_to_path(click_x, click_y, obj["coords"], threshold):
                return obj
        elif obj["type"] == "text":
            if is_text_close_to_click(click_x, click_y, obj, threshold):
                return obj
        elif obj["type"] in ["box", "circle"]:
            if is_box_close_to_click(click_x, click_y, obj["coords"], threshold):
                return obj
    return None

def normal_vec(x0, y0, x1, y1):
    dx, dy = x1 - x0, y1 - y0
    length = math.sqrt(dx**2 + dy**2)
    dx, dy = dx / length, dy / length
    return -dy, dx

def path_append(path, x, y, width):
    coords = path["coords"]

    if len(coords) == 0:
        coords.append((x, y))
    lp = coords[-1]
    if abs(x - lp[0]) < 2 and abs(y - lp[1]) < 2:
        return

    coords.append((x, y))
    width = width / 2

    if len(coords) == 2:
        p1, p2 = coords[0], coords[1]
        nx, ny = normal_vec(p1[0], p1[1], p2[0], p2[1])
        path["outline_l"].append((p1[0] + nx * width, p1[1] + ny * width))
        path["outline_l"].append((p2[0] + nx * width, p2[1] + ny * width))
        path["outline_r"].append((p1[0] - nx * width, p1[1] - ny * width))
        path["outline_r"].append((p2[0] - nx * width, p2[1] - ny * width))
    if len(coords) > 2:
        p1, p2 = coords[-2], coords[-1]
        nx, ny = normal_vec(p1[0], p1[1], p2[0], p2[1])
        path["outline_l"].append((p1[0] + nx * width, p1[1] + ny * width))
        path["outline_r"].append((p1[0] - nx * width, p1[1] - ny * width))
    if len(coords) >= 2:
        path["outline"] = path["outline_l"] + path["outline_r"][::-1]



def move_coords(coords, dx, dy):
    """Move a path by a given offset."""
    for i in range(len(coords)):
        coords[i] = (coords[i][0] + dx, coords[i][1] + dy)

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
        self.current_text = None
        self.changing_line_width = False
        self.selection = None
        self.mode      = "default"
        self.current_cursor = None
        self.hover = None

        # defaults for drawing
        self.transparent = 0
        self.font_size  = 24
        self.line_width = 4
        self.color      = (0.2, 0, 0)

        # distance for selecting objects
        self.max_dist   = 15

        self.load_state()

        self.connect("button-press-event", self.on_button_press)
        self.connect("button-release-event", self.on_button_release)
        self.connect("motion-notify-event", self.on_motion_notify)

        self.make_cursors()
        self.set_keep_above(True)
        self.maximize()

    def exit(self):
        ## close the savefile_f
        print("Exiting")
        self.save_state()
        Gtk.main_quit()
        

    def on_draw(self, widget, cr):
        cr.set_source_rgba(1, 1, 1, self.transparent)
        cr.set_operator(cairo.OPERATOR_SOURCE)
        cr.paint()
        cr.set_operator(cairo.OPERATOR_OVER)
        self.draw(cr)

    def draw_text(self, cr, text, hover):
        position, content, size, color, cursor_pos = text["coords"][0], text["content"], text["size"], text["color"], text["cursor_pos"]
        cr.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        cr.set_font_size(size)

        dy   = 0
        maxw = 0
        bb_x = position[0]
        bb_y = None
        bb_w = 0
        bb_h = 0
        
        for i in range(len(content)):
            fragment = content[i]

            if cursor_pos != None and self.current_text == text and i == text["line"]:
                fragment = fragment[:cursor_pos] + "|" + fragment[cursor_pos:]

            x_bearing, y_bearing, width, height, x_advance, y_advance = cr.text_extents(fragment)
            text["bb"] = (position[0] + x_bearing, position[1] + y_bearing + dy, width, height)

            if bb_y == None:
                bb_y = position[1] + y_bearing

            bb_w = max(bb_w, width)
            bb_h += 1.5 * height

            cr.set_font_size(size)
            cr.move_to(position[0], position[1] + dy)
            cr.set_source_rgb(*color)
            cr.show_text(fragment)
            cr.stroke()

            dy += y_advance + 1.5 * height
        text["bb"] = (bb_x, bb_y, bb_w, bb_h)
        if hover:
            cr.set_line_width(1)
            cr.rectangle(bb_x, bb_y, bb_w, bb_h)
            cr.stroke()

    def draw_path(self, cr, path, hover):
        if len(path["outline"]) < 4:
            return
        if len(path["coords"]) < 3:
            return
        if hover:
            dd = 1
        else:
            dd = 0
        cr.set_source_rgb(*path["color"])
        cr.move_to(path["outline"][0][0] + dd, path["outline"][0][1] + dd)
        for point in path["outline"][1:]:
            cr.line_to(point[0] + dd, point[1] + dd)
        cr.close_path()
        cr.fill()

    def draw_path_outline(self, cr, path, hover):
        lwd = path["lwd"]

        if len(path["coords"]) == 1:
            return

        cr.set_line_width(1)
        cr.set_source_rgb(*path["color"])

        poly = calculate_outline(path["coords"], lwd)

        x0, y0 = path["coords"][0]
        cr.move_to(x0, y0)
        i = 0
        for point in path["coords"][1:]:
            cr.line_to(point[0], point[1])
            x0, y0 = point
            i += 1
        cr.stroke()
        print (poly)

    def draw_path_bad(self, cr, path, hover):
        lwd = path["lwd"]
        if hover:
            base_width = path["line_width"] + 1
            cr.set_line_width(path["line_width"] + 1)
        else:
            base_width = path["line_width"]
            cr.set_line_width(path["line_width"])

        cr.set_source_rgb(*path["color"])
        x0, y0 = path["coords"][0]
        i = 0
        for point in path["coords"][1:]:
            cr.move_to(x0, y0)
            cr.set_line_width(lwd[i] * base_width)
            cr.line_to(point[0], point[1])
            x0, y0 = point
            cr.stroke()
            i += 1

    def draw_circle(self, cr, box, hover):
        if hover:
            cr.set_line_width(box["line_width"] + 1)
        else:
            cr.set_line_width(box["line_width"])
        cr.set_source_rgb(*box["color"])
        x1, y1 = box["coords"][0]
        x2, y2 = box["coords"][1]
        w, h = (abs(x1 - x2), abs(y1 - y2))
        x0, y0 = (min(x1, x2), min(y1, y2))
        #cr.rectangle(x0, y0, w, h)
        cr.save()
        cr.translate(x0 + w / 2, y0 + h / 2)
        cr.scale(w / 2, h / 2)
        cr.arc(0, 0, 1, 0, 2 * 3.14159)
        cr.restore()
        cr.stroke()

    def draw_box(self, cr, box, hover):
        if hover:
            cr.set_line_width(box["line_width"] + 1)
        else:
            cr.set_line_width(box["line_width"])
        cr.set_source_rgb(*box["color"])
        x1, y1 = box["coords"][0]
        x2, y2 = box["coords"][1]
        w, h = (abs(x1 - x2), abs(y1 - y2))
        x0, y0 = (min(x1, x2), min(y1, y2))
        cr.rectangle(x0, y0, w, h)
        cr.stroke()

    def draw_object(self, cr, obj, hover):
        if obj["type"] == "path":
            self.draw_path(cr, obj, hover)
        elif obj["type"] == "text":
            self.draw_text(cr, obj, hover)
        elif obj["type"] == "box":
            self.draw_box(cr, obj, hover)
        elif obj["type"] == "circle":
            self.draw_circle(cr, obj, hover)

    def draw(self, cr):

        for obj in self.objects:
            hover = False
            if obj == self.hover:
                hover = True
            self.draw_object(cr, obj, hover)

        # If changing line width, draw a preview of the new line width
        if self.changing_line_width:
            cr.set_line_width(self.line_width)
            cr.set_source_rgb(*self.color)
            self.draw_dot(cr, *self.cur_pos, self.line_width)

    def clear(self):
        self.selection      = None
        self.current_text   = None
        self.current_object = None
        self.objects = []
        self.queue_draw()

    def save_state(self): 
        config = {
                'transparent': self.transparent,
                'font_size': self.font_size,
                'line_width': self.line_width,
                'color': self.color
        }

        state = { 'config': config, 'objects': self.objects }
        with open(savefile, 'w') as f:
            yaml.dump(state, f)
        print("Saved drawing to", savefile)

    def load_state(self):
        if not os.path.exists(savefile):
            print("No saved drawing found at", savefile)
            return
        with open(savefile, 'r') as f:
            state = yaml.load(f, Loader=yaml.FullLoader)
        self.objects           = state['objects']
        self.transparent       = state['config']['transparent']
        self.font_size         = state['config']['font_size']
        self.line_width        = state['config']['line_width']
        self.color             = state['config']['color']

    # Event handlers
    def on_button_press(self, widget, event):
        modifiers = Gtk.accelerator_get_default_mod_mask()
        obj = find_obj_close_to_click(event.x, event.y, self.objects, self.max_dist)
        shift = event.state & modifiers == Gdk.ModifierType.SHIFT_MASK
        ctrl  = event.state & modifiers == Gdk.ModifierType.CONTROL_MASK
        print("mode:", self.mode)

        # Ignore clicks when text input is active
        if self.current_text:
            print("click, but text input active")
            return

        # Start changing line width
        if ctrl and event.button == 1:  # Ctrl + Left mouse button
            self.cur_pos = (event.x, event.y)
            self.changing_line_width = True
            return

        elif self.mode == "box" and event.button == 1 and not self.current_object:
            print("drawing box / circle")
            self.current_object = { "type": "box",
                                    "coords": [ (event.x, event.y), (event.x + 1, event.y + 1) ], 
                                    "line_width": self.line_width, 
                                    "color": self.color }
            self.objects.append(self.current_object)
            self.queue_draw()

        elif self.mode == "circle" and event.button == 1 and not self.current_object:
            print("drawing circle")
            self.current_object = { "type": "circle",
                                    "coords": [ (event.x, event.y), (event.x + 1, event.y + 1) ], 
                                    "line_width": self.line_width, 
                                    "color": self.color }
            self.objects.append(self.current_object)
            self.queue_draw()

        # Check if the Shift key is pressed
        elif (shift or self.mode == "text") and event.button == 1:  # Shift + Left mouse button
            self.change_cursor("none")

            if not self.current_text:
                self.current_text = { "coords": [ (event.x, event.y) ], 
                                      "type": "text",
                                      "content": [ "" ], 
                                      "line": 0,
                                      "size": self.font_size, 
                                      "color": self.color, 
                                      "cursor_pos": 1 }
                self.objects.append(self.current_text)
            self.queue_draw()

        # Left mouse button - just drawing
        elif ((event.button == 1 and self.mode == "move") or event.button == 3) and obj:  # Left mouse button
            self.selection = obj
            self.selection["origin"] = (event.x, event.y)

        elif obj and event.button == 1 and self.mode == "eraser":
            self.objects.remove(obj)
            self.selection = None
            self.revert_cursor()
            self.queue_draw()

        elif (self.mode == "draw" or self.mode == "default") and event.button == 1:  # Left mouse button
            if event.type == Gdk.EventType.DOUBLE_BUTTON_PRESS:
                if obj and obj["type"] == "text":
                    # put the cursor in the last line, end of the text
                    obj["line"] = len(obj["content"]) - 1
                    obj["cursor_pos"] = len(obj["content"][ obj["line"] ])
                    self.current_text = obj
                    self.queue_draw()
                    self.change_cursor("none")
            elif not self.current_object:
                self.current_object = { "type": "path",
                                      "coords": [ (event.x, event.y) ], 
                                      "outline_l": [],
                                      "outline_r": [],
                                      "outline": [],
                                      "line_width": self.line_width, 
                                      "lwd": [ ],
                                      "color": self.color }
                self.objects.append(self.current_object)

        self.queue_draw()

    # Event handlers
    def on_button_release(self, widget, event):
        """Handle mouse button release events."""
        obj = self.current_object
        if obj and obj["type"] == "path":
            print("finishing path")
            path_append(obj, event.x, event.y, 0)
            self.queue_draw()

        self.cur_pos      = None
        self.changing_line_width = False
        self.current_object = None

        if self.selection:
            # If the user was dragging a selected object and the drag ends
            # in the lower left corner, delete the object
            self.selection["origin"] = None
            if event.x < 10 and event.y > self.get_size()[1] - 10:
                self.objects.remove(self.selection)
                self.selection = None
                self.queue_draw()
        self.selection    = None


    def on_motion_notify(self, widget, event):
        """Handle mouse motion events."""
        obj = self.current_object
        if self.changing_line_width:
            self.line_width = max(3, min(40, self.line_width + (event.x - self.cur_pos[0])/250))
            self.queue_draw()
        elif obj and (obj["type"] == "box" or obj["type"] == "circle"):
            obj["coords"][1] = (event.x, event.y)
            self.queue_draw()
        elif obj and obj["type"] == "path":
            pressure = event.get_axis(Gdk.AxisUse.PRESSURE)
            if pressure is None:
                pressure = 1.0
            path_append(obj, event.x, event.y, obj["line_width"] * pressure)
            self.queue_draw()
        elif self.selection is not None:
            dx = event.x - self.selection["origin"][0]
            dy = event.y - self.selection["origin"][1]

            # Move the selected object
            move_coords(self.selection["coords"], dx, dy)
            if self.selection["type"] == "path":
                move_coords(self.selection["outline"], dx, dy)
            self.selection["origin"] = (event.x, event.y)
            self.queue_draw()
        else:
            object_underneath = find_obj_close_to_click(event.x, event.y, self.objects, self.max_dist)

            prev_hover = self.hover
            if object_underneath:
                if self.mode == "move":
                    self.change_cursor("hand")
                self.hover = object_underneath
            else:
                if self.mode == "move":
                    self.revert_cursor()
                self.hover = None
            if prev_hover != self.hover:
                self.queue_draw()

    def finish_text_input(self):
        """Clean up current text and finish text input."""
        if self.current_text:
            self.current_text["cursor_pos"] = None
            self.current_text = None
        self.revert_cursor()
        self.queue_draw()

    def update_text_input(self, keyname, char):
        """Update the current text input."""
        cur  = self.current_text
        text = cur["content"][ cur["line"] ]
        # length of text
    
        if keyname == "BackSpace" and cur["cursor_pos"] > 0:
            text = text[:cur["cursor_pos"] - 1] + text[cur["cursor_pos"]:]
            cur["cursor_pos"] -= 1
        elif keyname == "Right":
            cur["cursor_pos"] = min(cur["cursor_pos"] + 1, len(text))
        elif keyname == "Left":
            cur["cursor_pos"] = max(cur["cursor_pos"] - 1, 0)
        elif keyname == "Return":
            cur["line"] += 1
            cur["content"].insert(cur["line"], "")
            text = ""
            cur["cursor_pos"] = 0
        elif char and char.isprintable():
            text = text[:cur["cursor_pos"]] + char + text[cur["cursor_pos"]:]
            cur["cursor_pos"] += 1
        cur["content"][ cur["line"] ] = text
        self.queue_draw()

    def handle_shortcuts(self, keyname, ctrl):
        """Handle keyboard shortcuts."""
        if not ctrl:
            if keyname == 'd':
                print("Default mode")
                self.mode = "default"
                self.default_cursor("pencil")
            elif keyname == 't':
                print("Text input mode")
                self.mode = "text"
                self.default_cursor("text")
            elif keyname == 'e':
                print("Eraser mode")
                self.mode = "eraser"
                self.default_cursor("eraser")
            elif keyname == 'm':
                print("Move mode")
                self.mode = "move"
                self.default_cursor("move")
            elif keyname == 'c':
                print("Circle drawing mode")
                self.default_cursor("crosshair")
                self.mode = "circle"
            elif keyname == 'b':
                print("Box / circle drawing mode")
                self.default_cursor("crosshair")
                self.mode = "box"
        else:
            if keyname == "c" or keyname == "q":
                self.exit()
            elif keyname == "b":
                if self.transparent == 1:
                    self.transparent = 0
                elif self.transparent == 0:
                    self.transparent = .5
                else:
                    self.transparent = 1
                self.queue_draw()
            elif keyname == "k":
                self.select_color()
            elif keyname == "l":
                self.clear()
            elif self.current_text:
                if keyname == "plus":
                    self.current_text["size"] += 1
                    self.font_size = self.current_text["size"]
                    self.queue_draw()
                elif keyname == "minus":
                    self.current_text["size"] = max(1, self.current_text["size"] - 1)
                    self.font_size = self.current_text["size"]
                    self.queue_draw()
            elif keyname == "s":
                    self.save_drawing()
     
    def on_key_press(self, widget, event):
        """Handle keyboard events."""
        keyname = Gdk.keyval_name(event.keyval)
        char    = chr(Gdk.keyval_to_unicode(event.keyval))
        #print(keyname)

        # End text input
        if keyname == "Escape":
            self.finish_text_input()

        # Handle keyboard shortcuts
        elif event.state & Gdk.ModifierType.CONTROL_MASK:
            self.handle_shortcuts(keyname, True)
       
        # Handle text input
        elif self.current_text:
            self.update_text_input(keyname, char)

        else:
            self.handle_shortcuts(keyname, False)

    def draw_dot(self, cr, x, y, diameter):
        """Draws a dot at the specified position with the given diameter.
        
        Args:
            cr: The Cairo context to draw on.
            x (float): The x-coordinate of the center of the dot.
            y (float): The y-coordinate of the center of the dot.
            diameter (float): The diameter of the dot.
        """
        cr.arc(x, y, diameter / 2, 0, 2 * 3.14159)  # Draw a circle
        cr.fill()  # Fill the circle to make a dot

    def make_cursors(self):
        self.cursors = {
            "hand":      Gdk.Cursor.new_from_name(self.get_display(), "hand1"),
            "finger":    Gdk.Cursor.new_from_name(self.get_display(), "hand2"),
            "move":    Gdk.Cursor.new_from_name(self.get_display(), "move"),
            "text":      Gdk.Cursor.new_from_name(self.get_display(), "text"),
            "eraser":      Gdk.Cursor.new_from_name(self.get_display(), "not-allowed"),
            "pencil":    Gdk.Cursor.new_from_name(self.get_display(), "pencil"),
            "crosshair": Gdk.Cursor.new_from_name(self.get_display(), "crosshair"),
            "none": Gdk.Cursor.new_from_name(self.get_display(), "none"),
            "default":   Gdk.Cursor.new_from_name(self.get_display(), "pencil")
        }

    def revert_cursor(self):
        if self.current_cursor == "default":
            return
        print("reverting cursor")
        self.get_window().set_cursor(self.cursors["default"])
        self.current_cursor = "default"

    def change_cursor(self, cursor_name):
        if self.current_cursor == cursor_name:
            return
        print("changing cursor to", cursor_name)
        cursor = self.cursors[cursor_name]
        self.get_window().set_cursor(cursor)
        self.current_cursor = cursor_name

    def default_cursor(self, cursor_name):
        if self.current_cursor == cursor_name:
            return
        print("setting default cursor to", cursor_name)
        self.cursors["default"] = self.cursors[cursor_name]
        self.get_window().set_cursor(self.cursors["default"])
        self.current_cursor = cursor_name

    def select_color(self):
        # Create a new color chooser dialog
        color_chooser = Gtk.ColorChooserDialog("Select Current Foreground Color", None)

        # Show the dialog
        response = color_chooser.run()

        # Check if the user clicked the OK button
        if response == Gtk.ResponseType.OK:
            color = color_chooser.get_rgba()
            self.color = (color.red, color.green, color.blue)

        # Don't forget to destroy the dialog
        color_chooser.destroy()

    def save_drawing(self):
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

if __name__ == "__main__":
    win = TransparentWindow()
    win.show_all()
    win.present()
    win.change_cursor("default")
    win.stick()

    Gtk.main()

