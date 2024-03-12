#!/usr/bin/env python3

import gi
import yaml
gi.require_version('Gtk', '3.0')

from gi.repository import Gtk, Gdk

import cairo
import math
import os

savefile = os.path.expanduser("~/.screendrawer")
print(savefile)
# open file for appending if exists, or create if not


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

def find_paths_close_to_click(click_x, click_y, paths, threshold):
    """Find all paths close to a click."""
    close_paths = []
    for path in paths:
        if is_click_close_to_path(click_x, click_y, path["coords"], threshold):
            return {"type": "path", "path": path, "origin": (click_x, click_y)}
    return None

def find_text_close_to_click(click_x, click_y, texts, threshold):
    """Find all texts close to a click."""
    for text in texts:
        if "bb" not in text:
            continue
        x, y, width, height = text["bb"]
        if click_x >= x and click_x <= x + width and click_y >= y and click_y <= y + height:
            return {"type": "text", "text": text, "origin": (click_x, click_y)}
    return None

def move_path(path, dx, dy):
    """Move a path by a given offset."""
    for i in range(len(path)):
        path[i] = (path[i][0] + dx, path[i][1] + dy)

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


        # Drawing setup
        self.paths = []
        self.current_path = None
        self.set_events(Gdk.EventMask.BUTTON_PRESS_MASK | Gdk.EventMask.BUTTON_RELEASE_MASK | Gdk.EventMask.POINTER_MOTION_MASK)
        self.texts = []
        self.text_input    = None
        self.changing_line_width = False
        self.selection = None

        # defaults for drawing
        self.transparent = True
        self.font_size  = 24
        self.line_width = 4
        self.color      = (0.2, 0, 0)

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
        if self.transparent:
            cr.set_source_rgba(0, 0, 0, 0)  # Transparent background
        else:
            cr.set_source_rgb(1, 1, 1) # White background
        cr.set_operator(cairo.OPERATOR_SOURCE)
        cr.paint()
        cr.set_operator(cairo.OPERATOR_OVER)
        self.draw(cr)

    def draw(self, cr):

        # Draw the paths
        for path in self.paths:
            cr.set_line_width(path["line_width"])
            cr.set_source_rgb(*path["color"])  # Drawing color
            cr.move_to(path["coords"][0][0], path["coords"][0][1])
            for point in path["coords"][1:]:
                cr.line_to(point[0], point[1])
            cr.stroke()

        # Draw the text
        for text in self.texts:
            position, content, size, color = text["position"], text["content"], text["size"], text["color"]
            cr.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
            cr.set_font_size(size)

            x_bearing, y_bearing, width, height, x_advance, y_advance = cr.text_extents(content)
            text["bb"] = (position[0] + x_bearing, position[1] + y_bearing, width, height)

           #cr.set_line_width(1)
           #cr.rectangle(position[0] + x_bearing - 5, position[1] + y_bearing - 5, width + 10, height + 10)
           #cr.set_source_rgb(1, 0, 0)  # Black for the text outline
           #cr.stroke()

            cr.move_to(position[0], position[1])
            cr.set_source_rgb(*color)
            cr.show_text(content)
            cr.stroke()

        # If changing line width, draw a preview of the new line width
        if self.changing_line_width:
            cr.set_line_width(self.line_width)
            cr.set_source_rgb(*self.color)
            self.draw_dot(cr, *self.cur_pos, self.line_width)

    def save_state(self): 
        config = {
                'transparent': self.transparent,
                'font_size': self.font_size,
                'line_width': self.line_width,
                'color': self.color
        }

        state = { 'config': config, 'texts': self.texts, 'paths': self.paths }
        with open(savefile, 'w') as f:
            yaml.dump(state, f)
        print("Saved drawing to", savefile)

    def load_state(self):
        if not os.path.exists(savefile):
            print("No saved drawing found at", savefile)
            return
        with open(savefile, 'r') as f:
            state = yaml.load(f, Loader=yaml.FullLoader)
        self.texts, self.paths = state['texts'], state['paths']
        self.transparent       = state['config']['transparent']
        self.font_size         = state['config']['font_size']
        self.line_width        = state['config']['line_width']
        self.color             = state['config']['color']

    # Event handlers
    def on_button_press(self, widget, event):
        modifiers = Gtk.accelerator_get_default_mod_mask()

        # Check if the Ctrl key is pressed
        if event.state & modifiers == Gdk.ModifierType.CONTROL_MASK and event.button == 1:  # Ctrl + Left mouse button
            self.cur_pos = (event.x, event.y)
            self.changing_line_width = True

        # Check if the Shift key is pressed
        elif event.state & modifiers == Gdk.ModifierType.SHIFT_MASK and event.button == 1:  # Shift + Left mouse button
            self.change_cursor("text")
            print("Text input mode")
            if not self.texts or not self.texts[-1] or self.texts[-1]["content"] != "|":
                self.texts.append({"position": (event.x, event.y), "content": "|", "size": self.font_size, "color": self.color})
            self.text_input = True
            self.queue_draw()

        # Left mouse button - just drawing
        elif event.button == 1:  # Left mouse button
            obj = self.find_objects(event.x, event.y)
            if event.type == Gdk.EventType.DOUBLE_BUTTON_PRESS:
                print("Double-click detected!")
                double_click = True

            if obj:
                print("Found object of type", obj["type"])
                self.selection = obj

            else:
                self.current_path = { "coords": [ (event.x, event.y) ], "line_width": self.line_width, "color": self.color }
                self.paths.append(self.current_path)

            self.queue_draw()
    # Event handlers
    def on_button_release(self, widget, event):
        """Handle mouse button release events."""
        self.cur_pos      = None
        self.changing_line_width = False
        self.current_path = None

        if self.selection:
            # If the user was dragging a selected object and the drag ends
            # in the lower left corner, delete the object
            if event.x < 10 and event.y > self.get_size()[1] - 10:
                if self.selection["type"] == "path":
                    self.paths.remove(self.selection["path"])
                elif self.selection["type"] == "text":
                    self.texts.remove(self.selection["text"])
                self.selection = None
                self.queue_draw()
        self.selection    = None

    def find_objects(self, x0, y0):
        """Find all objects close to the given position."""
        maybepath = find_paths_close_to_click(x0, y0, self.paths, 10)
        maybetext = find_text_close_to_click(x0, y0, self.texts, 10)
        if maybepath:
            return maybepath
        if maybetext:
            return maybetext
        return None

    def on_motion_notify(self, widget, event):
        """Handle mouse motion events."""
        if self.changing_line_width:
            self.line_width = max(3, min(40, self.line_width + (event.x - self.cur_pos[0])/250))
            self.queue_draw()
        elif self.current_path is not None:
            self.current_path["coords"].append((event.x, event.y))
            self.queue_draw()
        elif self.selection is not None:
            dx = event.x - self.selection["origin"][0]
            dy = event.y - self.selection["origin"][1]

            # Move the selected object
            if self.selection["type"] == "path":
                move_path(self.selection["path"]["coords"], dx, dy)
                self.selection["origin"] = (event.x, event.y)
                self.queue_draw()
            elif self.selection["type"] == "text":
                self.selection["text"]["position"] = (self.selection["text"]["position"][0] + dx, self.selection["text"]["position"][1] + dy)
                self.selection["origin"] = (event.x, event.y)
                self.queue_draw()
        else:
            object_underneath = self.find_objects(event.x, event.y)

            if object_underneath:
                self.change_cursor("hand")
            else:
                self.change_cursor("pencil")

    def finish_text_input(self):
        """Clean up current text and finish text input."""
        if self.texts and self.texts[-1]["content"] and self.texts[-1]["content"][-1] == "|":
            self.texts[-1]["content"] = self.texts[-1]["content"][:-1]
        self.text_input = False
        self.change_cursor("pencil")
        self.queue_draw()

    def update_text_input(self, char, keyname):
        """Update the current text input."""
        text = self.texts[-1]["content"][:-1]

        if keyname == "BackSpace":
            text = text[:-1]
        elif keyname == "Return":
            pass
        else:
            text += char
        self.texts[-1]["content"] = text + '|'
        self.queue_draw()

    def handle_shortcuts(self, keyname):
        """Handle keyboard shortcuts."""
        if keyname == "c" or keyname == "q":
            self.exit()
        elif keyname == "b":
            self.transparent = not self.transparent
            self.queue_draw()
        elif keyname == "k":
            self.select_color()
        elif keyname == "l":
            self.selection = None
            self.text_input = False
            self.paths = []
            self.texts = []
            self.queue_draw()
        elif keyname == "plus":
            if self.text_input:
                self.texts[-1]["size"] += 1
                print("increasing current font size to", self.texts[-1]["size"])
                self.queue_draw()
            else:
                self.font_size += 1
                print("increasing default font size to", self.font_size)
        elif keyname == "minus":
            if self.text_input:
                self.texts[-1]["size"] = max(1, self.texts[-1]["size"] - 1)
                print("decreasing current font size to", self.texts[-1]["size"])
                self.queue_draw()
            else:
                self.font_size = max(1, self.font_size - 1)
                print("decreasing default font size to", self.font_size)
        elif keyname == "s":
                self.save_drawing()
 
    def on_key_press(self, widget, event):
        """Handle keyboard events."""
        keyname = Gdk.keyval_name(event.keyval)
        char = chr(Gdk.keyval_to_unicode(event.keyval))
        #print(keyname)

        # End text input
        if keyname == "Return" or keyname == "Escape":
            self.finish_text_input()

        # Handle keyboard shortcuts
        elif event.state & Gdk.ModifierType.CONTROL_MASK:
            self.handle_shortcuts(keyname)
       
        # Handle text input
        elif self.texts and self.text_input and (char and char.isprintable() or keyname in ["BackSpace", "Return"]):
            self.update_text_input(char, keyname)

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
            "hand": Gdk.Cursor.new_from_name(self.get_display(), "hand1"),
            "text": Gdk.Cursor.new_from_name(self.get_display(), "text"),
            "pencil": Gdk.Cursor.new_from_name(self.get_display(), "pencil"),
            "crosshair": Gdk.Cursor.new_from_name(self.get_display(), "crosshair"),
            "default": Gdk.Cursor.new_from_name(self.get_display(), "pencil")
        }

    def change_cursor(self, cursor_name):
        cursor = self.cursors[cursor_name]
        self.get_window().set_cursor(cursor)

    def select_color(self):
        # Create a new color chooser dialog
        color_chooser = Gtk.ColorChooserDialog("Select a Color", None)

        # Show the dialog
        response = color_chooser.run()

        # Check if the user clicked the OK button
        if response == Gtk.ResponseType.OK:
            # Get the selected color
            color = color_chooser.get_rgba()
            # Convert the color to a string (or use it directly in your application)
            color_str = "RGBA: ({:.2f}, {:.2f}, {:.2f}, {:.2f})".format(color.red, color.green, color.blue, color.alpha)
            self.color = (color.red, color.green, color.blue)
            print(color_str)
        else:
            print("Color selection canceled.")

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
            if not filename.endswith('.png'):
                filename += '.png'
            self.export_to_png(filename)
        dialog.destroy()

    def export_to_png(self, filename):
        # Create a Cairo surface of the same size as the window content
        width, height = self.get_size()
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        cr = cairo.Context(surface)

        # Redraw your window content here on the cr
        # This should mirror the drawing code in your on_draw method, e.g.,
        cr.set_source_rgba(1, 1, 1)
        cr.paint()
        #cr.set_source_rgb(0, 0, 0)  # Example drawing setup
        # Insert your drawing code here, similar to the on_draw method
        self.draw(cr)

        # Save the surface to the file
        surface.write_to_png(filename)

if __name__ == "__main__":
    win = TransparentWindow()
    win.show_all()
    win.present()
    win.change_cursor("default")
    Gtk.main()

