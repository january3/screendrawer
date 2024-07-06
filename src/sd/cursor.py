"""
CursorManager class to manage the cursor.

Basically, this class is responsible for creating and managing the cursor for the window.
It has methods to set the cursor to different modes.
"""

import gi                                                  # <remove>
gi.require_version('Gtk', '3.0')                           # <remove>
gi.require_version('Gdk', '3.0')                           # <remove>
from gi.repository import Gdk         # <remove>
from .icons import Icons            # <remove>
import logging                                                   # <remove>
log = logging.getLogger(__name__)                                # <remove>

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
        """Draw a cursor if necessary"""

        x, y = self.pos_absolute()
        w, h = state.get_win_size()
        cr.save()
        cr.set_source_rgb(0, 1, 0.2)
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
            self.__bus.emitOnce("query_cursor_pos")

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
        log.debug(f"setting default cursor to {cursor_name}")
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
