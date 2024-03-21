from gi.repository import Gdk # <remove>

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
            "polygon":     Gdk.Cursor.new_from_name(window.get_display(), "pencil"),
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



