from gi.repository import Gdk # <remove>
from sd.utils import base64_to_pixbuf # <remove>

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

        pipette = "iVBORw0KGgoAAAANSUhEUgAAABwAAAAbCAYAAABvCO8sAAADJUlEQVRIx7XWS2xVVRQG4O/c20LpxTYNLSaAVagVtYoDG1DQ+IgaFYOmIeJME4hKfIImDBgYlRgiJDghPsAoQdGRGsMAUQgywOADRPFBYiiVKpZauZDS2+c9ThZJY4DUXrpGO+vss/691v7/vRZjZ2V4AZ3owBe4oWyMwKrwMboxBw24Be9nxgCsBtvQFVk2YBpm40CpgNdieIxqfIYfMA8VqAx/FZpLAVyGD/ARHkEWH+JTLEBr+I5iJi7G9lIA78eL6MMVWI5BPIry8PcG8I3hW50tAXA2noiANZiLS+K+vow9x9GEWXgJO5MSACfiLTTGPe5Dc2TSjp7wz4vsN5fKxvtwOoJvx/dYjwHsx++hvabhP422pC1BlsEo5UnMiOCvY1KUcyUOlJrZ9UGGAu4a5r85WPs2FmITPg85jNquipO347qzfF8UGlyNO0MSd48WbGqAdWD6OfbMxF78hV/jHutHAzYeh7HqPAGWxXO2LrixFltGA1aPtgA7356uuMOS7OF48VfBxDd7irn1+cHcyz89N2zPlVG6nSNh/fmEvzjKshA7Kl/55UhSNn5aUjM1I8kmuo7c3r3i8hTvhQZb0D+SJnk2W4kHsQY7JqzY9XVmcmO9oYFEX0HxVEdv4dXbnsLV+CrYWRxpV/6vPR2amjN+6eZLs/WzjxmXqy2ePPa3npN1aX+3/k2PH0zzf9bhGywZKdjZXppn8QBaqjamUyRlewwNTsrUzShLKi/KpT15/VueaRs6vLc9NLZ4JGU8V4ZLoox3TNiYXjN06o+tErWZKU3S013SwimFdfccTztb9wVJlv+fzM7YmX7YjMdwb9U76axMX35bsf1gXaZuhuLx36SdrXpfm9+bdrZ+gkNRieJoaH+GpWvxbfWGdO9Q34n9xUO7q4rlE5Ik1SGbre19Y1FP2t21Jrr2k6WOcpVYIMm0DOTbssWjP1ZDUpE7ZLC/vLBufoWBwi5MxvOlCjuJjj0dksZb85mKcd9lG27KpSfaywd2b+iTppfh3ZBKeiHGun+GrX+OA+yJ9dIxmVqzDXNPBM3X4qGYS5rj0b6g9i9UTPumQhFEOwAAAABJRU5ErkJggg=="

        pipette = base64_to_pixbuf(pipette)

        self._cursors = {
            "hand":        Gdk.Cursor.new_from_name(window.get_display(), "hand1"),
            "move":        Gdk.Cursor.new_from_name(window.get_display(), "hand2"),
            "grabbing":    Gdk.Cursor.new_from_name(window.get_display(), "grabbing"),
            "moving":      Gdk.Cursor.new_from_name(window.get_display(), "grab"),
            "text":        Gdk.Cursor.new_from_name(window.get_display(), "text"),
            "eraser":      Gdk.Cursor.new_from_name(window.get_display(), "not-allowed"),
            "pencil":      Gdk.Cursor.new_from_name(window.get_display(), "pencil"),
            "picker":      Gdk.Cursor.new_from_name(window.get_display(), "crosshair"),
            #"colorpicker": Gdk.Cursor.new_from_name(window.get_display(), "crosshair"),
            "colorpicker": Gdk.Cursor.new_from_pixbuf(window.get_display(), pipette, 1, 26),
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



