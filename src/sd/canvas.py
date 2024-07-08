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

import gi                                                  # <remove>
gi.require_version('Gtk', '3.0')                           # <remove>
from gi.repository import Gtk, Gdk, GdkPixbuf, Pango, GLib # <remove> pylint: disable=wrong-import-position
import cairo                                                   # <remove>
from .grid     import Grid                                       # <remove>

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

    def on_draw(self, widget, cr):
        """Main draw method of the whole app."""

        if self.__state.graphics().hidden():
            return

        page = self.__state.current_page()
        tr = page.translate()

        cr.save()

        if tr:
            cr.translate(*tr)

        self.draw_bg(cr, tr)
        page.draw(cr, self.__state, force_redraw = self.__force_redraw)

        # emit the draw signal for objects that wish to be drawn in draw
        # coordinates
        self.__bus.emit("obj_draw", exclusive = False, cr = cr, state = self.__state)

        cobj = self.__state.current_obj()

        if cobj and not cobj in page.objects_all_layers():
            self.__state.current_obj().draw(cr)

        cr.restore()

        ws = self.__state.get_win_size()

        if ws != self.__winsize:
            self.__winsize = ws
            self.__bus.emit("update_size", exclusive = False, width = ws[0], height = ws[1])

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
        pass

        bg_color     = self.__state.graphics().bg_color()
        transparency = self.__state.graphics().alpha()
        show_grid    = self.__state.graphics().show_grid()
        size         = self.__state.get_win_size()

        cr.set_source_rgba(*bg_color, transparency)
        cr.set_operator(cairo.OPERATOR_SOURCE)
        cr.paint()
        cr.set_operator(cairo.OPERATOR_OVER)

        if show_grid:
            tr = tr or (0, 0)
            self.__grid.draw(cr, tr, size)
