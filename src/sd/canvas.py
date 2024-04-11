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
    def __init__(self, state, dm, wiglets):
        self.__state = state
        self.__grid = Grid()
        self.__dm = dm
        self.__wiglets = wiglets

    def on_draw(self, widget, cr):
        """Main draw method of the whole app."""
        if self.__state.hidden():
            return
        page = self.__state.current_page()
        tr = page.translate()

        cr.save()

        if tr:
            cr.translate(*tr)

        self.draw_bg(cr, tr)
        page.draw(cr, self.__state)

        cobj = self.__state.current_obj()
        if cobj and not cobj in page.objects_all_layers():
            self.__state.current_obj().draw(cr)

        if self.__dm.selection_tool():
            self.__dm.selection_tool().draw(cr)

        cr.restore()

        #self.__dm.draw(None, cr)

        ws = self.__state.get_win_size()
        for w in self.__wiglets:
            w.update_size(*ws)
            w.draw(cr, self.__state)

       # XXX this does not work.
       #if self.__wiglet_active:
       #    self.__wiglet_active.draw(cr)

        return True

    def draw_bg(self, cr, tr):
        """
        Draw the objects on the page.

        :param objects: The objects to draw.
        :param cr: The context on which to draw.
        :param tr: The translation (paning).
        """
        pass

        bg_color     = self.__state.bg_color()
        transparency = self.__state.alpha()
        show_grid    = self.__state.show_grid()
        size         = self.__state.get_win_size()

        cr.set_source_rgba(*bg_color, transparency)
        cr.set_operator(cairo.OPERATOR_SOURCE)
        cr.paint()
        cr.set_operator(cairo.OPERATOR_OVER)

        if show_grid:
            tr = tr or (0, 0)
            self.__grid.draw(cr, tr, size)
