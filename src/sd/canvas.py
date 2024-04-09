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

import cairo                                                   # <remove>
from .grid     import Grid                                       # <remove>

class Canvas:
    """
    Canvas for drawing shapes and text.
    """
    def __init__(self, status):
        self.__status = status
        self.__grid = Grid()

    def draw(self, cr, tr):
        """
        Draw the objects on the page.

        :param objects: The objects to draw.
        :param cr: The context on which to draw.
        :param tr: The translation (paning).
        """
        pass

        bg_color     = self.__status.bg_color()
        transparency = self.__status.transparent()
        show_grid    = self.__status.show_grid()
        size         = self.__status.get_win_size()

        cr.set_source_rgba(*self.__bg_color, self.__transparency)
        cr.set_operator(cairo.OPERATOR_SOURCE)
        cr.paint()
        cr.set_operator(cairo.OPERATOR_OVER)

        if show_grid:
            tr = tr or (0, 0)
            self.__grid.draw(cr, tr, size)
