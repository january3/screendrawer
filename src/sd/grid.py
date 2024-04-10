"""Grid class for drawing a grid on screen"""
import gi                                                  # <remove>
gi.require_version('Gtk', '3.0')                           # <remove>
import cairo                                            # <remove>

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
