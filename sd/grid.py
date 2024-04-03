"""Grid class for drawing a grid on screen"""

class Grid:
    """
    Grid object holds information about how tight a grid is, and how it is drawn.

    app is a necessary argument, because Grid needs to know the current size of 
    the screen to draw the grid properly.
    """
    def __init__(self):
        self.__spacing = 10
        self.__tocks  = 5
        self.__origin = (0, 0)
        self.__color = (.2, .2, .2, .75)
        self.__line_width = 0.2

    def draw(self, cr, tr, size):
        """Draw grid in the current cairo context"""

        width, height = size
        dx, dy = tr

        x0 =  - int(dx / self.__spacing) * self.__spacing
        y0 =  - int(dy / self.__spacing) * self.__spacing

        cr.set_source_rgba(*self.__color)
        cr.set_line_width(self.__line_width/2)

        # draw vertical lines
        x = x0
        i = 1
        while x < width + x0:
            if i == 5:
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
            x += self.__spacing

        # draw horizontal lines
        y = y0
        while y < height + y0:
            if i == 5:
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
            y += self.__spacing


