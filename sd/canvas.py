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

from sd.pen      import Pen                                        # <remove>

class Canvas:
    """
    Canvas for drawing shapes and text.
    """
    def __init__(self):
        self.__bg_color = (.8, .75, .65)
        self.__transparency = 0
        self.__outline = False

        self.__pen  = Pen(line_width = 4,  color = (0.2, 0, 0), font_size = 24, transparency  = 1)
        self.__pen2 = Pen(line_width = 40, color = (1, 1, 0),   font_size = 24, transparency = .2)


    def pen(self, alternate = False, pen = None):
        """Get or set the pen."""
        if pen:
            self.__pen_set(pen, alternate)
        return self.__pen2 if alternate else self.__pen

    def __pen_set(self, pen, alternate = False):
        """Set the pen."""
        if alternate:
            self.__pen2 = pen
        else:
            self.__pen = pen

    def switch_pens(self):
        """Switch between pens."""
        self.__pen, self.__pen2 = self.__pen2, self.__pen

    def apply_pen_to_bg(self):
        """Apply the pen to the background."""
        self.__bg_color = self.__pen.color

    def cycle_background(self):
        """Cycle through background transparency."""
        self.__transparency = {1: 0, 0: 0.5, 0.5: 1}[self.__transparency]

    def outline(self):
        """Get the outline mode."""
        return self.__outline

    def outline_toggle(self):
        """Toggle outline mode."""
        self.__outline = not self.__outline

    def bg_color(self, color=None):
        """Get or set the background color."""
        if color:
            self.__bg_color = color
        return self.__bg_color

    def transparent(self, value=None):
        """Get or set the transparency."""
        if value:
            self.__transparency = value
        return self.__transparency


