from .pen import Pen                           # <remove>
from .utils import draw_dot, is_click_in_bbox  # <remove>

import colorsys                                # <remove>

def adjust_color_brightness(rgb, factor):
    """
    Adjust the color brightness.
    :param rgb: A tuple of (r, g, b) in the range [0, 1]
    :param factor: Factor by which to adjust the brightness (>1 to make lighter, <1 to make darker)
    :return: Adjusted color as an (r, g, b) tuple in the range [0, 1]
    """
    # Convert RGB to HLS
    print("rgb:", rgb)
    h, l, s = colorsys.rgb_to_hls(*rgb)
    
    # Adjust lightness
    l = max(min(l * factor, 1), 0)  # Ensure lightness stays within [0, 1]
    newrgb = colorsys.hls_to_rgb(h, l, s)
    print("newrgb:", newrgb)
    
    # Convert back to RGB
    return newrgb


## ---------------------------------------------------------------------
class Wiglet:
    """drawable dialog-like objects on the canvas"""
    def __init__(self, type, coords):
        self.wiglet_type   = type
        self.coords = coords

    def update_size(self, width, height):
        raise NotImplementedError("update size method not implemented")

    def draw(self, cr):
        raise NotImplementedError("draw method not implemented")

    def event_update(self, x, y):
        raise NotImplementedError("event_update method not implemented")

    def event_finish(self):
        raise NotImplementedError("event_finish method not implemented")

class WigletTransparency(Wiglet):
    """Wiglet for changing the transparency."""
    def __init__(self, coords, pen):
        super().__init__("transparency", coords)

        if not pen or not isinstance(pen, Pen):
            raise ValueError("Pen is not defined or not of class Pen")

        self.pen      = pen
        self._last_pt = coords[0]
        self._initial_transparency = pen.transparency
        print("initial transparency:", self._initial_transparency)

    def draw(self, cr):
        cr.set_source_rgba(*self.pen.color, self.pen.transparency)
        draw_dot(cr, *self.coords, 50)

    def event_update(self, x, y):
        dx = x - self.coords[0]
        #print("changing transparency", dx)
        ## we want to change the transparency by 0.1 for every 20 pixels
        self.pen.transparency = max(0, min(1, self._initial_transparency + dx/500))
        #print("new transparency:", self.pen.transparency)

    def event_finish(self):
        pass

class WigletLineWidth(Wiglet):
    """Wiglet for changing the line width."""
    """directly operates on the pen of the object"""
    def __init__(self, coords, pen):
        super().__init__("line_width", coords)

        if not pen or not isinstance(pen, Pen):
            raise ValueError("Pen is not defined or not of class Pen")
        self.pen      = pen
        self._last_pt = coords[0]
        self._initial_width = pen.line_width

    def draw(self, cr):
        cr.set_source_rgb(*self.pen.color)
        draw_dot(cr, *self.coords, self.pen.line_width)

    def event_update(self, x, y):
        dx = x - self.coords[0]
        print("changing line width", dx)
        self.pen.line_width = max(1, min(60, self._initial_width + dx/20))
        return True

    def event_finish(self):
        pass

## ---------------------------------------------------------------------


class WigletColorSelector(Wiglet):
    """Wiglet for selecting the color."""
    def __init__(self, coords = (0, 0), width = 30, height = 500, func_color = None, func_bg = None):
        super().__init__("color_selector", coords)
        print("height:", height)

        self.__width, self.__height = width, height
        self.__bbox = (coords[0], coords[1], width, height)
        self.__colors = self.generate_colors()
        self.__dh = 25
        self.__func_color = func_color
        self.__func_bg    = func_bg
        self.recalculate()

        for color, ypos in self.__colors_hpos.items():
            print("color:", color, "ypos:", ypos)

    def recalculate(self):
        self.__bbox = (self.coords[0], self.coords[1], self.__width, self.__height)
        self.__color_dh = (self.__height - self.__dh) / len(self.__colors)
        self.__colors_hpos = { color : self.__dh + i * self.__color_dh for i, color in enumerate(self.__colors) }

    def update_size(self, width, height):
        _, self.__height = width, height
        self.recalculate()

    def draw(self, cr):
        # draw grey rectangle around my bbox
        cr.set_source_rgb(0.5, 0.5, 0.5)
        cr.rectangle(*self.__bbox)
        cr.fill()

        bg, fg = (0, 0, 0), (1, 1, 1)
        if self.__func_bg and callable(self.__func_bg):
            bg = self.__func_bg()
        if self.__func_color and callable(self.__func_color):
            fg = self.__func_color()

        print("current fg/bg:", fg, bg)

        # draw the colors
        dh = 25
        h = (self.__height - dh)/ len(self.__colors)
        for i, color in enumerate(self.__colors):
            cr.set_source_rgb(*color)
            cr.rectangle(self.__bbox[0] + 1, self.__colors_hpos[color], self.__width - 2, h)
            cr.fill()

    def on_click(self, x, y, ev):
        if not is_click_in_bbox(x, y, self.__bbox):
            return False
        print("clicked inside the bbox")

        dy = y - self.__bbox[1]
        # which color is at this position?
        print("dy:", dy)
        sel_color = None
        for color, ypos in self.__colors_hpos.items():
            print("color:", color, "ypos:", ypos)
            if ypos <= dy <= ypos + self.__color_dh:
                print("selected color:", color)
                sel_color = color
        if ev.shift():
            print("setting bg to color", sel_color)
            if sel_color and self.__func_bg and callable(self.__func_bg):
                self.__func_bg(sel_color)
        else:
            print("setting fg to color", sel_color)
            if sel_color and self.__func_color and callable(self.__func_color):
                self.__func_color(sel_color)
        return True


    def event_update(self, x, y):
        dx = x - self.coords[0]
        print("changing color", dx)
        ## we want to change the transparency by 0.1 for every 20 pixels
        self.pen.color = (self.pen.color[0] + dx/1000, self.pen.color[1], self.pen.color[2])
        print("new color:", self.pen.color)

    def event_finish(self):
        pass


    def generate_colors(self):
        """
        Generate a rainbow of 24 colors.
        """

        # list of 24 colors forming a rainbow
        colors = [  (0.88, 0.0, 0.83),
                     (0.29, 0.0, 0.51),
                     (0.0, 0.0, 1.0),
                     (0.0, 0.6, 1.0),
                     (0.0, 1.0, 0.4),
                     (0.0, 1.0, 0.0),
                     (1.0, 1.0, 0.0),
                     (1.0, 0.7, 0.0),
                     (1.0, 0.3, 0.0),
                     (1.0, 0.0, 0.0)]

        newc = [ ]
        for i in range(11):
            print(i/10)

            newc.append((i/10, i/10, i/10))
        for c in colors:
            print("color:", c)
            lighter = adjust_color_brightness(c, 1.5)
            for dd in range(2, 20, 2): #
                d = dd / 10
                newc.append(adjust_color_brightness(c, d))

        return newc
