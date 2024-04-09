"""
Wiglets are small dialog-like objects that can be drawn on the canvas.
They are used to provide interactive controls for changing drawing properties
such as line width, color, and transparency.
"""
import colorsys                                # <remove>
import cairo                                   # <remove>
from gi.repository import Gdk                  # <remove>
from .pen import Pen                           # <remove>
from .utils import draw_dot, is_click_in_bbox  # <remove>
from .icons import Icons                       # <remove>

def draw_rhomb(cr, bbox, fg = (0, 0, 0), bg = (1, 1, 1)):
    """
    Draw a rhombus shape
    """
    x0, y0, w, h = bbox
    cr.set_source_rgb(*bg)
    cr.move_to(x0, y0 + h/2)
    cr.line_to(x0 + w/2, y0)
    cr.line_to(x0 + w, y0 + h/2)
    cr.line_to(x0 + w/2, y0 + h)
    cr.close_path()
    cr.fill()

    cr.set_source_rgb(*fg)
    cr.set_line_width(1)
    cr.move_to(x0, y0 + h/2)
    cr.line_to(x0 + w/2, y0)
    cr.line_to(x0 + w, y0 + h/2)
    cr.line_to(x0 + w/2, y0 + h)
    cr.close_path()
    cr.stroke()


def adjust_color_brightness(rgb, factor):
    """
    Adjust the color brightness.
    :param rgb: A tuple of (r, g, b) in the range [0, 1]
    :param factor: Factor by which to adjust the brightness (>1 to make lighter, <1 to make darker)
    :return: Adjusted color as an (r, g, b) tuple in the range [0, 1]
    """
    # Convert RGB to HLS
    h, l, s = colorsys.rgb_to_hls(*rgb)
    #print("r,g,b:", *rgb, "h, l, s:", h, l, s)

    # Adjust lightness
    l = max(min(l * factor, 1), 0)  # Ensure lightness stays within [0, 1]
    newrgb = colorsys.hls_to_rgb(h, l, s)

    # Convert back to RGB
    return newrgb


## ---------------------------------------------------------------------
class Wiglet:
    """drawable dialog-like objects on the canvas"""
    def __init__(self, mytype, coords):
        self.wiglet_type   = mytype
        self.coords = coords

    def update_size(self, width, height):
        """update the size of the widget"""
        raise NotImplementedError("update size method not implemented")

    def draw(self, cr):
        """draw the widget"""
        raise NotImplementedError("draw method not implemented")

    def event_update(self, x, y):
        """update on mouse move"""
        raise NotImplementedError("event_update method not implemented")

    def event_finish(self):
        """update on mouse release"""
        raise NotImplementedError("event_finish method not implemented")

class WigletTransparency(Wiglet):
    """Wiglet for changing the transparency."""
    def __init__(self, coords, pen):
        super().__init__("transparency", coords)

        if not pen or not isinstance(pen, Pen):
            raise ValueError("Pen is not defined or not of class Pen")

        self.__pen      = pen
        self.__initial_transparency = pen.transparency
        print("initial transparency:", self.__initial_transparency)

    def draw(self, cr):
        """draw the widget"""
        cr.set_source_rgba(*self.__pen.color, self.__pen.transparency)
        draw_dot(cr, *self.coords, 50)

    def event_update(self, x, y):
        """update on mouse move"""
        dx = x - self.coords[0]
        ## we want to change the transparency by 0.1 for every 20 pixels
        self.__pen.transparency = max(0, min(1, self.__initial_transparency + dx/500))

    def update_size(self, width, height):
        """ignoring the update the size of the widget"""

    def event_finish(self):
        """update on mouse release"""

class WigletLineWidth(Wiglet):
    """
    Wiglet for changing the line width.
    directly operates on the pen of the object
    """
    def __init__(self, coords, pen):
        super().__init__("line_width", coords)

        if not pen or not isinstance(pen, Pen):
            raise ValueError("Pen is not defined or not of class Pen")
        self.__pen      = pen
        self.__initial_width = pen.line_width

    def draw(self, cr):
        cr.set_source_rgb(*self.__pen.color)
        draw_dot(cr, *self.coords, self.__pen.line_width)

    def event_update(self, x, y):
        dx = x - self.coords[0]
        print("changing line width", dx)
        self.__pen.line_width = max(1, min(60, self.__initial_width + dx/20))
        return True

    def update_size(self, width, height):
        """ignoring the update the size of the widget"""

    def event_finish(self):
        """ignoring the update on mouse release"""

## ---------------------------------------------------------------------
class WigletPageSelector(Wiglet):
    """Wiglet for selecting the page."""
    def __init__(self, coords = (500, 0), gom = None,
                 width = 20, height = 35, screen_wh_func = None,
                 set_page_func = None):
        if not gom:
            raise ValueError("GOM is not defined")
        super().__init__("page_selector", coords)

        self.__width, self.__height = width, height
        self.__height_per_page = height
        self.__gom = gom
        self.__screen_wh_func = screen_wh_func
        self.__set_page_func  = set_page_func

        # we need to recalculate often because the pages might have been
        # changing a lot
        self.recalculate()

    def recalculate(self):
        """recalculate the position of the widget"""
        if self.__screen_wh_func and callable(self.__screen_wh_func):
            w, _ = self.__screen_wh_func()
            self.coords = (w - self.__width, 0)
        self.__page_n = self.__gom.number_of_pages()
        self.__height = self.__height_per_page * self.__page_n
        self.__bbox = (self.coords[0], self.coords[1], self.__width, self.__height)
        self.__current_page_n = self.__gom.current_page_number()

    def on_click(self, x, y, ev):
        """handle the click event"""
        self.recalculate()
        if not is_click_in_bbox(x, y, self.__bbox):
            return False

        # which page is at this position?
        print("clicked inside the bbox, event", ev)
        dy = y - self.__bbox[1]

        page_no = int(dy / self.__height_per_page)
        print("selected page:", page_no)

        page_in_range = 0 <= page_no < self.__page_n
        if page_in_range and self.__set_page_func and callable(self.__set_page_func):
            print("setting page to", page_no)
            self.__set_page_func(page_no)

        return True

    def draw(self, cr):
        """draw the widget"""
        self.recalculate()

        wpos = self.__bbox[0]
        hpos = self.__bbox[1]

        for i in range(self.__page_n):
            cr.set_source_rgb(0.5, 0.5, 0.5)
            cr.rectangle(wpos, hpos, self.__width, self.__height_per_page)
            cr.fill()

            if i == self.__current_page_n:
                cr.set_source_rgb(0, 0, 0)
            else:
                cr.set_source_rgb(1, 1, 1)

            cr.rectangle(wpos + 1, hpos + 1,
                        self.__width - 2, self.__height_per_page - 2)
            cr.fill()
            if i == self.__current_page_n:
                cr.set_source_rgb(1, 1, 1)
            else:
                cr.set_source_rgb(0, 0, 0)
            cr.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
            cr.set_font_size(14)
            cr.move_to(wpos + 5, hpos + 20)
            cr.show_text(str(i + 1))

            hpos += self.__height_per_page

            # draw layer symbols for the current page
            if i == self.__current_page_n:
                page = self.__gom.page()
                n_layers = page.number_of_layers()
                cur_layer = page.layer_no()
                cr.set_source_rgb(0.5, 0.5, 0.5)
                cr.rectangle(wpos, hpos, self.__width, 
                             n_layers * 5 + 5)
                cr.fill()

                hpos = hpos + n_layers * 5 + 5
                for j in range(n_layers):
                    # draw a small rhombus for each layer
                    curpos = hpos - j * 5 - 10
                    if j == cur_layer:
                        # inverted for the current layer
                        draw_rhomb(cr, (wpos, curpos, self.__width, 10),
                                   (1, 1, 1), (0, 0, 0))
                    else:
                        draw_rhomb(cr, (wpos, curpos, self.__width, 10))



    def update_size(self, width, height):
        """update the size of the widget"""


## ---------------------------------------------------------------------
class WigletToolSelector(Wiglet):
    """Wiglet for selecting the tool."""
    def __init__(self, coords = (50, 0), width = 1000, height = 35, func_mode = None):
        super().__init__("tool_selector", coords)

        self.__width, self.__height = width, height
        self.__icons_only = True

        self.__modes = [ "move", "draw", "shape", "rectangle",
                        "circle", "text", "eraser", "colorpicker" ]
        self.__modes_dict = { "move": "Move", "draw": "Draw", "shape": "Shape",
                              "rectangle": "Rectangle", "circle": "Circle", "text": "Text",
                              "eraser": "Eraser", "colorpicker": "Col.Pick" }

        if self.__icons_only and width > len(self.__modes) * 35:
            self.__width = len(self.__modes) * 35

        self.__bbox = (coords[0], coords[1], self.__width, self.__height)
        self.recalculate()
        self.__mode_func = func_mode
        self.__icons = { }

        self._init_icons()

    def _init_icons(self):
        """initialize the icons"""
        icons = Icons()
        self.__icons = { mode: icons.get(mode) for mode in self.__modes }
        print("icons:", self.__icons)

    def recalculate(self):
        """recalculate the position of the widget"""
        self.__bbox = (self.coords[0], self.coords[1], self.__width, self.__height)
        self.__dw   = self.__width / len(self.__modes)

    def draw(self, cr):
        """draw the widget"""
        cr.set_source_rgb(0.5, 0.5, 0.5)
        cr.rectangle(*self.__bbox)
        cr.fill()

        cur_mode = None
        if self.__mode_func and callable(self.__mode_func):
            cur_mode = self.__mode_func()

        for i, mode in enumerate(self.__modes):
            label = self.__modes_dict[mode]
            # white rectangle
            if mode == cur_mode:
                cr.set_source_rgb(0, 0, 0)
            else:
                cr.set_source_rgb(1, 1, 1)

            cr.rectangle(self.__bbox[0] + 1 + i * self.__dw,
                         self.__bbox[1] + 1,
                         self.__dw - 2,
                         self.__height - 2)
            cr.fill()
            # black text

            if mode == cur_mode:
                cr.set_source_rgb(1, 1, 1)
            else:
                cr.set_source_rgb(0, 0, 0)
            # select small font

            icon = self.__icons.get(mode)
            if not self.__icons_only:
                cr.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
                cr.set_font_size(14)
                x_bearing, y_bearing, t_width, t_height, _, _ = cr.text_extents(label)
                x0 = self.__bbox[0]
                if icon:
                    iw = icon.get_width()
                    x0 += i * self.__dw + (self.__dw - t_width - iw) / 2 - x_bearing + iw
                else:
                    x0 += i * self.__dw + (self.__dw - t_width) / 2 - x_bearing

                cr.move_to(x0, self.__bbox[1] + (self.__height - t_height) / 2 - y_bearing)
                cr.show_text(label)
            if icon:
                Gdk.cairo_set_source_pixbuf(cr,
                                            self.__icons[mode],
                                            self.__bbox[0] + i * self.__dw + 5,
                                            self.__bbox[1] + 5)
                cr.paint()


    def on_click(self, x, y, ev):
        """handle the click event"""

        if not is_click_in_bbox(x, y, self.__bbox):
            return False

        # which mode is at this position?
        print("clicked inside the bbox, event", ev)
        dx = x - self.__bbox[0]
        print("dx:", dx)
        sel_mode = None
        i = int(dx / self.__dw)
        sel_mode = self.__modes[i]
        print("selected mode:", sel_mode)
        if self.__mode_func and callable(self.__mode_func):
            self.__mode_func(sel_mode)


        return True

    def update_size(self, width, height):
        """update the size of the widget"""

class WigletColorSelector(Wiglet):
    """Wiglet for selecting the color."""
    def __init__(self, coords = (0, 0),
                 width = 15,
                 height = 500,
                 func_color = None,
                 func_bg = None):
        super().__init__("color_selector", coords)
        print("height:", height)

        self.__width, self.__height = width, height
        self.__bbox = (coords[0], coords[1], width, height)
        self.__colors = self.generate_colors()
        self.__dh = 25
        self.__func_color = func_color
        self.__func_bg    = func_bg
        self.recalculate()

    def recalculate(self):
        """recalculate the position of the widget"""
        self.__bbox = (self.coords[0], self.coords[1], self.__width, self.__height)
        self.__color_dh = (self.__height - self.__dh) / len(self.__colors)
        self.__colors_hpos = { color : self.__dh + i * self.__color_dh
                              for i, color in enumerate(self.__colors) }

    def update_size(self, width, height):
        """update the size of the widget"""
        _, self.__height = width, height
        self.recalculate()

    def draw(self, cr):
        """draw the widget"""
        # draw grey rectangle around my bbox
        cr.set_source_rgb(0.5, 0.5, 0.5)
        cr.rectangle(*self.__bbox)
        cr.fill()

        bg, fg = (0, 0, 0), (1, 1, 1)
        if self.__func_bg and callable(self.__func_bg):
            bg = self.__func_bg()
        if self.__func_color and callable(self.__func_color):
            fg = self.__func_color()

        cr.set_source_rgb(*bg)
        cr.rectangle(self.__bbox[0] + 4, self.__bbox[1] + 9, self.__width - 5, 23)
        cr.fill()
        cr.set_source_rgb(*fg)
        cr.rectangle(self.__bbox[0] + 1, self.__bbox[1] + 1, self.__width - 5, 14)
        cr.fill()

        # draw the colors
        dh = 25
        h = (self.__height - dh)/ len(self.__colors)
        for color in self.__colors:
            cr.set_source_rgb(*color)
            cr.rectangle(self.__bbox[0] + 1, self.__colors_hpos[color], self.__width - 2, h)
            cr.fill()

    def on_click(self, x, y, ev):
        """handle the click event"""
        if not is_click_in_bbox(x, y, self.__bbox):
            return False
        print("clicked inside the bbox")

        dy = y - self.__bbox[1]
        # which color is at this position?
        sel_color = None
        for color, ypos in self.__colors_hpos.items():
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
        """update on mouse move"""

    def event_finish(self):
        """update on mouse release"""

    def generate_colors(self):
        """Generate a rainbow of 24 colors."""

        # list of 24 colors forming a rainbow
        colors = [  #(0.88, 0.0, 0.83),
                     (0.29, 0.0, 0.51),
                     (0.0, 0.0, 1.0),
                     (0.0, 0.6, 1.0),
                     (0.0, 0.7, 0.5),
                     (0.0, 1.0, 0.0),
                     (1.0, 1.0, 0.0),
                     (1.0, 0.6, 0.0),
                     (0.776, 0.612, 0.427),
                     (1.0, 0.3, 0.0),
                     (1.0, 0.0, 0.0)]

       #colors = [ ]

       #for i in range(1, 21):
       #    h = i/20
       #    rgb = colorsys.hls_to_rgb(h, 0.5, 1)
       #    print("h=", h, "rgb:", *rgb)
       #    colors.append(rgb)

        newc = [ ]
        for i in range(11):
            newc.append((i/10, i/10, i/10))

        for c in colors:
            for dd in range(30, 180, 15): #
                d = dd / 100
                newc.append(adjust_color_brightness(c, d))

        return newc
