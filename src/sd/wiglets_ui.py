"""Wiglets which constitute visible UI elements"""
import logging                                 # <remove>
import colorsys                                # <remove>
import gi                                      # <remove>
gi.require_version('Gtk', '3.0')               # <remove> pylint: disable=wrong-import-position
gi.require_version('Gdk', '3.0')               # <remove> pylint: disable=wrong-import-position
from gi.repository import Gdk                  # <remove>
import cairo                                   # <remove>

from .wiglets import Wiglet #<remove>
from .utils import draw_dot, is_click_in_bbox  # <remove>
from .icons import Icons                       # <remove>
log = logging.getLogger(__name__)              # <remove>

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



class WigletTransparency(Wiglet):
    """Wiglet for changing the transparency."""
    def __init__(self, bus, state):
        super().__init__("transparency", None)

        self.__state = state
        self.__pen    = None
        self.__initial_transparency = None
        self.__active = False
        self.__bus    = bus
        log.debug("initial transparency: %s", self.__initial_transparency)

        bus.on("left_mouse_click", self.on_click)

    def draw(self, cr, state):
        """draw the widget"""
        if not self.__active:
            return

        log.debug("drawing transparency widget")
        cr.set_source_rgba(*self.__pen.color, self.__pen.transparency)
        draw_dot(cr, *self.coords, 50)

    def on_click(self, ev):
        """handle the click event"""
        # make sure we are in the correct mode
        if not ev.shift() or ev.alt() or not ev.ctrl():
            return False

        if ev.hover() or ev.corner()[0] or ev.double():
            return False

        self.__bus.on("mouse_move", self.on_move)
        self.__bus.on("mouse_release", self.on_release)
        self.__bus.on("draw", self.draw)

        self.coords = (ev.x_abs, ev.y_abs)

        self.__pen    = self.__state.pen()
        self.__initial_transparency = self.__pen.transparency

        self.__active = True
        return True

    def on_move(self, ev):
        """update on mouse move"""
        if not self.__active:
            return False
        x = ev.x_abs
        dx = x - self.coords[0]

        ## we want to change the transparency by 0.1 for every 20 pixels
        transparency = max(0, min(1, self.__initial_transparency + dx/500))
        self.__bus.emit_mult("set_transparency", transparency)

        return True

    def on_release(self, ev):
        """handle the release event"""

        if not self.__active:
            return False

        log.debug("got release event")
        self.__active = False
        self.__bus.off("mouse_move", self.on_move)
        self.__bus.off("mouse_release", self.on_release)
        self.__bus.off("draw", self.draw)
        return True

class WigletLineWidth(Wiglet):
    """
    Wiglet for changing the line width.
    directly operates on the pen of the object
    """
    def __init__(self, bus, state):
        super().__init__("line_width", None)

        if not state:
            raise ValueError("Need state object")
        self.__state = state
        self.__pen    = None
        self.__initial_width = None
        self.__active = False
        self.__bus = bus

        bus.on("left_mouse_click", self.on_click)

    def draw(self, cr, state):
        """draw the widget"""
        if not self.__active:
            return

        cr.set_source_rgb(*self.__pen.color)
        draw_dot(cr, *self.coords, self.__pen.line_width)

    def on_click(self, ev):
        """handle the click event"""
        # make sure we are in the correct mode

        if ev.shift() or ev.alt() or not ev.ctrl():
            return False
        if ev.hover() or ev.corner()[0] or ev.double():
            return False

        self.coords = (ev.x_abs, ev.y_abs)
        self.__pen    = self.__state.pen()
        self.__initial_width = self.__pen.line_width

        self.__bus.on("mouse_move", self.on_move)
        self.__bus.on("mouse_release", self.on_release)
        self.__bus.on("draw", self.draw)

        log.debug("activating lw at %d, %d, initial width %s",
                  int(self.coords[0]), int(self.coords[1]),
                  self.__initial_width)
        self.__active = True
        return True

    def on_release(self, ev):
        """handle the release event"""
        if not self.__active:
            return False
        self.__active = False

        self.__bus.off("mouse_move", self.on_move)
        self.__bus.off("mouse_release", self.on_release)
        self.__bus.off("draw", self.draw)

        return True

    def on_move(self, ev):
        """update on mouse move"""
        if not self.__active:
            return False
        x = ev.x_abs
        dx = x - self.coords[0]
        width = max(1, min(60, self.__initial_width + dx/20))

        self.__bus.emit_mult("set_line_width", width)
        return True

## ---------------------------------------------------------------------
class WigletPageSelector(Wiglet):
    """Wiglet for selecting the page."""

    # we need five things for talking to to the outside world:
    # 0. getting the height and width of the screen
    # 1. getting the number of pages
    # 2. getting the current page number
    # 3. setting the current page number
    # 4. adding a new page
    # one possibility: get a separate function for each of these
    # or: use gom as a single object that can do all of these, but then we
    # need to be aware of the gom object
    def __init__(self, state, bus):

        coords = (500, 0)
        wh = (20, 35)

        super().__init__("page_selector", coords)

        self.__bbox = (coords[0], coords[1], wh[0], wh[1])
        self.__height_per_page = wh[1]
        self.__gom = state.gom()
        self.__page_screen_pos = [ ]

        # we need to recalculate often because the pages might have been
        # changing a lot
        self.recalculate()
        bus.on("left_mouse_click", self.on_click, priority = 99)
        bus.on("update_win_size", self.update_size)
        bus.on("draw", self.draw)

    def recalculate(self):
        """recalculate the position of the widget"""
        self.__page_n = self.__gom.number_of_pages()     # <- outside info

        tot_h = sum(x for x, _ in self.__page_screen_pos)

        self.__bbox = (self.coords[0], self.coords[1],
                       self.__bbox[2], tot_h)

        self.__current_page_n = self.__gom.current_page_number() # <- outside info

    def update_size(self, width, height):
        """update the size of the widget"""

        self.coords = (width - self.__bbox[2], 0)
        self.__bbox = (self.coords[0], self.coords[1],
                       self.__bbox[2], self.__bbox[3])

        self.recalculate()

    def on_click(self, ev):
        """handle the click event"""
        if not ev.state.graphics().show_wiglets():
            return False

        x, y = ev.x_abs, ev.y_abs

        self.recalculate()

        if not is_click_in_bbox(x, y, self.__bbox):
            return False

        # which page is at this position?
        log.debug("page_selector: clicked inside the bbox, event %s", ev)
        dy = y - self.__bbox[1]

        page_no = self.__page_n

        for i, (y0, y1) in enumerate(self.__page_screen_pos):
            if y0 <= dy <= y1:
                page_no = i
                break
        log.debug("selected page: %s", page_no)

        page_in_range = 0 <= page_no < self.__page_n

        if page_in_range:
            log.debug("setting page to %s", page_no)
            self.__gom.set_page_number(page_no)     # <- outside info

        if page_no == self.__page_n:
            log.debug("adding a new page")
            self.__gom.set_page_number(page_no - 1) # <- outside info
            self.__gom.next_page()                  # <- outside info

        return True

    def draw(self, cr, state):
        """
        Draw the widget on cr.

        For each page, make a little white rectangle; current page is
        highlighted by inverted colors.  If the current page is selected,
        draw a little symbol for the layers.

        Finally, draw a little "+" symbol for adding a new page.
        """
        if not state.graphics().show_wiglets():
            return

        self.recalculate()

        wpos  = self.__bbox[0]
        hpos  = self.__bbox[1]
        width = self.__bbox[2]

        # page_screen_pos records the exact screen positions of the pages,
        # so when the widget is clicked, we know on which page
        self.__page_screen_pos = [ ]

        for i in range(self.__page_n + 1):
            page_pos = hpos
            self.__draw_page(cr, i, hpos, self.__bbox)

            hpos += self.__height_per_page

            # draw layer symbols for the current page
            if i == self.__current_page_n:
                page      = self.__gom.page()
                n_layers  = page.number_of_layers()
                cur_layer = page.layer_no()

                hpos = hpos + n_layers * 5 + 5
                self.__draw_layers(cr, n_layers, cur_layer,
                                   (wpos, hpos, width, None))

            self.__page_screen_pos.append((page_pos, hpos))

    def __draw_page(self, cr, page_no, hpos, bbox):
        """draw a page"""

        wpos, _, width, _ = bbox

        # grey background
        cr.set_source_rgb(0.5, 0.5, 0.5)
        cr.rectangle(wpos, hpos, width,
                     self.__height_per_page)
        cr.fill()

        # the rectangle representing the page
        if page_no == self.__current_page_n:
            cr.set_source_rgb(0, 0, 0)
        else:
            cr.set_source_rgb(1, 1, 1)

        cr.rectangle(wpos + 1, hpos + 1,
                     width - 2,
                     self.__height_per_page - 2)
        cr.fill()

        # the page number or "+" symbol
        if page_no == self.__current_page_n:
            cr.set_source_rgb(1, 1, 1)
        else:
            cr.set_source_rgb(0, 0, 0)
        cr.select_font_face("Sans",
                            cairo.FONT_SLANT_NORMAL,
                            cairo.FONT_WEIGHT_NORMAL)
        cr.set_font_size(14)
        cr.move_to(wpos + 5, hpos + 20)
        if page_no == self.__page_n:
            cr.show_text("+")
        else:
            cr.show_text(str(page_no + 1))


    def __draw_layers(self, cr, n_layers, cur_layer, bbox):
        """
        Draw n_layers layers with current layer cur_layer starting from
        bottom at position hpos and left at position wpos, with width
        width.
        """
        wpos, hpos, width, _ = bbox
        cr.set_source_rgb(0.5, 0.5, 0.5)
        cr.rectangle(wpos, hpos, width,
                     n_layers * 5 + 5)
        cr.fill()

        for j in range(n_layers):
            # draw a small rhombus for each layer
            curpos = hpos - j * 5 - 10
            if j == cur_layer:
                # inverted for the current layer
                draw_rhomb(cr, (wpos, curpos, width, 10),
                           (1, 1, 1), (1, 0, 0))
            else:
                draw_rhomb(cr, (wpos, curpos, width, 10))

## ---------------------------------------------------------------------

## ---------------------------------------------------------------------
class WigletToolSelector(Wiglet):
    """Wiglet for selecting the tool."""
    def __init__(self, bus, coords = (50, 0), func_mode = None):
        super().__init__("tool_selector", coords)

        width, height = 1000, 35
        self.__icons_only = True

        self.__modes = [ "move", "draw", "segment", "shape", "rectangle",
                        "circle", "text", "eraser", "colorpicker", "zoom" ]
        self.__modes_dict = { "move": "Move", "draw": "Draw", "segment": "Seg.Path",
                             "shape": "Shape", "rectangle": "Rectangle",
                             "circle": "Circle", "text": "Text", "eraser": "Eraser",
                             "colorpicker": "Col.Pick", "zoom": "Zoom"}

        if self.__icons_only and width > len(self.__modes) * 35:
            width = len(self.__modes) * 35

        self.__bbox = (coords[0], coords[1], width, height)
        self.__mode_func = func_mode
        self.__icons = { }

        self._init_icons()
        bus.on("left_mouse_click", self.on_click, priority = 99)
        bus.on("draw", self.draw)

    def _init_icons(self):
        """initialize the icons"""
        icons = Icons()
        self.__icons = { mode: icons.get(mode) for mode in self.__modes }

    def draw(self, cr, state):
        """draw the widget"""
        if not state.graphics().show_wiglets():
            return

        cr.set_source_rgb(0.5, 0.5, 0.5)
        cr.rectangle(*self.__bbox)
        cr.fill()

        height = self.__bbox[3]
        dw   = self.__bbox[2] / len(self.__modes)

        cur_mode = None

        if self.__mode_func and callable(self.__mode_func):
            cur_mode = self.__mode_func()

        for i, mode in enumerate(self.__modes):
            # white rectangle
            icon  = self.__icons.get(mode)
            label = self.__modes_dict[mode] if not self.__icons_only else None

            bb = (self.__bbox[0] + i * dw,
                  self.__bbox[1],
                  dw, height)

            self.__draw_label(cr, bb, label, icon, mode == cur_mode)

    def __draw_label(self, cr, bbox, label, icon, inverse = False):
        """Paint one button within the bounding box"""

        x0, y0 = bbox[0], bbox[1]
        iw = 0

        if inverse:
            cr.set_source_rgb(0, 0, 0)
        else:
            cr.set_source_rgb(1, 1, 1)

        cr.rectangle(bbox[0] + 1, bbox[1] + 1, bbox[2] - 2, bbox[3] - 2)
        cr.fill()

        if icon:
            iw = icon.get_width()
            Gdk.cairo_set_source_pixbuf(cr, icon, x0 + 5, y0 + 5)
            cr.paint()

        if not label:
            return

        if inverse:
            cr.set_source_rgb(1, 1, 1)
        else:
            cr.set_source_rgb(0, 0, 0)
        # select small font

        cr.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        cr.set_font_size(14)

        x_bearing, y_bearing, t_width, t_height, _, _ = cr.text_extents(label)

        xpos = x0 + (bbox[2] - t_width - iw) / 2 - x_bearing + iw
        ypos = y0 + (bbox[3] - t_height) / 2 - y_bearing

        cr.move_to(xpos, ypos)
        cr.show_text(label)

    def on_click(self, ev):
        """handle the click event"""

        if not ev.state.graphics().show_wiglets():
            return False

        x, y = ev.x_abs, ev.y_abs

        dw   = self.__bbox[2] / len(self.__modes)

        if not is_click_in_bbox(x, y, self.__bbox):
            return False

        # which mode is at this position?
        dx = x - self.__bbox[0]
        sel_mode = None
        i = int(dx / dw)
        sel_mode = self.__modes[i]
        if self.__mode_func and callable(self.__mode_func):
            self.__mode_func(sel_mode)

        return True

class WigletStatusLine(Wiglet):
    """Show file name on the bottom of the screen"""

    def __init__(self, bus, state, bbox = (100, 100, 15, 500)):
        coords = (bbox[0], bbox[1])
        super().__init__("status_line", coords)

        self.__state = state
        self.__params = {
                "screen_wh": (100, 100),
                "text_par": [ None, None, None, None ],
                "bbox": bbox,
                "moves": [ ],
                "moves_per_second": 0,
                "zoom": [ 1, 1 ]
                }

        self.zoom_calc()

        bus.on("mouse_move", self.rec_move, 1999)
        bus.on("update_win_size", self.update_size)
        bus.on("draw", self.draw)
        bus.on("page_zoom", self.zoom_calc, priority = -1)
        bus.on("page_zoom_reset", self.zoom_calc, priority = -1)
        bus.on("page_translate", self.zoom_calc, priority = -1)

    def zoom_calc(self, *args): # pylint: disable=unused-argument
        """recalculate the current zoom"""
        trafo = self.__state.page().trafo()
        zx, zy = trafo.calc_zoom()
        self.__params["zoom"] = (zx, zy)
        return False

    def rec_move(self, ev):
        """record the move and calculate moves / second"""

        ## current time in seconds
        t = ev.event.time
        ## convert it to hh:mm:ss
        #log.debug("time: %s", t)
        moves = self.__params["moves"]
        moves.append(t)

        if len(moves) > 100:
            # remove first element
            moves.pop(0)
            #self.__param["moves"] = moves[1:]
            #log.debug("time for 100 moves: %.2f", (self.__moves[-1] - self.__moves[0]) / 1000)
            self.__params["moves_per_second"] = 1000 * 100 / (moves[-1] - moves[0])
        return False

    def update_size(self, width, height):
        """update the size of the widget"""

        self.__params["screen_wh"] = (width, height)
        return True

    def calc_size(self):
        """Calculate the vertical size of the widget"""
        p = self.__params
        _, height = p["screen_wh"]

        # we can only update the size if we have the text parameters
        if not p["text_par"]:
            return False

        (dx, dy, tw, th) = p["text_par"]
        x0 = 5 - dx
        y0 = height - 5 - th - dy
        self.coords = (x0, y0)
        p["bbox"] = (x0 + dx - 5, y0 + dy - 5, tw + 10, th + 10)
        return True

    def draw(self, cr, state):
        """draw the widget"""
        if not state.graphics().show_wiglets():
            return

        p = self.__params
        state = self.__state
        status_line = state.config().savefile() + f" |mode: {state.mode()}|"

        status_line += ' (!)' if state.graphics().modified() else ''

        bg_cols = [ int(x * 100) for x in state.graphics().bg_color()]
        tr      = int(state.graphics().alpha() * 100)
        bg_cols.append(tr)
        status_line += f"  bg: {bg_cols}"

        pen = state.pen()
        status_line += f"  pen: col={pen.color} lw={int(100*pen.line_width)/100} "
        status_line += f"tr={int(100*pen.transparency)} type: {pen.brush_type()}"
        status_line += f'| zoom: {int(p["zoom"][0] * 100)}%'

        hov = state.hover_obj()
        status_line += f"  hover: {hov.type}" if hov else ''

        status_line += f'  moves/s: {p["moves_per_second"]:.2f}'

        cr.select_font_face("monospace", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        cr.set_font_size(14)

        # x_bearing: The horizontal distance from the origin to the leftmost part of the glyphs.
        # y_bearing: The vertical distance from the origin to the topmost part of the glyphs.
        # width: The width of the text.
        # height: The height of the text.
        # x_advance: The distance to advance horizontally after drawing the text.
        # y_advance: The distance to advance vertically after drawing the text
        #            (usually 0 for horizontal text).
        dx, dy, tw, th, _, _ = cr.text_extents(status_line)
        p["text_par"] = (dx, dy, tw, th)
        self.calc_size()

        cr.set_source_rgba(1, 1, 1, 0.5)
        cr.rectangle(*p["bbox"])
        cr.fill()

        cr.set_source_rgb(0.2, 0.2, 0.2)
        cr.move_to(*self.coords)
        cr.show_text(status_line)

class WigletColorSelector(Wiglet):
    """Wiglet for selecting the color."""
    def __init__(self, bus, bbox = (0, 0, 15, 500),
                 func_color = None,
                 func_bg = None):

        coords = (bbox[0], bbox[1])
        super().__init__("color_selector", coords)

        self.__bbox = bbox
        self.__colors = self.generate_colors()
        self.__dh = 25
        self.__func_color = func_color
        self.__func_bg    = func_bg
        self.__bus = bus
        self.recalculate()
        bus.on("left_mouse_click", self.on_click, priority = 999)
        bus.on("update_win_size", self.update_size)
        bus.on("draw", self.draw)

    def recalculate(self):
        """recalculate the position of the widget"""
        self.__color_dh = (self.__bbox[3] - self.__dh) / len(self.__colors)
        self.__colors_hpos = { color : self.__dh + i * self.__color_dh
                              for i, color in enumerate(self.__colors) }

    def update_size(self, width, height):
        """update the size of the widget"""
        self.__bbox = (self.coords[0], self.coords[1], self.__bbox[2], height - 50)
        self.recalculate()

    def draw(self, cr, state):
        """draw the widget"""
        # draw grey rectangle around my bbox
        if not state.graphics().show_wiglets():
            return

        cr.set_source_rgb(0.5, 0.5, 0.5)
        cr.rectangle(*self.__bbox)
        cr.fill()

        bg, fg = (0, 0, 0), (1, 1, 1)
        if self.__func_bg and callable(self.__func_bg):
            bg = self.__func_bg()
        if self.__func_color and callable(self.__func_color):
            fg = self.__func_color()

        cr.set_source_rgb(*bg)
        cr.rectangle(self.__bbox[0] + 4,
                     self.__bbox[1] + 9,
                     self.__bbox[2] - 5, 23)
        cr.fill()
        cr.set_source_rgb(*fg)
        cr.rectangle(self.__bbox[0] + 1,
                     self.__bbox[1] + 1,
                     self.__bbox[2] - 5, 14)
        cr.fill()

        # draw the colors
        dh = 25
        h = (self.__bbox[3] - dh)/ len(self.__colors)
        for color in self.__colors:
            cr.set_source_rgb(*color)
            cr.rectangle(self.__bbox[0] + 1,
                         self.__colors_hpos[color],
                         self.__bbox[2] - 2, h)
            cr.fill()

    def on_click(self, ev):
        """handle the click event"""
        if not ev.state.graphics().show_wiglets():
            return False

        x, y = ev.x_abs, ev.y_abs

        if not is_click_in_bbox(x, y, self.__bbox):
            return False

        log.debug("color_selector: clicked inside the bbox")

        dy = y - self.__bbox[1]
        # which color is at this position?
        sel_color = None
        for color, ypos in self.__colors_hpos.items():
            if ypos <= dy <= ypos + self.__color_dh:
                log.debug("selected color: %s", color)
                sel_color = color

        if not sel_color:
            log.debug("no color selected")
            return True

        if ev.shift():
            log.debug("setting bg to color %s", sel_color)
            self.__bus.emit("set_bg_color", False, sel_color)
        else:
            log.debug("setting fg to color %s", sel_color)
            self.__bus.emit("set_color", False, sel_color)

        self.__bus.emit("queue_draw")
        return True

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
