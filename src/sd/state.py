"""Status singleton class for holding key app information."""
import logging                                                   # <remove>
log = logging.getLogger(__name__)                                # <remove>
from sd.pen      import Pen                                        # <remove>

class State:
    """Singleton class for holding key app information."""
    __new_instance = None

    def __new__(cls, *args, **kwargs):
        if not cls.__new_instance:
            cls.__new_instance = super(State, cls).__new__(cls)
        return cls.__new_instance

    def __init__(self, app, bus, gom, cursor):
        bus.on("queue_draw", self.queue_draw)
        self.__bus  = bus

        self.__vars = {
                "mode": "draw",
                "modified": True,
                "filename": None,
                "cur_dir": None,
                "export_dir": None,
                "export_fn": None,
                }

        self.__obj = {
                "gom": gom,
                "app": app,
                "cursor": cursor
                }

        self.__gr = {
                "bg_color": (.8, .75, .65),
                "transparency": 0,
                "outline": False,
                }

        self.__show = {
                "hidden": False,
                "grid": False,
                "wiglets": True,
                }

        self.__objs = {
                "hover": None,
                "current": None,
                "resize": None,
                }

        self.__pens = [ Pen(line_width = 4,  color = (0.2, 0, 0),
                            font_size = 24, transparency  = 1),
                        Pen(line_width = 40, color = (1, 1, 0),
                            font_size = 24, transparency = .2) ]

        bus.on("mode_set", self.mode, 999)
        bus.on("cycle_bg_transparency", self.cycle_background, 0)
        bus.on("toggle_wiglets", self.toggle_wiglets, 0)
        bus.on("toggle_grid", self.toggle_grid, 0)
        bus.on("switch_pens", self.switch_pens, 0)
        bus.on("apply_pen_to_bg", self.apply_pen_to_bg, 0)
        bus.on("clear_page", self.current_obj_clear, 0)
        bus.on("set_bg_color", self.bg_color, 0)
        bus.on("set_filename", self.filename, 0)
        bus.on("set_export_dir", self.export_dir, 0)
        bus.on("set_export_fn", self.export_fn, 0)


    # -------------------------------------------------------------------------

    def filename(self, name = None):
        """Get or set the filename."""
        if name:
            self.__vars["filename"] = name
        return self.__vars["filename"]

    def cur_dir(self, name = None):
        """Get or set the current directory."""
        if name:
            self.__vars["cur_dir"] = name
        return self.__vars["cur_dir"]

    def export_dir(self, name = None):
        """Get or set the export directory."""
        if name:
            self.__vars["export_dir"] = name
        return self.__vars["export_dir"]

    def export_fn(self, name = None):
        """Get or set the export file name."""
        if name:
            self.__vars["export_fn"] = name
        return self.__vars["export_fn"]

    def gom(self):
        """Return GOM"""
        return self.__obj["gom"]

    def page(self):
        """Current page"""
        return self.gom().page()

    def selection(self):
        """Current selection"""
        return self.gom().selection()

    def current_obj(self, obj = None):
        """Get or set the current object."""
        if obj:
            self.__objs["current"] = obj
        return self.__objs["current"]

    def current_obj_clear(self):
        """Clear the current object."""
        self.__objs["current"] = None
        self.queue_draw()

    def hover_obj(self, obj = None):
        """Get or set the hover object."""
        if obj:
            self.__objs["hover"] = obj
        return self.__objs["hover"]

    def hover_obj_clear(self):
        """Clear the hover object."""
        self.__objs["hover"] = None

    def current_page(self):
        """Get the current page object from gom."""
        return self.gom().page()

    def modified(self, value = None):
        """Get or set the modified flag."""
        if value is not None:
            self.__vars["modified"] = value
        return self.__vars["modified"]

    # -------------------------------------------------------------------------
    def mode(self, mode = None):
        """Get or set the cursor mode."""
        if mode:
            if mode == self.__vars["mode"]:
                return mode
            log.debug(f"setting mode to {mode}")
            self.__vars["mode"] = mode
            self.cursor().default(mode)
        return self.__vars["mode"]

    # -------------------------------------------------------------------------
    def cursor(self):
        """expose the cursor manager."""
        return self.__obj["cursor"]

    # -------------------------------------------------------------------------
    def show_grid(self):
        """What is the show grid status."""
        return self.__show["grid"]

    def toggle_grid(self):
        """Toggle the grid."""
        self.__show["grid"] = not self.__show["grid"]

    def show_wiglets(self, value = None):
        """Show or hide the wiglets."""
        if value is not None:
            self.__show["wiglets"] = value
        return self.__show["wiglets"]

    def toggle_wiglets(self):
        """Toggle the wiglets."""
        self.__show["wiglets"] = not self.__show["wiglets"]

    # -------------------------------------------------------------------------
    def __pen_set(self, pen, alternate = False):
        """Set the pen."""
        if alternate:
            self.__pens[1] = pen
        else:
            self.__pens[0] = pen

    def pen(self, alternate = False, pen = None):
        """Get or set the pen."""
        if pen:
            self.__pen_set(pen, alternate)
        return self.__pens[1] if alternate else self.__pens[0]

    def switch_pens(self):
        """Switch between pens."""
        self.__pens = [self.__pens[1], self.__pens[0]]

    def apply_pen_to_bg(self):
        """Apply the pen to the background."""
        self.__gr["bg_color"] = self.__pens[0].color
    # -------------------------------------------------------------------------

    def cycle_background(self):
        """Cycle through background transparency."""
        self.__gr["transparency"] = {1: 0, 0: 0.5, 0.5: 1}[self.__gr["transparency"]]

    def alpha(self, value=None):
        """Get or set the bg transparency."""
        if value:
            self.__gr["transparency"] = value
        return self.__gr["transparency"]

    def outline(self, value = None):
        """Get the outline mode."""
        if value is not None:
            self.__gr["outline"] = value
        return self.__gr["outline"]

    def bg_color(self, color=None):
        """Get or set the background color."""
        if color:
            self.__gr["bg_color"] = color
        return self.__gr["bg_color"]

    # -------------------------------------------------------------------------
    def get_win_size(self):
        """Get the window size."""
        return self.__obj["app"].get_size()

    def queue_draw(self):
        """Queue a draw."""
        self.__obj["app"].queue_draw()

    def hidden(self, value = None):
        """Hide or show the drawing."""
        if not value is None:
            self.__show["hidden"] = value
        return self.__show["hidden"]

# -----------------------------------------------------------------------------
class Setter:
    """
    Class for setting the state.


    The purpose is to pack a bunch of setter methods into a single class
    so that the state class doesn't get too cluttered.
    """
    __new_instance = None

    def __new__(cls, *args, **kwargs):
        if not cls.__new_instance:
            cls.__new_instance = super(Setter, cls).__new__(cls)
        return cls.__new_instance

    def __init__(self, bus, state):
        self.__bus  = bus
        self.__state = state

        bus.on("set_color", self.set_color)
        bus.on("set_brush", self.set_brush)
        bus.on("set_font", self.set_font)
        bus.on("set_line_width", self.set_line_width)
        bus.on("set_transparency", self.set_transparency)
        bus.on("toggle_outline", self.toggle_outline)
        bus.on("stroke_change",  self.stroke_change, 90)

    # -------------------------------------------------------------------------
    def set_font(self, font_description):
        """Set the font."""
        self.__state.pen().font_set_from_description(font_description)

        obj = self.__state.current_obj()
        if obj and obj.type == "text":
            obj.pen.font_set_from_description(font_description)

    def set_brush(self, brush = None):
        """Set the brush."""
        if brush is not None:
            log.debug(f"setting pen {self.__state.pen()} brush to {brush}")
            self.__state.pen().brush_type(brush)
        return self.__state.pen().brush_type()

    def set_color(self, color = None):
        """Get or set the color."""

        if color is not None:
            log.debug(f"Setting color to {color}")
            self.__state.pen().color_set(color)

        return self.__state.pen().color

    def set_transparency(self, transparency = None):
        """Set the line width."""

        if transparency is not None:
            self.__state.pen().transparency = transparency

        return self.__state.pen().line_width

    def set_line_width(self, width = None):
        """Set the line width."""

        if width is not None:
            log.debug(f"Setting line width to {width}")
            self.__state.pen().line_width = width

        return self.__state.pen().line_width

    def stroke_change(self, direction):
        """Modify the line width or text size."""
        log.debug(f"Changing stroke {direction}")
        cobj = self.__state.current_obj()
        # without a selected object, change the default pen, but only if in the correct mode
        if self.__state.mode() == "draw":
            self.__state.pen().line_width = max(1, self.__state.pen().line_width + direction)
        elif self.__state.mode() == "text":
            self.__state.pen().font_size = max(1, self.__state.pen().font_size + direction)

        if cobj and cobj.type == "text":
            log.debug(f"Changing text size")
            cobj.stroke_change(direction)
            self.__state.pen().font_size = cobj.pen.font_size
            return True

        return False

    def toggle_outline(self):
        """Toggle outline mode."""
        self.__state.outline(not self.__state.outline())
        self.__bus.emit("force_redraw")
