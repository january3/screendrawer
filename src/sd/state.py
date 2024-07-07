"""Status singleton class for holding key app information."""
import logging                                                   # <remove>
from .pen      import Pen                                      # <remove>
from .history  import History                                  # <remove>
from .gom      import GraphicsObjectManager                    # <remove>
from .cursor import CursorManager                                # <remove>

log = logging.getLogger(__name__)                                # <remove>

class StateGraphics:
    """
    Base class for holding key app graphics state information.

    Essentially options regarding whether to show UI elements, background
    etc.
    """

    def __init__(self):

        self.__gr = {
                "mode": "draw",             # drawing mode
                "modified": True,           # modified flag
                "bg_color": (.8, .75, .65), # bg color
                "transparency": 0,          # bg alpha
                "outline": False,           # outline mode
                "hidden": False,            # hide drawing
                                            # for screenshots
                "grid": False,              # show grid
                "wiglets": True,            # show wiglets
                }

    def bg_color(self, color=None):
        """Get or set the background color."""
        if color:
            self.__gr["bg_color"] = color
        return self.__gr["bg_color"]

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

    def show_grid(self):
        """What is the show grid status."""
        return self.__gr["grid"]

    def toggle_grid(self):
        """Toggle the grid."""
        self.__gr["grid"] = not self.__gr["grid"]

    def show_wiglets(self, value = None):
        """Show or hide the wiglets."""
        if value is not None:
            self.__gr["wiglets"] = value
        return self.__gr["wiglets"]

    def toggle_wiglets(self):
        """Toggle the wiglets."""
        self.__gr["wiglets"] = not self.__gr["wiglets"]

    def hidden(self, value = None):
        """Hide or show the drawing."""
        if not value is None:
            self.__gr["hidden"] = value
        return self.__gr["hidden"]

    def mode(self, mode = None):
        """Get or set the cursor mode."""
        if mode:
            if mode == self.__gr["mode"]:
                return mode
            log.debug("setting mode to %s", mode)
            self.__gr["mode"] = mode
        return self.__gr["mode"]

    def modified(self, value = None):
        """Get or set the modified flag."""
        if value is not None:
            self.__gr["modified"] = value
        return self.__gr["modified"]



class StateConfig:
    """import export dirs"""

    def __init__(self):
        self.__vars = {
                "savefile": None,
                "cur_dir": None,
                "import_dir": None,
                "export_dir": None,
                "export_fn": None,
                }

    def savefile(self, name = None):
        """Get or set the savefile."""
        if name:
            self.__vars["savefile"] = name
        return self.__vars["savefile"]

    def cur_dir(self, name = None):
        """Get or set the current directory."""
        if name:
            self.__vars["cur_dir"] = name
        return self.__vars["cur_dir"]

    def import_dir(self, name = None):
        """Get or set the import directory."""
        if name:
            self.__vars["import_dir"] = name
        return self.__vars["import_dir"]

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


class StateRoot:
    """Base class for holding key app information."""

    def __init__(self):
        self.__gr_state = StateGraphics()
        self.__config   = StateConfig()

        self.__objs = {
                "hover": None,
                "current": None,
                "resize": None,
                }

        self.__pens = [ Pen(line_width = 4,  color = (0.2, 0, 0),
                            font_size = 24, transparency  = 1),
                        Pen(line_width = 40, color = (1, 1, 0),
                            font_size = 24, transparency = .2) ]


    # -------------------------------------------------------------------------
    def graphics(self):
        """Return the graphics state."""
        return self.__gr_state

    def config(self):
        """Return the config state."""
        return self.__config

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
        self.__gr_state.bg_color(self.__pens[0].color)

# -----------------------------------------------------------------------------
class StateObj(StateRoot):

    """
    Adds big object handling to state
    """

    def __init__(self, app, bus):

        super().__init__()
        self.__bus      = bus

        history = History(bus)
        gom = GraphicsObjectManager(self.__bus)
        cursor = CursorManager(app, bus)

        self.__obj = {
                "gom": gom,
                "app": app,
                "cursor": cursor,
                "history": history,
                }

    def cursor(self):
        """Return the cursor."""
        return self.__obj["cursor"]

    def cursor_pos(self):
        """Return the cursor position."""
        return self.__obj["cursor"].pos()

    def history(self):
        """Return the history."""
        return self.__obj["history"]

    def app(self):
        """Return the app."""
        return self.__obj["app"]

    def bus(self):
        """Return the bus."""
        return self.__bus

    def gom(self):
        """Return GOM"""
        return self.__obj["gom"]

    def current_page(self):
        """Get the current page object from gom."""
        return self.gom().page()

    def cursor(self):
        """expose the cursor manager."""
        return self.__obj["cursor"]

    def page(self):
        """Current page"""
        return self.gom().page()

    def selection(self):
        """Current selection"""
        return self.gom().selection()

    def selected_objects(self):
        """Return the selected objects."""
        return self.gom().selected_objects()

    def objects(self):
        """Return the objects of the current layer."""
        return self.page().layer().objects()

    def get_win_size(self):
        """Get the window size."""
        return self.__obj["app"].get_size()

    def queue_draw(self):
        """Queue a draw."""
        self.__obj["app"].queue_draw()

    def mode(self, mode = None):
        """Get or set the mode."""
        # wrapper, because this is used frequently, and also because
        # graphics state does not hold the cursor object
        if mode is not None and mode != self.graphics().mode():
            self.cursor().default(mode)

        return self.graphics().mode(mode)


# -----------------------------------------------------------------------------
class State(StateObj):
    """
    Class for setting the state.


    The purpose is to pack a bunch of setter methods into a single class
    so that the state class doesn't get too cluttered.
    """
    __new_instance = None

    def __new__(cls, *args, **kwargs):
        if not cls.__new_instance:
            cls.__new_instance = super(State, cls).__new__(cls)
        return cls.__new_instance

    def __init__(self, app, bus):

        super().__init__(app, bus)

        bus.on("queue_draw", self.queue_draw)
        bus.on("mode_set", self.mode, 999)

        bus.on("cycle_bg_transparency", self.graphics().cycle_background, 0)
        bus.on("set_bg_color", self.graphics().bg_color, 0)
        bus.on("toggle_wiglets", self.graphics().toggle_wiglets, 0)
        bus.on("toggle_grid", self.graphics().toggle_grid, 0)

        bus.on("set_savefile", self.config().savefile, 0)
        bus.on("set_export_dir", self.config().export_dir, 0)
        bus.on("set_import_dir", self.config().import_dir, 0)
        bus.on("set_export_fn", self.config().export_fn, 0)

        bus.on("switch_pens", self.switch_pens, 0)
        bus.on("apply_pen_to_bg", self.apply_pen_to_bg, 0)
        bus.on("clear_page", self.current_obj_clear, 0)

        bus.on("set_color", self.set_color)
        bus.on("set_brush", self.set_brush)
        bus.on("set_font", self.set_font)
        bus.on("set_line_width", self.set_line_width)
        bus.on("set_transparency", self.set_transparency)
        bus.on("toggle_outline", self.toggle_outline)
        bus.on("stroke_change",  self.stroke_change, 90)
        bus.on("query_cursor_pos", self.get_cursor_pos)

    # -------------------------------------------------------------------------
    def get_cursor_pos(self):
        """Get the cursor position"""

        x, y = get_cursor_position(self.app())
        self.bus().emitMult("cursor_abs_pos_update", (x, y))

    def set_font(self, font_description):
        """Set the font."""
        self.pen().font_set_from_description(font_description)

        obj = self.current_obj()
        if obj and obj.type == "text":
            obj.pen.font_set_from_description(font_description)

    def set_brush(self, brush = None):
        """Set the brush."""
        if brush is not None:
            log.debug("setting pen %s brush to {brush}", self.pen())
            self.pen().brush_type(brush)
        return self.pen().brush_type()

    def set_color(self, color = None):
        """Get or set the pen color."""

        if color is not None:
            log.debug("Setting color to %s", color)
            self.pen().color_set(color)

        return self.pen().color

    def set_transparency(self, transparency = None):
        """Set the line width."""

        if transparency is not None:
            self.pen().transparency = transparency

        return self.pen().line_width

    def set_line_width(self, width = None):
        """Set the line width."""

        if width is not None:
            log.debug("Setting line width to %s", width)
            self.pen().line_width = width

        return self.pen().line_width

    def stroke_change(self, direction):
        """Modify the line width or text size."""
        log.debug("Changing stroke %s", direction)
        cobj = self.current_obj()
        # without a selected object, change the default pen, but only if in the correct mode
        if self.mode() == "draw":
            self.pen().line_width = max(1, self.pen().line_width + direction)
        elif self.mode() == "text":
            self.pen().font_size = max(1, self.pen().font_size + direction)

        if cobj and cobj.type == "text":
            log.debug("Changing text size")
            cobj.stroke_change(direction)
            self.pen().font_size = cobj.pen.font_size
            return True

        return False

    def toggle_outline(self):
        """Toggle outline mode."""
        self.graphics().outline(not self.graphics().outline())
        self.bus().emit("force_redraw")
