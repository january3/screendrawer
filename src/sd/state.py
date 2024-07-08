"""Status singleton class for holding key app information."""
import logging                                                 # <remove>
from .pen      import Pen                                      # <remove>
from .history  import History                                  # <remove>
from .gom      import GraphicsObjectManager                    # <remove>
from .cursor import CursorManager                              # <remove>
from .clipboard import Clipboard                               # <remove>
from .utils import get_cursor_position                         # <remove>

from .drawable_primitives import Text, Image                   # <remove>

log = logging.getLogger(__name__)                              # <remove>

def object_create_copies(objects, move = False, bb = None):
    """Create copies of given objects, possibly shifted"""

    new_objects = [ ]

    for obj in objects:
        new_obj = obj.duplicate()
        new_objects.append(new_obj)

        # move the new object to the current location
        if move:
            x, y = move
            if bb is None:
                bb  = new_obj.bbox()
            new_obj.move(x - bb[0], y - bb[1])

    return new_objects


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
        if self.graphics().mode() == "draw":
            self.pen().line_width = max(1, self.pen().line_width + direction)
        elif self.graphics().mode() == "text":
            self.pen().font_size = max(1, self.pen().font_size + direction)

        if cobj and cobj.type == "text":
            log.debug("Changing text size")
            cobj.stroke_change(direction)
            self.pen().font_size = cobj.pen.font_size
            return True

        return False


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
                "clipboard": None,
                "mouse": None,
                }

    def mouse(self, mouse = None):
        """Return the mouse object."""
        if mouse:
            self.__obj["mouse"] = mouse
        return self.__obj["mouse"]

    def cursor(self):
        """expose cursor"""
        return self.__obj["cursor"]

    def clipboard(self, clipboard = None):
        """Return the clipboard."""
        if clipboard:
            self.__obj["clipboard"] = clipboard
        return self.__obj["clipboard"]

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

        clipboard = Clipboard()
        self.clipboard(clipboard)
        self.__init_signals()

    def __init_signals(self):
        """Initialize the signals."""

        bus_signals = {
            "queue_draw": {"listener": self.queue_draw},
            "mode_set": {"listener": self.mode, "priority": 999},

            "cycle_bg_transparency": {"listener": self.graphics().cycle_background, "priority": 0},
            "set_bg_color": {"listener": self.graphics().bg_color, "priority": 0},
            "toggle_wiglets": {"listener": self.graphics().toggle_wiglets, "priority": 0},
            "toggle_grid": {"listener": self.graphics().toggle_grid, "priority": 0},

            "set_savefile": {"listener": self.config().savefile, "priority": 0},
            "set_export_dir": {"listener": self.config().export_dir, "priority": 0},
            "set_import_dir": {"listener": self.config().import_dir, "priority": 0},
            "set_export_fn": {"listener": self.config().export_fn, "priority": 0},

            "switch_pens": {"listener": self.switch_pens, "priority": 0},
            "apply_pen_to_bg": {"listener": self.apply_pen_to_bg, "priority": 0},
            "clear_page": {"listener": self.current_obj_clear, "priority": 0},

            "set_color": {"listener": self.set_color},
            "set_brush": {"listener": self.set_brush},
            "set_font": {"listener": self.set_font},
            "set_line_width": {"listener": self.set_line_width},
            "set_transparency": {"listener": self.set_transparency},
            "toggle_outline": {"listener": self.toggle_outline},
            "toggle_hide": {"listener": self.toggle_hide},
            "stroke_change": {"listener": self.stroke_change, "priority": 90},
            "query_cursor_pos": {"listener": self.get_cursor_pos},

            "copy_content": {"listener": self.copy_content},
            "cut_content": {"listener": self.cut_content},
            "duplicate_content": {"listener": self.duplicate_content},
            "paste_content": {"listener": self.paste_content},
        }

        bus = self.bus()
        for signal, params in bus_signals.items():
            if params.get("priority"):
                bus.on(signal, params["listener"], priority = params["priority"])
            else:
                bus.on(signal, params["listener"])

    # -------------------------------------------------------------------------
    def get_cursor_pos(self):
        """Get the cursor position"""

        x, y = get_cursor_position(self.app())
        self.bus().emitMult("cursor_abs_pos_update", (x, y))

    def toggle_outline(self):
        """Toggle outline mode."""
        self.graphics().outline(not self.graphics().outline())
        self.bus().emit("force_redraw")

    def toggle_hide(self, hide_state = None):
        """Toggle hide mode."""
        if hide_state is not None:
            self.graphics().hidden(hide_state)
        else:
            self.graphics().hidden(not self.graphics().hidden())

        self.bus().emit("force_redraw")

    def mode(self, mode = None):
        """Get or set the mode."""
        # wrapper, because this is used frequently, and also because
        # graphics state does not hold the cursor object
        if mode is not None and mode != self.graphics().mode():
            self.cursor().default(mode)

        return self.graphics().mode(mode)

    def cut_content(self):
        """Cut content to clipboard."""
        self.copy_content(True)

    def copy_content(self, destroy = False):
        """Copy content to clipboard."""
        content = self.selection()

        if content.is_empty():
            return

        log.debug("Copying content %s", content)
        self.clipboard().copy_content(content, cut = destroy)

        if destroy:
            self.gom().remove_selection()

    def __paste_text(self, clip_text):
        """Enter some text in the current object or create a new object."""

        obj = Text([ self.cursor_pos() ],
                        pen = self.pen(), content=clip_text.strip())
        return [ obj ]

    def __paste_image(self, clip_img):
        """Create an image object from a pixbuf image."""
        obj = Image([ self.cursor_pos() ], self.pen(), clip_img)
        return [ obj ]

    def __paste_internal(self, clip):
        """Paste internal content."""
        log.debug("Pasting internal content")

        if clip.type != "clipboard_group":
            raise ValueError("Internal clipboard is not a clipboard_group")

        bb = clip.bbox()
        log.debug("clipboard bbox %s", bb)

        if not clip.is_cut():
            move = self.cursor_pos()
        else:
            move = None

        new_objects = object_create_copies(clip.objects, move, bb)

        return new_objects

    def paste_content(self):
        """Paste content from clipboard."""
        clip_type, clip = self.clipboard().get_content()

        if not clip:
            return

        # internal paste
        if clip_type == "internal":
            new_objects = self.__paste_internal(clip)
        elif clip_type == "text":
            new_objects = self.__paste_text(clip)
        elif clip_type == "image":
            new_objects = self.__paste_image(clip)

        if new_objects:
            for obj in new_objects:
                self.bus().emit("add_object", True, obj)
            self.bus().emit("set_selection", True, new_objects)

    def duplicate_content(self):
        """Duplicate the selected content."""
        content = self.selection()

        if content.is_empty():
            return

        for obj in content.objects:
            new_obj = obj.duplicate()
            self.bus().emitOnce("add_object", new_obj)
