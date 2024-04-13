"""Status singleton class for holding key app information."""
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
        self.__mode = 'draw'
        self.__app = app
        self.__gom = gom
        self.__cursor = cursor
        self.__modified = True

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

    # -------------------------------------------------------------------------

    def gom(self):
        """Return GOM"""
        return self.__gom

    def page(self):
        """Current page"""
        return self.__gom.page()

    def selection(self):
        """Current selection"""
        return self.__gom.selection()

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

    def current_page(self):
        """Get the current page object from gom."""
        return self.__gom.page()

    def modified(self, value = None):
        """Get or set the modified flag."""
        if value is not None:
            self.__modified = value
        return self.__modified

    # -------------------------------------------------------------------------
    def mode(self, mode = None):
        """Get or set the cursor mode."""
        if mode:
            self.__mode = mode
            self.__cursor.default(mode)
        return self.__mode

    # -------------------------------------------------------------------------
    def cursor(self):
        """expose the cursor manager."""
        return self.__cursor

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
        """Get or set the transparency."""
        if value:
            self.__gr["transparency"] = value
        return self.__gr["transparency"]

    def outline(self):
        """Get the outline mode."""
        return self.__gr["outline"]

    def outline_toggle(self):
        """Toggle outline mode."""
        self.__gr["outline"] = not self.__gr["outline"]

    def bg_color(self, color=None):
        """Get or set the background color."""
        if color:
            self.__gr["bg_color"] = color
        return self.__gr["bg_color"]

    # -------------------------------------------------------------------------
    def get_win_size(self):
        """Get the window size."""
        return self.__app.get_size()

    def queue_draw(self):
        """Queue a draw."""
        self.__app.queue_draw()

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

    def __init__(self, app, gom, cursor, state):
        self.__state = state
        self.__app = app
        self.__gom = gom
        self.__cursor = cursor


    # -------------------------------------------------------------------------
    def set_font(self, font_description):
        """Set the font."""
        self.__state.pen().font_set_from_description(font_description)
        self.__gom.selection_font_set(font_description)

        obj = self.__state.current_obj()
        if obj and obj.type == "text":
            obj.pen.font_set_from_description(font_description)

    def set_brush(self, brush = None):
        """Set the brush."""
        if brush is not None:
            print("setting pen", self.__state.pen(), "brush to", brush)
            self.__state.pen().brush_type(brush)
        return self.__state.pen().brush_type()

    def set_color(self, color = None):
        """Get or set the color."""
        if color is None:
            return self.__state.pen().color
        self.__state.pen().color_set(color)
        self.__gom.selection_color_set(color)
        return color

    def stroke_change(self, direction):
        """Modify the line width or text size."""
        print("Changing stroke", direction)
        cobj = self.__state.current_obj()
        if cobj and cobj.type == "text":
            print("Changing text size")
            cobj.stroke_change(direction)
            self.__state.pen().font_size = cobj.pen.font_size
        else:
            for obj in self.__gom.selected_objects():
                obj.stroke_change(direction)

        # without a selected object, change the default pen, but only if in the correct mode
        if self.__state.mode() == "draw":
            self.__state.pen().line_width = max(1, self.__state.pen().line_width + direction)
        elif self.__state.mode() == "text":
            self.__state.pen().font_size = max(1, self.__state.pen().font_size + direction)


    # ---------------------------------------------------------------------
    def clear(self):
        """Clear the drawing."""
        self.__gom.selection().clear()
        #self.__resizeobj      = None
        self.__state.current_obj_clear()
        self.__gom.remove_all()
        self.__state.queue_draw()
