"""Status singleton class for holding key app information."""
from sd.pen      import Pen                                        # <remove>
from sd.cursor   import CursorManager                              # <remove>

class State:
    """Singleton class for holding key app information."""
    __new_instance = None

    def __new__(cls, *args, **kwargs):
        if not cls.__new_instance:
            cls.__new_instance = super(State, cls).__new__(cls)
        return cls.__new_instance

    def __init__(self, app, gom, cursor):
        self.__mode = 'draw'
        self.__app = app
        self.__gom = gom
        self.__cursor = cursor
        self.__bg_color     = (.8, .75, .65)
        self.__transparency = 0
        self.__outline      = False
        self.__hidden       = False
        self.__show_grid    = False
        self.__show_wiglets = True
        self.__hover_obj    = None
        self.__current_obj  = None

        self.__pens = [ Pen(line_width = 4,  color = (0.2, 0, 0), font_size = 24, transparency  = 1),
                        Pen(line_width = 40, color = (1, 1, 0),   font_size = 24, transparency = .2) ]

    # -------------------------------------------------------------------------
    def app(self):
        """Get the app."""
        # XXX temporary solution until we can remove drawing of wiglets
        # from dm
        return self.__app

    # -------------------------------------------------------------------------
    def current_obj(self, obj = None):
        """Get or set the current object."""
        if obj:
            self.__current_obj = obj
        return self.__current_obj

    def current_obj_clear(self):
        """Clear the current object."""
        self.__current_obj = None

    def hover_obj(self, obj = None):
        """Get or set the hover object."""
        if obj:
            self.__hover_obj = obj
        return self.__hover_obj

    def hover_obj_clear(self):
        """Clear the hover object."""
        self.__hover_obj = None

    def current_page(self):
        """Get the current page object from gom."""
        return self.__gom.page()

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
        return self.__show_grid

    def toggle_grid(self):
        """Toggle the grid."""
        self.__show_grid = not self.__show_grid

    def show_wiglets(self, value = None):
        """Show or hide the wiglets."""
        if value is not None:
            self.__show_wiglets = value
        return self.__show_wiglets

    def toggle_wiglets(self):
        """Toggle the wiglets."""
        self.__show_wiglets = not self.__show_wiglets

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
        self.__bg_color = self.__pens[0].color
    # -------------------------------------------------------------------------

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

    def alpha(self, value=None):
        """Get or set the transparency."""
        if value:
            self.__transparency = value
        return self.__transparency

    # -------------------------------------------------------------------------
    def get_win_size(self):
        """Get the window size."""
        return self.__app.get_size()

    def queue_draw(self):
        """Queue a draw."""
        self.__app.queue_draw()

    def hide(self, value = None):
        """Hide or show the drawing."""
        if not value is None:
            self.__hidden = value
        return self.__hidden
