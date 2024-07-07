"""Class which produces dialogs and UI elements"""
from os import path
import logging                                             # <remove>
import gi                                                  # <remove>
gi.require_version('Gtk', '3.0')                           # <remove> pylint: disable=wrong-import-position
from gi.repository import GdkPixbuf # <remove>

from .wiglets import WigletSelectionTool, WigletEraser          # <remove>
from .wiglets import WigletCreateObject, WigletCreateGroup      # <remove>
from .wiglets import WigletCreateSegments, WigletEditText       # <remove>
from .wiglets import WigletPan, WigletHover, WigletColorPicker  # <remove>
from .wiglets import WigletResizeRotate, WigletMove             # <remove>

from .wiglets_ui import WigletLineWidth, WigletPageSelector     # <remove>
from .wiglets_ui import WigletColorSelector, WigletToolSelector # <remove>
from .wiglets_ui import WigletTransparency, WigletStatusLine    # <remove>

from .dialogs import help_dialog, import_image_dialog, FontChooser, ColorChooser # <remove>
from .drawable_primitives import Image  # <remove>

log = logging.getLogger(__name__)                                # <remove>
log.setLevel(logging.DEBUG)                                      # <remove>

class UIBuilder():
    """Builds the UI."""

    __new_instance = None

    def __new__(cls, *args, **kwargs):
        if not cls.__new_instance:
            cls.__new_instance = super(UIBuilder, cls).__new__(cls)
        return cls.__new_instance

    def __init__(self, state):

        self.__state = state
        self.__app = state.app()
        self.__bus = state.bus()

        self.__init_wiglets()
        self.__register_bus_events()

    def __register_bus_events(self):
        """Register the bus events."""
        bus = self.__bus

        bus.on("show_help_dialog",  self.show_help_dialog)
        bus.on("import_image",      self.import_image)
        bus.on("select_font",       self.select_font)
        bus.on("select_color_bg",   self.select_color_bg)
        bus.on("select_color",      self.select_color)

    def __init_wiglets(self):
        """Initialize the wiglets."""
        bus = self.__bus
        state = self.__state

        WigletStatusLine(bus = bus, state = state)
        WigletEraser(bus = bus, state = state)
        WigletCreateObject(bus = bus, state = state)
        WigletCreateGroup(bus = bus, state = state)
        WigletCreateSegments(bus = bus, state = state)
        WigletEditText(bus = bus, state = state)
        #WigletEditText2(bus = bus, state = state, app =
        WigletPan(bus = bus, state = state)
        WigletHover(bus = bus, state = state)
        WigletSelectionTool(bus = bus, state = state)
        WigletResizeRotate(bus = bus, state = state)
        WigletMove(bus = bus, state = state)
        WigletColorSelector(bus = bus, func_color = state.set_color,
                             func_bg = state.graphics().bg_color)
        WigletToolSelector(bus = bus, func_mode = state.mode)
        WigletPageSelector(bus = bus, state = state)
        WigletColorPicker(bus = bus, func_color = state.set_color,
                          clipboard = state.clipboard())
        WigletTransparency(bus = bus, state = state)
        WigletLineWidth(bus = bus, state = state)

    def show_help_dialog(self):
        """Show the help dialog."""

        dialog = help_dialog(self.__app)
        dialog.run()
        dialog.destroy()

    def select_font(self):
        """Select a font for drawing using FontChooser dialog."""
        font_description = FontChooser(self.__state.pen(), self.__app)

        if font_description:
            self.__bus.emitMult("set_font", font_description)

    def select_color_bg(self):
        """Select a color for the background using ColorChooser."""

        color = ColorChooser(self.__app, "Select Background Color")

        if color:
            self.__bus.emitOnce("set_bg_color", (color.red, color.green, color.blue))

    def select_color(self):
        """Select a color for drawing using ColorChooser dialog."""
        color = ColorChooser(self.__app)
        if color:
            self.__bus.emitMult("set_color", (color.red, color.green, color.blue))

    def import_image(self):
        """Select an image file and create a pixbuf from it."""

        bus = self.__bus

        import_dir = self.__state.config().import_dir()
        image_file = import_image_dialog(self.__app, import_dir = import_dir)
        dirname, _ = path.split(image_file)
        bus.emitMult("set_import_dir", dirname)

        pixbuf = None

        if image_file:
            try:
                pixbuf = GdkPixbuf.Pixbuf.new_from_file(image_file)
                log.debug("Loaded image: %s", image_file)
            except Exception as e:
                log.error("Failed to load image: %s", e)

            if pixbuf is not None:
                pos = self.__state.cursor_pos()
                img = Image([ pos ], self.__state.pen(), pixbuf)
                bus.emitOnce("add_object", img)

        return pixbuf
