"""Class which produces dialogs and UI elements"""
from os import path
import logging                                             # <remove>
import gi                                                  # <remove>
gi.require_version('Gtk', '3.0')                           # <remove> pylint: disable=wrong-import-position
from gi.repository import Gtk, GLib, GdkPixbuf # <remove>

from .wiglets import WigletSelectionTool, WigletEraser          # <remove>
from .wiglets import WigletCreateObject, WigletCreateGroup      # <remove>
from .wiglets import WigletCreateSegments, WigletEditText       # <remove>
from .wiglets import WigletPan, WigletHover, WigletColorPicker  # <remove>
from .wiglets import WigletResizeRotate, WigletMove             # <remove>
from .wiglets import WigletZoom                                 # <remove>

from .wiglets_ui import WigletLineWidth, WigletPageSelector     # <remove>
from .wiglets_ui import WigletColorSelector, WigletToolSelector # <remove>
from .wiglets_ui import WigletTransparency, WigletStatusLine    # <remove>

from .dialogs import help_dialog, FontChooser, ColorChooser     # <remove>
from .dialogs import open_drawing_dialog, import_image_dialog   # <remove>
from .dialogs import export_dialog, save_dialog                 # <remove>
from .drawable_primitives import Image                          # <remove>
from .drawable_group import DrawableGroup                       # <remove>

from .import_export import read_file_as_sdrw, save_file_as_sdrw # <remove>
from .import_export import export_image                         # <remove>

from .utils import get_screenshot                               # <remove>

from .pen import Pen                                            # <remove>

log = logging.getLogger(__name__)                               # <remove>
log.setLevel(logging.DEBUG)                                     # <remove>

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
        self.__ui_state = {
                "export_options": {
                    "format": None,
                    "all_pages_pdf": False,
                    "export_screen": False
                    }
                }

    def __register_bus_events(self):
        """Register the bus events."""

        listeners = {
            "show_help_dialog":  self.show_help_dialog,
            "import_image":      self.import_image,
            "select_font":       self.select_font,
            "select_color_bg":   self.select_color_bg,
            "select_color":      self.select_color,
            "open_drawing":      self.open_drawing,
            "save_drawing_as":   self.save_drawing_as,
            "export_drawing":    self.export_drawing,
            "screenshot":        self.screenshot,
        }

        for event, listener in listeners.items():
            self.__bus.on(event, listener)

    def __init_wiglets(self):
        """Initialize the wiglets."""
        bus = self.__bus
        state = self.__state

        WigletZoom(bus = bus, state = state)
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
            except Exception as e: #pylint: disable=broad-except
                log.error("Failed to load image: %s", e)

            if pixbuf is not None:
                pos = self.__state.cursor_pos()
                img = Image([ pos ], self.__state.pen(), pixbuf)
                bus.emitOnce("add_object", img)

        return pixbuf

    def open_drawing(self):
        """Open a drawing from a file in native format."""
        file_name = open_drawing_dialog(self.__app)

        if file_name and self.read_file(file_name):
            log.debug("Setting savefile to %s", file_name)

            self.__bus.emit("set_savefile", False, file_name)
            self.__state.graphics().modified(True)

    def read_file(self, filename, load_config = True):
        """Read the drawing state from a file."""
        if not filename:
            raise ValueError("No filename provided")

        state = self.__state
        config, objects, pages = read_file_as_sdrw(filename)

        if pages:
            state.gom().set_pages(pages)
        elif objects:
            state.gom().set_objects(objects)

        if config and load_config:
            state.graphics().bg_color(config.get('bg_color') or (.8, .75, .65))
            state.graphics().alpha(config.get('transparent') or 0)
            show_wiglets = config.get('show_wiglets')

            if show_wiglets is None:
                show_wiglets = True

            state.graphics().show_wiglets(show_wiglets)
            state.pen(pen = Pen.from_dict(config['pen']))
            state.pen(pen = Pen.from_dict(config['pen2']), alternate = True)
            state.gom().set_page_number(config.get('page') or 0)
        if config or objects:
            state.graphics().modified(True)
            return True
        return False

    def autosave(self):
        """Autosave the drawing state."""
        state = self.__state

        if not state.graphics().modified():
            return

        if state.current_obj(): # not while drawing!
            return

        log.debug("Autosaving")

        self.save_state()
        state.graphics().modified(False)

    def save_state(self):
        """Save the current drawing state to a file."""
        state = self.__state

        savefile = state.config().savefile()

        if not savefile:
            log.debug("No savefile set")
            return

        log.debug("savefile: %s", savefile)
        config = {
                'bg_color':     state.graphics().bg_color(),
                'transparent':  state.graphics().alpha(),
                'show_wiglets': state.graphics().show_wiglets(),
                'bbox':         (0, 0, *self.__app.get_size()),
                'pen':          state.pen().to_dict(),
                'pen2':         state.pen(alternate = True).to_dict(),
                'page':         state.gom().current_page_number()
        }

        pages   = state.gom().export_pages()

        save_file_as_sdrw(savefile, config, pages = pages)

    def save_drawing_as(self):
        """Save the drawing to a file."""
        log.debug("opening save file dialog")
        file = save_dialog(self.__app)

        if file:
            log.debug("setting savefile to %s", file)
            self.__bus.emit("set_savefile", False, file)
            self.save_state()

    def __normalize_export_format(self, file_format):
        """Normalize the export format."""
        formats = { "By extension": "any",
                    "PDF": "pdf",
                    "SVG": "svg",
                    "PNG": "png",
                   }
        return formats[file_format]

    def __get_export_cfg(self, bbox):
        """Get the export configuration."""
        state = self.__state
        cfg = { "bg": state.graphics().bg_color(),
               "bbox": bbox,
               "transparency": state.graphics().alpha() }
        return cfg

    def __emit_export_dir(self, file_name):
        """Emit the export directory."""
        # extract dirname and base name from file name
        dirname, base_name = path.split(file_name)
        log.debug("dirname: %s base_name: %s", dirname, base_name)

        self.__bus.emitMult("set_export_dir", dirname)
        self.__bus.emitMult("set_export_fn", base_name)

    def __prepare_export_objects(self, all_as_pdf, exp_screen, selected):
        """Prepare the objects for export."""
        state = self.__state

        if all_as_pdf:
            # get all objects from all pages and layers
            # create drawable group for each page
            obj = state.gom().get_all_pages()
            obj = [ p.objects_all_layers() for p in obj ]
            obj = [ DrawableGroup(o) for o in obj ]
            bbox = None
        elif exp_screen:
            log.debug("Exporting screen")
            obj = DrawableGroup(state.objects_all_layers())
            bbox = state.visible_bbox()
        else:
            log.debug("Exporting selected objects")
            if selected:
                obj = DrawableGroup(selected)
            else:
                obj = DrawableGroup(state.objects_all_layers())
            bbox = obj.bbox()

        return obj, bbox

    def export_drawing(self):
        """Save the drawing to a file."""
        # Choose where to save the file
        #    self.export(filename, "svg")
        bbox = None
        state = self.__state

        selected = state.selected_objects()

        self.__ui_state["export_options"]["selected"] = selected

        export_dir = state.config().export_dir()
        file_name  = state.config().export_fn()

        ret = export_dialog(self.__app, export_dir = export_dir,
                            filename = file_name,
                            ui_opts = self.__ui_state["export_options"]
                            )

        file_name, file_format, all_as_pdf, exp_screen = ret

        if not file_name:
            return

        self.__ui_state["export_options"]["format"] = file_format
        self.__ui_state["export_options"]["all_pages_pdf"] = all_as_pdf
        self.__ui_state["export_options"]["export_screen"] = exp_screen

        file_format = self.__normalize_export_format(file_format)
        obj, bbox = self.__prepare_export_objects(all_as_pdf, exp_screen,
                                                  selected)

        self.__emit_export_dir(file_name)
        cfg = self.__get_export_cfg(bbox)

        export_image(obj, file_name, file_format, cfg, all_as_pdf)

    def __screenshot_finalize(self, bb):
        """Finish up the screenshot."""

        state = self.__state
        trafo = self.__state.page().trafo()
        bus   = self.__bus
        log.debug("Taking screenshot now")

        frame = trafo.apply([ (bb[0] - 3, bb[1] - 3),
                                     (bb[0] + bb[2] + 6, bb[1] + bb[3] + 6) ])
        frame = (frame[0][0], frame[0][1], frame[1][0], frame[1][1])
        log.debug("frame is %s", [ int(x) for x in frame ])

        pixbuf, filename = get_screenshot(self.__app, *frame)
        bus.emitOnce("toggle_hide", False)
        bus.emitOnce("queue_draw")

        # Create the image and copy the file name to clipboard
        if pixbuf is not None:
            img = Image([ (bb[0], bb[1]) ], state.pen(), pixbuf)
            bus.emitOnce("add_object", img)
            bus.emitOnce("queue_draw")
            state.clipboard().set_text(filename)

    def __find_screenshot_box(self):
        """Find a box suitable for selecting a screenshot."""

        for obj in self.__state.selected_objects():
            if obj.type == "rectangle":
                return obj

        return None

    def screenshot(self, obj = None):
        """Take a screenshot and add it to the drawing."""

        # function called twice: once, when requesting a screenshot
        # and second, when the screenshot has been made

        state = self.__state
        bus   = self.__bus

        if not obj:
            obj = self.__find_screenshot_box()

        bus.off("add_object", self.screenshot)

        if not obj:
            log.debug("no suitable box found")
            state.mode("rectangle")
            bus.on("add_object", self.screenshot, priority = 999)
            return

        bb = obj.bbox()
        log.debug("bbox is %s", [int(x) for x in bb])

        bus.emitOnce("toggle_hide", True)
        bus.emitOnce("queue_draw")

        while Gtk.events_pending():
            Gtk.main_iteration_do(False)
        GLib.timeout_add(100, self.__screenshot_finalize, bb)
