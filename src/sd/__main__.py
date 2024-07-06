"""
ScreenDrawer - a simple drawing program that allows you to draw on the screen

Usage:
  sd.py [options] [file.sdrw [file.[png, pdf, svg]]]

See README.md for more information.
"""

import copy
import pickle
import traceback
import colorsys
from sys import exc_info, argv, exit
import os
from os import path
import time
import math
import base64
import tempfile
import logging
import hashlib
from io import BytesIO

import warnings
import argparse

import pyautogui
from PIL import ImageGrab

import yaml
import cairo
import appdirs

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GdkPixbuf, Pango, GLib


# ---------------------------------------------------------------------
# These are various classes and utilities for the sd.py script. In the
# "skeleton" variant, they are just imported. In the "full" variant, they
# the files are directly, physically inserted in order to get one big fat
# Python script that can just be copied.

from sd.utils import *                    ###<placeholder sd/utils.py>
from sd.commands import *                 ###<placeholder sd/commands.py>
from sd.pen import Pen                    ###<placeholder sd/pen.py>
from sd.events import *                   ###<placeholder sd/events.py>
from sd.dialogs import *                  ###<placeholder sd/dialogs.py>
from sd.clipboard import Clipboard        ###<placeholder sd/clipboard.py>
from sd.cursor import CursorManager       ###<placeholder sd/cursor.py>
from sd.gom import GraphicsObjectManager  ###<placeholder sd/gom.py>
from sd.import_export import *            ###<placeholder sd/import_export.py>
from sd.em import *                       ###<placeholder sd/em.py>
from sd.menus import *                    ###<placeholder sd/menus.py>
from sd.wiglets import *                  ###<placeholder sd/wiglets.py>
from sd.wiglets_ui import *               ###<placeholder sd/wiglets.py>
from sd.icons import Icons                ###<placeholder sd/icons.py>
from sd.page import Page                  ###<placeholder sd/page.py>
from sd.canvas import Canvas              ###<placeholder sd/canvas.py>
from sd.brush import Brush                ###<placeholder sd/brush.py>
from sd.grid import Grid                  ###<placeholder sd/grid.py>
from sd.texteditor import TextEditor      ###<placeholder sd/texteditor.py>
from sd.imageobj import ImageObj          ###<placeholder sd/imageobj.py>
from sd.state import State, Setter        ###<placeholder sd/state.py>
from sd.drawable import *                 ###<placeholder sd/drawable.py>
from sd.drawer import Drawer              ###<placeholder sd/drawer.py>
from sd.drawable_factory import DrawableFactory         ###<placeholder sd/drawable_factory.py>
from sd.drawable_group import DrawableGroup ##<placeholder sd/drawable_group.py>
from sd.drawable_primitives import Image, Text                    #<placeholder sd/drawable_primitives.py>
from sd.drawable_primitives import Rectangle, Shape, Circle #<placeholder sd/drawable_primitives.py>
from sd.drawable_paths import Path                          #<placeholder sd/drawable_paths.py>
from sd.history import History                              #<placeholder sd/history.py>
from sd.bus import Bus #<placeholder sd/bus.py>


# ---------------------------------------------------------------------
# defaults

logging.basicConfig(level=logging.INFO,
                    #format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    format='%(levelname)s %(filename)s:%(lineno)d %(funcName)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
log = logging.getLogger(__name__)
log.info("Application is starting")

COLORS = {
        "black": (0, 0, 0),
        "white": (1, 1, 1),
        "red": (.7, 0, 0),
        "green": (0, .7, 0),
        "blue": (0, 0, .5),
        "yellow": (1, 1, 0),
        "cyan": (0, 1, 1),
        "magenta": (1, 0, 1),
        "purple": (0.5, 0, 0.5),
        "grey": (0.5, 0.5, 0.5)
}

AUTOSAVE_INTERVAL = 30 * 1000 # 30 seconds




## ---------------------------------------------------------------------

class TransparentWindow(Gtk.Window):
    """Main app window. Holds all information and everything that exists.
       One window to rule them all."""

    def __init__(self, save_file = None):
        super().__init__()

        self.savefile            = save_file
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        self.set_title("Transparent Drawing Window")
        self.set_decorated(False)
        self.connect("destroy", self.exit)
        self.set_default_size(800, 800)
        self.set_keep_above(True)
        self.maximize()

        # transparency
        screen = self.get_screen()
        visual = screen.get_rgba_visual()
        if visual is not None and screen.is_composited():
            self.set_visual(visual)
        self.set_app_paintable(True)

        self.fixed = Gtk.Fixed()
        self.add(self.fixed)

        # autosave
        GLib.timeout_add(AUTOSAVE_INTERVAL, self.__autosave)

        # Drawing setup
        self.bus                = Bus()
        self.cursor             = CursorManager(self, self.bus)
        self.history            = History(self.bus)
        self.gom                = GraphicsObjectManager(self.bus)

        # we pass the app to the state, because it has the queue_draw
        # method
        self.state              = State(app = self,
                                        bus = self.bus,
                                        gom = self.gom,
                                        cursor = self.cursor)

        self.clipboard           = Clipboard()

        self.setter = Setter(bus = self.bus, state = self.state)

        # initialize the wiglets - which listen to events
        self.__init_wiglets()

        # em has to know about all that to link actions to methods
        em  = EventManager(bus = self.bus, state  = self.state)
        mm  = MenuMaker(self.bus, self.gom, em, self)

        # mouse gets the mouse events
        self.mouse = MouseCatcher(bus = self.bus, state = self.state)

        # canvas orchestrates the drawing
        self.canvas = Canvas(state = self.state, bus = self.bus)

        # load the drawing from the savefile
        self.read_file(self.savefile)
        self.bus.emit("set_filename", False, self.savefile)

        # connecting events
        self.__add_bus_events()

        self.set_events(Gdk.EventMask.BUTTON_PRESS_MASK |
                        Gdk.EventMask.BUTTON_RELEASE_MASK |
                        Gdk.EventMask.POINTER_MOTION_MASK |
                        Gdk.EventMask.TOUCH_MASK)

        self.connect("key-press-event",      em.on_key_press)
        self.connect("draw",                 self.canvas.on_draw)
        self.connect("button-press-event",   self.mouse.on_button_press)
        self.connect("button-release-event", self.mouse.on_button_release)
        self.connect("motion-notify-event",  self.mouse.on_motion_notify)

      # self.button = Gtk.Button(label="Add Text Entry")
      # self.button.set_size_request(100, 130)
      # self.button.connect("clicked", self.exit)
      # self.fixed.put(self.button, 200, 200)

    def exit(self, event = None):
        """Exit the application."""
        ## close the savefile_f
        log.info("Exiting")
        self.__save_state()
        Gtk.main_quit()

    # ---------------------------------------------------------------------

    def __init_wiglets(self):
        """Initialize the wiglets."""
        wiglets = [
                   WigletStatusLine(bus = self.bus, state = self.state),
                   WigletEraser(bus = self.bus, state = self.state),
                   WigletCreateObject(bus = self.bus, state = self.state),
                   WigletCreateGroup(bus = self.bus, state = self.state),
                   WigletCreateSegments(bus = self.bus, state = self.state),
                   WigletEditText(bus = self.bus, state = self.state),
                   #WigletEditText2(bus = self.bus, state = self.state, app = self),
                   WigletPan(bus = self.bus, state = self.state),
                   WigletHover(bus = self.bus, state = self.state),
                   WigletSelectionTool(bus = self.bus, gom = self.gom),
                   WigletResizeRotate(bus = self.bus, gom = self.gom, state = self.state),
                   WigletMove(bus = self.bus, state = self.state),
                   WigletColorSelector(bus = self.bus, func_color = self.setter.set_color,
                                        func_bg = self.state.bg_color),
                   WigletToolSelector(bus = self.bus, func_mode = self.state.mode),
                   WigletPageSelector(bus = self.bus, gom = self.gom),
                   WigletColorPicker(bus = self.bus, func_color = self.setter.set_color, 
                                     clipboard = self.clipboard),
                   WigletTransparency(bus = self.bus, state = self.state),
                   WigletLineWidth(bus = self.bus, state = self.state),
        ]


    def __add_bus_events(self):
        """Add bus events."""

        self.bus.on("app_exit", self.exit)
        self.bus.on("show_help_dialog",  self.show_help_dialog)
        self.bus.on("export_drawing",    self.export_drawing)
        self.bus.on("save_drawing_as",   self.save_drawing_as)
        self.bus.on("select_color",      self.select_color)
        self.bus.on("select_color_bg",   self.select_color_bg)
        self.bus.on("select_font",       self.select_font)
        self.bus.on("import_image",      self.import_image)
        self.bus.on("open_drawing",      self.open_drawing)
        self.bus.on("copy_content",      self.copy_content)
        self.bus.on("cut_content",       self.cut_content)
        self.bus.on("duplicate_content", self.duplicate_content)
        self.bus.on("paste_content",     self.paste_content)
        self.bus.on("screenshot",        self.screenshot)


    def paste_text(self, clip_text):
        """Enter some text in the current object or create a new object."""

        cobj = self.state.current_obj()
        if cobj and cobj.type == "text":
            cobj.add_text(clip_text.strip())
        else:
            obj = Text([ self.cursor.pos() ],
                            pen = self.state.pen(), content=clip_text.strip())
            self.bus.emit("add_object", True, obj)
            self.bus.emitOnce("set_selection", [ obj ])

    def paste_image(self, clip_img):
        """Create an image object from a pixbuf image."""
        obj = Image([ self.cursor.pos() ], self.state.pen(), clip_img)
        self.bus.emitMult("add_object", obj)
        self.bus.emitOnce("set_selection", [ obj ])

    def __object_create_copy(self, obj, bb = None):
        """Copy the given object into a new object."""
        new_obj = obj.duplicate()

        # move the new object to the current location
        x, y = self.cursor.pos()
        if bb is None:
            bb  = new_obj.bbox()
        new_obj.move(x - bb[0], y - bb[1])

        self.bus.emit("add_object", True, new_obj)
        self.bus.emit("set_selection", True, [ new_obj ])

    def copy_content(self, destroy = False):
        """Copy content to clipboard."""
        content = self.gom.selection()
        if content.is_empty():
            return

        log.debug(f"Copying content {content}")
        self.clipboard.copy_content(content)

        if destroy:
            self.gom.remove_selection()

    def paste_content(self):
        """Paste content from clipboard."""
        clip_type, clip = self.clipboard.get_content()

        if not clip:
            return

        # internal paste
        if clip_type == "internal":
            log.debug("Pasting content internally")
            if clip.type != "group":
                raise ValueError("Internal clipboard is not a group")
            bb = clip.bbox()
            log.debug(f"clipboard bbox {bb}")
            for obj in clip.objects:
                self.__object_create_copy(obj, bb)
            return

        if clip_type == "text":
            self.paste_text(clip)
        elif clip_type == "image":
            self.paste_image(clip)

    def cut_content(self):
        """Cut content to clipboard."""
        self.copy_content(True)

    def duplicate_content(self):
        """Duplicate the selected content."""
        content = self.gom.selection()

        if content.is_empty():
            return

        for obj in content.objects:
            new_obj = obj.duplicate()
            self.bus.emitOnce("add_object", new_obj)

    def select_color_bg(self):
        """Select a color for the background using ColorChooser."""
        color = ColorChooser(self, "Select Background Color")
        if color:
            self.state.bg_color((color.red, color.green, color.blue))

    def select_color(self):
        """Select a color for drawing using ColorChooser dialog."""
        color = ColorChooser(self)
        if color:
            self.bus.emit("set_color", True, (color.red, color.green, color.blue))

    def select_font(self):
        """Select a font for drawing using FontChooser dialog."""
        font_description = FontChooser(self.state.pen(), self)

        if font_description:
            self.bus.emit("set_font", True, font_description)

    def show_help_dialog(self):
        """Show the help dialog."""
        dialog = help_dialog(self)
        dialog.run()
        dialog.destroy()

    def save_drawing_as(self):
        """Save the drawing to a file."""
        log.debug("opening save file dialog")
        file = save_dialog(self)
        if file:
            self.savefile = file
            log.debug(f"setting savefile to {self.savefile}")
            self.bus.emit("set_filename", False, self.savefile)
            self.__save_state()

    def export_drawing(self):
        """Save the drawing to a file."""
        # Choose where to save the file
        #    self.export(filename, "svg")
        bbox = None
        selected = False

        if self.gom.selected_objects():
            # only export the selected objects
            obj = self.gom.selected_objects()
            selected = True
        else:
            # set bbox so we export the whole screen
            obj = self.gom.objects()
            bbox = (0, 0, *self.get_size())

        if not obj:
            log.warning("Nothing to draw")
            return

        obj = DrawableGroup(obj)
        export_dir = self.state.export_dir()
        file_name  = self.state.export_fn()

        log.debug(f"export_dir: {export_dir}")
        ret = export_dialog(self, export_dir = export_dir,
                            filename = file_name, 
                            selected = selected)
        file_name, file_format, all_as_pdf = ret

        if not file_name:
            return

        cfg = { "bg": self.state.bg_color(), 
               "bbox": bbox, 
               "transparency": self.state.alpha() }

        if all_as_pdf:
            # get all objects from all pages and layers
            # create drawable group for each page
            obj = self.gom.get_all_pages()
            obj = [ p.objects_all_layers() for p in obj ]
            obj = [ DrawableGroup(o) for o in obj ]

        # extract dirname and base name from file name
        dirname, base_name = path.split(file_name)
        log.debug(f"dirname: {dirname}, base_name: {base_name}")
        self.bus.emitMult("set_export_dir", dirname)
        self.bus.emitMult("set_export_fn", base_name)

        export_image(obj, file_name, file_format, cfg, all_as_pdf)

    def import_image(self):
        """Select an image file and create a pixbuf from it."""

        import_dir = self.state.import_dir()
        image_file = import_image_dialog(self, import_dir = import_dir)
        dirname, base_name = path.split(image_file)
        self.bus.emitMult("set_import_dir", dirname)

        pixbuf = None

        if image_file:
            try:
                pixbuf = GdkPixbuf.Pixbuf.new_from_file(image_file)
                print(f"Loaded image: {image_file}")
            except Exception as e:
                print(f"Failed to load image: {e}")

            if pixbuf is not None:
                pos = self.cursor.pos()
                img = Image([ pos ], self.state.pen(), pixbuf)
                self.bus.emit("add_object", True, img)
                self.queue_draw()

        return pixbuf

    def __screenshot_finalize(self, bb):
        """Finish up the screenshot."""
        print("Taking screenshot now")
        dx, dy = self.gom.page().translate() or (0, 0)
        print("translate is", dx, dy)
        frame = (bb[0] - 3 + dx, bb[1] - 3 + dy, bb[0] + bb[2] + 6 + dx, bb[1] + bb[3] + 6 + dy)
        print("frame is", frame)
        pixbuf, filename = get_screenshot(self, *frame)
        self.state.hidden(False)
        self.queue_draw()

        # Create the image and copy the file name to clipboard
        if pixbuf is not None:
            img = Image([ (bb[0], bb[1]) ], self.state.pen(), pixbuf)
            self.bus.emit("add_object", True, img)
            self.queue_draw()
            self.clipboard.set_text(filename)

    def __find_screenshot_box(self):
        """Find a box suitable for selecting a screenshot."""
       #cobj = self.state.current_obj()
       #if cobj and cobj.type == "rectangle":
       #    return cobj
        for obj in self.gom.selected_objects():
            if obj.type == "rectangle":
                return obj
       #for obj in self.gom.objects()[::-1]:
       #    if obj.type == "rectangle":
       #        return obj
        return None

    def screenshot(self, obj = None):
        """Take a screenshot and add it to the drawing."""
        if not obj:
            obj = self.__find_screenshot_box()

        self.bus.off("add_object", self.screenshot)
        if not obj:
            print("no suitable box found")
            self.state.mode("rectangle")
            self.bus.on("add_object", self.screenshot, priority = 99)
            return
            ## use the whole screen
            #bb = (0, 0, *self.get_size())
        else:
            bb = obj.bbox()
            print("bbox is", bb)
        #self.hidden = True
        self.state.hidden(True)
        self.queue_draw()
        while Gtk.events_pending():
            Gtk.main_iteration_do(False)
        GLib.timeout_add(100, self.__screenshot_finalize, bb)

    def __autosave(self):
        """Autosave the drawing state."""
        if not self.state.modified():
            return

        if self.state.current_obj(): # not while drawing!
            return

        log.debug("Autosaving")
        self.__save_state()
        self.state.modified(False)

    def __save_state(self):
        """Save the current drawing state to a file."""
        if not self.savefile:
            log.debug("No savefile set")
            return

        log.debug(f"savefile: {self.savefile}")
        config = {
                'bg_color':    self.state.bg_color(),
                'transparent': self.state.alpha(),
                'show_wiglets': self.state.show_wiglets(),
                'bbox':        (0, 0, *self.get_size()),
                'pen':         self.state.pen().to_dict(),
                'pen2':        self.state.pen(alternate = True).to_dict(),
                'page':        self.gom.current_page_number()
        }

        pages   = self.gom.export_pages()

        save_file_as_sdrw(self.savefile, config, pages = pages)

    def open_drawing(self):
        """Open a drawing from a file in native format."""
        file_name = open_drawing_dialog(self)
        if self.read_file(file_name):
            log.debug(f"Setting savefile to {file_name}")
            self.savefile = file_name
            self.state.modified(True)
            self.bus.emit("set_filename", False, self.savefile)

    def read_file(self, filename, load_config = True):
        """Read the drawing state from a file."""
        config, objects, pages = read_file_as_sdrw(filename)

        if pages:
            self.gom.set_pages(pages)
        elif objects:
            self.gom.set_objects(objects)

        if config and load_config:
            self.state.bg_color(config.get('bg_color') or (.8, .75, .65))
            self.state.alpha(config.get('transparent') or 0)
            show_wiglets = config.get('show_wiglets')
            if show_wiglets is None:
                show_wiglets = True
            self.state.show_wiglets(show_wiglets)
            self.state.pen(pen = Pen.from_dict(config['pen']))
            self.state.pen(pen = Pen.from_dict(config['pen2']), alternate = True)
            self.gom.set_page_number(config.get('page') or 0)
        if config or objects:
            self.state.modified(True)
            return True
        return False


## ---------------------------------------------------------------------

def parse_arguments():
    """Handle command line arguments."""
    parser = argparse.ArgumentParser(
            description="Drawing on the screen",
            epilog=f"Alternative use: {argv[0]} file.sdrw file.[png, pdf, svg, yaml]")
    parser.add_argument("-l", "--loadfile", help="Load drawing from file")
    parser.add_argument("-c", "--convert",
                        help="""
Convert screendrawer file to given format (png, pdf, svg, yaml) and exit
(use -o to specify output file, otherwise a default name is used)
""",
                        metavar = "FORMAT"
    )
    parser.add_argument("-p", "--page", help="Page to convert (default: 1)", 
                        type=int)

    parser.add_argument("-b", "--border", 
                        help="""
                             Border width for conversion. If not
                             specified, the default is 10 px.
                             """, 
                        default=10,
                        type=int)
    parser.add_argument("-o", "--output", help="Output file for conversion")
    parser.add_argument("-d", "--debug", help="Debug mode", action="store_true")
    parser.add_argument("-s", "--sticky", help="Sticky window (yes or no)",
                        choices = [ "yes", "no" ], default = "yes")
    parser.add_argument("files", nargs="*")
    args     = parser.parse_args()
    return args

def process_args(args, app_name, app_author):
    """Process command line arguments."""
    page_no = None
    _savefile = None

    if args.page is not None:
        page_no = args.page - 1

    # explicit conversion
    if args.convert:
        if not args.convert in [ "png", "pdf", "svg", "yaml" ]:
            print("Invalid conversion format")
            exit(1)
        output = None
        if args.output:
            output = args.output

        if not args.files:
            print("No input file provided")
            exit(1)

        convert_file(args.files[0], 
                     output, 
                     args.convert, 
                     border = args.border, 
                     page_no = page_no)
        exit(0)

    # convert if exactly two file names are provided
    if args.files:
        if len(args.files) > 2:
            print("Too many files provided")
            exit(1)
        elif len(args.files) == 2:
            convert_file(args.files[0], args.files[1], 
                         border = args.border, 
                         page_no = page_no)
            exit(0)
        else:
            _savefile = args.files[0]
    else:
        _savefile = get_default_savefile(app_name, app_author)
    return _savefile

def main():
    """Main function for the application."""
    APP_NAME   = "ScreenDrawer"
    APP_AUTHOR = "JanuaryWeiner"  # used on Windows

    # Get user-specific config directory
    user_config_dir = appdirs.user_config_dir(APP_NAME, APP_AUTHOR)
    log.debug(f"User config directory: {user_config_dir}")

    #user_cache_dir = appdirs.user_cache_dir(APP_NAME, APP_AUTHOR)
    #user_log_dir   = appdirs.user_log_dir(APP_NAME, APP_AUTHOR)

# ---------------------------------------------------------------------
# Parsing command line
# ---------------------------------------------------------------------

    args = parse_arguments()
    log.info(args.debug)
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        log.info("Setting logging level to DEBUG")
    savefile = process_args(args, APP_NAME, APP_AUTHOR)
    log.debug(f"Save file is: {savefile}")

# ---------------------------------------------------------------------

    win = TransparentWindow(save_file = savefile)
    if args.loadfile:
        win.read_file(args.loadfile)

    CSS = b"""
    #myMenu {
        background-color: rgba(255, 255, 255, 0.9); /* Semi-transparent white */
        font-family: 'Ubuntu Mono', Monospace, 'Monospace Regular', monospace, 'Courier New'; /* Use 'Courier New', falling back to any monospace font */
    }
    """

    style_provider = Gtk.CssProvider()
    style_provider.load_from_data(CSS)
    Gtk.StyleContext.add_provider_for_screen(
        Gdk.Screen.get_default(),
        style_provider,
        Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
    )

    win.set_icon(Icons().get("app_icon"))
    win.show_all()
    win.present()
    win.cursor.set(win.state.mode())
    if args.sticky == "yes":
        win.stick()

    Gtk.main()

if __name__ == "__main__":
    main()
