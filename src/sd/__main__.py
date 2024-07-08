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
from sd.state import State                ###<placeholder sd/state.py>
from sd.drawable import *                 ###<placeholder sd/drawable.py>
from sd.drawer import Drawer              ###<placeholder sd/drawer.py>
from sd.drawable_factory import DrawableFactory         ###<placeholder sd/drawable_factory.py>
from sd.drawable_group import DrawableGroup ##<placeholder sd/drawable_group.py>
from sd.drawable_primitives import Image, Text                    #<placeholder sd/drawable_primitives.py>
from sd.drawable_primitives import Rectangle, Shape, Circle #<placeholder sd/drawable_primitives.py>
from sd.drawable_paths import Path                          #<placeholder sd/drawable_paths.py>
from sd.history import History                              #<placeholder sd/history.py>
from sd.bus import Bus #<placeholder sd/bus.py>
from sd.uibuilder import UIBuilder                          #<placeholder sd/uibuilder.py>


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

        self.set_title("Transparent Drawing Window")
        self.set_decorated(False)
        self.connect("destroy", self.exit)
        self.set_default_size(800, 800)
        self.set_keep_above(True)
        self.maximize()

        screen = self.get_screen()
        visual = screen.get_rgba_visual()

        if visual is not None and screen.is_composited():
            self.set_visual(visual)

        self.set_app_paintable(True)

        self.fixed = Gtk.Fixed()
        self.add(self.fixed)

        self.init_ui(save_file = save_file)

    def init_ui(self, save_file):
        """Initialize the user interface."""

        # Drawing setup
        self.bus                = Bus()

        # we pass the app to the state, because it has the queue_draw
        # method
        self.state              = State(app = self,
                                        bus = self.bus)

        # em has to know about all that to link actions to methods
        em  = EventManager(bus = self.bus, state  = self.state)
        mm  = MenuMaker(self.bus, self.state)

        # mouse gets the mouse events
        self.mouse = MouseCatcher(bus = self.bus, state = self.state)

        # canvas orchestrates the drawing
        self.canvas = Canvas(bus = self.bus, state = self.state)

        self.uibuilder = UIBuilder(state = self.state)
        # autosave
        GLib.timeout_add(AUTOSAVE_INTERVAL, self.uibuilder.autosave)

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

        # load the drawing from the savefile
        self.bus.emit("set_savefile", False, save_file)
        self.uibuilder.read_file(save_file)


    def exit(self, event = None):
        """Exit the application."""
        ## close the savefile_f
        log.info("Exiting")
        self.uibuilder.save_state()
        Gtk.main_quit()

    # ---------------------------------------------------------------------

    def __add_bus_events(self):
        """Add bus events."""

        self.bus.on("app_exit", self.exit)
        self.bus.on("copy_content",      self.copy_content)
        self.bus.on("cut_content",       self.cut_content)
        self.bus.on("duplicate_content", self.duplicate_content)
        self.bus.on("paste_content",     self.paste_content)

    def paste_text(self, clip_text):
        """Enter some text in the current object or create a new object."""

        cobj = self.state.current_obj()
        if cobj and cobj.type == "text":
            cobj.add_text(clip_text.strip())
        else:
            obj = Text([ self.state.cursor_pos() ],
                            pen = self.state.pen(), content=clip_text.strip())
            self.bus.emit("add_object", True, obj)
            self.bus.emitOnce("set_selection", [ obj ])

    def paste_image(self, clip_img):
        """Create an image object from a pixbuf image."""
        obj = Image([ self.state.cursor_pos() ], self.state.pen(), clip_img)
        self.bus.emitMult("add_object", obj)
        self.bus.emitOnce("set_selection", [ obj ])

    def __object_create_copies(self, clip, bb = None):
        """Copy the given object into a new object."""

        cut = clip.is_cut()
        objects = clip.objects

        new_objects = [ ]

        for obj in objects:
            new_obj = obj.duplicate()
            new_objects.append(new_obj)

            # move the new object to the current location
            if not cut:
                x, y = self.state.cursor_pos()
                if bb is None:
                    bb  = new_obj.bbox()
                new_obj.move(x - bb[0], y - bb[1])

            self.bus.emit("add_object", True, new_obj)

        self.bus.emit("set_selection", True, new_objects)

    def copy_content(self, destroy = False):
        """Copy content to clipboard."""
        content = self.state.selection()
        if content.is_empty():
            return

        log.debug("Copying content %s", content)
        self.state.clipboard().copy_content(content, cut = destroy)

        if destroy:
            self.state.gom().remove_selection()

    def paste_content(self):
        """Paste content from clipboard."""
        clip_type, clip = self.state.clipboard().get_content()

        if not clip:
            return

        # internal paste
        if clip_type == "internal":
            log.debug("Pasting content internally")

            if clip.type != "clipboard_group":
                raise ValueError("Internal clipboard is not a clipboard_group")

            bb = clip.bbox()
            log.debug("clipboard bbox %s", bb)

            self.__object_create_copies(clip, bb)

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
        content = self.state.selection()

        if content.is_empty():
            return

        for obj in content.objects:
            new_obj = obj.duplicate()
            self.bus.emitOnce("add_object", new_obj)


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
    log.debug("Save file is: %s", savefile) 

# ---------------------------------------------------------------------

    win = TransparentWindow(save_file = savefile)
    if args.loadfile:
        win.uibuilder.read_file(args.loadfile)

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
    win.state.cursor().set(win.state.mode())
    if args.sticky == "yes":
        win.stick()

    Gtk.main()

if __name__ == "__main__":
    main()
