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

from sd.utils import *                   ###<placeholder sd/utils.py>
from sd.commands import *                ###<placeholder sd/commands.py>
from sd.pen import Pen                   ###<placeholder sd/pen.py>
from sd.drawable import *                ###<placeholder sd/drawable.py>
from sd.events import *                  ###<placeholder sd/events.py>
from sd.dialogs import *                 ###<placeholder sd/dialogs.py>
from sd.clipboard import Clipboard       ###<placeholder sd/clipboard.py>
from sd.cursor import CursorManager      ###<placeholder sd/cursor.py>
from sd.gom import GraphicsObjectManager ###<placeholder sd/gom.py>
from sd.import_export import *           ###<placeholder sd/import_export.py>
from sd.em import *                      ###<placeholder sd/em.py>
from sd.menus import *                   ###<placeholder sd/menus.py>
###<placeholder sd/wiglets.py>
from sd.dm import *                      ###<placeholder sd/dm.py>
from sd.icons import Icons               ###<placeholder sd/icons.py>
from sd.page import Page                 ###<placeholder sd/page.py>
from sd.canvas import Canvas             ###<placeholder sd/canvas.py>
from sd.brush import Brush               ###<placeholder sd/brush.py>
from sd.grid import Grid                 ###<placeholder sd/grid.py>
from .texteditor import TextEditor       ###<placeholder sd/texteditor.py>


# ---------------------------------------------------------------------
# defaults


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

        # autosave
        GLib.timeout_add(AUTOSAVE_INTERVAL, self.__autosave)

        # Drawing setup
        self.clipboard           = Clipboard()
        self.canvas              = Canvas()
        self.gom                 = GraphicsObjectManager(self, canvas=self.canvas)
        self.cursor              = CursorManager(self)

        # dm needs to know about gom because gom manipulates the selection
        # and history (object creation etc)
        # it needs to know about the cursor to change the cursor if needed
        # it needs to know about the canvas to get the pen
        # it needs to know about the app to get the size of the window and
        # queue up redraws
        self.dm                  = DrawManager(gom = self.gom,  app = self,
                                               cursor = self.cursor, canvas = self.canvas)

        # em has to know about all that to link actions to methods
        self.em                  = EventManager(gom = self.gom, app = self,
                                                dm = self.dm, canvas = self.canvas)
        self.mm                  = MenuMaker(self.em, self)

        # distance for selecting objects
        self.max_dist   = 15

        # load the drawing from the savefile
        self.load_state()

        # connecting events

       #XXX doesn't work
       #self.gesture_pan = Gtk.GesturePan.new(self, orientation=Gtk.Orientation.VERTICAL)
       #self.gesture_pan.connect('pan', self.dm.on_pan)
       #self.gesture_pan.set_propagation_phase(Gtk.PropagationPhase.CAPTURE)

       ## Gesture for zoom
       #self.gesture_zoom = Gtk.GestureZoom.new(self)
       #self.gesture_zoom.connect('begin', self.dm.on_zoom)
       #self.gesture_zoom.set_propagation_phase(Gtk.PropagationPhase.CAPTURE)

        self.set_events(Gdk.EventMask.BUTTON_PRESS_MASK |
                        Gdk.EventMask.BUTTON_RELEASE_MASK |
                        Gdk.EventMask.POINTER_MOTION_MASK |
                        Gdk.EventMask.TOUCH_MASK)

        self.connect("key-press-event",      self.em.on_key_press)
        self.connect("draw",                 self.dm.on_draw)
        self.connect("button-press-event",   self.dm.on_button_press)
        self.connect("button-release-event", self.dm.on_button_release)
        self.connect("motion-notify-event",  self.dm.on_motion_notify)

    def exit(self):
        """Exit the application."""
        ## close the savefile_f
        print("Exiting")
        self.__save_state()
        Gtk.main_quit()

    # ---------------------------------------------------------------------

    def paste_text(self, clip_text):
        """Enter some text in the current object or create a new object."""

        cobj = self.dm.current_object()
        if cobj and cobj.type == "text":
            cobj.add_text(clip_text.strip())
        else:
            new_text = Text([ self.cursor.pos() ],
                            pen = self.canvas.pen(), content=clip_text.strip())
            self.gom.add_object(new_text)

    def paste_image(self, clip_img):
        """Create an image object from a pixbuf image."""
        obj = Image([ self.cursor.pos() ], self.canvas.pen(), clip_img)
        self.gom.add_object(obj)

    def __object_create_copy(self, obj, bb = None):
        """Copy the given object into a new object."""
        new_obj = copy.deepcopy(obj.to_dict())
        new_obj = Drawable.from_dict(new_obj)

        # move the new object to the current location
        x, y = self.cursor.pos()
        if bb is None:
            bb  = new_obj.bbox()
        new_obj.move(x - bb[0], y - bb[1])

        self.gom.add_object(new_obj)

    def copy_content(self, destroy = False):
        """Copy content to clipboard."""
        content = self.gom.selection()
        if content.is_empty():
            # nothing selected
            print("Nothing selected, selecting all objects")
            content = DrawableGroup(self.gom.objects())

        print("Copying content", content)
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
            print("Pasting content internally")
            if clip.type != "group":
                raise ValueError("Internal clipboard is not a group")
            bb = clip.bbox()
            print("clipboard bbox:", bb)
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

    def select_color_bg(self):
        """Select a color for the background using ColorChooser."""
        color = ColorChooser(self, "Select Background Color")
        if color:
            self.canvas.bg_color((color.red, color.green, color.blue))

    def select_color(self):
        """Select a color for drawing using ColorChooser dialog."""
        color = ColorChooser(self)
        if color:
            self.dm.set_color((color.red, color.green, color.blue))

    def select_font(self):
        """Select a font for drawing using FontChooser dialog."""
        font_description = FontChooser(self.canvas.pen(), self)

        if font_description:
            self.dm.set_font(font_description)

    def show_help_dialog(self):
        """Show the help dialog."""
        dialog = help_dialog(self)
        dialog.run()
        dialog.destroy()

    def save_drawing_as(self):
        """Save the drawing to a file."""
        print("opening save file dialog")
        file = save_dialog(self)
        if file:
            self.savefile = file
            print("setting savefile to", file)
            self.__save_state()

    def export_drawing(self):
        """Save the drawing to a file."""
        # Choose where to save the file
        #    self.export(filename, "svg")
        bbox = None
        if self.gom.selected_objects():
            # only export the selected objects
            obj = self.gom.selected_objects()
        else:
            # set bbox so we export the whole screen
            obj = self.gom.objects()
            bbox = (0, 0, *self.get_size())

        if not obj:
            print("Nothing to draw")
            return

        obj = DrawableGroup(obj)
        file_name, file_format = export_dialog(self)

        if file_name:
            export_image(obj, file_name, file_format,
                         bg = self.canvas.bg_color(),
                         bbox = bbox, transparency = self.canvas.transparent())

    def select_image_and_create_pixbuf(self):
        """Select an image file and create a pixbuf from it."""

        image_file = import_image_dialog(self)
        pixbuf = None

        if image_file:
            try:
                pixbuf = GdkPixbuf.Pixbuf.new_from_file(image_file)
                print(f"Loaded image: {image_file}")
            except Exception as e:
                print(f"Failed to load image: {e}")

            if pixbuf is not None:
                pos = self.cursor.pos()
                img = Image([ pos ], self.canvas.pen(), pixbuf)
                self.gom.add_object(img)
                self.queue_draw()

        return pixbuf

    def __screenshot_finalize(self, bb):
        """Finish up the screenshot."""
        print("Taking screenshot now")
        frame = (bb[0] - 3, bb[1] - 3, bb[0] + bb[2] + 6, bb[1] + bb[3] + 6)
        pixbuf, filename = get_screenshot(self, *frame)
        self.dm.hide(False)
        self.queue_draw()

        # Create the image and copy the file name to clipboard
        if pixbuf is not None:
            img = Image([ (bb[0], bb[1]) ], self.canvas.pen(), pixbuf)
            self.gom.add_object(img)
            self.queue_draw()
            self.clipboard.set_text(filename)

    def __find_screenshot_box(self):
        """Find a box suitable for selecting a screenshot."""
        cobj = self.dm.current_object()
        if cobj and cobj.type == "rectangle":
            return cobj
        for obj in self.gom.selected_objects():
            if obj.type == "rectangle":
                return obj
        for obj in self.gom.objects()[::-1]:
            if obj.type == "rectangle":
                return obj
        return None

    def screenshot(self):
        """Take a screenshot and add it to the drawing."""
        obj = self.__find_screenshot_box()
        if not obj:
            print("no suitable box found")
            # use the whole screen
            bb = (0, 0, *self.get_size())
        else:
            bb = obj.bbox()
            print("bbox is", bb)
        #self.hidden = True
        self.dm.hide(True)
        self.queue_draw()
        while Gtk.events_pending():
            Gtk.main_iteration_do(False)
        GLib.timeout_add(100, self.__screenshot_finalize, bb)

    def __autosave(self):
        """Autosave the drawing state."""
        if not self.dm.modified():
            return

        if self.dm.current_object(): # not while drawing!
            return

        print("Autosaving")
        self.__save_state()
        self.dm.modified(False)

    def __save_state(self):
        """Save the current drawing state to a file."""
        if not self.savefile:
            print("No savefile set")
            return

        print("savefile:", self.savefile)
        config = {
                'bg_color':    self.canvas.bg_color(),
                'transparent': self.canvas.transparent(),
                'show_wiglets': self.dm.show_wiglets(),
                'bbox':        (0, 0, *self.get_size()),
                'pen':         self.canvas.pen().to_dict(),
                'pen2':        self.canvas.pen(alternate = True).to_dict(),
                'page':        self.gom.current_page_number()
        }

        #objects = self.gom.export_objects()
        pages   = self.gom.export_pages()

        save_file_as_sdrw(self.savefile, config, pages = pages)

    def open_drawing(self):
        """Open a drawing from a file in native format."""
        file_name = open_drawing_dialog(self)
        if self.read_file(file_name):
            print("Setting savefile to", file_name)
            self.savefile = file_name
            self.dm.modified(True)

    def read_file(self, filename, load_config = True):
        """Read the drawing state from a file."""
        config, objects, pages = read_file_as_sdrw(filename)

        if pages:
            self.gom.set_pages(pages)
        elif objects:
            self.gom.set_objects(objects)

        if config and load_config:
            self.canvas.bg_color(config.get('bg_color') or (.8, .75, .65))
            self.canvas.transparent(config.get('transparent') or 0)
            show_wiglets = config.get('show_wiglets')
            if show_wiglets is None:
                show_wiglets = True
            self.dm.show_wiglets(show_wiglets)
            self.canvas.pen(pen = Pen.from_dict(config['pen']))
            self.canvas.pen(pen = Pen.from_dict(config['pen2']), alternate = True)
            self.gom.set_page_number(config.get('page') or 0)
        if config or objects:
            self.dm.modified(True)
            return True
        return False

    def load_state(self):
        """Load the drawing state from a file."""
        self.read_file(self.savefile)


## ---------------------------------------------------------------------

def main():
    """Main function for the application."""
    APP_NAME   = "ScreenDrawer"
    APP_AUTHOR = "JanuaryWeiner"  # Optional; used on Windows

    # Get user-specific config directory
    user_config_dir = appdirs.user_config_dir(APP_NAME, APP_AUTHOR)
    print(f"User config directory: {user_config_dir}")

    #user_cache_dir = appdirs.user_cache_dir(APP_NAME, APP_AUTHOR)
    #user_log_dir   = appdirs.user_log_dir(APP_NAME, APP_AUTHOR)


# ---------------------------------------------------------------------
# Parsing command line
# ---------------------------------------------------------------------

    parser = argparse.ArgumentParser(
            description="Drawing on the screen",
            epilog=f"Alternative use: {argv[0]} file.sdrw file.[png, pdf, svg]")
    parser.add_argument("-l", "--loadfile", help="Load drawing from file")
    parser.add_argument("-c", "--convert",
                        help="""
Convert screendrawer file to given format (png, pdf, svg) and exit
(use -o to specify output file, otherwise a default name is used)
"""
    )

    parser.add_argument("-b", "--border", help="Border width for conversion", type=int)
    parser.add_argument("-o", "--output", help="Output file for conversion")
    parser.add_argument("files", nargs="*")
    args     = parser.parse_args()

    if args.convert:
        if not args.convert in [ "png", "pdf", "svg" ]:
            print("Invalid conversion format")
            exit(1)
        OUTPUT = None
        if args.output:
            OUTPUT = args.output

        if not args.files:
            print("No input file provided")
            exit(1)
        convert_file(args.files[0], OUTPUT, args.convert, border = args.border)
        exit(0)

    if args.files:
        if len(args.files) > 2:
            print("Too many files provided")
            exit(1)
        elif len(args.files) == 2:
            convert_file(args.files[0], args.files[1], border = args.border)
            exit(0)
        else:
            savefile = args.files[0]
    else:
        savefile = get_default_savefile(APP_NAME, APP_AUTHOR)
    print("Save file is:", savefile)

# ---------------------------------------------------------------------

    win = TransparentWindow(save_file = savefile)
    if args.loadfile:
        win.read_file(args.loadfile)

    CSS = b"""
    #myMenu {
        background-color: rgba(255, 255, 255, 0.9); /* Semi-transparent white */
        font-family: Monospace, 'Monospace regular', monospace, 'Courier New'; /* Use 'Courier New', falling back to any monospace font */
    }
    """

    style_provider = Gtk.CssProvider()
    style_provider.load_from_data(CSS)
    Gtk.StyleContext.add_provider_for_screen(
        Gdk.Screen.get_default(),
        style_provider,
        Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
    )

    win.show_all()
    win.present()
    win.cursor.set(win.dm.mode())
    win.stick()

    Gtk.main()

if __name__ == "__main__":
    main()
