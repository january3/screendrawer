#!/usr/bin/env python3

##  MIT License
##
##  Copyright (c) 2024 January Weiner
##
##  Permission is hereby granted, free of charge, to any person obtaining a copy
##  of this software and associated documentation files (the "Software"), to deal
##  in the Software without restriction, including without limitation the rights
##  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
##  copies of the Software, and to permit persons to whom the Software is
##  furnished to do so, subject to the following conditions:
##
##  The above copyright notice and this permission notice shall be included in all
##  copies or substantial portions of the Software.
##
##  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
##  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
##  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
##  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
##  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
##  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
##  SOFTWARE.

# ---------------------------------------------------------------------
import gi
import copy
import yaml
import pickle
gi.require_version('Gtk', '3.0')

from gi.repository import Gtk, Gdk, GdkPixbuf, Pango, GLib
import cairo
import os
import time
import math
import base64
import tempfile
from io import BytesIO

import warnings
import appdirs
import argparse

import pyautogui
from PIL import ImageGrab
from os import path

# ---------------------------------------------------------------------
# These are various classes and utilities for the sd.py script. In the
# "skeleton" variant, they are just imported. In the "full" variant, they
# the files are directly, physically inserted in order to get one big fat 
# Python script that can just be copied.

from sd.utils import *              ###<placeholder sd/utils.py>
from sd.commands import *           ###<placeholder sd/commands.py>
from sd.pen import Pen              ###<placeholder sd/pen.py>
from sd.drawable import *           ###<placeholder sd/drawable.py>
from sd.events import *             ###<placeholder sd/events.py>
from sd.dialogs import *   ###<placeholder sd/dialogs.py>
from sd.clipboard import Clipboard  ###<placeholder sd/clipboard.py>
from sd.cursor import CursorManager ###<placeholder sd/cursor.py>
from sd.gom import GraphicsObjectManager ###<placeholder sd/gom.py>
from sd.import_export import *      ###<placeholder sd/import_export.py>

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

    def __init__(self, savefile = None):
        super(TransparentWindow, self).__init__()

        self.savefile            = savefile
        self.init_ui()
    
    def init_ui(self):
        self.set_title("Transparent Drawing Window")
        self.set_decorated(False)
        self.connect("destroy", self.exit)
        self.set_default_size(800, 600)

        # transparency
        screen = self.get_screen()
        visual = screen.get_rgba_visual()
        if visual != None and screen.is_composited():
            self.set_visual(visual)
        self.set_app_paintable(True)

        # connecting events
        self.connect("draw", self.on_draw)
        self.connect("key-press-event", self.on_key_press)
        self.set_events(Gdk.EventMask.BUTTON_PRESS_MASK | Gdk.EventMask.BUTTON_RELEASE_MASK | Gdk.EventMask.POINTER_MOTION_MASK)
        self.connect("button-press-event",   self.on_button_press)
        self.connect("button-release-event", self.on_button_release)
        self.connect("motion-notify-event",  self.on_motion_notify)

        # autosave
        GLib.timeout_add(AUTOSAVE_INTERVAL, self.autosave)

        # Drawing setup
        self.mode                = "draw"
        self.gom                 = GraphicsObjectManager(self)
        self.clipboard           = Clipboard()
        self.cursor              = CursorManager(self)
        self.hidden              = False

        self.current_object      = None
        self.wiglet_active       = None
        self.resizeobj           = None
        self.hover               = None
        self.selection_tool      = None

        # defaults for drawing
        self.pen  = Pen(line_width = 4,  color = (0.2, 0, 0), font_size = 24, transparency  = 1)
        self.pen2 = Pen(line_width = 40, color = (1, 1, 0),   font_size = 24, transparency = .2)
        self.transparent = 0
        self.outline     = False

        # distance for selecting objects
        self.max_dist   = 15

        self.objects = [ ]
        self.load_state()
        self.modified = False # for autosave

        self.create_context_menu()
        self.create_object_menu()
        self.set_keep_above(True)
        self.maximize()

    def on_menu_item_activated(self, widget, data):
        print("Menu item activated:", data)

        self.handle_shortcuts(data)


    def create_context_menu(self):
        menu_items = [
                { "label": "Move         [m]",        "callback": self.on_menu_item_activated, "data": "m" },
                { "label": "Pencil       [d]",        "callback": self.on_menu_item_activated, "data": "d" },
                { "label": "Polygon      [p]",        "callback": self.on_menu_item_activated, "data": "d" },
                { "label": "Text         [t]",        "callback": self.on_menu_item_activated, "data": "t" },
                { "label": "Rectangle    [r]",        "callback": self.on_menu_item_activated, "data": "r" },
                { "label": "Circle       [c]",        "callback": self.on_menu_item_activated, "data": "c" },
                { "label": "Eraser       [e]",        "callback": self.on_menu_item_activated, "data": "e" },
                { "label": "Color picker [i]",        "callback": self.on_menu_item_activated, "data": "i" },
                { "separator": True },
                { "label": "Select all    (Ctrl-a)",  "callback": self.on_menu_item_activated, "data": "Ctrl-a" },
                { "label": "Paste         (Ctrl-v)",  "callback": self.on_menu_item_activated, "data": "Ctrl-v" },
                { "label": "Clear drawing (Ctrl-l)",  "callback": self.on_menu_item_activated, "data": "Ctrl-l" },
                { "separator": True },
                { "label": "Color           (Ctrl-k)",  "callback": self.on_menu_item_activated, "data": "Ctrl-k" },
                { "label": "Image from file (Ctrl-i)",  "callback": self.on_menu_item_activated, "data": "Ctrl-i" },
                { "label": "Export drawing  (Ctrl-e)",  "callback": self.on_menu_item_activated, "data": "Ctrl-e" },
                { "label": "Font            (Ctrl-f)",  "callback": self.on_menu_item_activated, "data": "Ctrl-f" },
                { "label": "Help            [F1]",      "callback": self.on_menu_item_activated, "data": "h" },
                { "label": "Quit            (Ctrl-q)",  "callback": self.on_menu_item_activated, "data": "x" },
        ]

        self.context_menu = build_menu(menu_items)

    def create_object_menu(self):
        menu_items = [
                { "label": "Copy (Ctrl-c)",        "callback": self.on_menu_item_activated, "data": "Ctrl-c" },
                { "label": "Cut (Ctrl-x)",         "callback": self.on_menu_item_activated, "data": "Ctrl-x" },
                { "separator": True },
                { "label": "Delete (|Del|)",       "callback": self.on_menu_item_activated, "data": "Delete" },
                { "label": "Group (g)",            "callback": self.on_menu_item_activated, "data": "g" },
                { "label": "Ungroup (u)",          "callback": self.on_menu_item_activated, "data": "u" },
                { "separator": True },
                { "label": "Color (Ctrl-k)",       "callback": self.on_menu_item_activated, "data": "Ctrl-k" },
                { "label": "Font (Ctrl-f)",        "callback": self.on_menu_item_activated, "data": "Ctrl-f" },
                { "label": "Help [F1]",            "callback": self.on_menu_item_activated, "data": "h" },
                { "label": "Quit (Ctrl-q)",        "callback": self.on_menu_item_activated, "data": "x" },
        ]
        self.object_menu = build_menu(menu_items)


    def exit(self):
        ## close the savefile_f
        print("Exiting")
        self.save_state()
        Gtk.main_quit()

    def on_draw(self, widget, cr):
        """Handle draw events."""
        if self.hidden:
            print("I am hidden!")
            return True

        cr.set_source_rgba(.8, .75, .65, self.transparent)
        cr.set_operator(cairo.OPERATOR_SOURCE)
        cr.paint()
        cr.set_operator(cairo.OPERATOR_OVER)
        self.draw(cr)

        if self.current_object:
            self.current_object.draw(cr)

        if self.wiglet_active:
            self.wiglet_active.draw(cr)
        return True

    def draw(self, cr):
        """Draw the objects in the given context. Used also by export functions."""

        for obj in self.gom.objects():
            hover    = obj == self.hover and self.mode == "move"
            selected = self.gom.selection.contains(obj) and self.mode == "move"
            obj.draw(cr, hover=hover, selected=selected, outline = self.outline)
    #   if self.current_object:
    #       print("drawing current object:", self.current_object, "mode:", self.mode)
    #       self.current_object.draw(cr)

        # If changing line width, draw a preview of the new line width
      
    def clear(self):
        """Clear the drawing."""
        self.gom.selection.clear()
        self.resizeobj      = None
        self.current_object = None
        self.gom.remove_all()
        self.queue_draw()

    # ---------------------------------------------------------------------
    #                              Event handlers

    def on_right_click(self, event, hover_obj):
        """Handle right click events - context menus."""
        if hover_obj:
            self.mode = "move"
            self.cursor.default(self.mode)

            if not self.gom.selection.contains(hover_obj):
                self.gom.selection.set([ hover_obj ])

            self.object_menu.popup(None, None, None, None, event.button, event.time)
            self.queue_draw()
        else:
            self.context_menu.popup(None, None, None, None, event.button, event.time)

    # ---------------------------------------------------------------------

    def move_resize_rotate(self, ev):
        """Process events for moving, resizing and rotating objects."""
        corner_obj = ev.corner()
        hover_obj  = ev.hover()
        pos = ev.pos()
        shift, ctrl = ev.shift(), ev.ctrl()

        if corner_obj[0] and corner_obj[0].bbox():
            print("starting resize")
            obj    = corner_obj[0]
            corner = corner_obj[1]
            print("ctrl:", ctrl, "shift:", shift)
            # XXX this code here is one of the reasons why rotating or resizing the
            # whole selection does not work. The other reason is that the
            # selection object itself is not considered when detecting
            # hover or corner objects.
            if ctrl and shift:
                self.resizeobj = RotateCommand(obj, origin = pos, corner = corner)
            else:
                self.resizeobj = ResizeCommand(obj, origin = pos, corner = corner, proportional = ctrl)
            self.gom.selection.set([ obj ])
            # XXX - this should happen through GOM and upon mouse release 
            # self.history.append(self.resizeobj)
            self.cursor.set(corner)
        elif hover_obj:
            if ev.shift():
                # add if not present, remove if present
                print("adding object", hover_obj)
                self.gom.selection.add(hover_obj)
            if not self.gom.selection.contains(hover_obj):
                print("object not in selection, setting it", hover_obj)
                self.gom.selection.set([ hover_obj ])
            self.resizeobj = MoveCommand(self.gom.selection, pos)
            # XXX - this should happen through GOM and upon mouse release 
            # self.history.append(self.resizeobj)
            self.cursor.set("grabbing")
        else:
            self.gom.selection.clear()
            self.resizeobj   = None
            print("starting selection tool")
            self.selection_tool = SelectionTool([ pos, (pos[0] + 1, pos[1] + 1) ])
            self.current_object = self.selection_tool
            self.queue_draw()
        return True

    def create_object(self, ev):
        """Create an object based on the current mode."""
        # not managed by GOM: first create, then decide whether to add to GOM
        obj = DrawableFactory.create_drawable(self.mode, pen = self.pen, ev=ev)
        if obj:
            self.current_object = obj

    # XXX this code should be completely rewritten, cleaned up, refactored
    # and god knows what else. It's a mess.
    def on_button_press(self, widget, event):
        print("on_button_press: type:", event.type, "button:", event.button, "state:", event.state)
        self.modified = True # better safe than sorry

        ev = MouseEvent(event, self.gom.objects())
        shift, ctrl, pressure = ev.shift(), ev.ctrl(), ev.pressure()
        hover_obj = ev.hover()

        # double click on a text object: start editing
        if event.button == 1 and ev.double() and hover_obj and hover_obj.type == "text" and self.mode in ["draw", "text", "move"]:
            # put the cursor in the last line, end of the text
            # this should be a Command event
            hover_obj.move_caret("End")
            self.current_object = hover_obj
            self.queue_draw()
            self.cursor.set("none")
            return True

        # Ignore clicks when text input is active
        if self.current_object:
            if  self.current_object.type == "text":
                print("click, but text input active - finishing it first")
                self.finish_text_input()
            else:
                print("click, but text input active - ignoring it; object=", self.current_object)
            return True

        # right click: open context menu
        if event.button == 3 and not shift:
            self.on_right_click(event, hover_obj)
            return True

        if event.button != 1:
            return True

        # Start changing line width: single click with ctrl pressed
        if ctrl and event.button == 1 and self.mode == "draw": 
            if not shift: 
                self.wiglet_active = WigletLineWidth((event.x, event.y), self.pen)
            else:
                self.wiglet_active = WigletTransparency((event.x, event.y), self.pen)
            return True

        if self.mode == "picker":
            #print("picker mode")
            color = get_color_under_cursor()
            self.set_color(color) 
            color_hex = rgb_to_hex(color)
            self.clipboard.set_text(color_hex)
            return True

        elif self.mode == "move":
            return self.move_resize_rotate(ev)

        # erasing an object, if an object is underneath the cursor
        elif self.mode == "eraser" and hover_obj: 
                ## XXX -> GOM 
                # self.history.append(RemoveCommand([ hover_obj ], self.objects))
                self.gom.remove_objects([ hover_obj ], clear_selection = True)
                self.resizeobj   = None
                self.cursor.revert()

        # simple click: create modus
        else:
            self.create_object(ev)

        self.queue_draw()

        return True

    # Event handlers
    # XXX same comment as above
    def on_button_release(self, widget, event):
        """Handle mouse button release events."""
        obj = self.current_object

        if obj and obj.type in [ "polygon", "path" ]:
            print("finishing path / polygon")
            obj.path_append(event.x, event.y, 0)
            obj.finish()
            if len(obj.coords) < 3:
                obj = None
            self.queue_draw()

        if obj:
            # remove objects that are too small
            bb = obj.bbox()
            if bb and obj.type in [ "box", "circle" ] and bb[2] == 0 and bb[3] == 0:
                obj = None

        if obj:
            if obj != self.selection_tool:
                self.gom.add_object(obj)
            else:
                self.current_object = None

        if self.wiglet_active:
            self.wiglet_active.event_finish()
            self.wiglet_active = None
            self.queue_draw()
            return True

        # if selection tool is active, finish it
        if self.selection_tool:
            print("finishing selection tool")
            #self.objects.remove(self.selection_tool)
            #bb = self.selection_tool.bbox()
            objects = self.selection_tool.objects_in_selection(self.gom.objects())
            if len(objects) > 0:
                self.gom.selection.set(objects)
            else:
                self.gom.selection.clear()
            self.selection_tool = None
            self.queue_draw()
            return True

        # if the user clicked to create a text, we are not really done yet
        if self.current_object and self.current_object.type != "text":
            print("there is a current object: ", self.current_object)
            self.gom.selection.clear()
            self.current_object = None
            self.queue_draw()
            return True

        if self.resizeobj:
            # If the user was dragging a selected object and the drag ends
            # in the lower left corner, delete the object
            self.resizeobj.event_finish()
            obj = self.resizeobj.obj
            if self.resizeobj.command_type == "move" and  event.x < 10 and event.y > self.get_size()[1] - 10:
                # command group because these are two commands: first move,
                # then remove
                self.gom.command_append([ self.resizeobj, RemoveCommand([ obj ], self.gom.objects()) ])
                self.selection.clear()
            else:
                self.gom.command_append([ self.resizeobj ])
            self.resizeobj    = None
            self.cursor.revert()
            self.queue_draw()
        return True


    def on_motion_notify(self, widget, event):
        """Handle mouse motion events."""

        ev = MouseEvent(event, self.gom.objects())
        x, y = ev.pos()
        self.cursor.update_pos(x, y)

        if self.wiglet_active:
            self.wiglet_active.event_update(x, y)
            self.queue_draw()
            return True

        obj = self.current_object or self.selection_tool

        if obj:
            obj.update(x, y, ev.pressure())
            self.queue_draw()
        elif self.resizeobj:
            self.resizeobj.event_update(x, y)
            self.queue_draw()
        elif self.mode == "move":
            object_underneath = ev.hover()
            prev_hover = self.hover

            if object_underneath:
                if object_underneath.type == "text":
                    self.cursor.set("text")
                else:
                    self.cursor.set("moving")
                self.hover = object_underneath
            else:
                self.cursor.revert()
                self.hover = None

            corner_obj = ev.corner()

            if corner_obj[0] and corner_obj[0].bbox():
                self.cursor.set(corner_obj[1])
                self.hover = corner_obj[0]
                self.queue_draw()

            if prev_hover != self.hover:
                self.queue_draw()

        # stop event propagation
        return True

    # ---------------------------------------------------------------------

    def finish_text_input(self):
        """Clean up current text and finish text input."""
        print("finishing text input")
        if self.current_object and self.current_object.type == "text":
            self.current_object.caret_pos = None
            if self.current_object.strlen() == 0:
                self.gom.kill_object(self.current_object)
            self.current_object = None
        self.cursor.revert()

    def update_text_input(self, keyname, char):
        """Update the current text input."""
        cur  = self.current_object
        if not cur:
            raise ValueError("No current object")
        cur.update_by_key(keyname, char)

    def cycle_background(self):
        """Cycle through background transparency."""
        self.transparent = {1: 0, 0: 0.5, 0.5: 1}[self.transparent]

    def paste_text(self, clip_text):
        """Enter some text in the current object or create a new object."""

        if self.current_object and self.current_object.type == "text":
            self.current_object.add_text(clip_text.strip())
        else:
            new_text = Text([ self.cursor.pos() ], 
                            pen = self.pen, content=clip_text.strip())
            self.gom.add_object(new_text)

    def paste_image(self, clip_img):
        """Create an image object from a pixbuf image."""
        obj = Image([ self.cursor.pos() ], self.pen, clip_img)
        self.gom.add_object(obj)

    def object_create_copy(self, obj, bb = None):
        """Copy the given object into a new object."""
        new_obj = copy.deepcopy(obj.to_dict())
        new_obj = Drawable.from_dict(new_obj)

        # move the new object to the current location
        x, y = self.cursor.pos()
        if bb is None:
            bb  = new_obj.bbox()
        new_obj.move(x - bb[0], y - bb[1])

        self.gom.add_object(new_obj)

    def paste_content(self):
        """Paste content from clipboard."""
        clip_type, clip = self.clipboard.get_content()

        if not clip:
            return

        # internal paste
        if clip_type == "internal":
            print("Pasting content internally")
            if clip.type != "group":
                Raise("Internal clipboard is not a group")
            bb = clip.bbox()
            print("clipboard bbox:", bb)
            for obj in clip.objects:
                self.object_create_copy(obj, bb)
            return

        elif clip_type == "text":
            self.paste_text(clip)
        elif clip_type == "image":
            self.paste_image(clip)

    def copy_content(self, destroy = False):
        """Copy content to clipboard."""
        if self.gom.selection.is_empty():
            # nothing selected
            print("Nothing selected")
            return

        print("Copying content", self.gom.selection)
        self.clipboard.copy_content(self.gom.selection)

        if destroy:
            self.gom.remove_selection()

    def cut_content(self):
        """Cut content to clipboard."""
        self.copy_content(True)
   
    def stroke_increase(self):
        """Increase whatever is selected."""
        self.stroke_change(1)

    def stroke_decrease(self):
        """Decrease whatever is selected."""
        self.stroke_change(-1)

    def stroke_change(self, direction):
        """Modify the line width or text size."""
        print("Changing stroke", direction)
        if self.current_object and self.current_object.type == "text":
            print("Changing text size")
            self.current_object.stroke_change(direction)
            self.pen.font_size = self.current_object.pen.font_size
        else: 
            for obj in self.gom.selected_objects():
                obj.stroke_change(direction)

        # without a selected object, change the default pen, but only if in the correct mode
        if self.mode == "draw":
            self.pen.line_width = max(1, self.pen.line_width + direction)
        elif self.mode == "text":
            self.pen.font_size = max(1, self.pen.font_size + direction)

    def outline_toggle(self):
        """Toggle outline mode."""
        self.outline = not self.outline

    def set_color(self, color):
        self.pen.color_set(color)
        self.gom.selection_color_set(color)

    def set_font(self, font_description):
        """Set the font."""
        self.pen.font_set_from_description(font_description)
        self.gom.selection_font_set(font_description)
        if self.current_object and self.current_object.type == "text":
            self.current_object.pen.font_set_from_description(font_description)

    def transmute(self, mode):
        """Change the selected object(s) to a polygon."""
        print("transmuting to", mode)
        sel = self.gom.selected_objects()
        # note to self: sel is a list, not the selection
        if sel:
            self.gom.transmute(sel, mode)

#   def smoothen(self):
#       """Smoothen the selected object."""
#       if self.selection.n() > 0:
#           for obj in self.selection.objects:
#               obj.smoothen()

    def switch_pens(self):
        """Switch between pens."""
        self.pen, self.pen2 = self.pen2, self.pen
        self.queue_draw()

    def handle_shortcuts(self, keyname):
        """Handle keyboard shortcuts."""
        print(keyname)
        self.modified = True # better safe than sorry

        # these are single keystroke mode modifiers
        modes = { 'm': "move", 's': "move", 'space': "move", 
                  'd': "draw", 't': "text", 'e': "eraser", 
                  'i': "picker",
                  'c': "circle", 'r': "box", 'p': "polygon" }

        # these are single keystroke actions
        actions = {
            'h':                    {'action': self.show_help_dialog},
            'F1':                   {'action': self.show_help_dialog},
            'question':             {'action': self.show_help_dialog},
            'Shift-question':       {'action': self.show_help_dialog},
            'Ctrl-l':               {'action': self.clear},
            'Ctrl-b':               {'action': self.cycle_background},
            'x':                    {'action': self.exit},
            'q':                    {'action': self.exit},
            'Ctrl-q':               {'action': self.exit},
            'l':                    {'action': self.clear},
            # XXX something is rotten here
            #'f':                    {'action': self.gom.selection_fill, 'modes': ["box", "circle", "draw", "move"]},
            'o':                    {'action': self.outline_toggle, 'modes': ["box", "circle", "draw", "move"]},
            'Alt-p':                {'action': self.gom.transmute_selection, 'args': [ "polygon" ], 'modes': ["draw", "polygon", "move"]},
            'Alt-P':                {'action': self.gom.transmute_selection, 'args': [ "draw" ], 'modes': ["draw", "polygon", "move"]},

            'Up':                   {'action': self.gom.move_selection, 'args': [0, -10],  'modes': ["move"]},
            'Shift-Up':             {'action': self.gom.move_selection, 'args': [0, -1],   'modes': ["move"]},
            'Ctrl-Up':              {'action': self.gom.move_selection, 'args': [0, -100], 'modes': ["move"]},
            'Down':                 {'action': self.gom.move_selection, 'args': [0, 10],   'modes': ["move"]},
            'Shift-Down':           {'action': self.gom.move_selection, 'args': [0, 1],    'modes': ["move"]},
            'Ctrl-Down':            {'action': self.gom.move_selection, 'args': [0, 100],  'modes': ["move"]},
            'Left':                 {'action': self.gom.move_selection, 'args': [-10, 0],  'modes': ["move"]},
            'Shift-Left':           {'action': self.gom.move_selection, 'args': [-1, 0],   'modes': ["move"]},
            'Ctrl-Left':            {'action': self.gom.move_selection, 'args': [-100, 0], 'modes': ["move"]},
            'Right':                {'action': self.gom.move_selection, 'args': [10, 0],   'modes': ["move"]},
            'Shift-Right':          {'action': self.gom.move_selection, 'args': [1, 0],    'modes': ["move"]},
            'Ctrl-Right':           {'action': self.gom.move_selection, 'args': [100, 0],  'modes': ["move"]},

            'Page_Up':              {'action': self.gom.rotate_selection, 'args': [10],  'modes': ["move"]},
            'Shift-Page_Up':        {'action': self.gom.rotate_selection, 'args': [1],   'modes': ["move"]},
            'Ctrl-Page_Up':         {'action': self.gom.rotate_selection, 'args': [90],  'modes': ["move"]},
            'Page_Down':            {'action': self.gom.rotate_selection, 'args': [-10], 'modes': ["move"]},
            'Shift-Page_Down':      {'action': self.gom.rotate_selection, 'args': [-1],  'modes': ["move"]},
            'Ctrl-Page_Down':       {'action': self.gom.rotate_selection, 'args': [-90], 'modes': ["move"]},

            'Alt-Page_Up':          {'action': self.gom.selection_zmove, 'args': [ "top"    ], 'modes': ["move"]},
            'Alt-Page_Down':        {'action': self.gom.selection_zmove, 'args': [ "bottom" ], 'modes': ["move"]},
            'Alt-Up':               {'action': self.gom.selection_zmove, 'args': [ "raise"  ], 'modes': ["move"]},
            'Alt-Down':             {'action': self.gom.selection_zmove, 'args': [ "lower"  ], 'modes': ["move"]},


            'Shift-W':              {'action': self.set_color, 'args': [COLORS["white"]]},
            'Shift-B':              {'action': self.set_color, 'args': [COLORS["black"]]},
            'Shift-R':              {'action': self.set_color, 'args': [COLORS["red"]]},
            'Shift-G':              {'action': self.set_color, 'args': [COLORS["green"]]},
            'Shift-L':              {'action': self.set_color, 'args': [COLORS["blue"]]},
            'Shift-E':              {'action': self.set_color, 'args': [COLORS["grey"]]},
            'Shift-Y':              {'action': self.set_color, 'args': [COLORS["yellow"]]},
            'Shift-P':              {'action': self.set_color, 'args': [COLORS["purple"]]},

            # dialogs
            'Ctrl-e':               {'action': self.export_drawing},
            'Ctrl-k':               {'action': self.select_color},
            'Ctrl-f':               {'action': self.select_font},
            'Ctrl-i':               {'action': self.select_image_and_create_pixbuf},
            'Ctrl-p':               {'action': self.switch_pens},
            'Ctrl-o':               {'action': self.open_drawing},

            # selections and moving objects
            'Tab':                  {'action': self.gom.select_next_object, 'modes': ["move"]},
            'Shift-ISO_Left_Tab':   {'action': self.gom.select_next_object, 'modes': ["move"]},
            'g':                    {'action': self.gom.selection_group,    'modes': ["move"]},
            'u':                    {'action': self.gom.selection_ungroup,  'modes': ["move"]},
            'Delete':               {'action': self.gom.selection_delete,   'modes': ["move"]},
            'Ctrl-a':               {'action': self.gom.select_all},
            'Ctrl-r':               {'action': self.gom.select_reverse},
            'Ctrl-y':               {'action': self.gom.redo},
            'Ctrl-z':               {'action': self.gom.undo},
#            'Ctrl-m':               {'action': self.smoothen,           'modes': ["move"]},
            'Ctrl-c':               {'action': self.copy_content,       'modes': ["move"]},
            'Ctrl-x':               {'action': self.cut_content,        'modes': ["move"]},
            'Ctrl-v':               {'action': self.paste_content},
            'Ctrl-F':               {'action': self.screenshot},

            'Ctrl-plus':            {'action': self.stroke_increase},
            'Ctrl-minus':           {'action': self.stroke_decrease},
        }

        if keyname in modes:
            self.mode = modes[keyname]
            self.cursor.default(modes[keyname])
        elif keyname in actions:
            if not "modes" in actions[keyname] or self.mode in actions[keyname]["modes"]:
                if "args" in actions[keyname]:
                    actions[keyname]["action"](*actions[keyname]["args"])
                else:
                    actions[keyname]["action"]()
        self.queue_draw()
     
    def on_key_press(self, widget, event):
        """Handle keyboard events."""
        keyname = Gdk.keyval_name(event.keyval)
        char    = chr(Gdk.keyval_to_unicode(event.keyval))
        ctrl    = event.state & Gdk.ModifierType.CONTROL_MASK
        shift   = event.state & Gdk.ModifierType.SHIFT_MASK
        alt_l   = event.state & Gdk.ModifierType.MOD1_MASK

        keyfull = keyname
        if shift:
            keyfull = "Shift-" + keyname
        if ctrl:
            keyfull = "Ctrl-" + keyname
        if alt_l:
            keyfull = "Alt-" + keyname

        # End text input
        if keyname == "Escape":
            self.finish_text_input()

        # Handle ctrl-keyboard shortcuts: priority
        elif event.state & ctrl:
            self.handle_shortcuts(keyfull)
       
        # Handle text input
        elif self.current_object and self.current_object.type == "text":
            self.update_text_input(keyname, char)
        else:
        # handle single keystroke shortcuts
            self.handle_shortcuts(keyfull)

        self.queue_draw()
        return True

    def select_color(self):
        """Select a color for drawing."""
        color = ColorChooser(self)
        if color:
            self.set_color((color.red, color.green, color.blue))

    def select_font(self):
        font_description = FontChooser(self.pen, self)

        if font_description:
            self.set_font(font_description)

    def show_help_dialog(self):
        """Show the help dialog."""
        dialog = help_dialog(self)
        response = dialog.run()
        dialog.destroy()

    def export_drawing(self):
        """Save the drawing to a file."""
        # Choose where to save the file
        #    self.export(filename, "svg")
        file_name, file_format = export_dialog(self)
        width, height = self.get_size()
        export_image(width, height, file_name, self.draw, file_format)

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
                img = Image([ pos ], self.pen, pixbuf)
                self.gom.add_object(img)
                self.queue_draw()
        
        return pixbuf

    def screenshot_finalize(self, bb):
        print("Taking screenshot now")
        pixbuf, filename = get_screenshot(self, bb[0] - 3, bb[1] - 3, bb[0] + bb[2] + 6, bb[1] + bb[3] + 6)
        self.hidden = False
        self.queue_draw()

        # Create the image and copy the file name to clipboard
        if pixbuf is not None:
            img = Image([ (bb[0], bb[1]) ], self.pen, pixbuf)
            self.gom.add_object(img)
            self.queue_draw()
            self.clipboard.set_text(filename)

    def find_screenshot_box(self):
        if self.current_object and self.current_object.type == "box":
            return self.current_object
        if self.gom.selection.n() == 1 and self.gom.selected_objects()[0].type == "box":
            return self.gom.selection.objects()[0]

        for obj in self.gom.objects()[::-1]:
            if obj.type == "box":
                return obj
        return None

    def screenshot(self):
        obj = self.find_screenshot_box()
        if not obj:
            print("no suitable box found")
            return

        bb = obj.bbox()
        print("bbox is", bb)
        self.hidden = True
        self.queue_draw()
        while Gtk.events_pending():
            Gtk.main_iteration_do(False)
        GLib.timeout_add(100, self.screenshot_finalize, bb)

    def autosave(self):
        # XXX: not implemented, tracking modifications of state
        if not self.modified:
           return

        if self.current_object: # not while drawing!
            return

        print("Autosaving")
        self.save_state()
        self.modified = False

    def save_state(self): 
        """Save the current drawing state to a file."""
        if not self.savefile:
            print("No savefile set")
            return

        print("savefile:", self.savefile)
        config = {
                'transparent': self.transparent,
                'pen': self.pen.to_dict(),
                'pen2': self.pen2.to_dict()
        }

        objects = self.gom.export_objects()
        save_file_as_sdrw(self.savefile, config, objects)

    def open_drawing(self):
        file_name = open_drawing_dialog(self)
        if self.read_file(file_name):
            print("Setting savefile to", file_name)
            self.savefile = file_name
            self.modified = True

    def read_file(self, filename, load_config = True):
        """Read the drawing state from a file."""
        config, objects = read_file_as_sdrw(filename)

        if objects:
            self.gom.set_objects(objects)

        if config and load_config:
            self.transparent       = config['transparent']
            self.pen               = Pen.from_dict(config['pen'])
            self.pen2              = Pen.from_dict(config['pen2'])
        if config and objects:
            self.modified = True
            return True
        return False

    def load_state(self):
        """Load the drawing state from a file."""
        self.read_file(self.savefile)


## ---------------------------------------------------------------------

if __name__ == "__main__":
    app_name   = "ScreenDrawer"
    app_author = "JanuaryWeiner"  # Optional; used on Windows

    # Get user-specific config directory
    user_config_dir = appdirs.user_config_dir(app_name, app_author)
    print(f"User config directory: {user_config_dir}")

    #user_cache_dir = appdirs.user_cache_dir(app_name, app_author)
    #user_log_dir   = appdirs.user_log_dir(app_name, app_author)


# ---------------------------------------------------------------------

    parser = argparse.ArgumentParser(description="Drawing on the screen")
    parser.add_argument("-s", "--savefile", help="File for automatic save upon exit")
    parser.add_argument("files", nargs="*")

    args     = parser.parse_args()
    savefile = args.savefile or get_default_savefile(app_name, app_author)
    files    = args.files
    print("Save file is:", savefile)

# ---------------------------------------------------------------------

    win = TransparentWindow(savefile = savefile)
    if files:
        win.read_file(files[0])

    css = b"""
    #myMenu {
        background-color: rgba(255, 255, 255, 0.9); /* Semi-transparent white */
        font-family: Monospace, 'Monospace regular', monospace, 'Courier New'; /* Use 'Courier New', falling back to any monospace font */
    }
    """

    style_provider = Gtk.CssProvider()
    style_provider.load_from_data(css)
    Gtk.StyleContext.add_provider_for_screen(
        Gdk.Screen.get_default(),
        style_provider,
        Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
    )

    win.show_all()
    win.present()
    win.cursor.set(win.mode)
    win.stick()

    Gtk.main()

