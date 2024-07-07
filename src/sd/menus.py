"""
This module holds the MenuMaker class, which is a singleton class that creates menus.
"""

import gi                                                  # <remove>
gi.require_version('Gtk', '3.0')                           # <remove>
from gi.repository import Gtk # <remove>
import logging                                                   # <remove>
log = logging.getLogger(__name__)                                # <remove>

class MenuMaker:
    """A class holding methods to create menus. Singleton."""
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(MenuMaker, cls).__new__(cls)
        return cls._instance

    def __init__(self, bus, state):
        if not hasattr(self, '_initialized'):
            self.__bus = bus
            self._initialized = True
            self.__state = state
            self.__context_menu = None
            self.__object_menu = None
            self.__bus.on("right_mouse_click", self.on_right_mouse_click)

    def build_menu(self, menu_items):
        """Build a menu from a list of menu items."""
        menu = Gtk.Menu()
        menu.set_name("myMenu")

        for m in menu_items:
            if "separator" in m:
                menu_item = Gtk.SeparatorMenuItem()
            else:
                menu_item = Gtk.MenuItem(label=m["label"])
                menu_item.connect("activate", m["callback"], m["action"])
            menu.append(menu_item)
        menu.show_all()
        return menu

    def on_menu_item_activated(self, widget, params):
        """Callback for when a menu item is activated."""
        log.debug("Menu item activated: %s from %s", params, widget)

        self.__bus.emitMult("queue_draw")
        action = params[0]
        args   = params[1:]
        self.__bus.emit(action, *args)


    def context_menu(self):
        """Build the main context menu"""
        # general menu for everything
        ## build only once
        if self.__context_menu:
            return self.__context_menu

        cb = self.on_menu_item_activated
        menu_items = [
                { "label": "Toggle UI    [w]",                "callback": cb, "action": [ "toggle_wiglets" ] },
                { "separator": True },
                { "label": "Move         [m]",                "callback": cb, "action": [ "mode_set", False, "move" ] },
                { "label": "Pencil       [d]",                "callback": cb, "action": [ "mode_set", False, "draw" ] },
                { "label": "Shape        [s]",                "callback": cb, "action": [ "mode_set", False, "shape" ] },
                { "label": "Text         [t]",                "callback": cb, "action": [ "mode_set", False, "text" ] },
                { "label": "Rectangle    [r]",                "callback": cb, "action": [ "mode_set", False, "box" ] },
                { "label": "Circle       [c]",                "callback": cb, "action": [ "mode_set", False, "circle" ] },
                { "label": "Eraser       [e]",                "callback": cb, "action": [ "mode_set", False, "eraser" ] },
                { "label": "Color picker [i]",                "callback": cb, "action": [ "mode_set", False, "colorpicker" ] },
                { "separator": True },
                { "label": "Select all    (Ctrl-a)",          "callback": cb, "action": [ "set_selection", True, "all" ] },
                { "label": "Paste         (Ctrl-v)",          "callback": cb, "action": [ "paste_content" ] },
                { "label": "Clear drawing (Ctrl-l)",          "callback": cb, "action": [ "clear_page" ] },
                { "separator": True },
                { "label": "Next Page     (Shift-n)",         "callback": cb, "action": [ "next_page" ] },
                { "label": "Prev Page     (Shift-p)",         "callback": cb, "action": [ "prev_page" ] },
                { "label": "Delete Page   (Shift-d)",         "callback": cb, "action": [ "delete_page" ] },
                { "label": "Next Layer    (Ctrl-Shift-n)",    "callback": cb, "action": [ "next_layer" ] },
                { "label": "Prev Layer    (Ctrl-Shift-p)",    "callback": cb, "action": [ "prev_layer" ] },
                { "label": "Delete Layer  (Ctrl-Shift-d)",    "callback": cb, "action": [ "delete_layer" ] },
                { "separator": True },
                { "label": "Bg transparency (Ctrl-b)",        "callback": cb, "action": [ "cycle_bg_transparency" ] },
                { "label": "Toggle outline       [o]",        "callback": cb, "action": [ "toggle_outline" ] },
                { "separator": True },
                { "label": "Color           (Ctrl-k)",        "callback": cb, "action": [ "select_color" ] },
                { "label": "Bg Color  (Shift-Ctrl-k)",        "callback": cb, "action": [ "select_color_bg" ] },
                { "label": "Font            (Ctrl-f)",        "callback": cb, "action": [ "select_font" ] },
                { "separator": True },
                { "label": "Open drawing    (Ctrl-o)",        "callback": cb, "action": [ "open_drawing" ] },
                { "label": "Image from file (Ctrl-i)",        "callback": cb, "action": [ "import_image" ] },
                { "label": "Screenshot      (Ctrl-Shift-f)",  "callback": cb, "action": [ "screenshot" ] },
                { "label": "Save as         (Ctrl-s)",        "callback": cb, "action": [ "save_drawing_as" ] },
                { "label": "Export          (Ctrl-e)",        "callback": cb, "action": [ "export_drawing" ] },
                { "label": "Help            [F1]",            "callback": cb, "action": [ "show_help_dialog" ] },
                { "label": "Quit            (Ctrl-q)",        "callback": cb, "action": [ "app_exit" ] },
        ]

        self.__context_menu = self.build_menu(menu_items)
        return self.__context_menu

    def object_menu(self, objects):
        """Build the object context menu"""
        # when right-clicking on an object
        cb = self.on_menu_item_activated

        menu_items = [
                { "label": "Copy (Ctrl-c)",                  "callback": cb, "action": [ "copy_content" ] },
                { "label": "Cut (Ctrl-x)",                   "callback": cb, "action": [ "cut_content" ] },
                { "separator": True },
                { "label": "Delete (|Del|)",                 "callback": cb, "action": [ "selection_delete" ] },
                { "label": "Group (g)",                      "callback": cb, "action": [ "selection_group" ] },
                { "label": "Ungroup (u)",                    "callback": cb, "action": [ "selection_ungroup" ] },
                { "label": "Export (Ctrl-e)",                "callback": cb, "action": [ "export_drawing" ] },
                { "separator": True },
                { "label": "Move to top (Alt-Page_Up)",      "callback": cb, "action": [ "selection_zmove", False, "top" ] },
                { "label": "Raise (Alt-Up)",                 "callback": cb, "action": [ "selection_zmove", False, "raise" ] },
                { "label": "Lower (Alt-Down)",               "callback": cb, "action": [ "selection_zmove", False, "lower" ] },
                { "label": "Move to bottom (Alt-Page_Down)", "callback": cb, "action": [ "selection_zmove", False, "bottom" ] },
                { "separator": True },
                { "label": "To shape   (Alt-s)",             "callback": cb, "action": [ "transmute_selection", True, "shape" ] },
                { "label": "To drawing (Alt-d)",             "callback": cb, "action": [ "transmute_selection", True, "draw" ] },
                { "label": "Fill toggle (f)",                "callback": cb, "action": [ "selection_fill" ] },
                { "separator": True },
                { "label": "Color (Ctrl-k)",                 "callback": cb, "action": [ "select_color" ] },
                { "label": "Font (Ctrl-f)",                  "callback": cb, "action": [ "select_font" ] },
                { "label": "Help [F1]",                      "callback": cb, "action": [ "show_help_dialog" ] },
                { "label": "Quit (Ctrl-q)",                  "callback": cb, "action": [ "app_exit" ] },
        ]

        # if there is only one object, remove the group menu item
        if len(objects) == 1:
            log.debug("only one object")
            menu_items = [m for m in menu_items if "action" not in m or "selection_group" not in m["action"]]

        group_found = [o for o in objects if o.type == "group"]
        if not group_found:
            log.debug("no group found")
            menu_items = [m for m in menu_items if "action" not in m or "selection_ungroup" not in m["action"]]

        self.__object_menu = self.build_menu(menu_items)

        return self.__object_menu

    def on_right_mouse_click(self, ev):
        """Catch the right mouse click and display the menu"""

        event     = ev.event
        hover_obj = ev.hover()

        if hover_obj:
            self.__state.mode("move")

            if not self.__state.selection().contains(hover_obj):
                self.__state.selection().set([ hover_obj ])

            sel_objects = self.__state.selection().objects
            self.object_menu(sel_objects).popup(None, None,
                                                         None, None,
                                                         event.button, event.time)
        else:
            self.context_menu().popup(None, None,
                                               None, None,
                                               event.button, event.time)
        self.__bus.emit("queue_draw")
        return True
