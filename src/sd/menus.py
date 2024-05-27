"""
This module holds the MenuMaker class, which is a singleton class that creates menus.
"""

import gi                                                  # <remove>
gi.require_version('Gtk', '3.0')                           # <remove>
from gi.repository import Gtk # <remove>

class MenuMaker:
    """A class holding methods to create menus. Singleton."""
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(MenuMaker, cls).__new__(cls)
        return cls._instance

    def __init__(self, bus, gom, em, app):
        if not hasattr(self, '_initialized'):
            self.__bus = bus
            self._initialized = True
            self.__app = app # App
            self.__gom = gom
            self.__em  = em  # EventManager
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

    def on_menu_item_activated(self, widget, action):
        """Callback for when a menu item is activated."""
        print("Menu item activated:", action, "from", widget)

        self.__em.dispatch_action(action)
        self.__app.queue_draw()


    def context_menu(self):
        """Build the main context menu"""
        # general menu for everything
        ## build only once
        if self.__context_menu:
            return self.__context_menu

        menu_items = [
                { "label": "Toggle UI    [w]",          "callback": self.on_menu_item_activated, "action": "toggle_wiglets" },
                { "separator": True },
                { "label": "Move         [m]",          "callback": self.on_menu_item_activated, "action": "mode_move" },
                { "label": "Pencil       [d]",          "callback": self.on_menu_item_activated, "action": "mode_draw" },
                { "label": "Shape        [s]",          "callback": self.on_menu_item_activated, "action": "mode_shape" },
                { "label": "Text         [t]",          "callback": self.on_menu_item_activated, "action": "mode_text" },
                { "label": "Rectangle    [r]",          "callback": self.on_menu_item_activated, "action": "mode_box" },
                { "label": "Circle       [c]",          "callback": self.on_menu_item_activated, "action": "mode_circle" },
                { "label": "Eraser       [e]",          "callback": self.on_menu_item_activated, "action": "mode_eraser" },
                { "label": "Color picker [i]",          "callback": self.on_menu_item_activated, "action": "mode_colorpicker" },
                { "separator": True },
                { "label": "Select all    (Ctrl-a)",    "callback": self.on_menu_item_activated, "action": "select_all" },
                { "label": "Paste         (Ctrl-v)",    "callback": self.on_menu_item_activated, "action": "paste_content" },
                { "label": "Clear drawing (Ctrl-l)",    "callback": self.on_menu_item_activated, "action": "clear_page" },
                { "separator": True },
                { "label": "Next Page     (Shift-n)",    "callback": self.on_menu_item_activated, "action": "next_page" },
                { "label": "Prev Page     (Shift-p)",    "callback": self.on_menu_item_activated, "action": "prev_page" },
                { "label": "Delete Page   (Shift-d)",    "callback": self.on_menu_item_activated, "action": "delete_page" },
                { "label": "Next Layer    (Ctrl-Shift-n)",    "callback": self.on_menu_item_activated, "action": "next_layer" },
                { "label": "Prev Layer    (Ctrl-Shift-p)",    "callback": self.on_menu_item_activated, "action": "prev_layer" },
                { "label": "Delete Layer  (Ctrl-Shift-d)",    "callback": self.on_menu_item_activated, "action": "delete_layer" },
                { "separator": True },
                { "label": "Bg transparency (Ctrl-b)",  "callback": self.on_menu_item_activated, "action": "cycle_bg_transparency" },
                { "label": "Toggle outline       [o]",  "callback": self.on_menu_item_activated, "action": "toggle_outline" },
                { "separator": True },
                { "label": "Color           (Ctrl-k)",  "callback": self.on_menu_item_activated, "action": "select_color" },
                { "label": "Bg Color  (Shift-Ctrl-k)",  "callback": self.on_menu_item_activated, "action": "select_color_bg" },
                { "label": "Font            (Ctrl-f)",  "callback": self.on_menu_item_activated, "action": "select_font" },
                { "separator": True },
                { "label": "Open drawing    (Ctrl-o)",  "callback": self.on_menu_item_activated, "action": "open_drawing" },
                { "label": "Image from file (Ctrl-i)",  "callback": self.on_menu_item_activated, "action": "import_image" },
                { "label": "Screenshot      (Ctrl-Shift-f)",  "callback": self.on_menu_item_activated, "action": "screenshot" },
                { "label": "Save as         (Ctrl-s)",  "callback": self.on_menu_item_activated, "action": "save_drawing_as" },
                { "label": "Export          (Ctrl-e)",  "callback": self.on_menu_item_activated, "action": "export_drawing" },
                { "label": "Help            [F1]",      "callback": self.on_menu_item_activated, "action": "show_help_dialog" },
                { "label": "Quit            (Ctrl-q)",  "callback": self.on_menu_item_activated, "action": "app_exit" },
        ]

        self.__context_menu = self.build_menu(menu_items)
        return self.__context_menu

    def object_menu(self, objects):
        """Build the object context menu"""
        # when right-clicking on an object
        menu_items = [
                { "label": "Copy (Ctrl-c)",        "callback": self.on_menu_item_activated, "action": "copy_content" },
                { "label": "Cut (Ctrl-x)",         "callback": self.on_menu_item_activated, "action": "cut_content" },
                { "separator": True },
                { "label": "Delete (|Del|)",       "callback": self.on_menu_item_activated, "action": "selection_delete" },
                { "label": "Group (g)",            "callback": self.on_menu_item_activated, "action": "selection_group" },
                { "label": "Ungroup (u)",          "callback": self.on_menu_item_activated, "action": "selection_ungroup" },
                { "label": "Export (Ctrl-e)",      "callback": self.on_menu_item_activated, "action": "export_drawing" },
                { "separator": True },
                { "label": "Move to top (Alt-Page_Up)", "callback": self.on_menu_item_activated, "action": "zmove_selection_top" },
                { "label": "Raise (Alt-Up)",       "callback": self.on_menu_item_activated, "action": "zmove_selection_up" },
                { "label": "Lower (Alt-Down)",     "callback": self.on_menu_item_activated, "action": "zmove_selection_down" },
                { "label": "Move to bottom (Alt-Page_Down)", "callback": self.on_menu_item_activated, "action": "zmove_selection_bottom" },
                { "separator": True },
                { "label": "To shape   (Alt-s)",   "callback": self.on_menu_item_activated, "action": "transmute_to_shape" },
                { "label": "To drawing (Alt-d)",   "callback": self.on_menu_item_activated, "action": "transmute_to_drawing" },
                #{ "label": "Fill       (f)",       "callback": self.on_menu_item_activated, "action": "f" },
                { "separator": True },
                { "label": "Color (Ctrl-k)",       "callback": self.on_menu_item_activated, "action": "select_color" },
                { "label": "Font (Ctrl-f)",        "callback": self.on_menu_item_activated, "action": "select_font" },
                { "label": "Help [F1]",            "callback": self.on_menu_item_activated, "action": "show_help_dialog" },
                { "label": "Quit (Ctrl-q)",        "callback": self.on_menu_item_activated, "action": "app_exit" },
        ]

        # if there is only one object, remove the group menu item
        if len(objects) == 1:
            print("only one object")
            menu_items = [m for m in menu_items if "action" not in m or "selection_group" not in m["action"]]

        group_found = [o for o in objects if o.type == "group"]
        if not group_found:
            print("no group found")
            menu_items = [m for m in menu_items if "action" not in m or "selection_ungroup" not in m["action"]]

        self.__object_menu = self.build_menu(menu_items)

        return self.__object_menu

    def on_right_mouse_click(self, ev):
        """Catch the right mouse click and display the menu"""

        event     = ev.event
        hover_obj = ev.hover()

        if hover_obj:
            ev.state.mode("move")

            # it would be better not to access gom directly
            if not self.__gom.selection().contains(hover_obj):
                self.__gom.selection().set([ hover_obj ])

            sel_objects = self.__gom.selected_objects()
            self.object_menu(sel_objects).popup(None, None,
                                                         None, None,
                                                         event.button, event.time)
        else:
            self.context_menu().popup(None, None,
                                               None, None,
                                               event.button, event.time)
        self.__bus.emit("queue_draw")
        return True
