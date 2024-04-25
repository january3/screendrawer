"""
EM stands for EventManager. The EventManager class is a singleton that
manages the events and actions of the app. The actions are defined in the
make_actions_dictionary method.

The design of the app is as follows: the EventManager class is a singleton
that manages the events and actions of the app. The actions are defined in
the actions_dictionary method. 

So the EM is a know-it-all class, and the others (GOM, App) are just
listeners to the EM. The EM is the one that knows what to do when an event
happens.
"""

import traceback # <remove>
from sys import exc_info # <remove>
import gi                                                  # <remove>
gi.require_version('Gtk', '3.0')                           # <remove>
from gi.repository import Gdk # <remove>

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


class EventManager:
    """
    The EventManager class is a singleton that manages the events and actions
    of the app. The actions are defined in the make_actions_dictionary method.
    """
    # singleton pattern
    _instance = None

    def __new__(cls, *args, **kwargs):
        # singleton pattern
        if not cls._instance:
            cls._instance = super(EventManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, bus, gom, app, state, setter):
        # singleton pattern
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self.__state = state
            self.__setter = setter
            self.__bus = bus
            self.make_actions_dictionary(bus, gom, app, state, setter)
            self.make_default_keybindings()

    def dispatch_action(self, action_name, **kwargs):
        """
        Dispatches an action by name.
        """
        #print("dispatch_action", action_name)
        if not action_name in self.__actions:
            print(f"action {action_name} not found in actions")
            return

        action = self.__actions[action_name]['action']
        if not callable(action):
            raise ValueError(f"Action is not callable: {action_name}")
        try:
            if 'args' in self.__actions[action_name]:
                args = self.__actions[action_name]['args']
                action(*args)
            else:
                action(**kwargs)
        except Exception as e:
            exc_type, exc_value, exc_traceback = exc_info()
            print("Exception type: ", exc_type)
            print("Exception value:", exc_value)
            print("Traceback:")
            traceback.print_tb(exc_traceback)
            print(f"Error while dispatching action {action_name}: {e}")

    def dispatch_key_event(self, key_event, mode):
        """
        Dispatches an action by key event.
        """
        #print("dispatch_key_event", key_event, mode)

        if not key_event in self.__keybindings:
            print(f"key_event {key_event} not found in keybindings")
            return

        action_name = self.__keybindings[key_event]

        if not action_name in self.__actions:
            print(f"action {action_name} not found in actions")
            return

        # check whether action allowed in the current mode
        if 'modes' in self.__actions[action_name]:
            if not mode in self.__actions[action_name]['modes']:
                print("action not allowed in this mode")
                return

        print("keyevent", key_event, "dispatching action", action_name)
        self.dispatch_action(action_name)

    def on_key_press(self, widget, event):
        """
        This method is called when a key is pressed.
        """

        state = self.__state

        keyname = Gdk.keyval_name(event.keyval)
        char    = chr(Gdk.keyval_to_unicode(event.keyval))
        ctrl    = event.state & Gdk.ModifierType.CONTROL_MASK
        shift   = event.state & Gdk.ModifierType.SHIFT_MASK
        alt_l   = event.state & Gdk.ModifierType.MOD1_MASK
        print("keyname", keyname, "char", char, "ctrl", ctrl, "shift", shift, "alt_l", alt_l)

        mode = state.mode()

        keyfull = keyname

        if char.isupper():
            keyfull = keyname.lower()
        if shift:
            keyfull = "Shift-" + keyfull
        if ctrl:
            keyfull = "Ctrl-" + keyfull
        if alt_l:
            keyfull = "Alt-" + keyfull
        #print("keyfull", keyfull)

        # first, check whether there is a current object being worked on
        # and whether this object is a text object. In that case, we only
        # call the ctrl keybindings and pass the rest to the text object.
        cobj = state.current_obj()

        if cobj and cobj.type == "text" and not(ctrl or keyname == "Escape"):
            print("updating text input")
            cobj.update_by_key(keyname, char)
            state.queue_draw()
            return

        # otherwise, we dispatch the key event
        self.dispatch_key_event(keyfull, mode)

        # XXX this probably should be somewhere else
        state.queue_draw()


    def make_actions_dictionary(self, bus, gom, app, state, setter):
        """
        This dictionary maps key events to actions.
        """
        self.__actions = {
            'mode_draw':             {'action': state.mode, 'args': ["draw"]},
            'mode_rectangle':              {'action': state.mode, 'args': ["rectangle"]},
            'mode_circle':           {'action': state.mode, 'args': ["circle"]},
            'mode_move':             {'action': state.mode, 'args': ["move"]},
            'mode_text':             {'action': state.mode, 'args': ["text"]},
            'mode_select':           {'action': state.mode, 'args': ["select"]},
            'mode_eraser':           {'action': state.mode, 'args': ["eraser"]},
            'mode_shape':            {'action': state.mode, 'args': ["shape"]},
            'mode_colorpicker':      {'action': state.mode, 'args': ["colorpicker"]},

            'finish_text_input':     {'action': bus.emit, 'args': ["finish_text_input"]},

            'cycle_bg_transparency': {'action': state.cycle_background},
            'toggle_outline':        {'action': state.outline_toggle},

            'clear_page':            {'action': setter.clear},
            'toggle_wiglets':        {'action': state.toggle_wiglets},
            'toggle_grid':           {'action': state.toggle_grid},

            'show_help_dialog':      {'action': app.show_help_dialog},
            'app_exit':              {'action': app.exit},

            'selection_fill':        {'action': gom.selection_fill},
            'transmute_to_shape':    {'action': gom.transmute_selection, 'args': [ "shape" ]},
            'transmute_to_draw':     {'action': gom.transmute_selection, 'args': [ "draw" ]},
            'move_up_10':            {'action': gom.move_selection, 'args': [0, -10],   'modes': ["move"]},
            'move_up_1':             {'action': gom.move_selection, 'args': [0, -1],    'modes': ["move"]},
            'move_up_100':           {'action': gom.move_selection, 'args': [0, -100],  'modes': ["move"]},
            'move_down_10':          {'action': gom.move_selection, 'args': [0, 10],    'modes': ["move"]},
            'move_down_1':           {'action': gom.move_selection, 'args': [0, 1],     'modes': ["move"]},
            'move_down_100':         {'action': gom.move_selection, 'args': [0, 100],   'modes': ["move"]},
            'move_left_10':          {'action': gom.move_selection, 'args': [-10, 0],   'modes': ["move"]},
            'move_left_1':           {'action': gom.move_selection, 'args': [-1, 0],    'modes': ["move"]},
            'move_left_100':         {'action': gom.move_selection, 'args': [-100, 0],  'modes': ["move"]},
            'move_right_10':         {'action': gom.move_selection, 'args': [10, 0],    'modes': ["move"]},
            'move_right_1':          {'action': gom.move_selection, 'args': [1, 0],     'modes': ["move"]},
            'move_right_100':        {'action': gom.move_selection, 'args': [100, 0],   'modes': ["move"]},

            # XXX something is rotten here
            #'f':                    {'action': self.gom.selection_fill, 'modes': ["box", "circle", "draw", "move"]},

            'rotate_selection_ccw_10': {'action': gom.rotate_selection, 'args': [10],  'modes': ["move"]},
            'rotate_selection_ccw_1':  {'action': gom.rotate_selection, 'args': [1],   'modes': ["move"]},
            'rotate_selection_ccw_90': {'action': gom.rotate_selection, 'args': [90],  'modes': ["move"]},
            'rotate_selection_cw_10':  {'action': gom.rotate_selection, 'args': [-10], 'modes': ["move"]},
            'rotate_selection_cw_1':   {'action': gom.rotate_selection, 'args': [-1],  'modes': ["move"]},
            'rotate_selection_cw_90':  {'action': gom.rotate_selection, 'args': [-90], 'modes': ["move"]},

            'zmove_selection_top':    {'action': gom.selection_zmove, 'args': [ "top" ],    'modes': ["move"]},
            'zmove_selection_bottom': {'action': gom.selection_zmove, 'args': [ "bottom" ], 'modes': ["move"]},
            'zmove_selection_raise':  {'action': gom.selection_zmove, 'args': [ "raise" ],  'modes': ["move"]},
            'zmove_selection_lower':  {'action': gom.selection_zmove, 'args': [ "lower" ],  'modes': ["move"]},

            'set_color_white':       {'action': setter.set_color, 'args': [COLORS["white"]]},
            'set_color_black':       {'action': setter.set_color, 'args': [COLORS["black"]]},
            'set_color_red':         {'action': setter.set_color, 'args': [COLORS["red"]]},
            'set_color_green':       {'action': setter.set_color, 'args': [COLORS["green"]]},
            'set_color_blue':        {'action': setter.set_color, 'args': [COLORS["blue"]]},
            'set_color_yellow':      {'action': setter.set_color, 'args': [COLORS["yellow"]]},
            'set_color_cyan':        {'action': setter.set_color, 'args': [COLORS["cyan"]]},
            'set_color_magenta':     {'action': setter.set_color, 'args': [COLORS["magenta"]]},
            'set_color_purple':      {'action': setter.set_color, 'args': [COLORS["purple"]]},
            'set_color_grey':        {'action': setter.set_color, 'args': [COLORS["grey"]]},

            'set_brush_rounded':     {'action': setter.set_brush, 'args': ["rounded"] },
            'set_brush_marker':      {'action': setter.set_brush, 'args': ["marker"] },
            'set_brush_slanted':     {'action': setter.set_brush, 'args': ["slanted"] },
            'set_brush_pencil':      {'action': setter.set_brush, 'args': ["pencil"] },
            'set_brush_tapered':     {'action': setter.set_brush, 'args': ["tapered"] },

            'apply_pen_to_bg':       {'action': state.apply_pen_to_bg,        'modes': ["move"]},
            'toggle_pens':           {'action': state.switch_pens},

            # dialogs
            "export_drawing":        {'action': app.export_drawing},
            "save_drawing_as":       {'action': app.save_drawing_as},
            "select_color":          {'action': app.select_color},
            "select_color_bg":       {'action': app.select_color_bg},
            "select_font":           {'action': app.select_font},
            "import_image":          {'action': app.select_image_and_create_pixbuf},
            "open_drawing":          {'action': app.open_drawing},

            # selections and moving objects
            'select_next_object':     {'action': gom.select_next_object,     'modes': ["move"]},
            'select_previous_object': {'action': gom.select_previous_object, 'modes': ["move"]},
            'select_all':             {'action': gom.select_all},
            'select_reverse':         {'action': gom.select_reverse},
            'selection_group':        {'action': gom.selection_group,   'modes': ["move"]},
            'selection_ungroup':      {'action': gom.selection_ungroup, 'modes': ["move"]},
            'selection_delete':       {'action': gom.selection_delete,  'modes': ["move"]},
            'redo':                   {'action': gom.redo},
            'undo':                   {'action': gom.undo},

            'next_page':              {'action': gom.next_page},
            'prev_page':              {'action': gom.prev_page},
            'insert_page':            {'action': gom.insert_page},
            'delete_page':            {'action': gom.delete_page},

            'next_layer':             {'action': gom.next_layer},
            'prev_layer':             {'action': gom.prev_layer},
            'delete_layer':           {'action': gom.delete_layer},

            'apply_pen_to_selection': {'action': gom.selection_apply_pen,    'modes': ["move"]},

            'copy_content':           {'action': app.copy_content,        },
            'cut_content':            {'action': app.cut_content,         'modes': ["move"]},
            'paste_content':          {'action': app.paste_content},
            'screenshot':             {'action': app.screenshot},

            'stroke_increase':        {'action': setter.stroke_change, 'args': [1]},
            'stroke_decrease':        {'action': setter.stroke_change, 'args': [-1]},
        }

    def make_default_keybindings(self):
        """
        This dictionary maps key events to actions.
        """
        self.__keybindings = {
            'm':                    "mode_move",
            'r':                    "mode_rectangle",
            'c':                    "mode_circle",
            'd':                    "mode_draw",
            't':                    "mode_text",
            'e':                    "mode_eraser",
            's':                    "mode_shape",
            'i':                    "mode_colorpicker",
            'space':                "mode_move",

            'h':                    "show_help_dialog",
            'F1':                   "show_help_dialog",
            'question':             "show_help_dialog",
            'Shift-question':       "show_help_dialog",
            'Escape':               "finish_text_input",
            'Ctrl-l':               "clear_page",
            'Ctrl-b':               "cycle_bg_transparency",
            'x':                    "app_exit",
            'q':                    "app_exit",
            'Ctrl-q':               "app_exit",
            'l':                    "clear_page",
            'o':                    "toggle_outline",
            'w':                    "toggle_wiglets",
            'Ctrl-g':              "toggle_grid",
            'Alt-s':                "transmute_to_shape",
            'Alt-d':                "transmute_to_draw",
            'f':                    "selection_fill",

            'Up':                   "move_up_10",
            'Shift-Up':             "move_up_1",
            'Ctrl-Up':              "move_up_100",
            'Down':                 "move_down_10",
            'Shift-Down':           "move_down_1",
            'Ctrl-Down':            "move_down_100",
            'Left':                 "move_left_10",
            'Shift-Left':           "move_left_1",
            'Ctrl-Left':            "move_left_100",
            'Right':                "move_right_10",
            'Shift-Right':          "move_right_1",
            'Ctrl-Right':           "move_right_100",
            'Page_Up':              "rotate_selection_ccw_10",
            'Shift-Page_Up':        "rotate_selection_ccw_1",
            'Ctrl-Page_Up':         "rotate_selection_ccw_90",
            'Page_Down':            "rotate_selection_cw_10",
            'Shift-Page_Down':      "rotate_selection_cw_1",
            'Ctrl-Page_Down':       "rotate_selection_cw_90",

            'Alt-Page_Up':          "zmove_selection_top",
            'Alt-Page_Down':        "zmove_selection_bottom",
            'Alt-Up':               "zmove_selection_raise",
            'Alt-Down':             "zmove_selection_lower",

            'Shift-w':              "set_color_white",
            'Shift-b':              "set_color_black",
            'Shift-r':              "set_color_red",
            'Shift-g':              "set_color_green",
            'Shift-l':              "set_color_blue",
            'Shift-e':              "set_color_grey",
            'Shift-y':              "set_color_yellow",
#            'Shift-p':              "set_color_purple",

            '1':                    "set_brush_rounded",
            '2':                    "set_brush_marker",
            '3':                    "set_brush_slanted",
            '4':                    "set_brush_pencil",
            '5':                    "set_brush_tapered",

            'Shift-p':              "prev_page",
            'Shift-n':              "next_page",
            'Shift-i':              "insert_page",
            'Shift-d':              "delete_page",
            'Ctrl-Shift-p':         "prev_layer",
            'Ctrl-Shift-n':         "next_layer",
            'Ctrl-Shift-d':         "delete_layer",

            'Ctrl-e':               "export_drawing",
            'Ctrl-Shift-s':         "save_drawing_as",
            'Ctrl-k':               "select_color",
            'Ctrl-Shift-k':         "select_color_bg",
            'Ctrl-f':               "select_font",
            'Ctrl-i':               "import_image",
            'Ctrl-p':               "toggle_pens",
            'Ctrl-o':               "open_drawing",


            'Tab':                  "select_next_object",
            'Shift-ISO_Left_Tab':   "select_previous_object",
            'g':                    "selection_group",
            'u':                    "selection_ungroup",
            'Delete':               "selection_delete",

            'Alt-p':                "apply_pen_to_selection",
            'Alt-Shift-p':          "apply_pen_to_bg",

            'Ctrl-a':               "select_all",
            'Ctrl-r':               "select_reverse",
            'Ctrl-y':               "redo",
            'Ctrl-z':               "undo",
            'Ctrl-c':               "copy_content",
            'Ctrl-x':               "cut_content",
            'Ctrl-v':               "paste_content",
            'Ctrl-Shift-f':         "screenshot",
            'Ctrl-plus':            "stroke_increase",
            'Ctrl-minus':           "stroke_decrease",
        }
