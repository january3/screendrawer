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
import logging                                                   # <remove>
log = logging.getLogger(__name__)                                # <remove>

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

    def __init__(self, bus, state):
        # singleton pattern
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self.__state = state
            self.__bus = bus
            self.make_actions_dictionary(bus)
            self.make_default_keybindings()
            bus.on("key_press_event", self.process_key_event)

    def dispatch_action(self, action_name, **kwargs):
        """
        Dispatches an action by name.
        """
        #print("dispatch_action", action_name)
        if not action_name in self.__actions:
            log.warning(f"action {action_name} not found in actions")
            return

        action = self.__actions[action_name].get('action') or self.__bus.emit

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
            log.warning(f"Exception type: {exc_type}")
            log.warning(f"Exception value:{exc_value}")
            log.warning("Traceback:")
            traceback.print_tb(exc_traceback)
            log.warning(f"Error while dispatching action {action_name}: {e}")

    def dispatch_key_event(self, key_event, mode):
        """
        Dispatches an action by key event.
        """
        #print("dispatch_key_event", key_event, mode)

        if not key_event in self.__keybindings:
            return

        action_name = self.__keybindings[key_event]

        if not action_name in self.__actions:
            log.warning(f"action {action_name} not found in actions")
            return

        # check whether action allowed in the current mode
        if 'modes' in self.__actions[action_name]:
            if not mode in self.__actions[action_name]['modes']:
                log.warning("action not allowed in this mode")
                return

        log.debug(f"keyevent {key_event} dispatching action {action_name}")
        self.dispatch_action(action_name)

    def on_key_press(self, widget, event):
        """
        This method is called when a key is pressed.
        """

        log.debug(f"key pressed {event.keyval} state {event.state}")
        self.__bus.emit("key_press_event", True, event)

    def process_key_event(self, event):
        """Process the key event and send message to the bus"""
        state = self.__state

        keyname = Gdk.keyval_name(event.keyval)
        char    = chr(Gdk.keyval_to_unicode(event.keyval))
        ctrl    = event.state & Gdk.ModifierType.CONTROL_MASK
        shift   = event.state & Gdk.ModifierType.SHIFT_MASK
        alt_l   = event.state & Gdk.ModifierType.MOD1_MASK
        log.debug(f"keyname {keyname} char {char} ctrl {ctrl} shift {shift} alt_l {alt_l}")

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

        if cobj and cobj.type == "text" and not keyname == "Escape":
            log.debug(f"updating text input, keyname={keyname} char={char} keyfull={keyfull} mode={mode}")
            cobj.update_by_key(keyfull, char)
            state.queue_draw()
            return

        # otherwise, we dispatch the key event
        self.dispatch_key_event(keyfull, mode)

        # XXX this probably should be somewhere else
        state.queue_draw()


    def make_actions_dictionary(self, bus):
        """
        This dictionary maps key events to actions.
        """
        self.__actions = {
            'mode_draw':             {'args': [ "mode_set", False, "draw"]},
            'mode_rectangle':        {'args': [ "mode_set", False, "rectangle"]},
            'mode_circle':           {'args': [ "mode_set", False, "circle"]},
            'mode_move':             {'args': [ "mode_set", False, "move"]},
            'mode_text':             {'args': [ "mode_set", False, "text"]},
            'mode_select':           {'args': [ "mode_set", False, "select"]},
            'mode_eraser':           {'args': [ "mode_set", False, "eraser"]},
            'mode_shape':            {'args': [ "mode_set", False, "shape"]},
            'mode_segment':          {'args': [ "mode_set", False, "segment"]},
            'mode_colorpicker':      {'args': [ "mode_set", False, "colorpicker"]},

            'escape':                {'args': ["escape"]},
            'toggle_grouping':       {'args': ["toggle_grouping"]},
            'toggle_outline':        {'args': ["toggle_outline"]},

            'cycle_bg_transparency': {'args': ["cycle_bg_transparency"]},
            'toggle_wiglets':        {'args': ["toggle_wiglets"]},
            'toggle_grid':           {'args': ["toggle_grid"]},
            'switch_pens':           {'args': ["switch_pens"]},
            'apply_pen_to_bg':       {'args': ["apply_pen_to_bg"], 'modes': ["move"]},

            'clear_page':            {'args': ["clear_page"]},

            # dialogs and app events
            'app_exit':              {'args': ["app_exit"]},
            'show_help_dialog':      {'args': ["show_help_dialog"]},
            "export_drawing":        {'args': ["export_drawing"]},
            "save_drawing_as":       {'args': ["save_drawing_as"]},
            "select_color":          {'args': ["select_color"]},
            "select_color_bg":       {'args': ["select_color_bg"]},
            "select_font":           {'args': ["select_font"]},
            "import_image":          {'args': ["import_image"]},
            "open_drawing":          {'args': ["open_drawing"]},

            'copy_content':          {'args': ["copy_content"]},
            'cut_content':           {'args': ["cut_content"]},
            'paste_content':         {'args': ["paste_content"]},
            'screenshot':            {'args': ["screenshot"]},

            'selection_fill':        {'args': [ "selection_fill" ], 'modes': ["move"]},

            'transmute_to_shape':    {'args': [ "transmute_selection", True, "shape" ]},
            'transmute_to_draw':     {'args': [ "transmute_selection", True, "draw" ]},

            'move_up_10':            {'args': [ "move_selection", True, 0, -10],   'modes': ["move"]},
            'move_up_1':             {'args': [ "move_selection", True, 0, -1],    'modes': ["move"]},
            'move_up_100':           {'args': [ "move_selection", True, 0, -100],  'modes': ["move"]},
            'move_down_10':          {'args': [ "move_selection", True, 0, 10],    'modes': ["move"]},
            'move_down_1':           {'args': [ "move_selection", True, 0, 1],     'modes': ["move"]},
            'move_down_100':         {'args': [ "move_selection", True, 0, 100],   'modes': ["move"]},
            'move_left_10':          {'args': [ "move_selection", True, -10, 0],   'modes': ["move"]},
            'move_left_1':           {'args': [ "move_selection", True, -1, 0],    'modes': ["move"]},
            'move_left_100':         {'args': [ "move_selection", True, -100, 0],  'modes': ["move"]},
            'move_right_10':         {'args': [ "move_selection", True, 10, 0],    'modes': ["move"]},
            'move_right_1':          {'args': [ "move_selection", True, 1, 0],     'modes': ["move"]},
            'move_right_100':        {'args': [ "move_selection", True, 100, 0],   'modes': ["move"]},

            'rotate_selection_ccw_10': {'args': [ "rotate_selection", True, 10],  'modes': ["move"]},
            'rotate_selection_ccw_1':  {'args': [ "rotate_selection", True, 1],   'modes': ["move"]},
            'rotate_selection_ccw_90': {'args': [ "rotate_selection", True, 90],  'modes': ["move"]},
            'rotate_selection_cw_10':  {'args': [ "rotate_selection", True, -10], 'modes': ["move"]},
            'rotate_selection_cw_1':   {'args': [ "rotate_selection", True, -1],  'modes': ["move"]},
            'rotate_selection_cw_90':  {'args': [ "rotate_selection", True, -90], 'modes': ["move"]},

            'zmove_selection_top':     {'args': [ "selection_zmove", True, "top" ],    'modes': ["move"]},
            'zmove_selection_bottom':  {'args': [ "selection_zmove", True, "bottom" ], 'modes': ["move"]},
            'zmove_selection_raise':   {'args': [ "selection_zmove", True, "raise" ],  'modes': ["move"]},
            'zmove_selection_lower':   {'args': [ "selection_zmove", True, "lower" ],  'modes': ["move"]},

            'set_color_white':       {'args': [ "set_color", True, COLORS["white"]]},
            'set_color_black':       {'args': [ "set_color", True, COLORS["black"]]},
            'set_color_red':         {'args': [ "set_color", True, COLORS["red"]]},
            'set_color_green':       {'args': [ "set_color", True, COLORS["green"]]},
            'set_color_blue':        {'args': [ "set_color", True, COLORS["blue"]]},
            'set_color_yellow':      {'args': [ "set_color", True, COLORS["yellow"]]},
            'set_color_cyan':        {'args': [ "set_color", True, COLORS["cyan"]]},
            'set_color_magenta':     {'args': [ "set_color", True, COLORS["magenta"]]},
            'set_color_purple':      {'args': [ "set_color", True, COLORS["purple"]]},
            'set_color_grey':        {'args': [ "set_color", True, COLORS["grey"]]},

            'set_brush_rounded':     {'args': [ "set_brush", True, "rounded"] },
            'set_brush_marker':      {'args': [ "set_brush", True, "marker"] },
            'set_brush_slanted':     {'args': [ "set_brush", True, "slanted"] },
            'set_brush_pencil':      {'args': [ "set_brush", True, "pencil"] },
            'set_brush_tapered':     {'args': [ "set_brush", True, "tapered"] },

            'stroke_increase':       {'args': [ "stroke_change", True, 1]},
            'stroke_decrease':       {'args': [ "stroke_change", True, -1]},

            # selections and moving objects
            'select_next_object':     {'args': [ "set_selection", True, "next_object" ],     'modes': ["move"]},
            'select_previous_object': {'args': [ "set_selection", True, "previous_object" ], 'modes': ["move"]},
            'select_all':             {'args': [ "set_selection", True, "all" ]},
            'select_reverse':         {'args': [ "set_selection", True, "reverse" ]},

            'selection_clip':         {'args': [ "selection_clip"   ], 'modes': ["move"]},
            'selection_unclip':       {'args': [ "selection_unclip" ], 'modes': ["move"]},
            'selection_group':        {'args': [ "selection_group"  ], 'modes': ["move"]},
            'selection_ungroup':      {'args': [ "selection_ungroup"], 'modes': ["move"]},
            'selection_delete':       {'args': [ "selection_delete" ], 'modes': ["move"]},
            'redo':                   {'args': [ "history_redo" ]},
            'undo':                   {'args': [ "history_undo" ]},
            'next_page':              {'args': [ "next_page" ]},
            'prev_page':              {'args': [ "prev_page" ]},
            'insert_page':            {'args': [ "insert_page" ]},
            'delete_page':            {'args': [ "delete_page" ]},
            'next_layer':             {'args': [ "next_layer" ]},
            'prev_layer':             {'args': [ "prev_layer" ]},
            'delete_layer':           {'args': [ "delete_layer" ]},
            'apply_pen_to_selection': {'args': [ "selection_apply_pen" ],    'modes': ["move"]},
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
            'Shift-s':              "mode_segment",
            'i':                    "mode_colorpicker",
            'space':                "mode_move",

            'h':                    "show_help_dialog",
            'F1':                   "show_help_dialog",
            'question':             "show_help_dialog",
            'Shift-question':       "show_help_dialog",
            'Escape':               "escape",
            'Ctrl-l':               "clear_page",
            'Ctrl-b':               "cycle_bg_transparency",
            'x':                    "app_exit",
            'q':                    "app_exit",
            'Ctrl-q':               "app_exit",
            'l':                    "clear_page",
            'o':                    "toggle_outline",
            'w':                    "toggle_wiglets",
            'Ctrl-g':               "toggle_grid",
            'Alt-s':                "transmute_to_shape",
            'Alt-d':                "transmute_to_draw",
            'f':                    "selection_fill",

            'Ctrl-Shift-g':         "toggle_grouping",

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
            'Ctrl-p':               "switch_pens",
            'Ctrl-o':               "open_drawing",


            'Tab':                  "select_next_object",
            'Shift-ISO_Left_Tab':   "select_previous_object",
            'g':                    "selection_group",
            'u':                    "selection_ungroup",
            'Shift-c':              "selection_clip",
            'Shift-u':              "selection_unclip",
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
