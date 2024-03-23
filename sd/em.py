# the design of the app is as follows: the EventManager class is a singleton
# that manages the events and actions of the app. The actions are defined in
# the actions_dictionary method. 
#
# So the EM is a know-it-all class, and the others (GOM, App) are just
# listeners to the EM. The EM is the one that knows what to do when an event
# happens.

import traceback # <remove>
from sys import exc_info # <remove>
from gi.repository import Gdk, Gtk # <remove>

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

    def __init__(self, gom, app):
        # singleton pattern
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self.__gom = gom
            self.__app = app
            self.make_actions_dictionary(gom, app)
            self.make_default_keybindings()

    def dispatch_action(self, action_name, **kwargs):
        """
        Dispatches an action by name.
        """
        print("dispatch_action", action_name)
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
        print("dispatch_key_event", key_event, mode)

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

        print("dispatching action", action_name)
        self.dispatch_action(action_name)

    def on_key_press(self, widget, event):
        """
        This method is called when a key is pressed.
        """

        gom, app = self.__gom, self.__app

        keyname = Gdk.keyval_name(event.keyval)
        char    = chr(Gdk.keyval_to_unicode(event.keyval))
        ctrl    = event.state & Gdk.ModifierType.CONTROL_MASK
        shift   = event.state & Gdk.ModifierType.SHIFT_MASK
        alt_l   = event.state & Gdk.ModifierType.MOD1_MASK
        print("keyname", keyname, "char", char, "ctrl", ctrl, "shift", shift, "alt_l", alt_l)

        mode = app.get_mode()

        keyfull = keyname

        if char.isupper():
            keyfull = keyname.lower()
        if shift:
            keyfull = "Shift-" + keyfull
        if ctrl:
            keyfull = "Ctrl-" + keyfull
        if alt_l:
            keyfull = "Alt-" + keyfull
        print("keyfull", keyfull)

        # first, check whether there is a current object being worked on
        # and whether this object is a text object. In that case, we only
        # call the ctrl keybindings and pass the rest to the text object.
        if app.current_object and app.current_object.type == "text" and not(ctrl or keyname == "Escape"):
            print("updating text input")
            app.current_object.update_by_key(keyname, char)
            app.queue_draw()
            return

        # otherwise, we dispatch the key event
        self.dispatch_key_event(keyfull, mode)

        # XXX this probably should be somewhere else
        app.queue_draw()


    def make_actions_dictionary(self, gom, app):
        """
        This dictionary maps key events to actions.
        """
        self.__actions = {
            'mode_draw':             {'action': app.set_mode, 'args': ["draw"]},
            'mode_box':              {'action': app.set_mode, 'args': ["box"]},
            'mode_circle':           {'action': app.set_mode, 'args': ["circle"]},
            'mode_move':             {'action': app.set_mode, 'args': ["move"]},
            'mode_text':             {'action': app.set_mode, 'args': ["text"]},
            'mode_select':           {'action': app.set_mode, 'args': ["select"]},
            'mode_eraser':           {'action': app.set_mode, 'args': ["eraser"]},
            'mode_shape':            {'action': app.set_mode, 'args': ["shape"]},
            'mode_colorpicker':      {'action': app.set_mode, 'args': ["colorpicker"]},

            'finish_text_input':     {'action': app.finish_text_input},

            'show_help_dialog':      {'action': app.show_help_dialog},
            'app_exit':              {'action': app.exit},

            'clear_page':            {'action': app.clear},
            'cycle_bg_transparency': {'action': app.cycle_background},
            'toggle_outline':        {'action': app.outline_toggle},

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

            'set_color_white':       {'action': app.set_color, 'args': [COLORS["white"]]},
            'set_color_black':       {'action': app.set_color, 'args': [COLORS["black"]]},
            'set_color_red':         {'action': app.set_color, 'args': [COLORS["red"]]},
            'set_color_green':       {'action': app.set_color, 'args': [COLORS["green"]]},
            'set_color_blue':        {'action': app.set_color, 'args': [COLORS["blue"]]},
            'set_color_yellow':      {'action': app.set_color, 'args': [COLORS["yellow"]]},
            'set_color_cyan':        {'action': app.set_color, 'args': [COLORS["cyan"]]},
            'set_color_magenta':     {'action': app.set_color, 'args': [COLORS["magenta"]]},
            'set_color_purple':      {'action': app.set_color, 'args': [COLORS["purple"]]},
            'set_color_grey':        {'action': app.set_color, 'args': [COLORS["grey"]]},

            # dialogs
            "export_drawing":        {'action': app.export_drawing},
            "save_drawing_as":       {'action': app.save_drawing_as},
            "select_color":          {'action': app.select_color},
            "select_font":           {'action': app.select_font},
            "import_image":          {'action': app.select_image_and_create_pixbuf},
            "toggle_pens":           {'action': app.switch_pens},
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

#            'Ctrl-m':               {'action': self.smoothen,           'modes': ["move"]},
            'copy_content':          {'action': app.copy_content,        'modes': ["move"]},
            'cut_content':           {'action': app.cut_content,         'modes': ["move"]},
            'paste_content':         {'action': app.paste_content},
            'screenshot':            {'action': app.screenshot},

            'stroke_increase':       {'action': app.stroke_increase},
            'stroke_decrease':       {'action': app.stroke_decrease},
        }

    def get_keybindings(self):
        """
        Returns the keybindings dictionary.
        """
        return self.__keybindings

    def make_default_keybindings(self):
        """
        This dictionary maps key events to actions.
        """
        self.__keybindings = {
            'm':                    "mode_move",
            'b':                    "mode_box",
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
            'Shift-p':              "set_color_purple",

            'Ctrl-e':               "export_drawing",
            'Ctrl-s':               "save_drawing_as",
            'Ctrl-k':               "select_color",
            'Ctrl-f':               "select_font",
            'Ctrl-i':               "import_image",
            'Ctrl-p':               "toggle_pens",
            'Ctrl-o':               "open_drawing",

            'Tab':                  "select_next_object",
            'Shift-ISO_Left_Tab':   "select_previous_object",
            'g':                    "selection_group",
            'u':                    "selection_ungroup",
            'Delete':               "selection_delete",

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


