"""
This module contains the MouseEvent class.
It is used to handle mouse events in the drawing area.
"""

import logging                                                         # <remove>
import gi                                                              # <remove>
gi.require_version('Gtk', '3.0')                                       # <remove> pylint: disable=wrong-import-position
gi.require_version('Gdk', '3.0')                                       # <remove> pylint: disable=wrong-import-position
from gi.repository import Gdk                                          # <remove>
from .utils import find_obj_close_to_click, find_corners_next_to_click # <remove>

log = logging.getLogger(__name__)                                      # <remove>
log.setLevel(logging.INFO)                                             # <remove>

## ---------------------------------------------------------------------
class MouseEvent:
    """
    Simple class for handling mouse events.

    Takes the event and computes a number of useful things.

    One advantage of using it: the computationaly intensive stuff is
    computed only once and only if it is needed.
    """
    def __init__(self, event, state):
        self.event = event
        self.state = state

        self.x_abs, self.y_abs = event.x, event.y

        self.x, self.y = state.pos_abs_to_rel((event.x, event.y))


        self.__info = {
                "mode": state.mode(),
                "shift": (event.state & Gdk.ModifierType.SHIFT_MASK) != 0,
                "ctrl": (event.state & Gdk.ModifierType.CONTROL_MASK) != 0,
                "alt": (event.state & Gdk.ModifierType.MOD1_MASK) != 0,
                "double":   (event.type == Gdk.EventType.DOUBLE_BUTTON_PRESS),
                "pressure": event.get_axis(Gdk.AxisUse.PRESSURE),
                "hover":    None,
                "corner":   [ ], 
                "pos": (self.x, self.y),
                "pos_abs": (self.x_abs, self.y_abs)
                }

        if self.__info["pressure"] is None:  # note that 0 is perfectly valid
            self.__info["pressure"] = 1


    def hover(self):
        """Return the object that is hovered by the mouse."""
        objects = self.state.objects()
        pos = self.__info["pos"]
        if not self.__info.get("hover"):
            self.__info["hover"] = find_obj_close_to_click(pos[0],
                                                   pos[1],
                                                   objects, 20)
        return self.__info["hover"]

    def corner(self):
        """Return the corner that is hovered by the mouse."""
        objects = self.state.objects()
        pos = self.__info["pos"]
        if not self.__info.get("corner"):
            self.__info["corner"] = find_corners_next_to_click(pos[0],
                                                       pos[1],
                                                       objects, 20)
        return self.__info["corner"][0], self.__info["corner"][1]

    def pos_abs(self):
        """Return the position of the mouse."""
        return self.__info["pos_abs"]

    def pos(self):
        """Return the position of the mouse."""
        return self.__info["pos"]

    def shift(self):
        """Return True if the shift key is pressed."""
        return self.__info.get("shift")

    def ctrl(self):
        """Return True if the control key is pressed."""
        return self.__info.get("ctrl")

    def alt(self):
        """Return True if the alt key is pressed."""
        return self.__info.get("alt")

    def double(self):
        """Return True if the event is a double click."""
        return self.__info.get("double")

    def pressure(self):
        """Return the pressure of the pen."""
        return self.__info.get("pressure")

    def mode(self):
        """Return the mode in which the event happened."""
        return self.__info.get("mode")


class MouseCatcher:
    """
    Class that catches mouse events, creates the MouseEvent and sends it
    back to the bus.

    """
    def __init__(self, bus, state):
        self.__bus = bus
        self.__state = state

        # objects that indicate the state of the drawing area
        # drawing parameters

    # ---------------------------------------------------------------------
    #                              Event handlers

    def on_pan(self, gesture, direction, offset):
        """Handle panning events."""

    def on_zoom(self, gesture, scale):
        """Handle zoom events."""

    # ---------------------------------------------------------------------

    # Button press event handlers -------------------------------------------
    def on_button_press(self, widget, event): # pylint: disable=unused-argument
        """Handle mouse button press events."""
        log.debug("type:{event.type} button:{event.button} state:%s", event.state)
        self.__state.graphics().modified(True)
        ev = MouseEvent(event, state = self.__state)

        if event.button == 3:
            if self.__handle_button_3(ev):
                return True

        elif event.button == 1:
            if self.__handle_button_1(event, ev):
                return True

        return True

    def __handle_button_3(self, ev):
        """Handle right click events, unless shift is pressed."""
        if self.__bus.emit("right_mouse_click", True, ev):
            return True

        return False

    def __handle_button_1(self, event, ev):
        """Handle left click events."""

        if ev.double():
            log.debug("dblclick (%d, %d) raw (%d, %d)",
                      int(ev.x), int(ev.y), int(event.x), int(event.y))
            self.__bus.emit("cancel_left_mouse_single_click", True, ev)
            self.__bus.emit("left_mouse_double_click", True, ev)
            return True

        log.debug("sngle clck (%d, %d) raw (%d, %d)",
                  int(ev.x), int(ev.y), int(event.x), int(event.y))

        if self.__bus.emit("left_mouse_click", True, ev):
            log.debug("bus event caught the click")
            self.__bus.emit("queue_draw")

        return False

    def on_button_release(self, widget, event): # pylint: disable=unused-argument
        """Handle mouse button release events."""
        log.debug("button release: type:%s button:%s state:%s",
                  event.type, event.button, event.state)
        ev = MouseEvent(event, state = self.__state)

        if self.__bus.emit("mouse_release", True, ev):
            self.__bus.emit("queue_draw")
            return True

        return True

    # ---------------------------------------------------------------------
    # motion event handlers

    def on_motion_notify(self, widget, event): # pylint: disable=unused-argument
        """Handle mouse motion events."""

        ev = MouseEvent(event, state = self.__state)

        self.__bus.emit_mult("cursor_pos_update", ev.pos(), ev.pos_abs())

        if self.__bus.emit_once("mouse_move", ev):
            self.__bus.emit("queue_draw")
            return True

        return True
