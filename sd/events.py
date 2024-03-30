"""
This module contains the MouseEvent class.
It is used to handle mouse events in the drawing area.
"""
from gi.repository import Gdk # <remove>
from .utils import find_obj_close_to_click, find_corners_next_to_click # <remove>

## ---------------------------------------------------------------------
class MouseEvent:
    """
    Simple class for handling mouse events.

    Takes the event and computes a number of useful things.

    One advantage of using it: the computationaly intensive stuff is
    computed only once and only if it is needed.
    """
    def __init__(self, event, objects, translate = None):
        self.event   = event
        self.objects = objects
        self.__modifiers = {
                "shift": (event.state & Gdk.ModifierType.SHIFT_MASK) != 0,
                "ctrl": (event.state & Gdk.ModifierType.CONTROL_MASK) != 0,
                "alt": (event.state & Gdk.ModifierType.MOD1_MASK) != 0,
                }
        self.__info = {
                "double":   (event.type == Gdk.EventType.DOUBLE_BUTTON_PRESS),
                "pressure": event.get_axis(Gdk.AxisUse.PRESSURE),
                "hover":    None,
                "corner":   None 
                }

        if self.__info["pressure"] is None:  # note that 0 is perfectly valid
            self.__info["pressure"] = 1

        self.x, self.y = event.x, event.y

        if translate:
            self.x, self.y = self.x - translate[0], self.y - translate[1]

        self.__pos    = (self.x, self.y)


    def hover(self):
        """Return the object that is hovered by the mouse."""
        if not self.__info.get("hover"):
            self.__info["hover"] = find_obj_close_to_click(self.__pos[0],
                                                   self.__pos[1],
                                                   self.objects, 20)
        return self.__info["hover"]

    def corner(self):
        """Return the corner that is hovered by the mouse."""
        if not self.__info.get("corner"):
            self.__info["corner"] = find_corners_next_to_click(self.__pos[0],
                                                       self.__pos[1],
                                                       self.objects, 20)
        return self.__info["corner"]

    def pos(self):
        """Return the position of the mouse."""
        return self.__pos

    def shift(self):
        """Return True if the shift key is pressed."""
        return self.__modifiers.get("shift")

    def ctrl(self):
        """Return True if the control key is pressed."""
        return self.__modifiers.get("ctrl")

    def alt(self):
        """Return True if the alt key is pressed."""
        return self.__modifiers.get("alt")

    def double(self):
        """Return True if the event is a double click."""
        return self.__info.get("double")

    def pressure(self):
        """Return the pressure of the pen."""
        return self.__info.get("pressure")
