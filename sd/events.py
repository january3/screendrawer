from gi.repository import Gtk, Gdk # <remove>
from .utils import *               # <remove>

## ---------------------------------------------------------------------
class MouseEvent:
    """
    Simple class for handling mouse events.

    Takes the event and computes a number of useful things.

    One advantage of using it: the computationaly intensive stuff is
    computed only once and only if it is needed.
    """
    def __init__(self, event, objects):
        self.event   = event
        self.objects = objects
        self._pos    = (event.x, event.y)
        self._hover  = None
        self._corner = None
        self._shift  = (event.state & Gdk.ModifierType.SHIFT_MASK) != 0
        self._ctrl   = (event.state & Gdk.ModifierType.CONTROL_MASK) != 0
        self._alt    = (event.state & Gdk.ModifierType.MOD1_MASK) != 0
        self._double = event.type == Gdk.EventType.DOUBLE_BUTTON_PRESS
        self._pressure = event.get_axis(Gdk.AxisUse.PRESSURE)
        if self._pressure is None:  # note that 0 is perfectly valid
            self._pressure = 1


    def hover(self):
        if not self._hover:
            self._hover = find_obj_close_to_click(self._pos[0], self._pos[1], self.objects, 20)
        return self._hover

    def corner(self):
        if not self._corner:
            self._corner = find_corners_next_to_click(self._pos[0], self._pos[1], self.objects, 20)
        return self._corner

    def pos(self):
        return self._pos

    def shift(self):
        return self._shift

    def ctrl(self):
        return self._ctrl

    def alt(self):
        return self._alt

    def double(self):
        return self._double

    def pressure(self):
        return self._pressure


