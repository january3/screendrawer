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
    def __init__(self, event, objects, translate = None):
        self.event   = event
        self.objects = objects
        self.__hover  = None
        self.__corner = None
        self.__shift  = (event.state & Gdk.ModifierType.SHIFT_MASK) != 0
        self.__ctrl   = (event.state & Gdk.ModifierType.CONTROL_MASK) != 0
        self.__alt    = (event.state & Gdk.ModifierType.MOD1_MASK) != 0
        self.__double = event.type == Gdk.EventType.DOUBLE_BUTTON_PRESS
        self.__pressure = event.get_axis(Gdk.AxisUse.PRESSURE)
        if self.__pressure is None:  # note that 0 is perfectly valid
            self.__pressure = 1
        self.x = event.x
        self.y = event.y

        if translate:
            self.x, self.y = self.x - translate[0], self.y - translate[1]
        self.__pos    = (self.x, self.y)


    def hover(self):
        if not self.__hover:
            self.__hover = find_obj_close_to_click(self.__pos[0], self.__pos[1], self.objects, 20)
        return self.__hover

    def corner(self):
        if not self.__corner:
            self.__corner = find_corners_next_to_click(self.__pos[0], self.__pos[1], self.objects, 20)
        return self.__corner

    def pos(self):
        return self.__pos

    def shift(self):
        return self.__shift

    def ctrl(self):
        return self.__ctrl

    def alt(self):
        return self.__alt

    def double(self):
        return self.__double

    def pressure(self):
        return self.__pressure


