"""
DrawManager is a class that manages the drawing on the canvas.
"""

import gi                                                  # <remove>
gi.require_version('Gtk', '3.0')                           # <remove> pylint: disable=wrong-import-position
from gi.repository import GLib                  # <remove>

from .events   import MouseEvent                                 # <remove>
#from sd.cursor   import CursorManager                            # <remove>



class DrawManager:
    """
    Class that catches mouse events, creates the MouseEvent and sends it
    back to the bus.
    """
    def __init__(self, bus, gom, state, setter):
        self.__bus = bus
        self.__state = state
        self.__gom = gom
        self.__cursor = state.cursor()
        self.__setter = setter
        self.__timeout = None # for discerning double clicks

        # objects that indicate the state of the drawing area
        # drawing parameters

    # ---------------------------------------------------------------------
    #                              Event handlers

    def on_pan(self, gesture, direction, offset):
        """Handle panning events."""
        print(f"Panning: Direction: {direction}, Offset: {offset}, Gesture: {gesture}")

    def on_zoom(self, gesture, scale):
        """Handle zoom events."""
        print(f"Zooming: Scale: {scale}, gesture: {gesture}")

    # ---------------------------------------------------------------------

    # Button press event handlers -------------------------------------------
    def on_button_press(self, widget, event): # pylint: disable=unused-argument
        """Handle mouse button press events."""
        print("on_button_press: type:", event.type, "button:", event.button, "state:", event.state)
        self.__state.modified(True)
        ev = MouseEvent(event, self.__gom.objects(),
                        translate = self.__gom.page().translate(),
                        state = self.__state)

        if event.button == 3:
            if self.__handle_button_3(event, ev):
                return True

        elif event.button == 1:
            if self.__handle_button_1(event, ev):
                return True

        return True

    def __handle_button_3(self, event, ev):
        """Handle right click events, unless shift is pressed."""
        if self.__bus.emit("right_mouse_click", True, ev):
            print("bus event caught the click")
            return True

        return False

    def __handle_button_1(self, event, ev):
        """Handle left click events."""

        if ev.double():
            print("DOUBLE CLICK 1")
           #if self.__handle_text_input_on_click(ev):
           #    return True
            self.__timeout = None
            self.__bus.emit("left_mouse_double_click", True, ev)
            return True

        self.__timeout = event.time

        GLib.timeout_add(50, self.__handle_button_1_single_click, event, ev)
        return True

    def __handle_button_1_single_click(self, event, ev):
        """Handle left click events."""

        # this function needs to return false to stop the timeout func
        print("SINGLE CLICK 1")

        if not self.__timeout:
            print("timeout is None, canceling click")
            return False

        if self.__bus.emit("left_mouse_click", True, ev):
            print("bus event caught the click")
            self.__bus.emit("queue_draw")
            return False

        if self.__handle_mode_specials_on_click(event, ev):
            return False

        return False

    def __handle_mode_specials_on_click(self, event, ev):
        """Handle special events for the current mode."""

        return False

    def __handle_text_input_on_click(self, ev):
        """Check whether text object should be activated."""
        hover_obj = ev.hover()

        if not hover_obj or hover_obj.type != "text":
            return False

        if not self.__state.mode() in ["draw", "text", "move"]:
            return False

        if not ev.double():
            return False

        if ev.shift() or ev.ctrl():
            return False

        # only when double clicked with no modifiers - start editing the hover obj
        print("starting text editing existing object")
        hover_obj.move_caret("End")
        self.__state.current_obj(hover_obj)
        self.__bus.emit("queue_draw")
        self.__cursor.set("none")
        return True


    # Event handlers
    # XXX same comment as above
    # Button release handlers ------------------------------------------------
    def on_button_release(self, widget, event):
        """Handle mouse button release events."""
        print("button release: type:", event.type, "button:", event.button, "state:", event.state)
        ev = MouseEvent(event, self.__gom.objects(),
                        translate = self.__gom.page().translate(),
                        state = self.__state)

        if self.__bus.emit("mouse_release", True, ev):
            self.__bus.emit("queue_draw")
            return True

        return True

    # ---------------------------------------------------------------------
    # motion event handlers

    def on_motion_notify(self, widget, event):
        """Handle mouse motion events."""

        ev = MouseEvent(event, self.__gom.objects(),
                        translate = self.__gom.page().translate(),
                        state = self.__state)

        if self.__bus.emit("mouse_move", True, ev):
            self.__bus.emit("queue_draw")
            return True

        x, y = ev.pos()
        self.__cursor.update_pos(x, y)

        # stop event propagation
        return True
