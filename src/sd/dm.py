"""
DrawManager is a class that manages the drawing on the canvas.
"""

import gi                                                  # <remove>
gi.require_version('Gtk', '3.0')                           # <remove> pylint: disable=wrong-import-position
from gi.repository import GLib                  # <remove>

from .drawable_factory import DrawableFactory             # <remove>
from .drawable_primitives import SelectionTool             # <remove>
from .commands import RemoveCommand, MoveCommand, ResizeCommand, RotateCommand  # <remove>
from .events   import MouseEvent                                 # <remove>
#from sd.cursor   import CursorManager                            # <remove>



class DrawManager:
    """
    DrawManager is a class that manages the drawing canvas.
    It holds information about the state, the mouse events, the position of
    the cursor, whether there is a current object being generated, whether
    a resize operation is in progress, whether the view is paned or zoomed etc.

    DrawManager must be aware of GOM, because GOM holds all the objects

    methods from app used:
    app.clipboard.set_text() # used for color picker

    methods from state used:
    state.cursor()
    state.show_wiglets()
    state.mode()
    state.current_obj()
    state.current_obj_clear()
    state.pen()
    state.get_win_size()
    state.hover_obj()
    state.hover_obj_clear()

    methods from gom used:
    gom.set_page_number()
    gom.page()
    gom.draw()
    gom.selected_objects()
    gom.selection()
    gom.remove_objects()
    gom.selection().set()
    gom.selection().add()
    gom.selection().clear()
    gom.add_object()
    gom.kill_object()
    gom.remove_all()
    gom.command_append()
    """
    def __init__(self, bus, gom, state, setter):
        self.__bus = bus
        self.__state = state
        self.__gom = gom
        self.__cursor = state.cursor()
        self.__setter = setter
        self.__timeout = None # for discerning double clicks

        # objects that indicate the state of the drawing area
        self.__resizeobj = None
        self.__paning = None
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

    def __move_resize_rotate(self, ev):
        """Process events for moving, resizing and rotating objects."""

        if self.__state.mode() != "move":
            return False

        corner_obj, corner = ev.corner()
        hover_obj  = ev.hover()
        pos = ev.pos()
        shift, ctrl = ev.shift(), ev.ctrl()

        if False and corner_obj and corner_obj.bbox():
            print("starting resize")
            print("ctrl:", ctrl, "shift:", shift)
            # XXX this code here is one of the reasons why rotating or resizing the
            # whole selection does not work. The other reason is that the
            # selection object itself is not considered when detecting
            # hover or corner objects.
            if ctrl and shift:
                self.__resizeobj = RotateCommand(corner_obj, origin = pos,
                                                 corner = corner)
            else:
                self.__resizeobj = ResizeCommand(corner_obj, origin = pos,
                                                 corner = corner, proportional = ctrl)
            self.__gom.selection().set([ corner_obj ])
            self.__cursor.set(corner)
            return True

        if False and hover_obj:
            if ev.shift():
                # add if not present, remove if present
                print("adding object to selection:", hover_obj)
                self.__gom.selection().add(hover_obj)
            if not self.__gom.selection().contains(hover_obj):
                print("object not in selection, setting selection to it:", hover_obj)
                self.__gom.selection().set([ hover_obj ])
            # we are using the .selection().copy() because the selection
            # object can change in time
            self.__resizeobj = MoveCommand(self.__gom.selection().copy(), pos)
            self.__cursor.set("grabbing")
            return True

        return False

    def create_object(self, ev):
        """Create an object based on the current mode."""

        # not managed by GOM: first create, then decide whether to add to GOM
        mode = self.__state.mode()

        if ev.shift() and not ev.ctrl():
            mode = "text"

        if mode not in [ "draw", "shape", "rectangle", "circle", "text" ]:
            return False

        obj = DrawableFactory.create_drawable(mode, pen = self.__state.pen(), ev=ev)

        if obj:
            self.__state.current_obj(obj)
        else:
            print("No object created for mode", mode)

        return True

    # Button press event handlers -------------------------------------------
    def on_button_press(self, widget, event): # pylint: disable=unused-argument
        """Handle mouse button press events."""
        print("on_button_press: type:", event.type, "button:", event.button, "state:", event.state)
        self.__state.modified(True)
        ev = MouseEvent(event, self.__gom.objects(),
                        translate = self.__gom.page().translate(),
                        state = self.__state)

        # Ignore clicks when text input is active
        if self.__state.current_obj():
            if  self.__state.current_obj().type == "text":
                print("click, but text input active - finishing it first")
                self.__setter.finish_text_input()

        # right click: emit right click event
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

        if ev.shift():
            return False
        #self.on_right_click(ev)
        return True

    def __handle_button_1(self, event, ev):
        """Handle left click events."""

        if ev.double():
            print("DOUBLE CLICK 1")
            if self.__handle_text_input_on_click(ev):
                return True
            self.__timeout = None
            self.__bus.emit("left_mouse_double_click", True, ev)
            return True

        self.__timeout = event.time

        GLib.timeout_add(50, self.__handle_button_1_single_click, event, ev)
        return True

        return self.__handle_button_1_single_click(event, ev)

    def __handle_button_1_single_click(self, event, ev):
        """Handle left click events."""
        print("SINGLE CLICK 1")

        if not self.__timeout:
            print("timeout is None, canceling click")
            return False

        if self.__bus.emit("left_mouse_click", True, ev):
            print("bus event caught the click")
            self.__bus.emit("queue_draw")
            return False


        if ev.alt():
            self.__paning = (event.x, event.y)
            return False

        if self.__move_resize_rotate(ev):
            return False

        if self.__handle_mode_specials_on_click(event, ev):
            return False

        # simple click: create modus
        self.create_object(ev)
        self.__bus.emit("queue_draw")

        return False

    def __handle_mode_specials_on_click(self, event, ev):
        """Handle special events for the current mode."""

        if self.__handle_eraser_on_click(ev):
            return True

        return False

    def __handle_eraser_on_click(self, ev):
        """Handle eraser on click events."""
        if not self.__state.mode() == "eraser":
            return False

        hover_obj = ev.hover()
        if not hover_obj:
            return False

        self.__gom.remove_objects([ hover_obj ], clear_selection = True)
        self.__resizeobj   = None
        self.__cursor.revert()
        self.__bus.emit("queue_draw")
        return True

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

        if self.__paning:
            self.__paning = None
            return True

        if self.__handle_current_object_on_release(ev):
            return True

        return True

    def __handle_current_object_on_release(self, ev):
        """Handle the current object on mouse release."""
        obj = self.__state.current_obj()
        if not obj:
            return False

        if obj.type in [ "shape", "path" ]:
            print("finishing path / shape")
            obj.path_append(ev.x, ev.y, 0)
            obj.finish()
            # remove paths that are too small
            if len(obj.coords) < 3:
                print("removing object of type", obj.type, "because too small")
                self.__state.current_obj_clear()
                obj = None

        # remove objects that are too small
        if obj:
            bb = obj.bbox()
            if bb and obj.type in [ "rectangle", "box", "circle" ] and bb[2] == 0 and bb[3] == 0:
                print("removing object of type", obj.type, "because too small")
                self.__state.current_obj_clear()
                obj = None

        if obj:
            self.__gom.add_object(obj)
            # with text, we are not done yet! Need to keep current object
            # such that keyboard events can update it
            if self.__state.current_obj().type != "text":
                self.__gom.selection().clear()
                self.__state.current_obj_clear()

        self.__bus.emit("queue_draw")
        return True

    def __handle_drag_on_release(self, event):
        """Handle mouse release on drag events."""
        if not self.__resizeobj:
            return False

        # If the user was dragging a selected object and the drag ends
        # in the lower left corner, delete the object
        self.__resizeobj.event_finish()
        # self.__resizeobj is a command object
        # self.__resizeobj.obj is a copy of the selection group
        # self.__resizeobj.obj.objects is the list of objects in the copy of the selection group

        obj = self.__resizeobj.obj

        _, height = self.__state.get_win_size()
        if self.__resizeobj.command_type() == "move" and  event.x < 10 and event.y > height - 10:
            # command group because these are two commands: first move,
            # then remove
            self.__gom.command_append([ self.__resizeobj,
                                       RemoveCommand(obj.objects,
                                                     self.__gom.objects()) ])
            self.__gom.selection().clear()
        else:
            self.__gom.command_append([ self.__resizeobj ])
        self.__resizeobj    = None
        self.__cursor.revert()
        self.__bus.emit("queue_draw")

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

        # we are paning
        if self.__on_motion_paning(event):
            return True

        if self.__on_motion_update_object(ev):
            return True

        if self.__on_motion_update_resize(ev):
            return True

        self.__on_motion_process_hover(ev)
        # stop event propagation
        return True

    def __on_motion_process_hover(self, ev):
        """Process hover events."""

        if not self.__state.mode() == "move":
            return False

        object_underneath = ev.hover()

        if object_underneath:
            self.__cursor.set("moving")
            self.__state.hover_obj(object_underneath)
        else:
            self.__cursor.revert()
            self.__state.hover_obj_clear()

        corner_obj, corner = ev.corner()

        if corner_obj and corner_obj.bbox():
            self.__cursor.set(corner)
            self.__state.hover_obj(corner_obj)
            self.__bus.emit("queue_draw")

        self.__bus.emit("queue_draw")
        return True

    def __on_motion_update_resize(self, ev):
        """Handle on motion update for resizing."""
        if not self.__resizeobj:
            return False

        self.__resizeobj.event_update(ev.x, ev.y)
        self.__bus.emit("queue_draw")
        return True

    def __on_motion_update_object(self, event):
        """Handle on motion update for an object."""
        obj = self.__state.current_obj()
        if not obj:
            return False

        obj.update(event.x, event.y, event.pressure())
        self.__bus.emit("queue_draw")
        return True

    def __on_motion_paning(self, event):
        """Handle on motion update when paning"""
        if not self.__paning:
            return False

        tr = self.__gom.page().translate()
        if not tr:
            tr = self.__gom.page().translate((0, 0))
        dx, dy = event.x - self.__paning[0], event.y - self.__paning[1]
        tr = (tr[0] + dx, tr[1] + dy)
        self.__gom.page().translate(tr)
        self.__paning = (event.x, event.y)
        self.__bus.emit("queue_draw")
        return True

        return False
