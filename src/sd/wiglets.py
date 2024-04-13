"""
Wiglets are small dialog-like objects that can be drawn on the canvas.
They are used to provide interactive controls for changing drawing properties
such as line width, color, and transparency.
"""
from .utils    import get_color_under_cursor, rgb_to_hex         # <remove>
from .drawable_primitives import SelectionTool   # <remove>
from .commands import RotateCommand, ResizeCommand, MoveCommand                # <remove>
from .commands import RemoveCommand                # <remove>



## ---------------------------------------------------------------------
class Wiglet:
    """drawable dialog-like objects on the canvas"""
    def __init__(self, mytype, coords):
        self.wiglet_type   = mytype
        self.coords = coords

    def update_size(self, width, height): # pylint: disable=unused-argument
        """ignore update the size of the widget"""
        return False

    def draw(self, cr, state): # pylint: disable=unused-argument
        """do not draw the widget"""

    def on_click(self, ev): # pylint: disable=unused-argument
        """ignore the click event"""
        return False

    def on_release(self, ev): # pylint: disable=unused-argument
        """ignore the release event"""
        return False

    def on_move(self, ev): # pylint: disable=unused-argument
        """ignore update on mouse move"""
        return False

class WigletResizeRotate(Wiglet):
    """Catch resize events and update the size of the object."""

    def __init__(self, bus, gom, state):
        super().__init__("resize", None)
        self.__gom = gom
        self.__bus = bus
        self.__cmd = None
        self.__state = state

        bus.on("left_mouse_click", self.on_click)
        bus.on("mouse_move", self.on_move)
        bus.on("mouse_release", self.on_release)

    def on_click(self, ev):
        print("resize widget clicked")
        if ev.mode() != "move" or ev.alt():
            print("resizing - wrong modifiers")
            return False

        corner_obj, corner = ev.corner()
        if not corner_obj or not corner_obj.bbox() or ev.double():
            print("widget resizing wrong event", ev.hover(), ev.corner(), ev.double())
            return False
        print("widget resizing object", corner_obj)

        if ev.ctrl() and ev.shift():
            print("resizing with both shift and ctrl")
            self.__cmd = RotateCommand(corner_obj, origin = ev.pos(),
                                             corner = corner)
        else:
            self.__cmd = ResizeCommand(corner_obj, origin = ev.pos(),
                                   corner = corner,
                                   proportional = ev.ctrl())

        self.__gom.selection().set([corner_obj])
        self.__state.cursor().set(corner)
        self.__bus.emit("queue_draw")

        return True

    def on_move(self, ev):
        if not self.__cmd:
            return False

        self.__cmd.event_update(*ev.pos())
        self.__bus.emit("queue_draw")
        return True

    def on_release(self, ev):
        if not self.__cmd:
            return False
        self.__cmd.event_update(*ev.pos())
        self.__cmd.event_finish()
        self.__gom.command_append([ self.__cmd ])
        self.__state.cursor().revert()
        self.__cmd = None
        self.__bus.emit("queue_draw")
        return True

class WigletHover(Wiglet):
    """Change cursor when moving over objects."""

    def __init__(self, bus, state):
        super().__init__("hover", None)

        self.__bus   = bus
        self.__state = state

        bus.on("mouse_move", self.on_move, priority = -9)

    def on_move(self, ev):
        """When cursor hovers over an object"""

        if not self.__state.mode() == "move":
            return False

        corner_obj, corner = ev.corner()
        object_underneath  = ev.hover()

        cursor = self.__state.cursor()

        if corner_obj and corner_obj.bbox():
            cursor.set(corner)
        elif object_underneath:
            cursor.set("moving")
            self.__state.hover_obj(object_underneath)
        else:
            cursor.revert()
            self.__state.hover_obj_clear()


        self.__bus.emit("queue_draw")
        return True

class WigletMove(Wiglet):
    """Catch moving events and update the position of the object."""

    def __init__(self, bus, gom, state):
        super().__init__("move", None)
        self.__gom = gom
        self.__bus = bus
        self.__cmd = None
        self.__state = state
        self.__obj = None

        bus.on("left_mouse_click", self.on_click)
        bus.on("mouse_move", self.on_move)
        bus.on("mouse_release", self.on_release)

    def on_click(self, ev):
        if ev.mode() != "move" or ev.alt() or ev.ctrl():
            print("wrong modifiers")
            return False

        if not ev.hover() or ev.corner()[0] or ev.double():
            print("widget moving wrong event", ev.hover(), ev.corner(), ev.double())
            return False
        print("widget moving object", ev.hover())

        # first, update the current selection
        obj = ev.hover()
        selection = self.__gom.selection()

        if ev.shift():
            selection.add(obj)
        if not selection.contains(obj):
            selection.set([obj])

        self.__obj = selection.copy()
        self.__cmd = MoveCommand(self.__obj, ev.pos())
        self.__state.cursor().set("grabbing")
        self.__bus.emit("queue_draw")

        return True

    def on_move(self, ev):
        if not self.__cmd:
            return False

        self.__cmd.event_update(*ev.pos())
        self.__bus.emit("queue_draw")
        return True

    def on_release(self, ev):
        if not self.__cmd:
            return False

        _, height = self.__state.get_win_size()
        cmd = self.__cmd
        cmd.event_update(*ev.pos())
        cmd.event_finish()

        obj = self.__obj
        gom = self.__gom

        if ev.event.x < 10 and ev.event.y > height - 10:
            # command group because these are two commands: first move,
            # then remove
            gom.command_append([ cmd,
                                 RemoveCommand(obj.objects,
                                               gom.objects()) ])
            gom.selection().clear()
        else:
            gom.command_append([ cmd ])
        self.__state.cursor().revert()
        self.__cmd = None
        self.__obj = None
        self.__bus.emit("queue_draw")
        return True

class WigletSelectionTool(Wiglet):
    """Draw the selection tool when activated."""

    def __init__(self, bus, gom):
        super().__init__("selection_tool", None)

        self.__selection_tool = None
        self.__bus = bus
        self.__gom   = gom
        bus.on("left_mouse_click", self.on_click, priority = -1)
        bus.on("mouse_move",       self.on_move)
        bus.on("mouse_release",   self.on_release)
        bus.on("draw", self.draw)

    def draw(self, cr, state):
        """draw the widget"""
        if not self.__selection_tool:
            return
        self.__selection_tool.draw(cr, state)

    def on_click(self, ev):
        """handle the click event"""
        print("receiving call")
        # ev.shift() means "add to current selection"
        if ev.mode() != "move" or ev.alt() or ev.ctrl():
            print("wrong modifiers")
            return False

        if ev.hover() or ev.corner()[0] or ev.double():
            print("wrong event", ev.hover(), ev.corner(), ev.double())
            return False
        print("taking the call")
        self.__gom.selection().clear()
        x, y, = ev.event.x, ev.event.y
        self.coords = (x, y)
        self.__selection_tool = SelectionTool([ (x, y), (x + 1, y + 1) ])
        return True

    def on_move(self, ev):
        """update on mouse move"""
        if not self.__selection_tool:
            return False

        x, y = ev.event.x, ev.event.y
        obj = self.__selection_tool
        obj.update(x, y, ev.pressure())
        self.__bus.emit("queue_draw")
        return True

    def on_release(self, ev):
        """handle the release event"""
        if not self.__selection_tool:
            return False

        objects = self.__selection_tool.objects_in_selection(self.__gom.objects())

        if len(objects) > 0:
            self.__gom.selection().set(objects)
        else:
            self.__gom.selection().clear()

        self.__bus.emit("queue_draw")
        self.__selection_tool = None
        return True

# ---------------------------------------------------------------------
class WigletColorPicker(Wiglet):
    """Invisible wiglet that processes clicks in the color picker mode."""
    def __init__(self, bus, func_color, clipboard):
        super().__init__("colorpicker", None)

        self.__func_color = func_color
        self.__clipboard = clipboard
        bus.on("left_mouse_click", self.on_click, priority = 1)

    def on_click(self, ev):
        """handle the click event"""

        # only works in color picker mode
        if ev.mode() != "colorpicker":
            return False

        if ev.shift() or ev.alt() or ev.ctrl():
            return False

        color = get_color_under_cursor()
        self.__func_color(color)

        color_hex = rgb_to_hex(color)
        self.__clipboard.set_text(color_hex)
        #self.__state.queue_draw()
        return True
