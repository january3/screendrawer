"""
Wiglets are small dialog-like objects that can be drawn on the canvas.
They are used to provide interactive controls for changing drawing properties
such as line width, color, and transparency.
"""
from .utils    import get_color_under_cursor, rgb_to_hex         # <remove>
from .drawable_primitives import SelectionTool                   # <remove>
from .commands import RotateCommand, ResizeCommand, MoveCommand  # <remove>
from .commands import RemoveCommand                              # <remove>
from .drawable_factory import DrawableFactory                    # <remove>



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

class WigletPan(Wiglet):
    """Paning the page, i.e. tranposing it with alt-click"""
    def __init__(self, bus, state):
        super().__init__("pan", None)

        self.__bus    = bus
        self.__state  = state
        self.__origin = None
        self.__page   = None
        bus.on("left_mouse_click", self.on_click, priority = 9)
        bus.on("mouse_move",       self.on_move, priority = 99)
        bus.on("mouse_release",   self.on_release, priority = 99)

    def on_click(self, ev):
        """Start paning"""
        if ev.shift() or ev.ctrl() or not ev.alt():
            return False

        print("Panning: start")

        self.__origin = (ev.event.x, ev.event.y)
        self.__page   = self.__state.page()

        return True

    def on_move(self, ev):
        """Update paning"""

        if not self.__origin:
            return False

        print("my origin:", self.__origin)

        page = self.__page
        tr = page.translate()
        if not tr:
            tr = page.translate((0, 0))

        dx, dy = ev.event.x - self.__origin[0], ev.event.y - self.__origin[1]
        tr = (tr[0] + dx, tr[1] + dy)
        print("Translating page by", tr)

        page.translate(tr)

        self.__origin = (ev.event.x, ev.event.y)
        self.__bus.emit("queue_draw")
        return True

    def on_release(self, ev):
        """Handle mouse release event"""
        if not self.__origin:
            return False

        self.__origin = None
        return True

class WigletEditText(Wiglet):
    """Create or edit text objects"""
    def __init__(self, bus, state):
        super().__init__("pan", None)

        self.__bus   = bus
        self.__state = state
        self.__obj   = None
        self.__active = False
        bus.on("left_mouse_click", self.on_click, priority = 0)
        bus.on("mouse_move",       self.on_move, priority = 99)
        bus.on("mouse_release",   self.on_release, priority = 99)
        bus.on("finish_text_input", self.finish_text_input, priority = 99)
        bus.on("left_mouse_double_click",
               self.on_double_click, priority = 9)

    def on_double_click(self, ev):
        """Double click on text launches text editing"""

        if self.__active: # currently editing
            print("WigletEditText: we are active, double click finishes the input")
            self.__bus.emit("finish_text_input")
            return True

        if ev.shift() or ev.ctrl() or ev.alt():
            return False

        obj = ev.hover()
        if not (obj and obj.type == "text"):
            return False

        print("Starting to edit a text object")
        self.__active = True
        self.__state.current_obj(obj)
        self.__obj = obj
        obj.move_caret("End")
        self.__bus.emit("queue_draw")
        return True


    def on_click(self, ev):
        """Start drawing"""

        if self.__active: # currently editing
            self.__bus.emit("finish_text_input")
            return True

        mode = self.__state.mode()

        if ev.shift() and not ev.ctrl():
            mode = "text"

        if mode != "text":
            return False

        print("Creating a new text object")
        obj = DrawableFactory.create_drawable(mode, pen = self.__state.pen(), ev=ev)

        if obj:
            self.__state.current_obj(obj)
            self.__active = True
            self.__obj = obj
        else:
            print("No object created for mode", mode)

        return True

    def on_release(self, ev):
        """Finish drawing object"""

        obj = self.__obj

        if not obj:
            return False

        self.__bus.emit("queue_draw")
        return True

    def finish_text_input(self):
        """Finish text input"""
        if not self.__active:
            return False

        print("Wiglet: finishing text input")

        obj = self.__obj
        obj.show_caret(False)

        if obj.strlen() > 0:
            self.__state.gom().add_object(obj)

        self.__state.current_obj_clear()
        self.__state.cursor().revert()
        self.__active = False
        self.__obj = None

        self.__bus.emit("queue_draw")
        return True


class WigletCreateObject(Wiglet):
    """Create object when clicked"""

    def __init__(self, bus, state):
        super().__init__("pan", None)

        self.__bus   = bus
        self.__state = state
        self.__obj   = None
        bus.on("left_mouse_click", self.on_click, priority = 0)
        bus.on("mouse_move",       self.on_move, priority = 99)
        bus.on("mouse_release",   self.on_release, priority = 99)

    def on_click(self, ev):
        """Start drawing"""

       #if ev.hover() or ev.corner()[0] or ev.double():
       #    return False

        mode = self.__state.mode()

        if ev.shift() and not ev.ctrl():
            mode = "text"

        if mode not in [ "draw", "shape", "rectangle", "circle" ]:
            return False

        print("Creating a new object")
        obj = DrawableFactory.create_drawable(mode, pen = self.__state.pen(), ev=ev)

        if obj:
            self.__state.current_obj(obj)
            self.__obj = obj
        else:
            print("No object created for mode", mode)

        return True

    def on_move(self, ev):
        """Update drawing object"""
        obj = self.__obj

        if not obj:
            return False

        obj.update(ev.x, ev.y, ev.pressure())
        self.__bus.emit("queue_draw")
        return True

    def on_release(self, ev):
        """Finish drawing object"""

        obj = self.__obj

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
            self.__state.gom().add_object(obj)

            # with text, we are not done yet! Need to keep current object
            # such that keyboard events can update it
            if self.__state.current_obj().type != "text":
                self.__state.selection().clear()
                self.__state.current_obj_clear()

        self.__obj = None
        self.__bus.emit("queue_draw")
        return True

class WigletEraser(Wiglet):
    """Erase mode. Removes objects."""

    def __init__(self, bus, state):
        super().__init__("pan", None)

        self.__bus    = bus
        self.__state  = state
        self.__active = False
        bus.on("left_mouse_click", self.on_click, priority = 10)
        bus.on("mouse_move",       self.on_move, priority = 99)
        bus.on("mouse_release",   self.on_release, priority = 99)

    def on_click(self, ev):
        """Clicking above an object removes it"""

        if not self.__state.mode() == "eraser":
            return False

        self.__active = True
        self.__delete_hover(ev)

        return True

    def __delete_hover(self, ev):
        """Delete object underneath"""
        hover_obj = ev.hover()

        if not hover_obj:
            return

        print("removing object")
        gom = self.__state.gom()
        gom.remove_objects([ hover_obj ], clear_selection = True)
        self.__state.cursor().revert()
        self.__bus.emit("queue_draw")

    def on_move(self, ev):
        """Process move: if active, delete everything underneath"""

        if not self.__active:
            return False

        self.__delete_hover(ev)
        return True

    def on_release(self, ev):
        """Stop being active"""
        if not self.__active:
            return False

        self.__active = False
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
