"""
Wiglets are small dialog-like objects that can be drawn on the canvas.
They are used to provide interactive controls for changing drawing properties
such as line width, color, and transparency.
"""
import logging                                                   # <remove>

from .utils    import get_color_under_cursor, rgb_to_hex         # <remove>
from .drawable_primitives import SelectionTool                   # <remove>
from .commands import AddToGroupCommand                          # <remove>
from .commands import CommandGroup                               # <remove>
from .commands_obj import RemoveCommand, AddCommand              # <remove>
from .commands_obj import ResizeCommand, MoveCommand             # <remove>
from .commands_obj import RotateCommand                          # <remove>
from .commands import TextEditCommand                            # <remove>
from .drawable_factory import DrawableFactory                    # <remove>
from .drawable_group import DrawableGroup                        # <remove>
from .utils import bus_listeners_on, bus_listeners_off           # <remove>
log = logging.getLogger(__name__)                                # <remove>
#log.setLevel(logging.INFO)

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

    def __init__(self, bus, state):
        super().__init__("resize", None)
        self.__bus = bus
        self.__cmd = None
        self.__state = state # pylint: disable=unused-private-member

        bus.on("left_mouse_click", self.on_click)

    def on_click(self, ev):
        log.debug("resize widget clicked")
        if ev.mode() != "move" or ev.alt():
            log.debug("resizing - wrong modifiers")
            return False

        corner_obj, corner = ev.corner()

        if not corner_obj or not corner_obj.bbox() or ev.double():
            log.debug("widget resizing wrong event hover: %s, corner: %s, double: %s",
                    ev.hover, ev.corner, ev.double)
            return False

        log.debug("widget resizing object. corner: %s", corner_obj)

        if ev.ctrl() and ev.shift():
            log.debug("rotating with both shift and ctrl")
            self.__cmd = RotateCommand(corner_obj, origin = ev.pos(),
                                             corner = corner)
        else:
            self.__cmd = ResizeCommand(corner_obj, origin = ev.pos(),
                                   corner = corner,
                                   proportional = ev.ctrl())

        self.__bus.on("mouse_move", self.on_move)
        self.__bus.on("mouse_release", self.on_release)
        self.__bus.emit_once("set_selection", [ corner_obj ])
        self.__bus.emit_once("cursor_set", corner)
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
        self.__bus.emit_once("history_append", self.__cmd)
        self.__bus.emit_once("cursor_revert")
        self.__cmd = None
        self.__bus.off("mouse_move", self.on_move)
        self.__bus.off("mouse_release", self.on_release)
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

        if not ev.mode() == "move":
            return False

        corner_obj, corner = ev.corner()
        object_underneath  = ev.hover()

        if corner_obj and corner_obj.bbox():
            self.__bus.emit_once("cursor_set", corner)
        elif object_underneath:
            self.__bus.emit_once("cursor_set", "moving")
            self.__state.hover_obj(object_underneath)
        else:
            self.__bus.emit_once("cursor_revert")
            self.__state.hover_obj_clear()


        self.__bus.emit("queue_draw")
        return True

class WigletMove(Wiglet):
    """Catch moving events and update the position of the object."""

    def __init__(self, bus, state):
        super().__init__("move", None)
        self.__bus = bus
        self.__cmd = None
        self.__state = state
        self.__obj = None

        bus.on("left_mouse_click", self.on_click)

    def on_click(self, ev):
        """Start moving object"""
        obj = ev.hover()

        if ev.mode() != "move" or ev.alt() or ev.ctrl():
            log.debug("wrong modifiers")
            return False

        if not obj or ev.corner()[0] or ev.double():
            log.debug("widget moving wrong event")
            return False

        # first, update the current selection
        selection = self.__state.selection()

        if ev.shift():
            selection.add(obj)
        if not selection.contains(obj):
            selection.set([obj])

        self.__bus.on("cancel_left_mouse_single_click", self.cancel)
        self.__bus.on("mouse_move", self.on_move)
        self.__bus.on("mouse_release", self.on_release)

        self.__obj = selection.copy()
        log.debug("moving object: %s", self.__obj)
        self.__cmd = MoveCommand(self.__obj, ev.pos())
        self.__bus.emit_once("cursor_set", "grabbing")
        self.__bus.emit("queue_draw")

        return True

    def on_move(self, ev):
        """Update moving object"""
        if not self.__cmd:
            return False

        self.__cmd.event_update(*ev.pos())
        self.__bus.emit("queue_draw")
        return True

    def cancel(self, ev): # pylint: disable=unused-argument
        """Cancel the move operation"""
        if not self.__cmd:
            return False

        self.__bus.off("cancel_left_mouse_single_click", self.cancel)
        self.__bus.off("mouse_move", self.on_move)
        self.__bus.off("mouse_release", self.on_release)

        self.__cmd = None
        self.__obj = None

        return True

    def on_release(self, ev):
        """Finish moving object"""
        if not self.__cmd:
            return False

        self.__bus.off("cancel_left_mouse_single_click", self.cancel)
        self.__bus.off("mouse_move", self.on_move)
        self.__bus.off("mouse_release", self.on_release)

        _, height = self.__state.get_win_size()
        cmd = self.__cmd
        cmd.event_update(*ev.pos())
        cmd.event_finish()

        obj = self.__obj
        page = self.__state.page()

        if ev.event.x < 10 and ev.event.y > height - 10:
            # command group because these are two commands: first move,
            # then remove
            cmd = CommandGroup([ cmd,
                                 RemoveCommand(obj.objects,
                                               page.layer().objects()) ])
            self.__bus.emit_once("history_append", cmd)
            self.__bus.emit_once("set_selection", "nothing")
        else:
            self.__bus.emit_once("history_append", cmd)

        self.__bus.emit_once("cursor_revert")
        self.__cmd = None
        self.__obj = None
        self.__bus.emit("queue_draw")
        return True

class WigletSelectionTool(Wiglet):
    """Draw the selection tool when activated."""

    def __init__(self, bus, state):
        super().__init__("selection_tool", None)

        self.__selection_tool = None
        self.__bus = bus
        self.__state   = state
        bus.on("left_mouse_click", self.on_click, priority = -1)

    def draw(self, cr, state):
        """draw the widget"""
        if not self.__selection_tool:
            return
        self.__selection_tool.draw(cr, state)

    def on_click(self, ev):
        """handle the click event"""
        log.debug("receiving call")
        # ev.shift() means "add to current selection"
        if ev.mode() != "move" or ev.alt() or ev.ctrl():
            log.debug("wrong modifiers")
            return False

        if ev.hover() or ev.corner()[0] or ev.double():
            log.debug("wrong event; hover=%s corner=%s double=%s",
                      ev.hover(), ev.corner(), ev.double())
            return False

        log.debug("taking the call")

        self.__bus.on("mouse_move",       self.on_move)
        self.__bus.on("mouse_release",   self.on_release)
        self.__bus.on("obj_draw", self.draw)
        self.__bus.emit_once("set_selection", "nothing")

        x, y, = ev.x, ev.y
        self.coords = (x, y)
        self.__selection_tool = SelectionTool([ (x, y), (x + 1, y + 1) ])
        return True

    def on_move(self, ev):
        """update on mouse move"""
        if not self.__selection_tool:
            return False

        x, y = ev.x, ev.y
        obj = self.__selection_tool
        obj.update(x, y, ev.pressure())
        self.__bus.emit("queue_draw")
        return True

    def on_release(self, ev):
        """handle the release event"""
        if not self.__selection_tool:
            return False

        page = self.__state.page()
        objects = self.__selection_tool.objects_in_selection(page.layer().objects())

        if len(objects) > 0:
            self.__bus.emit_once("set_selection", objects)
        else:
            self.__bus.emit_once("set_selection", "nothing")

        self.__bus.off("mouse_move",       self.on_move)
        self.__bus.off("mouse_release",   self.on_release)
        self.__bus.off("obj_draw", self.draw)
        self.__bus.emit("queue_draw")

        self.__selection_tool = None
        return True

class WigletPan(Wiglet):
    """Paning the page, i.e. tranposing it with alt-click"""
    def __init__(self, bus):
        super().__init__("pan", None)

        self.__bus    = bus
        self.__origin = None
        bus.on("left_mouse_click", self.on_click, priority = 9)

    def on_click(self, ev):
        """Start paning"""
        if ev.shift() or ev.ctrl() or not ev.alt():
            return False
        self.__origin = (ev.event.x, ev.event.y)
        self.__bus.on("mouse_move",     self.on_move, priority = 99)
        self.__bus.on("mouse_release",  self.on_release, priority = 99)
        return True

    def on_move(self, ev):
        """Update paning"""

        if not self.__origin:
            return False

        dx, dy = ev.event.x - self.__origin[0], ev.event.y - self.__origin[1]
        self.__bus.emit_once("page_translate", (dx, dy))
        self.__origin = (ev.event.x, ev.event.y)
        self.__bus.emit("force_redraw")
        return True

    def on_release(self, ev):
        """Handle mouse release event"""
        self.__bus.off("mouse_move",     self.on_move)
        self.__bus.off("mouse_release",  self.on_release)

        if not self.__origin:
            log.warning("no origin")
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
        self.__edit_existing = False
        bus.on("left_mouse_click",  self.on_click, priority = 99)
        bus.on("left_mouse_double_click",
               self.on_double_click, priority = 9)

        self.__listeners = {
            "mouse_move": { "listener":        self.on_move, "priority": 99},
            "mouse_release": { "listener":     self.on_release, "priority": 99},
            "mode_set": { "listener":          self.finish_text_input, "    priority": 99},
            "finish_text_input": { "listener": self.finish_text_input, "priority": 99},
            "escape": { "listener":            self.finish_text_input, "priority": 99},
            }

    def start_listening(self):
        """Start listening to events"""
        bus_listeners_on(self.__bus, self.__listeners)

    def stop_listening(self):
        """Stop listening to events"""
        bus_listeners_off(self.__bus, self.__listeners)

    def on_double_click(self, ev):
        """Double click on text launches text editing"""

        if self.__active: # currently editing
            self.__bus.emit("finish_text_input")
            return True

        if ev.shift() or ev.ctrl() or ev.alt():
            return False

        obj = ev.hover()
        if not (obj and obj.type == "text"):
            return False

        self.__edit_existing = obj.to_string()
        self.__obj = obj
        self.__active = True
        self.__state.current_obj(obj)
        self.__obj.move_caret("End")
        self.__bus.emit("queue_draw")
        self.start_listening()
        return True

    def on_click(self, ev):
        """Start typing text"""

        if self.__active: # currently editing
            self.__bus.emit("finish_text_input")
            return False

        mode = self.__state.mode()
        log.debug("mode %s", mode)

        if ev.shift() and not ev.ctrl() and mode != "move":
            mode = "text"

        if mode != "text":
            return False

        log.debug("Creating a new text object")
        self.__edit_existing = False

        obj = DrawableFactory.create_drawable(mode, pen = self.__state.pen(), ev=ev)

        if obj:
            self.__state.current_obj(obj)
            self.__active = True
            self.__obj = obj
        else:
            log.debug("No object created for mode %s", mode)
            return False

        self.start_listening()
        return True

    def on_release(self, ev):
        """Finish drawing object"""

        obj = self.__obj

        if not obj:
            return False

        self.__bus.emit("queue_draw")
        return True

    def finish_text_input(self, new_mode = False): #pylint: disable=unused-argument
        """Finish text input"""
        if not self.__active:
            return False

        log.debug("finishing text input")

        obj = self.__obj
        obj.show_caret(False)

        if self.__edit_existing:
            page = self.__state.page()

            if obj.strlen() > 0:
                # create a command to allow undo
                cmd = TextEditCommand(obj, self.__edit_existing, obj.to_string())
                self.__bus.emit("history_append", True, cmd)
            else:
                # remove the object
                cmd1 = TextEditCommand(obj, self.__edit_existing, obj.to_string())
                cmd2 = RemoveCommand([ obj ], page.layer().objects())
                self.__bus.emit("history_append", True, CommandGroup([ cmd1, cmd2 ]))
        elif obj.strlen() > 0:
            self.__bus.emit("add_object", True, obj)

        self.__state.current_obj_clear()
        self.__bus.emit_once("cursor_revert")
        self.__active = False
        self.__obj = None

        self.__bus.emit("queue_draw")
        self.stop_listening()
        return True

class WigletCreateSegments(Wiglet):
    """Create a segmented path"""

    def __init__(self, bus, state):
        super().__init__("pan", None)

        self.__bus   = bus
        self.__state = state
        self.__obj   = None
        bus.on("left_mouse_click", self.on_click,   priority = 90)
        bus.on("mode_set",         self.cancel,     priority = 99)
        bus.on("escape", self.cancel,   priority = 99)
        bus.on("left_mouse_double_click", self.on_finish,   priority = 99)
        bus.on("mouse_move",       self.on_move,    priority = 99)
        bus.on("obj_draw",         self.draw_obj,   priority = 99)

    def cancel(self, new_mode = None):
        """Cancel creating a segmented path"""
        mode = self.__state.mode()

        if new_mode is not None and self.__obj:
            self.__bus.emit("add_object", True, self.__obj)
            self.__state.selection().clear()

            self.__obj = None
            self.__bus.emit("queue_draw")
            return False

        if self.__obj:
            log.debug("WigletCreateSegments: cancel")
            self.__obj = None

        if mode != "segment":
            return False

        return True

    def on_move(self, ev):
        """Update drawing object"""
        obj = self.__obj

        if not obj:
            return False

        obj.path_pop()
        obj.update(ev.x, ev.y, ev.pressure())
        self.__bus.emit("queue_draw")
        return True

    def draw_obj(self, cr, state):
        """Draw the object currently being created"""
        if not self.__obj:
            return False

        self.__obj.draw(cr)
        return True

    def on_click(self, ev):
        """Start drawing"""

        if ev.ctrl() or ev.alt():
            return False

        mode = self.__state.mode()
        log.debug("segment on_click here")

        if mode != "segment" or ev.shift() or ev.ctrl():
            return False

        if self.__obj:
            self.__obj.path_pop()
            ## append twice, once the "actual" point, once the moving end
            self.__obj.path_append(ev.x, ev.y, 1)
            self.__obj.path_append(ev.x, ev.y, 1)
        else:
            obj = DrawableFactory.create_drawable("segmented_path", pen = self.__state.pen(), ev=ev)

            if obj:
                self.__obj = obj
                self.__obj.path_append(ev.x, ev.y, 1)

        self.__bus.emit("queue_draw")
        return True

    def on_finish(self, ev):
        """Finish drawing object"""

        obj = self.__obj

        if not obj:
            return False

        obj.path_append(ev.x, ev.y, 0)
        obj.finish()

        self.__bus.emit("add_object", True, obj)
        self.__state.selection().clear()

        self.__obj = None
        self.__bus.emit("queue_draw")
        return True

class WigletCreateGroup(Wiglet):
    """
    Create a groups of objects while drawing

    Basically, by default, objects are grouped automatically
    until you change the mode or press escape.
    """

    def __init__(self, bus, state, grouping = True):
        super().__init__("pan", None)

        self.__bus   = bus
        self.__state = state
        self.__group_obj   = None
        self.__added = False
        self.__grouping = grouping

        # the first command in the group. In case we need to abort before
        # the second element is added, we will undo this command and
        # instead add a single object to the page.
        self.__first_cmd = None

        # the logic is as follows: listen to all events. If we catch an
        # event which is not in the ignore list, we finish the group. This
        # ensures that weird stuff doesn't happen.
        self.__ignore_events = [ "queue_draw", "mouse_move",
                                 "history_append", "add_object",
                                 "draw", "obj_draw",
                                 "left_mouse_click",
                                 "cursor_pos_update",
                                 "cursor_revert", "cursor_set",
                                 "update_win_size",
                                 "mouse_release" ]

        bus.on("toggle_grouping",  self.toggle_grouping, priority = 0)

        if self.__grouping:
            self.start_grouping()

    def toggle_grouping(self):
        """Toggle automatic grouping of objects"""

        self.__grouping = not self.__grouping

        if self.__grouping:
            self.start_grouping()
        else:
            self.end_grouping()

        return True

    def start_grouping(self, mode = None): # pylint: disable=unused-argument
        """Start automatic grouping of objects"""

        if self.__group_obj:
            raise ValueError("Group object already exists")

        self.__group_obj = DrawableGroup()
        self.__added = False
        self.__bus.on("add_object", self.add_object, priority = 99)
        self.__bus.on("escape",     self.end_grouping, priority = 99)
        self.__bus.on("clear_page", self.end_grouping, priority = 200)
        self.__bus.on("mode_set",   self.end_grouping, priority = 200)
        self.__bus.on("*",          self.abort)
        self.__bus.off("toggle_grouping",  self.start_grouping)
        return True

    def abort(self, event, *args, **kwargs):
        """Abort grouping if event is not in the ignore list"""
        if event in self.__ignore_events:
            return False
        log.debug("event: {event} %s %s, aborting grouping", args, kwargs)
        self.end_grouping()
        return False

    def end_grouping(self, mode = None): # pylint: disable=unused-argument
        """End automatic grouping of objects"""

        if not self.__group_obj:
            log.warning("end_grouping: no group object")

        self.__bus.off("add_object", self.add_object)
        self.__bus.off("escape",     self.end_grouping)
        self.__bus.off("clear_page", self.end_grouping)
        self.__bus.off("mode_set",   self.end_grouping)
        self.__bus.off("*",          self.abort)

        n = self.__group_obj.length()

        if n == 1:
            page = self.__state.current_page()
            obj = self.__group_obj.objects[0]
            #cmd1 = RemoveCommand([ self.__group_obj ], page.layer().objects())
            self.__bus.emit("history_undo_cmd", True, self.__first_cmd)
            cmd2 = AddCommand([ obj ], page.layer().objects())
            #cmd = CommandGroup([ cmd1, cmd2 ])
            self.__bus.emit("history_append", True, cmd2)

        self.__group_obj = None

        if self.__grouping:
            self.start_grouping()

        return True

    def add_object(self, obj):
        """Add object to the group"""

        mode = self.__state.mode()

        if not mode == "draw":
            return False

        if not self.__group_obj:
            return False

        if obj.type != "path":
            log.warning("object of type %s cannot be added to automatic path group",
                        obj.type)
            return False

        if not self.__added:
            page = self.__state.current_page()
            cmd1 = AddCommand([ self.__group_obj ], page.layer().objects())
            cmd2 = AddToGroupCommand(self.__group_obj, obj)
            cmd  = CommandGroup([ cmd1, cmd2 ])
            self.__bus.emit("history_append", True, cmd)
            self.__first_cmd = cmd
            self.__added = True
        else:
            cmd = AddToGroupCommand(self.__group_obj, obj)
            self.__bus.emit("history_append", True, cmd)

        return True

class WigletCreateObject(Wiglet):
    """Create object when clicked"""

    def __init__(self, bus, state):
        super().__init__("pan", None)

        self.__bus   = bus
        self.__state = state
        self.__obj   = None
        bus.on("left_mouse_click", self.on_click,   priority = 0)

    def draw_obj(self, cr, state): # pylint: disable=unused-argument
        """Draw the object currently being created"""
        if not self.__obj:
            return False
        self.__obj.draw(cr)
        return True

    def on_click(self, ev):
        """Start drawing"""

        if ev.ctrl() or ev.alt():
            return False

        mode = self.__state.mode()

        if ev.shift() and not ev.ctrl():
            mode = "text"

        if mode not in [ "draw", "shape", "rectangle", "circle" ]:
            return False

        log.debug("WigletCreateObject: creating a new object at %s, %s pressure %s",
                int(ev.x), int(ev.y), int(ev.pressure() * 1000))
        obj = DrawableFactory.create_drawable(mode, pen = self.__state.pen(), ev=ev)

        if obj:
            self.__obj = obj
            self.__bus.on("mouse_move",       self.on_move,    priority = 99)
            self.__bus.on("mouse_release",    self.on_release, priority = 99)
            self.__bus.on("obj_draw",         self.draw_obj,   priority = 99)
        else:
            log.debug("No object created for mode %s", mode)
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
            log.debug("finishing path / shape")
            obj.path_append(ev.x, ev.y, 0)
            obj.finish()
            # remove paths that are too small
            if len(obj.coords) < 3:
                obj = None

        # remove objects that are too small
        if obj:
            bb = obj.bbox()
            if bb and obj.type in [ "rectangle", "box", "circle" ] and bb[2] == 0 and bb[3] == 0:
                obj = None

        if obj:
            self.__bus.emit("add_object", True, obj)

            if self.__obj.type == "text":
                raise ValueError("Text object should not be finished here")
            self.__state.selection().clear()

        self.__obj = None
        self.__bus.off("mouse_move",    self.on_move)
        self.__bus.off("mouse_release", self.on_release)
        self.__bus.off("obj_draw",      self.draw_obj)
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

        self.__bus.emit_once("remove_objects", [ hover_obj ], clear_selection = True)
        self.__bus.emit_once("cursor_revert")
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

class WigletZoom(Wiglet):
    """Zoom in and out"""

    def __init__(self, bus):
        super().__init__("zoom", None)

        self.__bus   = bus
        self.__wsize = (100, 100)
        self.__start_pos = None
        self.__zoom_tool = None

        # listeners that are active when zooming
        self.__active_listeners = {
                "mouse_release": { "listener": self.on_release, "priority": 99},
                "mouse_move": { "listener": self.on_move, "priority": 99},
                "obj_draw": { "listener": self.draw},
                }

        listeners = {
            "zoom_reset": { "listener":  self.zoom_reset, "priority": 99},
            "zoom_in": { "listener":  self.zoom_in, "priority": 99},
            "zoom_out": { "listener":  self.zoom_out, "priority": 99},
            "update_win_size": { "listener": self.update_size},
            "left_mouse_click": { "listener": self.on_click, "priority": 1},
        }

        bus_listeners_on(bus, listeners)

    def draw(self, cr, state):
        """draw the widget"""
        if not self.__zoom_tool:
            return
        self.__zoom_tool.draw(cr, state)

    def on_click(self, ev):
        """handle the click event"""

        if ev.mode() != "zoom":
            return False
        x, y, = ev.x, ev.y
        self.__zoom_tool = SelectionTool([ (x, y), (x + 1, y + 1) ])

        bus_listeners_on(self.__bus, self.__active_listeners)
        log.debug("zooming in or out")
        return True

    def on_release(self, ev):
        """handle the release event"""
        bus_listeners_off(self.__bus, self.__active_listeners)

        bb = self.__zoom_tool.bbox()
        x, y, w, h = bb

        self.__bus.emit_once("page_zoom_reset")
        self.__bus.emit_once("page_translate", (-x, -y))

        z1 = self.__wsize[0] / w
        z2 = self.__wsize[1] / h
        zoom = min(z1, z2, 14)
        log.debug("zooming to %s", zoom)
        self.__bus.emit_once("page_zoom",
                            (0, 0), zoom)
        self.__bus.emit("force_redraw")
        return True

    def on_move(self, ev):
        """handle the move event"""

        if not self.__start_pos:
            self.__start_pos = ev.pos()
            return False

        x, y = ev.x, ev.y
        obj = self.__zoom_tool
        obj.update(x, y, ev.pressure())
        self.__bus.emit("queue_draw")
        return True

    def update_size(self, width, height):
        """update the size of the widget"""
        self.__wsize = (width, height)

    def zoom_reset(self):
        """Reset zoom to 100%"""

        self.__bus.emit_once("page_zoom_reset")
        self.__bus.emit("force_redraw")
        return True

    def zoom_out(self):
        """Zoom out"""

        pos = (self.__wsize[0]/2, self.__wsize[1]/2)
        self.__bus.emit_once("page_zoom", pos, 0.9)
        self.__bus.emit("force_redraw")
        return True

    def zoom_in(self):
        """Zoom in"""

        pos = (self.__wsize[0]/2, self.__wsize[1]/2)
        self.__bus.emit_once("page_zoom", pos, 1.1)
        self.__bus.emit("force_redraw")
        return True
