"""
This module contains the Page class, which is a container for Layers,
and Layer class, which actually handles the objects.
"""

from math import radians                                             # <remove>
import logging                                                   # <remove>
from .drawable import Drawable # <remove>
from .drawable_group import SelectionObject # <remove>
from .commands import GroupObjectCommand, UngroupObjectCommand, RemoveCommand     # <remove>
from .commands import ClipCommand, UnClipCommand, SetColorCommand, SetFontCommand # <remove>
from .commands import SetTransparencyCommand, SetLineWidthCommand, ChangeStrokeCommand # <remove>
from .commands import ToggleFillCommand, RotateCommand, MoveCommand, ZStackCommand # <remove>
from .commands import TransmuteCommand, AddCommand, CommandGroup, InsertPageCommand # <remove>
from .commands import DeletePageCommand, DeleteLayerCommand, FlushCommand # <remove>
from .drawer import Drawer                                           # <remove>
log = logging.getLogger(__name__)                                # <remove>

class LayerSelectionPropertyHandler:
    """
    Base class for handling properties of selected objects.
    """
    def __init__(self, layer, bus):
        self.__layer = layer
        self.__bus   = bus

    def selection_color_set(self, color):
        """Set the color of the selected objects."""
        log.debug("setting color selection")
        if not self.__layer.selection().is_empty():
            cmd = SetColorCommand(self.__layer.selection(), color)
            self.__layer.selection().modified(True)
            self.__bus.emit("history_append", True, cmd)

    def selection_font_set(self, font_description):
        """Set the font of the selected objects."""
        if not self.__layer.selection().is_empty():
            cmd = SetFontCommand(self.__layer.selection(), font_description)
            self.__layer.selection().modified(True)
            self.__bus.emit("history_append", True, cmd)

    def selection_set_transparency(self, transparency):
        """Set the line width of the selected objects."""
        if not self.__layer.selection().is_empty():
            cmd = SetTransparencyCommand(self.__layer.selection(), transparency)
            self.__layer.selection().modified(True)
            self.__bus.emit("history_append", True, cmd)

    def selection_set_line_width(self, width):
        """Set the line width of the selected objects."""
        if not self.__layer.selection().is_empty():
            cmd = SetLineWidthCommand(self.__layer.selection(), width)
            self.__layer.selection().modified(True)
            self.__bus.emit("history_append", True, cmd)

    def selection_change_stroke(self, direction):
        """Change the stroke size of the selected objects."""
        if not self.__layer.selection().is_empty():
            cmd = ChangeStrokeCommand(self.__layer.selection(), direction)
            self.__layer.selection().modified(True)
            self.__bus.emit("history_append", True, cmd)

  # XXX! this is not implemented
    def selection_apply_pen(self):
        """Apply the pen to the selected objects."""
  #     if not self.__layer.selection().is_empty():
  #         pen = self.__state.pen()
  #         self.__history.append(SetPenCommand(self.__layer.selection(), pen))

    def selection_fill(self):
        """Toggle the fill of the selected objects."""
        if not self.__layer.selection().is_empty():
            cmd = ToggleFillCommand(self.__layer.selection())
            self.__layer.selection().modified(True)
            self.__bus.emit("history_append", True, cmd)


class LayerEventHandler:
    """
    Base class for handling events on a layer.
    """
    def __init__(self, layer, bus):
        self.__layer = layer
        self.__ph    = LayerSelectionPropertyHandler(layer, bus)
        self.__bus   = bus

        # Dictionary containing the event signal name, listener, and priority (if given)
        self.__event_listeners = {
            "selection_group": {"listener": self.selection_group},
            "selection_ungroup": {"listener": self.selection_ungroup},
            "selection_delete": {"listener": self.selection_delete},
            "selection_clip": {"listener": self.selection_clip},
            "selection_unclip": {"listener": self.selection_unclip},
            "rotate_selection": {"listener": self.rotate_selection},
            "move_selection": {"listener": self.move_selection},
            "selection_zmove": {"listener": self.selection_zmove},
            "transmute_selection": {"listener": self.transmute_selection},
            "remove_objects": {"listener": self.remove_objects},
            "set_selection": {"listener": self.selection_set},
            "add_object": {"listener": self.add_object, "priority": 1},
            "clear_page": {"listener": self.clear, "priority": 9},
            "flush_selection": {"listener": self.flush_selection},
            "set_color": {"listener": self.__ph.selection_color_set},
            "set_font": {"listener": self.__ph.selection_font_set},
            "set_transparency": {"listener": self.__ph.selection_set_transparency},
            "set_line_width": {"listener": self.__ph.selection_set_line_width},
            "stroke_change": {"listener": self.__ph.selection_change_stroke},
            # "apply_pen_to_selection": {"listener": self.__ph.selection_apply_pen},
            "selection_fill": {"listener": self.__ph.selection_fill}
        }

        self.activate()


    def activate(self):
        """Activate the event handler."""

        for event, params in self.__event_listeners.items():
            if "priority" in params:
                self.__bus.on(event, params["listener"], 
                              priority=params["priority"])
            else:
                self.__bus.on(event, params["listener"])

    def deactivate(self):
        """Deactivate the event handler."""

        for event, params in self.__event_listeners.items():
            self.__bus.off(event, params["listener"])


    def selection_group(self):
        """Group selected objects."""
        if self.__layer.selection().n() < 2:
            return
        log.debug("Grouping n=%s objects", self.__layer.selection().n())
        cmd = GroupObjectCommand(self.__layer.selection().objects,
                                                 self.__layer.objects(),
                                                 selection_object=self.__layer.selection())
        self.__bus.emit("history_append", True, cmd)

    def selection_ungroup(self):
        """Ungroup selected objects."""
        if self.__layer.selection().is_empty():
            return
        cmd = UngroupObjectCommand(self.__layer.selection().objects,
                                                   self.__layer.objects(),
                                                   selection_object=self.__layer.selection())
        self.__bus.emit("history_append", True, cmd)

    def selection_delete(self):
        """Delete selected objects."""

        if self.__layer.selection().objects:
            cmd = RemoveCommand(self.__layer.selection().objects,
                                                self.__layer.objects())
            self.__bus.emit("history_append", True, cmd)
            self.__layer.selection().clear()

    def selection_clip(self):
        """Clip the selected objects."""
        if self.__layer.selection().is_empty():
            return

        obj = self.__layer.selection().objects

        if len(obj) < 2:
            log.warning("need at least two objects to clip")
            return

        log.debug("object: %s", obj[-1].type)
        if not obj[-1].type in [ "rectangle", "shape", "circle" ]:
            log.warning("Need a shape, rectangle or circle to clip, not %s", obj[-1].type)
            return

        cmd = ClipCommand(obj[-1], obj[:-1],
                                       self.__layer.objects(),
                                       selection_object=self.__layer.selection())
        self.__bus.emit("history_append", True, cmd)

    def selection_unclip(self):
        """Unclip the selected objects."""
        if self.__layer.selection().is_empty():
            return
        cmd = UnClipCommand(self.__layer.selection().objects,
                                         self.__layer.objects(),
                                         selection_object=self.__layer.selection())
        self.__bus.emit("history_append", True, cmd)

    def rotate_selection(self, angle):
        """Rotate the selected objects by the given angle (degrees)."""
        if self.__layer.selection().is_empty():
            return

        obj = self.__layer.selection()
        event_obj = RotateCommand(obj, angle=radians(angle))
        event_obj.event_finish()
        self.__bus.emit("history_append", True, event_obj)

    def move_selection(self, dx, dy):
        """Move the selected objects by the given amount."""
        if self.__layer.selection().is_empty():
            return

        obj = self.__layer.selection().copy()
        event_obj = MoveCommand(obj, (0, 0))
        event_obj.event_update(dx, dy)
        self.__bus.emit("history_append", True, event_obj)

    def selection_zmove(self, operation):
        """move the selected objects long the z-axis."""
        if self.__layer.selection().is_empty() or not operation:
            return

        cmd = ZStackCommand(self.__layer.selection().objects,
                                            self.__layer.objects(), operation)
        self.__bus.emit("history_append", True, cmd)

    def transmute(self, objects, mode):
        """
        Transmute the object to the given mode.

        This is a dangerous operation, because we are replacing the objects
        and we need to make sure that the old objects are removed from the
        list of objects, selections etc.

        Args:
            objects (list): The list of objects.
            mode (str): The mode to transmute to.
        """

    def transmute_selection(self, mode):
        """
        Transmute the selected objects to the given mode.

        Args:
            mode (str): The mode to transmute to.
        """
        if self.__layer.selection().is_empty():
            return
        objects = self.__layer.selection().objects
        cmd = TransmuteCommand(objects=objects,
                               stack=self.__layer.objects(),
                               new_type=mode,
                               selection_objects=self.__layer.selection().objects)
        self.__bus.emit("history_append", True, cmd)

    def remove_objects(self, objects, clear_selection = False):
        """Remove an object from the list of objects."""
        cmd = RemoveCommand(objects, self.__layer.objects())
        self.__bus.emit("history_append", True, cmd)
        if clear_selection:
            self.__layer.selection().clear()

    def selection_set(self, what):
        """Dispatch to the correct selection function"""

        if not what:
            return False

        if what == "all":
            self.__layer.selection().all()
        elif what == "next":
            self.__layer.selection().next()
        elif what == "previous":
            self.__layer.selection().previous()
        elif what == "reverse":
            self.__layer.selection().reverse()
        elif what == "nothing":
            self.__layer.selection().clear()
        else:
            log.debug("Setting selection to %s", what)
            self.__layer.selection().set(what)

        self.__bus.emit("mode_set", False, "move")
        return True

    def add_object(self, obj):
        """Add an object to the list of objects."""

        log.debug("Adding object %s", obj)

        if obj in self.__layer.objects():
            log.warning("object %s already in list", obj)
            return None
        if not isinstance(obj, Drawable):
            raise ValueError("Only Drawables can be added to the stack")

        cmd = AddCommand([obj], self.__layer.objects())
        self.__bus.emit("history_append", True, cmd)

        return obj

    def clear(self):
        """Clear the list of objects."""
        self.__layer.selection().clear()

    def flush_selection(self, flush_direction):
        """Flush the selection in the given direction."""
        if self.__layer.selection().n() < 2:
            return

        cmd = FlushCommand(self.__layer.selection().objects, flush_direction)
        self.__bus.emit("history_append", True, cmd)



class Layer:
    """
    A layer is a container for objects.
    """
    def __init__(self):
        self.__objects = []
        self.__selection = SelectionObject(self.__objects)
        self.__bus = None
        self.__layer_handler = None

    def objects(self, objects = None):
        """Return or set the list of objects on the layer."""
        if objects:
            self.__objects = objects
            self.__selection = SelectionObject(self.__objects)
        return self.__objects

    def objects_import(self, object_list):
        """Import objects from a dict"""
        self.objects([ Drawable.from_dict(d) for d in object_list ] or [ ])

    def selection(self):
        """Return the selection object."""
        return self.__selection

    def export(self):
        """Exports the layer as a dict"""
        return [ obj.to_dict() for obj in self.__objects ]

    def activate(self, bus):
        """Activate the layer."""
        log.debug("layer %s activating", self)
        bus.emitMult("layer_deactivate")
        bus.on("layer_deactivate", self.deactivate)
        self.__bus = bus
        self.__layer_handler = LayerEventHandler(self, bus)

    def deactivate(self):
        """Deactivate the layer."""
        log.debug("layer %s dectivating", self)

        self.__bus.off("layer_deactivate", self.deactivate)
        self.__layer_handler.deactivate()
        self.__layer_handler = None
        self.__bus = None

## ---------------------------------------------------------------------
##
##       Page class for handling entire pages with multiple layers
##
## ---------------------------------------------------------------------

class PageChain:
    """Base class for Page. Handles the linked list structure of pages."""

    def __init__(self, prev = None, next_p = None):
        self.__prev = prev
        self.__next = next_p

    def next(self, create = True):
        """
        Return the next page.

        If create is True, create a new page if it doesn't exist.
        """
        if not self.__next and create:
            log.debug("Creating new page")
            self.__next = Page(self)
        return self.__next

    def prev(self):
        """Return the previous page."""
        return self.__prev or self # can't go beyond first page

    def next_set(self, page):
        """Set the next page."""
        self.__next = page

    def prev_set(self, page):
        """Set the previous page."""
        self.__prev = page

    def insert(self):
        """Insert a new page after the current page."""
        # we are already the last page
        cmd = InsertPageCommand(self)
        ret = self.__next
        return ret, cmd

    def delete(self):
        """Delete the page and create links between prev and next pages."""
        if not self.__prev and not self.__next:
            log.debug("only one page remaining")
            return self, None

        cmd = DeletePageCommand(self)
        ret = self.__prev or self.__next

        return ret, cmd


class Page(PageChain):
    """
    A page is a container for layers.

    It serves as an interface between layers and whatever wants to
    manipulate objects or selection on a layer by choosing the current
    layer and managing layers.
    """
    def __init__(self, prev = None, layers = None):
        super().__init__(prev)

        self.__layers = [ layers or Layer() ]
        self.__current_layer = 0
        self.__translate = None
        self.__drawer = Drawer()
        self.__bus = None

    def activate(self, bus):
        """Activate the page so that it responds to the bus"""
        # the active page responds to signals requesting layer manipulation
        log.debug("page %s activating", self)

        # shout out to the previous current page to get lost
        bus.emitOnce("page_deactivate")
        bus.on("page_deactivate", self.deactivate)
        bus.on("next_layer", self.next_layer)
        bus.on("prev_layer", self.prev_layer)
        bus.on("delete_layer", self.delete_layer_cmd)
        bus.on("clear_page", self.clear, priority = 8)

        self.layer().activate(bus)

        self.__bus = bus

    def deactivate(self):
        """Stop reacting to the bus"""
        bus = self.__bus

        if bus is None:
            return

        log.debug("page %s deactivating", self)
        bus.off("page_deactivate", self.deactivate)
        bus.off("next_layer", self.next_layer)
        bus.off("prev_layer", self.prev_layer)
        bus.off("delete_layer", self.delete_layer_cmd)
        bus.off("clear_page", self.clear)

        self.layer().deactivate()

        self.__bus = None

    def objects_all_layers(self):
        """Return all objects on all layers."""
        objects = [ obj for layer in self.__layers for obj in layer.objects() ]
        return objects

    def number_of_layers(self):
        """Return the number of layers."""
        return len(self.__layers)

    def next_layer(self):
        """Switch to the next layer."""
        self.__current_layer += 1
        if self.__current_layer == len(self.__layers):
            self.__layers.append(Layer())
            log.debug("appending a new layer, total now %s", len(self.__layers))
        self.__layers[self.__current_layer].selection().all()
        if self.__bus:
            self.__layers[self.__current_layer].activate(self.__bus)
        return self.__current_layer

    def prev_layer(self):
        """Switch to the previous layer."""
        self.__current_layer = max(0, self.__current_layer - 1)
        self.__layers[self.__current_layer].selection().all()
        if self.__bus:
            self.__layers[self.__current_layer].activate(self.__bus)
        return self.__current_layer

    def layer(self, new_layer = None, pos = None):
        """
        Get or insert the current layer.

        Arguments:
        new_layer -- if not None, insert new_layer and set
                     the current layer to new_layer.
        pos -- if not None, insert a new layer at pos.
        """
        if new_layer is not None:
            if pos is not None and pos < len(self.__layers):
                self.__layers.insert(pos, new_layer)
                self.__current_layer = pos
            else:
                self.__layers.append(new_layer)
                self.__current_layer = len(self.__layers) - 1
            if self.__bus:
                self.__layers[self.__current_layer].activate(self.__bus)

        if pos is not None:
            return self.__layers[pos]

        return self.__layers[self.__current_layer]

    def layer_no(self, layer_no = None):
        """Get or set the current layer number."""
        if layer_no is None:
            return self.__current_layer

        layer_no = max(0, layer_no)

        if layer_no >= len(self.__layers):
            self.__layers.append(Layer())

        self.__current_layer = layer_no
        if self.__bus:
            self.__layers[self.__current_layer].activate(self.__bus)
        return self.__current_layer

    def delete_layer_cmd(self):
        """Delete the current layer."""
        # the "logic", if you can call it thusly, is as follows.
        # the delete_layer is actually called by the DeleteLayerCommand.
        # Therefore, we need a wrapper around DeleteLayerCommand that does
        # not actually delete the layer.

        cmd = DeleteLayerCommand(self, self.layer_no())
        self.__bus.emit("history_append", True, cmd)

    def delete_layer(self, layer_no = None):
        """Delete the current layer."""
        log.debug("deleting layer %s", layer_no)

        if len(self.__layers) == 1:
            return None, None

        if layer_no is None or layer_no < 0 or layer_no >= len(self.__layers):
            layer_no = self.__current_layer

        layer = self.__layers[layer_no]
        pos   = layer_no

        del self.__layers[layer_no]

        # make sure layer is within boundaries
        self.__current_layer = max(0, layer_no - 1)
        self.__current_layer = min(self.__current_layer,
                                   len(self.__layers) - 1)

        if self.__bus:
            self.__layers[self.__current_layer].activate(self.__bus)
        return layer, pos

    def translate(self, new_val = None):
        """Get or set the translate"""
        if new_val is not None:
            self.__translate = new_val
        return self.__translate

    def export(self):
        """Exports the page with all layers as a dict"""
        layers = [ l.export() for l in self.__layers ]
        ret = {
                 "layers": layers,
                 "translate": self.translate(),
                 "cur_layer": self.__current_layer
              }
        return ret

    def import_page(self, page_dict):
        """Imports a dict to self"""
        log.debug("importing pages")
        self.__translate = page_dict.get("translate")
        if "objects" in page_dict:
            self.layer().objects(page_dict["objects"])
        elif "layers" in page_dict:
            log.debug('%s layers found', len(page_dict["layers"]))
            log.debug("however, we only have %s layers", len(self.__layers))
            log.debug('creating %s new layers', len(page_dict["layers"]) - len(self.__layers))
            self.__current_layer = 0
            for _ in range(len(page_dict["layers"]) - len(self.__layers)):
                self.next_layer()
            self.__current_layer = 0
            for l_list in page_dict["layers"]:
                layer = self.__layers[self.__current_layer]
                layer.objects_import(l_list)
                self.__current_layer += 1

        cl = page_dict.get("cur_layer")
        self.__current_layer = cl if cl is not None else 0
        if self.__bus:
            self.__layers[self.__current_layer].activate(self.__bus)

    def clear(self):
        """
        Remove all objects from all layers.

        Returns a CommandGroup object that can be used to undo the
        operation.
        """
        ret_commands = []
        for layer in self.__layers:
            cmd = RemoveCommand(layer.objects()[:], layer.objects())
            ret_commands.append(cmd)
        cmd = CommandGroup(ret_commands[::-1])
        self.__bus.emit("history_append", True, cmd)

    def draw(self, cr, state, force_redraw = False):
        """Draw the objects on the page."""

        self.__drawer.draw(cr, self, state, force_redraw)
