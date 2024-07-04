from math import radians                                             # <remove>
from .commands import *                                              # <remove>
from .drawable_group import DrawableGroup                            # <remove>
from .drawable import Drawable                                       # <remove>
from .page import Page                                               # <remove>
from .utils import sort_by_stack                                     # <remove>
from .history import History                                         # <remove>
import logging                                                   # <remove>
log = logging.getLogger(__name__)                                # <remove>


## ---------------------------------------------------------------------

class GraphicsObjectManager:
    """
    Class to manage graphics objects.

    Attributes:
        _objects (list): The list of objects.
    """

    def __init__(self, bus):

        # private attr
        self.__bus        = bus
        self.__page = None
        self.page_set(Page())
        self.__add_bus_listeners()

    def __add_bus_listeners(self):
        """Add listeners to the bus."""
        self.__bus.on("add_object", self.add_object, priority = 1)
        self.__bus.on("clear_page", self.clear, priority = 9)
        self.__bus.on("clear_page", self.remove_all, priority = 8)
        self.__bus.on("rotate_selection", self.rotate_selection)
        self.__bus.on("selection_zmove", self.selection_zmove)
        self.__bus.on("move_selection", self.move_selection)
        self.__bus.on("selection_fill", self.selection_fill)
        self.__bus.on("transmute_selection", self.transmute_selection)
        self.__bus.on("set_selection", self.select)
        self.__bus.on("selection_clip", self.selection_clip)
        self.__bus.on("selection_unclip", self.selection_unclip)
        self.__bus.on("selection_group", self.selection_group)
        self.__bus.on("selection_ungroup", self.selection_ungroup)
        self.__bus.on("selection_delete", self.selection_delete)
        self.__bus.on("next_page", self.next_page)
        self.__bus.on("prev_page", self.prev_page)
        self.__bus.on("insert_page", self.insert_page)
        self.__bus.on("page_set", self.page_set)
        self.__bus.on("delete_page", self.delete_page)
        self.__bus.on("next_layer", self.next_layer)
        self.__bus.on("prev_layer", self.prev_layer)
        self.__bus.on("delete_layer", self.delete_layer)
        self.__bus.on("apply_pen_to_selection", self.selection_apply_pen)
        self.__bus.on("set_color", self.selection_color_set)
        self.__bus.on("set_font", self.selection_font_set)
        self.__bus.on("set_line_width", self.selection_set_line_width)
        self.__bus.on("set_transparency", self.selection_set_transparency)
        self.__bus.on("stroke_change", self.selection_change_stroke)

    def clear(self):
        """Clear the list of objects."""
        self.selection().clear()

    def page_set(self, page):
        """Set the current page."""
        self.__bus.emit("page_changed", False, page)
        self.__page = page

    def next_page(self):
        """Go to the next page."""
        self.page_set(self.__page.next(create = True))

    def insert_page(self):
        """Insert a new page."""
        curpage, cmd = self.__page.insert()
        bus.__emit("history_append", True, cmd)
        self.page_set(curpage)

    def prev_page(self):
        """Go to the prev page."""
        self.page_set(self.__page.prev())

    def delete_page(self):
        """Delete the current page."""
        curpage, cmd = self.__page.delete()
        self.page_set(curpage)
        self.__bus.emit("history_append", True, cmd)

    def page(self):
        """Return the current page."""
        return self.__page

    def set_page_number(self, n):
        """Choose page number n."""
        tot_n = self.number_of_pages()
        if n < 0 or n >= tot_n:
            return
        cur_n = self.current_page_number()

        if n == cur_n:
            return
        if n > cur_n:
            for _ in range(n - cur_n):
                self.next_page()
        else:
            for _ in range(cur_n - n):
                self.prev_page()

    def number_of_pages(self):
        """Return the total number of pages."""
        p = self.start_page()

        n = 1
        while p.next(create = False):
            n += 1
            p = p.next(create = False)
        return n

    def current_page_number(self):
        """Return the current page number."""
        p = self.__page
        n = 0
        while p.prev() != p:
            n += 1
            p = p.prev()
        return n

    def start_page(self):
        """Return the first page."""
        p = self.__page
        while p.prev() != p:
            p = p.prev()
        return p

    def next_layer(self):
        """Go to the next layer."""
        log.debug("creating a new layer")
        self.__page.next_layer()

    def prev_layer(self):
        """Go to the previous layer."""
        self.__page.prev_layer()

    def delete_layer(self):
        """Delete the current layer."""
        #cmd = 
        #self.__page.delete_layer(self.__page.layer_no())
        cmd = DeleteLayerCommand(self.__page, self.__page.layer_no())
        self.__bus.emit("history_append", True, cmd)

    def objects(self):
        """Return the list of objects."""
        return self.__page.objects()

    def selection(self):
        """Return the selection object."""
        return self.__page.selection()

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
        cmd = TransmuteCommand(objects=objects,
                               stack=self.__page.objects(),
                               new_type=mode,
                               selection_objects=self.__page.selection().objects)
        self.__bus.emit("history_append", True, cmd)

    def transmute_selection(self, mode):
        """
        Transmute the selected objects to the given mode.

        Args:
            mode ( str ): The mode to transmute to.
        """
        if self.__page.selection().is_empty():
            return
        self.transmute(self.__page.selection().objects, mode)

    def set_objects(self, objects):
        """Set the list of objects."""
        ## no undo
        log.debug(f"GOM: setting n={len(objects)} objects")
        self.__page.objects(objects)

    def set_pages(self, pages):
        """Set the content of pages."""
        self.__page = Page()
        self.__page.import_page(pages[0])
        for p in pages[1:]:
            self.__page = self.__page.next()
            self.__page.import_page(p)

    def add_object(self, obj):
        """Add an object to the list of objects."""

        log.debug(f"Adding object {obj}")

        if obj in self.__page.objects():
            log.warning(f"object {obj} already in list")
            return None
        if not isinstance(obj, Drawable):
            raise ValueError("Only Drawables can be added to the stack")

        cmd = AddCommand([obj], self.__page.objects())
        self.__bus.emit("history_append", True, cmd)

        return obj

    def get_all_pages(self):
        """Return all pages."""
        p = self.__page
        while p.prev() != p:
            p = p.prev()

        pages = [ ]

        while p:
            pages.append(p)
            p = p.next(create = False)

        return pages

    def export_pages(self):
        """Export all pages."""

        pages = [ p.export() for p in self.get_all_pages() ]
        return pages

    def kill_object(self, obj):
        """Directly remove an object from the list of objects."""
        self.__page.kill_object(obj)

    def selected_objects(self):
        """Return the selected objects."""
        return self.__page.selection().objects

    def remove_selection(self):
        """Remove the selected objects from the list of objects."""
        if self.__page.selection().is_empty():
            return
        cmd = RemoveCommand(self.__page.selection().objects,
                                            self.__page.objects())
        self.__bus.emit("history_append", True, cmd)
        self.__page.selection().clear()

    def remove_objects(self, objects, clear_selection = False):
        """Remove an object from the list of objects."""
        cmd = RemoveCommand(objects, self.__page.objects())
        self.__bus.emit("history_append", True, cmd)
        if clear_selection:
            self.__page.selection().clear()

    def remove_all(self):
        """Clear the list of objects."""
        self.__bus.emit("history_append", True, self.__page.clear())

    def command_append(self, command_list):
        """Append a group of commands to the history."""
        ## append in reverse order!
        cmd = CommandGroup(command_list[::-1])
        self.__bus.emit("history_append", True, cmd)

    def selection_group(self):
        """Group selected objects."""
        if self.__page.selection().n() < 2:
            return
        log.debug(f"Grouping n={self.__page.selection().n()} objects")
        cmd = GroupObjectCommand(self.__page.selection().objects,
                                                 self.__page.objects(),
                                                 selection_object=self.__page.selection(),
                                                 page=self.__page)
        self.__bus.emit("history_append", True, cmd)

    def selection_ungroup(self):
        """Ungroup selected objects."""
        if self.__page.selection().is_empty():
            return
        cmd = UngroupObjectCommand(self.__page.selection().objects,
                                                   self.__page.objects(),
                                                   selection_object=self.__page.selection(),
                                                   page=self.__page)
        self.__bus.emit("history_append", True, cmd)

    def selection_clip(self):
        """Clip the selected objects."""
        page = self.__page
        if page.selection().is_empty():
            return
        obj = page.selection().objects

        if len(obj) < 2:
            log.warning("need at least two objects to clip")
            return

        log.debug(f"object: {obj[-1].type}")
        if not obj[-1].type in [ "rectangle", "shape", "circle" ]:
            log.warning(f"Need a shape, rectangle or circle to clip, not {obj[-1].type}")
            return

        cmd = ClipCommand(obj[-1], obj[:-1],
                                       page.objects(),
                                       selection_object=page.selection(),
                                       page=page)
        self.__bus.emit("history_append", True, cmd)

        log.debug("clipping selection")

    def selection_unclip(self):
        page = self.__page
        """Unclip the selected objects."""
        if page.selection().is_empty():
            return
        log.debug("unclipping selection")
        cmd = UnClipCommand(page.selection().objects,
                                         page.objects(),
                                         selection_object=page.selection(),
                                         page=page)
        self.__bus.emit("history_append", True, cmd)

    def select(self, what):
        """Dispatch to the correct selection function"""

        if not what:
            return False

        if what == "all":
            self.select_all()
        elif what == "next":
            self.select_next_object()
        elif what == "previous":
            self.select_previous_object()
        elif what == "reverse":
            self.select_reverse()
        else:
            log.error(f"Unknown selection command: {what}")

        self.__bus.emit("mode_set", False, "move")
        return True

    def select_reverse(self):
        """Reverse the selection."""
        self.__page.selection().reverse()
        self.__bus.emit("mode_set", False, "move")

    def select_all(self):
        """Select all objects."""
        if not self.__page.objects():
            log.debug("no objects found")
            return

        self.__page.selection().all()
        self.__bus.emit("mode_set", False, "move")

    def selection_delete(self):
        """Delete selected objects."""
        #self.__page.selection_delete()
        if self.__page.selection().objects:
            cmd = RemoveCommand(self.__page.selection().objects,
                                                self.__page.objects())
            self.__bus.emit("history_append", True, cmd)
            self.__page.selection().clear()

    def select_next_object(self):
        """Select the next object."""
        self.__page.selection().next()
        self.__bus.emit("mode_set", False, "move")

    def select_previous_object(self):
        """Select the previous object."""
        self.__page.selection().previous()
        self.__bus.emit("mode_set", False, "move")

    def selection_fill(self):
        """Toggle the fill of the selected objects."""
       #for obj in self.__page.selection().objects:
       #    obj.fill_toggle()
        if not self.__page.selection().is_empty():
            cmd = ToggleFillCommand(self.__page.selection())
            self.__page.selection().modified(True)
            self.__bus.emit("history_append", True, cmd)

    def selection_color_set(self, color):
        """Set the color of the selected objects."""
        if not self.__page.selection().is_empty():
            cmd = SetColorCommand(self.__page.selection(), color)
            self.__page.selection().modified(True)
            self.__bus.emit("history_append", True, cmd)

    def selection_font_set(self, font_description):
        """Set the font of the selected objects."""
        if not self.__page.selection().is_empty():
            cmd = SetFontCommand(self.__page.selection(), font_description)
            self.__page.selection().modified(True)
            self.__bus.emit("history_append", True, cmd)

    def selection_set_transparency(self, transparency):
        """Set the line width of the selected objects."""
        if not self.__page.selection().is_empty():
            cmd = SetTransparencyCommand(self.__page.selection(), transparency)
            self.__page.selection().modified(True)
            self.__bus.emit("history_append", True, cmd)

    def selection_set_line_width(self, width):
        """Set the line width of the selected objects."""
        if not self.__page.selection().is_empty():
            cmd = SetLineWidthCommand(self.__page.selection(), width)
            self.__page.selection().modified(True)
            self.__bus.emit("history_append", True, cmd)

    def selection_change_stroke(self, direction):
        """Change the stroke size of the selected objects."""
        if not self.__page.selection().is_empty():
            cmd = ChangeStrokeCommand(self.__page.selection(), direction)
            self.__page.selection().modified(True)
            self.__bus.emit("history_append", True, cmd)

  # XXX! this is not implemented
    def selection_apply_pen(self):
        """Apply the pen to the selected objects."""
  #     if not self.__page.selection().is_empty():
  #         pen = self.__state.pen()
  #         self.__history.append(SetPenCommand(self.__page.selection(), pen))

    def move_obj(self, obj, dx, dy):
        """Move the object by the given amount."""
        event_obj = MoveCommand(obj, (0, 0))
        event_obj.event_update(dx, dy)
        self.__bus.emit("history_append", True, event_obj)

    def move_selection(self, dx, dy):
        """Move the selected objects by the given amount."""
        if self.__page.selection().is_empty():
            return
        self.move_obj(self.__page.selection().copy(), dx, dy)

    def rotate_obj(self, obj, angle):
        """Rotate the object by the given angle (degrees)."""
        log.debug(f"rotating by {angle}")
        event_obj = RotateCommand(obj, angle=radians(angle))
        event_obj.event_finish()
        self.__bus.emit("history_append", True, event_obj)

    def rotate_selection(self, angle):
        """Rotate the selected objects by the given angle (degrees)."""
        if self.__page.selection().is_empty():
            return
        self.rotate_obj(self.__page.selection(), angle)

    def selection_zmove(self, operation):
        """move the selected objects long the z-axis."""
        if self.__page.selection().is_empty():
            return
        cmd = ZStackCommand(self.__page.selection().objects,
                                            self.__page.objects(), operation, page=self.__page)
        self.__bus.emit("history_append", True, cmd)
