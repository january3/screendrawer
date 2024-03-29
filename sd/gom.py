from .commands import *                                              # <remove>
from .drawable import SelectionObject, DrawableGroup, SelectionTool  # <remove>
from .drawable import DrawableFactory                                # <remove>
from .page import Page                                               # <remove>
from .utils import sort_by_stack                                     # <remove>


## ---------------------------------------------------------------------

class GraphicsObjectManager:
    """
    Class to manage graphics objects.

    Attributes:
        _objects (list): The list of objects.
    """

    def __init__(self, app):

        # private attr
        self.__app = app
       #self._objects    = []
        self.__history    = []
        self.__redo_stack = []
        self.page_set(Page())

    def page_set(self, page):
        self.__page = page
        self.__selection = self.__page.selection()

    def next_page(self):
        """Go to the next page."""
        self.page_set(self.__page.next(create = True))

    def prev_page(self):
        """Go to the prev page."""
        self.page_set(self.__page.prev())

    def delete_page(self):
        """Delete the current page."""
        self.page_set(self.__page.delete())

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
            for i in range(n - cur_n):
                self.next_page()
        else:
            for i in range(cur_n - n):
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

    def objects(self):
        """Return the list of objects."""
        return self.__page.objects()

    def selection(self):
        """Return the selection object."""
        return self.__selection

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
        self.__history.append(TransmuteCommand(objects=objects, stack=self.__page.objects(), 
                                               new_type=mode, 
                                               selection_objects=self.__selection.objects, 
                                               page = self.__page))

    def transmute_selection(self, mode):
        """
        Transmute the selected objects to the given mode.

        Args:
            mode ( str ): The mode to transmute to.
        """
        if self.__selection.is_empty():
            return
        self.transmute(self.__selection.objects, mode)

    def set_objects(self, objects):
        """Set the list of objects."""
        ## no undo
        print("GOM: setting n=", len(objects), "objects")
        self.__page.objects(objects)

    def set_pages(self, pages):
        self.__page = Page()
        self.__page.objects(pages[0]['objects'])
        for p in pages[1:]:
            self.__page = self.__page.next()
            self.__page.objects(p['objects'])
        self.__selection = self.__page.selection()

    def add_object(self, obj):
        """Add an object to the list of objects."""
        self.__history.append(AddCommand(obj, self.__page.objects(), page=self.__page))
 
    def export_pages(self):
        """Export all pages."""
        # XXX
        # find the first page
        p = self.__page
        while p.prev() != p:
            p = p.prev()

        # create a list of pages for all pages
        pages = [ ]
        while p:
            objects = [ obj.to_dict() for obj in p.objects() ]
            pages.append({ "objects": objects })
            p = p.next(create = False)
        return pages

    def export_objects(self):
        """Just the objects from the current page."""
        objects = [ obj.to_dict() for obj in self.__page.objects() ]
        return objects

    def kill_object(self, obj):
        """Directly remove an object from the list of objects."""
        self.__page.kill_object(obj)

    def selected_objects(self):
        """Return the selected objects."""
        return self.__selection.objects

    def remove_selection(self):
        """Remove the selected objects from the list of objects."""
        if self.__selection.is_empty():
            return
        self.__history.append(RemoveCommand(self.__selection.objects, self.__page.objects(), page=self.__page))
        self.__selection.clear()

    def remove_objects(self, objects, clear_selection = False):
        """Remove an object from the list of objects."""
        self.__history.append(RemoveCommand(objects, self.__page.objects(), page=self.__page))
        if clear_selection:
            self.__selection.clear()

    def remove_all(self):
        """Clear the list of objects."""
        self.__history.append(RemoveCommand(self.__page.objects()[:], self.__page.objects(), page = self.__page))

    def command_append(self, command_list):
        """Append a group of commands to the history."""
        ## append in reverse order
        self.__history.append(CommandGroup(command_list[::-1]))

    def selection_group(self):
        """Group selected objects."""
        if self.__selection.n() < 2:
            return
        print("Grouping", self.__selection.n(), "objects")
        self.__history.append(GroupObjectCommand(self.__selection.objects, self.__page.objects(), 
                                                 selection_object=self.__selection, page=self.__page))

    def selection_ungroup(self):
        """Ungroup selected objects."""
        if self.__selection.is_empty():
            return
        self.__history.append(UngroupObjectCommand(self.__selection.objects, self.__page.objects(),
                                                   selection_object=self.__selection, page=self.__page))

    def select_reverse(self):
        """Reverse the selection."""
        self.__selection.reverse()
        self.__app.dm.mode("move")

    def select_all(self):
        """Select all objects."""
        if not self.__page.objects():
            return

        self.__selection.all()
        self.__app.dm.mode("move")

    def selection_delete(self):
        """Delete selected objects."""
        #self.__page.selection_delete()
        if self.__selection.objects:
            self.__history.append(RemoveCommand(self.__selection.objects, 
                                                self.__page.objects(), page=self.__page))
            self.__selection.clear()

    def select_next_object(self):
        """Select the next object."""
        self.__selection.next()

    def select_previous_object(self):
        """Select the previous object."""
        self.__selection.previous()

    def selection_fill(self):
        """Fill the selected object."""
        # XXX gom should not call dm directly
        # this code should be gone!
        color = self.__app.dm.pen().color
        for obj in self.__selection.objects:
            obj.fill(color)

    def selection_color_set(self, color):
        """Set the color of the selected objects."""
        if not self.__selection.is_empty():
            self.__history.append(SetColorCommand(self.__selection, color))

    def selection_font_set(self, font_description):
        """Set the font of the selected objects."""
        # XXX: no undo!
        for obj in self.__selection.objects:
            obj.pen.font_set_from_description(font_description)

    def selection_apply_pen(self, pen):
        """Apply the pen to the selected objects."""
        if not self.__selection.is_empty():
            # self._history.append(SetColorCommand(self.selection, pen.color))
            # self._history.append(SetLWCommand(self.selection, pen.color))
            for obj in self.__selection.objects:
                obj.set_pen(pen)

    def redo(self):
        """Redo the last action."""
        print("Redo stack, size is", len(self.__redo_stack))
        if self.__redo_stack:
            command = self.__redo_stack.pop()
            page = command.redo()
            self.__history.append(command)

            # switch to the relevant page
            if page:
                self.page_set(page)

    def undo(self):
        """Undo the last action."""
        print("Undo, history size is", len(self.__history))
        if self.__history:
            command = self.__history.pop()
            page = command.undo()
            self.__redo_stack.append(command)

            # switch to the relevant page
            if page:
                self.page_set(page)

    def move_obj(self, obj, dx, dy):
        """Move the object by the given amount."""
        eventObj = MoveCommand(obj, (0, 0), page=self.__page)
        eventObj.event_update(dx, dy)
        self.__history.append(eventObj)

    def move_selection(self, dx, dy):
        """Move the selected objects by the given amount."""
        if self.__selection.is_empty():
            return
        self.move_obj(self.__selection.copy(), dx, dy)

    def rotate_obj(self, obj, angle):
        """Rotate the object by the given angle (degrees)."""
        print("rotating by", angle)
        eventObj = RotateCommand(obj, angle=math.radians(angle), page = self.__page)
        eventObj.event_finish()
        self.__history.append(eventObj)

    def rotate_selection(self, angle):
        """Rotate the selected objects by the given angle (degrees)."""
        if self.__selection.is_empty():
            return
        self.__rotate_obj(self.__selection, angle)

    def selection_zmove(self, operation):
        """move the selected objects long the z-axis."""
        if self.__selection.is_empty():
            return
        self.__history.append(ZStackCommand(self.__selection.objects, 
                                            self.__page.objects(), operation, page=self.__page))


