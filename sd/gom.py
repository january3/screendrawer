from .commands import *                                              # <remove>
from .drawable import SelectionObject, DrawableGroup, SelectionTool  # <remove>
from .clipboard import Clipboard                                     # <remove>
from .drawable import DrawableFactory                                # <remove>
from .page import Page                                               # <remove>


## ---------------------------------------------------------------------

class GraphicsObjectManager:
    """
    Class to manage graphics objects.

    Attributes:
        _objects (list): The list of objects.
    """

    def __init__(self, app):
        # public attr
        self.clipboard = Clipboard()

        # private attr
        self.__app = app
        self.__page = Page()
       #self._objects    = []
       #self._history    = []
       #self._redo_stack = []
       #self.selection = SelectionObject(self._objects)


    def objects(self):
        """Return the list of objects."""
        return self.__page.objects()

    def selection(self):
        """Return the selection object."""
        return self.__page.selection()

    def transmute_selection(self, mode):
        """
        Transmute the selected objects to the given mode.

        Args:
            mode ( str ): The mode to transmute to.
        """
        if self.__page.selection().is_empty():
            return
        self.__page.transmute(objects, mode)

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

    def add_object(self, obj):
        """Add an object to the list of objects."""
        self.__page.add_object(obj)

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
            print("next page!")
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
        return self.__page.selection().objects

    def remove_selection(self):
        """Remove the selected objects from the list of objects."""
        self.__page.remove_selection()

    def remove_objects(self, objects, clear_selection = False):
        """Remove an object from the list of objects."""
        self.__page.remove_objects(objects, clear_selection)

    def remove_all(self):
        """Clear the list of objects."""
        self.__page.remove_all()

    def command_append(self, command_list):
        """Append a group of commands to the history."""
        ## append in reverse order
        self.__page.command_append(command_list)

    def selection_group(self):
        """Group selected objects."""
        self.__page.selection_group()

    def selection_ungroup(self):
        """Ungroup selected objects."""
        self.__page.selection_ungroup()

    def select_reverse(self):
        """Reverse the selection."""
        self.__page.selection().reverse()
        self.__app.dm.mode("move")

    def select_all(self):
        """Select all objects."""
        if not self.__page.objects():
            return

        self.__page.selection().all()
        self.__app.dm.mode("move")

    def selection_delete(self):
        """Delete selected objects."""
        self.__page.selection_delete()

    def select_next_object(self):
        """Select the next object."""
        self.__page.selection().next()

    def select_previous_object(self):
        """Select the previous object."""
        self.__page.selection().previous()

    def selection_fill(self):
        """Fill the selected object."""
        # XXX gom should not call dm directly
        # this code should be gone!
        color = self.__app.dm.pen().color
        for obj in self.__page.selection().objects:
            obj.fill(color)

    def selection_color_set(self, color):
        """Set the color of the selected objects."""
        self.__page.selection_color_set(color)

    def selection_font_set(self, font_description):
        """Set the font of the selected objects."""
        self.__page.selection_font_set(font_description)

    def selection_apply_pen(self):
        """Apply the pen to the selected objects."""
        pen = self.__app.dm.pen()
        self.__page.selection_apply_pen(pen)

    def do(self, command):
        """Do a command."""
        self.__page.do(command)

    def redo(self):
        """Redo the last action."""
        self.__page.redo()

    def undo(self):
        """Undo the last action."""
        self.__page.undo()

    def move_obj(self, obj, dx, dy):
        """Move the object by the given amount."""
        self.__page.move_obj(obj, dx, dy)

    def move_selection(self, dx, dy):
        """Move the selected objects by the given amount."""
        if self.__page.selection().is_empty():
            return
        self.__page.move_obj(self.__page.selection(), dx, dy)

    def rotate_obj(self, obj, angle):
        """Rotate the object by the given angle (degrees)."""
        self.__page.rotate_obj(obj, angle)

    def rotate_selection(self, angle):
        """Rotate the selected objects by the given angle (degrees)."""
        if self.__page.selection().is_empty():
            return
        self.__rotate_obj(self.__page.selection(), angle)

    def selection_zmove(self, operation):
        """move the selected objects long the z-axis."""
        self.__page.selection_zmove(operation)

    def next_page(self):
        """Go to the next page."""
        self.__page = self.__page.next()

    def prev_page(self):
        """Go to the prev page."""
        self.__page = self.__page.prev()

    def delete_page(self):
        """Delete the current page."""
        self.__page = self.__page.delete()

