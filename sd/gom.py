from .commands import *                                              # <remove>
from .drawable import SelectionObject, DrawableGroup, SelectionTool  # <remove>
from .clipboard import Clipboard                                     # <remove>
from .drawable import DrawableFactory                                # <remove>


## ---------------------------------------------------------------------

class GraphicsObjectManager:
    """
    Class to manage graphics objects.

    Attributes:
        _objects (list): The list of objects.
        _history_stack (list): The list of commands in the history.
        _hidden (bool): True if the objects are hidden.
        _resizeobj (Drawable): The object being resized.
        _mode (str): The current mode.
        _hover (Drawable): The object being hovered over.
        _clipboard (Clipboard): The clipboard.
        _selection_tool (SelectionTool): The selection tool.
    """

    def __init__(self, app):
        # public attr
        self.clipboard = Clipboard()

        # private attr
        self.__app = app
        self._objects    = []
        self._history    = []
        self._redo_stack = []
        self.selection = SelectionObject(self._objects)

        self._hidden = False
        self._resizeobj = None
        self._hover = None
        self._selection_tool = None

    def objects(self):
        """Return the list of objects."""
        return self._objects

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
        self._history.append(TransmuteCommand(objects, self._objects, mode, self.selection.objects))
        # XXX the problem is that we need to remove the old objects from the
        # selection as well. However, it turns out to be more complicated than

    def transmute_selection(self, mode):
        """
        Transmute the selected objects to the given mode.

        Args:
            mode ( str ): The mode to transmute to.
        """
        if self.selection.is_empty():
            return
        self.transmute(self.selection.objects, mode)

    def set_objects(self, objects):
        """Set the list of objects."""
        ## no undo
        print("GOM: setting n=", len(objects), "objects")
        self._objects = objects
        self.selection = SelectionObject(self._objects)

    def add_object(self, obj):
        """Add an object to the list of objects."""
        self._history.append(AddCommand(obj, self._objects))
        ##self._objects.append(obj)

    def export_objects(self):
        objects = [ obj.to_dict() for obj in self._objects ]
        return objects

    def kill_object(self, obj):
        """Directly remove an object from the list of objects."""
        self._objects.remove(obj)
        ##self._objects.remove(obj)

    def selected_objects(self):
        """Return the selected objects."""
        return self.selection.objects

    def remove_selection(self):
        """Remove the selected objects from the list of objects."""
        if self.selection.is_empty():
            return
        self._history.append(RemoveCommand(self.selection.objects, self._objects))
        self.selection.clear()

    def remove_objects(self, objects, clear_selection = False):
        """Remove an object from the list of objects."""
        self._history.append(RemoveCommand(objects, self._objects))
        if clear_selection:
            self.selection.clear()
        ##self._objects.remove(obj)

    def remove_all(self):
        """Clear the list of objects."""
        self._history.append(RemoveCommand(self._objects[:], self._objects))

    def command_append(self, command_list):
        """Append a group of commands to the history."""
        ## append in reverse order
        self._history.append(CommandGroup(command_list[::-1]))

    def hide(self):
        """Hide the objects."""
        self._hidden = True

    def show(self):
        """Show the objects."""
        self._hidden = False

    def toggle_visibility(self):
        """Toggle the visibility of the objects."""
        self._hidden = not self._hidden

    def selection_group(self):
        """Group selected objects."""
        if self.selection.n() < 2:
            return
        print("Grouping", self.selection.n(), "objects")
        objects = sort_by_stack(self.selection.objects, self._objects)
        new_grp_obj = DrawableGroup(objects)

        for obj in self.selection.objects:
            self._objects.remove(obj)

        # XXX history append CommandGroup: Remove obj + add group
        self._objects.append(new_grp_obj)
        self.selection.set([ new_grp_obj ])

    def selection_ungroup(self):
        """Ungroup selected objects."""
        if self.selection.is_empty():
            return
        for obj in self.selection.objects:
            if obj.type == "group":
                print("Ungrouping", obj)
                self._objects.extend(obj.objects)
                self._objects.remove(obj)
        return

    def select_reverse(self):
        """Reverse the selection."""
        self.selection.reverse()
        self.__app.dm.mode("move")

    def select_all(self):
        """Select all objects."""
        if not self._objects:
            return

        self.selection.all()
        self.__app.dm.mode("move")

    def selection_delete(self):
        """Delete selected objects."""
        if self.selection.objects:
            self._history.append(RemoveCommand(self.selection.objects, self._objects))
            self.selection.clear()

    def select_next_object(self):
        """Select the next object."""
        self.selection.next()

    def selection_fill(self):
        """Fill the selected object."""
        # XXX gom should not call dm directly
        color = self.__app.dm.pen().color
        for obj in self.selection.objects:
            obj.fill(color)

    def select_previous_object(self):
        """Select the previous object."""
        self.selection.previous()

    def selection_color_set(self, color):
        """Set the color of the selected objects."""
        if not self.selection.is_empty():
            self._history.append(SetColorCommand(self.selection, color))

    def selection_font_set(self, font_description):
        for obj in self.selection.objects:
            obj.pen.font_set_from_description(font_description)

    def selection_apply_pen(self):
        pen = self.__app.dm.pen()
        """Apply the pen to the selected objects."""
        if not self.selection.is_empty():
            # self._history.append(SetColorCommand(self.selection, pen.color))
            # self._history.append(SetLWCommand(self.selection, pen.color))
            for obj in self.selection.objects:
                obj.set_pen(pen)

    def do(self, command):
        """Do a command."""
        self._history.append(command)

    def redo(self):
        """Redo the last action."""
        print("Redo stack, size is", len(self._redo_stack))
        if self._redo_stack:
            command = self._redo_stack.pop()
            command.redo()
            self._history.append(command)

    def undo(self):
        """Undo the last action."""
        print("Undo, history size is", len(self._history))
        if self._history:
            command = self._history.pop()
            command.undo()
            self._redo_stack.append(command)

    def move_obj(self, obj, dx, dy):
        """Move the object by the given amount."""
        eventObj = MoveCommand(obj, (0, 0))
        eventObj.event_update(dx, dy)
        self._history.append(eventObj)

    def move_selection(self, dx, dy):
        """Move the selected objects by the given amount."""
        if self.selection.is_empty():
            return
        self.move_obj(self.selection, dx, dy)

    def rotate_obj(self, obj, angle):
        """Rotate the object by the given angle (degrees)."""
        print("rotating by", angle)
        eventObj = RotateCommand(obj, angle=math.radians(angle))
        eventObj.event_finish()
        self._history.append(eventObj)

    def rotate_selection(self, angle):
        """Rotate the selected objects by the given angle (degrees)."""
        if self.selection.is_empty():
            return
        self.rotate_obj(self.selection, angle)

    def selection_zmove(self, operation):
        """move the selected objects long the z-axis."""
        if self.selection.is_empty():
            return
        self._history.append(ZStackCommand(self.selection.objects, self._objects, operation))


