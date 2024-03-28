from .drawable import Drawable, DrawableGroup, SelectionObject # <remove>
from .commands import *                                              # <remove>
from .utils import sort_by_stack                                      # <remove>


class Page:
    def __init__(self):
        self.__objects = []
        self.__history = []
        self.__redo_stack = []
        self.__selection = SelectionObject(self.__objects)

    def objects(self, objects = None):
        if objects:
            self.__objects = objects
            self.__selection = SelectionObject(self.__objects)
        return self.__objects

    def history(self):
        return self.__history
    
    def redo_stack(self):
        return self.__redo_stack

    def selection(self):
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
        self.__history.append(TransmuteCommand(objects, self.__objects, mode, self.__selection.objects))
        # XXX the problem is that we need to remove the old objects from the
        # selection as well. However, it turns out to be more complicated than

    def add_object(self, obj):
        """Add an object to the list of objects."""
        self.__history.append(AddCommand(obj, self.__objects))
 
    def kill_object(self, obj):
        """Directly remove an object from the list of objects."""
        self.__objects.remove(obj)

    def remove_selection(self):
        """Remove the selected objects from the list of objects."""
        if self.__selection.is_empty():
            return
        self.__history.append(RemoveCommand(self.__selection.objects, self.__objects))
        self.__selection.clear()

    def remove_objects(self, objects, clear_selection = False):
        """Remove an object from the list of objects."""
        self.__history.append(RemoveCommand(objects, self.__objects))
        if clear_selection:
            self.__selection.clear()

    def remove_all(self):
        """Clear the list of objects."""
        self.__history.append(RemoveCommand(self.__objects[:], self.__objects))

    def selection_group(self):
        """Group selected objects."""
        if self.__selection.n() < 2:
            return
        print("Grouping", self.__selection.n(), "objects")
        objects = sort_by_stack(self.__selection.objects, self.__objects)
        new_grp_obj = DrawableGroup(objects)

        for obj in self.__selection.objects:
            self.__objects.remove(obj)

        # XXX history append CommandGroup: Remove obj + add group
        self.__objects.append(new_grp_obj)
        self.__selection.set([ new_grp_obj ])

    def selection_ungroup(self):
        """Ungroup selected objects."""
        if self.__selection.is_empty():
            return
        for obj in self.__selection.objects:
            if obj.type == "group":
                print("Ungrouping", obj)
                self.__objects.extend(obj.objects)
                self.__objects.remove(obj)
        return

    def selection_delete(self):
        """Delete selected objects."""
        if self.__selection.objects:
            self.__history.append(RemoveCommand(self.__selection.objects, self.__objects))
            self.__selection.clear()

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

    def do(self, command):
        """Do a command."""
        self.__history.append(command)

    def command_append(self, command_list):
        """Append a group of commands to the history."""
        ## append in reverse order
        self.__history.append(CommandGroup(command_list[::-1]))

    def redo(self):
        """Redo the last action."""
        print("Redo stack, size is", len(self.__redo_stack))
        if self.__redo_stack:
            command = self.__redo_stack.pop()
            command.redo()
            self.__history.append(command)

    def undo(self):
        """Undo the last action."""
        print("Undo, history size is", len(self.__history))
        if self.__history:
            command = self.__history.pop()
            command.undo()
            self.__redo_stack.append(command)

    def move_obj(self, obj, dx, dy):
        """Move the object by the given amount."""
        eventObj = MoveCommand(obj, (0, 0))
        eventObj.event_update(dx, dy)
        self.__history.append(eventObj)

    def rotate_obj(self, obj, angle):
        """Rotate the object by the given angle (degrees)."""
        print("rotating by", angle)
        eventObj = RotateCommand(obj, angle=math.radians(angle))
        eventObj.event_finish()
        self.__history.append(eventObj)

    def selection_zmove(self, operation):
        """move the selected objects long the z-axis."""
        if self.__selection.is_empty():
            return
        self.__history.append(ZStackCommand(self.__selection.objects, self.__objects, operation))



