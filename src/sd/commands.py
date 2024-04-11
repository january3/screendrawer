"""
This module contains the commands that can be executed on the objects. They
should be undoable and redoable. It is their responsibility to update the
state of the objects they are acting on.
"""


from .utils import calc_rotation_angle, sort_by_stack ## <remove>
from .drawable_factory import DrawableFactory ## <remove>
from .drawable_group import DrawableGroup ## <remove>

def swap_stacks(stack1, stack2):
    """Swap two stacks"""
    stack1[:], stack2[:] = stack2[:], stack1[:]


## ---------------------------------------------------------------------
## These are the commands that can be executed on the objects. They should
## be undoable and redoable. It is their responsibility to update the
## state of the objects they are acting on.

class Command:
    """Base class for commands."""
    def __init__(self, mytype, objects):
        self.obj   = objects
        self.__type   = mytype
        self.__undone = False

    def command_type(self):
        """Return the type of the command."""
        return self.__type

    def undo(self):
        """Undo the command."""
        raise NotImplementedError("undo method not implemented")

    def redo(self):
        """Redo the command."""
        raise NotImplementedError("redo method not implemented")

    def undone(self):
        """Return whether the command has been undone."""
        return self.__undone

    def undone_set(self, value):
        """Set the undone status of the command."""
        self.__undone = value


class CommandGroup(Command):
    """Simple class for handling groups of commands."""
    def __init__(self, commands, page=None):
        super().__init__("group", objects=None)
        self.__commands = commands
        self.__page = page

    def undo(self):
        """Undo the command."""
        if self.undone():
            return None
        for cmd in self.__commands:
            cmd.undo()
        self.undone_set(True)
        return self.__page

    def redo(self):
        """Redo the command."""
        if not self.undone():
            return None
        for cmd in self.__commands:
            cmd.redo()
        self.undone_set(False)
        return self.__page

class DeletePageCommand(Command):
    """
    Handling deleting pages.
    """
    def __init__(self, page):
        super().__init__("delete_page", None)
        prev_page = page.prev()

        if prev_page == page:
            prev_page = None

        next_page = page.next(create = False)

        self.__prev = prev_page
        self.__next = next_page
        self.__page = page

        if self.__prev:
            # set the previous page's next to our next
            self.__prev.next_set(self.__next)
        if self.__next:
            # set the next page's previous to our previous
            self.__next.prev_set(self.__prev)

    def undo(self):
        """Undo the command."""
        if self.undone():
            return None
        if self.__prev:
            self.__prev.next_set(self.__page)
        if self.__next:
            self.__next.prev_set(self.__page)
        self.undone_set(True)
        return self.__page

    def redo(self):
        """Redo the command."""
        if not self.undone():
            return None
        if self.__prev:
            self.__prev.next_set(self.__next)
        if self.__next:
            self.__next.prev_set(self.__prev)
        self.undone_set(False)
        return self.__page

class DeleteLayerCommand(Command):
    """Simple class for handling deleting layers of a page."""
    def __init__(self, page, layer_pos = None):
        """Simple class for handling deleting layers of a page."""
        super().__init__("delete_layer", None)

        if layer_pos is None:
            layer_pos = page.layer_no()
        self.__layer, self.__layer_pos = page.delete_layer(layer_pos)
        self.__page  = page

    def undo(self):
        """Undo the command."""
        if self.undone():
            return None
        self.__page.layer(self.__layer, self.__layer_pos)
        self.undone_set(True)
        return self.__page

    def redo(self):
        """Redo the command."""
        if not self.undone():
            return None
        self.__page.delete_layer(self.__layer_pos)
        self.undone_set(False)
        return self.__page

class GroupObjectCommand(Command):
    """Simple class for handling grouping objects."""
    def __init__(self, objects, stack, selection_object = None, page = None):
        objects = sort_by_stack(objects, stack)

        super().__init__("group", objects)
        self.__stack      = stack
        self.__stack_copy = stack[:]

        self.__page      = page
        self.__selection = selection_object

        self.__group = DrawableGroup(self.obj)
        #self.__grouped_objects = self.obj[:]

        # position of the last object in stack
        idx = self.__stack.index(self.obj[-1])

        # add group to the stack at the position of the last object
        self.__stack.insert(idx, self.__group)

        for obj in self.obj:
            self.__stack.remove(obj)

        if self.__selection:
            self.__selection.set([ self.__group ])

    def swap_stacks(self):
        """Swap the stack with the copy of the stack."""
        # put the copy of the stack into the stack,
        # and the stack into the copy
        tmp = self.__stack[:]
        self.__stack[:] = self.__stack_copy[:]
        self.__stack_copy = tmp

    def undo(self):
        """Undo the command."""
        if self.undone():
            return None
        self.swap_stacks()
        self.undone_set(True)
        if self.__selection:
            self.__selection.set(self.obj)
        return self.__page

    def redo(self):
        """Redo the command."""
        if not self.undone():
            return None
        self.swap_stacks()
        self.undone_set(False)
        if self.__selection:
            self.__selection.set([ self.__group ])
        return self.__page

class UngroupObjectCommand(Command):
    """
    Class for handling ungrouping objects.

    :param objects: Objects to be ungrouped (objects which are not groups
                    will be ignored)
    """
    def __init__(self, objects, stack, selection_object = None, page = None):
        super().__init__("ungroup", objects)
        self.__stack = stack
        self.__stack_copy = stack[:]
        self.__selection = selection_object
        self.__page = page

        new_objects = []
        n = 0

        for obj in self.obj:
            if not obj.type == "group":
                print("Object is not a group, ignoring:", obj)
                print("object type:", obj.type)
                continue

            n += 1
            # position of the group in the stack
            idx = self.__stack.index(obj)

            # remove the group from the stack
            self.__stack.remove(obj)

            # add the objects back to the stack
            for subobj in obj.objects[::-1]:
                self.__stack.insert(idx, subobj)
                new_objects.append(subobj)

        if n > 0 and self.__selection:
            self.__selection.set(new_objects)

    def swap_stacks(self):
        """Swap the stack with the copy of the stack."""
        # put the copy of the stack into the stack,
        # and the stack into the copy
        tmp = self.__stack[:]
        self.__stack[:] = self.__stack_copy[:]
        self.__stack_copy = tmp

    def undo(self):
        """Undo the command."""
        if self.undone():
            return None
        self.swap_stacks()
        self.undone_set(True)
        if self.__selection:
            self.__selection.set([ self.obj ])
        return self.__page

    def redo(self):
        """Redo the command."""
        if not self.undone():
            return None
        self.swap_stacks()
        self.undone_set(False)
        if self.__selection:
            self.__selection.set(self.obj)
        return self.__page



class RemoveCommand(Command):
    """
    Class for handling deleting objects.

    :param objects: a list of objects to be removed.
    """
    def __init__(self, objects, stack, page = None):
        super().__init__("remove", objects)
        self.__stack = stack
        self.__stack_copy = self.__stack[:]
        self.__page = page

        # remove the objects from the stack
        for obj in self.obj:
            self.__stack.remove(obj)

    def undo(self):
        if self.undone():
            return None
        swap_stacks(self.__stack, self.__stack_copy)
        self.undone_set(True)
        return self.__page

    def redo(self):
        if not self.undone():
            return None
        swap_stacks(self.__stack, self.__stack_copy)
        self.undone_set(False)
        return self.__page

class AddCommand(Command):
    """
    Class for handling creating objects.

    :param objects: a list of objects to be removed.
    """
    def __init__(self, objects, stack, page = None):
        super().__init__("add", objects)
        self.__stack = stack
        self.__page = page
        self.__add_objects()

    def __add_objects(self):
        for o in self.obj:
            self.__stack.append(o)

    def undo(self):
        if self.undone():
            return None

        for o in self.obj:
            self.__stack.remove(o)

        self.undone_set(True)
        return self.__page

    def redo(self):
        if not self.undone():
            return None
        self.__add_objects()
        self.undone_set(False)
        return self.__page

class TransmuteCommand(Command):
    """
    Turning object(s) into another type.

    Internally, for each object we create a new object of the new type, and
    replace the old object with the new one in the stack.

    For undo method, we should store the old object as well as its position in the stack.
    However, we don't. Instead we just slap the old object back onto the stack.
    """

    def __init__(self, objects, stack, new_type, selection_objects = None, page = None):
        super().__init__("transmute", objects)
        #self.__new_type = new_type
        self.__old_objs = [ ]
        self.__new_objs = [ ]
        self.__stack    = stack
        self.__selection_objects = selection_objects
        self.__page = page
        print("executing transmute; undone = ", self.undone())

        for obj in self.obj:
            new_obj = DrawableFactory.transmute(obj, new_type)

            if not obj in self.__stack:
                raise ValueError("TransmuteCommand: Got Object not in stack:", obj)

            if obj == new_obj: # ignore if no transmutation
                continue

            self.__old_objs.append(obj)
            self.__new_objs.append(new_obj)
            self.__stack.remove(obj)
            self.__stack.append(new_obj)

        if self.__selection_objects:
            self.map_selection()

    def map_selection(self):
        """Map the selection objects to the new objects."""
        obj_map = self.obj_map()
        # XXX this should not change the order of the objects
        self.__selection_objects[:] = [ obj_map.get(obj, obj) for obj in self.__selection_objects ]

    def obj_map(self):
        """Return a dictionary mapping old objects to new objects."""
        return { self.__old_objs[i]: self.__new_objs[i] for i in range(len(self.__old_objs)) }

    def undo(self):
        """replace all the new objects with the old ones in the stack"""
        if self.undone():
            return None
        for obj in self.__new_objs:
            self.__stack.remove(obj)
        for obj in self.__old_objs:
            self.__stack.append(obj)
        self.undone_set(True)
        if self.__selection_objects:
            self.map_selection()
        return self.__page

    def redo(self):
        """put the new objects again on the stack and remove the old ones"""
        if not self.undone():
            return None
        for obj in self.__old_objs:
            self.__stack.remove(obj)
        for obj in self.__new_objs:
            self.__stack.append(obj)
        self.undone_set(False)
        if self.__selection_objects:
            self.map_selection()
        return self.__page

class ZStackCommand(Command):
    """Simple class for handling z-stack operations."""
    def __init__(self, objects, stack, operation, page = None):
        super().__init__("z_stack", objects)
        self._operation  = operation
        self.__stack      = stack
        self.__page = page

        for obj in objects:
            if not obj in stack:
                raise ValueError("Object not in stack:", obj)

        self._objects = sort_by_stack(objects, stack)
        # copy of the old stack
        self.__stack_orig = stack[:]

        if operation == "raise":
            self.hoist() # raise is reserved
        elif operation == "lower":
            self.lower()
        elif operation == "top":
            self.top()
        elif operation == "bottom":
            self.bottom()
        else:
            raise ValueError("Invalid operation:", operation)

    ## here is the problem: not all objects that we get exist in the stack.
    ## u

    def hoist(self):
        """Move the objects towards the top of the stack."""
        li = self.__stack.index(self._objects[-1])
        n  = len(self.__stack)

        # if the last element is already on top, we just move everything to
        # the top
        if li == n - 1:
            self.top()
            return

        # otherwise, we move all the objects to the position of the element
        # following the last one. Then, we just copy the elements from the
        # stack to the new stack, and when we see the indicator object, we
        # add our new objects.
        ind_obj = self.__stack[li + 1]

        new_list = []
        for i in range(n):
            o = self.__stack[i]
            if not o in self._objects:
                new_list.append(o)
            if o == ind_obj:
                new_list.extend(self._objects)

        self.__stack[:] = new_list[:]

    def lower(self):
        """Move the objects towards the bottom of the stack."""
        fi = self.__stack.index(self._objects[0])
        n  = len(self.__stack)

        if fi == 0:
            self.bottom()
            return

        # otherwise, we move all the objects to the position of the element
        # preceding the first one. Then, we just copy the elements from the
        # stack to the new stack, and when we see the indicator object, we
        # this could be done more efficiently, but that way it is clearer

        ind_obj = self.__stack[fi - 1]
        new_list = []
        for i in range(n):
            o = self.__stack[i]
            if o == ind_obj:
                new_list.extend(self._objects)
            if not o in self._objects:
                new_list.append(o)

        self.__stack[:] = new_list[:]

    def top(self):
        """Move the objects to the top of the stack."""
        for obj in self._objects:
            self.__stack.remove(obj)
            self.__stack.append(obj)

    def bottom(self):
        """Move the objects to the bottom of the stack."""
        for obj in self._objects[::-1]:
            self.__stack.remove(obj)
            self.__stack.insert(0, obj)

    def undo(self):
        """Undo the command."""
        if self.undone():
            return None
        swap_stacks(self.__stack, self.__stack_orig)
        self.undone_set(True)
        return self.__page

    def redo(self):
        """Redo the command."""
        if not self.undone():
            return None
        swap_stacks(self.__stack, self.__stack_orig)
        self.undone_set(False)
        return self.__page

# --------------------------------------------------------------

class MoveResizeCommand(Command):
    """
    Simple class for handling move and resize events.

    Attributes:
        start_point (tuple): the point where the first click was made
        origin (tuple): the original position of the object
        bbox (tuple): the bounding box of the object

    Arguments:
        type (str): the type of the command
        obj (Drawable): the object to be moved or resized
        origin (tuple): the original position of the object

    This class is different from other classes because it takes a single
    object as the argument. This is because Move or Resize commands need to
    react to continuous updates. It is therefore the responsibility of the
    caller to ensure that a list of objects is grouped as a DrawableGroup.

    Also, the subclasses need to implement two methods: event_update and
    event_finish, which have to handle the changes during move / resize and
    call on objects to finalize the command.
    """

    def __init__(self, mytype, obj, origin):
        super().__init__(mytype, obj)
        self.start_point = origin
        self.origin      = origin
        self.bbox        = obj.bbox()

    def event_update(self, x, y):
        """Update the move or resize event."""
        raise NotImplementedError("event_update method not implemented for type", self.__type)

    def event_finish(self):
        """Finish the move or resize event."""
        raise NotImplementedError("event_finish method not implemented for type", self.__type)

class RotateCommand(MoveResizeCommand):
    """
    Simple class for handling rotate events.

    Attributes:

        corner (str): the corner which is being dragged, e.g. "upper_left"
        _rotation_centre (tuple): the point around which the rotation is done

    Arguments:
        obj (Drawable): object to be rotated
        origin (tuple, optional): where the first click was made
        corner (str, optional): which corner has been clicked
        angle (float, optional): set the rotation angle directly
    """

    def __init__(self, obj, origin=None, corner=None, angle = None, page = None):
        super().__init__("rotate", obj, origin)
        self.corner      = corner
        bb = obj.bbox()
        self.__rotation_centre = (bb[0] + bb[2] / 2, bb[1] + bb[3] / 2)
        obj.rotate_start(self.__rotation_centre)
        self.__page = page

        if not angle is None:
            self.obj.rotate(angle, set = False)

        self._angle = 0

    def event_update(self, x, y):
        angle = calc_rotation_angle(self.__rotation_centre, self.start_point, (x, y))
        d_a = angle - self._angle
        self._angle = angle
        self.obj.rotate(d_a, set_angle = False)

    def event_finish(self):
        self.obj.rotate_end()

    def undo(self):
        if self.undone():
            return None
        self.obj.rotate_start(self.__rotation_centre)
        self.obj.rotate(-self._angle)
        self.obj.rotate_end()
        self.undone_set(True)
        return self.__page

    def redo(self):
        if not self.undone():
            return None
        self.obj.rotate_start(self.__rotation_centre)
        self.obj.rotate(self._angle)
        self.obj.rotate_end()
        self.undone_set(False)
        return self.__page

class MoveCommand(MoveResizeCommand):
    """Simple class for handling move events."""
    def __init__(self, obj, origin, page = None):
        super().__init__("move", obj, origin)
        self.__last_pt = origin
        self.__page = page
        print("MoveCommand: origin is", origin)

    def event_update(self, x, y):
        """Update the move event."""
        dx = x - self.__last_pt[0]
        dy = y - self.__last_pt[1]

        self.obj.move(dx, dy)
        self.__last_pt = (x, y)

    def event_finish(self):
        """Finish the move event."""
        print("MoveCommand: finish")

    def undo(self):
        """Undo the command."""
        if self.undone():
            return None
        print("MoveCommand: undo")
        dx = self.start_point[0] - self.__last_pt[0]
        dy = self.start_point[1] - self.__last_pt[1]
        self.obj.move(dx, dy)
        self.undone_set(True)
        return self.__page

    def redo(self):
        """Redo the command."""
        if not self.undone():
            return None
        dx = self.start_point[0] - self.__last_pt[0]
        dy = self.start_point[1] - self.__last_pt[1]
        self.obj.move(-dx, -dy)
        self.undone_set(False)
        return self.__page


class ResizeCommand(MoveResizeCommand):
    """Simple class for handling resize events."""
    def __init__(self, obj, origin, corner, proportional = False, page = None):
        super().__init__("resize", obj, origin)
        self.corner = corner
        obj.resize_start(corner, origin)
        self._orig_bb = obj.bbox()
        self._prop    = proportional
        if self._orig_bb[2] == 0:
            raise ValueError("Bounding box with no width")

        self._orig_bb_ratio = self._orig_bb[3] / self._orig_bb[2]
        self.__newbb = None
        self.__page = page

    def undo(self):
        """Undo the command."""
        if self.undone():
            return None
        obj = self.obj
        pt  = (self._orig_bb[0], self._orig_bb[1])
        obj.resize_start(self.corner, pt)
        self.obj.resize_update(self._orig_bb)
        obj.resize_end()
        self.undone_set(True)
        return self.__page

    def redo(self):
        """Redo the command."""
        if not self.undone():
            return None
        obj = self.obj
        obj.resize_start(self.corner, self.start_point)
        obj.resize_update(self.__newbb)
        obj.resize_end()
        self.undone_set(False)
        return self.__page

    def event_finish(self):
        """Finish the resize event."""
        self.obj.resize_end()

    def event_update(self, x, y):
        """Update the resize event."""
        bb = self._orig_bb
        corner = self.corner

        if corner in ["upper_left", "lower_right"]:
            dx = x - self.origin[0]
            dy = y - self.origin[1]
        else:
            dx = (self.origin[0] + bb[2]) - x
            dy = y - self.origin[1] + bb[3]

        if dx == 0 or dy == 0:
            return

        if self._prop:
            if dy / dx > self._orig_bb_ratio:
                dy = dx * self._orig_bb_ratio
            else:
                dx = dy / self._orig_bb_ratio

        if corner == "lower_left":
            newbb = (bb[0] + bb[2] - dx, bb[1], dx, dy)
        elif corner == "upper_right":
            newbb = (bb[0], bb[1] + dy - bb[3], bb[2] * 2 - dx, bb[3] - dy + bb[3])
        elif corner == "upper_left":
            newbb = (bb[0] + dx, bb[1] + dy, bb[2] - dx, bb[3] - dy)
        elif corner == "lower_right":
            newbb = (bb[0], bb[1], bb[2] + dx, bb[3] + dy)
        else:
            raise ValueError("Invalid corner:", corner)

        self.__newbb = newbb
        self.obj.resize_update(newbb)

# -------------------------------------------------------------------------------------

class SetPropCommand(Command):
    """
    Superclass for handling property changes of drawing primitives.

    The superclass handles everything, while the subclasses set up the
    functions that do the actual manipulation of the primitives.

    In principle, we need one function to extract the current property, and
    one to set the property.
    """
    def __init__(self, mytype, objects, prop, get_prop_func, set_prop_func, page = None):
        super().__init__(mytype, objects.get_primitive())
        self.__prop = prop
        self.__page = page
        #self.__get_prop_func = get_prop_func
        self.__set_prop_func = set_prop_func
        self.__undo_dict = { obj: get_prop_func(obj) for obj in self.obj }

        for obj in self.obj:
            print("setting prop for", obj)
            set_prop_func(obj, prop)

    def undo(self):
        """Undo the command."""
        if self.undone():
            return None
        for obj in self.obj:
            if obj in self.__undo_dict:
                self.__set_prop_func(obj, self.__undo_dict[obj])
        self.undone_set(True)
        return self.__page

    def redo(self):
        """Redo the command."""
        if not self.undone():
            return None
        for obj in self.obj:
            self.__set_prop_func(obj, self.__prop)
        self.undone_set(False)
        return self.__page


def pen_set_func(obj, prop):
    """Set the pen of the object."""
    obj.pen_set(prop)

def pen_get_func(obj):
    """Get the pen of the object."""
    return obj.pen

class SetPenCommand(SetPropCommand):
    """Simple class for handling color changes."""
    def __init__(self, objects, pen):
        set_prop_func = pen_set_func
        get_prop_func = pen_get_func
        pen = pen.copy()
        super().__init__("set_pen", objects, pen, get_prop_func, set_prop_func)


def color_set_func(obj, prop):
    """Set the color of the object."""
    obj.color_set(prop)

def color_get_func(obj):
    """Get the color of the object."""
    return obj.pen.color

class SetColorCommand(SetPropCommand):
    """Simple class for handling color changes."""
    def __init__(self, objects, color):
        set_prop_func = color_set_func
        get_prop_func = color_get_func
        super().__init__("set_color", objects, color, get_prop_func, set_prop_func)


def set_font_func(obj, prop):
    """Set the font of the object."""
    obj.pen.font_set(prop)

def get_font_func(obj):
    """Get the font of the object."""
    return obj.pen.font_get()

class SetFontCommand(SetPropCommand):
    """Simple class for handling font changes."""
    def __init__(self, objects, font):
        set_prop_func = set_font_func
        get_prop_func = get_font_func
        super().__init__("set_font", objects, font, get_prop_func, set_prop_func)
