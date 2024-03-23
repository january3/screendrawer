from .utils import * ## <remove>
from .drawable import DrawableFactory ## <remove>

## ---------------------------------------------------------------------
## These are the commands that can be executed on the objects. They should
## be undoable and redoable. It is their responsibility to update the
## state of the objects they are acting on.

class Command:
    """Base class for commands."""
    def __init__(self, type, objects):
        self.obj   = objects
        self._type = type
        self._undone = False

    def command_type(self):
        return self._type

    def undo(self):
        raise NotImplementedError("undo method not implemented")

    def redo(self):
        raise NotImplementedError("redo method not implemented")

class CommandGroup(Command):
    """Simple class for handling groups of commands."""
    def __init__(self, commands):
        self._commands = commands

    def undo(self):
        for cmd in self._commands:
            cmd.undo()
        self._undone = True

    def redo(self):
        if not self._undone:
            return
        for cmd in self._commands:
            cmd.redo()
        self._undone = False
        
class SetColorCommand(Command):
    """Simple class for handling color changes."""
    # XXX: what happens if an object is added to group after the command,
    # but before the undo? well, bad things happen
    def __init__(self, objects, color):
        super().__init__("set_color", objects.get_primitive())
        self._color = color
        self._undo_color = { obj: obj.pen.color for obj in self.obj }

        for obj in self.obj:
            obj.color_set(color)

    def undo(self):
        for obj in self.obj:
            if obj in self._undo_color:
                obj.color_set(self._undo_color[obj])
        self._undone = True

    def redo(self):
        if not self._undone:
            return
        for obj in self.obj:
            obj.color_set(self._color)
        self._undone = False

class RemoveCommand(Command):
    """Simple class for handling deleting objects."""
    def __init__(self, objects, stack):
        super().__init__("remove", objects)
        self._stack = stack

        # remove the objects from the stack
        for obj in self.obj:
            self._stack.remove(obj)

    def undo(self):
        for obj in self.obj:
        # XXX: it should insert at the same position!
            self._stack.append(obj)
        self._undone = True

    def redo(self):
        if not self._undone:
            return
        for obj in self.obj:
            self._stack.remove(obj)
        self._undone = False

class AddCommand(Command):
    """Simple class for handling creating objects."""
    def __init__(self, objects, stack):
        super().__init__("add", objects)
        self._stack = stack
        self._stack.append(self.obj)

    def undo(self):
        self._stack.remove(self.obj)
        self._undone = True

    def redo(self):
        if not self._undone:
            return
        self._stack.append(self.obj)
        self._undone = False

class TransmuteCommand(Command):
    """
    Turning object(s) into another type.

    Internally, for each object we create a new object of the new type, and
    replace the old object with the new one in the stack.

    For undo method, we should store the old object as well as its position in the stack.
    However, we don't. Instead we just slap the old object back onto the stack.
    """

    def __init__(self, objects, stack, new_type, selection_objects = None):
        super().__init__("transmute", objects)
        self._new_type = new_type
        self._old_objs = [ ]
        self._new_objs = [ ]
        self._stack    = stack
        self._selection_objects = selection_objects

        for obj in self.obj:
            new_obj = DrawableFactory.transmute(obj, new_type)

            if not obj in self._stack:
                raise ValueError("TransmuteCommand: Got Object not in stack:", obj)
                continue

            if obj == new_obj: # ignore if no transmutation
                continue

            self._old_objs.append(obj)
            self._new_objs.append(new_obj)
            self._stack.remove(obj)
            self._stack.append(new_obj)

        if self._selection_objects:
            self.map_selection()

    def map_selection(self):
        obj_map = self.obj_map()
        # XXX this should not change the order of the objects
        self._selection_objects[:] = [ obj_map.get(obj, obj) for obj in self._selection_objects ]

    def obj_map(self):
        """Return a dictionary mapping old objects to new objects."""
        return { self._old_objs[i]: self._new_objs[i] for i in range(len(self._old_objs)) }

    def undo(self):
        """replace all the new objects with the old ones in the stack"""
        if self._undone:
            return
        for obj in self._new_objs:
            self._stack.remove(obj)
        for obj in self._old_objs:
            self._stack.append(obj)
        self._undone = True
        if self._selection_objects:
            self.map_selection()

    def redo(self):
        """put the new objects again on the stack and remove the old ones"""
        if not self._undone:
            return
        for obj in self._old_objs:
            self._stack.remove(obj)
        for obj in self._new_objs:
            self._stack.append(obj)
        self._undone = False
        if self._selection_objects:
            self.map_selection()

class ZStackCommand(Command):
    """Simple class for handling z-stack operations."""
    def __init__(self, objects, stack, operation):
        super().__init__("z_stack", objects)
        self._operation  = operation
        self._stack      = stack

        for obj in objects:
            if not obj in stack:
                raise ValueError("Object not in stack:", obj)

        self._objects = sort_by_stack(objects, stack)
        # copy of the old stack
        self._stack_orig = stack[:]

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
        li = self._stack.index(self._objects[-1])
        n  = len(self._stack)

        # if the last element is already on top, we just move everything to
        # the top
        if li == n - 1:
            self.top()
            return

        # otherwise, we move all the objects to the position of the element
        # following the last one. Then, we just copy the elements from the
        # stack to the new stack, and when we see the indicator object, we
        # add our new objects.
        ind_obj = self._stack[li + 1]

        new_list = []
        for i in range(n):
            o = self._stack[i]
            if not o in self._objects:
                new_list.append(o)
            if o == ind_obj:
                new_list.extend(self._objects)

        self._stack[:] = new_list[:]

    def lower(self):
        fi = self._stack.index(self._objects[0])
        n  = len(self._stack)

        if fi == 0:
            self.bottom()
            return

        # otherwise, we move all the objects to the position of the element
        # preceding the first one. Then, we just copy the elements from the
        # stack to the new stack, and when we see the indicator object, we
        # this could be done more efficiently, but that way it is clearer

        ind_obj = self._stack[fi - 1]
        new_list = []
        for i in range(n):
            o = self._stack[i]
            if o == ind_obj:
                new_list.extend(self._objects)
            if not o in self._objects:
                new_list.append(o)

        self._stack[:] = new_list[:]

    def top(self):
        for obj in self._objects:
            self._stack.remove(obj)
            self._stack.append(obj)

    def bottom(self):
        for obj in self.obj[::-1]:
            self._stack.remove(obj)
            self._stack.insert(0, obj)

    def undo(self):
        self.swap_stacks(self._stack, self._stack_orig)
        self._undone = True

    def redo(self):
        if not self._undone:
            return
        self.swap_stacks(self._stack, self._stack_orig)
        self._undone = False

    def swap_stacks(self, stack1, stack2):
        stack1[:], stack2[:] = stack2[:], stack1[:]


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
    """

    def __init__(self, type, obj, origin):
        super().__init__("move", obj)
        self.start_point = origin
        self.origin      = origin
        self.bbox        = obj.bbox()

    def event_update(self, x, y):
        raise NotImplementedError("event_update method not implemented for type", self._type)

    def event_finish(self):
        raise NotImplementedError("event_finish method not implemented for type", self._type)

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

    def __init__(self, obj, origin=None, corner=None, angle = None):
        super().__init__("rotate", obj, origin)
        self.corner      = corner
        bb = obj.bbox()
        self._rotation_centre = (bb[0] + bb[2] / 2, bb[1] + bb[3] / 2)
        obj.rotate_start(self._rotation_centre)

        if not angle is None:
            self.obj.rotate(angle, set = False)

        self._angle = 0

    def event_update(self, x, y):
        angle = calc_rotation_angle(self._rotation_centre, self.start_point, (x, y))
        d_a = angle - self._angle
        self._angle = angle
        self.obj.rotate(d_a, set = False)

    def event_finish(self):
        self.obj.rotate_end()

    def undo(self):
        self.obj.rotate_start(self._rotation_centre)
        self.obj.rotate(-self._angle)
        self.obj.rotate_end()
        self._undone = True

    def redo(self):
        if not self._undone:
            return
        self.obj.rotate_start(self._rotation_centre)
        self.obj.rotate(self._angle)
        self.obj.rotate_end()
        self._undone = False

class MoveCommand(MoveResizeCommand):
    """Simple class for handling move events."""
    def __init__(self, obj, origin):
        super().__init__("move", obj, origin)
        self._last_pt = origin
        print("MoveCommand: origin is", origin)

    def event_update(self, x, y):
        dx = x - self._last_pt[0]
        dy = y - self._last_pt[1]

        self.obj.move(dx, dy)
        self._last_pt = (x, y)

    def event_finish(self):
        print("MoveCommand: finish")
        pass

    def undo(self):
        if self._undone:
            return
        print("MoveCommand: undo")
        dx = self.start_point[0] - self._last_pt[0]
        dy = self.start_point[1] - self._last_pt[1]
        self.obj.move(dx, dy)
        self._undone = True

    def redo(self):
        if not self._undone:
            return
        dx = self.start_point[0] - self._last_pt[0]
        dy = self.start_point[1] - self._last_pt[1]
        self.obj.move(-dx, -dy)
        self._undone = False


class ResizeCommand(MoveResizeCommand):
    """Simple class for handling resize events."""
    def __init__(self, obj, origin, corner, proportional = False):
        super().__init__("resize", obj, origin)
        self.corner = corner
        obj.resize_start(corner, origin)
        self._orig_bb = obj.bbox()
        self._prop    = proportional
        ## XXX check the bb for pitfalls
        self._orig_bb_ratio = self._orig_bb[3] / self._orig_bb[2]


    def undo(self):
        obj = self.obj
        pt  = (self._orig_bb[0], self._orig_bb[1])
        obj.resize_start(self.corner, pt)
        self.obj.resize_update(self._orig_bb)
        obj.resize_end()
        self._undone = True

    def redo(self):
        if not self._undone:
            return
        obj = self.obj
        obj.resize_start(self.corner, self.start_point)
        obj.resize_update(self._newbb)
        obj.resize_end()
        self._undone = False

    def event_finish(self):
        self.obj.resize_end()

    def event_update(self, x, y):
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

        self._newbb = newbb
        self.obj.resize_update(newbb)


