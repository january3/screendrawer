"""Add, Move, Resize, Rotate, and Delete commands for drawable objects."""
import logging                                                   # <remove>
from .commands import Command                            #<remove>
from .utils import swap_stacks                ## <remove>
from .utils import calc_rotation_angle ## <remove>
log = logging.getLogger(__name__)                                # <remove>

class AddCommand(Command):
    """
    Class for handling creating objects.

    :param objects: a list of objects to be removed.
    """
    def __init__(self, objects, stack):
        super().__init__("add", objects)
        self.__stack = stack
        self.__add_objects()

    def __add_objects(self):
        for o in self.obj:
            self.__stack.append(o)

    def undo(self):
        if self.undone():
            return
        for o in self.obj:
            self.__stack.remove(o)
        self.undone_set(True)

    def redo(self):
        if not self.undone():
            return
        self.__add_objects()
        self.undone_set(False)

class RemoveCommand(Command):
    """
    Class for handling deleting objects.

    :param objects: a list of objects to be removed.
    """
    def __init__(self, objects, stack):
        super().__init__("remove", objects)
        self.__stack = stack
        self.__stack_copy = self.__stack[:]

        # remove the objects from the stack
        for obj in self.obj:
            self.__stack.remove(obj)

    def undo(self):
        if self.undone():
            return
        swap_stacks(self.__stack, self.__stack_copy)
        self.undone_set(True)

    def redo(self):
        if not self.undone():
            return
        swap_stacks(self.__stack, self.__stack_copy)
        self.undone_set(False)


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

    def __init__(self, obj, origin=None, corner=None, angle = None):
        super().__init__("rotate", obj, origin)
        self.corner      = corner
        bb = obj.bbox(actual = True)
        self.bbox        = obj.bbox()
        self.__rotation_centre = (bb[0] + bb[2] / 2, bb[1] + bb[3] / 2)
        obj.rotate_start(self.__rotation_centre)

        self.__angle = 0

        if not angle is None:
            self.obj.rotate(angle, set_angle = False)
            self.__angle = angle

    def event_update(self, x, y):
        angle = calc_rotation_angle(self.__rotation_centre, self.start_point, (x, y))
        d_a = angle - self.__angle
        self.__angle = angle
        self.obj.rotate(d_a, set_angle = False)

    def event_finish(self):
        self.obj.rotate_end()

    def undo(self):
        if self.undone():
            return
        if not self.__angle:
            return
        self.obj.rotate_start(self.__rotation_centre)
        self.obj.rotate(0 - self.__angle)
        self.obj.rotate_end()
        self.undone_set(True)

    def redo(self):
        if not self.undone():
            return
        self.obj.rotate_start(self.__rotation_centre)
        self.obj.rotate(self.__angle)
        self.obj.rotate_end()
        self.undone_set(False)

class MoveCommand(MoveResizeCommand):
    """Simple class for handling move events."""
    def __init__(self, obj, origin):
        obj = obj.objects
        super().__init__("move", obj, origin)
        self.__last_pt = origin
        log.debug("MoveCommand: origin is %s hash %s",
               [int(x) for x in origin], self.hash())

    def __add__(self, other):
        """Add two move commands"""
        if not isinstance(other, MoveCommand):
            return super().__add__(other)

        dx = other.last_pt()[0] - other.start_point[0]
        dy = other.last_pt()[1] - other.start_point[1]
        self.__last_pt = (
                self.__last_pt[0] + dx,
                self.__last_pt[1] + dy
                )
        return self

    def last_pt(self):
        """Return last point"""
        return self.__last_pt

    def event_update(self, x, y):
        """Update the move event."""
        dx = x - self.__last_pt[0]
        dy = y - self.__last_pt[1]

        for obj in self.obj:
            obj.move(dx, dy)
        self.__last_pt = (x, y)

    def event_finish(self):
        """Finish the move event."""
        log.debug("MoveCommand: finish")

    def undo(self):
        """Undo the command."""
        if self.undone():
            return
        log.debug("MoveCommand: undo")
        dx = self.start_point[0] - self.__last_pt[0]
        dy = self.start_point[1] - self.__last_pt[1]
        for obj in self.obj:
            obj.move(dx, dy)
        self.undone_set(True)

    def redo(self):
        """Redo the command."""
        if not self.undone():
            return
        dx = self.start_point[0] - self.__last_pt[0]
        dy = self.start_point[1] - self.__last_pt[1]
        for obj in self.obj:
            obj.move(-dx, -dy)
        self.undone_set(False)

class ResizeCommand(MoveResizeCommand):
    """Simple class for handling resize events."""
    def __init__(self, obj, origin, corner, proportional = False):
        super().__init__("resize", obj, origin)
        self.corner = corner
        self.bbox     = obj.bbox(actual = True)
        obj.resize_start(corner, origin)
        self._orig_bb = obj.bbox(actual = True)
        self._prop    = proportional
        if self._orig_bb[2] == 0:
            raise ValueError("Bounding box with no width")

        self._orig_bb_ratio = self._orig_bb[3] / self._orig_bb[2]
        self.__newbb = None

    def undo(self):
        """Undo the command."""
        if self.undone():
            return
        obj = self.obj
        pt  = (self._orig_bb[0], self._orig_bb[1])
        obj.resize_start(self.corner, pt)
        self.obj.resize_update(self._orig_bb)
        obj.resize_end()
        self.undone_set(True)

    def redo(self):
        """Redo the command."""
        if not self.undone():
            return
        obj = self.obj
        obj.resize_start(self.corner, self.start_point)
        obj.resize_update(self.__newbb)
        obj.resize_end()
        self.undone_set(False)

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
