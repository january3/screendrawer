"""
This module contains the commands that can be executed on the objects. They
should be undoable and redoable. It is their responsibility to update the
state of the objects they are acting on.
"""
import logging                                # <remove>
import hashlib                                # <remove>
from .utils import sort_by_stack              # <remove>
from .utils import swap_stacks                # <remove>
from .drawable_factory import DrawableFactory # <remove>
from .drawable_group import DrawableGroup     # <remove>
from .drawable_group import ClippingGroup     # <remove>
log = logging.getLogger(__name__)             # <remove>

## ---------------------------------------------------------------------
## These are the commands that can be executed on the objects. They should
## be undoable and redoable. It is their responsibility to update the
## state of the objects they are acting on.

## The hash is used to identify the command. By default, it is unique for
## for a type of command and the objects it is acting on.
## That way, a command affecting a certain group of primitives can be
## joined with another command that does the same thing on the same group.
## ---------------------------------------------------------------------

def compute_id_hash(objects):
    """calculate a unique hash based on the drawables carreid by the object"""
    # Extract IDs and concatenate them into a single string
    if isinstance(objects, list):
        ids_concatenated = ''.join(str(id(obj)) for obj in objects)
    else:
        ids_concatenated = str(id(objects))

    # Compute the hash of the concatenated string
    hash_object = hashlib.md5(ids_concatenated.encode())
    hash_hex    = hash_object.hexdigest()

    return hash_hex

class Command:
    """Base class for commands."""
    def __init__(self, mytype, objects):
        self.obj   = objects
        self.__type   = mytype
        self.__undone = False

        if objects:
            self.__hash = compute_id_hash(objects)
        else:
            self.__hash = compute_id_hash(self)

        self.__hash = mytype + ':' + self.__hash

    def __eq__(self, other):
        """Return whether the command is equal to another command."""
        return self.hash() == other.hash()

    def __gt__(self, other):
        """Return whether the command is a group that contains commands with identical hashes."""
        return self.hash() ==  'group:' + other.hash()

    def __add__(self, other):
        """Add two commands together."""

        if other.com_type() == "group":
            other.add(self)
            return other

        return CommandGroup([ self, other ])

    def com_type(self):
        """Return my type"""
        return self.__type

    def hash(self):
        """Return a hash of the command."""
        return self.__hash

    def type(self):
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
    def __init__(self, commands):
        super().__init__("group", objects=None)
        self.__commands = commands

        self.__hash = compute_id_hash([ self ])
        self.__hash = "group" + ':' + self.__hash

    def __add__(self, other):
        """Add two commands together."""
        if other.type() == "group":
            return CommandGroup(self.__commands + other.commands())
        return CommandGroup(self.__commands + [ other ])

    def hash(self):
        """Return a hash of the command."""
        cmds = self.__commands
        hashes = [ cmd.hash() for cmd in cmds ]

        ## how many unique values in the hashes array?
        unique_hashes = set(hashes)
        if len(unique_hashes) == 1:
            return 'group:' + hashes[0]

        return self.__hash

    def commands(self, cmd = None):
        """Return or set the commands in the group."""
        if cmd:
            self.__commands = cmd
        return self.__commands

    def add(self, cmd):
        """Add a command to the group."""
        self.__commands.append(cmd)
        return self

    def undo(self):
        """Undo the command."""
        if self.undone():
            return
        for cmd in self.__commands[::-1]:
            cmd.undo()
        self.undone_set(True)
        return

    def redo(self):
        """Redo the command."""
        if not self.undone():
            return
        for cmd in self.__commands:
            cmd.redo()
        self.undone_set(False)
        return

class InsertPageCommand(Command):
    """
    Handling inserting pages.
    """
    def __init__(self, page):
        super().__init__("insert_page", None)
        self.__prev = page
        self.__next = page.next(create = False)

        # create the new page
        page.next_set(None)
        self.__page = page.next(create = True)
        self.__page.prev_set(page)
        self.__page.next_set(self.__next)
        page.next_set(self.__page)

        if self.__next:
            self.__next.prev_set(self.__page)

    def undo(self):
        """Undo the command."""
        if self.undone():
            return None
        self.__prev.next_set(self.__next)
        if self.__next:
            self.__next.prev_set(self.__prev)
        self.undone_set(True)
        return self.__prev

    def redo(self):
        """Redo the command."""
        if not self.undone():
            return None
        self.__prev.next_set(self.__page)
        if self.__next:
            self.__next.prev_set(self.__page)
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

class ClipCommand(Command):
    """Simple class for handling clipping objects."""
    def __init__(self, clip, objects, stack, selection_object = None):
        super().__init__("clip", objects)
        self.__selection = selection_object
        self.__stack = stack
        self.__stack_copy = stack[:]

        # position of the last object in stack
        idx = self.__stack.index(self.obj[-1])

        self.__group = ClippingGroup(clip, self.obj)
        # add group to the stack at the position of the last object
        self.__stack.insert(idx, self.__group)

        for obj in self.obj:
            if not obj in stack:
                raise ValueError("Object not in stack:", obj)
            stack.remove(obj)
        stack.remove(clip)

    def undo(self):
        """Undo the command."""
        if self.undone():
            return
        swap_stacks(self.__stack, self.__stack_copy)
        self.undone_set(True)
        if self.__selection:
            self.__selection.set(self.obj)

    def redo(self):
        """Redo the command."""
        if not self.undone():
            return
        swap_stacks(self.__stack, self.__stack_copy)
        self.undone_set(False)
        if self.__selection:
            self.__selection.set([ self.__group ])

class UnClipCommand(Command):
    """Simple class for handling clipping objects."""
    def __init__(self, objects, stack, selection_object = None):
        super().__init__("unclip", objects)
        self.__stack = stack
        self.__stack_copy = stack[:]
        self.__selection = selection_object

        new_objects = []
        n = 0

        for obj in self.obj:
            if not obj.type == "clipping_group":
                log.warning("Object is not a clipping_group, ignoring: %s", obj)
                log.warning("object type: %s", obj.type)
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

        self.__group = new_objects

    def undo(self):
        """Undo the command."""
        if self.undone():
            return
        swap_stacks(self.__stack, self.__stack_copy)
        self.undone_set(True)
        if self.__selection:
            self.__selection.set(self.obj)

    def redo(self):
        """Redo the command."""
        if not self.undone():
            return
        swap_stacks(self.__stack, self.__stack_copy)
        self.undone_set(False)
        if self.__selection:
            self.__selection.set(self.__group)

class TextEditCommand(Command):
    """Simple class for handling text editing."""

    def __init__(self, obj, oldtext, newtext):
        super().__init__("text_edit", obj)
        self.__oldtext = oldtext
        self.__newtext = newtext

    def undo(self):
        """Undo the command."""
        if self.undone():
            return
        self.obj.set_text(self.__oldtext)
        self.undone_set(True)

    def redo(self):
        """Redo the command."""
        if not self.undone():
            return
        self.obj.set_text(self.__newtext)
        self.undone_set(False)

class AddToGroupCommand(Command):
    """ Add an object to an existing group """

    def __init__(self, group, obj, page=None):
        super().__init__("add_to_group", objects=None)
        self.__page      = page
        self.__group     = group
        self.__obj       = obj

        group.add(obj)

    def undo(self):
        """Undo the command."""
        if self.undone():
            return None
        self.__group.remove(self.__obj)
        self.undone_set(True)

        return self.__page

    def redo(self):
        """Redo the command."""
        if not self.undone():
            return None
        self.__group.add(self.__obj)
        self.undone_set(False)

        return self.__page

class GroupObjectCommand(Command):
    """Simple class for handling grouping objects."""
    def __init__(self, objects, stack, selection_object = None):
        objects = sort_by_stack(objects, stack)

        super().__init__("group", objects)
        self.__stack      = stack
        self.__stack_copy = stack[:]

        self.__selection = selection_object

        self.__group = DrawableGroup(self.obj)

        # position of the last object in stack
        idx = self.__stack.index(self.obj[-1])

        # add group to the stack at the position of the last object
        self.__stack.insert(idx, self.__group)

        for obj in self.obj:
            if not obj in stack:
                raise ValueError("Object not in stack:", obj)
            self.__stack.remove(obj)

        if self.__selection:
            self.__selection.set([ self.__group ])

    def undo(self):
        """Undo the command."""
        if self.undone():
            return
        swap_stacks(self.__stack, self.__stack_copy)
        self.undone_set(True)
        if self.__selection:
            self.__selection.set(self.obj)

    def redo(self):
        """Redo the command."""
        if not self.undone():
            return
        swap_stacks(self.__stack, self.__stack_copy)
        self.undone_set(False)
        if self.__selection:
            self.__selection.set([ self.__group ])

class UngroupObjectCommand(Command):
    """
    Class for handling ungrouping objects.

    :param objects: Objects to be ungrouped (objects which are not groups
                    will be ignored)
    """
    def __init__(self, objects, stack, selection_object = None):
        super().__init__("ungroup", objects)
        self.__stack = stack
        self.__stack_copy = stack[:]
        self.__selection = selection_object

        new_objects = []
        n = 0

        for obj in self.obj:
            if not obj.type == "group":
                log.warning("Object is not a group, ignoring: %s", obj)
                log.warning("object type: %s", obj.type)
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

    def undo(self):
        """Undo the command."""
        if self.undone():
            return
        swap_stacks(self.__stack, self.__stack_copy)
        self.undone_set(True)
        if self.__selection:
            self.__selection.set([ self.obj ])

    def redo(self):
        """Redo the command."""
        if not self.undone():
            return
        swap_stacks(self.__stack, self.__stack_copy)
        self.undone_set(False)
        if self.__selection:
            self.__selection.set(self.obj)

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
        #self.__new_type = new_type
        self.__old_objs = [ ]
        self.__new_objs = [ ]
        self.__stack    = stack
        self.__selection_objects = selection_objects
        log.debug("executing transmute; undone = %s", self.undone())

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
        self.__selection_objects[:] = [ obj_map.get(obj, obj) for obj in self.__selection_objects ]

    def obj_map(self):
        """Return a dictionary mapping old objects to new objects."""
        return { self.__old_objs[i]: self.__new_objs[i] for i in range(len(self.__old_objs)) }

    def undo(self):
        """replace all the new objects with the old ones in the stack"""
        if self.undone():
            return
        for obj in self.__new_objs:
            self.__stack.remove(obj)
        for obj in self.__old_objs:
            self.__stack.append(obj)
        self.undone_set(True)
        if self.__selection_objects:
            self.map_selection()
        return

    def redo(self):
        """put the new objects again on the stack and remove the old ones"""
        if not self.undone():
            return
        for obj in self.__old_objs:
            self.__stack.remove(obj)
        for obj in self.__new_objs:
            self.__stack.append(obj)
        self.undone_set(False)
        if self.__selection_objects:
            self.map_selection()
        return

class ZStackCommand(Command):
    """Simple class for handling z-stack operations."""
    def __init__(self, objects, stack, operation):
        super().__init__("z_stack", objects)
        self._operation  = operation
        self.__stack      = stack

        for obj in objects:
            if not obj in stack:
                raise ValueError("Object not in stack:", obj)

        self._objects = sort_by_stack(objects, stack)
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
            return
        swap_stacks(self.__stack, self.__stack_orig)
        self.undone_set(True)

    def redo(self):
        """Redo the command."""
        if not self.undone():
            return
        swap_stacks(self.__stack, self.__stack_orig)
        self.undone_set(False)

class FlushCommand(Command):
    """
    Class for flushing a group of object to left / right / top / bottom.
    """

    def __init__(self, objects, flush_direction):
        name = "flush_" + flush_direction
        super().__init__(name, objects)
        log.debug("flushing objects %s to %s", objects, flush_direction)

        self.__undo_dict = None
        self.__redo_dict = None

        # call the appropriate method
        if flush_direction == "left":
            self.flush_left()
        elif flush_direction == "right":
            self.flush_right()
        elif flush_direction == "top":
            self.flush_top()
        elif flush_direction == "bottom":
            self.flush_bottom()
        else:
            raise ValueError("Invalid flush direction:", flush_direction)

    def flush_left(self):
        """Flush the objects to the left."""
        log.debug("flushing left")
        self.__undo_dict = { obj: obj.bbox() for obj in self.obj }
        min_x = min(obj.bbox()[0] for obj in self.obj)

        for obj in self.obj:
            obj.move(min_x - obj.bbox()[0], 0)

        self.__redo_dict = { obj: obj.bbox() for obj in self.obj }

    def flush_right(self):
        """Flush the objects to the right."""
        self.__undo_dict = { obj: obj.bbox() for obj in self.obj }
        max_x = max(obj.bbox()[2] + obj.bbox()[0] for obj in self.obj)

        for obj in self.obj:
            obj.move(max_x - (obj.bbox()[2] + obj.bbox()[0]), 0)

        self.__redo_dict = { obj: obj.bbox() for obj in self.obj }

    def flush_top(self):
        """Flush the objects to the top."""
        self.__undo_dict = { obj: obj.bbox() for obj in self.obj }
        min_y = min(obj.bbox()[1] for obj in self.obj)

        for obj in self.obj:
            obj.move(0, min_y - obj.bbox()[1])

        self.__redo_dict = { obj: obj.bbox() for obj in self.obj }

    def flush_bottom(self):
        """Flush the objects to the bottom."""
        self.__undo_dict = { obj: obj.bbox() for obj in self.obj }
        max_y = max(obj.bbox()[3] + obj.bbox()[1] for obj in self.obj)

        for obj in self.obj:
            obj.move(0, max_y - (obj.bbox()[3] + obj.bbox()[1]))

        self.__redo_dict = { obj: obj.bbox() for obj in self.obj }

    def move_to_bb(self, obj, bb):
        """Move the object to the bounding box."""
        bb_obj = obj.bbox()
        dx = bb[0] - bb_obj[0]
        dy = bb[1] - bb_obj[1]
        obj.move(dx, dy)

    def undo(self):
        """Undo the command."""
        if self.undone():
            return
        for obj in self.obj:
            self.move_to_bb(obj, self.__undo_dict[obj])
        self.undone_set(True)

    def redo(self):
        """Redo the command."""
        if not self.undone():
            return
        for obj in self.obj:
            self.move_to_bb(obj, self.__redo_dict[obj])
        self.undone_set(False)
