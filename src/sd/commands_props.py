"""Commands for setting properties of objects."""
import logging                      # <remove>
from .commands import Command       # <remove>
log = logging.getLogger(__name__)   # <remove>

# -------------------------------------------------------------------------------------
class SetPropCommand(Command):
    """
    Superclass for handling property changes of drawing primitives.

    The superclass handles everything, while the subclasses set up the
    functions that do the actual manipulation of the primitives.

    In principle, we need one function to extract the current property, and
    one to set the property.
    """
    def __init__(self, mytype, objects, prop, prop_func):
        super().__init__(mytype, objects.get_primitive())
        self.__prop = prop
        self.__prop_func = prop_func
        self.__undo_dict = { obj: prop_func(obj) for obj in self.obj }
        log.debug("undo_dict: %s", self.__undo_dict)

        for obj in self.obj:
            log.debug("setting prop type %s for %s", mytype, obj)
            prop_func(obj, prop)
            obj.modified(True)

    def __add__(self, other):

        if self == other:
            self.__prop = other.prop()
            return self

        return super().__add__(other)

    def prop(self):
        """Return the property"""
        return self.__prop

    def undo(self):
        """Undo the command."""
        if self.undone():
            return
        for obj in self.obj:
            if obj in self.__undo_dict:
                self.__prop_func(obj, self.__undo_dict[obj])
                obj.modified(True)
        self.undone_set(True)

    def redo(self):
        """Redo the command."""
        if not self.undone():
            return
        for obj in self.obj:
            self.__prop_func(obj, self.__prop)
            obj.modified(True)
        self.undone_set(False)

def set_pen(obj, prop = None):
    """Set the pen property."""
    if prop:
        obj.pen_set(prop)
    return obj.pen

class SetPenCommand(SetPropCommand):
    """Simple class for handling color changes."""
    def __init__(self, objects, pen):
        pen = pen.copy()
        super().__init__("set_pen", objects, pen, set_pen)

def set_transparency(obj, prop = None):
    """Set the transparency property."""
    if prop:
        obj.pen.transparency_set(prop)
    return obj.pen.transparency

class SetTransparencyCommand(SetPropCommand):
    """Simple class for handling line width changes."""
    def __init__(self, objects, width):
        super().__init__("set_transparency", objects, width,
                         set_transparency)

def set_line_width(obj, prop = None):
    """Set the line width property."""
    if prop:
        obj.stroke(prop)
    return obj.stroke()

class SetLineWidthCommand(SetPropCommand):
    """Simple class for handling line width changes."""
    def __init__(self, objects, width):
        super().__init__("set_line_width", objects, width,
                         set_line_width)

def set_color(obj, prop = None):
    """Set the color property."""
    if prop:
        obj.pen.color_set(prop)
    return obj.pen.color

class SetColorCommand(SetPropCommand):
    """Simple class for handling color changes."""
    def __init__(self, objects, color):
        super().__init__("set_color", objects, color, set_color)

def set_font(obj, prop = None):
    """Set the font property."""
    if prop:
        obj.pen.font_set(prop)
    return obj.pen.font_get()

class SetFontCommand(SetPropCommand):
    """Simple class for handling font changes."""
    def __init__(self, objects, font):
        super().__init__("set_font", objects, font, set_font)

class ToggleFillCommand(Command):
    """Simple class for handling toggling fill."""
    def __init__(self, objects):
        super().__init__("fill_toggle", objects.get_primitive())

        for obj in self.obj:
            obj.fill_toggle()

    def undo(self):
        """Undo the command."""
        if self.undone():
            return
        for obj in self.obj:
            obj.fill_toggle()
        self.undone_set(True)

    def redo(self):
        """Redo the command."""
        if not self.undone():
            return
        for obj in self.obj:
            obj.fill_toggle()
        self.undone_set(False)

class ChangeStrokeCommand(Command):
    """Simple class for handling line width changes."""
    def __init__(self, objects, direction):
        super().__init__("change_stroke", objects.get_primitive())

        self.__direction = direction
        self.__undo_dict = { obj: obj.stroke_change(direction) for obj in self.obj }

    def __add__(self, other):
        """Add two commands together."""
        if self == other:
            self.__direction += other.direction()
            return self
        return super().__add__(other)

    def direction(self):
        """Return the direction"""
        return self.__direction

    def undo(self):
        """Undo the command."""
        if self.undone():
            return
        for obj in self.obj:
            if obj in self.__undo_dict:
                obj.stroke(self.__undo_dict[obj])
        self.undone_set(True)

    def redo(self):
        """Redo the command."""
        if not self.undone():
            return
        for obj in self.obj:
            obj.stroke_change(self.__direction)
        self.undone_set(False)
