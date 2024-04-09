"""Factory for drawable objects"""
from .drawable_primitives import Text, Rectangle, Shape, Circle # <remove>
from .drawable_paths import Path # <remove>


class DrawableFactory:
    """
    Factory class for creating drawable objects.
    """
    @classmethod
    def create_drawable(cls, mode, pen, ev):
        """
        Create a drawable object of the specified type.
        """
        pos = ev.pos()
        pressure = ev.pressure()
        ret_obj = None

        print("create object in mode", mode)
        #if mode == "text" or (mode == "draw" and shift_click and no_active_area):

        if mode == "text":
            print("creating text object")
            ret_obj = Text([ pos ], pen = pen, content = "")
            ret_obj.move_caret("Home")

        elif mode == "draw":
            print("creating path object")
            ret_obj = Path([ pos ], pen = pen, pressure = [ pressure ])

        elif mode == "rectangle":
            print("creating rectangle object")
            ret_obj = Rectangle([ pos ], pen = pen)

        elif mode == "shape":
            print("creating shape object")
            ret_obj = Shape([ pos ], pen = pen)

        elif mode == "circle":
            print("creating circle object")
            ret_obj = Circle([ pos, (pos[0], pos[1]) ], pen = pen)

        else:
            raise ValueError("Unknown mode:", mode)

        return ret_obj

    @classmethod
    def transmute(cls, obj, mode):
        """
        Transmute an object into another type.

        For groups, the behaviour is special: rather than converting the group
        into a single object, we convert all objects within the group into the
        new type by calling the transmute_to method of the group object.
        """
        print("transmuting object to", mode)

        if obj.type == "group":
            # for now, we do not pass transmutations to groups, because
            # we then cannot track the changes.
            return obj

        if mode == "text":
            obj = Text.from_object(obj)
        elif mode == "draw":
            obj = Path.from_object(obj)
        elif mode == "rectangle":
            obj = Rectangle.from_object(obj)
        elif mode == "shape":
            print("calling Shape.from_object")
            obj = Shape.from_object(obj)
        elif mode == "circle":
            obj = Circle.from_object(obj)
        else:
            raise ValueError("Unknown mode:", mode)

        return obj
