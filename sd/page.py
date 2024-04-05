"""
This module contains the Page class, which is a container for objects.
"""
from .drawable import SelectionObject # <remove>


class Layer:
    """
    A layer is a container for objects.
    """
    def __init__(self):
        self.__objects = []
        self.__selection = SelectionObject(self.__objects)

    def objects(self, objects = None):
        """Return or set the list of objects on the layer."""
        if objects:
            self.__objects = objects
            self.__selection = SelectionObject(self.__objects)
        return self.__objects

    def selection(self):
        """Return the selection object."""
        return self.__selection

    def kill_object(self, obj):
        """Directly remove an object from the list of objects."""
        if obj in self.__objects:
            self.__objects.remove(obj)

class Page:
    """
    A page is a container for layers.

    It serves as an interface between layers and whatever wants to
    manipulate objects or selection on a layer by choosing the current
    layer and managing layers.
    """
    def __init__(self, prev = None, layers = None):
        self.__prev    = prev
        self.__next = None
        #self.__objects = []
        #self.__selection = SelectionObject(self.__objects)
        self.__layers = [ layers or Layer() ]
        self.__current_layer = 0

    def next(self, create = True):
        """
        Return the next page.

        If create is True, create a new page if it doesn't exist.
        """
        if not self.__next and create:
            print("Creating new page")
            self.__next = Page(self)
        return self.__next

    def prev(self):
        """Return the previous page."""
        return self.__prev or self # can't go beyond first page

    def next_set(self, page):
        """Set the next page."""
        self.__next = page

    def prev_set(self, page):
        """Set the previous page."""
        self.__prev = page

    def delete(self):
        """Delete the page and create links between prev and next pages."""
        if not self.__prev and not self.__next:
            print("only one page remaining")
            return self

        if self.__prev:
            self.__prev.next_set(self.__next)
        if self.__next:
            self.__next.prev_set(self.__prev)
        return self.__prev or self.__next

    def objects(self, objects = None):
        """Return or set the list of objects on the page."""
        layer = self.__layers[self.__current_layer]
        return layer.objects(objects)

    def objects_all_layers(self):
        """Return all objects on all layers."""
        return [ obj for layer in self.__layers for obj in layer.objects() ]

    def selection(self):
        """Return the selection object."""
        layer = self.__layers[self.__current_layer]
        return layer.selection()

    def kill_object(self, obj):
        """Directly remove an object from the list of objects."""
        layer = self.__layers[self.__current_layer]
        layer.kill_object(obj)

    def next_layer(self):
        """Switch to the next layer."""
        print("appending a new layer")
        self.__current_layer += 1
        if self.__current_layer == len(self.__layers):
            self.__layers.append(Layer())
        return self.__current_layer

    def prev_layer(self):
        """Switch to the previous layer."""
        self.__current_layer -= 1
        if self.__current_layer < 0:
            self.__current_layer = 0
        return self.__current_layer

    def set_layer(self, layer_no):
        """Set the current layer."""
        if layer_no < 0:
            layer_no = 0
        if layer_no >= len(self.__layers):
            self.__layers.append(Layer())
        self.__current_layer = layer_no
        return self.__current_layer

    def delete_layer(self):
        """Delete the current layer."""
        if len(self.__layers) == 1:
            return
        del self.__layers[self.__current_layer]
        if self.__current_layer >= len(self.__layers):
            self.__current_layer = len(self.__layers) - 1
        return self.__current_layer

