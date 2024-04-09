"""
This module contains the Page class, which is a container for objects.
"""

from .drawable import Drawable # <remove>
from .drawable_group import SelectionObject # <remove>
from .commands import RemoveCommand, CommandGroup # <remove>
from .commands import DeletePageCommand, DeleteLayerCommand # <remove>
from .drawer import Drawer                                           # <remove>


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

    def objects_import(self, object_list):
        """Import objects from a dict"""
        self.objects([ Drawable.from_dict(d) for d in object_list ] or [ ])

    def selection(self):
        """Return the selection object."""
        return self.__selection

    def kill_object(self, obj):
        """Directly remove an object from the list of objects."""
        if obj in self.__objects:
            self.__objects.remove(obj)

    def export(self):
        """Exports the layer as a dict"""
        return [ obj.to_dict() for obj in self.__objects ]

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
        self.__layers = [ layers or Layer() ]
        self.__current_layer = 0
        self.__translate = None
        self.__drawer = Drawer()

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

        cmd = DeletePageCommand(self, self.__prev, self.__next)
        ret = self.__prev or self.__next

        return ret, cmd

    def objects(self, objects = None):
        """Return or set the list of objects on the page."""
        layer = self.__layers[self.__current_layer]
        return layer.objects(objects)

    def objects_all_layers(self):
        """Return all objects on all layers."""
        objects = [ obj for layer in self.__layers for obj in layer.objects() ]
        return objects

    def selection(self):
        """Return the selection object."""
        layer = self.__layers[self.__current_layer]
        return layer.selection()

    def kill_object(self, obj):
        """Directly remove an object from the list of objects."""
        layer = self.__layers[self.__current_layer]
        layer.kill_object(obj)

    def number_of_layers(self):
        """Return the number of layers."""
        return len(self.__layers)

    def next_layer(self):
        """Switch to the next layer."""
        print("appending a new layer")
        self.__current_layer += 1
        if self.__current_layer == len(self.__layers):
            self.__layers.append(Layer())
        self.__layers[self.__current_layer].selection().all()
        return self.__current_layer

    def prev_layer(self):
        """Switch to the previous layer."""
        self.__current_layer = max(0, self.__current_layer - 1)
        self.__layers[self.__current_layer].selection().all()
        return self.__current_layer

    def layer(self, new_layer = None, pos = None):
        """
        Get or insert the current layer.

        Arguments:
        new_layer -- if not None, insert new_layer and set
                     the current layer to new_layer.
        pos -- if not None, insert a new layer at pos.
        """
        if new_layer is not None:
            if pos is not None and pos < len(self.__layers):
                self.__layers.insert(pos, new_layer)
                self.__current_layer = pos
            else:
                self.__layers.append(new_layer)
                self.__current_layer = len(self.__layers) - 1

        return self.__layers[self.__current_layer]

    def layer_no(self, layer_no = None):
        """Get or set the current layer."""
        if layer_no is None:
            return self.__current_layer

        layer_no = max(0, layer_no)

        if layer_no >= len(self.__layers):
            self.__layers.append(Layer())

        self.__current_layer = layer_no
        return self.__current_layer

    def delete_layer(self, layer_no = None):
        """Delete the current layer."""

        if len(self.__layers) == 1:
            return None

        if layer_no is None or layer_no < 0 or layer_no >= len(self.__layers):
            layer_no = self.__current_layer

        layer = self.__layers[layer_no]
        pos   = layer_no

        del self.__layers[layer_no]

        self.__current_layer = max(0, layer_no - 1)
        self.__current_layer = min(self.__current_layer,
                                   len(self.__layers) - 1)

        cmd = DeleteLayerCommand(self, layer, pos)

        return cmd

    def translate(self, new_val = None):
        """Get or set the translate"""
        if new_val is not None:
            self.__translate = new_val
        return self.__translate

    def translate_set(self, new_val):
        """Set the translate"""
        self.__translate = new_val
        return self.__translate

    def export(self):
        """Exports the page with all layers as a dict"""
        layers = [ l.export() for l in self.__layers ]
        ret = {
                 "layers": layers,
                 "translate": self.translate(),
                 "cur_layer": self.__current_layer
              }
        return ret

    def import_page(self, page_dict):
        """Imports a dict to self"""
        print("importing pages")
        self.__translate = page_dict.get("translate")
        if "objects" in page_dict:
            self.objects(page_dict["objects"])
        elif "layers" in page_dict:
            print(len(page_dict["layers"]), "layers found")
            print("however, we only have", len(self.__layers), "layers")
            print("creating", len(page_dict["layers"]) - len(self.__layers), "new layers")
            self.__current_layer = 0
            for _ in range(len(page_dict["layers"]) - len(self.__layers)):
                self.next_layer()
            self.__current_layer = 0
            for l_list in page_dict["layers"]:
                layer = self.__layers[self.__current_layer]
                layer.objects_import(l_list)
                self.__current_layer += 1

        cl = page_dict.get("cur_layer")
        self.__current_layer = cl if cl is not None else 0

    def clear(self):
        """
        Remove all objects from all layers.

        Returns a CommandGroup object that can be used to undo the
        operation.
        """
        ret_commands = []
        for layer in self.__layers:
            cmd = RemoveCommand(layer.objects()[:], layer.objects(), page = self)
            ret_commands.append(cmd)
        return CommandGroup(ret_commands[::-1])

    def draw(self, cr, state):
        """Draw the objects on the page."""

        self.__drawer.draw(cr, self, state)
