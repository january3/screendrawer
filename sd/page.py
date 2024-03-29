from .drawable import Drawable, DrawableGroup, SelectionObject # <remove>
from .commands import *                                              # <remove>
from .utils import sort_by_stack                                      # <remove>


class Page:
    def __init__(self, prev = None):
        self.__prev    = prev
        self.__next = None
        self.__objects = []
        self.__history = []
        self.__redo_stack = []
        self.__selection = SelectionObject(self.__objects)

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

    def kill_object(self, obj):
        """Directly remove an object from the list of objects."""
        self.__objects.remove(obj)

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


